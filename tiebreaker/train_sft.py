"""CryoForge RL Tiebreaker Trainer — QLoRA SFT (Local GPU & Modal L4) + Experience Policy Compiler.

Supports three execution modes trained strictly on `S01`–`S05` rollouts (zero `S06`–`S08` access):
1. `--local-gpu`:
   Runs 4-bit QLoRA / PEFT (`LoraConfig(r=16, lora_alpha=32, target_modules=[...])`) +
   `trl.SFTTrainer` on `runs/tiebreaker/sft_train_messages.jsonl` and saves the adapter
   to `tiebreaker/checkpoints/cryoforge-qwen25-3b-lora`.
2. `--modal` (or `modal run tiebreaker/train_sft.py`):
   Executes QLoRA fine-tuning remotely on a Modal L4 cloud GPU (`@app.function(gpu="L4", ...)`),
   downloads the trained LoRA adapter files locally, and writes loss curves + metrics to
   `runs/tiebreaker/training_metrics.json`.
3. `--compile-experience`:
   Deterministic CPU fallback that collects/verifies `S01`–`S05` rollouts and contrastive
   failure analyses, compiling `runs/tiebreaker/experience_playbook.json` and recording
   dataset telemetry in `runs/tiebreaker/training_metrics.json`.
"""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import time
from typing import Any

from env_core.contracts import EVAL_SEEDS
from tiebreaker import (
    DEFAULT_CHECKPOINT_DIR,
    FORBIDDEN_HOLDOUT_SEEDS,
    TIEBREAKER_RUNS_DIR,
    TRAJECTORIES_DIR,
    assert_train_seed_only,
)
from tiebreaker.collect_trajectories import (
    collect_and_export,
    verify_zero_holdout_leakage_in_text,
)

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

try:
    import modal  # type: ignore[import-not-found]

    _HAS_MODAL = True
except ImportError:
    modal = None  # type: ignore[assignment]
    _HAS_MODAL = False


# ---------------------------------------------------------------------------
# Dataset Loading, Leakage Verification & Chat-Template Normalization
# ---------------------------------------------------------------------------


def load_and_verify_sft_records(dataset_path: Path) -> list[dict[str, Any]]:
    """Load `sft_train_messages.jsonl` and strictly verify zero `S06`–`S08` holdout leakage."""
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"SFT dataset not found at {dataset_path}. Run `python -m tiebreaker.collect_trajectories` first."
        )
    raw_text = dataset_path.read_text(encoding="utf-8")
    verify_zero_holdout_leakage_in_text(raw_text, str(dataset_path))

    records: list[dict[str, Any]] = []
    for line_no, line in enumerate(raw_text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        row = json.loads(stripped)
        seed = str(row.get("seed", ""))
        assert_train_seed_only(seed)
        if not isinstance(row.get("messages"), list) or len(row["messages"]) < 3:
            raise ValueError(f"Invalid messages array at {dataset_path}:{line_no}")
        records.append(row)

    if not records:
        raise ValueError(f"SFT dataset {dataset_path} contains 0 training trajectories.")
    return records


def _normalize_messages_for_template(
    messages: list[dict[str, Any]],
    *,
    arguments_as_dict: bool,
) -> list[dict[str, Any]]:
    """Normalize `tool_calls[].function.arguments` between JSON strings and dicts."""
    normalized: list[dict[str, Any]] = []
    for msg in messages:
        mcopy = dict(msg)
        if "tool_calls" in mcopy and isinstance(mcopy["tool_calls"], list):
            new_tcs: list[dict[str, Any]] = []
            for tc in mcopy["tool_calls"]:
                tc_copy = dict(tc)
                fn = dict(tc_copy.get("function", {}))
                raw_args = fn.get("arguments", "{}")
                if arguments_as_dict and isinstance(raw_args, str):
                    try:
                        fn["arguments"] = json.loads(raw_args)
                    except json.JSONDecodeError:
                        fn["arguments"] = {}
                elif not arguments_as_dict and not isinstance(raw_args, str):
                    fn["arguments"] = json.dumps(raw_args, ensure_ascii=False, sort_keys=True)
                tc_copy["function"] = fn
                new_tcs.append(tc_copy)
            mcopy["tool_calls"] = new_tcs
        normalized.append(mcopy)
    return normalized


def _fallback_chatml_tool_format(messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> str:
    """Fallback ChatML/Hermes-style multi-turn tool formatting for tokenizers without `tools=`."""
    chunks: list[str] = []
    tools_json = json.dumps(tools, ensure_ascii=False)
    for idx, msg in enumerate(messages):
        role = msg.get("role", "user")
        content = msg.get("content") or ""
        if idx == 0 and role == "system":
            chunks.append(
                f"<|im_start|>system\n{content}\n\n# Tools\n<tools>\n{tools_json}\n</tools><|im_end|>"
            )
        elif role == "assistant" and msg.get("tool_calls"):
            tc_blocks: list[str] = []
            for tc in msg["tool_calls"]:
                fn = tc.get("function", {})
                raw_args = fn.get("arguments", "{}")
                args_obj = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                call_payload = json.dumps({"name": fn.get("name"), "arguments": args_obj}, ensure_ascii=False)
                tc_blocks.append(f"<tool_call>\n{call_payload}\n</tool_call>")
            body = f"{content}\n" + "\n".join(tc_blocks) if content else "\n".join(tc_blocks)
            chunks.append(f"<|im_start|>assistant\n{body}<|im_end|>")
        elif role == "tool":
            resp_payload = json.dumps({"name": msg.get("name"), "observation": content}, ensure_ascii=False)
            chunks.append(f"<|im_start|>user\n<tool_response>\n{resp_payload}\n</tool_response><|im_end|>")
        else:
            chunks.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    return "\n".join(chunks) + "\n"


def render_sft_texts(tokenizer: Any, records: list[dict[str, Any]]) -> list[str]:
    """Render multi-turn tool-calling `messages` + `tools` into tokenized training strings."""
    rendered: list[str] = []
    for row in records:
        messages = row["messages"]
        tools = row.get("tools", [])
        text: str | None = None
        for as_dict in (True, False):
            norm_msgs = _normalize_messages_for_template(messages, arguments_as_dict=as_dict)
            try:
                text = tokenizer.apply_chat_template(
                    norm_msgs,
                    tools=tools,
                    tokenize=False,
                    add_generation_prompt=False,
                )
                break
            except Exception:
                continue
        if text is None:
            text = _fallback_chatml_tool_format(messages, tools)
        verify_zero_holdout_leakage_in_text(text, f"rendered_sft:{row.get('id', 'row')}")
        rendered.append(text)
    return rendered


# ---------------------------------------------------------------------------
# Core QLoRA / PEFT + TRL SFTTrainer Routine (Shared by Local GPU & Modal L4)
# ---------------------------------------------------------------------------


def run_qlora_sft_training(
    *,
    records: list[dict[str, Any]],
    base_model: str = DEFAULT_BASE_MODEL,
    output_dir: Path = DEFAULT_CHECKPOINT_DIR,
    epochs: int = 3,
    learning_rate: float = 2e-4,
    per_device_batch_size: int = 1,
    gradient_accumulation_steps: int = 4,
    max_seq_length: int = 8192,
) -> dict[str, Any]:
    """Train a 4-bit QLoRA adapter on `records` (`S01`–`S05` ONLY) using PEFT + `trl.SFTTrainer`."""
    import torch  # type: ignore[import-not-found]
    from datasets import Dataset  # type: ignore[import-not-found]
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training  # type: ignore[import-not-found]
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments  # type: ignore[import-not-found]
    import trl  # type: ignore[import-not-found]

    for r in records:
        assert_train_seed_only(str(r.get("seed", "")))

    t0 = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)

    has_cuda = torch.cuda.is_available()
    bf16_ok = bool(has_cuda and torch.cuda.is_bf16_supported())
    compute_dtype = torch.bfloat16 if bf16_ok else (torch.float16 if has_cuda else torch.float32)

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rendered_texts = render_sft_texts(tokenizer, records)
    train_dataset = Dataset.from_list(
        [
            {"text": txt, "id": rec["id"], "seed": rec["seed"]}
            for txt, rec in zip(rendered_texts, records)
        ]
    )

    quant_config = None
    quant_mode = "lora_fp16_or_bf16"
    if has_cuda:
        try:
            from transformers import BitsAndBytesConfig  # type: ignore[import-not-found]
            import bitsandbytes as _bnb  # noqa: F401  # type: ignore[import-not-found]

            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=compute_dtype,
            )
            quant_mode = "qlora_4bit_nf4"
        except ImportError:
            quant_config = None

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "device_map": "auto" if has_cuda else None,
    }
    if quant_config is not None:
        model_kwargs["quantization_config"] = quant_config
    else:
        model_kwargs["torch_dtype"] = compute_dtype

    model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
    model.config.use_cache = False
    if quant_config is not None:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(LORA_TARGET_MODULES),
    )
    peft_probe = get_peft_model(model, lora_config)
    trainable_params, total_params = peft_probe.get_nb_trainable_parameters()

    # Compatible across TRL 0.11.x .. 0.16.x+
    sft_Trainer_cls = trl.SFTTrainer
    sft_sig = inspect.signature(sft_Trainer_cls.__init__)
    sft_config_cls = getattr(trl, "SFTConfig", None)

    common_train_kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "num_train_epochs": epochs,
        "per_device_train_batch_size": per_device_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": learning_rate,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.08,
        "logging_steps": 1,
        "save_strategy": "no",
        "bf16": bf16_ok,
        "fp16": bool(has_cuda and not bf16_ok),
        "gradient_checkpointing": has_cuda,
        "optim": "paged_adamw_8bit" if quant_config is not None else "adamw_torch",
        "report_to": [],
        "seed": 42,
    }

    trainer_kwargs: dict[str, Any] = {
        "model": peft_probe,
        "train_dataset": train_dataset,
    }
    if "processing_class" in sft_sig.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in sft_sig.parameters:
        trainer_kwargs["tokenizer"] = tokenizer

    if sft_config_cls is not None:
        cfg_sig = inspect.signature(sft_config_cls.__init__)
        if "dataset_text_field" in cfg_sig.parameters:
            common_train_kwargs["dataset_text_field"] = "text"
        if "max_seq_length" in cfg_sig.parameters:
            common_train_kwargs["max_seq_length"] = max_seq_length
        elif "max_length" in cfg_sig.parameters:
            common_train_kwargs["max_length"] = max_seq_length
        training_args = sft_config_cls(**common_train_kwargs)
    else:
        training_args = TrainingArguments(**common_train_kwargs)
        if "dataset_text_field" in sft_sig.parameters:
            trainer_kwargs["dataset_text_field"] = "text"
        if "max_seq_length" in sft_sig.parameters:
            trainer_kwargs["max_seq_length"] = max_seq_length

    trainer_kwargs["args"] = training_args
    trainer = sft_Trainer_cls(**trainer_kwargs)
    train_result = trainer.train()

    trainer.model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    loss_curve = [
        {
            "step": int(entry["step"]),
            "epoch": round(float(entry.get("epoch", 0.0)), 4),
            "loss": round(float(entry["loss"]), 6),
            "learning_rate": float(entry.get("learning_rate", 0.0)),
        }
        for entry in trainer.state.log_history
        if "loss" in entry and "step" in entry
    ]
    elapsed_s = round(time.perf_counter() - t0, 2)
    gpu_name = torch.cuda.get_device_name(0) if has_cuda else "cpu"

    return {
        "mode": "qlora_sft",
        "quantization_mode": quant_mode,
        "base_model": base_model,
        "adapter_checkpoint_dir": str(output_dir),
        "gpu_device": gpu_name,
        "train_seeds_used": sorted({str(r["seed"]).split("_")[0] for r in records}),
        "holdout_seeds_touched": [],
        "zero_leakage_verified": True,
        "num_train_trajectories": len(records),
        "epochs": epochs,
        "learning_rate": learning_rate,
        "max_seq_length": max_seq_length,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_target_modules": list(LORA_TARGET_MODULES),
        "trainable_parameters": int(trainable_params),
        "total_parameters": int(total_params),
        "trainable_pct": round(100.0 * trainable_params / max(1, total_params), 4),
        "final_train_loss": round(float(getattr(train_result, "training_loss", loss_curve[-1]["loss"] if loss_curve else 0.0)), 6),
        "initial_train_loss": loss_curve[0]["loss"] if loss_curve else None,
        "loss_curve": loss_curve,
        "wall_clock_seconds": elapsed_s,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Modal L4 Cloud GPU Entrypoint (`modal run tiebreaker/train_sft.py` & `--modal`)
# ---------------------------------------------------------------------------

if _HAS_MODAL:
    modal_image = (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install(
            "torch>=2.3.0",
            "transformers>=4.45.0",
            "peft>=0.13.0",
            "trl>=0.11.0",
            "datasets>=3.0.0",
            "bitsandbytes>=0.44.0",
            "accelerate>=0.34.0",
            "pydantic>=2.7.0",
        )
        .add_local_python_source("env_core", "tiebreaker")
    )
    app = modal.App("cryoforge-rl-tiebreaker-sft", image=modal_image)

    @app.function(gpu="L4", image=modal_image, timeout=1800)
    def train_qlora_modal_l4(
        sft_jsonl_content: str,
        base_model: str = DEFAULT_BASE_MODEL,
        epochs: int = 3,
        learning_rate: float = 2e-4,
        max_seq_length: int = 8192,
    ) -> dict[str, Any]:
        """Remote Modal L4 GPU QLoRA SFT trainer (~6 min on Qwen2.5-3B-Instruct)."""
        verify_zero_holdout_leakage_in_text(sft_jsonl_content, "modal_sft_jsonl_payload")
        records = [json.loads(line) for line in sft_jsonl_content.splitlines() if line.strip()]
        remote_ckpt = Path("/tmp/cryoforge-qwen25-3b-lora")
        metrics = run_qlora_sft_training(
            records=records,
            base_model=base_model,
            output_dir=remote_ckpt,
            epochs=epochs,
            learning_rate=learning_rate,
            per_device_batch_size=1,
            gradient_accumulation_steps=4,
            max_seq_length=max_seq_length,
        )
        metrics["mode"] = "modal_l4_qlora_sft"

        adapter_blobs: dict[str, str] = {}
        for fpath in sorted(remote_ckpt.glob("*")):
            if fpath.is_file():
                adapter_blobs[fpath.name] = base64.b64encode(fpath.read_bytes()).decode("ascii")
        return {"metrics": metrics, "adapter_files_b64": adapter_blobs}

    @app.local_entrypoint()
    def modal_cli_entrypoint(
        base_model: str = DEFAULT_BASE_MODEL,
        epochs: int = 3,
        lr: float = 2e-4,
        augment: bool = False,
    ) -> None:
        """CLI entrypoint for `modal run tiebreaker/train_sft.py`."""
        sft_path = TIEBREAKER_RUNS_DIR / "sft_train_messages.jsonl"
        if not sft_path.exists() or augment:
            collect_and_export(augment=augment)
        raw_jsonl = sft_path.read_text(encoding="utf-8")
        verify_zero_holdout_leakage_in_text(raw_jsonl, str(sft_path))
        payload = train_qlora_modal_l4.remote(
            sft_jsonl_content=raw_jsonl,
            base_model=base_model,
            epochs=epochs,
            learning_rate=lr,
        )
        _save_modal_artifacts(payload, DEFAULT_CHECKPOINT_DIR, TIEBREAKER_RUNS_DIR / "training_metrics.json")


def _save_modal_artifacts(
    payload: dict[str, Any],
    checkpoint_dir: Path,
    metrics_path: Path,
) -> dict[str, Any]:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    for fname, b64_data in payload.get("adapter_files_b64", {}).items():
        (checkpoint_dir / fname).write_bytes(base64.b64decode(b64_data))
    metrics = dict(payload["metrics"])
    metrics["adapter_checkpoint_dir"] = str(checkpoint_dir)
    metrics["downloaded_adapter_files"] = sorted(payload.get("adapter_files_b64", {}).keys())
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


# ---------------------------------------------------------------------------
# Mode C: Deterministic CPU Experience-Conditioned Policy Compiler
# ---------------------------------------------------------------------------


def compile_experience_policy(
    *,
    trajectories_dir: Path = TRAJECTORIES_DIR,
    output_dir: Path = TIEBREAKER_RUNS_DIR,
    metrics_path: Path | None = None,
    augment: bool = False,
) -> dict[str, Any]:
    """Deterministic CPU compilation of `S01`–`S05` rollouts into `experience_playbook.json`."""
    t0 = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    collection_summary = collect_and_export(
        trajectories_dir=trajectories_dir,
        output_dir=output_dir,
        augment=augment,
    )

    sft_path = Path(collection_summary["sft_train_messages_path"])
    playbook_path = Path(collection_summary["experience_playbook_path"])
    records = load_and_verify_sft_records(sft_path)

    total_turns = sum(len(r["messages"]) for r in records)
    total_tool_calls = sum(
        sum(1 for m in r["messages"] if m.get("role") == "assistant" and m.get("tool_calls"))
        for r in records
    )
    approx_tokens = sum(len(json.dumps(r["messages"], ensure_ascii=False)) // 4 for r in records)

    playbook_bytes = playbook_path.read_bytes()
    playbook_text = playbook_bytes.decode("utf-8")
    verify_zero_holdout_leakage_in_text(playbook_text, str(playbook_path))
    for bad_seed in FORBIDDEN_HOLDOUT_SEEDS:
        assert bad_seed not in playbook_text

    elapsed_s = round(time.perf_counter() - t0, 2)
    metrics = {
        "mode": "compile_experience_cpu",
        "train_seeds_used": list(EVAL_SEEDS),
        "holdout_seeds_touched": [],
        "zero_leakage_verified": True,
        "sft_examples_count": len(records),
        "preference_pairs_count": collection_summary["preference_pairs_count"],
        "total_dialogue_messages": total_turns,
        "total_assistant_tool_calls": total_tool_calls,
        "approx_sft_tokens": approx_tokens,
        "expert_mean_score_s01_s05": collection_summary["expert_mean_score"],
        "rejected_mean_score_s01_s05": collection_summary["rejected_mean_score"],
        "experience_playbook_path": str(playbook_path),
        "experience_playbook_sha256": hashlib.sha256(playbook_bytes).hexdigest(),
        "sft_train_messages_path": str(sft_path),
        "wall_clock_seconds": elapsed_s,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    target_metrics = metrics_path or (output_dir / "training_metrics.json")
    target_metrics.parent.mkdir(parents=True, exist_ok=True)
    target_metrics.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------


def _load_env_file(env_file: Path | None) -> None:
    if env_file is None or not env_file.exists():
        return
    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip("'").strip('"')
        if key and val and not os.environ.get(key):
            os.environ[key] = val


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="CryoForge RL Tiebreaker Trainer: Local QLoRA SFT, Modal L4 Cloud GPU SFT, or CPU Experience Compilation."
    )
    mode_group = parser.add_mutually_exclusive_group(required=False)
    mode_group.add_argument(
        "--local-gpu",
        action="store_true",
        help="Mode A: Run HuggingFace transformers + PEFT QLoRA + TRL SFTTrainer on local GPU.",
    )
    mode_group.add_argument(
        "--modal",
        action="store_true",
        help="Mode B: Run 4-bit QLoRA SFT remotely on a Modal L4 GPU and download adapter + loss curves.",
    )
    mode_group.add_argument(
        "--compile-experience",
        action="store_true",
        help="Mode C: Deterministic CPU compilation of S01-S05 rollouts into runs/tiebreaker/experience_playbook.json.",
    )

    parser.add_argument("--base-model", type=str, default=DEFAULT_BASE_MODEL, help="HuggingFace model ID for QLoRA SFT.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=TIEBREAKER_RUNS_DIR / "sft_train_messages.jsonl",
        help="Path to multi-turn SFT messages JSONL (generated strictly from S01-S05).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIR,
        help="Output directory for trained LoRA adapter checkpoint.",
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=TIEBREAKER_RUNS_DIR / "training_metrics.json",
        help="Output JSON path for training loss curve / compilation metrics.",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Number of SFT epochs.")
    parser.add_argument("--lr", type=float, default=2e-4, help="Peak cosine learning rate for LoRA.")
    parser.add_argument("--batch-size", type=int, default=1, help="Per-device micro-batch size.")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps.")
    parser.add_argument("--max-seq-length", type=int, default=8192, help="Maximum context length for SFTTrainer.")
    parser.add_argument("--augment", action="store_true", help="Include S01-S05 augmented rollouts when building dataset.")
    parser.add_argument("--env-file", type=Path, default=None, help="Optional .env file for MODAL_TOKEN_ID / HF_TOKEN.")
    args = parser.parse_args(argv)

    _load_env_file(args.env_file)

    # Default to deterministic CPU experience compilation if no GPU mode flag is explicitly selected
    use_compile = args.compile_experience or (not args.local_gpu and not args.modal)

    if use_compile:
        metrics = compile_experience_policy(
            output_dir=args.dataset.parent,
            metrics_path=args.metrics_path,
            augment=args.augment,
        )
        print(json.dumps(metrics, indent=2))
        return 0

    if not args.dataset.exists() or args.augment:
        collect_and_export(output_dir=args.dataset.parent, augment=args.augment)

    if args.local_gpu:
        records = load_and_verify_sft_records(args.dataset)
        metrics = run_qlora_sft_training(
            records=records,
            base_model=args.base_model,
            output_dir=args.output_dir,
            epochs=args.epochs,
            learning_rate=args.lr,
            per_device_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            max_seq_length=args.max_seq_length,
        )
        args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(metrics, indent=2))
        return 0

    if args.modal:
        if not _HAS_MODAL:
            raise RuntimeError(
                "Modal package is not installed in the current environment. "
                "Install with `pip install -e '.[tiebreaker]'` or run `--compile-experience`."
            )
        raw_jsonl = args.dataset.read_text(encoding="utf-8")
        verify_zero_holdout_leakage_in_text(raw_jsonl, str(args.dataset))
        with app.run():
            payload = train_qlora_modal_l4.remote(
                sft_jsonl_content=raw_jsonl,
                base_model=args.base_model,
                epochs=args.epochs,
                learning_rate=args.lr,
                max_seq_length=args.max_seq_length,
            )
        metrics = _save_modal_artifacts(payload, args.output_dir, args.metrics_path)
        print(json.dumps(metrics, indent=2))
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
