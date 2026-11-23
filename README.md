<div align="center">

# 🌟 QuantumForge AI

### **Next-Generation Multi-Task Planning Benchmark for Autonomous AI Agents**

*Developed by Keshav*

---

[![Live Dashboard](https://img.shields.io/badge/🚀_Live-Dashboard-00D9FF?style=for-the-badge)](https://keshav.github.io/QuantumForge-AI/)
[![License](https://img.shields.io/badge/License-MIT-success?style=for-the-badge)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![MCP](https://img.shields.io/badge/MCP-Compatible-purple?style=for-the-badge)](https://modelcontextprotocol.io)

---

</div>

## 🎯 **What is QuantumForge AI?**

**QuantumForge AI** is a cutting-edge benchmark that evaluates the ability of AI agents to handle **real-world complexity**: multi-constraint planning, resource optimization, and **dynamic recovery from mid-task disruptions**.

Unlike traditional static benchmarks, QuantumForge AI places agents in a **deterministic industrial commissioning environment** called **CryoForge**, where they must:

✨ **Schedule 4 interdependent jobs** across 2 execution lanes  
⚡ **Optimize under strict budgets** and customer SLA deadlines  
🔄 **Adapt dynamically** when mid-episode disruptions occur  
🎲 **Work with limited simulation calls** (only 3 attempts!)  

### 🔥 **Why This Matters**

Frontier AI models struggle with this benchmark:
- **Reference Planner Score:** `0.9491` (analytical solver)
- **Frontier Agent (Gemini 3.8):** `0.3162 ± 0.2496` (15 live rollouts)
- **Performance Gap:** **+0.6329** (63% worse than optimal!)

The issue? Models exhaust simulation budget on guessing, double-book resources, and fail to adjust plans after disruptions—despite having **all rules transparently available** from the start.

---

## 🏗️ **Architecture Overview**

```
┌─────────────────────────────────────────────────────────────┐
│                    QUANTUMFORGE AI                          │
│                                                             │
│   ┌───────────┐    ┌──────────────┐    ┌──────────────┐  │
│   │  Agent    │───▶│  CryoForge   │───▶│  Verifier    │  │
│   │  (AI/LLM) │    │  Environment │    │  (HMAC)      │  │
│   └───────────┘    └──────────────┘    └──────────────┘  │
│                                                             │
│   Tools Available via MCP:                                 │
│   • query_handbook() - Get all rules & constraints         │
│   • schedule_job() - Book a job on a lane                  │
│   • run_physics_sim() - Simulate outcome (3 calls max!)    │
│   • finalize_commissioning() - Submit & score              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 🎨 **Key Features**

| Feature | Description |
|---------|-------------|
| 🧩 **Multi-Resource Planning** | Coordinate 4 jobs across 2 parallel execution lanes |
| 💥 **Mid-Episode Disruptions** | System changes at Step 4—agents must replan dynamically |
| 🔒 **Zero-Trust Scoring** | HMAC-SHA256 verification prevents reward hacking (8/8 exploits blocked) |
| 🎯 **Transparent Rules** | All constraints available via API—success requires reasoning, not memorization |
| 🧪 **Deterministic Physics** | Reproducible scenarios for scientific evaluation |
| 📊 **Rich Visualization** | Interactive web dashboard with trajectory analysis |

---

## 🚀 **Quick Start**

### **1️⃣ Install & Launch Dashboard (25 seconds, no API key needed)**

```bash
git clone https://github.com/keshav/QuantumForge-AI.git
cd QuantumForge-AI
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt  # Windows
# source .venv/bin/pip install -r requirements.txt  # Linux/macOS
python scripts/run_eval.py --verify-recorded --serve
```

✅ Opens interactive dashboard at **`http://127.0.0.1:8095`**  
✅ Re-verifies 17 unit tests  
✅ Runs Reference Planner & baselines  
✅ Audits all recorded trajectories  

### **2️⃣ Install as Agent Skill**

```bash
npx skills add https://github.com/keshav/QuantumForge-AI.git
```

**Three execution modes:**
1. **Offline Verify & Dashboard** - No API key needed
2. **Live Agent Playtest via MCP** - Connect your active AI session
3. **Standalone LLM Benchmarking** - Requires API key

### **3️⃣ Run in Harbor Container Sandbox**

```bash
# Verify Reference Planner (no API key)
harbor run -p ./harbor_task -a oracle --jobs-dir runs/harbor_jobs

# Benchmark external model (API key required)
harbor run -p ./harbor_task -a opencode -m gemini/gemini-3.8-flash --jobs-dir runs/harbor_jobs
```

---

## 📊 **Benchmark Results**

<div align="center">

### **Performance Comparison**

| System | Score (S01-S05) | Notes |
|--------|-----------------|-------|
| 🏆 **Reference Planner** | **0.9491** | Analytical solver using public MCP tools |
| 🤖 **Gemini 3.8 Flash** | **0.3162 ± 0.2496** | 15 live rollouts (min: 0.016, max: 0.892) |
| 📉 **Performance Gap** | **+0.6329** | 63% optimization opportunity |
| ⚫ **Do-Nothing Baseline** | **0.0000** | Zero credit for unbooked episodes |
| 🛑 **Stop-Early Baseline** | **0.0800** | Single-job cherry-picking |
| 🛡️ **Anti-Cheat Suite** | **8/8 Blocked** | All reward-hacking attempts prevented |

</div>

### **📈 Held-Out Learning Results**

Training on scenarios S01-S05 improved small model performance on unseen scenarios S06-S08:

- **Before Training:** `0.1400` (3/12 valid jobs)
- **After Training:** `0.3400` (6/12 valid jobs)
- **Improvement:** **+142.9%** relative gain! 🎉

---

## 🌍 **Evaluation Scenarios**

| Scenario | Disruption Type | Reference | Frontier Mean | Gap |
|----------|-----------------|-----------|---------------|-----|
| **S01** | Supplier delay + route capacity cut | 0.9445 | 0.3400 | +0.6045 |
| **S02** | Lane outage + budget freeze | 0.9483 | 0.2488 | +0.6995 |
| **S03** | Efficiency degradation + permit cut | 0.9514 | 0.1627 | +0.7887 |
| **S04** | Safety uprate + clearance lock | 0.9480 | 0.4334 | +0.5146 |
| **S05** | Emergency budget cut + hoist cap | 0.9533 | 0.3963 | +0.5570 |
| **Mean** | **5 scenarios, 15 rollouts** | **0.9491** | **0.3162** | **+0.6329** |

---

## 🗂️ **Repository Structure**

```
QuantumForge-AI/
├── 📁 .agents/skills/        # Portable agent skills
├── 📁 .github/workflows/     # CI/CD & GitHub Pages deployment
├── 📁 docs/assets/           # Visualizations & figures
├── 📁 env_core/              # Physics engine & verifier
│   ├── contracts.py
│   ├── engine.py
│   ├── physics.py
│   ├── scenarios.py
│   └── verifier.py
├── 📁 eval/                  # Evaluation suite
│   ├── baselines.py
│   ├── reference_solver.py
│   └── exploit_tests.py
├── 📁 harbor_task/           # Harbor v1.0 task container
├── 📁 mcp_server/            # FastMCP stdio/SSE server
├── 📁 runs/                  # Recorded trajectories & scorecards
├── 📁 scripts/               # Evaluation & dashboard launcher
├── 📁 tests/                 # 17 unit tests
├── 📁 viewer/                # React + TypeScript dashboard
├── 📄 README.md              # This file
├── 📄 requirements.txt       # Python dependencies
└── 📄 pyproject.toml         # Project configuration
```

---

## 🔬 **Testing & Verification**

All test cases pass with HMAC-SHA256 verification:

```bash
pytest tests/ -v
```

**Test Coverage:**
- ✅ Deterministic physics validation
- ✅ Anti-cheat exploit prevention (8 attack vectors)
- ✅ Verifier state signature checks
- ✅ Scenario reproducibility
- ✅ MCP tool contract compliance

---

## 🎓 **Why Use QuantumForge AI?**

### **For Researchers**
- 📚 Transparent, deterministic environment
- 🔬 Reproducible results with HMAC verification
- 📊 Rich evaluation metrics & trajectory analysis
- 🧪 Scientific benchmark for multi-task planning

### **For AI Developers**
- 🛠️ MCP-compatible integration
- 🐳 Docker containerization support
- 🔌 Easy agent skill installation
- 📈 Progressive difficulty scenarios

### **For Industry**
- 🏭 Real-world constraint modeling
- 💼 Supply chain & logistics applications
- ⚡ Dynamic replanning under disruptions
- 🎯 Resource optimization at scale

---

## 📖 **Citation**

If you use QuantumForge AI in your research, please cite:

```bibtex
@software{quantumforge_ai_2025,
  title   = {QuantumForge AI: Multi-Task Planning Benchmark for Autonomous AI Agents},
  author  = {Keshav},
  year    = {2025},
  url     = {https://github.com/keshav/QuantumForge-AI},
  note    = {Next-generation AI agent evaluation framework}
}
```

---

## 🤝 **Contributing**

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📜 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🌐 **Links & Resources**

- 🌟 **Live Dashboard:** [https://keshav.github.io/QuantumForge-AI/](https://keshav.github.io/QuantumForge-AI/)
- 📚 **Documentation:** [Agent Skill Guide](.agents/skills/quantumforge-ai/SKILL.md)
- 📊 **Scorecard:** [runs/SCORECARD.md](runs/SCORECARD.md)
- 🐳 **Harbor Task:** [harbor_task/task.toml](harbor_task/task.toml)

---

<div align="center">

### 💫 **Built with Precision by Keshav**

*Pushing the boundaries of autonomous AI agent capabilities*

[![GitHub Stars](https://img.shields.io/github/stars/keshav/QuantumForge-AI?style=social)](https://github.com/keshav/QuantumForge-AI)
[![Twitter Follow](https://img.shields.io/twitter/follow/keshav?style=social)](https://twitter.com/keshav)

**[⭐ Star this repo](https://github.com/keshav/QuantumForge-AI)** if you find it useful!

</div>
