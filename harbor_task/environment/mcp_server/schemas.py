"""Strict Pydantic v2 tool-argument schemas for the CryoForge RL MCP server.

Every MCP tool call is validated through a `StrictModel` subclass (`extra="forbid"`,
`allow_inf_nan=False`, strict primitive checks, no `bool`-as-`int`/`float` coercion,
no null bytes, and finite numeric bounds) BEFORE delegating to `CryoForgeEngine`.
"""

from __future__ import annotations

import math
from typing import Annotated, Any

from pydantic import (
    BeforeValidator,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    ValidationError,
)

from env_core.contracts import StrictModel

MAX_IDENTIFIER_LEN: int = 128
MAX_NUMERIC_MAGNITUDE: float = 10_000.0


def _validate_clean_str(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("Expected a string value.")
    if "\x00" in value or any(ord(ch) < 32 for ch in value):
        raise ValueError("Control characters and null bytes are forbidden.")
    stripped = value.strip()
    if not stripped:
        raise ValueError("String identifier must not be empty or whitespace-only.")
    if len(stripped) > MAX_IDENTIFIER_LEN:
        raise ValueError(f"String exceeds max length of {MAX_IDENTIFIER_LEN} characters.")
    return stripped


def _validate_optional_clean_str(value: Any) -> str | None:
    if value is None:
        return None
    return _validate_clean_str(value)


def _validate_finite_positive_float(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Expected a finite numeric value (int or float, not bool).")
    fval = float(value)
    if not math.isfinite(fval):
        raise ValueError("Numeric value must be finite (NaN and Inf are forbidden).")
    if fval <= 0.0:
        raise ValueError(f"Numeric value must be strictly positive (> 0), got {fval}.")
    if fval > MAX_NUMERIC_MAGNITUDE:
        raise ValueError(f"Numeric value {fval} exceeds upper sanity bound {MAX_NUMERIC_MAGNITUDE}.")
    return fval


CleanStr = Annotated[StrictStr, BeforeValidator(_validate_clean_str)]
OptionalCleanStr = Annotated[str | None, BeforeValidator(_validate_optional_clean_str)]
FinitePositiveFloat = Annotated[float, BeforeValidator(_validate_finite_positive_float)]


class StrictToolInput(StrictModel):
    """Base schema for all MCP tool inputs with `extra='forbid'` and `allow_inf_nan=False`."""

    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
    )


class GetEpisodeBriefInput(StrictToolInput):
    """Arguments for `get_episode_brief()`."""


class QueryHandbookInput(StrictToolInput):
    """Arguments for `query_handbook(section)`."""

    section: CleanStr = Field(default="all", description="Handbook section name or 'all'.")


class InspectOrderInput(StrictToolInput):
    """Arguments for `inspect_order(order_id)`."""

    order_id: CleanStr = Field(..., description="Target customer order identifier.")


class InspectRouteInput(StrictToolInput):
    """Arguments for `inspect_route(route_id)`."""

    route_id: CleanStr = Field(..., description="Target transport route identifier.")


class GetFactoryStateInput(StrictToolInput):
    """Arguments for `get_factory_state()`."""


class RunPhysicsSimInput(StrictToolInput):
    """Arguments for `run_physics_sim(...)`."""

    order_id: CleanStr
    unit_count: StrictInt = Field(..., ge=1, le=100)
    unit_volume_m3: FinitePositiveFloat
    design_pressure_mpa: FinitePositiveFloat
    wall_thickness_mm: FinitePositiveFloat
    insulation: CleanStr
    fittings: CleanStr


class ReserveBayAndOrderInput(StrictToolInput):
    """Arguments for `reserve_bay_and_order(...)`."""

    order_id: CleanStr
    bay_id: CleanStr
    bay_start_week: StrictInt = Field(..., ge=1, le=520)
    unit_count: StrictInt = Field(..., ge=1, le=100)
    unit_volume_m3: FinitePositiveFloat
    design_pressure_mpa: FinitePositiveFloat
    wall_thickness_mm: FinitePositiveFloat
    insulation: CleanStr
    fittings: CleanStr


class AssignTransportInput(StrictToolInput):
    """Arguments for `assign_transport(...)`."""

    order_id: CleanStr
    route_id: CleanStr
    use_escort_permit: StrictBool = Field(default=False)


class ReviseOrderInput(StrictToolInput):
    """Arguments for `revise_order(...)`."""

    order_id: CleanStr
    bay_id: CleanStr
    bay_start_week: StrictInt = Field(..., ge=1, le=520)
    unit_count: StrictInt = Field(..., ge=1, le=100)
    unit_volume_m3: FinitePositiveFloat
    design_pressure_mpa: FinitePositiveFloat
    wall_thickness_mm: FinitePositiveFloat
    insulation: CleanStr
    fittings: CleanStr
    route_id: OptionalCleanStr = Field(default=None)
    use_escort_permit: StrictBool | None = Field(default=None)


class FinalizeCommissioningInput(StrictToolInput):
    """Arguments for `finalize_commissioning()`."""


TOOL_INPUT_SCHEMAS: dict[str, type[StrictToolInput]] = {
    "get_episode_brief": GetEpisodeBriefInput,
    "query_handbook": QueryHandbookInput,
    "inspect_order": InspectOrderInput,
    "inspect_route": InspectRouteInput,
    "get_factory_state": GetFactoryStateInput,
    "run_physics_sim": RunPhysicsSimInput,
    "reserve_bay_and_order": ReserveBayAndOrderInput,
    "assign_transport": AssignTransportInput,
    "revise_order": ReviseOrderInput,
    "finalize_commissioning": FinalizeCommissioningInput,
}


def format_validation_error(tool_name: str, exc: ValidationError) -> dict[str, Any]:
    """Convert a Pydantic ValidationError into a clean JSON-serializable error response."""
    issues: list[dict[str, Any]] = []
    for err in exc.errors():
        loc = ".".join(str(part) for part in err.get("loc", ()))
        issues.append(
            {
                "field": loc or "<root>",
                "type": str(err.get("type", "validation_error")),
                "message": str(err.get("msg", "Invalid parameter value.")),
            }
        )
    return {
        "ok": False,
        "error": "VALIDATION_ERROR",
        "tool": tool_name,
        "message": f"Invalid arguments for tool '{tool_name}'.",
        "details": issues,
    }


def validate_tool_arguments(
    tool_name: str,
    raw_arguments: dict[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Validate raw tool arguments against the tool's `StrictModel` schema.

    Returns `(validated_kwargs, None)` on success, or `(None, error_response_dict)` on failure.
    """
    schema_cls = TOOL_INPUT_SCHEMAS.get(tool_name)
    if schema_cls is None:
        return None, {
            "ok": False,
            "error": "UNKNOWN_TOOL",
            "tool": tool_name,
            "message": f"Unknown tool '{tool_name}'.",
        }
    if raw_arguments is not None and not isinstance(raw_arguments, dict):
        return None, {
            "ok": False,
            "error": "VALIDATION_ERROR",
            "tool": tool_name,
            "message": "Tool arguments payload must be a JSON object.",
            "details": [],
        }
    try:
        validated = schema_cls.model_validate(raw_arguments or {})
        return validated.model_dump(), None
    except ValidationError as exc:
        return None, format_validation_error(tool_name, exc)
