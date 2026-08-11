"""Deterministic failure classification and repair policy for Phase III.

The policy deliberately separates a fixed-point *state update* from a design
repair.  It returns one auditable action at a time in the order declared in the
case file.  Callers apply the action, re-run the affected native model, and
then repeat the complete coupled solve.  If no declared action remains, the
case receives the terminal status ``failed_no_feasible_repair`` and must not be
exported as accepted.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class RepairDecision:
    priority: int
    action: str
    sector: str
    target_id: str
    reason: str
    phase: str
    automatic: bool
    terminal: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def next_catalogue_value(current: float, catalogue: Iterable[float]) -> float | None:
    """Return the first declared catalogue value strictly above ``current``."""
    for value in sorted(float(item) for item in catalogue):
        if value > float(current) + 1e-9:
            return value
    return None


def validate_interface_state(state: Any) -> list[str]:
    """Return deterministic interface-contract errors before any solver call."""
    required = {"interface_id", "bus_id", "sector", "asset_id", "power_mw"}
    missing_columns = sorted(required - set(state.columns))
    errors = [f"missing_column:{name}" for name in missing_columns]
    if missing_columns:
        return errors
    if state.empty:
        errors.append("no_active_interfaces")
    if state["interface_id"].astype(str).duplicated().any():
        errors.append("duplicate_interface_id")
    if state["asset_id"].astype(str).duplicated().any():
        errors.append("duplicate_asset_id")
    if state["bus_id"].isna().any() or state["bus_id"].astype(str).str.len().eq(0).any():
        errors.append("missing_bus_id")
    if state["power_mw"].isna().any() or (state["power_mw"].astype(float) < 0).any():
        errors.append("invalid_interface_power")
    return errors


def phase_return_for_failure(failure_code: str) -> RepairDecision:
    """Map an exhausted local repair to one explicit upstream regeneration step."""
    mapping = {
        "electricity": ("return_to_phase_ii_a", "Phase II-A"),
        "drinking_water": ("return_to_phase_ii_b", "Phase II-B"),
        "wastewater": ("return_to_phase_ii_c", "Phase II-C"),
        "district_heating": ("return_to_phase_ii_d", "Phase II-D"),
        "interface": ("return_to_phase_iii_a", "Phase III-A"),
        "fixed_point": ("retry_with_declared_numerical_damping", "Phase III-B"),
    }
    action, phase = mapping.get(
        failure_code, ("return_to_responsible_generation_step", "Phase II")
    )
    return RepairDecision(
        priority=5,
        action=action,
        sector=failure_code,
        target_id="generated_case",
        reason=f"all bounded Phase-III repairs for {failure_code} were exhausted",
        phase=phase,
        automatic=False,
    )


def terminal_failure(reason: str, sector: str = "coupled_system") -> RepairDecision:
    """Return the mandatory terminal outcome after the complete ladder fails."""
    return RepairDecision(
        priority=6,
        action="failed_no_feasible_repair",
        sector=sector,
        target_id="generated_case",
        reason=reason,
        phase="release_gate",
        automatic=False,
        terminal=True,
    )


def declared_policy(config: dict[str, Any]) -> dict[str, Any]:
    """Return and minimally validate the versioned policy from the case file."""
    policy = config["bidirectional_coupling"]["repair_policy"]
    required = {
        "policy_id", "priority_order", "relaxation_schedule",
        "water_speed_command", "heat_speed_command",
        "heat_lift_factor", "maximum_component_upgrades_per_sector",
        "maximum_total_repairs", "terminal_status",
    }
    missing = sorted(required - set(policy))
    if missing:
        raise ValueError(f"repair_policy is missing required keys: {missing}")
    expected = [
        "interface_contract",
        "numerical_damping",
        "bounded_operating_setpoint",
        "next_catalogue_component",
        "phase_ii_topology_regeneration",
        "failed_no_feasible_repair",
    ]
    if list(policy["priority_order"]) != expected:
        raise ValueError(
            "repair_policy.priority_order must preserve the declared six-step hierarchy"
        )
    if policy["terminal_status"] != "failed_no_feasible_repair":
        raise ValueError("repair_policy.terminal_status must be failed_no_feasible_repair")
    return policy
