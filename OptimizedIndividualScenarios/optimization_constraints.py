"""Reusable PuLP building blocks for individual optimization models.

These helpers add variables and constraints without encoding a device-specific
dispatch policy. This keeps tariff bookkeeping and grid-direction handling
consistent across the boiler, BESS and HP models.
"""

from dataclasses import dataclass
from typing import Any

import pulp


@dataclass(frozen=True)
class TwoTierTariffBlock:
    """Variables for a cumulative low-price block and its high-price overflow."""

    low_step: list[pulp.LpVariable]
    high_step: list[pulp.LpVariable]
    remaining_low: list[pulp.LpVariable]


def create_two_tier_tariff_block(
    horizon: int,
    low_cap_kwh: float,
    prefix: str,
) -> TwoTierTariffBlock:
    """Create variables for a cumulative two-tier import tariff."""
    if horizon < 0:
        raise ValueError("horizon must be non-negative")
    if low_cap_kwh < 0:
        raise ValueError("low_cap_kwh must be non-negative")

    steps = range(horizon)
    return TwoTierTariffBlock(
        low_step=[pulp.LpVariable(f"{prefix}_low_step_{t}", lowBound=0) for t in steps],
        high_step=[pulp.LpVariable(f"{prefix}_high_step_{t}", lowBound=0) for t in steps],
        remaining_low=[
            pulp.LpVariable(
                f"{prefix}_remaining_low_{t}",
                lowBound=0,
                upBound=float(low_cap_kwh),
            )
            for t in steps
        ],
    )


def add_two_tier_step_constraints(
    problem: pulp.LpProblem,
    block: TwoTierTariffBlock,
    step: int,
    import_energy: Any,
    low_cap_kwh: float,
    prefix: str,
) -> None:
    """Add one timestep's split and cumulative low-block constraints."""
    low = block.low_step[step]
    high = block.high_step[step]
    remaining = block.remaining_low[step]

    problem += low + high == import_energy, f"{prefix}_step_split_{step}"
    if step == 0:
        problem += remaining == float(low_cap_kwh) - low, f"{prefix}_remaining_low_init"
        problem += low <= float(low_cap_kwh), f"{prefix}_low_step_cap_{step}"
    else:
        previous = block.remaining_low[step - 1]
        problem += remaining == previous - low, f"{prefix}_remaining_low_balance_{step}"
        problem += low <= previous, f"{prefix}_low_step_cap_{step}"


def add_import_export_exclusivity(
    problem: pulp.LpProblem,
    import_flow: Any,
    export_flow: Any,
    direction_selector: Any,
    max_import: float,
    max_export: float,
    step: int,
    prefix: str = "grid",
    selector_is_import: bool = True,
) -> None:
    """Prevent simultaneous grid import/export with a binary direction selector."""
    if selector_is_import:
        import_gate = direction_selector
        export_gate = 1 - direction_selector
    else:
        import_gate = 1 - direction_selector
        export_gate = direction_selector

    problem += import_flow <= float(max_import) * import_gate, f"{prefix}_import_gate_{step}"
    problem += export_flow <= float(max_export) * export_gate, f"{prefix}_export_gate_{step}"


__all__ = [
    "TwoTierTariffBlock",
    "add_import_export_exclusivity",
    "add_two_tier_step_constraints",
    "create_two_tier_tariff_block",
]
