from __future__ import annotations

from typing import Literal

import numpy as np
from Utility.energy_allocation import allocate_pool as _allocate_pool, match_energy

from Economics.calculate_economics import DEFAULT_TARIFFS, Tariffs, calculate_economics


AllocationMode = Literal["proportional", "equal"]
EPS = 1e-12


def _energy_matrix(values, name: str, shape: tuple[int, int] | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} csak (T, U) alakú kétdimenziós tömb lehet.")
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} alakja {array.shape}, de {shape} szükséges.")
    if np.any(array < -EPS):
        raise ValueError(f"{name} nem tartalmazhat negatív energiát.")
    return np.maximum(array, 0.0)


def allocate_virtual_community_energy(
    e_grid_import_a: np.ndarray,
    e_grid_export: np.ndarray,
    *,
    allocation_mode: AllocationMode = "proportional",
) -> dict:
    """Match individually optimized imports and exports without changing dispatch.

    Inputs and outputs are kWh per timestep. Only A-tariff import is eligible for
    community energy; B/GEO circuits remain individually settled.
    """
    e_import = _energy_matrix(e_grid_import_a, "e_grid_import_a")
    e_export = _energy_matrix(e_grid_export, "e_grid_export", e_import.shape)
    shared_in = np.zeros_like(e_import)
    shared_out = np.zeros_like(e_export)

    for t in range(e_import.shape[0]):
        shared_in[t], shared_out[t] = match_energy(
            e_import[t], e_export[t], allocation_mode, allocation_mode)

    residual_import = np.maximum(e_import - shared_in, 0.0)
    residual_export = np.maximum(e_export - shared_out, 0.0)

    if not np.allclose(shared_in.sum(axis=1), shared_out.sum(axis=1), atol=1e-9):
        raise RuntimeError("A virtuális közösségi energia időlépésenként nincs egyensúlyban.")

    return {
        "e_shared_in": shared_in,
        "e_shared_out": shared_out,
        "e_grid_import_a": residual_import,
        "e_grid_export": residual_export,
    }


def settle_individual_optimization_as_community(
    e_grid_import_a: np.ndarray,
    e_grid_import_b: np.ndarray,
    e_grid_export: np.ndarray,
    *,
    user_names: list[str] | None = None,
    tariffs: Tariffs = DEFAULT_TARIFFS,
    allocation_mode: AllocationMode = "proportional",
    pairing_mode: AllocationMode | None = None,
    shared_low_cap_mode: Literal["proportional", "grid_first"] = "proportional",
) -> dict:
    """3d-K: community settlement of independently optimized dispatches."""
    e_import_a = _energy_matrix(e_grid_import_a, "e_grid_import_a")
    e_import_b = _energy_matrix(e_grid_import_b, "e_grid_import_b", e_import_a.shape)
    e_export = _energy_matrix(e_grid_export, "e_grid_export", e_import_a.shape)
    flows = allocate_virtual_community_energy(
        e_import_a,
        e_export,
        allocation_mode=allocation_mode,
    )
    economics = calculate_economics(
        e_grid_import_a=flows["e_grid_import_a"],
        e_grid_import_b=e_import_b,
        e_grid_export=flows["e_grid_export"],
        e_shared_in=flows["e_shared_in"],
        e_shared_out=flows["e_shared_out"],
        user_names=user_names,
        tariffs=tariffs,
        pairing_mode=pairing_mode or allocation_mode,
        shared_low_cap_mode=shared_low_cap_mode,
    )
    economics["timeseries"] = {**flows, "e_grid_import_b": e_import_b}
    economics["settlement_mode"] = "community"
    economics["scenario_code"] = "3d-K"
    return economics


def settle_community_optimization_as_individual(
    e_grid_import_a: np.ndarray,
    e_grid_import_b: np.ndarray,
    e_grid_export: np.ndarray,
    e_shared_to_a: np.ndarray,
    e_shared_to_b: np.ndarray,
    e_shared_out: np.ndarray,
    *,
    user_names: list[str] | None = None,
    tariffs: Tariffs = DEFAULT_TARIFFS,
) -> dict:
    """4d-I: individual settlement of a community-optimized dispatch.

    Internal purchases are reclassified as grid import on the corresponding
    meter, while internal sales are reclassified as grid export. The optimized
    device schedules are left untouched and no community payments are booked.
    """
    e_import_a = _energy_matrix(e_grid_import_a, "e_grid_import_a")
    shape = e_import_a.shape
    e_import_b = _energy_matrix(e_grid_import_b, "e_grid_import_b", shape)
    e_export = _energy_matrix(e_grid_export, "e_grid_export", shape)
    shared_a = _energy_matrix(e_shared_to_a, "e_shared_to_a", shape)
    shared_b = _energy_matrix(e_shared_to_b, "e_shared_to_b", shape)
    shared_out = _energy_matrix(e_shared_out, "e_shared_out", shape)

    individual_import_a = e_import_a + shared_a
    individual_import_b = e_import_b + shared_b
    individual_export = e_export + shared_out
    economics = calculate_economics(
        e_grid_import_a=individual_import_a,
        e_grid_import_b=individual_import_b,
        e_grid_export=individual_export,
        user_names=user_names,
        tariffs=tariffs,
    )
    economics["timeseries"] = {
        "e_grid_import_a": individual_import_a,
        "e_grid_import_b": individual_import_b,
        "e_grid_export": individual_export,
        "e_shared_to_a_reclassified": shared_a,
        "e_shared_to_b_reclassified": shared_b,
        "e_shared_out_reclassified": shared_out,
    }
    economics["settlement_mode"] = "individual"
    economics["scenario_code"] = "4d-I"
    return economics
