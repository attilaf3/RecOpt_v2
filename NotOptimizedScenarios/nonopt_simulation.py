"""Közös fizikai szimuláció a négy, nem optimalizált szcenárióhoz."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from nonopt_boiler import build_boiler_profiles
from nonopt_data import DT, NonoptInput
from nonopt_finance import settle
from nonopt_reporting import save_results
from nonopt_sharing import SharingMode, share_timestep


EPS = 1e-12


def _select_bess(data: NonoptInput, percentage: float) -> np.ndarray:
    if not 0.0 <= percentage <= 100.0:
        raise ValueError("A BESS-részaránynak 0 és 100% között kell lennie.")
    candidates = np.flatnonzero((data.e_pv.sum(axis=0) > EPS) & (data.size_bess > EPS))
    selected = np.zeros(len(data.user_names), dtype=bool)
    selected[candidates[:int(round(len(candidates) * percentage / 100.0))]] = True
    return selected


def simulate(
    data: NonoptInput,
    *,
    case_name: str,
    out_dir: str | Path,
    community_settlement: bool,
    rule_based_boiler: bool,
    boiler_tariff: str,
    bess_share_pct: float,
    sharing_mode: SharingMode = "proportional",
    boiler_min_activation_hours: float = 2.0,
) -> dict:
    tariff = boiler_tariff.upper()
    if rule_based_boiler and tariff != "A":
        raise ValueError("A rule-based bojleres eset tarifája A kell legyen.")
    if not rule_based_boiler and tariff != "B":
        raise ValueError("A baseline mért bojlerprofilja B tarifás.")
    if not rule_based_boiler and bess_share_pct > EPS:
        raise ValueError("A baseline esetben nem lehet BESS.")

    e_boiler, boiler_diag = build_boiler_profiles(
        data, rule_based=rule_based_boiler, tariff=tariff,
        min_activation_hours=boiler_min_activation_hours,
    )
    e_pv, e_base = data.e_pv, data.e_base
    n, users = e_base.shape
    dynamic_boiler = np.asarray(boiler_diag["dynamic"], dtype=bool)
    e_boiler_a = np.where(dynamic_boiler[None, :], e_boiler, 0.0)
    e_boiler_b = np.where(dynamic_boiler[None, :], 0.0, e_boiler)
    e_load = e_base + e_boiler_a + e_boiler_b
    bess_enabled = _select_bess(data, bess_share_pct)

    names = (
        "e_pv_to_load e_pv_to_bess e_bess_to_load e_bess_local_to_load "
        "e_grid_to_bess e_local_to_base e_local_to_boiler "
        "e_need_before_share e_surplus_before_share e_shared_in e_shared_out "
        "e_shared_to_base e_shared_to_boiler e_grid_to_base e_grid_to_boiler "
        "e_grid_to_boiler_a e_grid_to_boiler_b "
        "e_grid_export e_bess d_bess_ch d_bess_dis"
    ).split()
    flow = {name: np.zeros((n, users)) for name in names}

    size = np.maximum(data.size_bess, 0.0)
    soc_min = np.maximum(data.soc_bess_min, 0.0) * size
    soc_max = np.maximum(soc_min, data.soc_bess_max * size)
    soc = np.clip(0.5 * size, soc_min, soc_max)
    soc_local, soc_grid = soc.copy(), np.zeros(users)
    max_step = np.full(users, np.inf)
    valid_time = data.t_bess_min > EPS
    max_step[valid_time] = size[valid_time] / data.t_bess_min[valid_time] * DT

    for t in range(n):
        base_need = np.zeros(users)
        boiler_need = np.zeros(users)
        for u in range(users):
            use_bess = bool(bess_enabled[u])
            dispatch_load = e_base[t, u] + e_boiler_a[t, u]
            pv = e_pv[t, u]

            if use_bess:
                soc[u] *= data.eta_bess_stor[u]
                soc_local[u] *= data.eta_bess_stor[u]
                soc_grid[u] *= data.eta_bess_stor[u]
                if soc[u] < soc_min[u] - EPS:
                    grid_charge = min(
                        (soc_min[u] - soc[u]) / max(data.eta_bess_in[u], EPS),
                        max_step[u],
                    )
                    stored = grid_charge * data.eta_bess_in[u]
                    soc[u] += stored
                    soc_grid[u] += stored
                    flow["e_grid_to_bess"][t, u] = grid_charge
                    flow["d_bess_ch"][t, u] = float(grid_charge > EPS)

            pv_to_load = min(pv, dispatch_load)
            remaining_load = max(dispatch_load - pv_to_load, 0.0)
            remaining_pv = max(pv - pv_to_load, 0.0)
            flow["e_pv_to_load"][t, u] = pv_to_load

            discharge = 0.0
            local_discharge = 0.0
            if use_bess and remaining_pv > EPS:
                charge = min(
                    remaining_pv,
                    max_step[u],
                    max(soc_max[u] - soc[u], 0.0) / max(data.eta_bess_in[u], EPS),
                )
                stored = charge * data.eta_bess_in[u]
                soc[u] += stored
                soc_local[u] += stored
                remaining_pv -= charge
                flow["e_pv_to_bess"][t, u] = charge
                flow["d_bess_ch"][t, u] = float(charge > EPS)
            elif use_bess and remaining_load > EPS:
                discharge = min(
                    remaining_load,
                    max_step[u],
                    max(soc[u] - soc_min[u], 0.0) * data.eta_bess_out[u],
                )
                if discharge > EPS:
                    needed_soc = discharge / max(data.eta_bess_out[u], EPS)
                    origin = soc_local[u] + soc_grid[u]
                    local_soc = min(soc_local[u], needed_soc * soc_local[u] / origin) if origin > EPS else 0.0
                    grid_soc = min(soc_grid[u], needed_soc - local_soc)
                    remainder = needed_soc - local_soc - grid_soc
                    add_local = min(max(soc_local[u] - local_soc, 0.0), remainder)
                    local_soc += add_local
                    grid_soc += min(max(soc_grid[u] - grid_soc, 0.0), remainder - add_local)
                    soc_local[u] -= local_soc
                    soc_grid[u] -= grid_soc
                    soc[u] -= needed_soc
                    local_discharge = local_soc * data.eta_bess_out[u]
                    remaining_load -= discharge
                    flow["d_bess_dis"][t, u] = 1.0
                flow["e_bess_to_load"][t, u] = discharge
                flow["e_bess_local_to_load"][t, u] = local_discharge

            local_supply = pv_to_load + discharge
            local_base = min(e_base[t, u], local_supply)
            local_boiler = min(e_boiler_a[t, u], max(local_supply - local_base, 0.0))
            base_need[u] = max(e_base[t, u] - local_base, 0.0)
            boiler_need[u] = max(e_boiler_a[t, u] - local_boiler, 0.0)
            flow["e_local_to_base"][t, u] = local_base
            flow["e_local_to_boiler"][t, u] = local_boiler
            flow["e_need_before_share"][t, u] = base_need[u] + boiler_need[u]
            flow["e_surplus_before_share"][t, u] = remaining_pv
            flow["e_bess"][t, u] = soc[u] if use_bess else 0.0

        if community_settlement:
            received, sent = share_timestep(
                flow["e_surplus_before_share"][t], flow["e_need_before_share"][t], sharing_mode,
            )
            flow["e_shared_in"][t], flow["e_shared_out"][t] = received, sent

        received = flow["e_shared_in"][t]
        shared_base = np.minimum(base_need, received)
        shared_boiler = np.minimum(boiler_need, np.maximum(received - shared_base, 0.0))
        flow["e_shared_to_base"][t] = shared_base
        flow["e_shared_to_boiler"][t] = shared_boiler
        flow["e_grid_to_base"][t] = np.maximum(base_need - shared_base, 0.0)
        flow["e_grid_to_boiler_a"][t] = np.maximum(boiler_need - shared_boiler, 0.0)
        flow["e_grid_to_boiler_b"][t] = e_boiler_b[t]
        flow["e_grid_to_boiler"][t] = (
            flow["e_grid_to_boiler_a"][t] + flow["e_grid_to_boiler_b"][t]
        )
        flow["e_grid_export"][t] = np.maximum(
            flow["e_surplus_before_share"][t] - flow["e_shared_out"][t], 0.0,
        )

    grid_a = (
        flow["e_grid_to_base"] + flow["e_grid_to_bess"]
        + flow["e_grid_to_boiler_a"]
    )
    grid_b = flow["e_grid_to_boiler_b"].copy()

    if not np.allclose(flow["e_shared_in"].sum(axis=1), flow["e_shared_out"].sum(axis=1), atol=1e-9):
        raise RuntimeError("A közösségbe adott és onnan kapott energia nem egyezik.")
    if not np.allclose(grid_b, e_boiler_b, atol=1e-9):
        raise RuntimeError("A mért bojler nem teljes egészében a B hálózati körre került.")
    if not np.allclose(
        e_boiler_a, flow["e_local_to_boiler"] + flow["e_shared_to_boiler"]
        + flow["e_grid_to_boiler_a"], atol=1e-9,
    ):
        raise RuntimeError("A modellezett bojler energiamérlege nem zár.")
    if rule_based_boiler:
        blocked_boiler = e_boiler_a * boiler_diag["heating_blocked"]
        if np.max(np.abs(blocked_boiler)) > 1e-9:
            raise RuntimeError("A rule-based bojler energiát kapott a tiltott időszakban.")
    finance = settle(grid_a, grid_b, flow["e_grid_export"], flow["e_shared_in"], flow["e_shared_out"])
    per_user, summary = _summaries(
        data, e_boiler, e_load, flow, finance, bess_enabled,
        case_name, tariff, community_settlement, bess_share_pct, boiler_diag,
    )
    flow.update({"e_load": e_load, "e_base": e_base, "e_boiler": e_boiler,
                 "e_boiler_a": e_boiler_a, "e_boiler_b": e_boiler_b, "e_pv": e_pv,
                 "e_grid_import_a": grid_a, "e_grid_import_b": grid_b})
    save_results(
        out_dir, data.user_names, per_user, summary, flow,
        sharing=community_settlement, bess=bool(bess_enabled.any()),
        boiler_diagnostics=boiler_diag if rule_based_boiler else None,
        pair_kwh=finance["pair_kwh"], pair_payment=finance["pair_payment"],
    )
    return {"per_user": per_user, "summary": summary, "timeseries": flow}


def _summaries(data, e_boiler, e_load, flow, finance, bess_enabled, case_name, tariff,
               community_settlement, bess_share_pct, boiler_diag):
    rows = []
    for u, name in enumerate(data.user_names):
        pv = float(data.e_pv[:, u].sum())
        load = float(e_load[:, u].sum())
        self_consumed = float(flow["e_pv_to_load"][:, u].sum() + flow["e_pv_to_bess"][:, u].sum() + flow["e_shared_out"][:, u].sum())
        local_supply = float(flow["e_pv_to_load"][:, u].sum() + flow["e_bess_local_to_load"][:, u].sum() + flow["e_shared_in"][:, u].sum())
        row = {
            "user_name": name, "has_pv": bool(pv > EPS), "has_bess": bool(bess_enabled[u]),
            "has_boiler": bool(e_boiler[:, u].sum() > EPS), "base_load_kwh": float(data.e_base[:, u].sum()),
            "boiler_kwh": float(e_boiler[:, u].sum()), "total_load_kwh": load, "pv_kwh": pv,
            "group_label": "PV+BESS" if bess_enabled[u] else "PV" if pv > EPS else "Nincs PV",
            "Pmax_kw": float(np.max(e_load[:, u] / DT)) if len(e_load) else 0.0,
            "grid_import_a_kwh": float((flow["e_grid_to_base"][:, u] + flow["e_grid_to_bess"][:, u] + flow["e_grid_to_boiler_a"][:, u]).sum()),
            "grid_import_b_kwh": float(flow["e_grid_to_boiler_b"][:, u].sum()),
            "grid_export_kwh": float(flow["e_grid_export"][:, u].sum()),
            "SCI": float(np.clip(self_consumed / pv if pv > EPS else 0.0, 0.0, 1.0)),
            "SSI": float(np.clip(local_supply / load if load > EPS else 0.0, 0.0, 1.0)),
            "grid_import_cost_ft": float(finance["grid_a_cost"][u] + finance["grid_b_cost"][u]),
            "grid_export_revenue_ft": float(finance["grid_export_revenue"][u]),
            "brt_bill_ft": float(finance["bill"][u]),
        }
        if bess_enabled.any():
            row.update({"pv_to_bess_kwh": float(flow["e_pv_to_bess"][:, u].sum()),
                        "bess_to_load_kwh": float(flow["e_bess_to_load"][:, u].sum())})
        if community_settlement:
            row.update({"shared_in_kwh": float(flow["e_shared_in"][:, u].sum()),
                        "shared_out_kwh": float(flow["e_shared_out"][:, u].sum()),
                        "shared_purchase_cost_ft": float(finance["shared_energy_cost"][u] + finance["shared_rhd_cost"][u]),
                        "shared_revenue_ft": float(finance["shared_revenue"][u])})
        if bool(boiler_diag["dynamic"][u]):
            row.update({
                "boiler_setpoint_C": float(data.t_set[u]),
                "dhw_energy_shortfall_kwhth": float(
                    boiler_diag["dhw_energy_shortfall"][:, u].sum()
                ),
                "minimum_dhw_energy_margin_kwhth": float(
                    boiler_diag["dhw_energy_margin"][:, u].min()
                ),
            })
        rows.append(row)
    frame = pd.DataFrame(rows)
    summary = {"case_name": case_name, "n_users": len(rows), "boiler_tariff": tariff,
               "community_settlement": community_settlement, "bess_share_pct": float(bess_share_pct),
               "dynamic_boiler_users": int(boiler_diag["dynamic"].sum())}
    for column in frame.select_dtypes(include=[np.number]).columns:
        if column not in {
            "SCI", "SSI", "Pmax_kw", "boiler_setpoint_C",
            "minimum_dhw_energy_margin_kwhth",
        }:
            summary[column] = float(frame[column].sum())
    dynamic_setpoints = data.t_set[boiler_diag["dynamic"]]
    summary["boiler_setpoint_min_C"] = (
        float(dynamic_setpoints.min()) if dynamic_setpoints.size else 0.0
    )
    summary["boiler_setpoint_max_C"] = (
        float(dynamic_setpoints.max()) if dynamic_setpoints.size else 0.0
    )
    dynamic_margins = boiler_diag["dhw_energy_margin"][:, boiler_diag["dynamic"]]
    summary["minimum_dhw_energy_margin_kwhth"] = (
        float(dynamic_margins.min()) if dynamic_margins.size else 0.0
    )
    summary["Pmax_kw"] = float(np.max(e_load.sum(axis=1) / DT)) if len(e_load) else 0.0
    total_pv = summary.get("pv_kwh", 0.0)
    total_load = summary.get("total_load_kwh", 0.0)
    consumed = float(flow["e_pv_to_load"].sum() + flow["e_pv_to_bess"].sum() + flow["e_shared_out"].sum())
    supplied = float(flow["e_pv_to_load"].sum() + flow["e_bess_local_to_load"].sum() + flow["e_shared_in"].sum())
    summary["SCI"] = consumed / total_pv if total_pv > EPS else 0.0
    summary["SSI"] = supplied / total_load if total_load > EPS else 0.0
    return frame, summary
