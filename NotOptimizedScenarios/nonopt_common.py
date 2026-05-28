from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

import matplotlib.pyplot as plt


DT = 0.25  # 15 perc


def _as_15min_energy(profile: np.ndarray) -> np.ndarray:

    p = np.asarray(profile, dtype=float).ravel()
    if p.size != 35040:
        raise ValueError(f"A profil hossza {p.size}, de 35040 kell.")
    return np.maximum(p, 0.0)


def _find_user_yaml(roots: Iterable[os.PathLike], name: str) -> Optional[Path]:
    for r in roots:
        for cand in (name, f"{name}.yaml"):
            p = Path(r) / cand
            if p.exists():
                return p
    return None


def build_inputs(
    sim_yaml_path: os.PathLike,
    profiles_csv_path: os.PathLike,
    dhw_profile_path: os.PathLike,
    max_users: int = 10,
    search_roots: Iterable[os.PathLike] | None = None,
) -> Tuple[
    np.ndarray,  # e_pv (35040, U) [kWh / 15 perc]
    np.ndarray,  # e_ue (35040, U) [kWh / 15 perc]
    np.ndarray,  # e_el_heater (35040, U) [kWh / 15 perc]
    np.ndarray,  # size_bess (U,)
    np.ndarray,  # eta_bess_in (U,)
    np.ndarray,  # eta_bess_out (U,)
    np.ndarray,  # eta_bess_stor (U,)
    np.ndarray,  # soc_bess_min (U,)
    np.ndarray,  # soc_bess_max (U,)
    np.ndarray,  # t_bess_min (U,)
    list[float],
    list[float],
    list[float],
    list[float],
    list[float],
    list[float],
    list[float],
    list[str],
]:

    sim_yaml_path = Path(sim_yaml_path)
    profiles_csv_path = Path(profiles_csv_path)

    if search_roots is None:
        search_roots = [sim_yaml_path.parent / "Users", sim_yaml_path.parent]

    sim = yaml.safe_load(sim_yaml_path.read_text(encoding="utf-8")) or {}
    users_list = list(sim.get("users_list", []))[: int(max_users)]

    EXCLUDE = {"battery", "bess", "community"}
    users_list = [u for u in users_list if str(u).strip().lower() not in EXCLUDE]
    if not users_list:
        raise RuntimeError("A simulation YAML nem tartalmaz users_list-et vagy max_users=0.")

    df = pd.read_csv(profiles_csv_path, index_col=0)
    df.columns = [str(c) for c in df.columns]
    df = df[[c for c in df.columns if c.lower() not in EXCLUDE]]

    e_pv_cols: list[np.ndarray] = []
    e_ue_cols: list[np.ndarray] = []
    e_el_heater_cols: list[np.ndarray] = []

    user_names = []

    size_bess = []
    eta_bess_in_u = []
    eta_bess_out_u = []
    eta_bess_stor_u = []
    soc_bess_min_u = []
    soc_bess_max_u = []
    t_bess_min_u = []

    for user_key in users_list:
        ypath = _find_user_yaml(search_roots, user_key)
        if not ypath:
            print(f"[WARN] YAML nem található: {user_key} — kihagyom.")
            continue

        u = yaml.safe_load(ypath.read_text(encoding="utf-8")) or {}
        units = u.get("units") or {}
        name = (units.get("name") or {}).get("name", str(user_key))
        user_names.append(name)

        # --- UE (villamos fogyasztás), 15 perc [kWh / lépés] ---
        ue = units.get("ue") or {}
        ue_prof = str(ue.get("profile")) if ue.get("profile") is not None else None
        if ue_prof and ue_prof in df.columns:
            e_ue_cols.append(_as_15min_energy(df[ue_prof].to_numpy()))
        else:
            e_ue_cols.append(np.zeros(35040, dtype=float))

        # --- PV (termelés), 15 perc [kWh / lépés] ---
        n_pv = 1.0
        pv = units.get("pv") or {}
        pv_prof = str(pv.get("profile")) if pv.get("profile") is not None else None
        if pv_prof and pv_prof in df.columns:
            e_pv_cols.append(_as_15min_energy(df[pv_prof].to_numpy()) / n_pv)
        else:
            e_pv_cols.append(np.zeros(35040, dtype=float))

        # --- BESS paraméterek ---
        bess = units.get("bess") or {}
        size_bess.append(float(bess.get("bess_size", 0.0)))
        eta_bess_in_u.append(float(bess.get("eta_bess_in", 0.98)))
        eta_bess_out_u.append(float(bess.get("eta_bess_out", 0.96)))
        eta_bess_stor_u.append(float(bess.get("eta_bess_stor", 0.995)))
        soc_bess_min_u.append(float(bess.get("soc_bess_min", 0.1)))
        soc_bess_max_u.append(float(bess.get("soc_bess_max", 0.9)))
        t_bess_min_u.append(float(bess.get("t_bess_min", 2.0)))

        # nincs HSS, az UT profil is 15 perces energia [kWh / lépés]
        heater = units.get("ut") or {}

        p_el_heater_prof = str(heater.get("profile")) if heater.get("profile") is not None else None
        if p_el_heater_prof and p_el_heater_prof in df.columns:
            e_el_heater_cols.append(_as_15min_energy(df[p_el_heater_prof].to_numpy()))
        else:
            e_el_heater_cols.append(np.zeros(35040, dtype=float))


    if not user_names:
        raise RuntimeError("Nincs érvényes felhasználó.")

    U = len(user_names)

    e_pv = np.column_stack(e_pv_cols).astype(float)
    e_ue = np.column_stack(e_ue_cols).astype(float)

    e_el_heater = (
        np.column_stack(e_el_heater_cols).astype(float)
        if e_el_heater_cols
        else np.zeros((35040, U), dtype=float)
    )

    return (
        e_pv,
        e_ue,
        e_el_heater,
        np.asarray(size_bess, dtype=float),
        np.asarray(eta_bess_in_u, dtype=float),
        np.asarray(eta_bess_out_u, dtype=float),
        np.asarray(eta_bess_stor_u, dtype=float),
        np.asarray(soc_bess_min_u, dtype=float),
        np.asarray(soc_bess_max_u, dtype=float),
        np.asarray(t_bess_min_u, dtype=float),
        user_names,
    )


LOW_TARIFF_LIMIT_KWH = 2523.0
LOW_TARIFF_FT_PER_KWH = 36.0
HIGH_TARIFF_FT_PER_KWH = 71.0

B_LOW_TARIFF_LIMIT_KWH = 2523.0
B_LOW_TARIFF_FT_PER_KWH = 23.0
B_HIGH_TARIFF_FT_PER_KWH = 61.0

EXPORT_FT_PER_KWH = 5.0


def _calc_import_cost(
    e_import_steps: np.ndarray,
    low_limit_kwh: float,
    low_price_ft_per_kwh: float,
    high_price_ft_per_kwh: float,
) -> tuple[float, float, float]:

    remaining_low = float(low_limit_kwh)
    low_kwh = 0.0
    high_kwh = 0.0
    cost_ft = 0.0

    for e_imp in np.maximum(np.asarray(e_import_steps, dtype=float), 0.0):
        low_part = min(float(e_imp), max(remaining_low, 0.0))
        high_part = max(float(e_imp) - low_part, 0.0)

        cost_ft += low_part * low_price_ft_per_kwh
        cost_ft += high_part * high_price_ft_per_kwh

        low_kwh += low_part
        high_kwh += high_part
        remaining_low -= low_part

    return low_kwh, high_kwh, cost_ft


def calc_bill_15min_brutto(
    e_grid_to_load: np.ndarray,
    e_inj: np.ndarray,
    e_grid_to_boiler: np.ndarray | None = None,
) -> dict:
    """
    15 perces bruttó elszámolás.

    A bemenetek kWh / időlépés egységűek.
    - Normál háztartási fogyasztás: A tarifa, 36 / 71 Ft/kWh.
    - Bojler: B tarifa, 23 / 61 Ft/kWh.

    Ha nincs külön `e_grid_to_boiler`, akkor minden import A-tarifára kerül.
    Ha van, akkor `e_grid_to_load` a teljes import, ebből levonjuk a bojler
    importját, és csak a maradék kerül A-tarifára.
    """
    e_grid_to_load = np.maximum(np.asarray(e_grid_to_load, dtype=float), 0.0)
    e_inj = np.maximum(np.asarray(e_inj, dtype=float), 0.0)

    if e_grid_to_boiler is None:
        e_grid_to_boiler = np.zeros_like(e_grid_to_load)
    else:
        e_grid_to_boiler = np.maximum(np.asarray(e_grid_to_boiler, dtype=float), 0.0)

    e_grid_to_boiler = np.minimum(e_grid_to_boiler, e_grid_to_load)
    e_grid_to_base = np.maximum(e_grid_to_load - e_grid_to_boiler, 0.0)

    a_low_kwh, a_high_kwh, a_import_cost_ft = _calc_import_cost(
        e_import_steps=e_grid_to_base,
        low_limit_kwh=LOW_TARIFF_LIMIT_KWH,
        low_price_ft_per_kwh=LOW_TARIFF_FT_PER_KWH,
        high_price_ft_per_kwh=HIGH_TARIFF_FT_PER_KWH,
    )
    b_low_kwh, b_high_kwh, b_import_cost_ft = _calc_import_cost(
        e_import_steps=e_grid_to_boiler,
        low_limit_kwh=B_LOW_TARIFF_LIMIT_KWH,
        low_price_ft_per_kwh=B_LOW_TARIFF_FT_PER_KWH,
        high_price_ft_per_kwh=B_HIGH_TARIFF_FT_PER_KWH,
    )

    export_revenue_ft = e_inj.sum() * EXPORT_FT_PER_KWH
    import_cost_ft = a_import_cost_ft + b_import_cost_ft
    brt_bill_ft = import_cost_ft - export_revenue_ft

    return {
        "grid_import_low_kwh": a_low_kwh + b_low_kwh,
        "grid_import_high_kwh": a_high_kwh + b_high_kwh,
        "grid_import_a_low_kwh": a_low_kwh,
        "grid_import_a_high_kwh": a_high_kwh,
        "grid_import_b_low_kwh": b_low_kwh,
        "grid_import_b_high_kwh": b_high_kwh,
        "grid_import_a_kwh": float(e_grid_to_base.sum()),
        "grid_import_b_kwh": float(e_grid_to_boiler.sum()),
        "import_cost_a_ft": a_import_cost_ft,
        "import_cost_b_ft": b_import_cost_ft,
        "import_cost_ft": import_cost_ft,
        "export_revenue_ft": export_revenue_ft,
        "brt_bill_ft": brt_bill_ft,
    }


def split_grid_import_base_boiler(
    e_base_load: np.ndarray,
    e_boiler_load: np.ndarray,
    e_pv_to_load: np.ndarray,
    e_bess_to_load: np.ndarray,
) -> dict:
    """
    A teljes lokális ellátást először az alapfogyasztásra, majd a bojlerre osztja.

    Így a bojler csak akkor kap PV/BESS energiát, ha az adott 15 perces lépésben
    az alapfogyasztás már ki van szolgálva. A maradék bojlerigény B-tarifás
    hálózati import lesz.
    """
    e_base_load = np.maximum(np.asarray(e_base_load, dtype=float), 0.0)
    e_boiler_load = np.maximum(np.asarray(e_boiler_load, dtype=float), 0.0)
    local_supply = np.maximum(
        np.asarray(e_pv_to_load, dtype=float) + np.asarray(e_bess_to_load, dtype=float),
        0.0,
    )

    e_local_to_base = np.minimum(e_base_load, local_supply)
    remaining_local = np.maximum(local_supply - e_local_to_base, 0.0)
    e_local_to_boiler = np.minimum(e_boiler_load, remaining_local)

    e_grid_to_base = np.maximum(e_base_load - e_local_to_base, 0.0)
    e_grid_to_boiler = np.maximum(e_boiler_load - e_local_to_boiler, 0.0)

    return {
        "e_local_to_base": e_local_to_base,
        "e_local_to_boiler": e_local_to_boiler,
        "e_grid_to_base": e_grid_to_base,
        "e_grid_to_boiler": e_grid_to_boiler,
    }


def simulate_one_user_greedy(
    e_load: np.ndarray,
    e_pv: np.ndarray,
    use_bess: bool,
    bess_size_kwh: float,
    eta_bess_in: float,
    eta_bess_out: float,
    eta_bess_stor: float,
    soc_bess_min: float,
    soc_bess_max: float,
    t_bess_min_h: float,
    e_boiler_load: np.ndarray | None = None,
    boiler_tariff: str = "B",
) -> dict:
    T = len(e_load)

    e_load = np.asarray(e_load, dtype=float)
    e_pv = np.asarray(e_pv, dtype=float)
    if e_boiler_load is None:
        e_boiler_load = np.zeros_like(e_load)
    else:
        e_boiler_load = np.maximum(np.asarray(e_boiler_load, dtype=float), 0.0)
    e_base_load = np.maximum(e_load - e_boiler_load, 0.0)

    boiler_tariff = str(boiler_tariff).upper().strip()
    if boiler_tariff not in {"A", "B"}:
        raise ValueError(f"boiler_tariff csak 'A' vagy 'B' lehet, nem: {boiler_tariff}")

    if boiler_tariff == "A":
        # A tarifás bojler: sima fogyasztó, kaphat PV-t és BESS-t.
        e_dispatch_load = e_load
    else:
        # B tarifás bojler: külön mérő, külön áramkör.
        # PV/BESS csak az általános fogyasztást látja el.
        e_dispatch_load = e_base_load

    e_pv_to_load = np.zeros(T, dtype=float)
    e_pv_to_bess = np.zeros(T, dtype=float)
    e_bess_to_load = np.zeros(T, dtype=float)
    e_grid_to_load = np.zeros(T, dtype=float)
    e_inj = np.zeros(T, dtype=float)
    e_bess = np.zeros(T, dtype=float)
    d_bess_ch = np.zeros(T, dtype=float)
    d_bess_dis = np.zeros(T, dtype=float)


    if not use_bess or bess_size_kwh <= 1e-12:
        e_pv_to_load = np.minimum(e_dispatch_load, e_pv)
        e_grid_to_load = np.maximum(e_dispatch_load - e_pv_to_load, 0.0)
        e_inj = np.maximum(e_pv - e_pv_to_load, 0.0)
        e_bess[:] = 0.0
    else:
        soc_min_kwh = max(0.0, soc_bess_min) * bess_size_kwh
        soc_max_kwh = max(soc_min_kwh, soc_bess_max * bess_size_kwh)

        if t_bess_min_h is None or t_bess_min_h <= 1e-12:
            p_bess_max_kw = 1e18
        else:
            p_bess_max_kw = bess_size_kwh / t_bess_min_h

        e_bess_max_step = p_bess_max_kw * DT

        soc = 0.5 * bess_size_kwh
        soc = min(max(soc, soc_min_kwh), soc_max_kwh)

        for t in range(T):
            soc *= eta_bess_stor

            load_t = max(e_dispatch_load[t], 0.0)
            pv_t = max(e_pv[t], 0.0)

            pv_to_load = min(load_t, pv_t)
            deficit = load_t - pv_to_load
            surplus = pv_t - pv_to_load

            charge = 0.0
            if surplus > 1e-12:
                room_kwh = max(soc_max_kwh - soc, 0.0)
                max_charge_by_soc = room_kwh / max(eta_bess_in, 1e-12)
                charge = min(surplus, e_bess_max_step, max_charge_by_soc)
                soc += charge * eta_bess_in
                surplus -= charge

            discharge = 0.0
            if deficit > 1e-12:
                avail_kwh = max(soc - soc_min_kwh, 0.0)
                max_discharge_by_soc = avail_kwh * eta_bess_out
                discharge = min(deficit, e_bess_max_step, max_discharge_by_soc)
                soc -= discharge / max(eta_bess_out, 1e-12)
                deficit -= discharge

            e_pv_to_load[t] = pv_to_load
            e_pv_to_bess[t] = charge
            e_bess_to_load[t] = discharge
            e_grid_to_load[t] = max(deficit, 0.0)
            e_inj[t] = max(surplus, 0.0)
            e_bess[t] = soc

            d_bess_ch[t] = 1.0 if charge > 1e-12 else 0.0
            d_bess_dis[t] = 1.0 if discharge > 1e-12 else 0.0

    load_kwh = e_load.sum()
    pv_kwh = e_pv.sum()
    pv_to_load_kwh = e_pv_to_load.sum()
    pv_to_bess_kwh = e_pv_to_bess.sum()
    bess_to_load_kwh = e_bess_to_load.sum()
    injection_kwh = e_inj.sum()

    if boiler_tariff == "A":
        # A tarifás bojler: nincs külön B tarifás import.
        e_grid_to_base = e_grid_to_load.copy()
        e_grid_to_boiler = np.zeros_like(e_grid_to_load)

        e_local_total = e_pv_to_load + e_bess_to_load
        e_local_to_base = np.minimum(e_base_load, e_local_total)
        e_local_to_boiler = np.minimum(
            np.maximum(e_local_total - e_local_to_base, 0.0),
            e_boiler_load,
        )

    else:
        # B tarifás bojler: teljes bojlerfogyasztás B tarifás hálózati import.
        e_grid_to_base = e_grid_to_load.copy()
        e_grid_to_boiler = e_boiler_load.copy()

        e_local_to_base = e_pv_to_load + e_bess_to_load
        e_local_to_boiler = np.zeros_like(e_boiler_load)

        # Teljes hálózati import = A tarifás alapimport + B tarifás bojlerimport.
        e_grid_to_load = e_grid_to_base + e_grid_to_boiler

    grid_import_kwh = e_grid_to_load.sum()

    bill = calc_bill_15min_brutto(
        e_grid_to_load=e_grid_to_load,
        e_inj=e_inj,
        e_grid_to_boiler=e_grid_to_boiler,
    )

    # SCI: önfogyasztási index.
    # A megtermelt PV mekkora része marad helyben:
    # - közvetlen PV -> fogyasztás
    # - PV -> BESS töltés
    #
    # Fontos: SCI-ben a PV->BESS töltést számoljuk,
    # nem a későbbi BESS->load kisütést, mert az már veszteségekkel csökkentett energia.
    self_consumed_pv_kwh = pv_to_load_kwh + pv_to_bess_kwh
    self_consumption_ratio = self_consumed_pv_kwh / pv_kwh if pv_kwh > 1e-12 else 0.0
    self_consumption_ratio = min(max(self_consumption_ratio, 0.0), 1.0)

    # SSI: önellátási index.
    # A fogyasztás mekkora részét fedezi helyi energia:
    # - közvetlen PV -> fogyasztás
    # - BESS -> fogyasztás
    #
    # Itt a BESS kisütés számít, mert ez ténylegesen fogyasztást lát el.
    locally_supplied_load_kwh = pv_to_load_kwh + bess_to_load_kwh
    self_sufficiency_ratio = locally_supplied_load_kwh / load_kwh if load_kwh > 1e-12 else 0.0
    self_sufficiency_ratio = min(max(self_sufficiency_ratio, 0.0), 1.0)

    return {
        "timeseries": {
            "e_load": e_load,
            "e_pv": e_pv,
            "e_pv_to_load": e_pv_to_load,
            "e_pv_to_bess": e_pv_to_bess,
            "e_bess_to_load": e_bess_to_load,
            "e_grid_to_load": e_grid_to_load,
            "e_grid_to_base": e_grid_to_base,
            "e_grid_to_boiler": e_grid_to_boiler,
            "e_local_to_base": e_local_to_base,
            "e_local_to_boiler": e_local_to_boiler,
            "e_inj": e_inj,
            "e_bess": e_bess,
            "d_bess_ch": d_bess_ch,
            "d_bess_dis": d_bess_dis,
        },

        "annual": {
            "load_kwh": load_kwh,
            "pv_kwh": pv_kwh,
            "pv_to_load_kwh": pv_to_load_kwh,
            "pv_to_bess_kwh": pv_to_bess_kwh,
            "bess_to_load_kwh": bess_to_load_kwh,
            "grid_import_kwh": grid_import_kwh,
            "grid_import_a_kwh": bill["grid_import_a_kwh"],
            "grid_import_b_kwh": bill["grid_import_b_kwh"],
            "injection_kwh": injection_kwh,
            "self_consumed_pv_kwh": self_consumed_pv_kwh,
            "locally_supplied_load_kwh": locally_supplied_load_kwh,
            "self_consumption_ratio": self_consumption_ratio,
            "self_sufficiency_ratio": self_sufficiency_ratio,
            **bill,
        },
    }

def plot_household_percentiles_by_group(per_user_df: pd.DataFrame, out_case: Path):
    df = per_user_df.copy().sort_values("brt_bill_ft").reset_index(drop=True)
    n = len(df)

    if n == 0:
        return

    df["percentile"] = 100.0 * (np.arange(n) + 0.5) / n

    marker_map = {
        "Nincs PV": "o",
        "PV": "s",
        "PV+BESS": "^",
    }

    plt.figure(figsize=(9, 5.5))

    for group in ["Nincs PV", "PV", "PV+BESS"]:
        sub = df[df["group_label"] == group].copy()
        if len(sub) == 0:
            continue

        plt.scatter(
            sub["brt_bill_ft"],
            sub["percentile"],
            s=28,
            marker=marker_map[group],
            alpha=0.8,
            label=group,
        )
    x_lo = np.percentile(df["brt_bill_ft"], 1)
    x_hi = np.percentile(df["brt_bill_ft"], 99)
    plt.xlim(x_lo, x_hi)
    plt.xlabel("Éves bruttó villanyszámla [Ft/év]")
    plt.ylabel("Háztartások aránya [%]")
    plt.title("Háztartások villanyszámla szerinti eloszlása")
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_case / "household_bill_percentiles_by_group.png", dpi=200)
    plt.close()

def plot_household_percentiles_by_group_with_global_scurve(per_user_df: pd.DataFrame, out_case: Path):
    from scipy.optimize import curve_fit

    df = per_user_df.copy().sort_values("brt_bill_ft").reset_index(drop=True)
    n = len(df)

    if n == 0:
        return

    df["percentile"] = 100.0 * (np.arange(n) + 0.5) / n

    marker_map = {
        "Nincs PV": "o",
        "PV": "s",
        "PV+BESS": "^",
    }

    plt.figure(figsize=(9, 5.5))

    # pontok csoportonként
    for group in ["Nincs PV", "PV", "PV+BESS"]:
        sub = df[df["group_label"] == group].copy()
        if len(sub) == 0:
            continue

        plt.scatter(
            sub["brt_bill_ft"],
            sub["percentile"],
            s=28,
            marker=marker_map[group],
            alpha=0.8,
            label=group,
        )

    # egyetlen globális S-görbe az összes háztartásra
    x = df["brt_bill_ft"].to_numpy(dtype=float)
    y = df["percentile"].to_numpy(dtype=float)

    def logistic(x_, x0, k):
        return 100.0 / (1.0 + np.exp(-k * (x_ - x0)))

    if len(df) >= 4:
        x0_init = np.median(x)
        spread = max(np.std(x), 1.0)
        k_init = 1.0 / spread

        try:
            popt, _ = curve_fit(logistic, x, y, p0=[x0_init, k_init], maxfev=20000)
            x_grid = np.linspace(x.min(), x.max(), 400)
            y_fit = logistic(x_grid, *popt)
            plt.plot(x_grid, y_fit, color="darkred",linewidth=2, label="Illesztett S-görbe (összes háztartás)")
        except Exception:
            pass

    x_lo = np.percentile(df["brt_bill_ft"], 1)
    x_hi = np.percentile(df["brt_bill_ft"], 100)
    plt.xlim(x_lo, x_hi)

    plt.xlabel("Éves bruttó villanyszámla [Ft/év]")
    plt.ylabel("Háztartások aránya [%]")
    plt.title("Háztartások villanyszámla szerinti eloszlása")
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_case / "household_bill_percentiles_by_group_global_scurve.png", dpi=200)
    plt.close()

def run_case(
    case_name: str,
    sim_yaml: str,
    profiles_csv: str,
    dhw_profiles_csv: str,
    out_dir: str,
    max_users: int,
    include_bess: bool,
    bess_share_pct: float = 100.0,
    boiler_tariff: str = "B",
):
    out_case = Path(out_dir)
    out_case.mkdir(parents=True, exist_ok=True)

    boiler_tariff = str(boiler_tariff).upper().strip()
    if boiler_tariff not in {"A", "B"}:
        raise ValueError(f"boiler_tariff csak 'A' vagy 'B' lehet, nem: {boiler_tariff}")

    (
        e_pv,
        e_ue,
        e_el_heater,
        size_bess,
        eta_bess_in_u,
        eta_bess_out_u,
        eta_bess_stor_u,
        soc_bess_min_u,
        soc_bess_max_u,
        t_bess_min_u,
        user_names,
    ) = build_inputs(
        sim_yaml_path=sim_yaml,
        profiles_csv_path=profiles_csv,
        dhw_profile_path=dhw_profiles_csv,
        max_users=max_users,
    )

    U = len(user_names)
    T = e_ue.shape[0]

    e_boiler = e_el_heater
    e_total_load = e_ue + e_boiler
    e_base_load = e_ue

    rows = []

    ts_e_load = np.zeros((T, U), dtype=float)
    ts_e_pv = np.zeros((T, U), dtype=float)
    ts_e_pv_to_load = np.zeros((T, U), dtype=float)
    ts_e_pv_to_bess = np.zeros((T, U), dtype=float)
    ts_e_bess_to_load = np.zeros((T, U), dtype=float)
    ts_e_grid_to_load = np.zeros((T, U), dtype=float)
    ts_e_grid_to_base = np.zeros((T, U), dtype=float)
    ts_e_grid_to_boiler = np.zeros((T, U), dtype=float)
    ts_e_inj = np.zeros((T, U), dtype=float)
    ts_e_bess = np.zeros((T, U), dtype=float)
    ts_d_bess_ch = np.zeros((T, U), dtype=float)
    ts_d_bess_dis = np.zeros((T, U), dtype=float)
    ts_e_base_load = np.zeros((T, U), dtype=float)
    ts_e_boiler = np.zeros((T, U), dtype=float)

    pv_annual_kwh = e_pv.sum(axis=0)
    has_pv_arr = pv_annual_kwh > 1e-9

    bess_enabled_arr = np.zeros(len(user_names), dtype=bool)

    if include_bess:
        pv_user_idx = np.where(has_pv_arr)[0]
        n_pv_users = len(pv_user_idx)

        n_bess_users = int(round(n_pv_users * bess_share_pct / 100.0))
        n_bess_users = max(0, min(n_bess_users, n_pv_users))

        # egyszerű, determinisztikus megoldás:
        # az első n darab PV-s háztartás kap BESS-t
        bess_enabled_arr[pv_user_idx[:n_bess_users]] = True

        print(f"[INFO] PV-s háztartások száma: {n_pv_users}")
        print(f"[INFO] BESS arány: {bess_share_pct}%")
        print(f"[INFO] BESS-t kapó PV-s háztartások száma: {n_bess_users}")

    for u in range(U):
        sim = simulate_one_user_greedy(
            e_load=e_total_load[:, u],
            e_pv=e_pv[:, u],
            e_boiler_load=e_boiler[:, u],
            use_bess=bool(bess_enabled_arr[u]),
            bess_size_kwh=float(size_bess[u]),
            eta_bess_in=float(eta_bess_in_u[u]),
            eta_bess_out=float(eta_bess_out_u[u]),
            eta_bess_stor=float(eta_bess_stor_u[u]),
            soc_bess_min=float(soc_bess_min_u[u]),
            soc_bess_max=float(soc_bess_max_u[u]),
            t_bess_min_h=float(t_bess_min_u[u]),
            boiler_tariff=boiler_tariff,
        )


        annual = sim["annual"]
        times = sim["timeseries"]

        base_load_kwh = e_ue[:, u].sum()
        boiler_kwh = e_boiler[:, u].sum()
        pmax_kw = float(np.max(e_total_load[:, u] / DT))

        rows.append({
            "user_name": user_names[u],
            "has_pv": bool(e_pv[:, u].sum() > 1e-9),
            "has_bess": bool(bess_enabled_arr[u] and size_bess[u] > 1e-9),
            "has_boiler": bool(e_boiler[:, u].sum() > 1e-9),
            "group_label": (
                "PV+BESS" if (has_pv_arr[u] and bess_enabled_arr[u] and size_bess[u] > 1e-9)
                else "PV" if has_pv_arr[u]
                else "Nincs PV"
            ),
            "base_load_kwh": base_load_kwh,
            "boiler_kwh": boiler_kwh,
            "total_load_kwh": annual["load_kwh"],
            "pv_kwh": annual["pv_kwh"],
            "Pmax_kw": pmax_kw,
            "pv_to_load_kwh": annual["pv_to_load_kwh"],
            "pv_to_bess_kwh": annual["pv_to_bess_kwh"],
            "bess_to_load_kwh": annual["bess_to_load_kwh"],
            "grid_import_kwh": annual["grid_import_kwh"],
            "grid_import_a_kwh": annual["grid_import_a_kwh"],
            "grid_import_b_kwh": annual["grid_import_b_kwh"],
            "grid_import_low_kwh": annual["grid_import_low_kwh"],
            "grid_import_high_kwh": annual["grid_import_high_kwh"],
            "grid_import_a_low_kwh": annual["grid_import_a_low_kwh"],
            "grid_import_a_high_kwh": annual["grid_import_a_high_kwh"],
            "grid_import_b_low_kwh": annual["grid_import_b_low_kwh"],
            "grid_import_b_high_kwh": annual["grid_import_b_high_kwh"],
            "injection_kwh": annual["injection_kwh"],
            "self_consumed_pv_kwh": annual["self_consumed_pv_kwh"],
            "locally_supplied_load_kwh": annual["locally_supplied_load_kwh"],
            "self_consumption_ratio": annual["self_consumption_ratio"],
            "self_sufficiency_ratio": annual["self_sufficiency_ratio"],
            "SCI": annual["self_consumption_ratio"],
            "SSI": annual["self_sufficiency_ratio"],
            "import_cost_a_ft": annual["import_cost_a_ft"],
            "import_cost_b_ft": annual["import_cost_b_ft"],
            "import_cost_ft": annual["import_cost_ft"],
            "export_revenue_ft": annual["export_revenue_ft"],
            "brt_bill_ft": annual["brt_bill_ft"],
        })

        ts_e_load[:, u] = times["e_load"]
        ts_e_pv[:, u] = times["e_pv"]
        ts_e_pv_to_load[:, u] = times["e_pv_to_load"]
        ts_e_pv_to_bess[:, u] = times["e_pv_to_bess"]
        ts_e_bess_to_load[:, u] = times["e_bess_to_load"]
        ts_e_grid_to_load[:, u] = times["e_grid_to_load"]
        ts_e_grid_to_base[:, u] = times["e_grid_to_base"]
        ts_e_grid_to_boiler[:, u] = times["e_grid_to_boiler"]
        ts_e_inj[:, u] = times["e_inj"]
        ts_e_bess[:, u] = times["e_bess"]
        ts_d_bess_ch[:, u] = times["d_bess_ch"]
        ts_d_bess_dis[:, u] = times["d_bess_dis"]
        ts_e_base_load[:, u] = e_ue[:, u]
        ts_e_boiler[:, u] = e_boiler[:, u]

    per_user_df = pd.DataFrame(rows)
    per_user_df.to_csv(out_case / "per_user_summary.csv", index=False)

    plot_household_percentiles_by_group(per_user_df, out_case)
    plot_household_percentiles_by_group_with_global_scurve(per_user_df, out_case)

    total = {
        "case_name": case_name,
        "boiler_tariff": boiler_tariff,
        "n_users": int(U),
        "base_load_kwh": float(per_user_df["base_load_kwh"].sum()),
        "boiler_kwh": float(per_user_df["boiler_kwh"].sum()),
        "total_load_kwh": float(per_user_df["total_load_kwh"].sum()),
        "pv_kwh": float(per_user_df["pv_kwh"].sum()),
        "pv_to_load_kwh": float(per_user_df["pv_to_load_kwh"].sum()),
        "pv_to_bess_kwh": float(per_user_df["pv_to_bess_kwh"].sum()),
        "bess_to_load_kwh": float(per_user_df["bess_to_load_kwh"].sum()),
        "locally_supplied_load_kwh": float(per_user_df["locally_supplied_load_kwh"].sum()),
        "grid_import_kwh": float(per_user_df["grid_import_kwh"].sum()),
        "grid_import_a_kwh": float(per_user_df["grid_import_a_kwh"].sum()),
        "grid_import_b_kwh": float(per_user_df["grid_import_b_kwh"].sum()),
        "grid_import_low_kwh": float(per_user_df["grid_import_low_kwh"].sum()),
        "grid_import_high_kwh": float(per_user_df["grid_import_high_kwh"].sum()),
        "grid_import_a_low_kwh": float(per_user_df["grid_import_a_low_kwh"].sum()),
        "grid_import_a_high_kwh": float(per_user_df["grid_import_a_high_kwh"].sum()),
        "grid_import_b_low_kwh": float(per_user_df["grid_import_b_low_kwh"].sum()),
        "grid_import_b_high_kwh": float(per_user_df["grid_import_b_high_kwh"].sum()),
        "injection_kwh": float(per_user_df["injection_kwh"].sum()),
        "import_cost_a_ft": float(per_user_df["import_cost_a_ft"].sum()),
        "import_cost_b_ft": float(per_user_df["import_cost_b_ft"].sum()),
        "import_cost_ft": float(per_user_df["import_cost_ft"].sum()),
        "export_revenue_ft": float(per_user_df["export_revenue_ft"].sum()),
        "brt_bill_ft": float(per_user_df["brt_bill_ft"].sum()),
        "bess_share_pct": float(bess_share_pct),
        "pv_user_count": int(has_pv_arr.sum()),
        "bess_enabled_user_count": int(bess_enabled_arr.sum()),
    }

    total["self_consumed_pv_kwh"] = (
            total["pv_to_load_kwh"] + total["pv_to_bess_kwh"]
    )

    total["self_consumption_ratio"] = (
        total["self_consumed_pv_kwh"] / total["pv_kwh"]
        if total["pv_kwh"] > 1e-12 else 0.0
    )

    total["self_consumption_ratio"] = min(max(total["self_consumption_ratio"], 0.0), 1.0)
    total["self_sufficiency_ratio"] = (
        total["locally_supplied_load_kwh"] / total["total_load_kwh"]
        if total["total_load_kwh"] > 1e-12 else 0.0
    )

    total["self_sufficiency_ratio"] = min(max(total["self_sufficiency_ratio"], 0.0), 1.0)
    total["SCI"] = total["self_consumption_ratio"]
    total["SSI"] = total["self_sufficiency_ratio"]

    (out_case / "summary.json").write_text(
        json.dumps(total, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    def save_ts(arr: np.ndarray, filename: str):
        pd.DataFrame(arr, columns=user_names).to_csv(out_case / filename, index=False)

    save_ts(ts_e_load, "e_load.csv")
    save_ts(ts_e_pv, "e_pv.csv")
    save_ts(ts_e_pv_to_load, "e_pv_to_load.csv")
    save_ts(ts_e_pv_to_bess, "e_pv_to_bess.csv")
    save_ts(ts_e_bess_to_load, "e_bess_to_load.csv")
    save_ts(ts_e_grid_to_load, "e_grid_to_load.csv")
    save_ts(ts_e_grid_to_base, "e_grid_to_base.csv")
    save_ts(ts_e_grid_to_boiler, "e_grid_to_boiler.csv")
    save_ts(ts_e_inj, "e_inj.csv")
    save_ts(ts_e_bess, "e_bess.csv")
    save_ts(ts_d_bess_ch, "d_bess_ch.csv")
    save_ts(ts_d_bess_dis, "d_bess_dis.csv")
    save_ts(ts_e_base_load, "e_base_load.csv")
    save_ts(ts_e_boiler, "e_boiler.csv")

    community_ts = pd.DataFrame({
        "e_load_total": ts_e_load.sum(axis=1),
        "e_pv_total": ts_e_pv.sum(axis=1),
        "e_pv_to_load_total": ts_e_pv_to_load.sum(axis=1),
        "e_pv_to_bess_total": ts_e_pv_to_bess.sum(axis=1),
        "e_bess_to_load_total": ts_e_bess_to_load.sum(axis=1),
        "e_grid_to_load_total": ts_e_grid_to_load.sum(axis=1),
        "e_grid_to_base_total": ts_e_grid_to_base.sum(axis=1),
        "e_grid_to_boiler_total": ts_e_grid_to_boiler.sum(axis=1),
        "e_inj_total": ts_e_inj.sum(axis=1),
        "e_bess_total": ts_e_bess.sum(axis=1),
    })
    community_ts.to_csv(out_case / "community_timeseries.csv", index=False)

    print(f"[INFO] Kész: {case_name}")
    print(f"[INFO] Output: {out_case}")
    print(json.dumps(total, indent=2, ensure_ascii=False))