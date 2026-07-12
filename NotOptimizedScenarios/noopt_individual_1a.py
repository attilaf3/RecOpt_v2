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


def _dhw_liter_to_thermal_power_kw(
    dhw: pd.DataFrame,
    hss: dict,
    dt: float = DT,
) -> np.ndarray:
    """Liter/step DHW profil -> kW hőigény."""
    prof = hss.get("profile")
    if prof is None or str(prof) not in dhw.columns:
        return np.zeros(35040, dtype=float)

    liters = np.maximum(np.asarray(dhw[str(prof)].to_numpy(), dtype=float).ravel(), 0.0)
    if liters.size != 35040:
        raise ValueError(f"A DHW profil hossza {liters.size}, de 35040 kell.")

    c_hss = 0.00116667  # kWh/(liter*K)
    T_in = float(hss.get("T_in", 10.0))
    T_out = float(hss.get("T_out", 40.0))
    return liters * c_hss * max(T_out - T_in, 0.0) / float(dt)


def build_inputs(
    sim_yaml_path: os.PathLike,
    profiles_csv_path: os.PathLike,
    dhw_profile_path: os.PathLike,
    max_users: int = 10,
    search_roots: Iterable[os.PathLike] | None = None,
):
    """A nem optimalizált BESS+bojler eset összes bemenetének beolvasása."""
    sim_yaml_path = Path(sim_yaml_path)
    profiles_csv_path = Path(profiles_csv_path)
    dhw_profile_path = Path(dhw_profile_path)

    if search_roots is None:
        search_roots = [sim_yaml_path.parent / "users_v2", sim_yaml_path.parent]

    sim = yaml.safe_load(sim_yaml_path.read_text(encoding="utf-8")) or {}
    users_list = list(sim.get("users_list", []))[: int(max_users)]
    exclude = {"battery", "bess", "community"}
    users_list = [u for u in users_list if str(u).strip().lower() not in exclude]
    if not users_list:
        raise RuntimeError("A simulation YAML nem tartalmaz users_list-et vagy max_users=0.")

    df = pd.read_csv(profiles_csv_path, index_col=0)
    df.columns = [str(c) for c in df.columns]
    df = df[[c for c in df.columns if c.lower() not in exclude]]

    dhw = pd.read_csv(dhw_profile_path, index_col=0)
    dhw.columns = [str(c) for c in dhw.columns]

    e_pv_cols, e_ue_cols, e_el_heater_fixed_cols, p_dhw_cols = [], [], [], []
    user_names = []

    size_bess, eta_bess_in_u, eta_bess_out_u = [], [], []
    eta_bess_stor_u, soc_bess_min_u, soc_bess_max_u, t_bess_min_u = [], [], [], []

    size_elh_u, vol_hss_water_u = [], []
    T_env_u, T_max_u, T_min_u, T_in_u, T_set_u = [], [], [], [], []
    a_hss_u, eta_elh_u = [], []

    for user_key in users_list:
        ypath = _find_user_yaml(search_roots, user_key)
        if not ypath:
            print(f"[WARN] YAML nem található: {user_key} — kihagyom.")
            continue

        user_yaml = yaml.safe_load(ypath.read_text(encoding="utf-8")) or {}
        units = user_yaml.get("units") or {}
        user_names.append(str((units.get("name") or {}).get("name", user_key)))

        ue = units.get("ue") or {}
        pv = units.get("pv") or {}
        ut = units.get("ut") or {}
        hss = units.get("hss") or {}
        bess = units.get("bess") or {}

        ue_prof = ue.get("profile")
        pv_prof = pv.get("profile")
        ut_prof = ut.get("profile")

        e_ue_cols.append(
            _as_15min_energy(df[str(ue_prof)].to_numpy())
            if ue_prof is not None and str(ue_prof) in df.columns
            else np.zeros(35040, dtype=float)
        )
        e_pv_cols.append(
            _as_15min_energy(df[str(pv_prof)].to_numpy())
            if pv_prof is not None and str(pv_prof) in df.columns
            else np.zeros(35040, dtype=float)
        )
        e_el_heater_fixed_cols.append(
            _as_15min_energy(df[str(ut_prof)].to_numpy())
            if ut_prof is not None and str(ut_prof) in df.columns
            else np.zeros(35040, dtype=float)
        )
        p_dhw_cols.append(_dhw_liter_to_thermal_power_kw(dhw, hss, dt=DT))

        size_bess.append(float(bess.get("bess_size", 0.0)))
        eta_bess_in_u.append(float(bess.get("eta_bess_in", 0.98)))
        eta_bess_out_u.append(float(bess.get("eta_bess_out", 0.96)))
        eta_bess_stor_u.append(float(bess.get("eta_bess_stor", 0.995)))
        soc_bess_min_u.append(float(bess.get("soc_bess_min", 0.1)))
        soc_bess_max_u.append(float(bess.get("soc_bess_max", 0.9)))
        t_bess_min_u.append(float(bess.get("t_bess_min", 2.0)))

        size_elh_u.append(float(hss.get("size_elh", 0.0)))
        vol_hss_water_u.append(float(hss.get("vol_hss_water", 0.0)))
        T_env_u.append(float(hss.get("T_env", 20.0)))
        T_max_u.append(float(hss.get("T_max", 65.0)))
        T_min_u.append(float(hss.get("T_min", 38.0)))
        T_in_u.append(float(hss.get("T_in", 10.0)))
        T_set_u.append(float(hss.get("T_set", hss.get("T_setpoint", 50.0))))
        a_hss_u.append(float(hss.get("a_hss", 0.01275)))
        eta_elh_u.append(float(hss.get("eta_elh", 0.95)))

    if not user_names:
        raise RuntimeError("Nincs érvényes felhasználó.")

    return (
        np.column_stack(e_pv_cols).astype(float),
        np.column_stack(e_ue_cols).astype(float),
        np.column_stack(e_el_heater_fixed_cols).astype(float),
        np.column_stack(p_dhw_cols).astype(float),
        np.asarray(size_bess, dtype=float),
        np.asarray(eta_bess_in_u, dtype=float),
        np.asarray(eta_bess_out_u, dtype=float),
        np.asarray(eta_bess_stor_u, dtype=float),
        np.asarray(soc_bess_min_u, dtype=float),
        np.asarray(soc_bess_max_u, dtype=float),
        np.asarray(t_bess_min_u, dtype=float),
        np.asarray(size_elh_u, dtype=float),
        np.asarray(vol_hss_water_u, dtype=float),
        np.asarray(T_env_u, dtype=float),
        np.asarray(T_max_u, dtype=float),
        np.asarray(T_min_u, dtype=float),
        np.asarray(T_in_u, dtype=float),
        np.asarray(T_set_u, dtype=float),
        np.asarray(a_hss_u, dtype=float),
        np.asarray(eta_elh_u, dtype=float),
        user_names,
    )


C_HSS = 0.00116667  # kWh/(liter*K)


def _b_tariff_available(step: int, dt: float = DT) -> bool:
    hour = (step * dt) % 24.0
    return hour < 4.0 or (10.0 <= hour < 16.0) or hour >= 22.0


def simulate_rule_based_hss(
    p_dhw_kw: np.ndarray,
    *,
    size_elh_kw: float,
    vol_hss_water_l: float,
    T_env: float,
    T_max: float,
    T_min: float,
    T_in: float,
    T_set: float,
    a_hss_kw_per_k: float,
    eta_elh: float,
    boiler_tariff: str,
    dt: float = DT,
) -> dict:
    """Termosztatikus, nem optimalizált HSS-modell kW/kWh idősorokkal."""
    p_dhw_kw = np.maximum(np.asarray(p_dhw_kw, dtype=float).ravel(), 0.0)
    T = len(p_dhw_kw)
    tariff = str(boiler_tariff).upper().strip()
    if tariff not in {"A", "B"}:
        raise ValueError("boiler_tariff csak 'A' vagy 'B' lehet.")

    out = {
        "e_boiler": np.zeros(T),
        "p_elh_in": np.zeros(T),
        "p_hss_in": np.zeros(T),
        "p_hss_out": p_dhw_kw.copy(),
        "t_hss": np.zeros(T),
        "d_boiler_available": np.zeros(T),
        "d_boiler_on": np.zeros(T),
        "temperature_violation": np.zeros(T),
    }

    if size_elh_kw <= 1e-9 or vol_hss_water_l <= 1e-9:
        return out

    cap_kwh_per_k = vol_hss_water_l * C_HSS
    temp = float(np.clip(T_set, T_min, T_max))

    for t in range(T):
        available = tariff == "A" or _b_tariff_available(t, dt)
        out["d_boiler_available"][t] = float(available)
        out["t_hss"][t] = temp

        # Az aktuális lépés hőigénye, vesztesége és a setpoint visszaállítása.
        p_needed_th = (
            p_dhw_kw[t]
            + a_hss_kw_per_k * (temp - T_env)
            + cap_kwh_per_k * max(T_set - temp, 0.0) / dt
        )
        p_elh = min(size_elh_kw, max(p_needed_th, 0.0) / max(eta_elh, 1e-9)) if available else 0.0

        out["p_elh_in"][t] = p_elh
        out["p_hss_in"][t] = eta_elh * p_elh
        out["e_boiler"][t] = p_elh * dt
        out["d_boiler_on"][t] = float(p_elh > 1e-9)

        next_temp = temp + dt * (
            eta_elh * p_elh
            - p_dhw_kw[t]
            - a_hss_kw_per_k * (temp - T_env)
        ) / cap_kwh_per_k

        if next_temp < T_min - 1e-6 or next_temp > T_max + 1e-6:
            out["temperature_violation"][t] = 1.0
        temp = float(np.clip(next_temp, T_in, T_max))

    return out


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

    # Bruttó elszámolás:
    #   importköltség = A tarifás import költsége + B tarifás import költsége
    #   exportbevétel = PV exportált energia * átvételi ár
    #   bruttó villanyszámla = importköltség - exportbevétel
    #
    # A tarifa:
    #   normál háztartási fogyasztás
    # B tarifa:
    #   külön bojler import

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
    e_grid_to_bess = np.zeros(T, dtype=float)
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

        min_mode_steps = 4
        mode = "idle"
        lock_steps_left = 0

        # Greedy logika:
        # 1) PV először közvetlenül a fogyasztást látja el.
        # 2) Ha PV többlet van, az BESS-be tölt.
        # 3) Ha a BESS már nem tud több energiát felvenni, a maradék PV export.
        # 4) Ha fogyasztási hiány van, a BESS kisüt.
        # 5) Ha még mindig hiány van, hálózati import történik.
        # 6) A BESS SOC minden időlépésben önkisüléssel csökken.

        # A tarifás bojler esetén:
        #   a bojler a teljes háztartási fogyasztás része,
        #   ezért PV és BESS is kiszolgálhatja.
        #
        # B tarifás bojler esetén:
        #   a bojler külön mérőn van,
        #   ezért nem kap PV-t és nem kap BESS energiát,
        #   a teljes bojlerigény B tarifás hálózati import.

        for t in range(T):
            soc *= eta_bess_stor

            load_t = max(e_dispatch_load[t], 0.0)
            pv_t = max(e_pv[t], 0.0)

            pv_to_load = min(load_t, pv_t)
            deficit = load_t - pv_to_load
            surplus = pv_t - pv_to_load

            can_charge = (lock_steps_left <= 0) or (mode == "charge")
            can_discharge = (lock_steps_left <= 0) or (mode == "discharge")

            charge = 0.0
            if surplus > 1e-12 and can_charge:
                room_kwh = max(soc_max_kwh - soc, 0.0)
                max_charge_by_soc = room_kwh / max(eta_bess_in, 1e-12)
                charge = min(surplus, e_bess_max_step, max_charge_by_soc)
                soc += charge * eta_bess_in
                surplus -= charge

            discharge = 0.0
            if deficit > 1e-12 and can_discharge:
                avail_kwh = max(soc - soc_min_kwh, 0.0)
                max_discharge_by_soc = avail_kwh * eta_bess_out
                discharge = min(deficit, e_bess_max_step, max_discharge_by_soc)
                soc -= discharge / max(eta_bess_out, 1e-12)
                deficit -= discharge


            grid_charge = 0.0
            if soc < soc_min_kwh - 1e-12 and can_charge and surplus <= 1e-12:
                need_to_soc = soc_min_kwh - soc
                grid_charge = min(
                    e_bess_max_step,
                    need_to_soc / max(eta_bess_in, 1e-12),
                )
                soc += grid_charge * eta_bess_in

            if charge > 1e-12 or grid_charge > 1e-12:
                if mode != "charge":
                    mode = "charge"
                    lock_steps_left = min_mode_steps - 1

            elif discharge > 1e-12:
                if mode != "discharge":
                    mode = "discharge"
                    lock_steps_left = min_mode_steps - 1

            else:
                if lock_steps_left <= 0:
                    mode = "idle"

            if lock_steps_left > 0:
                lock_steps_left -= 1

            e_pv_to_load[t] = pv_to_load
            e_pv_to_bess[t] = charge
            e_bess_to_load[t] = discharge
            e_grid_to_load[t] = max(deficit, 0.0) + grid_charge
            e_inj[t] = max(surplus, 0.0)
            e_bess[t] = soc
            e_grid_to_bess[t] = grid_charge

            d_bess_ch[t] = 1.0 if (charge > 1e-12 or grid_charge > 1e-12) else 0.0
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
            "e_grid_to_bess": e_grid_to_bess,
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
            "grid_to_bess_kwh": float(e_grid_to_bess.sum()),
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
    use_boiler_model: bool = True,
):
    out_case = Path(out_dir)
    out_case.mkdir(parents=True, exist_ok=True)

    boiler_tariff = str(boiler_tariff).upper().strip()
    if boiler_tariff not in {"A", "B"}:
        raise ValueError(f"boiler_tariff csak 'A' vagy 'B' lehet, nem: {boiler_tariff}")

    (
        e_pv,
        e_ue,
        e_el_heater_fixed,
        p_dhw_kw,
        size_bess,
        eta_bess_in_u,
        eta_bess_out_u,
        eta_bess_stor_u,
        soc_bess_min_u,
        soc_bess_max_u,
        t_bess_min_u,
        size_elh_u,
        vol_hss_water_u,
        T_env_u,
        T_max_u,
        T_min_u,
        T_in_u,
        T_set_u,
        a_hss_u,
        eta_elh_u,
        user_names,
    ) = build_inputs(
        sim_yaml_path=sim_yaml,
        profiles_csv_path=profiles_csv,
        dhw_profile_path=dhw_profiles_csv,
        max_users=max_users,
    )

    U = len(user_names)
    T = e_ue.shape[0]

    # PV-s és érvényes HSS-paraméterekkel rendelkező háztartásnál dinamikus
    # bojlermodell készül. Minden más háztartásnál megmarad a mért fix profil.
    e_boiler = np.zeros_like(e_ue)
    ts_t_hss = np.zeros((T, U), dtype=float)
    ts_p_dhw = np.zeros((T, U), dtype=float)
    ts_p_elh_in = np.zeros((T, U), dtype=float)
    ts_p_hss_in = np.zeros((T, U), dtype=float)
    ts_d_boiler_available = np.zeros((T, U), dtype=float)
    ts_d_boiler_on = np.zeros((T, U), dtype=float)
    ts_temperature_violation = np.zeros((T, U), dtype=float)
    dynamic_boiler_arr = np.zeros(U, dtype=bool)

    pv_annual_kwh = e_pv.sum(axis=0)
    has_pv_arr = pv_annual_kwh > 1e-9

    for u in range(U):
        use_dynamic_boiler = (
            bool(use_boiler_model)
            and bool(has_pv_arr[u])
            and float(size_elh_u[u]) > 1e-9
            and float(vol_hss_water_u[u]) > 1e-9
            and float(np.sum(p_dhw_kw[:, u]) * DT) > 1e-9
        )
        dynamic_boiler_arr[u] = use_dynamic_boiler

        if use_dynamic_boiler:
            boiler_res = simulate_rule_based_hss(
                p_dhw_kw[:, u],
                size_elh_kw=float(size_elh_u[u]),
                vol_hss_water_l=float(vol_hss_water_u[u]),
                T_env=float(T_env_u[u]),
                T_max=float(T_max_u[u]),
                T_min=float(T_min_u[u]),
                T_in=float(T_in_u[u]),
                T_set=float(T_set_u[u]),
                a_hss_kw_per_k=float(a_hss_u[u]),
                eta_elh=float(eta_elh_u[u]),
                boiler_tariff=boiler_tariff,
                dt=DT,
            )
            e_boiler[:, u] = boiler_res["e_boiler"]
            ts_t_hss[:, u] = boiler_res["t_hss"]
            ts_p_dhw[:, u] = boiler_res["p_hss_out"]
            ts_p_elh_in[:, u] = boiler_res["p_elh_in"]
            ts_p_hss_in[:, u] = boiler_res["p_hss_in"]
            ts_d_boiler_available[:, u] = boiler_res["d_boiler_available"]
            ts_d_boiler_on[:, u] = boiler_res["d_boiler_on"]
            ts_temperature_violation[:, u] = boiler_res["temperature_violation"]
        else:
            e_boiler[:, u] = e_el_heater_fixed[:, u]

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
    ts_e_grid_to_bess = np.zeros((T, U), dtype=float)
    ts_e_grid_to_boiler = np.zeros((T, U), dtype=float)
    ts_e_inj = np.zeros((T, U), dtype=float)
    ts_e_bess = np.zeros((T, U), dtype=float)
    ts_d_bess_ch = np.zeros((T, U), dtype=float)
    ts_d_bess_dis = np.zeros((T, U), dtype=float)
    ts_e_base_load = np.zeros((T, U), dtype=float)
    ts_e_boiler = np.zeros((T, U), dtype=float)

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
            "dynamic_boiler": bool(dynamic_boiler_arr[u]),
            "fixed_boiler_profile": bool(not dynamic_boiler_arr[u] and e_el_heater_fixed[:, u].sum() > 1e-9),
            "boiler_temperature_violation_steps": int(ts_temperature_violation[:, u].sum()),
            "boiler_min_temperature_c": float(ts_t_hss[:, u][ts_t_hss[:, u] > 0].min()) if dynamic_boiler_arr[u] else None,
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
            "grid_to_bess_kwh": annual["grid_to_bess_kwh"],
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
        ts_e_grid_to_bess[:, u] = times["e_grid_to_bess"]
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
        "use_boiler_model": bool(use_boiler_model),
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
        "grid_to_bess_kwh": float(per_user_df["grid_to_bess_kwh"].sum()),
        "injection_kwh": float(per_user_df["injection_kwh"].sum()),
        "import_cost_a_ft": float(per_user_df["import_cost_a_ft"].sum()),
        "import_cost_b_ft": float(per_user_df["import_cost_b_ft"].sum()),
        "import_cost_ft": float(per_user_df["import_cost_ft"].sum()),
        "export_revenue_ft": float(per_user_df["export_revenue_ft"].sum()),
        "brt_bill_ft": float(per_user_df["brt_bill_ft"].sum()),
        "bess_share_pct": float(bess_share_pct),
        "pv_user_count": int(has_pv_arr.sum()),
        "bess_enabled_user_count": int(bess_enabled_arr.sum()),
        "dynamic_boiler_user_count": int(dynamic_boiler_arr.sum()),
        "fixed_boiler_user_count": int(((~dynamic_boiler_arr) & (e_el_heater_fixed.sum(axis=0) > 1e-9)).sum()),
        "boiler_temperature_violation_steps": int(ts_temperature_violation.sum()),
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
    save_ts(ts_e_grid_to_bess, "e_grid_to_bess.csv")
    save_ts(ts_e_grid_to_boiler, "e_grid_to_boiler.csv")
    save_ts(ts_e_inj, "e_inj.csv")
    save_ts(ts_e_bess, "e_bess.csv")
    save_ts(ts_d_bess_ch, "d_bess_ch.csv")
    save_ts(ts_d_bess_dis, "d_bess_dis.csv")
    save_ts(ts_e_base_load, "e_base_load.csv")
    save_ts(ts_e_boiler, "e_boiler.csv")
    save_ts(e_el_heater_fixed, "e_boiler_fixed_input.csv")
    save_ts(ts_t_hss, "t_hss.csv")
    save_ts(ts_p_dhw, "p_dhw_kw.csv")
    save_ts(ts_p_elh_in, "p_elh_in_kw.csv")
    save_ts(ts_p_hss_in, "p_hss_in_kw.csv")
    save_ts(ts_d_boiler_available, "d_boiler_available.csv")
    save_ts(ts_d_boiler_on, "d_boiler_on.csv")
    save_ts(ts_temperature_violation, "boiler_temperature_violation.csv")

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
        "e_grid_to_bess_total": ts_e_grid_to_bess.sum(axis=1),
    })
    community_ts.to_csv(out_case / "community_timeseries.csv", index=False)

    print(
        f"[INFO] Bojlerprofil mód: "
        f"{'dinamikus modell PV-seknél' if use_boiler_model else 'fix profil mindenkinél'}"
    )
    print(f"[INFO] Kész: {case_name}")
    print(f"[INFO] Output: {out_case}")
    print(json.dumps(total, indent=2, ensure_ascii=False))

SIM_YAML = "../Input/simulation_config_disaggregated_with_userlist.yaml"
PROFILES_CSV = "../Input/measurements_disaggregated_v2.csv"
DHW_PROFILES_CSV = "../Input/dhw_v2.csv"

MAX_USERS = 105
INCLUDE_BESS = True
BESS_SHARE_PCT = 40.0
BOILER_TARIFF = "B"

# True:
#   PV-s és megfelelő HSS/DHW-adattal rendelkező háztartásnál
#   dinamikus bojlermodell; a többieknél fix profil.
# False:
#   minden háztartásnál a mért fix UT-profil.
USE_BOILER_MODEL = True

CASE_NAME = (
    f"nonopt_individual_{BOILER_TARIFF}_{BESS_SHARE_PCT}%bess_"
    f"{'boiler_model' if USE_BOILER_MODEL else 'fixed_boiler'}"
)
if not USE_BOILER_MODEL and not INCLUDE_BESS:
    CASE_NAME = f"nonopt_individual_{BOILER_TARIFF}_basecase"
OUT_DIR = f"results_{CASE_NAME}"

if __name__ == "__main__":
    run_case(
        case_name=CASE_NAME,
        sim_yaml=SIM_YAML,
        profiles_csv=PROFILES_CSV,
        dhw_profiles_csv=DHW_PROFILES_CSV,
        out_dir=OUT_DIR,
        max_users=MAX_USERS,
        include_bess=INCLUDE_BESS,
        bess_share_pct=BESS_SHARE_PCT,
        boiler_tariff=BOILER_TARIFF,
        use_boiler_model=USE_BOILER_MODEL,
    )