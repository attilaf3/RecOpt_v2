from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from Disaggregated.call_disaggregated import build_inputs
import matplotlib.pyplot as plt


DT = 0.25  # 15 perc
LOW_TARIFF_LIMIT_KWH = 2523.0
LOW_TARIFF_FT_PER_KWH = 36.0
HIGH_TARIFF_FT_PER_KWH = 71.0
EXPORT_FT_PER_KWH = 5.0


def calc_bill_15min_brutto(p_grid_to_load: np.ndarray, p_inj: np.ndarray, dt: float = 0.25) -> dict:
    """
    15 perces bruttó elszámolás.
    - minden időlépésben külön számoljuk az importot és exportot
    - export bevétel: 5 Ft/kWh
    - import díj: 36 Ft/kWh 2523 kWh/év-ig, utána 71 Ft/kWh

    Fontos:
    itt az ársávos importot éves kumulált import alapján követjük,
    de maga az elszámolás 15 perces bruttó energiacserére épül.
    """

    p_grid_to_load = np.asarray(p_grid_to_load, dtype=float)
    p_inj = np.asarray(p_inj, dtype=float)

    e_import_steps = np.maximum(p_grid_to_load, 0.0) * dt
    e_export_steps = np.maximum(p_inj, 0.0) * dt

    remaining_low = LOW_TARIFF_LIMIT_KWH
    import_cost_ft = 0.0

    e_grid_low = 0.0
    e_grid_high = 0.0

    for e_imp in e_import_steps:
        low_part = min(e_imp, max(remaining_low, 0.0))
        high_part = max(e_imp - low_part, 0.0)

        import_cost_ft += low_part * LOW_TARIFF_FT_PER_KWH
        import_cost_ft += high_part * HIGH_TARIFF_FT_PER_KWH

        e_grid_low += low_part
        e_grid_high += high_part
        remaining_low -= low_part

    export_revenue_ft = e_export_steps.sum() * EXPORT_FT_PER_KWH
    net_bill_ft = import_cost_ft - export_revenue_ft

    return {
        "grid_import_low_kwh": e_grid_low,
        "grid_import_high_kwh": e_grid_high,
        "import_cost_ft": import_cost_ft,
        "export_revenue_ft": export_revenue_ft,
        "net_bill_ft": net_bill_ft,
    }


def simulate_one_user_greedy(
    p_load: np.ndarray,
    p_pv: np.ndarray,
    use_bess: bool,
    bess_size_kwh: float,
    eta_bess_in: float,
    eta_bess_out: float,
    eta_bess_stor: float,
    soc_bess_min: float,
    soc_bess_max: float,
    t_bess_min_h: float,
) -> dict:
    T = len(p_load)

    p_load = np.asarray(p_load, dtype=float)
    p_pv = np.asarray(p_pv, dtype=float)

    p_pv_to_load = np.zeros(T, dtype=float)
    p_pv_to_bess = np.zeros(T, dtype=float)
    p_bess_to_load = np.zeros(T, dtype=float)
    p_grid_to_load = np.zeros(T, dtype=float)
    p_inj = np.zeros(T, dtype=float)
    e_bess = np.zeros(T, dtype=float)

    if not use_bess or bess_size_kwh <= 1e-12:
        p_pv_to_load = np.minimum(p_load, p_pv)
        p_grid_to_load = np.maximum(p_load - p_pv_to_load, 0.0)
        p_inj = np.maximum(p_pv - p_pv_to_load, 0.0)
        e_bess[:] = 0.0
    else:
        soc_min_kwh = max(0.0, soc_bess_min) * bess_size_kwh
        soc_max_kwh = max(soc_min_kwh, soc_bess_max * bess_size_kwh)

        if t_bess_min_h is None or t_bess_min_h <= 1e-12:
            p_bess_max_kw = 1e18
        else:
            p_bess_max_kw = bess_size_kwh / t_bess_min_h

        soc = soc_min_kwh

        for t in range(T):
            soc *= eta_bess_stor

            load_t = max(p_load[t], 0.0)
            pv_t = max(p_pv[t], 0.0)

            pv_to_load = min(load_t, pv_t)
            deficit = load_t - pv_to_load
            surplus = pv_t - pv_to_load

            charge = 0.0
            if surplus > 1e-12:
                room_kwh = max(soc_max_kwh - soc, 0.0)
                max_charge_by_capacity_kw = room_kwh / max(eta_bess_in * DT, 1e-12)
                charge = min(surplus, p_bess_max_kw, max_charge_by_capacity_kw)
                soc += charge * eta_bess_in * DT
                surplus -= charge

            discharge = 0.0
            if deficit > 1e-12:
                avail_kwh = max(soc - soc_min_kwh, 0.0)
                max_discharge_by_energy_kw = avail_kwh * eta_bess_out / DT
                discharge = min(deficit, p_bess_max_kw, max_discharge_by_energy_kw)
                soc -= (discharge / max(eta_bess_out, 1e-12)) * DT
                deficit -= discharge

            grid_to_load = max(deficit, 0.0)
            inj = max(surplus, 0.0)

            p_pv_to_load[t] = pv_to_load
            p_pv_to_bess[t] = charge
            p_bess_to_load[t] = discharge
            p_grid_to_load[t] = grid_to_load
            p_inj[t] = inj
            e_bess[t] = soc

    load_kwh = p_load.sum() * DT
    pv_kwh = p_pv.sum() * DT
    pv_to_load_kwh = p_pv_to_load.sum() * DT
    pv_to_bess_kwh = p_pv_to_bess.sum() * DT
    bess_to_load_kwh = p_bess_to_load.sum() * DT
    grid_import_kwh = p_grid_to_load.sum() * DT
    injection_kwh = p_inj.sum() * DT

    bill = calc_bill_15min_brutto(
        p_grid_to_load=p_grid_to_load,
        p_inj=p_inj,
        dt=DT,
    )

    self_consumed_pv_kwh = pv_to_load_kwh + pv_to_bess_kwh
    self_consumption_ratio = self_consumed_pv_kwh / pv_kwh if pv_kwh > 1e-12 else 0.0
    self_sufficiency_ratio = (load_kwh - grid_import_kwh) / load_kwh if load_kwh > 1e-12 else 0.0

    return {
        "timeseries": {
            "p_load": p_load,
            "p_pv": p_pv,
            "p_pv_to_load": p_pv_to_load,
            "p_pv_to_bess": p_pv_to_bess,
            "p_bess_to_load": p_bess_to_load,
            "p_grid_to_load": p_grid_to_load,
            "p_inj": p_inj,
            "e_bess": e_bess,
        },
        "annual": {
            "load_kwh": load_kwh,
            "pv_kwh": pv_kwh,
            "pv_to_load_kwh": pv_to_load_kwh,
            "pv_to_bess_kwh": pv_to_bess_kwh,
            "bess_to_load_kwh": bess_to_load_kwh,
            "grid_import_kwh": grid_import_kwh,
            "injection_kwh": injection_kwh,
            "self_consumed_pv_kwh": self_consumed_pv_kwh,
            "self_consumption_ratio": self_consumption_ratio,
            "self_sufficiency_ratio": self_sufficiency_ratio,
            **bill,
        },
    }

def plot_household_percentiles_by_group(per_user_df: pd.DataFrame, out_case: Path):
    df = per_user_df.copy().sort_values("net_bill_ft").reset_index(drop=True)
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
            sub["net_bill_ft"],
            sub["percentile"],
            s=28,
            marker=marker_map[group],
            alpha=0.8,
            label=group,
        )
    x_lo = np.percentile(df["net_bill_ft"], 1)
    x_hi = np.percentile(df["net_bill_ft"], 99)
    plt.xlim(x_lo, x_hi)
    plt.xlabel("Éves nettó villanyszámla [Ft/év]")
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

    df = per_user_df.copy().sort_values("net_bill_ft").reset_index(drop=True)
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
            sub["net_bill_ft"],
            sub["percentile"],
            s=28,
            marker=marker_map[group],
            alpha=0.8,
            label=group,
        )

    # egyetlen globális S-görbe az összes háztartásra
    x = df["net_bill_ft"].to_numpy(dtype=float)
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

    x_lo = np.percentile(df["net_bill_ft"], 1)
    x_hi = np.percentile(df["net_bill_ft"], 100)
    plt.xlim(x_lo, x_hi)

    plt.xlabel("Éves nettó villanyszámla [Ft/év]")
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
    pv_ratio: float,
    include_bess: bool,
    include_boiler: bool,
    bess_share_pct: float = 100.0,
):
    out_case = Path(out_dir)
    out_case.mkdir(parents=True, exist_ok=True)

    (
        p_pv,
        p_ue,
        p_dhw,
        p_el_heater,
        size_elh,
        vol_hss_water,
        size_bess,
        eta_bess_in_u,
        eta_bess_out_u,
        eta_bess_stor_u,
        soc_bess_min_u,
        soc_bess_max_u,
        t_bess_min_u,
        T_env_u,
        T_max_u,
        T_min_u,
        T_in_u,
        a_hss_u,
        eta_elh_u,
        t_hss_min_in_u,
        user_names,
    ) = build_inputs(
        sim_yaml_path=sim_yaml,
        profiles_csv_path=profiles_csv,
        dhw_profile_path=dhw_profiles_csv,
        max_users=max_users,
        pv_ratio=pv_ratio,
        use_hss=False,
    )

    U = len(user_names)
    T = p_ue.shape[0]

    p_boiler = p_el_heater if include_boiler else np.zeros_like(p_el_heater)
    p_total_load = p_ue + p_boiler

    rows = []

    ts_p_load = np.zeros((T, U), dtype=float)
    ts_p_pv = np.zeros((T, U), dtype=float)
    ts_p_pv_to_load = np.zeros((T, U), dtype=float)
    ts_p_pv_to_bess = np.zeros((T, U), dtype=float)
    ts_p_bess_to_load = np.zeros((T, U), dtype=float)
    ts_p_grid_to_load = np.zeros((T, U), dtype=float)
    ts_p_inj = np.zeros((T, U), dtype=float)
    ts_e_bess = np.zeros((T, U), dtype=float)

    pv_annual_kwh = p_pv.sum(axis=0) * DT
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
            p_load=p_total_load[:, u],
            p_pv=p_pv[:, u],
            use_bess=bool(bess_enabled_arr[u]),
            bess_size_kwh=float(size_bess[u]),
            eta_bess_in=float(eta_bess_in_u[u]),
            eta_bess_out=float(eta_bess_out_u[u]),
            eta_bess_stor=float(eta_bess_stor_u[u]),
            soc_bess_min=float(soc_bess_min_u[u]),
            soc_bess_max=float(soc_bess_max_u[u]),
            t_bess_min_h=float(t_bess_min_u[u]),
        )

        annual = sim["annual"]
        times = sim["timeseries"]

        base_load_kwh = p_ue[:, u].sum() * DT
        boiler_kwh = p_boiler[:, u].sum() * DT

        rows.append({
            "user_name": user_names[u],
            "has_pv": bool((p_pv[:, u].sum() * DT) > 1e-9),
            "has_bess": bool(bess_enabled_arr[u] and size_bess[u] > 1e-9),
            "has_boiler": bool(include_boiler and (p_boiler[:, u].sum() * DT) > 1e-9),
            "group_label": (
                "PV+BESS" if (has_pv_arr[u] and bess_enabled_arr[u] and size_bess[u] > 1e-9)
                else "PV" if has_pv_arr[u]
                else "Nincs PV"
            ),
            "base_load_kwh": base_load_kwh,
            "boiler_kwh": boiler_kwh,
            "total_load_kwh": annual["load_kwh"],
            "pv_kwh": annual["pv_kwh"],
            "pv_to_load_kwh": annual["pv_to_load_kwh"],
            "pv_to_bess_kwh": annual["pv_to_bess_kwh"],
            "bess_to_load_kwh": annual["bess_to_load_kwh"],
            "grid_import_kwh": annual["grid_import_kwh"],
            "grid_import_low_kwh": annual["grid_import_low_kwh"],
            "grid_import_high_kwh": annual["grid_import_high_kwh"],
            "injection_kwh": annual["injection_kwh"],
            "self_consumed_pv_kwh": annual["self_consumed_pv_kwh"],
            "self_consumption_ratio": annual["self_consumption_ratio"],
            "self_sufficiency_ratio": annual["self_sufficiency_ratio"],
            "import_cost_ft": annual["import_cost_ft"],
            "export_revenue_ft": annual["export_revenue_ft"],
            "net_bill_ft": annual["net_bill_ft"],
        })

        ts_p_load[:, u] = times["p_load"]
        ts_p_pv[:, u] = times["p_pv"]
        ts_p_pv_to_load[:, u] = times["p_pv_to_load"]
        ts_p_pv_to_bess[:, u] = times["p_pv_to_bess"]
        ts_p_bess_to_load[:, u] = times["p_bess_to_load"]
        ts_p_grid_to_load[:, u] = times["p_grid_to_load"]
        ts_p_inj[:, u] = times["p_inj"]
        ts_e_bess[:, u] = times["e_bess"]

    per_user_df = pd.DataFrame(rows)
    per_user_df.to_csv(out_case / "per_user_summary.csv", index=False)


    plot_household_percentiles_by_group(per_user_df, out_case)
    plot_household_percentiles_by_group_with_global_scurve(per_user_df, out_case)

    total = {
        "case_name": case_name,
        "n_users": int(U),
        "base_load_kwh": float(per_user_df["base_load_kwh"].sum()),
        "boiler_kwh": float(per_user_df["boiler_kwh"].sum()),
        "total_load_kwh": float(per_user_df["total_load_kwh"].sum()),
        "pv_kwh": float(per_user_df["pv_kwh"].sum()),
        "pv_to_load_kwh": float(per_user_df["pv_to_load_kwh"].sum()),
        "pv_to_bess_kwh": float(per_user_df["pv_to_bess_kwh"].sum()),
        "bess_to_load_kwh": float(per_user_df["bess_to_load_kwh"].sum()),
        "grid_import_kwh": float(per_user_df["grid_import_kwh"].sum()),
        "grid_import_low_kwh": float(per_user_df["grid_import_low_kwh"].sum()),
        "grid_import_high_kwh": float(per_user_df["grid_import_high_kwh"].sum()),
        "injection_kwh": float(per_user_df["injection_kwh"].sum()),
        "import_cost_ft": float(per_user_df["import_cost_ft"].sum()),
        "export_revenue_ft": float(per_user_df["export_revenue_ft"].sum()),
        "net_bill_ft": float(per_user_df["net_bill_ft"].sum()),
        "bess_share_pct": float(bess_share_pct),
        "pv_user_count": int(has_pv_arr.sum()),
        "bess_enabled_user_count": int(bess_enabled_arr.sum()),
    }

    total["self_consumption_ratio"] = (
        (total["pv_to_load_kwh"] + total["pv_to_bess_kwh"]) / total["pv_kwh"]
        if total["pv_kwh"] > 1e-12 else 0.0
    )
    total["self_sufficiency_ratio"] = (
        (total["total_load_kwh"] - total["grid_import_kwh"]) / total["total_load_kwh"]
        if total["total_load_kwh"] > 1e-12 else 0.0
    )

    (out_case / "summary.json").write_text(
        json.dumps(total, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    def save_ts(arr: np.ndarray, filename: str):
        pd.DataFrame(arr, columns=user_names).to_csv(out_case / filename, index=False)

    save_ts(ts_p_load, "p_load.csv")
    save_ts(ts_p_pv, "p_pv.csv")
    save_ts(ts_p_pv_to_load, "p_pv_to_load.csv")
    save_ts(ts_p_pv_to_bess, "p_pv_to_bess.csv")
    save_ts(ts_p_bess_to_load, "p_bess_to_load.csv")
    save_ts(ts_p_grid_to_load, "p_grid_to_load.csv")
    save_ts(ts_p_inj, "p_inj.csv")
    save_ts(ts_e_bess, "e_bess.csv")

    community_ts = pd.DataFrame({
        "p_load_total": ts_p_load.sum(axis=1),
        "p_pv_total": ts_p_pv.sum(axis=1),
        "p_pv_to_load_total": ts_p_pv_to_load.sum(axis=1),
        "p_pv_to_bess_total": ts_p_pv_to_bess.sum(axis=1),
        "p_bess_to_load_total": ts_p_bess_to_load.sum(axis=1),
        "p_grid_to_load_total": ts_p_grid_to_load.sum(axis=1),
        "p_inj_total": ts_p_inj.sum(axis=1),
        "e_bess_total": ts_e_bess.sum(axis=1),
    })
    community_ts.to_csv(out_case / "community_timeseries.csv", index=False)

    print(f"[INFO] Kész: {case_name}")
    print(f"[INFO] Output: {out_case}")
    print(json.dumps(total, indent=2, ensure_ascii=False))