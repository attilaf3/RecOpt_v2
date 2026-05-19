from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

# --- import optimizer locally ---------------------------------------------------
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.append(str(HERE))
from individual_opt_boiler import \
    individual_opt_boiler # expects (T,U) arrays, sizes, etc. :contentReference[oaicite:1]{index=1}


# --- helpers -------------------------------------------------------------------
def _keep_15min(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).ravel()
    if v.size != 35040:
        raise ValueError(f"A profil hossza {v.size}, de itt 35040 kell.")
    return v

def _energy_profile_kwh_step(v: np.ndarray) -> np.ndarray:
    """
    Beolvasott idősor energiaként kezelve: e [kWh/lépés].
    """
    return np.maximum(_keep_15min(v), 0.0)


def _power_profile_kw_from_energy(v: np.ndarray, dt: float) -> np.ndarray:
    """
    e [kWh/lépés] -> p [kW].
    """
    e_kwh_step = _energy_profile_kwh_step(v)
    return e_kwh_step / float(dt)


def _find_user_yaml(roots: Iterable[os.PathLike], name: str) -> Optional[Path]:
    for r in roots:
        for cand in (name, f"{name}.yaml"):
            p = Path(r) / cand
            if p.exists():
                return p
    return None


# --- input builder -------------------------------------------------------------
def build_inputs(
        sim_yaml_path: os.PathLike,
        profiles_csv_path: os.PathLike,
        dhw_profile_path: os.PathLike,
        max_users: int = 10,
        pv_ratio: float = 1.0,
        search_roots: Iterable[os.PathLike] | None = None,
        dt: float = 0.25,
) -> Tuple[
    np.ndarray,  # p_pv (35040, U)
    np.ndarray,  # p_ue (35040, U)
    np.ndarray,  # p_dhw (35040, U)
    np.ndarray,  # p_el_heater (35040, U)
    np.ndarray,  # size_elh (U,)
    np.ndarray,  # vol_hss_water (U,)
    list[float], # T_env_u
    list[float], # T_max_u
    list[float], # T_min_u
    list[float], # T_in_u
    list[float], # a_hss_u
    list[float], # eta_elh_u
    list[str],   # user_names
]:
    """
    Return:
      p_pv, p_ue, p_ut: np.ndarray (35040, U)
      size_elh: np.ndarray (U,)
      vol_hss_water: np.ndarray (U,)
      user_names: list[str]
    """
    sim_yaml_path = Path(sim_yaml_path)
    profiles_csv_path = Path(profiles_csv_path)
    if search_roots is None:
        search_roots = [sim_yaml_path.parent / "Users", sim_yaml_path.parent]

    sim = yaml.safe_load(sim_yaml_path.read_text(encoding="utf-8")) or {}
    users_list = list(sim.get("users_list", []))[: int(max_users)]

    # exclude some pseudo users by name (optional)
    EXCLUDE = {"battery", "bess", "community"}
    users_list = [u for u in users_list if str(u).strip().lower() not in EXCLUDE]
    if not users_list:
        raise RuntimeError("A simulation YAML nem tartalmaz users_list-et vagy max_users=0.")

    df = pd.read_csv(profiles_csv_path, index_col=0)
    df.columns = [str(c) for c in df.columns]  # oszlopnevek legyenek stringek
    df = df[[c for c in df.columns if c.lower() not in {"battery", "bess", "community"}]]

    n = 1
    dhw_profile_path = Path(dhw_profile_path)
    dhw = pd.read_csv(dhw_profile_path, index_col=0) / n
    dhw.columns = [str(c) for c in dhw.columns]

    p_pv_cols, p_ue_cols, p_el_heater_cols, dhw_cols = [], [], [], []

    size_elh = []
    vol_hss_water = []
    user_names = []

    T_env_u = []
    T_min_u = []
    T_max_u = []
    a_hss_u = []
    T_in_u = []
    T_out_u = []
    eta_elh_u = []
    t_hss_min_in_u = []

    for user_key in users_list:
        ypath = _find_user_yaml(search_roots, user_key)
        if not ypath:
            print(f"[WARN] YAML nem található: {user_key} — kihagyom.")
            continue

        u = yaml.safe_load(ypath.read_text(encoding="utf-8")) or {}
        units = (u.get("units") or {})
        name = (units.get("name") or {}).get("name", str(user_key))
        user_names.append(name)

        # --- UE (villamos fogyasztás) ---
        ue = units.get("ue") or {}
        ue_prof = str(ue.get("profile")) if ue.get("profile") is not None else None

        if ue_prof and ue_prof in df.columns:
            e_ue_kwh_step = _energy_profile_kwh_step(df[ue_prof].to_numpy())
            p_ue_kw = e_ue_kwh_step / dt
            p_ue_cols.append(p_ue_kw)
        else:
            p_ue_cols.append(np.zeros(35040, dtype=float))

        # --- PV (termelés) ---
        pv = units.get("pv") or {}
        pv_prof = str(pv.get("profile")) if pv.get("profile") is not None else None

        if pv_prof and pv_prof in df.columns:
            e_pv_kwh_step = _energy_profile_kwh_step(df[pv_prof].to_numpy()) * float(pv_ratio)
            p_pv_kw = e_pv_kwh_step / dt
            p_pv_cols.append(p_pv_kw)
        else:
            p_pv_cols.append(np.zeros(35040, dtype=float))


        # --- HSS / bojler adatok ---
        hss = units.get("hss") or {}
        heater = units.get("ut") or {}

        # DHW profil: liter/lépés -> e_dhw [kWh/lépés] -> p_dhw [kW]
        if hss.get("profile") is not None and str(hss["profile"]) in dhw.columns:
            L_dhw_step = np.maximum(_keep_15min(dhw[str(hss["profile"])].to_numpy()), 0.0)

            RHO_WATER_KG_PER_L = 1.0
            CP_WATER_J_PER_KGK = 4186.0
            J_PER_KWH = 3_600_000.0
            KWH_PER_L_PER_K = RHO_WATER_KG_PER_L * CP_WATER_J_PER_KGK / J_PER_KWH

            T_in = float(hss.get("T_in", 10))
            T_out = float(hss.get("T_out", 55))
            dT = max(0.0, T_out - T_in)

            e_dhw_kwh_step = L_dhw_step * KWH_PER_L_PER_K * dT
            p_dhw_kw = e_dhw_kwh_step / dt
            dhw_cols.append(p_dhw_kw.astype(float))
        else:
            dhw_cols.append(np.zeros(35040, dtype=float))

        # Fix villamos bojlerprofil / UT profil
        p_el_heater_prof = str(heater.get("profile")) if heater.get("profile") is not None else None
        if p_el_heater_prof and p_el_heater_prof in df.columns:
            p_el_heater_cols.append(
                _power_profile_kw_from_energy(df[p_el_heater_prof].to_numpy(), dt)
            )
        else:
            p_el_heater_cols.append(np.zeros(35040, dtype=float))

        # Ha nincs HSS, akkor nulla méretű bojlert feltételezünk
        size_elh.append(float(hss.get("size_elh", 0.0)))
        vol_hss_water.append(float(hss.get("vol_hss_water", 0.0)))
        # HSS paraméterek
        T_env_u.append(float(hss.get("T_env", 20)))
        T_max_u.append(float(hss.get("T_max", 65)))
        T_min_u.append(float(hss.get("T_min", 10)))
        T_in_u.append(float(hss.get("T_in", 10)))
        T_out_u.append(float(hss.get("T_out", 55)))
        a_hss_u.append(float(hss.get("a_hss", 0.01275)))
        eta_elh_u.append(float(hss.get("eta_elh", 0.95)))
        t_hss_min_in_u.append(float(hss.get("t_hss_min_in", 0.0)))

    if not user_names:
        raise RuntimeError("Nincs érvényes felhasználó.")

    # (T,U) mátrixok
    size_elh = np.asarray(size_elh, float)
    vol_hss_water = np.asarray(vol_hss_water, float)

    U = len(user_names)

    p_pv = np.column_stack(p_pv_cols).astype(float)
    p_ue = np.column_stack(p_ue_cols).astype(float)

    p_dhw = np.column_stack(dhw_cols).astype(float)
    p_el_heater = np.column_stack(p_el_heater_cols).astype(float)

    return (
        p_pv,
        p_ue,
        p_dhw,
        p_el_heater,
        np.asarray(size_elh, float),
        np.asarray(vol_hss_water, float),
        T_env_u,
        T_max_u,
        T_min_u,
        T_in_u,
        a_hss_u,
        eta_elh_u,
        t_hss_min_in_u,
        user_names,
    )


# --- runner --------------------------------------------------------------------

def run(
        sim_yaml: os.PathLike,
        profiles_csv: os.PathLike,
        dhw_profiles_csv: os.PathLike,
        out_dir: os.PathLike,
        max_users: int = 10,
        run_lp: bool = True,
        pv_ratio: float = 1.0,
) -> dict:
    dt = 0.25
    (
        p_pv, p_ue, p_dhw, p_el_heater,
        size_elh, vol_hss_water,
        T_env_u, T_max_u, T_min_u, T_in_u, a_hss_u, eta_elh_u, t_hss_min_in_u,
        user_names
    ) = build_inputs(
        sim_yaml_path=sim_yaml,
        profiles_csv_path=profiles_csv,
        dhw_profile_path=dhw_profiles_csv,
        max_users=max_users,
        pv_ratio=pv_ratio,
        dt=dt,
    )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Háztartások száma: {len(user_names)}")

    energy_rows = []
    finance_rows = []

    for u, name in enumerate(user_names):
        print(f"[INFO] Optimalizálás: {u+1}/{len(user_names)} - {name}")

        has_pv = bool(np.sum(p_pv[:, u]) > 1e-9)
        has_boiler = bool((size_elh[u] > 1e-9) and (vol_hss_water[u] > 1e-9))

        if has_pv and has_boiler:
            # PV + bojler: optimalizált HSS modell
            p_dhw_eff = p_dhw[:, u]
            p_el_heater_fixed_eff = np.zeros_like(p_ue[:, u])
            size_elh_eff = float(size_elh[u])
            vol_hss_water_eff = float(vol_hss_water[u])

        elif has_boiler and not has_pv:
            # Bojler van, PV nincs: nincs mit optimalizálni, fix villamos bojlerprofil
            p_dhw_eff = np.zeros_like(p_ue[:, u])
            p_el_heater_fixed_eff = p_el_heater[:, u]
            size_elh_eff = 0.0
            vol_hss_water_eff = 0.0

        else:
            # Nincs bojler
            p_dhw_eff = np.zeros_like(p_ue[:, u])
            p_el_heater_fixed_eff = np.zeros_like(p_ue[:, u])
            size_elh_eff = 0.0
            vol_hss_water_eff = 0.0

        res = individual_opt_boiler(
            p_pv=p_pv[:, u],
            p_ue=p_ue[:, u],
            p_dhw=p_dhw_eff,
            dt=dt,
            size_elh=size_elh_eff,
            vol_hss_water=vol_hss_water_eff,
            T_env=float(T_env_u[u]),
            T_max=float(T_max_u[u]),
            T_min=float(T_min_u[u]),
            T_in=float(T_in_u[u]),
            a_hss=float(a_hss_u[u]),
            eta_elh=float(eta_elh_u[u]),
            p_el_heater_fixed=p_el_heater_fixed_eff,

            price_grid_a_low=36.0,
            price_grid_a_high=71.0,
            price_grid_b_low=23.0,
            price_grid_b_high=61.0,
            price_pv_grid=5.0,
            grid_a_low_cap_kwh=2523.0,
            grid_b_low_cap_kwh=2523.0,

            run_lp=run_lp,
            msg=False,
            enforce_cl_rules=True,
            cl_max_on_hours_per_day=8.0,
            cl_min_midday_hours_per_day=4.0,
            gapRel=0.005,
            timeLimit=None,
            objective="grid",
        )

        # opcionális: külön idősor mentés háztartásonként
        ts = pd.DataFrame({
            "p_pv": p_pv[:, u],
            "p_ue": p_ue[:, u],
            "p_dhw": p_dhw_eff,

            "p_pv_load": res["p_pv_load"],
            "p_pv_elh": res["p_pv_elh"],
            "p_pv_grid": res["p_pv_grid"],

            "p_grid_load": res["p_grid_load"],
            "p_grid_elh": res["p_grid_elh"],

            # explicit grid idősorok, BESS modellhez hasonlóan
            "p_grid_import": res["p_grid_import"],
            "p_grid_export": res["p_grid_export"],

            "p_elh_in": res["p_elh_in"],
            "p_hss_in": res["p_hss_in"],
            "p_hss_out": res["p_hss_out"],

            "t_hss": res["t_hss"],
            "e_hss_stor": res["e_hss_stor"],

            "d_cl": res["d_cl"],
            "d_export": res["d_export"],

            # tarifa bontás idősorosan is
            "p_el_heater_fixed": p_el_heater_fixed_eff,
            "p_pv_elh_fixed": res["p_pv_elh_fixed"],
            "p_grid_elh_fixed": res["p_grid_elh_fixed"],
            "p_grid_to_base": res["p_grid_to_base"],
            "p_grid_to_boiler": res["p_grid_to_boiler"],
            "p_el_heater_total": res["p_el_heater_total"],
            "p_pv_to_boiler": res["p_pv_to_boiler"],

            "e_grid_a_low_step": res["e_grid_a_low_step"],
            "e_grid_a_high_step": res["e_grid_a_high_step"],
            "e_grid_b_low_step": res["e_grid_b_low_step"],
            "e_grid_b_high_step": res["e_grid_b_high_step"],
            "remaining_a_low_block_kwh": res["remaining_a_low_block_kwh"],
            "remaining_b_low_block_kwh": res["remaining_b_low_block_kwh"],
        })

        safe_name = str(name).replace("/", "_").replace("\\", "_")
        ts.to_csv(out / f"timeseries_{safe_name}.csv", index=False)

        energy_rows.append({
            "household": name,
            "has_pv": int(np.sum(p_pv[:, u]) > 1e-9),
            "has_boiler": int((size_elh[u] > 1e-9) and (vol_hss_water[u] > 1e-9)),

            "pv_gen_kwh": float(np.sum(p_pv[:, u]) * dt),
            "load_kwh": float(np.sum(p_ue[:, u]) * dt),
            "dhw_kwh_th": float(np.sum(p_dhw_eff) * dt),

            "grid_import_a_low_kwh": res["grid_import_a_low_kwh"],
            "grid_import_a_high_kwh": res["grid_import_a_high_kwh"],
            "grid_import_b_low_kwh": res["grid_import_b_low_kwh"],
            "grid_import_b_high_kwh": res["grid_import_b_high_kwh"],

            "grid_import_low_kwh": res["grid_import_low_kwh"],
            "grid_import_high_kwh": res["grid_import_high_kwh"],
            "grid_import_a_kwh": res["grid_import_a_kwh"],
            "grid_import_b_kwh": res["grid_import_b_kwh"],
            "grid_import_total_kwh": float(np.sum(res["p_grid_import"]) * dt),
            "grid_export_kwh": float(np.sum(res["p_grid_export"]) * dt),

            "pv_to_load_kwh": float(np.sum(res["p_pv_load"]) * dt),
            "pv_to_boiler_kwh": float(np.sum(res["p_pv_to_boiler"]) * dt),

            "grid_to_load_kwh": float(np.sum(res["p_grid_load"]) * dt),
            "grid_to_boiler_kwh": float(np.sum(res["p_grid_to_boiler"]) * dt),

            "boiler_el_input_kwh": float(np.sum(res["p_el_heater_total"]) * dt),
            "boiler_th_input_kwh": float(np.sum(res["p_hss_in"]) * dt),
            "boiler_th_output_kwh": float(np.sum(res["p_hss_out"]) * dt),

            "final_hss_energy_kwh": float(res["e_hss_stor"][-1]) if len(res["e_hss_stor"]) else 0.0,
            "fixed_boiler_el_kwh": float(np.sum(p_el_heater_fixed_eff) * dt),
            "hss_active": int(res["hss_active"]),

            "objective_value": res["objective_value"],
            "objective_type": res["objective_type"],
            "status": res["status"],
        })

        finance_rows.append({
            "household": name,
            "import_cost_a_ft": res["import_cost_a_ft"],
            "import_cost_b_ft": res["import_cost_b_ft"],
            "import_cost_ft": res["import_cost_ft"],
            "export_revenue_ft": res["export_revenue_ft"],
            "brt_bill_ft": res["brt_bill_ft"],
            "net_cost_Ft": res["net_cost_Ft"],
        })

    energy_df = pd.DataFrame(energy_rows)
    finance_df = pd.DataFrame(finance_rows)

    energy_df.to_csv(out / "household_energy_summary.csv", index=False)
    finance_df.to_csv(out / "household_finance_summary.csv", index=False)

    # egy kombinált tábla is hasznos lehet
    combined_df = energy_df.merge(finance_df, on="household", how="left")
    combined_df.to_csv(out / "household_summary_combined.csv", index=False)

    summary = {
        "n_households": len(user_names),
        "out_dir": str(out),
        "total_grid_import_kwh": float(energy_df["grid_import_total_kwh"].sum()),
        "total_grid_export_kwh": float(energy_df["grid_export_kwh"].sum()),
        "total_brt_bill_ft": float(finance_df["brt_bill_ft"].sum()),
        "total_import_cost_ft": float(finance_df["import_cost_ft"].sum()),
        "total_export_revenue_ft": float(finance_df["export_revenue_ft"].sum()),
    }

    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    return summary


# --- CLI -----------------------------------------------------------------------
def main(argv: list[str] | None = None):
    ap = argparse.ArgumentParser(description="Optimize multi-user from YAMLs (subset).")
    ap.add_argument("--sim", required=True, help="Path to simulation_config YAML.")
    ap.add_argument("--profiles", required=True, help="Path to disaggregated profiles CSV.")
    ap.add_argument("--dhw_profiles", help="Domestic hot water profiles CSV.")
    ap.add_argument("--out", default="results_subset", help="Output directory.")
    ap.add_argument("--max-users", type=int, default=10, help="Első N user a users_list-ből.")
    ap.add_argument("--pv-ratio", type=float, default=1.0, help="PV éves energia szorzó.")
    ap.add_argument("--use-hss", action="store_true", help="HSS logika (ΔT·c_víz) használata.")
    ap.add_argument("--mip", action="store_true", help="Bináris változók bekapcsolása (alap: LP).")
    args = ap.parse_args(argv)

    summary = run(
        sim_yaml=args.sim,
        profiles_csv=args.profiles,
        out_dir=args.out,
        max_users=args.max_users,
        run_lp=not args.mip,
        pv_ratio=args.pv_ratio,
        dhw_profiles_csv=args.dhw_profiles,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    summary = run(
        sim_yaml="../Inputs/simulation_config_disaggregated_pv_original_increase_1.0.yaml",
        profiles_csv="../Inputs/measurements_disaggregated_pv_original_increase_1.0.csv",
        dhw_profiles_csv="../Inputs/dhw.csv",
        out_dir="results_individual_opt_boiler",
        max_users=105,
        run_lp=False,
        pv_ratio=1.0,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
