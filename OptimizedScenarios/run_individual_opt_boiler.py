from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from InputReading import read_simulation_inputs
from Utility.configuration import config
# --- import optimizer locally ---------------------------------------------------
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.append(str(HERE))
from OptimizedIndividualScenarios.individual_opt_boiler import \
    individual_opt_boiler # expects (T,U) arrays, sizes, etc. :contentReference[oaicite:1]{index=1}


# --- input builder -------------------------------------------------------------
def build_inputs(
        sim_yaml_path: os.PathLike,
        profiles_csv_path: os.PathLike,
        dhw_profile_path: os.PathLike,
        max_users: int | None = 10,
        target_user: str | None = None,
        search_roots: Iterable[os.PathLike] | None = None,
        dt: float = config.getfloat("simulation", "dt_hours"),
) -> tuple:
    """Compatibility adapter backed exclusively by :mod:`InputReading`."""
    inputs = read_simulation_inputs(
        sim_yaml_path=sim_yaml_path,
        profiles_csv_path=profiles_csv_path,
        dhw_profile_path=dhw_profile_path,
        max_users=max_users,
        target_user=target_user,
        search_roots=search_roots,
        dt=dt,
    )
    return (
        inputs.p_pv_kw, inputs.p_ue_kw, inputs.p_dhw_kw, inputs.p_el_heater_kw,
        inputs.size_elh, inputs.vol_hss_water, inputs.T_env, inputs.T_max,
        inputs.T_min, inputs.T_in, inputs.a_hss, inputs.eta_elh,
        inputs.t_hss_min_in, inputs.user_names,
    )


# --- runner --------------------------------------------------------------------

def run(
        sim_yaml: os.PathLike,
        profiles_csv: os.PathLike,
        dhw_profiles_csv: os.PathLike,
        out_dir: os.PathLike,
        max_users: int = 10,
        target_user: str | None = None,
        run_lp: bool = True,
        boiler_tariff: str = "B",
) -> dict:
    dt = config.getfloat("simulation", "dt_hours")

    boiler_tariff = str(boiler_tariff).upper().strip()
    if boiler_tariff not in {"A", "B"}:
        raise ValueError(f"boiler_tariff csak 'A' vagy 'B' lehet, nem: {boiler_tariff}")

    inputs = read_simulation_inputs(
        sim_yaml_path=sim_yaml,
        profiles_csv_path=profiles_csv,
        dhw_profile_path=dhw_profiles_csv,
        max_users=max_users,
        dt=dt,
        target_user=target_user,
    )
    p_pv = inputs.p_pv_kw
    p_ue = inputs.p_ue_kw
    p_dhw = inputs.p_dhw_kw
    p_el_heater = inputs.p_el_heater_kw
    size_elh = inputs.size_elh
    vol_hss_water = inputs.vol_hss_water
    T_env_u = inputs.T_env
    T_max_u = inputs.T_max
    T_min_u = inputs.T_min
    T_in_u = inputs.T_in
    a_hss_u = inputs.a_hss
    eta_elh_u = inputs.eta_elh
    t_hss_min_in_u = inputs.t_hss_min_in
    user_names = inputs.user_names

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

            price_grid_a_low=config.getfloat("tariffs", "grid_a_low_ft_per_kwh"),
            price_grid_a_high=config.getfloat("tariffs", "grid_a_high_ft_per_kwh"),
            price_grid_b_low=config.getfloat("tariffs", "grid_b_low_ft_per_kwh"),
            price_grid_b_high=config.getfloat("tariffs", "grid_b_high_ft_per_kwh"),
            price_pv_grid=config.getfloat("tariffs", "pv_export_ft_per_kwh"),
            grid_a_low_cap_kwh=config.getfloat("tariffs", "grid_a_low_limit_kwh"),
            grid_b_low_cap_kwh=config.getfloat("tariffs", "grid_b_low_limit_kwh"),

            run_lp=run_lp,
            msg=False,
            enforce_cl_rules=True,
            cl_max_on_hours_per_day=8.0,
            cl_min_midday_hours_per_day=4.0,
            gapRel=0.005,
            timeLimit=None,
            boiler_tariff=boiler_tariff,
            objective="bill",
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
            "boiler_tariff": boiler_tariff,

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
        "boiler_tariff": boiler_tariff,
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
    ap.add_argument("--out", default=str(config.getpath("paths", "individual_boiler_output")), help="Output directory.")
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
        dhw_profiles_csv=args.dhw_profiles,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    # A tarifa
    summary = run(
        sim_yaml=config.getpath("paths", "simulation_yaml"),
        profiles_csv=config.getpath("paths", "profiles_csv"),
        dhw_profiles_csv=config.getpath("paths", "dhw_profiles_v2_csv"),
        out_dir=config.getpath("paths", "individual_boiler_output"),
        max_users=105,
        run_lp=False,
        boiler_tariff="A",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    # B tarifa
    # summary = run(
    #     sim_yaml="../Input/simulation_config_disaggregated_with_userlist.yaml",
    #     profiles_csv="../Input/measurements_disaggregated.csv",
    #     dhw_profiles_csv="../Input/dhw_v2.csv",
    #     out_dir="results_individual_opt_boiler_B_tariff",
    #     max_users=105,
    #     run_lp=False,
    #     boiler_tariff="B",
    # )
    # print(json.dumps(summary, indent=2, ensure_ascii=False))
