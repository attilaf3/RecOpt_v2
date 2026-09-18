from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from Economics import calculate_economics, settle_individual_optimization_as_community
from InputReading import read_simulation_inputs
from Utility.configuration import config
from OptimizedIndividualScenarios.individual_optimizer import optimize_household

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
        settlement_mode: str = "individual",
        sharing_mode: str = "proportional",
        pairing_mode: str | None = None,
        include_boiler: bool = True,
        include_bess: bool = False,
        include_heat_pump: bool = False,
        objective: str = "bill",
        bess_share_pct: float = 100.0,
        bess_terminal: str = "cyclic",
        bess_grid_charge: str | None = None,
        allow_grid_charge_for_min_soc: bool = False,
        pv_ratio: float = 1.0,
        use_hss: bool = True,
) -> dict:
    if include_heat_pump:
        raise NotImplementedError("Flexible heat-pump optimization is not implemented")
    if not 0 <= bess_share_pct <= 100:
        raise ValueError("bess_share_pct must be between 0 and 100")
    dt = config.getfloat("simulation", "dt_hours")

    boiler_tariff = str(boiler_tariff).upper().strip()
    if boiler_tariff not in {"A", "B"}:
        raise ValueError(f"boiler_tariff csak 'A' vagy 'B' lehet, nem: {boiler_tariff}")
    settlement_mode = str(settlement_mode).lower().strip()
    if settlement_mode not in {"individual", "community"}:
        raise ValueError("settlement_mode csak 'individual' vagy 'community' lehet.")

    inputs = read_simulation_inputs(
        sim_yaml_path=sim_yaml,
        profiles_csv_path=profiles_csv,
        dhw_profile_path=dhw_profiles_csv,
        max_users=max_users,
        dt=dt,
        target_user=target_user,
        pv_ratio=pv_ratio,
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

    candidates = np.flatnonzero((p_pv.sum(axis=0) > 1e-9) & (inputs.size_bess > 0))
    bess_users = set(candidates[:round(len(candidates)*bess_share_pct/100)]) if include_bess else set()

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Háztartások száma: {len(user_names)}")

    energy_rows = []
    finance_rows = []
    user_timeseries: list[tuple[str, pd.DataFrame]] = []
    bess_boundaries = []
    shape = p_pv.shape
    e_grid_import_a = np.zeros(shape, dtype=float)
    e_grid_import_b = np.zeros(shape, dtype=float)
    e_grid_export = np.zeros(shape, dtype=float)

    for u, name in enumerate(user_names):
        print(f"[INFO] Optimalizálás: {u+1}/{len(user_names)} - {name}")

        has_pv = bool(np.sum(p_pv[:, u]) > 1e-9)
        has_boiler = bool((size_elh[u] > 1e-9) and (vol_hss_water[u] > 1e-9))

        if include_boiler and use_hss and has_pv and has_boiler:
            # PV + bojler: optimalizált HSS modell
            p_dhw_eff = p_dhw[:, u]
            p_el_heater_fixed_eff = np.zeros_like(p_ue[:, u])
            size_elh_eff = float(size_elh[u])
            vol_hss_water_eff = float(vol_hss_water[u])

        elif has_boiler:
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

        res = optimize_household(
            include_boiler=include_boiler and use_hss,
            include_bess=u in bess_users,
            size_bess=float(inputs.size_bess[u]),
            eta_bess_in=float(inputs.eta_bess_in[u]),
            eta_bess_out=float(inputs.eta_bess_out[u]),
            eta_bess_stor=float(inputs.eta_bess_stor[u]),
            soc_bess_min=float(inputs.soc_bess_min[u]),
            soc_bess_max=float(inputs.soc_bess_max[u]),
            soc_bess_init=float(inputs.soc_bess_min[u]),
            t_bess_min=float(inputs.t_bess_min[u]),
            bess_terminal=bess_terminal, bess_grid_charge=bess_grid_charge,
            allow_grid_charge_for_min_soc=allow_grid_charge_for_min_soc,
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
            objective=objective,
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

        for key in ("p_pv_bess", "p_grid_bess", "p_bess_in", "p_bess_out",
                    "p_bess_to_boiler", "e_bess", "e_bess_start", "e_bess_end", "d_bess_ch", "d_bess_dis"):
            ts[key] = res[key]
        safe_name = str(name).replace("/", "_").replace("\\", "_")
        pd.DataFrame({"e_bess_boundary": res["e_bess_boundary"]}).to_csv(
            out / f"bess_boundary_{safe_name}.csv", index=False)
        user_timeseries.append((safe_name, ts))
        bess_boundaries.append(res["e_bess_boundary"])

        if boiler_tariff == "A":
            e_grid_import_a[:, u] = np.asarray(res["p_grid_import"], dtype=float) * dt
        else:
            e_grid_import_a[:, u] = (np.asarray(res["p_grid_to_base"], dtype=float)+res["p_grid_bess"]) * dt
            e_grid_import_b[:, u] = np.asarray(res["p_grid_to_boiler"], dtype=float) * dt
        e_grid_export[:, u] = np.asarray(res["p_grid_export"], dtype=float) * dt

        energy_rows.append({
            "household": name,
            "has_pv": int(np.sum(p_pv[:, u]) > 1e-9),
            "has_boiler": int((size_elh[u] > 1e-9) and (vol_hss_water[u] > 1e-9)),
            "boiler_tariff": boiler_tariff,
            "has_bess": res["bess_active"],
            "bess_charge_kwh": float(res["p_bess_in"].sum()*dt),
            "bess_discharge_kwh": float(res["p_bess_out"].sum()*dt),

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
    optimization_finance_df = pd.DataFrame(finance_rows)

    if settlement_mode == "community":
        economics = settle_individual_optimization_as_community(
            e_grid_import_a=e_grid_import_a,
            e_grid_import_b=e_grid_import_b,
            e_grid_export=e_grid_export,
            user_names=list(user_names),
            allocation_mode=sharing_mode,
            pairing_mode=pairing_mode,
        )
        settled_ts = economics["timeseries"]
    else:
        economics = calculate_economics(
            e_grid_import_a=e_grid_import_a,
            e_grid_import_b=e_grid_import_b,
            e_grid_export=e_grid_export,
            user_names=list(user_names),
        )
        settled_ts = {
            "e_grid_import_a": e_grid_import_a,
            "e_grid_import_b": e_grid_import_b,
            "e_grid_export": e_grid_export,
            "e_shared_in": np.zeros(shape, dtype=float),
            "e_shared_out": np.zeros(shape, dtype=float),
        }

    finance_df = economics["per_user_df"]
    from Utility.result_schema import save_bess_result
    save_bess_result({"e_bess_boundary": np.column_stack(bess_boundaries),
        **{key: np.column_stack([ts[key].to_numpy() for _, ts in user_timeseries])
           for key in ("p_pv_bess", "p_grid_bess", "p_bess_out")}}, out, dt, user_names)
    for u, (safe_name, ts) in enumerate(user_timeseries):
        ts["p_settled_grid_import_a"] = settled_ts["e_grid_import_a"][:, u] / dt
        ts["p_settled_grid_import_b"] = settled_ts["e_grid_import_b"][:, u] / dt
        ts["p_settled_grid_export"] = settled_ts["e_grid_export"][:, u] / dt
        ts["p_shared_in"] = settled_ts["e_shared_in"][:, u] / dt
        ts["p_shared_out"] = settled_ts["e_shared_out"][:, u] / dt
        ts.to_csv(out / f"timeseries_{safe_name}.csv", index=False)

    energy_df.to_csv(out / "household_energy_summary.csv", index=False)
    finance_df.to_csv(out / "household_finance_summary.csv", index=False)
    optimization_finance_df.to_csv(out / "individual_optimization_finance_reference.csv", index=False)

    # egy kombinált tábla is hasznos lehet
    combined_df = energy_df.merge(finance_df, on="household", how="left")
    combined_df.to_csv(out / "household_summary_combined.csv", index=False)

    if settlement_mode == "community":
        pd.DataFrame(
            economics["shared_settlement"]["pair_kwh"],
            index=user_names,
            columns=user_names,
        ).to_csv(out / "shared_pair_kwh_seller_x_buyer.csv")
        pd.DataFrame(
            economics["shared_settlement"]["pair_energy_payment_ft"],
            index=user_names,
            columns=user_names,
        ).to_csv(out / "shared_pair_energy_payment_ft_seller_x_buyer.csv")

    summary = {
        "n_households": len(user_names),
        "out_dir": str(out),
        "scenario_code": ("3d-K" if settlement_mode == "community" else "3d-I") if objective == "bill" else None,
        "settlement_mode": settlement_mode,
        "sharing_mode": sharing_mode if settlement_mode == "community" else None,
        "physical_grid_import_kwh": float(energy_df["grid_import_total_kwh"].sum()),
        "physical_grid_export_kwh": float(energy_df["grid_export_kwh"].sum()),
        "total_grid_import_kwh": economics["summary"]["grid_import_kwh"],
        "total_grid_export_kwh": economics["summary"]["grid_export_kwh"],
        "total_shared_in_kwh": economics["summary"]["shared_in_kwh"],
        "total_shared_out_kwh": economics["summary"]["shared_out_kwh"],
        "total_brt_bill_ft": economics["summary"]["brt_bill_ft"],
        "total_import_cost_ft": economics["summary"]["grid_import_cost_ft"],
        "total_export_revenue_ft": economics["summary"]["grid_export_revenue_ft"],
        "total_shared_purchase_cost_ft": economics["summary"]["shared_purchase_cost_ft"],
        "total_shared_revenue_ft": economics["summary"]["shared_revenue_ft"],
        "boiler_tariff": boiler_tariff,
        "include_boiler": include_boiler,
        "include_bess": include_bess,
        "objective": objective,
        "bess_terminal": bess_terminal,
        "bess_grid_charge": bess_grid_charge,
        "allow_grid_charge_for_min_soc": allow_grid_charge_for_min_soc,
    }

    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    return summary


def run_3d_k(**kwargs) -> dict:
    """Run individual financial optimization with community settlement (3d-K)."""
    kwargs = dict(kwargs)
    kwargs["settlement_mode"] = "community"
    kwargs.setdefault("boiler_tariff", "A")
    kwargs.setdefault("out_dir", config.getpath("paths", "scenario_3d_k_output"))
    return run(**kwargs)


# --- CLI -----------------------------------------------------------------------
def main(argv: list[str] | None = None):
    ap = argparse.ArgumentParser(description="Optimize multi-user from YAMLs (subset).")
    ap.add_argument("--sim", required=True, help="Path to simulation_config YAML.")
    ap.add_argument("--profiles", required=True, help="Path to disaggregated profiles CSV.")
    ap.add_argument("--dhw_profiles", required=True, help="Domestic hot water profiles CSV.")
    ap.add_argument("--out", default=str(config.getpath("paths", "individual_boiler_output")), help="Output directory.")
    ap.add_argument("--max-users", type=int, default=10, help="Első N user a users_list-ből.")
    ap.add_argument("--pv-ratio", type=float, default=1.0, help="PV éves energia szorzó.")
    ap.add_argument("--use-hss", action="store_true", default=True, help="HSS hőtároló-logika használata.")
    ap.add_argument("--no-use-hss", dest="use_hss", action="store_false")
    ap.add_argument("--bess", action="store_true", help="Enable battery optimization.")
    ap.add_argument("--allow-grid-charge-for-min-soc", action="store_true")
    ap.add_argument("--objective", choices=["bill", "grid"], default="bill")
    ap.add_argument("--boiler-tariff", choices=["A", "B"], default="B")
    ap.add_argument("--mip", action="store_true", help="Bináris változók bekapcsolása (alap: LP).")
    ap.add_argument("--settlement-mode", choices=["individual", "community"], default="individual")
    ap.add_argument("--sharing-mode", choices=["proportional", "equal"], default="proportional")
    args = ap.parse_args(argv)

    summary = run(
        sim_yaml=args.sim,
        profiles_csv=args.profiles,
        out_dir=args.out,
        max_users=args.max_users,
        run_lp=not args.mip,
        dhw_profiles_csv=args.dhw_profiles,
        settlement_mode=args.settlement_mode,
        sharing_mode=args.sharing_mode,
        use_hss=args.use_hss,
        pv_ratio=args.pv_ratio,
        include_bess=args.bess,
        allow_grid_charge_for_min_soc=args.allow_grid_charge_for_min_soc,
        objective=args.objective,
        boiler_tariff=args.boiler_tariff,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
    else:
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
