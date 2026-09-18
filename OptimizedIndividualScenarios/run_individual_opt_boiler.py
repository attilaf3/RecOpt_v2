from __future__ import annotations

import argparse
import json
import os
from os.path import join
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from InputReading import read_simulation_inputs
from OptimizedIndividualScenarios.individual_opt_boiler import individual_opt_boiler
from Utility.configuration import config


# --- input builder -------------------------------------------------------------
def build_inputs(
        sim_yaml_path: os.PathLike,
        profiles_csv_path: os.PathLike,
        dhw_profile_path: os.PathLike,
        max_users: int | None = 10,
        pv_ratio: float = 1.0,
        use_hss: bool = True,
        search_roots: Iterable[os.PathLike] | None = None,
) -> tuple:
    """Compatibility adapter backed exclusively by :mod:`InputReading`."""
    inputs = read_simulation_inputs(
        sim_yaml_path=sim_yaml_path,
        profiles_csv_path=profiles_csv_path,
        dhw_profile_path=dhw_profile_path,
        max_users=max_users,
        pv_ratio=pv_ratio,
        search_roots=search_roots,
        dt=config.getfloat("simulation", "dt_hours"),
    )
    p_dhw = inputs.p_dhw_kw if use_hss else np.zeros_like(inputs.p_dhw_kw)
    p_el_heater = np.zeros_like(inputs.p_el_heater_kw) if use_hss else inputs.p_el_heater_kw
    return (
        inputs.p_pv_kw, inputs.p_ue_kw, p_dhw, p_el_heater,
        inputs.size_elh, inputs.vol_hss_water, inputs.size_bess,
        inputs.eta_bess_in, inputs.eta_bess_out, inputs.eta_bess_stor,
        inputs.soc_bess_min, inputs.soc_bess_max, inputs.t_bess_min,
        inputs.T_env, inputs.T_max, inputs.T_min, inputs.T_in,
        inputs.a_hss, inputs.eta_elh, inputs.t_hss_min_in, inputs.user_names,
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
        use_hss: bool = True,
) -> dict:
    from OptimizedScenarios.run_individual_optimization import run as run_common
    return run_common(sim_yaml=sim_yaml, profiles_csv=profiles_csv,
        dhw_profiles_csv=dhw_profiles_csv, out_dir=out_dir, max_users=max_users,
        run_lp=run_lp, pv_ratio=pv_ratio, use_hss=use_hss,
        include_boiler=use_hss, include_bess=False)


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
    ap.add_argument("--no-use-hss", dest="use_hss", action="store_false")
    ap.set_defaults(use_hss=True)
    ap.add_argument("--mip", action="store_true", help="Bináris változók bekapcsolása (alap: LP).")
    args = ap.parse_args(argv)

    summary = run(
        sim_yaml=args.sim,
        profiles_csv=args.profiles,
        out_dir=args.out,
        max_users=args.max_users,
        run_lp=not args.mip,
        pv_ratio=args.pv_ratio,
        use_hss=args.use_hss,
        dhw_profiles_csv=args.dhw_profiles,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    summary = run(
        sim_yaml=config.getpath("paths", "simulation_yaml_base"),
        profiles_csv=config.getpath("paths", "profiles_csv"),
        dhw_profiles_csv=config.getpath("paths", "dhw_profiles_csv"),
        out_dir=config.getpath("paths", "individual_boiler_output"),
        max_users=105,
        run_lp=False,
        pv_ratio=1.0,
        use_hss=True,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
