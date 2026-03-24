from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.append(str(HERE))

from optimize_aggregated_ecoobj import optimize_aggregated


def _keep_15min(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).ravel()
    if v.size != 35040:
        raise ValueError(f"A profil hossza {v.size}, de itt 35040 (15 perces éves) hossz kell.")
    return v


def _norm_to_annual_energy(profile: np.ndarray, annual_kwh: float | None, dt: float = 0.25) -> np.ndarray:
    p = np.maximum(np.asarray(profile, dtype=float), 0.0)
    if annual_kwh is None:
        return np.zeros_like(p)

    annual_energy_from_profile = p.sum() * dt
    if annual_energy_from_profile <= 0:
        return np.zeros_like(p)

    return p / annual_energy_from_profile * float(annual_kwh)


def _find_user_yaml(roots: Iterable[os.PathLike], name: str) -> Optional[Path]:
    for r in roots:
        for cand in (name, f"{name}.yaml"):
            p = Path(r) / cand
            if p.exists():
                return p
    return None


def _dhw_to_thermal_power_15min(liter_per_step: np.ndarray, t_in: float, t_out: float) -> np.ndarray:
    rho_water_kg_per_l = 1.0
    cp_water_j_per_kgk = 4186.0
    j_per_kwh = 3_600_000.0
    kwh_per_l_per_k = rho_water_kg_per_l * cp_water_j_per_kgk / j_per_kwh

    dT = max(0.0, float(t_out) - float(t_in))
    e_kwh_per_step = np.asarray(liter_per_step, dtype=float) * kwh_per_l_per_k * dT
    return e_kwh_per_step / 0.25


def build_aggregated_inputs(
    sim_yaml_path: os.PathLike,
    profiles_csv_path: os.PathLike,
    dhw_profile_path: os.PathLike | None = None,
    max_users: int = 105,
    pv_ratio: float = 1.0,
    use_hss: bool = True,
    search_roots: Iterable[os.PathLike] | None = None,
):
    sim_yaml_path = Path(sim_yaml_path)
    profiles_csv_path = Path(profiles_csv_path)

    if search_roots is None:
        search_roots = [sim_yaml_path.parent / "Users", sim_yaml_path.parent]

    sim = yaml.safe_load(sim_yaml_path.read_text(encoding="utf-8")) or {}
    users_list = list(sim.get("users_list", []))[: int(max_users)]

    exclude = {"battery", "bess", "community"}
    users_list = [u for u in users_list if str(u).strip().lower() not in exclude]

    if not users_list:
        raise RuntimeError("A simulation YAML-ben nincs feldolgozható users_list.")

    df = pd.read_csv(profiles_csv_path, index_col=0)
    df.columns = [str(c) for c in df.columns]

    dhw = None
    if dhw_profile_path is not None:
        dhw = pd.read_csv(dhw_profile_path, index_col=0)
        dhw.columns = [str(c) for c in dhw.columns]

    p_pv_sum = None
    p_ue_sum = None
    p_dhw_sum = None
    p_el_heater_sum = None

    size_elh_sum = 0.0
    vol_hss_water_sum = 0.0
    size_bess_sum = 0.0

    valid_users = []

    hss_params = {
        "T_env": [],
        "T_max": [],
        "T_min": [],
        "T_in": [],
        "a_hss": [],
        "eta_elh": [],
    }

    bess_params = {
        "eta_bess_in": [],
        "eta_bess_out": [],
        "eta_bess_stor": [],
        "soc_bess_min": [],
        "soc_bess_max": [],
        "t_bess_min": [],
    }

    for user_key in users_list:
        ypath = _find_user_yaml(search_roots, user_key)
        if not ypath:
            print(f"[WARN] YAML nem található: {user_key}")
            continue

        u = yaml.safe_load(ypath.read_text(encoding="utf-8")) or {}
        units = u.get("units") or {}

        valid_users.append(str(user_key))

        ue = units.get("ue") or {}
        pv = units.get("pv") or {}
        ut = units.get("ut") or {}
        hss = units.get("hss") or {}
        bess = units.get("bess") or {}

        # UE
        ue_prof = str(ue.get("profile")) if ue.get("profile") is not None else None
        ue_size = float(ue.get("size")) if ue.get("size") is not None else None
        if ue_prof and ue_prof in df.columns and ue_size is not None:
            base = _keep_15min(df[ue_prof].to_numpy())
            p_ue_u = _norm_to_annual_energy(base, ue_size, dt=0.25)
        else:
            p_ue_u = np.zeros(35040, dtype=float)

        # PV
        pv_prof = str(pv.get("profile")) if pv.get("profile") is not None else None
        pv_size = float(pv.get("size")) if pv.get("size") is not None else None
        if pv_prof and pv_prof in df.columns and pv_size is not None:
            base = _keep_15min(df[pv_prof].to_numpy())
            p_pv_u = _norm_to_annual_energy(base, pv_size * float(pv_ratio), dt=0.25)
        else:
            p_pv_u = np.zeros(35040, dtype=float)

        # Boiler / HSS
        if use_hss:
            if dhw is not None and hss.get("profile") is not None and str(hss["profile"]) in dhw.columns:
                liters = _keep_15min(dhw[str(hss["profile"])].to_numpy())
                t_in = float(hss.get("T_in", 10.0))
                t_out = float(hss.get("T_out", 40.0))
                p_dhw_u = _dhw_to_thermal_power_15min(liters, t_in=t_in, t_out=t_out)
            else:
                p_dhw_u = np.zeros(35040, dtype=float)

            p_el_heater_u = np.zeros(35040, dtype=float)
            size_elh_sum += float(hss.get("size_elh", 0.0))
            vol_hss_water_sum += float(hss.get("vol_hss_water", 0.0))

            hss_params["T_env"].append(float(hss.get("T_env", 20.0)))
            hss_params["T_max"].append(float(hss.get("T_max", 65.0)))
            hss_params["T_min"].append(float(hss.get("T_min", 10.0)))
            hss_params["T_in"].append(float(hss.get("T_in", 10.0)))
            hss_params["a_hss"].append(float(hss.get("a_hss", 0.00125)))
            hss_params["eta_elh"].append(float(hss.get("eta_elh", 0.95)))
        else:
            ut_prof = str(ut.get("profile")) if ut.get("profile") is not None else None
            ut_size = float(ut.get("size")) if ut.get("size") is not None else None
            if ut_prof and ut_prof in df.columns and ut_size is not None:
                base = _keep_15min(df[ut_prof].to_numpy())
                p_el_heater_u = _norm_to_annual_energy(base, ut_size, dt=0.25)
            else:
                p_el_heater_u = np.zeros(35040, dtype=float)

            p_dhw_u = np.zeros(35040, dtype=float)

        # BESS a user yaml-ből
        bess_size_u = float(bess.get("bess_size", 0.0))
        size_bess_sum += bess_size_u

        if bess_size_u > 0:
            bess_params["eta_bess_in"].append(float(bess.get("eta_bess_in", 0.98)))
            bess_params["eta_bess_out"].append(float(bess.get("eta_bess_out", 0.96)))
            bess_params["eta_bess_stor"].append(float(bess.get("eta_bess_stor", 0.995)))
            bess_params["soc_bess_min"].append(float(bess.get("soc_bess_min", 0.2)))
            bess_params["soc_bess_max"].append(float(bess.get("soc_bess_max", 1.0)))
            bess_params["t_bess_min"].append(float(bess.get("t_bess_min", 2.0)))

        # összeadás
        if p_pv_sum is None:
            p_pv_sum = p_pv_u.copy()
            p_ue_sum = p_ue_u.copy()
            p_dhw_sum = p_dhw_u.copy()
            p_el_heater_sum = p_el_heater_u.copy()
        else:
            p_pv_sum += p_pv_u
            p_ue_sum += p_ue_u
            p_dhw_sum += p_dhw_u
            p_el_heater_sum += p_el_heater_u

    if not valid_users:
        raise RuntimeError("Nem találtam érvényes user YAML-t.")

    n_household = len(valid_users)

    # aggregált HSS paraméterek: egyszerűen tipikus közös paraméter
    agg_hss = dict(
        T_env=float(np.mean(hss_params["T_env"])) if hss_params["T_env"] else 20.0,
        T_max=float(np.mean(hss_params["T_max"])) if hss_params["T_max"] else 65.0,
        T_min=float(np.mean(hss_params["T_min"])) if hss_params["T_min"] else 10.0,
        T_in=float(np.mean(hss_params["T_in"])) if hss_params["T_in"] else 10.0,
        a_hss=float(np.mean(hss_params["a_hss"])) if hss_params["a_hss"] else 0.00125,
        eta_elh=float(np.mean(hss_params["eta_elh"])) if hss_params["eta_elh"] else 0.95,
    )

    agg_bess = dict(
        eta_bess_in=float(np.mean(bess_params["eta_bess_in"])) if bess_params["eta_bess_in"] else 0.98,
        eta_bess_out=float(np.mean(bess_params["eta_bess_out"])) if bess_params["eta_bess_out"] else 0.96,
        eta_bess_stor=float(np.mean(bess_params["eta_bess_stor"])) if bess_params["eta_bess_stor"] else 0.995,
        soc_bess_min=float(np.mean(bess_params["soc_bess_min"])) if bess_params["soc_bess_min"] else 0.2,
        soc_bess_max=float(np.mean(bess_params["soc_bess_max"])) if bess_params["soc_bess_max"] else 1.0,
        t_bess_min=float(np.mean(bess_params["t_bess_min"])) if bess_params["t_bess_min"] else 2.0,
    )

    return {
        "p_pv": p_pv_sum,
        "p_ue": p_ue_sum,
        "p_dhw": p_dhw_sum,
        "p_el_heater": p_el_heater_sum,
        "size_elh": float(size_elh_sum),
        "vol_hss_water": float(vol_hss_water_sum),
        "size_bess": float(size_bess_sum),
        "n_household": int(n_household),
        "valid_users": valid_users,
        "agg_hss": agg_hss,
        "agg_bess": agg_bess,
    }


def run_aggregated(
    sim_yaml: str | Path,
    profiles_csv: str | Path,
    dhw_profiles_csv: str | Path | None,
    out_dir: str | Path,
    max_users: int = 105,
    run_lp: bool = True,
    pv_ratio: float = 1.0,
    use_hss: bool = True,
):
    data = build_aggregated_inputs(
        sim_yaml_path=sim_yaml,
        profiles_csv_path=profiles_csv,
        dhw_profile_path=dhw_profiles_csv,
        max_users=max_users,
        pv_ratio=pv_ratio,
        use_hss=use_hss,
    )

    print(f"[AGG] Háztartások száma: {data['n_household']}")
    print(f"[AGG] Összes BESS kapacitás [kWh]: {data['size_bess']:.3f}")
    print(f"[AGG] Összes ELH teljesítmény [kW]: {data['size_elh']:.3f}")
    print(f"[AGG] Összes HSS térfogat [l]: {data['vol_hss_water']:.3f}")

    boiler_mode = "thermal_optimized" if use_hss else "electric_load"

    results, status, objective_detail, n_vars, n_cons, infeas_gap, sum_batt_to_grid = optimize_aggregated(
        p_pv=data["p_pv"],
        p_ue=data["p_ue"],
        p_el_heater=data["p_el_heater"],
        p_dhw=data["p_dhw"],
        dt=0.25,
        size_elh=data["size_elh"],
        size_bess=data["size_bess"],
        vol_hss_water=data["vol_hss_water"],
        boiler_mode=boiler_mode,
        n_household=data["n_household"],
        annual_cheap_limit_kwh=2523.0 * data["n_household"],
        eta_bess_in=data["agg_bess"]["eta_bess_in"],
        eta_bess_out=data["agg_bess"]["eta_bess_out"],
        eta_bess_stor=data["agg_bess"]["eta_bess_stor"],
        soc_bess_min=data["agg_bess"]["soc_bess_min"],
        soc_bess_max=data["agg_bess"]["soc_bess_max"],
        t_bess_min=data["agg_bess"]["t_bess_min"],
        run_lp=run_lp,
        msg=False,
        gapRel=0.01,
        timeLimit=None,
        **data["agg_hss"],
    )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({
        "p_pv": results["p_pv"],
        "p_ue": results["p_ue"],
        "p_total_load": results["p_total_load"],
        "p_grid_in": results["p_grid_in"],
        "p_grid_out": results["p_grid_out"],
        "p_grid_bess": results["p_grid_bess"],
        "p_bess_in": results["p_bess_in"],
        "p_bess_out": results["p_bess_out"],
        "e_bess_stor": results["e_bess_stor"],
        "p_elh_in": results["p_elh_in"],
        "p_elh_out": results["p_elh_out"],
        "p_hss_in": results["p_hss_in"],
        "p_hss_out": results["p_hss_out"],
        "e_hss_stor": results["e_hss_stor"],
        "t_hss": results["t_hss"],
        "d_bess": results["d_bess"],
        "d_grid": results["d_grid"],
    }).to_csv(out / "aggregated_timeseries.csv", index=False)

    summary = {
        "status": int(status),
        "objective_detail": objective_detail,
        "n_vars": int(n_vars),
        "n_cons": int(n_cons),
        "infeas_gap": float(infeas_gap) if infeas_gap is not None else None,
        "sum_batt_to_grid": float(sum_batt_to_grid),
        "n_household": int(data["n_household"]),
        "valid_users": data["valid_users"],
        "size_bess_total_kwh": float(data["size_bess"]),
        "size_elh_total_kw": float(data["size_elh"]),
        "vol_hss_water_total_l": float(data["vol_hss_water"]),
        "T_steps": int(len(data["p_pv"])),
        "dt_hours": 0.25,
        "mode": "AGGREGATED",
        "boiler_mode": boiler_mode,
        "out_dir": str(out.resolve()),
    }

    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


def main():
    ap = argparse.ArgumentParser(description="Aggregált 15 perces futtató")
    ap.add_argument("--sim", required=True, help="Szimulációs YAML")
    ap.add_argument("--profiles", required=True, help="15 perces profil CSV")
    ap.add_argument("--dhw_profiles", default=None, help="15 perces DHW CSV")
    ap.add_argument("--out", default="results_agg", help="Kimeneti mappa")
    ap.add_argument("--max-users", type=int, default=105)
    ap.add_argument("--pv-ratio", type=float, default=1.0)
    ap.add_argument("--use-hss", action="store_true")
    ap.add_argument("--no-use-hss", dest="use_hss", action="store_false")
    ap.set_defaults(use_hss=True)
    ap.add_argument("--mip", action="store_true", help="Bináris változók engedélyezése")

    args = ap.parse_args()

    summary = run_aggregated(
        sim_yaml=args.sim,
        profiles_csv=args.profiles,
        dhw_profiles_csv=args.dhw_profiles,
        out_dir=args.out,
        max_users=args.max_users,
        run_lp=not args.mip,
        pv_ratio=args.pv_ratio,
        use_hss=args.use_hss,
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
