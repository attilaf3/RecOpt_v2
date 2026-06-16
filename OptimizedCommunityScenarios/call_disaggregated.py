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

from InputReading.user_input_reading import read_users

DT = 1.0

# --- import optimizer locally ---------------------------------------------------
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.append(str(HERE))
from optimize_disaggregated import \
    optimize_multi_users_economic  # expects (T,U) arrays, sizes, etc. :contentReference[oaicite:1]{index=1}


# --- helpers -------------------------------------------------------------------
def _keep_15min(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).ravel()
    if v.size != 35040:
        raise ValueError(f"A profil hossza {v.size}, de itt 35040 kell.")
    return v


def _aggregate_to_hourly(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).ravel()
    if v.size == 8760:
        return v
    if v.size != 35040:
        raise ValueError(f"A profil hossza {v.size}, de itt 35040 vagy 8760 kell.")
    return v.reshape(8760, 4).sum(axis=1)


def _norm_to_annual(profile: np.ndarray, annual_kwh: float | None) -> np.ndarray:
    p = np.maximum(np.asarray(profile, float), 0.0)
    if annual_kwh is None:
        return np.zeros_like(p)
    s = p.sum()
    return np.zeros_like(p) if s <= 0 else p / s * float(annual_kwh)


def _find_user_yaml(roots: Iterable[os.PathLike], name: str) -> Optional[Path]:
    for r in roots:
        for cand in (name, f"{name}.yaml"):
            p = Path(r) / cand
            if p.exists():
                return p
    return None


# --- input builder -------------------------------------------------------------
def build_inputs(sim_yaml_path: os.PathLike, profiles_csv_path: os.PathLike, dhw_profile_path: os.PathLike,
        max_users: int = 10, pv_ratio: float = 1.0, use_hss: bool = True,
        search_roots: Iterable[os.PathLike] | None = None, ) -> Tuple[np.ndarray,  # p_pv (8760, U)
np.ndarray,  # p_ue (8760, U)
np.ndarray,  # p_dhw (8760, U)
np.ndarray,  # p_el_heater (8760, U)
np.ndarray,  # size_elh (U,)
np.ndarray,  # vol_hss_water (U,)
list[float],  # T_env_u
list[float],  # T_max_u
list[float],  # T_min_u
list[float],  # T_in_u
list[float],  # a_hss_u
list[float],  # eta_elh_u
list[str],  # user_names
]:
    """
    Return:
      p_pv, p_ue, p_ut: np.ndarray (8760, U)
      size_elh: np.ndarray (U,)
      vol_hss_water: np.ndarray (U,)
      user_names: list[str]
    """
    sim_yaml_path = Path(sim_yaml_path)
    profiles_csv_path = Path(profiles_csv_path)
    if search_roots is None:
        search_roots = [sim_yaml_path.parent / "Users", sim_yaml_path.parent]

    users_list = read_users(sim_yaml_path, max_users)

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
        units = (u.get("units") or {})
        name = (units.get("name") or {}).get("name", str(user_key))
        user_names.append(name)

        # --- UE (villamos fogyasztás) ---
        ue = units.get("ue") or {}
        ue_prof = str(ue.get("profile")) if ue.get("profile") is not None else None
        ue_size = float(ue.get("size")) if ue.get("size") is not None else None
        if ue_prof and ue_prof in df.columns and ue_size is not None:
            base = _aggregate_to_hourly(df[ue_prof].to_numpy())
            p_ue_cols.append(_norm_to_annual(base, ue_size))
        else:
            p_ue_cols.append(np.zeros(8760))

        # --- PV (termelés) ---
        n_pv = 7.0
        pv = units.get("pv") or {}
        pv_prof = str(pv.get("profile")) if pv.get("profile") is not None else None
        pv_size = float(pv.get("size")) if pv.get("size") is not None else None
        if pv_prof and pv_prof in df.columns and pv_size is not None:
            base = _aggregate_to_hourly(df[pv_prof].to_numpy())
            p_pv_cols.append(_norm_to_annual(base, pv_size / n_pv))
        else:
            p_pv_cols.append(np.zeros(8760))

        # BESS

        bess = units.get("bess") or {}
        size_bess.append(float(bess.get("bess_size", 0.0)))
        eta_bess_in_u.append(float(bess.get("eta_bess_in", 0.98)))
        eta_bess_out_u.append(float(bess.get("eta_bess_out", 0.96)))
        eta_bess_stor_u.append(float(bess.get("eta_bess_stor", 0.995)))
        soc_bess_min_u.append(float(bess.get("soc_bess_min", 0.2)))
        soc_bess_max_u.append(float(bess.get("soc_bess_max", 1.0)))
        t_bess_min_u.append(float(bess.get("t_bess_min", 2.0)))

        # --- HSS / UT (bojleres hőtároló, ELH szolgálja ki) ---
        hss = units.get("hss") or {}
        heater = units.get("ut") or {}

        if use_hss:
            # DHW profil: liter → kW (kWh/h), órára aggregálva
            if hss.get("profile") is not None and str(hss["profile"]) in dhw.columns:
                base_L_per_step = dhw[str(hss["profile"])].to_numpy()
                base_L_per_hour = _aggregate_to_hourly(base_L_per_step)

                RHO_WATER_KG_PER_L = 1.0
                CP_WATER_J_PER_KGK = 4186.0
                J_PER_KWH = 3_600_000.0
                KWH_PER_L_PER_K = RHO_WATER_KG_PER_L * CP_WATER_J_PER_KGK / J_PER_KWH

                T_in = float(hss.get("T_in", 10))
                T_out = float(hss.get("T_out", 55))
                dT = max(0.0, T_out - T_in)

                # órás energiaigény [kWh / óra-lépés]
                e_kwh_per_hour = base_L_per_hour * KWH_PER_L_PER_K * dT
                dhw_cols.append(e_kwh_per_hour.astype(float))
            else:
                dhw_cols.append(np.zeros(8760, dtype=float))
        else:
            # Ha nincs HSS, de van UT profil éves energiával (elektromos betét profil)
            p_el_heater_prof = str(heater.get("profile")) if heater.get("profile") is not None else None
            heater_number = float(heater.get("size")) if heater.get("size") is not None else None
            if p_el_heater_prof and p_el_heater_prof in df.columns and heater_number is not None:
                base = _aggregate_to_hourly(df[p_el_heater_prof].to_numpy())
                p_el_heater_cols.append(_norm_to_annual(base, heater_number))
            else:
                p_el_heater_cols.append(np.zeros(8760, dtype=float))

        # Ha nincs HSS, akkor nulla méretű bojlert feltételezünk
        size_elh.append(float(hss.get("size_elh", 0.0)))
        vol_hss_water.append(float(hss.get("vol_hss_water", 0.0)))
        # kiegészítő HSS paraméterek (nem kötelezőek)
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

    if use_hss:
        p_dhw = np.column_stack(dhw_cols).astype(float) if dhw_cols else np.zeros((8760, U), float)
        p_el_heater = np.zeros((8760, U), float)  # HSS módban nem használjuk az ELH profilt
    else:
        p_dhw = np.zeros((8760, U), float)
        p_el_heater = np.column_stack(p_el_heater_cols).astype(float) if p_el_heater_cols else np.zeros((8760, U),
                                                                                                        float)

    return (p_pv, p_ue, p_dhw, p_el_heater, np.asarray(size_elh, float), np.asarray(vol_hss_water, float),
            np.asarray(size_bess, float), np.asarray(eta_bess_in_u, float), np.asarray(eta_bess_out_u, float),
            np.asarray(eta_bess_stor_u, float), np.asarray(soc_bess_min_u, float), np.asarray(soc_bess_max_u, float),
            np.asarray(t_bess_min_u, float), T_env_u, T_max_u, T_min_u, T_in_u, a_hss_u, eta_elh_u, t_hss_min_in_u,
            user_names,)


# --- runner --------------------------------------------------------------------

def run(sim_yaml: os.PathLike, profiles_csv: os.PathLike, dhw_profiles_csv: os.PathLike, out_dir: os.PathLike,
        max_users: int = 10, run_lp: bool = True, pv_ratio: float = 1.0, use_hss: bool = True, ) -> dict:
    (p_pv, p_ue, p_dhw, p_el_heater, size_elh, vol_hss_water, size_bess, eta_bess_in_u, eta_bess_out_u, eta_bess_stor_u,
     soc_bess_min_u, soc_bess_max_u, t_bess_min_u, T_env_u, T_max_u, T_min_u, T_in_u, a_hss_u, eta_elh_u,
     t_hss_min_in_u, user_names) = build_inputs(sim_yaml_path=sim_yaml, profiles_csv_path=profiles_csv,
        dhw_profile_path=dhw_profiles_csv, max_users=max_users, pv_ratio=pv_ratio, use_hss=use_hss, )

    print(f"[INFO] Betöltött felhasználók száma: {len(user_names)} → "
          f"{', '.join(user_names[:10])}{'...' if len(user_names) > 10 else ''}")

    import numpy as np
    ue_active = int(((p_ue > 1e-9).sum(axis=0) > 0).sum())
    pv_active = int(((p_pv > 1e-9).sum(axis=0) > 0).sum())
    hss_active = int(((p_dhw > 1e-9).sum(axis=0) > 0).sum())
    elh_active = int(((p_el_heater > 1e-9).sum(axis=0) > 0).sum())
    print(
        f"[INFO] Aktív oszlopok — UE:{ue_active}, PV:{pv_active}, HSS:{hss_active}, ELH:{elh_active}, Use HSS:{use_hss} / {p_ue.shape[1]}")

    # --- hívjuk az optimalizálót ------------------------------------------------
    # optimize_multi_users pontos interface-e: p_pv/p_ue/p_ut alak (T,U), dt, size_elh, size_bess, vol_hss_water stb. :contentReference[oaicite:2]{index=2}
    results, status, objective, n_vars, n_cons, infeas_gap = optimize_multi_users_economic(p_pv=p_pv, p_ue=p_ue,
        p_dhw=p_dhw, p_el_heater=p_el_heater, dt=1.0, hss_flag=use_hss, size_elh=size_elh, vol_hss_water=vol_hss_water,
        size_bess=size_bess, eta_bess_in=eta_bess_in_u, eta_bess_out=eta_bess_out_u, eta_bess_stor=eta_bess_stor_u,
        soc_bess_min=soc_bess_min_u, soc_bess_max=soc_bess_max_u, t_bess_min=t_bess_min_u, T_env=T_env_u, T_max=T_max_u,
        T_min=T_min_u, T_in=T_in_u, a_hss=a_hss_u, eta_elh=eta_elh_u, run_lp=run_lp, msg=True, gapRel=0.01,
        timeLimit=None, t_hss_min_in=t_hss_min_in_u, )

    # --- kimenetek mentése ------------------------------------------------------
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Community (1D sorozatok)
    comm = pd.DataFrame(
        {"p_inj_comm": results["p_inj_comm"], "p_with_comm": results["p_with_comm"], "p_grid_in": results["p_grid_in"],
            "p_grid_out": results["p_grid_out"], "p_grid_bess_total": results["p_grid_bess_total"],
            "p_bess_in_total": results["p_bess_in_total"], "p_bess_out_total": results["p_bess_out_total"],
            "e_bess_total": results["e_bess_total"], "d_bess_any": results["d_bess_any"],
            "d_grid": results["d_grid"], })
    comm.to_csv(out / "community_timeseries.csv", index=False)

    # Per-user mátrixok (T×U) → CSV (oszlopok: user_names)
    def save_mat(filename: str, key: str):
        if key in results and isinstance(results[key], np.ndarray) and results[key].ndim == 2:
            pd.DataFrame(results[key], columns=user_names).to_csv(out / filename, index=False)

    save_mat("p_pv_load.csv", "p_pv_load")
    save_mat("p_pv_bess.csv", "p_pv_bess")
    save_mat("p_pv_elh.csv", "p_pv_elh")
    save_mat("p_pv_rec.csv", "p_pv_rec")
    save_mat("p_grid_load.csv", "p_grid_load")
    save_mat("p_grid_bess.csv", "p_grid_bess")
    save_mat("p_rec_load.csv", "p_rec_load")
    save_mat("p_rec_elh.csv", "p_rec_elh")
    save_mat("p_grid_elh.csv", "p_grid_elh")
    save_mat("p_bess_load.csv", "p_bess_load")
    save_mat("e_bess.csv", "e_bess")
    save_mat("p_hss_in.csv", "p_hss_in")
    save_mat("p_hss_out.csv", "p_hss_out")
    save_mat("t_hss.csv", "t_hss")
    save_mat("p_with_user.csv", "p_with_user")
    save_mat("p_inj_user.csv", "p_inj_user")

    pd.DataFrame([results["e_grid_low"]], columns=user_names).to_csv(out / "e_grid_low.csv", index=False)
    pd.DataFrame([results["e_grid_high"]], columns=user_names).to_csv(out / "e_grid_high.csv", index=False)

    # Itt vannak gazdasági számítások is
    pd.DataFrame({"user_name": user_names, "grid_cost_Ft": results["grid_cost_user"],
        "rec_buy_cost_Ft": results["rec_buy_cost_user"], "rec_sell_revenue_Ft": results["rec_sell_revenue_user"],
        "grid_export_revenue_Ft": results["grid_export_revenue_user"],
        "net_cost_Ft": results["net_cost_user"], }).to_csv(out / "user_bills.csv", index=False)

    summary = {"status": int(status), "objective": float(objective), "n_vars": int(n_vars), "n_cons": int(n_cons),
        "infeas_gap": float(infeas_gap) if isinstance(infeas_gap, (int, float, np.floating)) else None,
        "U_users": int(p_pv.shape[1]), "T_steps": int(p_pv.shape[0]), "user_names": user_names, "out_dir": str(out), }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


# --- CLI -----------------------------------------------------------------------
def main(argv: list[str] | None = None):
    # Ha nincs argumentum, kattintásos mód
    if argv is None and len(sys.argv) == 1:
        summary = run(sim_yaml="../Inputs/simulation_config_disaggregated_pv_original_increase_1.0.yaml",
            profiles_csv="../Inputs/measurements_disaggregated_pv_original_increase_1.0.csv",
            dhw_profiles_csv="../Inputs/dhw.csv", out_dir=r".\results_disaggregated_hourly", max_users=105,
            run_lp=False, pv_ratio=1.0, use_hss=True, )
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    ap = argparse.ArgumentParser(description="Optimize multi-user from YAMLs (subset).")
    ap.add_argument("--sim", required=True, help="Path to simulation_config YAML.")
    ap.add_argument("--profiles", required=True, help="Path to disaggregated profiles CSV.")
    ap.add_argument("--dhw_profiles", help="Domestic hot water profiles CSV.")
    ap.add_argument("--out", default="results_subset", help="Output directory.")
    ap.add_argument("--max-users", type=int, default=10, help="Első N user a users_list-ből.")
    ap.add_argument("--pv-ratio", type=float, default=1.0, help="PV éves energia szorzó.")
    ap.add_argument("--use-hss", action="store_true", help="HSS logika (ΔT·c_víz) használata.")
    ap.add_argument("--no-use-hss", dest="use_hss", action="store_false")
    ap.set_defaults(use_hss=True)
    ap.add_argument("--mip", action="store_true", help="Bináris változók bekapcsolása (alap: LP).")
    args = ap.parse_args(argv)

    summary = run(sim_yaml=args.sim, profiles_csv=args.profiles, out_dir=args.out, max_users=args.max_users,
        run_lp=not args.mip, pv_ratio=args.pv_ratio, use_hss=args.use_hss, dhw_profiles_csv=args.dhw_profiles, )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


# Notes: a 2b esethez fogjuk használni
if __name__ == "__main__":
    main()
