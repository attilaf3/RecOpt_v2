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

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.append(str(HERE))

from optimize_disaggregated import (
    disaggregated_opt_bess_shared,
    save_disaggregated_opt_results,
)


def _keep_15min(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).ravel()
    if v.size != 35040:
        raise ValueError(f"A profil hossza {v.size}, de itt 35040 kell.")
    return v


def _energy_profile_kwh_step(v: np.ndarray) -> np.ndarray:
    return np.maximum(_keep_15min(v), 0.0)


def _power_profile_kw_from_energy(v: np.ndarray, dt: float) -> np.ndarray:
    return _energy_profile_kwh_step(v) / float(dt)


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
    max_users: int = 10,
    pv_ratio: float = 1.0,
    search_roots: Iterable[os.PathLike] | None = None,
    dt: float = 0.25,
) -> Tuple[
    np.ndarray,  # p_pv (35040, U) kW
    np.ndarray,  # p_ue (35040, U) kW
    np.ndarray,  # p_el_heater (35040, U) kW
    np.ndarray,  # size_elh (U,)
    np.ndarray,  # vol_hss_water (U,)
    np.ndarray,  # size_bess (U,)
    np.ndarray,  # eta_bess_in_u (U,)
    np.ndarray,  # eta_bess_out_u (U,)
    np.ndarray,  # eta_bess_stor_u (U,)
    np.ndarray,  # soc_bess_min_u (U,)
    np.ndarray,  # soc_bess_max_u (U,)
    np.ndarray,  # t_bess_min_u (U,)
    list[str],
]:
    sim_yaml_path = Path(sim_yaml_path)
    profiles_csv_path = Path(profiles_csv_path)

    if search_roots is None:
        search_roots = [sim_yaml_path.parent / "Users_v2", sim_yaml_path.parent]

    sim = yaml.safe_load(sim_yaml_path.read_text(encoding="utf-8")) or {}
    users_list = list(sim.get("users_list", []))[: int(max_users)]

    EXCLUDE = {"battery", "bess", "community"}
    users_list = [u for u in users_list if str(u).strip().lower() not in EXCLUDE]
    if not users_list:
        raise RuntimeError("A simulation YAML nem tartalmaz users_list-et vagy max_users=0.")

    df = pd.read_csv(profiles_csv_path, index_col=0)
    df.columns = [str(c) for c in df.columns]
    df = df[[c for c in df.columns if c.lower() not in EXCLUDE]]

    p_pv_cols: list[np.ndarray] = []
    p_ue_cols: list[np.ndarray] = []
    p_el_heater_cols: list[np.ndarray] = []

    size_elh: list[float] = []
    vol_hss_water: list[float] = []
    size_bess: list[float] = []
    eta_bess_in_u: list[float] = []
    eta_bess_out_u: list[float] = []
    eta_bess_stor_u: list[float] = []
    soc_bess_min_u: list[float] = []
    soc_bess_max_u: list[float] = []
    t_bess_min_u: list[float] = []
    user_names: list[str] = []

    for user_key in users_list:
        ypath = _find_user_yaml(search_roots, user_key)
        if not ypath:
            print(f"[WARN] YAML nem található: {user_key} — kihagyom.")
            continue

        u = yaml.safe_load(ypath.read_text(encoding="utf-8")) or {}
        units = u.get("units") or {}
        name = (units.get("name") or {}).get("name", str(user_key))
        user_names.append(name)

        ue = units.get("ue") or {}
        ue_prof = str(ue.get("profile")) if ue.get("profile") is not None else None
        if ue_prof and ue_prof in df.columns:
            p_ue_cols.append(_power_profile_kw_from_energy(df[ue_prof].to_numpy(), dt))
        else:
            p_ue_cols.append(np.zeros(35040, dtype=float))

        pv = units.get("pv") or {}
        pv_prof = str(pv.get("profile")) if pv.get("profile") is not None else None
        if pv_prof and pv_prof in df.columns:
            p_pv_cols.append(_power_profile_kw_from_energy(df[pv_prof].to_numpy(), dt) * float(pv_ratio))
        else:
            p_pv_cols.append(np.zeros(35040, dtype=float))

        bess = units.get("bess") or {}
        size_bess.append(float(bess.get("bess_size", 0.0)))
        eta_bess_in_u.append(float(bess.get("eta_bess_in", 0.98)))
        eta_bess_out_u.append(float(bess.get("eta_bess_out", 0.96)))
        eta_bess_stor_u.append(float(bess.get("eta_bess_stor", 0.995)))
        soc_bess_min_u.append(float(bess.get("soc_bess_min", 0.10)))
        soc_bess_max_u.append(float(bess.get("soc_bess_max", 0.90)))
        t_bess_min_u.append(float(bess.get("t_bess_min", 2.0)))

        hss = units.get("hss") or {}
        heater = units.get("ut") or {}
        p_el_heater_prof = str(heater.get("profile")) if heater.get("profile") is not None else None
        if p_el_heater_prof and p_el_heater_prof in df.columns:
            p_el_heater_cols.append(_power_profile_kw_from_energy(df[p_el_heater_prof].to_numpy(), dt))
        else:
            p_el_heater_cols.append(np.zeros(35040, dtype=float))

        size_elh.append(float(hss.get("size_elh", 0.0)))
        vol_hss_water.append(float(hss.get("vol_hss_water", 0.0)))

    if not user_names:
        raise RuntimeError("Nincs érvényes felhasználó.")

    return (
        np.column_stack(p_pv_cols).astype(float),
        np.column_stack(p_ue_cols).astype(float),
        np.column_stack(p_el_heater_cols).astype(float),
        np.asarray(size_elh, dtype=float),
        np.asarray(vol_hss_water, dtype=float),
        np.asarray(size_bess, dtype=float),
        np.asarray(eta_bess_in_u, dtype=float),
        np.asarray(eta_bess_out_u, dtype=float),
        np.asarray(eta_bess_stor_u, dtype=float),
        np.asarray(soc_bess_min_u, dtype=float),
        np.asarray(soc_bess_max_u, dtype=float),
        np.asarray(t_bess_min_u, dtype=float),
        user_names,
    )


def _select_bess_users(
    p_pv: np.ndarray,
    size_bess: np.ndarray,
    bess_share_pct: float,
    include_bess: bool = True,
    dt: float = 0.25,
) -> np.ndarray:
    """Deterministikus BESS-kiosztás: csak PV-s háztartások közül választ."""
    U = p_pv.shape[1]
    enabled = np.zeros(U, dtype=bool)
    if not include_bess:
        return enabled

    has_pv_arr = p_pv.sum(axis=0) * dt > 1e-9
    candidates = np.where(has_pv_arr)[0]
    candidates = np.array([u for u in candidates if float(size_bess[u]) > 1e-9], dtype=int)

    n = int(round(len(candidates) * float(bess_share_pct) / 100.0))
    n = max(0, min(n, len(candidates)))
    enabled[candidates[:n]] = True
    return enabled


def run(
    sim_yaml: os.PathLike,
    profiles_csv: os.PathLike,
    out_dir: os.PathLike,
    max_users: int = 10,
    run_lp: bool = False,
    pv_ratio: float = 1.0,
    bess_share_pct: float = 100.0,
    include_bess: bool = True,
    boiler_tariff: str = "B",
    objective: str = "bill",
    sharing_mode: str = "proportional",
    pairing_mode: str | None = None,
    bess_min_mode_steps: int = 4,
    gap_rel: float | None = 0.005,
    time_limit: float | None = None,
    msg: bool = False,
    save_user_timeseries: bool = True,
) -> dict:
    dt = 0.25
    boiler_tariff = str(boiler_tariff).upper().strip()
    if boiler_tariff not in {"A", "B"}:
        raise ValueError(f"boiler_tariff csak 'A' vagy 'B' lehet, nem: {boiler_tariff}")

    (
        p_pv,
        p_ue,
        p_el_heater,
        _size_elh,
        _vol_hss_water,
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
        max_users=max_users,
        pv_ratio=pv_ratio,
        dt=dt,
    )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    bess_enabled_arr = _select_bess_users(
        p_pv=p_pv,
        size_bess=size_bess,
        bess_share_pct=bess_share_pct,
        include_bess=include_bess,
        dt=dt,
    )

    has_pv_arr = p_pv.sum(axis=0) * dt > 1e-9
    print(f"[INFO] Háztartások száma: {len(user_names)}")
    print(f"[INFO] PV-s háztartások száma: {int(has_pv_arr.sum())}")
    print(f"[INFO] BESS arány: {bess_share_pct}%")
    print(f"[INFO] BESS-t kapó háztartások száma: {int(bess_enabled_arr.sum())}")
    print(f"[INFO] Objective: {objective}")
    print(f"[INFO] Boiler tariff: {boiler_tariff}")
    print(f"[INFO] Sharing mode: {sharing_mode}")

    result = disaggregated_opt_bess_shared(
        p_pv=p_pv,
        p_ue=p_ue,
        p_el_heater=p_el_heater,
        dt=dt,
        user_names=user_names,
        bess_enabled=bess_enabled_arr,
        size_bess=size_bess,
        eta_bess_in=eta_bess_in_u,
        eta_bess_out=eta_bess_out_u,
        eta_bess_stor=eta_bess_stor_u,
        soc_bess_min=soc_bess_min_u,
        soc_bess_max=soc_bess_max_u,
        soc_bess_init=0.5,
        t_bess_min=t_bess_min_u,
        boiler_tariff=boiler_tariff,
        objective=objective,  # bill vagy grid
        run_lp=run_lp,
        msg=msg,
        gapRel=gap_rel,
        timeLimit=time_limit,
        bess_min_mode_steps=bess_min_mode_steps,
        sharing_mode=sharing_mode,
        pairing_mode=pairing_mode,
    )

    result["summary"]["out_dir"] = str(out)
    result["summary"]["bess_share_pct"] = float(bess_share_pct)
    result["summary"]["pv_ratio"] = float(pv_ratio)
    result["summary"]["include_bess"] = bool(include_bess)

    save_disaggregated_opt_results(result, out, save_user_timeseries=save_user_timeseries)
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))
    return result["summary"]


def run_case(
    case_name: str,
    sim_yaml: os.PathLike,
    profiles_csv: os.PathLike,
    dhw_profiles_csv: os.PathLike | None = None,
    out_dir: os.PathLike = "results_disaggregated_opt_bess_shared",
    max_users: int = 105,
    *,
    include_bess: bool = True,
    bess_share_pct: float = 100.0,
    boiler_tariff: str = "B",
    sharing_mode: str = "proportional",
    objective: str = "bill",
    run_lp: bool = False,
    pv_ratio: float = 1.0,
    bess_min_mode_steps: int = 4,
    gap_rel: float | None = 0.005,
    time_limit: float | None = None,
    msg: bool = False,
    save_user_timeseries: bool = True,
) -> dict:
    """Egyszerű hívó wrapper. A dhw_profiles_csv itt csak kompatibilitási paraméter;
    az optimalizált kód a bojlerprofilt a profiles_csv-ből olvassa."""
    summary = run(
        sim_yaml=sim_yaml,
        profiles_csv=profiles_csv,
        out_dir=out_dir,
        max_users=max_users,
        run_lp=run_lp,
        pv_ratio=pv_ratio,
        bess_share_pct=bess_share_pct,
        include_bess=include_bess,
        boiler_tariff=boiler_tariff,
        objective=objective,
        sharing_mode=sharing_mode,
        pairing_mode=None,
        bess_min_mode_steps=bess_min_mode_steps,
        gap_rel=gap_rel,
        time_limit=time_limit,
        msg=msg,
        save_user_timeseries=save_user_timeseries,
    )
    summary["case_name"] = case_name
    return summary


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Disaggregált optimalizált BESS + energiaközösségi megosztás.")
    ap.add_argument("--sim", required=True, help="Path to simulation_config YAML.")
    ap.add_argument("--profiles", required=True, help="Path to disaggregated profiles CSV.")
    ap.add_argument("--out", default="results_disaggregated_opt_bess_shared", help="Output directory.")
    ap.add_argument("--max-users", type=int, default=10)
    ap.add_argument("--pv-ratio", type=float, default=1.0)
    ap.add_argument("--bess-share-pct", type=float, default=100.0)
    ap.add_argument("--no-bess", action="store_true", help="BESS kikapcsolása.")
    ap.add_argument("--lp", action="store_true", help="LP relaxáció, binárisok nélkül.")
    ap.add_argument("--boiler-tariff", choices=["A", "B"], default="B")
    ap.add_argument("--objective", choices=["bill", "grid"], default="bill")
    ap.add_argument("--sharing-mode", choices=["proportional", "equal"], default="proportional", help="Fizikai megosztási mód a vevői oldalon: proportional vagy equal.")
    ap.add_argument("--pairing-mode", choices=["proportional", "equal"], default=None, help="Eladó-vevő pénzügyi párosítás. Ha nincs megadva, megegyezik a sharing-mode-dal.")
    ap.add_argument("--bess-min-mode-steps", type=int, default=4, help="Minimum BESS üzemmódhossz 15 perces lépésekben. 4 = 1 óra.")
    ap.add_argument("--gap-rel", type=float, default=0.005)
    ap.add_argument("--time-limit", type=float, default=None)
    ap.add_argument("--msg", action="store_true")
    ap.add_argument("--no-user-timeseries", action="store_true", help="Ne írjon külön timeseries_*.csv fájlokat háztartásonként.")
    args = ap.parse_args(argv)

    run(
        sim_yaml=args.sim,
        profiles_csv=args.profiles,
        out_dir=args.out,
        max_users=args.max_users,
        run_lp=args.lp,
        pv_ratio=args.pv_ratio,
        bess_share_pct=args.bess_share_pct,
        include_bess=not args.no_bess,
        boiler_tariff=args.boiler_tariff,
        objective=args.objective,
        sharing_mode=args.sharing_mode,
        pairing_mode=args.pairing_mode,
        bess_min_mode_steps=args.bess_min_mode_steps,
        gap_rel=args.gap_rel,
        time_limit=args.time_limit,
        msg=args.msg,
        save_user_timeseries=not args.no_user_timeseries,
    )


SIM_YAML = "../Input/simulation_config_disaggregated_with_userlist.yaml"
PROFILES_CSV = "../Input/measurements_disaggregated_v2.csv"
DHW_PROFILES_CSV = "../Input/dhw_v2.csv"

MAX_USERS = 105

INCLUDE_BESS = True
BESS_SHARE_PCT = 40.0

# "A": bojler ugyanazon a körön van, saját PV/BESS is kiszolgálhatja.
# "B": bojler külön körön van, saját PV/BESS és közösségi megosztás sem szolgálhatja ki;
#      kizárólag B tarifás hálózati import látja el.
BOILER_TARIFF: str = "B"

# "proportional": fogyasztásarányosan osztja a megosztott energiát.
# "equal": egyenlő kvótát próbál adni minden aktív hiányos vevőnek.
SHARING_MODE: str = "equal"

OBJECTIVE = "bill"
RUN_LP = False
PV_RATIO = 1.0
BESS_MIN_MODE_STEPS = 4
GAP_REL = 0.005
TIME_LIMIT = None
MSG = False
SAVE_USER_TIMESERIES = True

CASE_NAME = f"opt_community_{BOILER_TARIFF}_{SHARING_MODE}"
OUT_DIR = f"results_opt_community_{BOILER_TARIFF}_{SHARING_MODE}"

if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
    else:
        summary = run_case(
            case_name=CASE_NAME,
            sim_yaml=SIM_YAML,
            profiles_csv=PROFILES_CSV,
            dhw_profiles_csv=DHW_PROFILES_CSV,
            out_dir=OUT_DIR,
            max_users=MAX_USERS,
            include_bess=INCLUDE_BESS,
            bess_share_pct=BESS_SHARE_PCT,
            boiler_tariff=BOILER_TARIFF,
            sharing_mode=SHARING_MODE,
            objective=OBJECTIVE,
            run_lp=RUN_LP,
            pv_ratio=PV_RATIO,
            bess_min_mode_steps=BESS_MIN_MODE_STEPS,
            gap_rel=GAP_REL,
            time_limit=TIME_LIMIT,
            msg=MSG,
            save_user_timeseries=SAVE_USER_TIMESERIES,
        )
