from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional, Dict, Any, List

import pandas as pd
import yaml


def _find_user_yaml(roots: Iterable[os.PathLike], name: str) -> Optional[Path]:
    for r in roots:
        for cand in (name, f"{name}.yaml"):
            p = Path(r) / cand
            if p.exists():
                return p
    return None


def _get(d: Dict[str, Any], path: List[str], default=None):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def run_yaml_inventory(
    sim_yaml: os.PathLike,
    out_dir: os.PathLike,
    max_users: Optional[int] = None,
):
    sim_yaml = Path(sim_yaml)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sim = yaml.safe_load(sim_yaml.read_text(encoding="utf-8")) or {}
    users_list = list(sim.get("users_list", []))
    if max_users is not None:
        users_list = users_list[: int(max_users)]

    # ugyanaz a kizárás, mint a futtatókban
    EXCLUDE = {"battery", "bess", "community"}
    users_list = [u for u in users_list if str(u).strip().lower() not in EXCLUDE]

    # users_roots = [sim_yaml.parent / "Users", sim_yaml.parent]
    users_roots = [sim_yaml.parent / "Users_v2", sim_yaml.parent]

    rows = []
    missing = 0

    for key in users_list:
        ypath = _find_user_yaml(users_roots, str(key))
        if not ypath:
            print(f"[WARN] YAML nem található: {key}")
            missing += 1
            continue

        u = yaml.safe_load(ypath.read_text(encoding="utf-8")) or {}
        units = u.get("units") or {}

        name = _get(units, ["name", "name"], default=str(key))

        ue_size = _get(units, ["ue", "size"], default=None)          # kWh/év
        ue_prof = _get(units, ["ue", "profile"], default=None)

        pv_size = _get(units, ["pv", "size"], default=None)          # kWh/év
        pv_prof = _get(units, ["pv", "profile"], default=None)

        hss = units.get("hss") or {}
        size_elh = hss.get("size_elh", 0.0)                          # kW
        vol_hss = hss.get("vol_hss_water", 0.0)                      # liter
        hss_prof = hss.get("profile", None)

        ut = units.get("ut") or {}
        ut_size = ut.get("size", None)                               # kWh/év
        ut_prof = ut.get("profile", None)

        has_pv = (pv_size is not None and float(pv_size) > 0) or (pv_prof is not None)
        has_hss = (float(size_elh or 0.0) > 0.0) and (float(vol_hss or 0.0) > 0.0)
        has_ut = (ut_size is not None and float(ut_size) > 0) or (ut_prof is not None)

        rows.append({
            "user_key": str(key),
            "name": str(name),

            "ue_size_kwh_per_year": float(ue_size) if ue_size is not None else None,
            "ue_profile": ue_prof,

            "has_pv": int(has_pv),
            "pv_size_kwh_per_year": float(pv_size) if pv_size is not None else None,
            "pv_profile": pv_prof,

            "has_hss": int(has_hss),
            "hss_size_elh_kw": float(size_elh) if size_elh is not None else 0.0,
            "hss_vol_liter": float(vol_hss) if vol_hss is not None else 0.0,
            "hss_profile": hss_prof,

            "has_ut_electric_heater": int(has_ut),
            "ut_size_kwh_per_year": float(ut_size) if ut_size is not None else None,
            "ut_profile": ut_prof,

            "yaml_path": str(ypath),
        })

    df = pd.DataFrame(rows)

    print("\n========== YAML INVENTORY ==========")
    print(f"Users in sim YAML: {len(users_list)}")
    print(f"Found YAMLs      : {len(df)}")
    print(f"Missing YAMLs    : {missing}")

    if len(df) > 0:
        print(f"PV households    : {int(df['has_pv'].sum())}")
        print(f"HSS households   : {int(df['has_hss'].sum())}")
        print(f"UT households    : {int(df['has_ut_electric_heater'].sum())}")
        if df["ue_size_kwh_per_year"].notna().any():
            print(f"Total UE (kWh/y) : {df['ue_size_kwh_per_year'].fillna(0).sum():.2f}")

    out_csv = out_dir / "yaml_inventory.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nCSV mentve: {out_csv}\n")

    return df


if __name__ == "__main__":
    SIM_YAML = "Input/simulation_config_disaggregated_with_userlist.yaml"
    OUT_DIR = "results_yaml_inventory"


    run_yaml_inventory(sim_yaml=SIM_YAML, out_dir=OUT_DIR, max_users=None)