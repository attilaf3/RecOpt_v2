"""Bemeneti YAML- és idősor-adatok beolvasása a nem optimalizált esetekhez."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml


DT = 0.25
N_STEPS = 35040
C_HSS = 0.00116667


@dataclass(frozen=True)
class NonoptInput:
    user_names: list[str]
    e_pv: np.ndarray
    e_base: np.ndarray
    e_boiler_measured: np.ndarray
    p_dhw: np.ndarray
    size_bess: np.ndarray
    eta_bess_in: np.ndarray
    eta_bess_out: np.ndarray
    eta_bess_stor: np.ndarray
    soc_bess_min: np.ndarray
    soc_bess_max: np.ndarray
    t_bess_min: np.ndarray
    size_elh: np.ndarray
    vol_hss_water: np.ndarray
    t_env: np.ndarray
    t_max: np.ndarray
    t_min: np.ndarray
    t_in: np.ndarray
    t_set: np.ndarray
    a_hss: np.ndarray
    eta_elh: np.ndarray


def _profile(frame: pd.DataFrame, name: object) -> np.ndarray:
    if name is None or str(name) not in frame.columns:
        return np.zeros(N_STEPS)
    values = np.maximum(frame[str(name)].to_numpy(dtype=float).ravel(), 0.0)
    if values.size != N_STEPS:
        raise ValueError(f"A(z) {name} profil hossza {values.size}, de {N_STEPS} kell.")
    return values


def _find_yaml(roots: Iterable[Path], user_key: str) -> Path | None:
    for root in roots:
        for name in (user_key, f"{user_key}.yaml"):
            candidate = root / name
            if candidate.exists():
                return candidate
    return None


def load_inputs(
    sim_yaml: str | Path,
    profiles_csv: str | Path,
    dhw_profiles_csv: str | Path,
    max_users: int = 105,
) -> NonoptInput:
    sim_path = Path(sim_yaml)
    profiles = pd.read_csv(profiles_csv, index_col=0)
    profiles.columns = profiles.columns.map(str)
    dhw = pd.read_csv(dhw_profiles_csv, index_col=0)
    dhw.columns = dhw.columns.map(str)

    config = yaml.safe_load(sim_path.read_text(encoding="utf-8")) or {}
    excluded = {"battery", "bess", "community"}
    user_keys = [
        str(key) for key in config.get("users_list", [])[:max_users]
        if str(key).strip().lower() not in excluded
    ]
    roots = [
        sim_path.parent / "users_v3",
    ]

    names: list[str] = []
    columns: dict[str, list] = {key: [] for key in (
        "e_pv e_base e_boiler_measured p_dhw size_bess eta_bess_in eta_bess_out "
        "eta_bess_stor soc_bess_min soc_bess_max t_bess_min size_elh vol_hss_water "
        "t_env t_max t_min t_in t_set a_hss eta_elh"
    ).split()}

    for key in user_keys:
        yaml_path = _find_yaml(roots, key)
        if yaml_path is None:
            print(f"[WARN] YAML nem található: {key} — kihagyom.")
            continue
        units = (yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}).get("units") or {}
        ue, pv, ut = units.get("ue") or {}, units.get("pv") or {}, units.get("ut") or {}
        hss, bess = units.get("hss") or {}, units.get("bess") or {}

        names.append(str((units.get("name") or {}).get("name", key)))
        columns["e_base"].append(_profile(profiles, ue.get("profile")))
        columns["e_pv"].append(_profile(profiles, pv.get("profile")))
        columns["e_boiler_measured"].append(_profile(profiles, ut.get("profile")))

        liters = _profile(dhw, hss.get("profile"))
        delta_t = max(float(hss.get("T_out", 40.0)) - float(hss.get("T_in", 10.0)), 0.0)
        columns["p_dhw"].append(liters * C_HSS * delta_t / DT)

        defaults = {
            "size_bess": (bess, "bess_size", 0.0),
            "eta_bess_in": (bess, "eta_bess_in", 0.98),
            "eta_bess_out": (bess, "eta_bess_out", 0.96),
            "eta_bess_stor": (bess, "eta_bess_stor", 0.995),
            "soc_bess_min": (bess, "soc_bess_min", 0.10),
            "soc_bess_max": (bess, "soc_bess_max", 0.90),
            "t_bess_min": (bess, "t_bess_min", 2.0),
            "size_elh": (hss, "size_elh", 0.0),
            "vol_hss_water": (hss, "vol_hss_water", 0.0),
            "t_env": (hss, "T_env", 20.0),
            "t_max": (hss, "T_max", 65.0),
            "t_min": (hss, "T_min", 38.0),
            "t_in": (hss, "T_in", 10.0),
            "t_set": (hss, "T_set", hss.get("T_setpoint", 50.0)),
            "a_hss": (hss, "a_hss", 0.01275),
            "eta_elh": (hss, "eta_elh", 0.95),
        }
        for field, (source, source_key, default) in defaults.items():
            columns[field].append(float(source.get(source_key, default)))

    if not names:
        raise RuntimeError("Nincs feldolgozható felhasználó.")

    matrices = {key: np.column_stack(columns[key]).astype(float) for key in (
        "e_pv", "e_base", "e_boiler_measured", "p_dhw"
    )}
    vectors = {key: np.asarray(columns[key], dtype=float) for key in columns if key not in matrices}
    return NonoptInput(user_names=names, **matrices, **vectors)
