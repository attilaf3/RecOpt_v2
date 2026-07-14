from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml


__all__ = [
    "SimulationInputs",
    "read_simulation_inputs",
]


DT_DEFAULT = 0.25
N_STEPS_DEFAULT = 35040

EXCLUDE_DEFAULT = {"battery", "bess", "community"}

RHO_WATER_KG_PER_L = 1.0
CP_WATER_J_PER_KGK = 4186.0
J_PER_KWH = 3_600_000.0
KWH_PER_L_PER_K = RHO_WATER_KG_PER_L * CP_WATER_J_PER_KGK / J_PER_KWH


@dataclass
class SimulationInputs:
    """
    Központi input objektum minden szcenárióhoz.

    A villamos CSV-profilokat energiaként kezeljük:
        e_*_kwh: kWh / időlépés

    Az optimalizálókhoz ezekből teljesítményt számolunk:
        p_*_kw = e_*_kwh / dt

    A DHW profil liter / időlépés alapú:
        liter / step -> kWhth / step -> kWth
    """

    dt: float

    user_keys: list[str]
    user_names: list[str]
    user_yaml_paths: list[Path]

    # Villamos energia-idősorok [kWh / step]
    e_pv_kwh: np.ndarray
    e_ue_kwh: np.ndarray
    e_el_heater_kwh: np.ndarray

    # DHW hőenergia-idősor [kWhth / step]
    e_dhw_kwh: np.ndarray

    # Villamos teljesítmény-idősorok [kW]
    p_pv_kw: np.ndarray
    p_ue_kw: np.ndarray
    p_el_heater_kw: np.ndarray

    # DHW hőteljesítmény-idősor [kWth]
    p_dhw_kw: np.ndarray

    # HSS / bojler paraméterek
    size_elh: np.ndarray
    vol_hss_water: np.ndarray
    T_env: np.ndarray
    T_min: np.ndarray
    T_max: np.ndarray
    T_in: np.ndarray
    T_out: np.ndarray
    a_hss: np.ndarray
    eta_elh: np.ndarray
    t_hss_min_in: np.ndarray

    # BESS paraméterek
    size_bess: np.ndarray
    eta_bess_in: np.ndarray
    eta_bess_out: np.ndarray
    eta_bess_stor: np.ndarray
    soc_bess_min: np.ndarray
    soc_bess_max: np.ndarray
    t_bess_min: np.ndarray

    @property
    def n_steps(self) -> int:
        return int(self.e_ue_kwh.shape[0])

    @property
    def n_users(self) -> int:
        return int(self.e_ue_kwh.shape[1])

    @property
    def has_pv(self) -> np.ndarray:
        return self.e_pv_kwh.sum(axis=0) > 1e-9

    @property
    def has_bess_config(self) -> np.ndarray:
        return self.size_bess > 1e-9

    @property
    def has_measured_boiler(self) -> np.ndarray:
        return self.e_el_heater_kwh.sum(axis=0) > 1e-9

    @property
    def has_dhw_profile(self) -> np.ndarray:
        return self.e_dhw_kwh.sum(axis=0) > 1e-9

    @property
    def has_hss_config(self) -> np.ndarray:
        return (self.size_elh > 1e-9) & (self.vol_hss_water > 1e-9)


def _as_path(path: os.PathLike | str) -> Path:
    return Path(path).expanduser().resolve()


def _read_yaml(path: os.PathLike | str) -> dict:
    path = _as_path(path)
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _normalise_exclude(exclude: Iterable[str] | None) -> set[str]:
    if exclude is None:
        exclude = EXCLUDE_DEFAULT
    return {str(x).strip().lower() for x in exclude}


def _safe_float(value, default: float) -> float:
    if value is None:
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _keep_n_steps(
    values: np.ndarray,
    n_steps: int,
    name: str,
) -> np.ndarray:
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size != int(n_steps):
        raise ValueError(f"{name}: a profil hossza {arr.size}, de {n_steps} kell.")
    return arr


def _energy_profile_kwh_step(
    values: np.ndarray,
    n_steps: int,
    name: str,
) -> np.ndarray:
    """
    CSV-idősor energiaként kezelve.

    Kimenet:
        kWh / step
    """
    arr = _keep_n_steps(values, n_steps=n_steps, name=name)
    return np.maximum(arr, 0.0)


def _find_user_yaml(
    roots: Iterable[os.PathLike | str],
    user_key: str,
) -> Path | None:
    """
    User YAML keresése.

    Elfogadott alakok:
        user_001
        user_001.yaml
        user_001.yml
    """
    for root in roots:
        root = Path(root)

        candidates = [
            root / str(user_key),
            root / f"{user_key}.yaml",
            root / f"{user_key}.yml",
        ]

        for path in candidates:
            if path.exists() and path.is_file():
                return path

    return None


def _read_users(
    sim_yaml_path: os.PathLike | str,
    max_users: int | None,
    exclude: Iterable[str] | None,
    target_user: str | None = None,
) -> list[str]:
    """
    Felhasználólista beolvasása.

    Prioritás:
        1) simulation YAML users_list mező
        2) simulation YAML melletti Users/*.yaml vagy Users/*.yml fájlok

    Fontos:
        előbb szűrjük ki a pseudo-usereket,
        és csak utána alkalmazzuk a max_users limitet.
    """
    sim_yaml_path = _as_path(sim_yaml_path)
    sim = _read_yaml(sim_yaml_path)
    exclude_set = _normalise_exclude(exclude)

    raw_users_list = sim.get("users_list")

    if raw_users_list:
        users_list = [str(u) for u in raw_users_list]
    else:
        users_dir = sim_yaml_path.parent / "Users"

        if not users_dir.exists():
            raise RuntimeError(
                "A simulation YAML nem tartalmaz users_list-et, "
                f"és a Users könyvtár sem található: {users_dir}"
            )

        users_list = sorted(
            p.stem
            for p in users_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".yaml", ".yml"}
        )

        if not users_list:
            raise RuntimeError(
                "A simulation YAML nem tartalmaz users_list-et, "
                "és a Users könyvtárban sincs .yaml vagy .yml fájl."
            )

    users_list = [
        u for u in users_list
        if str(u).strip().lower() not in exclude_set
    ]

    if target_user is not None:
        target = str(target_user).strip()
        users_list = [u for u in users_list if str(u).strip() == target]
        if not users_list:
            raise RuntimeError(f"A megadott háztartás nem található a users_list-ben: {target}")

    if max_users is not None:
        users_list = users_list[: int(max_users)]

    if not users_list:
        raise RuntimeError(
            "A simulation YAML nem tartalmaz érvényes users_list-et, "
            "vagy max_users=0."
        )

    return users_list


def _read_profiles_csv(
    profiles_csv_path: os.PathLike | str,
    exclude: Iterable[str] | None,
) -> pd.DataFrame:
    """
    Villamos profil CSV beolvasása.

    Elvárt:
        index_col=0
        oszlopok: user YAML-ekben hivatkozott profilnevek

    Az értékeket energiaként kezeljük:
        kWh / step
    """
    profiles_csv_path = _as_path(profiles_csv_path)
    exclude_set = _normalise_exclude(exclude)

    df = pd.read_csv(profiles_csv_path, index_col=0)
    df.columns = [str(c) for c in df.columns]

    keep_cols = [
        c for c in df.columns
        if str(c).strip().lower() not in exclude_set
    ]

    return df[keep_cols]


def _read_dhw_csv(
    dhw_profile_path: os.PathLike | str | None,
) -> pd.DataFrame | None:
    """
    DHW CSV beolvasása.

    Elvárt:
        index_col=0
        értékek: liter / step

    Ha nincs megadva dhw_profile_path, akkor None.
    """
    if dhw_profile_path is None:
        return None

    dhw_profile_path = _as_path(dhw_profile_path)

    dhw = pd.read_csv(dhw_profile_path, index_col=0)
    dhw.columns = [str(c) for c in dhw.columns]

    return dhw


def _profile_energy_from_df(
    df: pd.DataFrame,
    profile_name: str | None,
    n_steps: int,
    label: str,
    *,
    default_zero: bool = True,
) -> np.ndarray:
    """
    Egy CSV-oszlop kiolvasása energiaként.

    Kimenet:
        kWh / step
    """
    if profile_name is None or str(profile_name).strip() == "":
        if default_zero:
            return np.zeros(n_steps, dtype=float)
        raise KeyError(f"{label}: nincs megadva profilnév.")

    profile_name = str(profile_name)

    if profile_name not in df.columns:
        if default_zero:
            return np.zeros(n_steps, dtype=float)
        raise KeyError(f"{label}: nincs ilyen oszlop a CSV-ben: {profile_name}")

    return _energy_profile_kwh_step(
        df[profile_name].to_numpy(),
        n_steps=n_steps,
        name=f"{label} ({profile_name})",
    )


def _dhw_liters_to_thermal_energy_kwh(
    liters_per_step: np.ndarray,
    T_in: float,
    T_out: float,
    n_steps: int,
) -> np.ndarray:
    """
    DHW literprofilból hőenergia-idősor.

    Képlet:
        E_kWh = V_liter * rho * cp * (T_out - T_in) / 3_600_000

    Kimenet:
        kWhth / step
    """
    liters = _keep_n_steps(
        liters_per_step,
        n_steps=n_steps,
        name="DHW liter profil",
    )
    liters = np.maximum(liters, 0.0)

    dT = max(float(T_out) - float(T_in), 0.0)

    return liters * KWH_PER_L_PER_K * dT


def read_simulation_inputs(
    sim_yaml_path: os.PathLike | str,
    profiles_csv_path: os.PathLike | str,
    dhw_profile_path: os.PathLike | str | None = None,
    *,
    max_users: int | None = None,
    pv_ratio: float = 1.0,
    dt: float = DT_DEFAULT,
    n_steps: int = N_STEPS_DEFAULT,
    search_roots: Iterable[os.PathLike | str] | None = None,
    exclude: Iterable[str] | None = None,
    warn_missing_user_yaml: bool = True,
    target_user: str | None = None,
) -> SimulationInputs:
    """
    Egyetlen központi input-beolvasó minden szcenárióhoz.

    Ez olvassa:
        - simulation YAML
        - users_list vagy Users/*.yaml
        - measurements/profiles CSV
        - opcionálisan dhw.csv
        - user YAML-ek
        - UE, PV, UT/ELH profilok
        - BESS paraméterek
        - HSS/DHW paraméterek

    Ez adja:
        - e_*_kwh idősorokat nem optimalizált modellekhez
        - p_*_kw idősorokat optimalizált modellekhez
        - p_dhw_kw idősorokat bojler/HSS optimalizáláshoz
    """
    sim_yaml_path = _as_path(sim_yaml_path)
    profiles_csv_path = _as_path(profiles_csv_path)

    if search_roots is None:
        search_roots = [
            sim_yaml_path.parent / "Users",
            sim_yaml_path.parent,
        ]
    else:
        search_roots = [Path(p) for p in search_roots]

    users_list = _read_users(
        sim_yaml_path=sim_yaml_path,
        max_users=max_users,
        exclude=exclude,
        target_user=target_user,
    )

    profiles_df = _read_profiles_csv(
        profiles_csv_path=profiles_csv_path,
        exclude=exclude,
    )

    dhw_df = _read_dhw_csv(dhw_profile_path)

    user_keys: list[str] = []
    user_names: list[str] = []
    user_yaml_paths: list[Path] = []

    e_pv_cols: list[np.ndarray] = []
    e_ue_cols: list[np.ndarray] = []
    e_el_heater_cols: list[np.ndarray] = []
    e_dhw_cols: list[np.ndarray] = []

    size_elh: list[float] = []
    vol_hss_water: list[float] = []

    T_env: list[float] = []
    T_min: list[float] = []
    T_max: list[float] = []
    T_in: list[float] = []
    T_out: list[float] = []
    a_hss: list[float] = []
    eta_elh: list[float] = []
    t_hss_min_in: list[float] = []

    size_bess: list[float] = []
    eta_bess_in: list[float] = []
    eta_bess_out: list[float] = []
    eta_bess_stor: list[float] = []
    soc_bess_min: list[float] = []
    soc_bess_max: list[float] = []
    t_bess_min: list[float] = []

    for user_key in users_list:
        user_yaml_path = _find_user_yaml(search_roots, str(user_key))

        if user_yaml_path is None:
            if warn_missing_user_yaml:
                print(f"[WARN] YAML nem található: {user_key} — kihagyom.")
            continue

        user_yaml = _read_yaml(user_yaml_path)
        units = user_yaml.get("units") or {}

        name = (units.get("name") or {}).get("name", str(user_key))

        user_keys.append(str(user_key))
        user_names.append(str(name))
        user_yaml_paths.append(user_yaml_path)

        # ---------------------------------------------------------------------
        # UE: alap villamos fogyasztás
        # CSV egység: kWh / step
        # ---------------------------------------------------------------------
        ue = units.get("ue") or {}
        ue_profile = str(ue.get("profile")) if ue.get("profile") is not None else None

        e_ue = _profile_energy_from_df(
            profiles_df,
            ue_profile,
            n_steps=n_steps,
            label=f"UE - {name}",
            default_zero=True,
        )
        e_ue_cols.append(e_ue)

        # ---------------------------------------------------------------------
        # PV-termelés
        # CSV egység: kWh / step
        # pv_ratio itt skálázza a termelést.
        # ---------------------------------------------------------------------
        pv = units.get("pv") or {}
        pv_profile = str(pv.get("profile")) if pv.get("profile") is not None else None

        e_pv = _profile_energy_from_df(
            profiles_df,
            pv_profile,
            n_steps=n_steps,
            label=f"PV - {name}",
            default_zero=True,
        )
        e_pv = e_pv * float(pv_ratio)
        e_pv_cols.append(e_pv)

        # ---------------------------------------------------------------------
        # UT / mért villamos bojlerprofil
        # CSV egység: kWh / step
        # ---------------------------------------------------------------------
        ut = units.get("ut") or {}
        ut_profile = str(ut.get("profile")) if ut.get("profile") is not None else None

        e_el_heater = _profile_energy_from_df(
            profiles_df,
            ut_profile,
            n_steps=n_steps,
            label=f"UT/ELH - {name}",
            default_zero=True,
        )
        e_el_heater_cols.append(e_el_heater)

        # ---------------------------------------------------------------------
        # BESS paraméterek
        # ---------------------------------------------------------------------
        bess = units.get("bess") or {}

        size_bess.append(_safe_float(bess.get("bess_size"), 0.0))
        eta_bess_in.append(_safe_float(bess.get("eta_bess_in"), 0.98))
        eta_bess_out.append(_safe_float(bess.get("eta_bess_out"), 0.96))
        eta_bess_stor.append(_safe_float(bess.get("eta_bess_stor"), 0.995))
        soc_bess_min.append(_safe_float(bess.get("soc_bess_min"), 0.10))
        soc_bess_max.append(_safe_float(bess.get("soc_bess_max"), 0.90))
        t_bess_min.append(_safe_float(bess.get("t_bess_min"), 2.0))

        # ---------------------------------------------------------------------
        # HSS / bojler paraméterek
        # ---------------------------------------------------------------------
        hss = units.get("hss") or {}

        size_elh_u = _safe_float(hss.get("size_elh"), 0.0)
        vol_hss_water_u = _safe_float(hss.get("vol_hss_water"), 0.0)

        T_env_u = _safe_float(hss.get("T_env"), 20.0)
        T_min_u = _safe_float(hss.get("T_min"), 10.0)
        T_max_u = _safe_float(hss.get("T_max"), 65.0)
        T_in_u = _safe_float(hss.get("T_in"), 10.0)
        T_out_u = _safe_float(hss.get("T_out"), 55.0)
        a_hss_u = _safe_float(hss.get("a_hss"), 0.01275)
        eta_elh_u = _safe_float(hss.get("eta_elh"), 0.95)
        t_hss_min_in_u = _safe_float(hss.get("t_hss_min_in"), 0.0)

        size_elh.append(size_elh_u)
        vol_hss_water.append(vol_hss_water_u)

        T_env.append(T_env_u)
        T_min.append(T_min_u)
        T_max.append(T_max_u)
        T_in.append(T_in_u)
        T_out.append(T_out_u)
        a_hss.append(a_hss_u)
        eta_elh.append(eta_elh_u)
        t_hss_min_in.append(t_hss_min_in_u)

        # ---------------------------------------------------------------------
        # DHW literprofil -> hőenergia-idősor
        # dhw CSV egység: liter / step
        # kimenet: kWhth / step
        # ---------------------------------------------------------------------
        hss_profile = str(hss.get("profile")) if hss.get("profile") is not None else None

        if dhw_df is not None and hss_profile is not None and hss_profile in dhw_df.columns:
            e_dhw = _dhw_liters_to_thermal_energy_kwh(
                liters_per_step=dhw_df[hss_profile].to_numpy(),
                T_in=T_in_u,
                T_out=T_out_u,
                n_steps=n_steps,
            )
        else:
            e_dhw = np.zeros(n_steps, dtype=float)

        e_dhw_cols.append(e_dhw)

    if not user_names:
        raise RuntimeError("Nincs érvényes felhasználó a megadott inputok alapján.")

    e_pv_kwh = np.column_stack(e_pv_cols).astype(float)
    e_ue_kwh = np.column_stack(e_ue_cols).astype(float)
    e_el_heater_kwh = np.column_stack(e_el_heater_cols).astype(float)
    e_dhw_kwh = np.column_stack(e_dhw_cols).astype(float)

    p_pv_kw = e_pv_kwh / float(dt)
    p_ue_kw = e_ue_kwh / float(dt)
    p_el_heater_kw = e_el_heater_kwh / float(dt)
    p_dhw_kw = e_dhw_kwh / float(dt)

    return SimulationInputs(
        dt=float(dt),

        user_keys=user_keys,
        user_names=user_names,
        user_yaml_paths=user_yaml_paths,

        e_pv_kwh=e_pv_kwh,
        e_ue_kwh=e_ue_kwh,
        e_el_heater_kwh=e_el_heater_kwh,
        e_dhw_kwh=e_dhw_kwh,

        p_pv_kw=p_pv_kw,
        p_ue_kw=p_ue_kw,
        p_el_heater_kw=p_el_heater_kw,
        p_dhw_kw=p_dhw_kw,

        size_elh=np.asarray(size_elh, dtype=float),
        vol_hss_water=np.asarray(vol_hss_water, dtype=float),
        T_env=np.asarray(T_env, dtype=float),
        T_min=np.asarray(T_min, dtype=float),
        T_max=np.asarray(T_max, dtype=float),
        T_in=np.asarray(T_in, dtype=float),
        T_out=np.asarray(T_out, dtype=float),
        a_hss=np.asarray(a_hss, dtype=float),
        eta_elh=np.asarray(eta_elh, dtype=float),
        t_hss_min_in=np.asarray(t_hss_min_in, dtype=float),

        size_bess=np.asarray(size_bess, dtype=float),
        eta_bess_in=np.asarray(eta_bess_in, dtype=float),
        eta_bess_out=np.asarray(eta_bess_out, dtype=float),
        eta_bess_stor=np.asarray(eta_bess_stor, dtype=float),
        soc_bess_min=np.asarray(soc_bess_min, dtype=float),
        soc_bess_max=np.asarray(soc_bess_max, dtype=float),
        t_bess_min=np.asarray(t_bess_min, dtype=float),
    )
