"""Mert bojlerprofilok skalazasa es B/A tarifas szamla-osszehasonlitasa.

A program ket, egymastol elvalasztott feladatot vegez:

1. A mert bojler-villamosprofil alakjat megtartva ugy skalazza annak eves
   energiajat, hogy megegyezzen a tiltott idosav nelkuli, setpointkoveto
   hotarolomodell villamosenergia-felvetelevel. A tiltasi sav nelkuli modell
   kizarolag a skalazasi cel meghatarozasara szolgal, szamlazasi esetkent nem
   szerepel.
2. BESS nelkul osszehasonlitja:
     - a skalazott mert, B tarifas bojlerprofilt;
     - a PV-s es ervenyes HSS-sel rendelkezo haztartasoknal a 06-09 es 17-22
       ora kozotti tiltast alkalmazo, A tarifas modellezett bojlert.

A nem PV-s, illetve nem modellezheto bojlerek mindket szamlazasi esetben a
skalazott mert profilon es B tarifan maradnak. A 40 C-os komfortszintet mindket
hotarolomodellnel gordulo 6 oras ablakkal ellenorizzuk.

A profilok kWh/idolepes, a teljesitmenyek kW mertekegyseguek. DT=0.25 ora.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml


DT = 0.25
EXPECTED_STEPS = 35040
C_WATER_KWH_PER_L_C = 0.00116667
EPS = 1e-12

# ---------------------------------------------------------------------------
# PYCHARM / SHIFT+F10 BEALLITASOK
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = (
    SCRIPT_DIR.parent if (SCRIPT_DIR.parent / "Input").exists() else SCRIPT_DIR
)
INPUT_DIR = PROJECT_ROOT / "Input"

SIM_YAML = INPUT_DIR / "simulation_config_disaggregated_with_userlist.yaml"
USERS_DIR = INPUT_DIR / "users_v2"
DHW_PROFILES_CSV = INPUT_DIR / "dhw_v2.csv"

# A PV-kiosztas altal letrehozott v3-at olvassuk es frissitjuk.
# Azonos forras/kimenet eseten az elso modositas elott masolat keszul.
SOURCE_MEASUREMENTS_CSV = INPUT_DIR / "measurements_disaggregated_v2.csv"
OUTPUT_MEASUREMENTS_CSV = INPUT_DIR / "measurements_disaggregated_v2.csv"
BACKUP_MEASUREMENTS_CSV = (
    INPUT_DIR / "measurements_disaggregated_v2_before_boiler_scaling.csv"
)

OUT_DIR = SCRIPT_DIR / "results_scaled_boiler_bill_comparison"
MAX_USERS: int | None = None

# None: minden ervenyes, mert profillal rendelkezo HSS skalazasa.
# Lista: csak a felsorolt YAML-kodok vagy felhasznalonevek skalazasa.
SCALE_HOUSEHOLDS: list[str] | None = None

OVERWRITE_OUTPUT = True
FAIL_IF_NO_BLOCK_MODEL_INVALID = True

# A tesztben hasznalt homersekletek. Nem a YAML T_set erteket olvassuk.
T_IN_C = 10.0
T_ENV_C = 20.0
T_COMFORT_C = 40.0
T_SETPOINT_DEFAULT_C = 50.0
T_SETPOINT_MIN_C = 50.0
T_MAX_C = 65.0
COMFORT_REACH_INTERVAL_HOURS = 6.0
MORNING_BLOCK = (6.0, 9.0)
EVENING_BLOCK = (17.0, 22.0)

# Haztartasonkenti eltero setpoint. A kulcs YAML-kod vagy felhasznalonev.
HOUSEHOLD_SETPOINT_C: dict[str, float] = {
    "load_0420144888235070": 57.0,
    "load_0420144888397058": 59.0,
    "load_0420144888439778": 55.0,
    "load_0420144888340898": 52.0,
    "load_0420144888377089": 52.0,
}

# Brutto elszamolas [Ft/kWh]. Az A- es B-savorszamitas haztartasonkent kulon
# tortenik. A modellezett bojler A, a mert bojler B tarifat kap.
A_LOW_LIMIT_KWH = 2523.0
A_LOW_PRICE_FT_PER_KWH = 36.0
A_HIGH_PRICE_FT_PER_KWH = 71.0
B_LOW_LIMIT_KWH = 2523.0
B_LOW_PRICE_FT_PER_KWH = 23.0
B_HIGH_PRICE_FT_PER_KWH = 61.0
EXPORT_PRICE_FT_PER_KWH = 5.0

EXCLUDE_USERS = {"battery", "bess", "community"}


@dataclass
class Household:
    key: str
    name: str
    has_pv: bool
    has_hss: bool
    base_profile: str | None
    pv_profile: str | None
    measured_boiler_profile: str | None
    dhw_profile: str | None
    base_kwh_step: np.ndarray
    pv_kwh_step: np.ndarray
    measured_boiler_kwh_step: np.ndarray
    dhw_litre_step: np.ndarray
    size_elh_kw: float
    volume_litre: float
    eta_elh: float
    a_hss_kw_per_k: float
    setpoint_c: float


@dataclass
class BoilerSimulation:
    heater_kwh_step: np.ndarray
    temperature_c: np.ndarray
    blocked: np.ndarray
    dhw_shortfall_kwhth: np.ndarray
    rolling_6h_violation: np.ndarray
    pass_energy: bool
    pass_6h_40: bool

    @property
    def pass_all(self) -> bool:
        return self.pass_energy and self.pass_6h_40


def _read_indexed_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Nem talalhato CSV: {path.resolve()}")
    frame = pd.read_csv(path, index_col=0)
    frame.columns = frame.columns.map(str)
    if len(frame) != EXPECTED_STEPS:
        raise ValueError(
            f"{path.name}: {len(frame)} sor van, de {EXPECTED_STEPS} szukseges."
        )
    return frame


def _profile(frame: pd.DataFrame, name: object) -> np.ndarray:
    if name in (None, "") or str(name) not in frame.columns:
        return np.zeros(EXPECTED_STEPS)
    return np.maximum(
        pd.to_numeric(frame[str(name)], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float),
        0.0,
    )


def _find_user_yaml(users_dir: Path, sim_parent: Path, key: str) -> Path | None:
    roots = (
        users_dir,
        sim_parent / "users_v2",
        sim_parent / "Users_v2",
        sim_parent,
    )
    for root in roots:
        for filename in (key, f"{key}.yaml", f"{key}.yml"):
            candidate = root / filename
            if candidate.exists():
                return candidate
    return None


def _setpoint_for(key: str, name: str) -> float:
    value = float(
        HOUSEHOLD_SETPOINT_C.get(
            key, HOUSEHOLD_SETPOINT_C.get(name, T_SETPOINT_DEFAULT_C)
        )
    )
    if not T_SETPOINT_MIN_C <= value <= T_MAX_C:
        raise ValueError(
            f"{key}: a setpoint {value:g} C; "
            f"{T_SETPOINT_MIN_C:g}-{T_MAX_C:g} C szukseges."
        )
    return value


def load_households(
    sim_yaml: Path,
    users_dir: Path,
    measurements: pd.DataFrame,
    dhw_profiles: pd.DataFrame,
    max_users: int | None,
) -> list[Household]:
    config = yaml.safe_load(sim_yaml.read_text(encoding="utf-8")) or {}
    user_keys = [
        str(key)
        for key in config.get("users_list", [])
        if str(key).strip().lower() not in EXCLUDE_USERS
    ]
    if max_users is not None:
        user_keys = user_keys[: int(max_users)]

    households: list[Household] = []
    for key in user_keys:
        yaml_path = _find_user_yaml(users_dir, sim_yaml.parent, key)
        if yaml_path is None:
            print(f"[WARN] YAML nem talalhato, kihagyva: {key}")
            continue
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        units = data.get("units", data)
        if not isinstance(units, dict):
            print(f"[WARN] Ervenytelen units blokk, kihagyva: {key}")
            continue

        name = str((units.get("name") or {}).get("name", key))
        ue = dict(units.get("ue") or {})
        pv = dict(units.get("pv") or {})
        ut = dict(units.get("ut") or {})
        hss = dict(units.get("hss") or {})

        base_profile = ue.get("profile")
        pv_profile = pv.get("profile")
        boiler_profile = ut.get("profile")
        dhw_profile = hss.get("profile")

        base = _profile(measurements, base_profile)
        pv_energy = _profile(measurements, pv_profile)
        measured_boiler = _profile(measurements, boiler_profile)
        dhw_litres = _profile(dhw_profiles, dhw_profile)

        size_elh = float(hss.get("size_elh", 0.0) or 0.0)
        volume = float(hss.get("vol_hss_water", 0.0) or 0.0)
        has_hss = bool(
            size_elh > EPS
            and volume > EPS
            and dhw_profile not in (None, "")
            and str(dhw_profile) in dhw_profiles.columns
        )

        households.append(
            Household(
                key=key,
                name=name,
                has_pv=bool(pv_energy.sum() > EPS),
                has_hss=has_hss,
                base_profile=(str(base_profile) if base_profile not in (None, "") else None),
                pv_profile=(str(pv_profile) if pv_profile not in (None, "") else None),
                measured_boiler_profile=(
                    str(boiler_profile) if boiler_profile not in (None, "") else None
                ),
                dhw_profile=(str(dhw_profile) if dhw_profile not in (None, "") else None),
                base_kwh_step=base,
                pv_kwh_step=pv_energy,
                measured_boiler_kwh_step=measured_boiler,
                dhw_litre_step=dhw_litres,
                size_elh_kw=size_elh,
                volume_litre=volume,
                eta_elh=float(hss.get("eta_elh", 0.95) or 0.95),
                a_hss_kw_per_k=float(hss.get("a_hss", 0.01275) or 0.01275),
                setpoint_c=_setpoint_for(key, name),
            )
        )
    if not households:
        raise RuntimeError("Nincs feldolgozhato haztartas.")
    return households


def _in_window(hour: float, window: tuple[float, float]) -> bool:
    return window[0] <= hour < window[1]


def _rolling_6h_violation(reached_40: np.ndarray) -> np.ndarray:
    window_steps = int(round(COMFORT_REACH_INTERVAL_HOURS / DT))
    if abs(window_steps * DT - COMFORT_REACH_INTERVAL_HOURS) > 1e-9:
        raise ValueError("A 6 oras komfortablak nem illeszkedik a DT idoracsra.")
    result = np.zeros(reached_40.size, dtype=bool)
    if reached_40.size < window_steps:
        return result
    hit_count = np.convolve(
        np.asarray(reached_40, dtype=np.int16),
        np.ones(window_steps, dtype=np.int16),
        mode="valid",
    )
    bad_starts = np.flatnonzero(hit_count == 0)
    result[bad_starts + window_steps - 1] = True
    return result


def simulate_boiler(h: Household, *, use_blocked_windows: bool) -> BoilerSimulation:
    if not h.has_hss:
        raise ValueError(f"{h.key}: nincs ervenyes HSS a modellezeshez.")
    if not (T_IN_C <= T_COMFORT_C <= h.setpoint_c <= T_MAX_C):
        raise ValueError(f"{h.key}: ervenytelen homersekletsorrend.")
    if h.eta_elh <= EPS:
        raise ValueError(f"{h.key}: eta_elh nem pozitiv.")

    n = EXPECTED_STEPS
    steps_per_day = int(round(24.0 / DT))
    capacity = h.volume_litre * C_WATER_KWH_PER_L_C
    energy_setpoint = capacity * (h.setpoint_c - T_IN_C)
    energy_max = capacity * (T_MAX_C - T_IN_C)
    q_dhw = h.dhw_litre_step * C_WATER_KWH_PER_L_C * (
        T_COMFORT_C - T_IN_C
    )

    energy = np.zeros(n + 1)
    temperature = np.zeros(n)
    heater_power = np.zeros(n)
    blocked = np.zeros(n, dtype=bool)
    shortfall = np.zeros(n)
    energy[0] = min(max(energy_setpoint, 0.0), energy_max)

    for t in range(n):
        hour = (t % steps_per_day) * DT
        blocked[t] = use_blocked_windows and (
            _in_window(hour, MORNING_BLOCK) or _in_window(hour, EVENING_BLOCK)
        )
        energy_now = energy[t]
        temp_now = T_IN_C + energy_now / capacity
        temperature[t] = temp_now
        loss_kw = h.a_hss_kw_per_k * (temp_now - T_ENV_C)

        requested = q_dhw[t]
        delivered = min(requested, max(energy_now, 0.0))
        shortfall[t] = max(requested - delivered, 0.0)

        max_power = 0.0 if blocked[t] else h.size_elh_kw
        max_by_temperature = max(
            (energy_max - energy_now + delivered + DT * loss_kw)
            / (DT * h.eta_elh),
            0.0,
        )
        max_power = min(max_power, max_by_temperature)

        required_power = max(
            (energy_setpoint - energy_now + delivered + DT * loss_kw)
            / (DT * h.eta_elh),
            0.0,
        )
        heater_power[t] = min(required_power, max_power)
        energy[t + 1] = np.clip(
            energy_now
            + DT * (h.eta_elh * heater_power[t] - loss_kw)
            - delivered,
            0.0,
            energy_max,
        )

    reached_40 = temperature >= T_COMFORT_C - 1e-6
    rolling_violation = _rolling_6h_violation(reached_40)
    return BoilerSimulation(
        heater_kwh_step=heater_power * DT,
        temperature_c=temperature,
        blocked=blocked,
        dhw_shortfall_kwhth=shortfall,
        rolling_6h_violation=rolling_violation,
        pass_energy=bool(not np.any(shortfall > 1e-9)),
        pass_6h_40=bool(not np.any(rolling_violation)),
    )


def _selected_for_scaling(h: Household, selected: set[str] | None) -> bool:
    if selected is None:
        return True
    return h.key in selected or h.name in selected


def scale_measured_boiler_profiles(
    households: list[Household],
    measurements: pd.DataFrame,
    selected_households: list[str] | None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, BoilerSimulation]]:
    selected = (
        None
        if selected_households is None
        else {str(value).strip() for value in selected_households if str(value).strip()}
    )
    if selected is not None:
        known = {h.key for h in households} | {h.name for h in households}
        unknown = sorted(selected.difference(known))
        if unknown:
            raise ValueError("Ismeretlen SCALE_HOUSEHOLDS azonosito(k): " + ", ".join(unknown))

    scaled = measurements.copy()
    rows: list[dict] = []
    no_block_models: dict[str, BoilerSimulation] = {}
    profile_owner: dict[str, str] = {}

    for h in households:
        measured_energy = float(h.measured_boiler_kwh_step.sum())
        chosen = _selected_for_scaling(h, selected)
        valid_profile = bool(
            h.measured_boiler_profile
            and h.measured_boiler_profile in scaled.columns
            and measured_energy > EPS
        )
        status = "NOT_SELECTED"
        target_energy = np.nan
        scale_factor = np.nan
        no_block: BoilerSimulation | None = None

        if chosen and h.has_hss and valid_profile:
            profile = str(h.measured_boiler_profile)
            previous_owner = profile_owner.get(profile)
            if previous_owner is not None and previous_owner != h.key:
                raise ValueError(
                    f"A(z) {profile} mert bojlerprofilt tobb haztartas hasznalja: "
                    f"{previous_owner}, {h.key}. Haztartasonkenti skalazashoz egyedi profil kell."
                )
            profile_owner[profile] = h.key
            no_block = simulate_boiler(h, use_blocked_windows=False)
            no_block_models[h.key] = no_block
            target_energy = float(no_block.heater_kwh_step.sum())
            if FAIL_IF_NO_BLOCK_MODEL_INVALID and not no_block.pass_all:
                raise RuntimeError(
                    f"{h.key}: a tiltasi sav nelkuli modell nem teljesiti az "
                    "energia- vagy a 6 oras 40 C-os kovetelmenyt; nem skalazok."
                )
            scale_factor = target_energy / measured_energy
            scaled[profile] = (
                pd.to_numeric(scaled[profile], errors="coerce")
                .fillna(0.0)
                .clip(lower=0.0)
                * scale_factor
            )
            h.measured_boiler_kwh_step = np.maximum(
                scaled[profile].to_numpy(dtype=float), 0.0
            )
            status = "SCALED"
        elif chosen and not h.has_hss:
            status = "SKIP_NO_VALID_HSS"
        elif chosen and not valid_profile:
            status = "SKIP_NO_POSITIVE_MEASURED_PROFILE"

        scaled_energy = float(h.measured_boiler_kwh_step.sum())
        rows.append(
            {
                "user_key": h.key,
                "household": h.name,
                "has_pv": int(h.has_pv),
                "has_hss": int(h.has_hss),
                "measured_boiler_profile": h.measured_boiler_profile,
                "setpoint_c": h.setpoint_c,
                "scaling_status": status,
                "original_measured_boiler_kwh": measured_energy,
                "no_block_target_boiler_kwh": target_energy,
                "scale_factor": scale_factor,
                "scaled_measured_boiler_kwh": scaled_energy,
                "scaling_error_kwh": (
                    scaled_energy - target_energy
                    if np.isfinite(target_energy) else np.nan
                ),
                "no_block_pass_energy": (
                    int(no_block.pass_energy) if no_block is not None else np.nan
                ),
                "no_block_pass_6h_40": (
                    int(no_block.pass_6h_40) if no_block is not None else np.nan
                ),
                "no_block_min_temperature_c": (
                    float(no_block.temperature_c.min())
                    if no_block is not None else np.nan
                ),
            }
        )
    return scaled, pd.DataFrame(rows), no_block_models


def _tier_cost(
    energy_kwh: float,
    limit_kwh: float,
    low_price: float,
    high_price: float,
) -> tuple[float, float, float]:
    energy = max(float(energy_kwh), 0.0)
    low = min(energy, limit_kwh)
    high = max(energy - low, 0.0)
    return low, high, low * low_price + high * high_price


def _settle_household(
    base_kwh_step: np.ndarray,
    pv_kwh_step: np.ndarray,
    boiler_a_kwh_step: np.ndarray,
    boiler_b_kwh_step: np.ndarray,
) -> dict[str, float]:
    # Saját PV-prioritas: alapfogyasztas, majd az A tarifas modellezett bojler.
    a_demand = base_kwh_step + boiler_a_kwh_step
    pv_to_a = np.minimum(pv_kwh_step, a_demand)
    grid_a = np.maximum(a_demand - pv_to_a, 0.0)
    export = np.maximum(pv_kwh_step - pv_to_a, 0.0)
    grid_b = np.maximum(boiler_b_kwh_step, 0.0)

    a_low, a_high, cost_a = _tier_cost(
        grid_a.sum(), A_LOW_LIMIT_KWH,
        A_LOW_PRICE_FT_PER_KWH, A_HIGH_PRICE_FT_PER_KWH,
    )
    b_low, b_high, cost_b = _tier_cost(
        grid_b.sum(), B_LOW_LIMIT_KWH,
        B_LOW_PRICE_FT_PER_KWH, B_HIGH_PRICE_FT_PER_KWH,
    )
    export_kwh = float(export.sum())
    export_revenue = export_kwh * EXPORT_PRICE_FT_PER_KWH
    return {
        "grid_import_a_kwh": float(grid_a.sum()),
        "grid_import_a_low_kwh": a_low,
        "grid_import_a_high_kwh": a_high,
        "grid_import_b_kwh": float(grid_b.sum()),
        "grid_import_b_low_kwh": b_low,
        "grid_import_b_high_kwh": b_high,
        "grid_import_cost_a_ft": cost_a,
        "grid_import_cost_b_ft": cost_b,
        "grid_export_kwh": export_kwh,
        "grid_export_revenue_ft": export_revenue,
        "bill_ft": cost_a + cost_b - export_revenue,
    }


def compare_bills(
    households: list[Household],
    scaling_report: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    scaling_status = scaling_report.set_index("user_key")["scaling_status"].to_dict()
    rows: list[dict] = []

    for h in households:
        zeros = np.zeros(EXPECTED_STEPS)
        scaled_measured = h.measured_boiler_kwh_step

        # 1. eset: minden bojler a skalazott mert profil szerint, B tarifan.
        measured_case = _settle_household(
            h.base_kwh_step, h.pv_kwh_step, zeros, scaled_measured
        )

        # 2. eset: kizarolag a PV+HSS, sikeresen skalazott haztartasok
        # kapnak tiltott idosavos modellt A tarifan. BESS nincs.
        modelled = bool(
            h.has_pv
            and h.has_hss
            and scaling_status.get(h.key) == "SCALED"
        )
        blocked_model = (
            simulate_boiler(h, use_blocked_windows=True) if modelled else None
        )
        boiler_a = blocked_model.heater_kwh_step if blocked_model else zeros
        boiler_b = zeros if blocked_model else scaled_measured
        blocked_case = _settle_household(
            h.base_kwh_step, h.pv_kwh_step, boiler_a, boiler_b
        )

        row = {
            "user_key": h.key,
            "household": h.name,
            "has_pv": int(h.has_pv),
            "has_hss": int(h.has_hss),
            "modelled_boiler_in_blocked_case": int(modelled),
            "scaled_measured_boiler_kwh": float(scaled_measured.sum()),
            "blocked_model_boiler_kwh": (
                float(boiler_a.sum()) if modelled else np.nan
            ),
            "blocked_model_pass_energy": (
                int(blocked_model.pass_energy) if blocked_model else np.nan
            ),
            "blocked_model_pass_6h_40": (
                int(blocked_model.pass_6h_40) if blocked_model else np.nan
            ),
            "blocked_model_min_temperature_c": (
                float(blocked_model.temperature_c.min())
                if blocked_model else np.nan
            ),
        }
        for key, value in measured_case.items():
            row[f"scaled_measured_B_{key}"] = value
        for key, value in blocked_case.items():
            row[f"blocked_model_A_{key}"] = value
        row["bill_saving_with_blocked_model_ft"] = (
            measured_case["bill_ft"] - blocked_case["bill_ft"]
        )
        rows.append(row)

    frame = pd.DataFrame(rows)
    pv_frame = frame[frame["has_pv"] == 1]
    modelled_frame = frame[frame["modelled_boiler_in_blocked_case"] == 1]
    summary = {
        "n_households": int(len(frame)),
        "n_pv_households": int(frame["has_pv"].sum()),
        "n_modelled_pv_boilers": int(frame["modelled_boiler_in_blocked_case"].sum()),
        "n_modelled_boilers_passing_6h_40": int(
            modelled_frame["blocked_model_pass_6h_40"].fillna(0).sum()
        ),
        "n_modelled_boilers_passing_energy": int(
            modelled_frame["blocked_model_pass_energy"].fillna(0).sum()
        ),
        "scaled_measured_B_total_bill_ft": float(
            frame["scaled_measured_B_bill_ft"].sum()
        ),
        "blocked_model_A_total_bill_ft": float(
            frame["blocked_model_A_bill_ft"].sum()
        ),
        "total_bill_saving_with_blocked_model_ft": float(
            frame["bill_saving_with_blocked_model_ft"].sum()
        ),
        "scaled_measured_B_pv_households_bill_ft": float(
            pv_frame["scaled_measured_B_bill_ft"].sum()
        ),
        "blocked_model_A_pv_households_bill_ft": float(
            pv_frame["blocked_model_A_bill_ft"].sum()
        ),
        "pv_households_bill_saving_ft": float(
            pv_frame["bill_saving_with_blocked_model_ft"].sum()
        ),
        "scaled_measured_B_modelled_pv_boilers_bill_ft": float(
            modelled_frame["scaled_measured_B_bill_ft"].sum()
        ),
        "blocked_model_A_modelled_pv_boilers_bill_ft": float(
            modelled_frame["blocked_model_A_bill_ft"].sum()
        ),
        "modelled_pv_boilers_bill_saving_ft": float(
            modelled_frame["bill_saving_with_blocked_model_ft"].sum()
        ),
        "mean_bill_saving_modelled_pv_boilers_ft": (
            float(modelled_frame["bill_saving_with_blocked_model_ft"].mean())
            if len(modelled_frame) else None
        ),
        "share_modelled_pv_boilers_with_lower_bill_pct": (
            float(
                100.0
                * (modelled_frame["bill_saving_with_blocked_model_ft"] > 0).mean()
            )
            if len(modelled_frame) else None
        ),
        "tariff_logic": (
            "scaled measured boiler=B; blocked PV+HSS model=A; "
            "non-PV/non-modelled boiler=B; no BESS"
        ),
    }
    return frame, summary


def save_measurements(
    measurements: pd.DataFrame,
    source: Path,
    output: Path,
    backup: Path | None,
    overwrite: bool,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    same_file = source.resolve() == output.resolve()
    if output.exists() and not overwrite and not same_file:
        raise FileExistsError(
            f"A kimeneti meresi fajl mar letezik: {output.resolve()}"
        )
    if same_file and backup is not None and not backup.exists():
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, backup)
        print(f"[INFO] Biztonsagi masolat: {backup}")

    temporary = output.with_name(f"{output.stem}.__boiler_scaling_tmp{output.suffix}")
    measurements.to_csv(temporary, index=True)
    temporary.replace(output)


def run(
    sim_yaml: Path = SIM_YAML,
    users_dir: Path = USERS_DIR,
    source_measurements_csv: Path = SOURCE_MEASUREMENTS_CSV,
    output_measurements_csv: Path = OUTPUT_MEASUREMENTS_CSV,
    dhw_profiles_csv: Path = DHW_PROFILES_CSV,
    out_dir: Path = OUT_DIR,
    backup_measurements_csv: Path | None = BACKUP_MEASUREMENTS_CSV,
    max_users: int | None = MAX_USERS,
    scale_households: list[str] | None = SCALE_HOUSEHOLDS,
    overwrite: bool = OVERWRITE_OUTPUT,
) -> dict:
    sim_yaml = Path(sim_yaml)
    users_dir = Path(users_dir)
    source_measurements_csv = Path(source_measurements_csv)
    output_measurements_csv = Path(output_measurements_csv)
    dhw_profiles_csv = Path(dhw_profiles_csv)
    out_dir = Path(out_dir)
    backup = Path(backup_measurements_csv) if backup_measurements_csv else None

    measurements = _read_indexed_csv(source_measurements_csv)
    dhw_profiles = _read_indexed_csv(dhw_profiles_csv)
    households = load_households(
        sim_yaml, users_dir, measurements, dhw_profiles, max_users
    )
    missing_pv = [
        f"{h.key}: {h.pv_profile}"
        for h in households
        if h.pv_profile and h.pv_profile not in measurements.columns
    ]
    if missing_pv:
        raise RuntimeError(
            "A YAML-ban hivatkozott PV-profilok hianyoznak a forras CSV-bol; "
            "a v3-at nem irom felul. Hianyzo profilok: "
            + ", ".join(missing_pv)
        )
    if output_measurements_csv.exists() and (
        source_measurements_csv.resolve() != output_measurements_csv.resolve()
    ):
        existing_columns = set(pd.read_csv(output_measurements_csv, nrows=0).columns)
        dropped_pv = sorted(
            col for col in existing_columns - set(measurements.columns)
            if col.startswith("pv_v3_")
        )
        if dropped_pv:
            raise RuntimeError(
                "A forras CSV nem tartalmazza a kimeneti v3-ban mar meglevo "
                "uj PV-oszlopokat; a v3-at nem irom felul: "
                + ", ".join(dropped_pv)
            )
    scaled_measurements, scaling_report, _ = scale_measured_boiler_profiles(
        households, measurements, scale_households
    )
    bill_frame, summary = compare_bills(households, scaling_report)

    save_measurements(
        scaled_measurements,
        source_measurements_csv,
        output_measurements_csv,
        backup,
        overwrite,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    scaling_report.to_csv(out_dir / "boiler_profile_scaling_report.csv", index=False)
    bill_frame.to_csv(out_dir / "boiler_bill_comparison_per_household.csv", index=False)
    summary.update(
        {
            "source_measurements_csv": str(source_measurements_csv),
            "output_measurements_csv": str(output_measurements_csv),
            "n_scaled_boiler_profiles": int(
                (scaling_report["scaling_status"] == "SCALED").sum()
            ),
        }
    )
    (out_dir / "boiler_bill_comparison_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    pd.DataFrame([summary]).to_csv(
        out_dir / "boiler_bill_comparison_summary.csv", index=False
    )

    print("\n========== BOJLERPROFIL-SKALAZAS ES SZAMLA ==========")
    print(f"Feldolgozott haztartasok: {summary['n_households']}")
    print(f"Skalazott mert bojlerprofilok: {summary['n_scaled_boiler_profiles']}")
    print(f"Tiltasos A tarifas PV+bojler modellek: {summary['n_modelled_pv_boilers']}")
    print(
        "6 oras 40 C kovetelmenyt teljesito tiltott modellek: "
        f"{summary['n_modelled_boilers_passing_6h_40']}"
    )
    print(
        "Skalazott mert, B tarifas osszes szamla: "
        f"{summary['scaled_measured_B_total_bill_ft']:,.0f} Ft"
    )
    print(
        "Tiltasos modellezett, A tarifas osszes szamla: "
        f"{summary['blocked_model_A_total_bill_ft']:,.0f} Ft"
    )
    print(
        "Megtakartitas (pozitiv = tiltasos modell kedvezobb): "
        f"{summary['total_bill_saving_with_blocked_model_ft']:,.0f} Ft"
    )
    print("\nCsak a tenylegesen modellezett PV+bojler haztartasok:")
    print(
        "  Skalazott mert B tarifas szamla: "
        f"{summary['scaled_measured_B_modelled_pv_boilers_bill_ft']:,.0f} Ft"
    )
    print(
        "  Tiltasos modellezett A tarifas szamla: "
        f"{summary['blocked_model_A_modelled_pv_boilers_bill_ft']:,.0f} Ft"
    )
    print(
        "  Megtakartitas: "
        f"{summary['modelled_pv_boilers_bill_saving_ft']:,.0f} Ft"
    )
    print(f"Modositott meresi CSV: {output_measurements_csv}")
    print(f"Riportmappa: {out_dir}")
    print("=====================================================\n")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim", type=Path, default=SIM_YAML)
    parser.add_argument("--users-dir", type=Path, default=USERS_DIR)
    parser.add_argument("--source-measurements", type=Path, default=SOURCE_MEASUREMENTS_CSV)
    parser.add_argument("--output-measurements", type=Path, default=OUTPUT_MEASUREMENTS_CSV)
    parser.add_argument("--dhw-profiles", type=Path, default=DHW_PROFILES_CSV)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--backup", type=Path, default=BACKUP_MEASUREMENTS_CSV)
    parser.add_argument("--max-users", type=int, default=MAX_USERS)
    parser.add_argument("--households", nargs="*", default=SCALE_HOUSEHOLDS)
    parser.add_argument("--no-overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(
        sim_yaml=args.sim,
        users_dir=args.users_dir,
        source_measurements_csv=args.source_measurements,
        output_measurements_csv=args.output_measurements,
        dhw_profiles_csv=args.dhw_profiles,
        out_dir=args.out_dir,
        backup_measurements_csv=args.backup,
        max_users=args.max_users,
        scale_households=args.households,
        overwrite=not args.no_overwrite,
    )


if __name__ == "__main__":
    main()
