"""A teljes 3d-I eset futtatása IDE-ből vagy terminálból."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml

# Terminálból indítva is tegye elérhetővé a projekt gyökerében lévő
# OptimizedIndividualScenarios csomagot (Windows és Linux alatt is).
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from OptimizedIndividualScenarios.individual_opt_bess_boiler import (
    individual_opt_bess_boiler,
)

DT = 0.25
EXPECTED_STEPS = 35040

# ---------------------------------------------------------------------------
# FUTTATÁSI BEÁLLÍTÁSOK
# ---------------------------------------------------------------------------
INPUT_DIR = Path("/home/cloud/PycharmData/input_attila")

SIM_YAML = INPUT_DIR / "simulation_config_disaggregated_with_userlist.yaml"
PROFILES_CSV = INPUT_DIR / "measurements_disaggregated_v3.csv"
DHW_PROFILES_CSV = INPUT_DIR / "dhw_v2.csv"
OUT_DIR = SCRIPT_DIR / "results_3d_I"

MAX_USERS = 105
ONLY_USER: str | None = None
PV_RATIO = 1.0

BESS_SHARE_PCT = 100.0
# None: százalékos, sorrend szerinti kiosztás.
# Lista: kizárólag a felsorolt háztartások kapnak BESS-t.
# Üres lista: senki sem kap BESS-t.
BESS_HOUSEHOLDS: list[str] | None = None

# A felsorolt PV-s háztartásoknál a modellezett bojler helyett
# a YAML ut.profile mért villamos profilja működik B tarifán.
MEASURED_BOILER_HOUSEHOLDS: list[str] = ["load_0420144888439778"]
# Példa:
# MEASURED_BOILER_HOUSEHOLDS = ["load_0420144653416475"]

BESS_DEFAULT_SIZE_KWH: float | None = None
RUN_LP = False
SOLVER = "gurobi"
SOLVER_MSG = False

# A 40 °C a használati meleg víz energiaszükségletének számításához kell.
# A sávhatárnál elérendő tárolóhőmérséklet 50 °C.
BOILER_T_IN_C = 10.0
BOILER_T_ABS_MIN_C = 10.0
BOILER_T_COMFORT_C = 40.0
BOILER_T_SETPOINT_MIN_C = 50.0
BOILER_T_MAX_C = 65.0
BOILER_T_INITIAL_C = 50.0
BOILER_T_ENV_C = 20.0


@dataclass
class Household:
    key: str
    name: str
    p_pv: np.ndarray
    p_ue: np.ndarray
    p_dhw: np.ndarray
    p_boiler_fixed: np.ndarray
    hss: dict
    bess: dict


def _profile(
    frame: pd.DataFrame,
    name,
    dt: float,
    scale: float = 1.0,
) -> np.ndarray:
    if name in (None, ""):
        return np.zeros(EXPECTED_STEPS)

    if str(name) not in frame.columns:
        raise KeyError(
            "A YAML-ban hivatkozott profil hiányzik "
            f"a measurements CSV-ből: {name}"
        )

    energy = np.maximum(
        frame[str(name)].to_numpy(dtype=float).ravel(),
        0.0,
    )

    if energy.size != EXPECTED_STEPS:
        raise ValueError(
            f"A(z) {name} profil hossza {energy.size}; "
            f"{EXPECTED_STEPS} szükséges."
        )

    return energy * float(scale) / dt


def _find_yaml(
    roots: Iterable[Path],
    key: str,
) -> Path | None:
    for root in roots:
        for filename in (key, f"{key}.yaml"):
            path = root / filename
            if path.exists():
                return path

    return None


def load_households(
    sim_yaml,
    profiles_csv,
    dhw_profiles_csv,
    *,
    pv_ratio=1.0,
    dt=DT,
    dhw_t_in_c=BOILER_T_IN_C,
    dhw_t_out_c=BOILER_T_COMFORT_C,
):
    sim_path = Path(sim_yaml)
    sim = yaml.safe_load(
        sim_path.read_text(encoding="utf-8")
    ) or {}

    keys = [
        str(k)
        for k in sim.get("users_list", [])
        if str(k).strip().lower()
        not in {"battery", "bess", "community"}
    ]

    profiles = pd.read_csv(
        profiles_csv,
        index_col=0,
    )
    profiles.columns = profiles.columns.astype(str)

    dhw = pd.read_csv(
        dhw_profiles_csv,
        index_col=0,
    )
    dhw.columns = dhw.columns.astype(str)

    roots = [sim_path.parent / "users_v3"]
    households = []

    for key in keys:
        yaml_path = _find_yaml(
            roots,
            key,
        )

        if yaml_path is None:
            print(
                f"[FIGYELEM] Nem található user YAML, "
                f"kihagyva: {key}"
            )
            continue

        cfg = yaml.safe_load(
            yaml_path.read_text(encoding="utf-8")
        ) or {}

        units = cfg.get("units") or {}
        hss = dict(units.get("hss") or {})
        bess = dict(units.get("bess") or {})
        name = str(
            (units.get("name") or {}).get("name", key)
        )

        p_pv = _profile(
            profiles,
            (units.get("pv") or {}).get("profile"),
            dt,
            pv_ratio,
        )
        p_ue = _profile(
            profiles,
            (units.get("ue") or {}).get("profile"),
            dt,
        )
        p_fixed = _profile(
            profiles,
            (units.get("ut") or {}).get("profile"),
            dt,
        )

        dhw_name = hss.get("profile")

        if (
            dhw_name not in (None, "")
            and str(dhw_name) not in dhw.columns
        ):
            raise KeyError(
                "A YAML-ban hivatkozott DHW-profil "
                f"hiányzik: {key}: {dhw_name}"
            )

        if (
            dhw_name is not None
            and str(dhw_name) in dhw.columns
        ):
            litres = np.maximum(
                dhw[str(dhw_name)]
                .to_numpy(dtype=float)
                .ravel(),
                0.0,
            )

            if litres.size != EXPECTED_STEPS:
                raise ValueError(
                    f"A(z) {dhw_name} DHW-profil "
                    "hossza hibás."
                )

            # Liter/15 perc profilból a 10 -> 40 °C
            # melegítés hőteljesítménye.
            delta_t = max(
                float(dhw_t_out_c)
                - float(dhw_t_in_c),
                0.0,
            )
            p_dhw = (
                litres
                * (4186.0 / 3_600_000.0)
                * delta_t
                / dt
            )
        else:
            p_dhw = np.zeros(EXPECTED_STEPS)

        households.append(
            Household(
                key,
                name,
                p_pv,
                p_ue,
                p_dhw,
                p_fixed,
                hss,
                bess,
            )
        )

    if not households:
        raise RuntimeError(
            "Nem sikerült egyetlen háztartást "
            "sem beolvasni."
        )

    return households


def _selected_bess_keys(
    households,
    share_pct: float,
) -> set[str]:
    pv_users = [
        h
        for h in households
        if h.p_pv.sum() > 1e-9
    ]

    count = max(
        0,
        min(
            len(pv_users),
            int(
                round(
                    len(pv_users)
                    * share_pct
                    / 100.0
                )
            ),
        ),
    )

    return {
        h.key
        for h in pv_users[:count]
    }


def _explicit_bess_keys(
    households,
    selected_codes,
) -> set[str]:
    requested = [
        str(code).strip()
        for code in selected_codes
    ]

    if any(not code for code in requested):
        raise ValueError(
            "A BESS-háztartások listája "
            "nem tartalmazhat üres kódot."
        )

    resolved = set()
    missing = []
    ambiguous = []

    for code in requested:
        matches = [
            h
            for h in households
            if h.key == code
            or h.name == code
        ]

        if not matches:
            missing.append(code)
        elif len(matches) > 1:
            ambiguous.append(code)
        else:
            resolved.add(
                matches[0].key
            )

    if missing:
        raise ValueError(
            "Ismeretlen BESS-háztartáskód(ok): "
            + ", ".join(missing)
        )

    if ambiguous:
        raise ValueError(
            "Nem egyértelmű BESS-háztartáskód(ok): "
            + ", ".join(ambiguous)
        )

    without_pv = [
        h.key
        for h in households
        if (
            h.key in resolved
            and h.p_pv.sum() <= 1e-9
        )
    ]

    if without_pv:
        raise ValueError(
            "BESS csak PV-s háztartáshoz "
            "rendelhető; nem PV-s kód(ok): "
            + ", ".join(without_pv)
        )

    return resolved


def _safe_name(value: str) -> str:
    return "".join(
        c
        if c.isalnum()
        or c in "-_."
        else "_"
        for c in value
    )


def _measured_boiler_keys(
    households,
    codes,
):
    selected = {
        str(code).strip()
        for code in (codes or [])
    }

    if "" in selected:
        raise ValueError(
            "A MEASURED_BOILER_HOUSEHOLDS "
            "nem tartalmazhat üres kódot."
        )

    unknown = selected - {
        h.key
        for h in households
    }

    if unknown:
        raise ValueError(
            "Ismeretlen mért bojleres "
            "háztartáskód(ok): "
            + ", ".join(sorted(unknown))
        )

    for h in households:
        if h.key not in selected:
            continue

        modelable = (
            h.p_pv.sum() > 1e-9
            and h.p_dhw.sum() > 1e-9
            and float(
                h.hss.get(
                    "size_elh",
                    0,
                )
            ) > 1e-9
            and float(
                h.hss.get(
                    "vol_hss_water",
                    0,
                )
            ) > 1e-9
        )

        if not modelable:
            raise ValueError(
                f"{h.key}: nincs modellezhető "
                "bojler, nem választható át "
                "mért profilra."
            )

        if h.p_boiler_fixed.sum() <= 1e-9:
            raise ValueError(
                f"{h.key}: hiányzik vagy nulla "
                "a mért bojler villamos profilja."
            )

    return selected


def run(
    sim_yaml,
    profiles_csv,
    dhw_profiles_csv,
    out_dir,
    *,
    max_users=105,
    only_user=None,
    pv_ratio=1.0,
    bess_share_pct=40.0,
    bess_households=None,
    measured_boiler_households=(
        MEASURED_BOILER_HOUSEHOLDS
    ),
    bess_default_size_kwh=None,
    force_bess_for_only_user=False,
    run_lp=False,
    solver="gurobi",
    solver_msg=False,
    dt=DT,
    boiler_t_in_c=BOILER_T_IN_C,
    boiler_t_abs_min_c=BOILER_T_ABS_MIN_C,
    boiler_t_comfort_c=BOILER_T_COMFORT_C,
    boiler_t_setpoint_min_c=(
        BOILER_T_SETPOINT_MIN_C
    ),
    boiler_t_max_c=BOILER_T_MAX_C,
    boiler_t_initial_c=BOILER_T_INITIAL_C,
    boiler_t_env_c=BOILER_T_ENV_C,
):
    all_households = load_households(
        sim_yaml,
        profiles_csv,
        dhw_profiles_csv,
        pv_ratio=pv_ratio,
        dt=dt,
        dhw_t_in_c=boiler_t_in_c,
        dhw_t_out_c=boiler_t_comfort_c,
    )

    measured_boiler_keys = (
        _measured_boiler_keys(
            all_households,
            measured_boiler_households,
        )
    )

    explicit_bess = (
        bess_households is not None
    )

    if explicit_bess:
        bess_keys = _explicit_bess_keys(
            all_households,
            bess_households,
        )

    if only_user is not None:
        if not explicit_bess:
            bess_keys = _selected_bess_keys(
                all_households,
                float(bess_share_pct),
            )

        needle = str(only_user).strip()
        households = [
            h
            for h in all_households
            if (
                h.key == needle
                or h.name == needle
            )
        ]

        if not households:
            raise RuntimeError(
                "A kiválasztott háztartás "
                f"nem található: {needle}"
            )

        if len(households) > 1:
            raise RuntimeError(
                "A kiválasztás nem "
                f"egyértelmű: {needle}"
            )

        if (
            force_bess_for_only_user
            and households[0].p_pv.sum()
            > 1e-9
        ):
            bess_keys.add(
                households[0].key
            )

    else:
        households = all_households[
            :int(max_users)
        ]

        if not explicit_bess:
            bess_keys = (
                _selected_bess_keys(
                    households,
                    float(bess_share_pct),
                )
            )

    configured_sizes = [
        float(
            h.bess.get(
                "bess_size",
                0.0,
            )
        )
        for h in all_households
        if float(
            h.bess.get(
                "bess_size",
                0.0,
            )
        ) > 0
    ]

    fallback_size = (
        float(bess_default_size_kwh)
        if bess_default_size_kwh
        is not None
        else (
            float(
                np.median(configured_sizes)
            )
            if configured_sizes
            else None
        )
    )

    out = Path(out_dir)
    out.mkdir(
        parents=True,
        exist_ok=True,
    )
    rows = []

    for number, h in enumerate(
        households,
        1,
    ):
        print(
            "[INFO] Optimalizálás "
            f"{number}/{len(households)}: "
            f"{h.name}"
        )

        has_pv = (
            h.p_pv.sum() > 1e-9
        )

        use_hss = bool(
            h.key not in measured_boiler_keys
            and has_pv
            and h.p_dhw.sum() > 1e-9
            and float(
                h.hss.get(
                    "size_elh",
                    0.0,
                )
            ) > 1e-9
            and float(
                h.hss.get(
                    "vol_hss_water",
                    0.0,
                )
            ) > 1e-9
        )

        p_dhw = (
            h.p_dhw
            if use_hss
            else np.zeros_like(
                h.p_dhw
            )
        )

        p_fixed = (
            np.zeros_like(
                h.p_boiler_fixed
            )
            if use_hss
            else h.p_boiler_fixed
        )

        use_bess = (
            h.key in bess_keys
        )

        bess_size = (
            float(
                h.bess.get(
                    "bess_size",
                    0.0,
                )
            )
            if use_bess
            else 0.0
        )

        if (
            use_bess
            and bess_size <= 1e-9
        ):
            if fallback_size is None:
                raise RuntimeError(
                    f"{h.name}: BESS-re kijelölt "
                    "user, de nincs bess_size. "
                    "Adj meg "
                    "BESS_DEFAULT_SIZE_KWH értéket."
                )

            bess_size = fallback_size

        soc_min = float(
            h.bess.get(
                "soc_bess_min",
                0.10,
            )
        )
        soc_max = float(
            h.bess.get(
                "soc_bess_max",
                0.90,
            )
        )
        soc_init = min(
            max(
                float(
                    h.bess.get(
                        "soc_bess_init",
                        0.50,
                    )
                ),
                soc_min,
            ),
            soc_max,
        )

        result = (
            individual_opt_bess_boiler(
                h.p_pv,
                h.p_ue,
                p_dhw,
                dt=dt,
                p_el_heater_fixed=(
                    p_fixed
                ),
                size_elh=(
                    float(
                        h.hss.get(
                            "size_elh",
                            0.0,
                        )
                    )
                    if use_hss
                    else 0.0
                ),
                vol_hss_water=(
                    float(
                        h.hss.get(
                            "vol_hss_water",
                            0.0,
                        )
                    )
                    if use_hss
                    else 0.0
                ),
                T_env=float(
                    boiler_t_env_c
                ),
                T_max=float(
                    boiler_t_max_c
                ),
                T_abs_min=float(
                    boiler_t_abs_min_c
                ),
                T_in=float(
                    boiler_t_in_c
                ),
                T_comfort=float(
                    boiler_t_comfort_c
                ),
                T_setpoint_min=float(
                    boiler_t_setpoint_min_c
                ),
                T_initial=float(
                    boiler_t_initial_c
                ),
                morning_block=(
                    6.0,
                    9.0,
                ),
                evening_block=(
                    17.0,
                    22.0,
                ),
                a_hss=float(
                    h.hss.get(
                        "a_hss",
                        0.01275,
                    )
                ),
                eta_elh=float(
                    h.hss.get(
                        "eta_elh",
                        0.95,
                    )
                ),
                size_bess=bess_size,
                eta_bess_in=float(
                    h.bess.get(
                        "eta_bess_in",
                        0.98,
                    )
                ),
                eta_bess_out=float(
                    h.bess.get(
                        "eta_bess_out",
                        0.96,
                    )
                ),
                eta_bess_stor=float(
                    h.bess.get(
                        "eta_bess_stor",
                        0.995,
                    )
                ),
                soc_bess_min=soc_min,
                soc_bess_max=soc_max,
                soc_bess_init=soc_init,
                t_bess_min=float(
                    h.bess.get(
                        "t_bess_min",
                        2.0,
                    )
                ),
                run_lp=run_lp,
                solver=solver,
                msg=solver_msg,
            )
        )

        n = len(h.p_pv)

        p_grid_import_a = (
            result["p_grid_base"]
            + result["p_grid_bess"]
            + (
                result["p_grid_boiler"]
                if use_hss
                else 0.0
            )
        )

        p_grid_import_b = (
            result["p_grid_boiler"]
            if p_fixed.sum() > 1e-9
            else np.zeros(n)
        )

        ts = pd.DataFrame({
            "p_pv_kw": h.p_pv,
            "p_base_load_kw": h.p_ue,
            "p_dhw_kwth": p_dhw,
            "p_fixed_boiler_kw": p_fixed,
            "p_pv_base_kw": result[
                "p_pv_base"
            ],
            "p_pv_boiler_kw": result[
                "p_pv_boiler"
            ],
            "p_pv_bess_kw": result[
                "p_pv_bess"
            ],
            "p_pv_export_kw": result[
                "p_pv_export"
            ],
            "p_grid_base_kw": result[
                "p_grid_base"
            ],
            "p_grid_boiler_kw": result[
                "p_grid_boiler"
            ],
            "p_grid_bess_kw": result[
                "p_grid_bess"
            ],
            "p_grid_import_kw": result[
                "p_grid_import"
            ],
            "p_grid_import_a_kw": (
                p_grid_import_a
            ),
            "p_grid_import_b_kw": (
                p_grid_import_b
            ),
            "p_bess_base_kw": result[
                "p_bess_base"
            ],
            "p_bess_boiler_kw": result[
                "p_bess_boiler"
            ],
            "p_bess_charge_kw": result[
                "p_bess_charge"
            ],
            "p_bess_discharge_kw": result[
                "p_bess_discharge"
            ],
            "e_bess_kwh": result[
                "e_bess"
            ][:n],
            "d_bess_charge": result[
                "d_bess_charge"
            ],
            "d_bess_discharge": result[
                "d_bess_discharge"
            ],
            "p_elh_kw": result[
                "p_elh"
            ],
            "p_hss_in_kwth": result[
                "p_hss_in"
            ],
            "p_hss_out_kwth": result[
                "p_hss_out"
            ],
            "e_hss_kwhth": result[
                "e_hss"
            ][:n],
            "t_hss_C": result[
                "t_hss"
            ][:n],
            "d_cl": result[
                "d_cl"
            ],
            "d_cl_start": result[
                "d_cl_start"
            ],
            "d_heat": result[
                "d_heat"
            ],
            "d_below_comfort": result[
                "d_below_comfort"
            ],
            "below_comfort_actual": (
                result["below_comfort"]
            ),
            "heating_blocked": result[
                "heating_blocked"
            ],
            "boiler_target_C": result[
                "boiler_target_C"
            ],
        })

        ts.to_csv(
            out
            / (
                f"timeseries_"
                f"{_safe_name(h.name)}.csv"
            ),
            index=False,
        )

        rows.append({
            "household_key": h.key,
            "household": h.name,
            "has_pv": int(has_pv),
            "hss_optimized": int(
                use_hss
            ),
            "has_bess": int(
                use_bess
            ),
            "boiler_tariff": result[
                "boiler_tariff"
            ],
            "bess_size_kwh": (
                bess_size
            ),
            "pv_generation_kwh": (
                float(
                    h.p_pv.sum()
                    * dt
                )
            ),
            "base_load_kwh": (
                float(
                    h.p_ue.sum()
                    * dt
                )
            ),
            "boiler_electric_kwh": (
                float(
                    (
                        result["p_elh"]
                        + p_fixed
                    ).sum()
                    * dt
                )
            ),
            "pv_to_base_kwh": (
                float(
                    result[
                        "p_pv_base"
                    ].sum()
                    * dt
                )
            ),
            "pv_to_boiler_kwh": (
                float(
                    result[
                        "p_pv_boiler"
                    ].sum()
                    * dt
                )
            ),
            "pv_to_bess_kwh": (
                float(
                    result[
                        "p_pv_bess"
                    ].sum()
                    * dt
                )
            ),
            "grid_to_bess_kwh": (
                float(
                    result[
                        "p_grid_bess"
                    ].sum()
                    * dt
                )
            ),
            "bess_to_base_kwh": (
                float(
                    result[
                        "p_bess_base"
                    ].sum()
                    * dt
                )
            ),
            "bess_to_boiler_kwh": (
                float(
                    result[
                        "p_bess_boiler"
                    ].sum()
                    * dt
                )
            ),
            "grid_import_kwh": result[
                "grid_import_total_kwh"
            ],
            "grid_import_a_kwh": result[
                "grid_import_a_kwh"
            ],
            "grid_import_b_kwh": result[
                "grid_import_b_kwh"
            ],
            "grid_export_kwh": result[
                "grid_export_kwh"
            ],
            "import_cost_a_ft": result[
                "import_cost_a_ft"
            ],
            "import_cost_b_ft": result[
                "import_cost_b_ft"
            ],
            "import_cost_ft": result[
                "import_cost_ft"
            ],
            "export_revenue_ft": result[
                "export_revenue_ft"
            ],
            "bill_ft": result[
                "bill_ft"
            ],
            "status": result[
                "status"
            ],
            "boiler_control_status": (
                result[
                    "boiler_control_status"
                ]
            ),
            "longest_continuous_below_40_h": (
                result[
                    "longest_continuous_below_40_h"
                ]
            ),
            "dhw_while_below_40_steps": (
                result[
                    "dhw_while_below_40_steps"
                ]
            ),
            "minimum_dhw_energy_margin_kwhth": (
                result[
                    "minimum_dhw_energy_margin_kwhth"
                ]
            ),
            "dhw_energy_shortfall_steps": (
                result[
                    "dhw_energy_shortfall_steps"
                ]
            ),
            "optimized_setpoint_min_C": (
                result[
                    "optimized_setpoint_min_C"
                ]
            ),
            "optimized_setpoint_max_C": (
                result[
                    "optimized_setpoint_max_C"
                ]
            ),
        })

    summary = pd.DataFrame(
        rows
    )

    summary.to_csv(
        out / "household_summary.csv",
        index=False,
    )

    totals = {
        "case": "3d-I",
        "boiler_tariff": (
            "A_dynamic_B_measured"
        ),
        "n_households": len(
            summary
        ),
        "bess_assignment_mode": (
            "explicit"
            if explicit_bess
            else "percentage"
        ),
        "bess_household_keys": sorted(
            bess_keys
        ),
        "measured_boiler_household_keys": (
            sorted(
                measured_boiler_keys
                & {
                    h.key
                    for h in households
                }
            )
        ),
        "total_grid_import_kwh": (
            float(
                summary.grid_import_kwh.sum()
            )
        ),
        "total_grid_export_kwh": (
            float(
                summary.grid_export_kwh.sum()
            )
        ),
        "total_bill_ft": float(
            summary.bill_ft.sum()
        ),
        "out_dir": str(out),
    }

    (
        out / "summary.json"
    ).write_text(
        json.dumps(
            totals,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return totals


def main():
    result = run(
        SIM_YAML,
        PROFILES_CSV,
        DHW_PROFILES_CSV,
        OUT_DIR,
        max_users=MAX_USERS,
        only_user=ONLY_USER,
        pv_ratio=PV_RATIO,
        bess_share_pct=(
            BESS_SHARE_PCT
        ),
        bess_households=(
            BESS_HOUSEHOLDS
        ),
        measured_boiler_households=(
            MEASURED_BOILER_HOUSEHOLDS
        ),
        bess_default_size_kwh=(
            BESS_DEFAULT_SIZE_KWH
        ),
        force_bess_for_only_user=False,
        run_lp=RUN_LP,
        solver=SOLVER,
        solver_msg=SOLVER_MSG,
    )

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()