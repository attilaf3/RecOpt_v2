"""A 4d-K-E / 4d-K-C közösségi optimalizálás Linuxos futtatója."""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from optimize_community import optimize_community, save_community_results


DT = 0.25
EXPECTED_STEPS = 35040
LOW_TARIFF_LIMIT_KWH = 2523.0

MONTH_NAMES = (
    "01_january", "02_february", "03_march", "04_april",
    "05_may", "06_june", "07_july", "08_august",
    "09_september", "10_october", "11_november", "12_december",
)

MONTH_DAYS_2021 = (
    31, 28, 31, 30, 31, 30,
    31, 31, 30, 31, 30, 31,
)

# A 40 °C a használati meleg víz energiaszükségletének számításához kell.
# Az optimalizált tartály célja a sávhatároknál legalább 50 °C.
BOILER_T_IN_C = 10.0
BOILER_T_ABS_MIN_C = 10.0
BOILER_T_COMFORT_C = 40.0
BOILER_T_SETPOINT_MIN_C = 50.0
BOILER_T_MAX_C = 65.0
BOILER_T_INITIAL_C = 50.0
BOILER_T_ENV_C = 20.0

INPUT_DIR = Path("/home/cloud/PycharmData/input_attila")
SIM_YAML = INPUT_DIR / "simulation_config_disaggregated_with_userlist.yaml"
PROFILES_CSV = INPUT_DIR / "measurements_disaggregated_v3.csv"
DHW_PROFILES_CSV = INPUT_DIR / "dhw_v2.csv"

MAX_USERS = 105
OBJECTIVE = "bill"             # "bill" = 4d-K-C; "grid" = 4d-K-E
SHARING_MODE = "proportional"
PV_RATIO = 1.0
INCLUDE_BESS = True
BESS_SHARE_PCT = 100.0
INCLUDE_BOILER_MODEL = True
AGGREGATE_PASSIVE = False

# None: a PV-s háztartások százalékos, users_list szerinti kiválasztása.
# Lista: a megadott háztartások kapnak BESS-t.
# Üres lista: senki sem kap BESS-t.
BESS_HOUSEHOLDS: list[str] | None = None

# Ezeknél a háztartásoknál a bojlert nem modellezzük.
# Helyette a YAML-ban megadott ut.profile mért villamos profilja kerül
# a B tarifás körbe.
#
# Példa:
# MEASURED_BOILER_HOUSEHOLDS = [
#     "load_0420144653416475",
#     "load_0420144888235070",
# ]
MEASURED_BOILER_HOUSEHOLDS: list[str] = ["load_0420144888439778"]

BESS_DEFAULT_SIZE_KWH: float | None = None

RUN_LP = False
SOLVER = "gurobi"
MSG = True
SAVE_USER_TIMESERIES = True

MEMORY_LOG_INTERVAL_SECONDS = 30.0
RUN_LOG_PATH = Path("community_run.log")


def _gib(value):
    return float(value) / (1024.0 ** 3)


def _read_proc_rss_bytes(pid):
    try:
        lines = Path(f"/proc/{pid}/status").read_text(
            encoding="utf-8"
        ).splitlines()

        for line in lines:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024

    except (
        FileNotFoundError,
        PermissionError,
        ProcessLookupError,
        ValueError,
    ):
        pass

    return 0


def _process_tree_rss_bytes():
    """Az aktuális Python-folyamat és gyermekfolyamatainak együttes RSS-e."""
    root = os.getpid()
    pending = [root]
    visited = set()
    total = 0

    while pending:
        pid = pending.pop()

        if pid in visited:
            continue

        visited.add(pid)
        total += _read_proc_rss_bytes(pid)

        try:
            task_dir = Path(f"/proc/{pid}/task")

            for child_file in task_dir.glob("*/children"):
                text = child_file.read_text(encoding="utf-8").strip()

                if text:
                    pending.extend(int(item) for item in text.split())

        except (
            FileNotFoundError,
            PermissionError,
            ProcessLookupError,
            ValueError,
        ):
            pass

    return total


def _system_memory_bytes():
    try:
        values = {}

        for line in Path("/proc/meminfo").read_text(
            encoding="utf-8"
        ).splitlines():
            key, raw = line.split(":", 1)
            values[key] = int(raw.strip().split()[0]) * 1024

        return values.get("MemTotal", 0), values.get("MemAvailable", 0)

    except (FileNotFoundError, PermissionError, ValueError):
        return 0, 0


class RunMonitor:
    def __init__(self, log_path, interval_seconds=30.0):
        self.log_path = Path(log_path)
        self.interval_seconds = max(float(interval_seconds), 1.0)
        self.started = time.monotonic()
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.thread = None

    def _write(self, level, message):
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        elapsed = time.monotonic() - self.started
        line = (
            f"[{stamp}] [{level}] "
            f"[elapsed={elapsed / 60:.1f} min] {message}"
        )

        with self.lock:
            print(line, flush=True)
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
                handle.flush()

    def stage(self, message):
        process_rss = _process_tree_rss_bytes()
        total, available = _system_memory_bytes()

        memory = f"process_tree={_gib(process_rss):.2f} GiB"

        if total > 0:
            used = total - available
            memory += (
                f"; system={_gib(used):.2f}/{_gib(total):.2f} GiB"
                f" ({100.0 * used / total:.1f}%)"
                f"; available={_gib(available):.2f} GiB"
            )

        self._write("STAGE", f"{message} | {memory}")

    def _loop(self):
        while not self.stop_event.wait(self.interval_seconds):
            self.stage("Futásban")

    def start(self):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.write_text("", encoding="utf-8")
        self.stage("Futás elindult")

        self.thread = threading.Thread(
            target=self._loop,
            name="memory-monitor",
            daemon=True,
        )
        self.thread.start()

    def stop(self):
        self.stop_event.set()

        if self.thread is not None:
            self.thread.join(timeout=2.0)


@dataclass
class Household:
    key: str
    name: str
    p_pv: np.ndarray
    p_base: np.ndarray
    p_dhw: np.ndarray
    p_boiler_fixed: np.ndarray
    hss: dict
    bess: dict


def _profile(frame, column, *, dt=DT, scale=1.0):
    if column in (None, ""):
        return np.zeros(EXPECTED_STEPS)

    if str(column) not in frame.columns:
        raise KeyError(
            "A YAML-ban hivatkozott profil hiányzik "
            f"a measurements CSV-ből: {column}"
        )

    energy = np.maximum(
        frame[str(column)].to_numpy(dtype=float).ravel(),
        0.0,
    )

    if energy.size != EXPECTED_STEPS:
        raise ValueError(
            f"A(z) {column} profil hossza {energy.size}; "
            f"{EXPECTED_STEPS} szükséges."
        )

    return energy * float(scale) / dt


def _find_user_yaml(sim_path, key):
    roots = [sim_path.parent / "users_v3"]

    for root in roots:
        for filename in (
            str(key),
            f"{key}.yaml",
            f"{key}.yml",
        ):
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
    boiler_t_in_c=BOILER_T_IN_C,
    boiler_t_comfort_c=BOILER_T_COMFORT_C,
):
    sim_path = Path(sim_yaml)
    sim = yaml.safe_load(
        sim_path.read_text(encoding="utf-8")
    ) or {}

    keys = [
        str(key)
        for key in sim.get("users_list", [])
        if str(key).strip().lower()
        not in {"battery", "bess", "community"}
    ]

    profiles = pd.read_csv(profiles_csv, index_col=0)
    profiles.columns = profiles.columns.astype(str)

    dhw = pd.read_csv(dhw_profiles_csv, index_col=0)
    dhw.columns = dhw.columns.astype(str)

    households = []

    for key in keys:
        path = _find_user_yaml(sim_path, key)

        if path is None:
            print(
                f"[FIGYELEM] Nem található user YAML, kihagyva: {key}"
            )
            continue

        cfg = yaml.safe_load(
            path.read_text(encoding="utf-8")
        ) or {}

        units = cfg.get("units") or {}
        hss = dict(units.get("hss") or {})
        bess = dict(units.get("bess") or {})

        p_pv = _profile(
            profiles,
            (units.get("pv") or {}).get("profile"),
            scale=pv_ratio,
        )
        p_base = _profile(
            profiles,
            (units.get("ue") or {}).get("profile"),
        )
        p_fixed = _profile(
            profiles,
            (units.get("ut") or {}).get("profile"),
        )

        dhw_name = hss.get("profile")

        if (
            dhw_name not in (None, "")
            and str(dhw_name) not in dhw.columns
        ):
            raise KeyError(
                f"A YAML-ban hivatkozott DHW-profil hiányzik: "
                f"{key}: {dhw_name}"
            )

        if dhw_name is not None and str(dhw_name) in dhw.columns:
            litres = np.maximum(
                dhw[str(dhw_name)].to_numpy(dtype=float).ravel(),
                0.0,
            )

            if litres.size != EXPECTED_STEPS:
                raise ValueError(
                    f"A(z) {dhw_name} DHW-profil hossza hibás."
                )

            # A használati meleg víz 40 °C-os energiaszükséglete.
            delta_t = max(
                float(boiler_t_comfort_c)
                - float(boiler_t_in_c),
                0.0,
            )
            p_dhw = (
                litres
                * (4186.0 / 3_600_000.0)
                * delta_t
                / DT
            )
        else:
            p_dhw = np.zeros(EXPECTED_STEPS)

        name = str(
            (units.get("name") or {}).get("name", key)
        )

        households.append(
            Household(
                key,
                name,
                p_pv,
                p_base,
                p_dhw,
                p_fixed,
                hss,
                bess,
            )
        )

    if not households:
        raise RuntimeError(
            "Nem sikerült egyetlen háztartást sem beolvasni."
        )

    return households


def _resolve_bess(
    households,
    selected,
    share_pct,
    include_bess,
):
    if not include_bess:
        return set()

    if selected is None:
        candidates = [
            h for h in households
            if h.p_pv.sum() > 1e-9
        ]
        count = max(
            0,
            min(
                len(candidates),
                int(
                    round(
                        len(candidates)
                        * float(share_pct)
                        / 100
                    )
                ),
            ),
        )
        return {
            h.key
            for h in candidates[:count]
        }

    requested = [
        str(code).strip()
        for code in selected
    ]

    if any(not code for code in requested):
        raise ValueError(
            "A BESS_HOUSEHOLDS nem tartalmazhat üres kódot."
        )

    resolved = set()

    for code in requested:
        matches = [
            h for h in households
            if h.key == code or h.name == code
        ]

        if len(matches) != 1:
            raise ValueError(
                f"Ismeretlen vagy nem egyértelmű "
                f"BESS-háztartás: {code}"
            )

        if matches[0].p_pv.sum() <= 1e-9:
            raise ValueError(
                f"BESS csak PV-s háztartáshoz "
                f"rendelhető: {code}"
            )

        resolved.add(matches[0].key)

    return resolved


def _param(households, source, key, default):
    return np.asarray([
        float(
            getattr(h, source).get(key, default)
        )
        for h in households
    ])


def _measured_boiler_keys(households, codes):
    selected = set(codes or ())
    unknown = selected - {
        h.key for h in households
    }

    if unknown:
        raise ValueError(
            "Ismeretlen mért bojleres háztartáskód(ok): "
            + ", ".join(sorted(unknown))
        )

    for h in households:
        if h.key not in selected:
            continue

        modelable = (
            h.p_pv.sum() > 1e-9
            and h.p_dhw.sum() > 1e-9
            and float(
                h.hss.get("size_elh", 0)
            ) > 1e-9
            and float(
                h.hss.get("vol_hss_water", 0)
            ) > 1e-9
        )

        if not modelable:
            raise ValueError(
                f"{h.key}: nincs modellezhető bojler, "
                "nem választható át mért profilra."
            )

        if h.p_boiler_fixed.sum() <= 1e-9:
            raise ValueError(
                f"{h.key}: hiányzik vagy nulla "
                "a mért bojler villamos profilja."
            )

    return selected


def _month_slices():
    boundaries = np.cumsum(
        (0,)
        + tuple(
            day * int(round(24 / DT))
            for day in MONTH_DAYS_2021
        )
    )

    if int(boundaries[-1]) != EXPECTED_STEPS:
        raise RuntimeError(
            "A havi szeletek nem adják ki "
            "a teljes 2021-es évet."
        )

    return [
        slice(
            int(boundaries[i]),
            int(boundaries[i + 1]),
        )
        for i in range(12)
    ]


def _combine_monthly_results(monthly_results):
    """A 12 havi eredményből éves eredményt állít elő."""
    if len(monthly_results) != 12:
        raise ValueError(
            "Pontosan 12 havi eredmény szükséges."
        )

    names = list(
        monthly_results[0]["user_names"]
    )

    if any(
        list(result["user_names"]) != names
        for result in monthly_results
    ):
        raise ValueError(
            "A háztartások sorrendje megváltozott "
            "a hónapok között."
        )

    timeseries = {
        key: np.concatenate(
            [
                result["timeseries"][key]
                for result in monthly_results
            ],
            axis=0,
        )
        for key in monthly_results[0]["timeseries"]
    }

    pair_kwh = sum(
        (
            result["pair_kwh"]
            for result in monthly_results
        ),
        np.zeros_like(
            monthly_results[0]["pair_kwh"]
        ),
    )

    pair_payment = sum(
        (
            result["pair_payment_ft"]
            for result in monthly_results
        ),
        np.zeros_like(
            monthly_results[0]["pair_payment_ft"]
        ),
    )

    frames = [
        result["household_summary"].set_index(
            "household"
        )
        for result in monthly_results
    ]

    first = frames[0]

    sum_columns = [
        "pv_generation_kwh",
        "base_load_kwh",
        "boiler_electric_kwh",
        "total_load_kwh",
        "used_pv_kwh",
        "local_supply_kwh",
        "shared_in_kwh",
        "shared_out_kwh",
        "shared_to_boiler_kwh",
        "grid_import_a_kwh",
        "grid_import_b_kwh",
        "grid_import_kwh",
        "grid_export_kwh",
        "import_cost_a_ft",
        "import_cost_b_ft",
        "shared_purchase_cost_ft",
        "shared_revenue_ft",
        "export_revenue_ft",
        "bill_ft",
        "dhw_energy_shortfall_kwhth",
        "setpoint_shortfall_kwhth",
    ]

    rows = []

    for name in names:
        monthly = [
            frame.loc[name]
            for frame in frames
        ]

        row = {
            "household": name,
            "has_pv": int(
                first.loc[name, "has_pv"]
            ),
            "has_bess": int(
                first.loc[name, "has_bess"]
            ),
            "hss_optimized": int(
                first.loc[name, "hss_optimized"]
            ),
            "boiler_tariff": first.loc[
                name,
                "boiler_tariff",
            ],
        }

        for column in sum_columns:
            row[column] = float(
                sum(
                    float(item[column])
                    for item in monthly
                )
            )

        row["self_consumption_ratio"] = (
            row["used_pv_kwh"]
            / row["pv_generation_kwh"]
            if row["pv_generation_kwh"] > 1e-9
            else 0.0
        )

        row["self_sufficiency_ratio"] = (
            row["local_supply_kwh"]
            / row["total_load_kwh"]
            if row["total_load_kwh"] > 1e-9
            else 0.0
        )

        failures = [
            item["boiler_control_status"]
            for item in monthly
            if item["boiler_control_status"]
            != "PASS"
        ]

        row["boiler_control_status"] = (
            failures[0]
            if failures
            else "PASS"
        )

        row["a_low_remaining_kwh"] = float(
            monthly[-1][
                "a_low_remaining_kwh"
            ]
        )

        row["b_low_remaining_kwh"] = float(
            monthly[-1][
                "b_low_remaining_kwh"
            ]
        )

        row["status"] = (
            "Optimal"
            if all(
                item["status"] == "Optimal"
                for item in monthly
            )
            else monthly[-1]["status"]
        )

        rows.append(row)

    household_summary = pd.DataFrame(rows)
    first_summary = monthly_results[0]["summary"]

    sum_summary_keys = (
        "objective_value_solver",
        "optimizer_shared_kwh",
        "maximum_share_before_equal_quotas_kwh",
        "settled_shared_kwh",
        "equal_unused_quota_kwh",
        "total_grid_import_a_kwh",
        "total_grid_import_b_kwh",
        "total_grid_export_kwh",
        "total_shared_to_boiler_kwh",
        "total_bill_ft",
        "total_dhw_energy_shortfall_kwhth",
        "total_setpoint_shortfall_kwhth",
    )

    summary = dict(first_summary)
    summary.pop("month", None)
    summary.pop("month_index", None)

    for key in sum_summary_keys:
        summary[key] = float(
            sum(
                result["summary"].get(key, 0.0)
                for result in monthly_results
            )
        )

    summary.update({
        "status": (
            "Optimal"
            if all(
                result["summary"]["status"]
                == "Optimal"
                for result in monthly_results
            )
            else monthly_results[-1][
                "summary"
            ]["status"]
        ),
        "monthly_optimization_count": 12,
        "horizon": "2021_12_sequential_months",
        "state_carry_over": (
            "HSS_energy_BESS_energy_"
            "A_cap_B_cap"
        ),
        "settlement_differs_from_optimizer": any(
            result["summary"][
                "settlement_differs_from_optimizer"
            ]
            for result in monthly_results
        ),
    })

    summary["final_grid_interaction_kwh"] = (
        summary["total_grid_import_a_kwh"]
        + summary["total_grid_import_b_kwh"]
        + summary["total_grid_export_kwh"]
    )

    pv_total = float(
        household_summary[
            "pv_generation_kwh"
        ].sum()
    )

    load_total = float(
        household_summary[
            "total_load_kwh"
        ].sum()
    )

    summary["SCI"] = (
        float(
            household_summary[
                "used_pv_kwh"
            ].sum()
        )
        / pv_total
        if pv_total > 1e-9
        else 0.0
    )

    summary["SSI"] = (
        float(
            household_summary[
                "local_supply_kwh"
            ].sum()
        )
        / load_total
        if load_total > 1e-9
        else 0.0
    )

    summary["monthly_statuses"] = [
        result["summary"]["status"]
        for result in monthly_results
    ]

    return {
        "summary": summary,
        "household_summary": household_summary,
        "timeseries": timeseries,
        "pair_kwh": pair_kwh,
        "pair_payment_ft": pair_payment,
        "user_names": names,
        "state": monthly_results[-1]["state"],
    }


def run(
    sim_yaml=SIM_YAML,
    profiles_csv=PROFILES_CSV,
    dhw_profiles_csv=DHW_PROFILES_CSV,
    out_dir=None,
    *,
    max_users=MAX_USERS,
    objective=OBJECTIVE,
    sharing_mode=SHARING_MODE,
    pv_ratio=PV_RATIO,
    include_bess=INCLUDE_BESS,
    bess_share_pct=BESS_SHARE_PCT,
    include_boiler_model=INCLUDE_BOILER_MODEL,
    bess_households=BESS_HOUSEHOLDS,
    bess_default_size_kwh=BESS_DEFAULT_SIZE_KWH,
    measured_boiler_households=(
        MEASURED_BOILER_HOUSEHOLDS
    ),
    run_lp=RUN_LP,
    solver=SOLVER,
    msg=MSG,
    save_user_timeseries=SAVE_USER_TIMESERIES,
    aggregate_passive=AGGREGATE_PASSIVE,
    memory_log_interval=(
        MEMORY_LOG_INTERVAL_SECONDS
    ),
    run_log_path=RUN_LOG_PATH,
):
    monitor = RunMonitor(
        run_log_path,
        memory_log_interval,
    )
    monitor.start()

    try:
        monitor.stage(
            "Inputfájlok beolvasása"
        )

        all_households = load_households(
            sim_yaml,
            profiles_csv,
            dhw_profiles_csv,
            pv_ratio=pv_ratio,
        )

        measured_boiler_keys = (
            _measured_boiler_keys(
                all_households,
                measured_boiler_households,
            )
        )

        monitor.stage(
            "Inputok beolvasva: "
            f"{len(all_households)} "
            "érvényes háztartás"
        )

        selected_keys = _resolve_bess(
            all_households,
            bess_households,
            bess_share_pct,
            include_bess,
        )

        households = all_households[
            :int(max_users)
        ]

        selected_keys &= {
            h.key for h in households
        }

        configured = [
            float(
                h.bess.get(
                    "bess_size",
                    0,
                )
            )
            for h in all_households
            if float(
                h.bess.get(
                    "bess_size",
                    0,
                )
            ) > 0
        ]

        fallback = (
            float(bess_default_size_kwh)
            if bess_default_size_kwh
            is not None
            else (
                float(np.median(configured))
                if configured
                else None
            )
        )

        monitor.stage(
            "Eszközlisták és paraméterek "
            "összeállítása"
        )

        hss_enabled = np.asarray([
            bool(include_boiler_model)
            and h.key
            not in measured_boiler_keys
            and h.p_pv.sum() > 1e-9
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
            for h in households
        ])

        bess_enabled = np.asarray([
            h.key in selected_keys
            for h in households
        ])

        bess_sizes = _param(
            households,
            "bess",
            "bess_size",
            0,
        )

        for u, enabled in enumerate(
            bess_enabled
        ):
            if (
                enabled
                and bess_sizes[u] <= 1e-9
            ):
                if fallback is None:
                    raise RuntimeError(
                        f"{households[u].key}: "
                        "nincs bess_size és "
                        "nincs "
                        "BESS_DEFAULT_SIZE_KWH."
                    )

                bess_sizes[u] = fallback

        p_dhw = np.column_stack([
            (
                h.p_dhw
                if hss_enabled[u]
                else np.zeros(
                    EXPECTED_STEPS
                )
            )
            for u, h in enumerate(
                households
            )
        ])

        p_fixed = np.column_stack([
            (
                np.zeros(
                    EXPECTED_STEPS
                )
                if hss_enabled[u]
                else h.p_boiler_fixed
            )
            for u, h in enumerate(
                households
            )
        ])

        case = (
            "4d-K-E"
            if objective == "grid"
            else "4d-K-C"
        )

        output = (
            Path(out_dir)
            if out_dir is not None
            else Path(
                f"results_{case}_"
                f"{sharing_mode}"
            )
        )

        passive_count = sum(
            h.p_pv.sum() <= 1e-9
            and not hss_enabled[u]
            and not bess_enabled[u]
            and p_fixed[:, u].sum() <= 1e-9
            for u, h in enumerate(
                households
            )
        )

        monitor.stage(
            f"Eset={case}; "
            f"userek={len(households)}; "
            f"HSS(A)={hss_enabled.sum()}; "
            f"BESS={bess_enabled.sum()}; "
            f"passzív={passive_count}; "
            f"megosztás={sharing_mode}"
        )

        p_pv_full = np.column_stack([
            h.p_pv for h in households
        ])

        p_base_full = np.column_stack([
            h.p_base for h in households
        ])

        user_names = [
            h.name for h in households
        ]

        size_elh = _param(
            households,
            "hss",
            "size_elh",
            0,
        )

        volume = _param(
            households,
            "hss",
            "vol_hss_water",
            0,
        )

        soc_min = _param(
            households,
            "bess",
            "soc_bess_min",
            0.10,
        )

        soc_max = _param(
            households,
            "bess",
            "soc_bess_max",
            0.90,
        )

        soc_init = _param(
            households,
            "bess",
            "soc_bess_init",
            0.50,
        )

        hss_capacity = (
            volume * 0.00116667
        )

        hss_energy = np.where(
            hss_enabled,
            hss_capacity
            * (
                BOILER_T_INITIAL_C
                - BOILER_T_IN_C
            ),
            0.0,
        )

        bess_energy = np.where(
            bess_enabled,
            np.clip(
                bess_sizes * soc_init,
                bess_sizes * soc_min,
                bess_sizes * soc_max,
            ),
            0.0,
        )

        a_low_remaining = np.full(
            len(households),
            LOW_TARIFF_LIMIT_KWH,
        )

        b_low_remaining = np.full(
            len(households),
            LOW_TARIFF_LIMIT_KWH,
        )

        monthly_results = []
        monthly_root = (
            output / "monthly"
        )

        for (
            month_index,
            (month_name, month_slice),
        ) in enumerate(
            zip(
                MONTH_NAMES,
                _month_slices(),
            )
        ):
            monitor.stage(
                "Havi optimalizálás "
                f"{month_index + 1}/12 "
                f"({month_name}); "
                "lépések="
                f"{month_slice.stop - month_slice.start}"
            )

            month_result = (
                optimize_community(
                    p_pv_full[month_slice],
                    p_base_full[month_slice],
                    p_dhw[month_slice],
                    p_fixed[month_slice],
                    user_names=user_names,
                    hss_enabled=hss_enabled,
                    bess_enabled=bess_enabled,
                    size_elh=size_elh,
                    vol_hss_water=volume,
                    size_bess=bess_sizes,
                    T_env=BOILER_T_ENV_C,
                    T_max=BOILER_T_MAX_C,
                    T_abs_min=(
                        BOILER_T_ABS_MIN_C
                    ),
                    T_in=BOILER_T_IN_C,
                    T_comfort=(
                        BOILER_T_COMFORT_C
                    ),
                    T_setpoint_min=(
                        BOILER_T_SETPOINT_MIN_C
                    ),
                    T_initial=(
                        BOILER_T_INITIAL_C
                    ),
                    a_hss=_param(
                        households,
                        "hss",
                        "a_hss",
                        0.01275,
                    ),
                    eta_elh=_param(
                        households,
                        "hss",
                        "eta_elh",
                        0.95,
                    ),
                    eta_bess_in=_param(
                        households,
                        "bess",
                        "eta_bess_in",
                        0.98,
                    ),
                    eta_bess_out=_param(
                        households,
                        "bess",
                        "eta_bess_out",
                        0.96,
                    ),
                    eta_bess_stor=_param(
                        households,
                        "bess",
                        "eta_bess_stor",
                        0.995,
                    ),
                    soc_bess_min=soc_min,
                    soc_bess_max=soc_max,
                    soc_bess_init=soc_init,
                    t_bess_min=_param(
                        households,
                        "bess",
                        "t_bess_min",
                        2.0,
                    ),
                    initial_hss_energy_kwhth=(
                        hss_energy
                    ),
                    initial_bess_energy_kwh=(
                        bess_energy
                    ),
                    a_low_remaining_kwh=(
                        a_low_remaining
                    ),
                    b_low_remaining_kwh=(
                        b_low_remaining
                    ),
                    enforce_terminal_state=(
                        month_index == 11
                    ),
                    objective=objective,
                    sharing_mode=(
                        sharing_mode
                    ),
                    run_lp=run_lp,
                    solver=solver,
                    msg=msg,
                    aggregate_passive=(
                        aggregate_passive
                    ),
                    progress_callback=(
                        monitor.stage
                    ),
                )
            )

            month_result["summary"][
                "month"
            ] = month_name

            month_result["summary"][
                "month_index"
            ] = month_index + 1

            save_community_results(
                month_result,
                monthly_root
                / month_name,
                save_user_timeseries=False,
            )

            monthly_results.append(
                month_result
            )

            state = month_result["state"]

            hss_energy = np.asarray(
                state[
                    "final_hss_energy_kwhth"
                ],
                dtype=float,
            )

            bess_energy = np.asarray(
                state[
                    "final_bess_energy_kwh"
                ],
                dtype=float,
            )

            a_low_remaining = (
                np.asarray(
                    state[
                        "a_low_remaining_kwh"
                    ],
                    dtype=float,
                )
            )

            b_low_remaining = (
                np.asarray(
                    state[
                        "b_low_remaining_kwh"
                    ],
                    dtype=float,
                )
            )

        monitor.stage(
            "A 12 havi eredmény éves "
            "összefűzése"
        )

        result = _combine_monthly_results(
            monthly_results
        )

        result["summary"].update({
            "out_dir": str(output),
            "bess_assignment_mode": (
                "explicit"
                if bess_households
                is not None
                else "percentage"
            ),
            "bess_household_keys": (
                sorted(selected_keys)
            ),
            "pv_ratio": float(pv_ratio),
            "boiler_model_enabled": (
                bool(
                    include_boiler_model
                )
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
            "run_log_path": str(
                Path(run_log_path)
            ),
        })

        monitor.stage(
            "Eredményfájlok mentése"
        )

        save_community_results(
            result,
            output,
            save_user_timeseries=(
                save_user_timeseries
            ),
        )

        pd.DataFrame([
            {
                "month": month_name,
                **monthly_results[index][
                    "summary"
                ],
            }
            for index, month_name
            in enumerate(MONTH_NAMES)
        ]).to_csv(
            output
            / "monthly_summary.csv",
            index=False,
        )

        monitor.stage(
            "Futás sikeresen "
            "befejeződött; "
            f"eredmények: {output}"
        )

        print(
            json.dumps(
                result["summary"],
                indent=2,
                ensure_ascii=False,
            ),
            flush=True,
        )

        return result["summary"]

    except BaseException as exc:
        monitor.stage(
            "Futás hibával leállt: "
            f"{type(exc).__name__}: "
            f"{exc}"
        )
        raise

    finally:
        monitor.stop()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "4d-K közösségi optimalizálás "
            "automatikus A/B bojlerkörrel."
        )
    )

    parser.add_argument(
        "--sim",
        default=str(SIM_YAML),
    )

    parser.add_argument(
        "--profiles",
        default=str(PROFILES_CSV),
    )

    parser.add_argument(
        "--dhw-profiles",
        default=str(
            DHW_PROFILES_CSV
        ),
    )

    parser.add_argument(
        "--out",
    )

    parser.add_argument(
        "--max-users",
        type=int,
        default=MAX_USERS,
    )

    parser.add_argument(
        "--objective",
        choices=[
            "bill",
            "grid",
        ],
        default=OBJECTIVE,
    )

    parser.add_argument(
        "--sharing-mode",
        choices=[
            "equal",
            "proportional",
        ],
        default=SHARING_MODE,
    )

    parser.add_argument(
        "--pv-ratio",
        type=float,
        default=PV_RATIO,
    )

    parser.add_argument(
        "--bess-share-pct",
        type=float,
        default=BESS_SHARE_PCT,
    )

    parser.add_argument(
        "--no-bess",
        action="store_true",
        default=not INCLUDE_BESS,
    )

    parser.add_argument(
        "--no-boiler-model",
        action="store_true",
        default=not INCLUDE_BOILER_MODEL,
    )

    parser.add_argument(
        "--lp",
        action="store_true",
    )

    parser.add_argument(
        "--solver",
        choices=[
            "gurobi",
            "cbc",
        ],
        default=SOLVER,
    )

    parser.add_argument(
        "--msg",
        action="store_true",
        default=MSG,
    )

    parser.add_argument(
        "--save-user-timeseries",
        action="store_true",
        default=(
            SAVE_USER_TIMESERIES
        ),
    )

    args = parser.parse_args(
        argv
    )

    run(
        args.sim,
        args.profiles,
        args.dhw_profiles,
        args.out,
        max_users=args.max_users,
        objective=args.objective,
        sharing_mode=(
            args.sharing_mode
        ),
        pv_ratio=args.pv_ratio,
        include_bess=(
            not args.no_bess
        ),
        bess_share_pct=(
            args.bess_share_pct
        ),
        include_boiler_model=(
            not args.no_boiler_model
        ),
        measured_boiler_households=(
            MEASURED_BOILER_HOUSEHOLDS
        ),
        run_lp=args.lp,
        solver=args.solver,
        msg=args.msg,
        save_user_timeseries=(
            args.save_user_timeseries
        ),
    )


if __name__ == "__main__":
    main(sys.argv[1:])