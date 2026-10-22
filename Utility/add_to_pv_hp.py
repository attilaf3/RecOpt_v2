
"""
Új users_v4:
- új PV-k kiosztása HP-s háztartásoknak;
- meglévő HP + PV háztartások PV-jének újraméretezése.

A collect_hp_households.py legyen ugyanebben a mappában.
Az eredeti YAML-eket, mérési CSV-t és HP-profilokat nem módosítja.
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import copy
import csv
import json
import math
import random
import shutil
import tempfile

import numpy as np
import pandas as pd
import yaml

import collect_hp_households as source


# ============================================================
# BEÁLLÍTÁSOK
# ============================================================

# A forrásútvonalakat a collect_hp_households.py elején állítsd be.

OUTPUT_USERS_DIR = source.INPUT_DIR / "users_v4"

OUTPUT_MEASUREMENTS_CSV = (
    source.INPUT_DIR / "measurements_disaggregated_v4.csv"
)

OUTPUT_SIM_YAML = (
    source.INPUT_DIR
    / "simulation_config_disaggregated_with_userlist_v4.yaml"
)

OUTPUT_REPORT_CSV = source.INPUT_DIR / "pv_hp_assignment_v4.csv"

OUTPUT_SUMMARY_JSON = (
    source.INPUT_DIR / "pv_hp_assignment_v4_summary.json"
)


# KÉZI KIOSZTÁS:
# A felsorolt HP-s, eredetileg PV nélküli háztartások kapnak új PV-t.
#
# Példa:
# RECIPIENT_HOUSEHOLDS = [
#     "load_0123456789012345",
#     "load_0987654321098765",
# ]
#
# VÉLETLEN KIOSZTÁS:
# None esetén N_NEW_PV_USERS háztartás kap új PV-t.
#
# [] esetén senki nem kap új PV-t,
# de a meglévő HP + PV rendszerek újraméretezése megtörténik.

RECIPIENT_HOUSEHOLDS = ["load_0420144888370690",
"load_0420144888373402",
"load_0420144888381069",
"load_0420144653433567",
"load_0420144653434432",
"load_0420144653434478",
"load_0420144653449093",
"load_0420144653462282",
"load_0420144653463289",
"load_0420144888235070",
"load_0420144888235617",
"load_0420144888285727",
"load_0420144888287806",
"load_0420144888295341",
]


# Kizárólag az ÚJ PV-k száma.
# A meglévő három HP + PV háztartás ezen felül szerepel.
# Kézi recipientlista mellett figyelmen kívül marad.
N_NEW_PV_USERS = 10

RANDOM_SEED = 42

# A meglévő HP + PV háztartások automatikus újraméretezése.
RESIZE_EXISTING_HP_PV = True

# A kiválasztott PV-profil megengedett szorzója.
# Meglévő PV esetén a saját eredeti profilhoz viszonyít.
PV_SCALE_MIN = 0.80
PV_SCALE_MAX = 1.20

# Éves méretezési cél:
# ue = alapfogyasztás
# hp = hőszivattyú villamosenergia-fogyasztása
# ut = mért bojlerprofil, ha azt is bele szeretnéd venni
TARGET_COMPONENTS = ("ue", "hp")

EXPECTED_STEPS = 35040
DT = 0.25  # óra

# Ha a CSV értékei időlépésenkénti energiák:
MEASUREMENTS_UNIT = "kWh/step"
HP_PROFILE_UNIT = "kWh/step"

# Ha a profilok átlagos teljesítményt tartalmaznak,
# az adott beállítás értéke "kW" legyen.

# True esetén a meglévő v4 kimenetekről mentés készül,
# majd a program az eredeti forrásokból újraépíti őket.
OVERWRITE_OUTPUTS = True

# True: csak ellenőriz és kiírja a kiosztási tervet.
DRY_RUN = False

# A pv.size ebben a projektben éves energia, kWh.
# Ezeket a külön teljesítménymezőket is skálázzuk, ha vannak.
PV_POWER_FIELDS = (
    "peak_power_kw",
    "p_peak_kw",
    "p_nom_kw",
    "capacity_kw",
    "p_max_kw",
    "rated_power_kw",
)


# ============================================================
# SEGÉDFÜGGVÉNYEK
# ============================================================

def read_profiles(path, expected_steps):
    path = Path(path)

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        header = next(csv.reader(file), [])

    if len(header) != len(set(header)):
        raise ValueError(f"{path}: ismétlődő oszlopnév")

    frame = pd.read_csv(path)

    if frame.empty or len(frame) != expected_steps:
        raise ValueError(
            f"{path}: {len(frame)} sor, várt {expected_steps}"
        )

    if frame.columns.duplicated().any():
        raise ValueError(f"{path}: ismétlődő oszlopnév")

    frame.columns = frame.columns.astype(str)

    return frame


def profile_values(frame, profile, label):
    if not profile or profile not in frame:
        raise KeyError(
            f"{label}: hiányzó profil/oszlop: {profile!r}"
        )

    values = pd.to_numeric(
        frame[profile],
        errors="raise",
    ).to_numpy(dtype=float)

    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError(
            f"{label}: negatív vagy nem véges profilérték: {profile}"
        )

    return values


def annual_energy(values, unit, dt):
    if unit not in {"kWh/step", "kW"}:
        raise ValueError(
            "Profil mértékegysége: 'kWh/step' vagy 'kW'"
        )

    multiplier = dt if unit == "kW" else 1.0

    return float(values.sum() * multiplier)


def resolve_recipients(
    users,
    requested,
    count=N_NEW_PV_USERS,
    seed=RANDOM_SEED,
):
    """
    Csak az új PV-k címzettjeit választja ki.

    A meglévő HP + PV rendszerek újraméretezése ettől
    függetlenül, külön történik.
    """

    candidates = [
        user
        for user in users
        if user.has_hp and not user.has_pv
    ]

    if requested is None:
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or count < 0
        ):
            raise ValueError(
                "N_NEW_PV_USERS nemnegatív egész szám legyen"
            )

        if count > len(candidates):
            raise ValueError(
                f"{count} új PV-t kértél, de csak "
                f"{len(candidates)} HP-s, eredetileg PV nélküli "
                "háztartás választható"
            )

        return random.Random(seed).sample(
            sorted(candidates, key=lambda user: user.key),
            count,
        )

    result = []
    seen = set()

    for key in requested:
        stem = source.canonical_code(key)

        matches = [
            user
            for user in users
            if source.canonical_code(user.key) == stem
        ]

        if len(matches) != 1:
            raise ValueError(
                f"{key}: hiányzó vagy nem egyértelmű "
                "háztartáskód az eredeti listában"
            )

        user = matches[0]

        if user.key in seen:
            raise ValueError(f"Duplikált PV-címzett: {key}")

        seen.add(user.key)

        if user.has_pv:
            print(
                f"[INFO] {user.key}: már PV-s, "
                "nem új PV-címzett; a meglévő HP + PV-k "
                "méretezése külön történik."
            )
            continue

        if not user.has_hp:
            raise ValueError(
                f"{user.key}: nem HP-s és nincs eredeti PV-je; "
                "új PV-t nem kaphat"
            )

        result.append(user)

    return result


def validate_paths(users_dir, input_files, outputs):
    root = Path(users_dir).resolve()

    inputs = [
        root,
        *(
            Path(path).resolve()
            for path in input_files
            if path is not None
        ),
    ]

    output_paths = [
        Path(path).resolve()
        for path in outputs
    ]

    if len(set(output_paths)) != len(output_paths):
        raise ValueError(
            "A kimeneti útvonalak nem lehetnek azonosak"
        )

    for output_path in output_paths:
        for input_path in inputs:
            if (
                output_path == input_path
                or output_path in input_path.parents
                or input_path in output_path.parents
            ):
                raise ValueError(
                    "Forrás és kimenet átfed: "
                    f"{input_path} / {output_path}"
                )

    for index, path in enumerate(output_paths):
        if any(
            path in other.parents or other in path.parents
            for other in output_paths[index + 1:]
        ):
            raise ValueError(
                "A kimeneti célok nem lehetnek egymás almappái"
            )


def publish(staged, overwrite):
    """
    Elkészült kimenetek áthelyezése a végleges helyükre.

    Felülírás esetén a korábbi kimenetekről mentés készül.
    Telepítési hiba esetén visszaállítja a korábbi kimeneteket.
    """

    existing = [
        destination
        for _, destination in staged
        if destination.exists()
    ]

    if existing and not overwrite:
        raise FileExistsError(
            "Már létező kimenetek: "
            + ", ".join(map(str, existing))
        )

    backups = []
    installed = []
    backup_root = None

    try:
        if existing:
            stamp = datetime.now(timezone.utc).strftime(
                "%Y%m%d_%H%M%S_%f"
            )

            backup_root = (
                staged[0][1].parent
                / f"pv_hp_v4_backup_{stamp}"
            )

            backup_root.mkdir()

            for index, destination in enumerate(existing):
                backup = (
                    backup_root
                    / f"{index:02d}_{destination.name}"
                )

                destination.rename(backup)
                backups.append((backup, destination))

        for temporary_path, destination in staged:
            destination.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            temporary_path.rename(destination)
            installed.append(destination)

    except Exception:
        for destination in reversed(installed):
            if destination.is_dir():
                shutil.rmtree(destination)
            else:
                destination.unlink(missing_ok=True)

        for backup, destination in reversed(backups):
            backup.rename(destination)

        raise

    if backup_root is not None:
        print(f"Korábbi kimenetek mentése: {backup_root}")


# ============================================================
# FŐPROGRAM
# ============================================================

def run(
    *,
    users_dir=source.SOURCE_USERS_DIR,
    sim_yaml=source.SOURCE_SIM_YAML,
    selection=source.SOURCE_SELECTION,
    hp_users_dir=source.SOURCE_HP_USERS_DIR,
    measurements_csv=source.SOURCE_MEASUREMENTS_CSV,
    hp_csv=source.SOURCE_HP_PROFILES_CSV,
    output_users_dir=OUTPUT_USERS_DIR,
    output_measurements_csv=OUTPUT_MEASUREMENTS_CSV,
    output_sim_yaml=OUTPUT_SIM_YAML,
    output_report_csv=OUTPUT_REPORT_CSV,
    output_summary_json=OUTPUT_SUMMARY_JSON,
    recipients=RECIPIENT_HOUSEHOLDS,
    count=N_NEW_PV_USERS,
    seed=RANDOM_SEED,
    resize_existing=RESIZE_EXISTING_HP_PV,
    scale_min=PV_SCALE_MIN,
    scale_max=PV_SCALE_MAX,
    target_components=TARGET_COMPONENTS,
    expected_steps=EXPECTED_STEPS,
    dt=DT,
    measurements_unit=MEASUREMENTS_UNIT,
    hp_unit=HP_PROFILE_UNIT,
    overwrite=OVERWRITE_OUTPUTS,
    dry_run=DRY_RUN,
):
    if not (
        math.isfinite(scale_min)
        and math.isfinite(scale_max)
        and 0 < scale_min <= scale_max
    ):
        raise ValueError("Hibás PV-skálázási tartomány")

    if dt <= 0 or expected_steps <= 0:
        raise ValueError("Hibás dt vagy sorhossz")

    if (
        not target_components
        or set(target_components) - {"ue", "hp", "ut"}
        or len(set(target_components)) != len(target_components)
    ):
        raise ValueError(
            "TARGET_COMPONENTS: egyedi ue/hp/ut elemek"
        )

    outputs = list(
        map(
            Path,
            [
                output_users_dir,
                output_measurements_csv,
                output_sim_yaml,
                output_report_csv,
                output_summary_json,
            ],
        )
    )

    input_files = [
        hp_users_dir,
        measurements_csv,
        hp_csv,
        sim_yaml if selection == "simulation" else None,
    ]

    validate_paths(users_dir, input_files, outputs)

    if (
        not dry_run
        and not overwrite
        and any(path.exists() for path in outputs)
    ):
        raise FileExistsError(
            "Már létezik v4 kimenet. Új célt válassz, "
            "vagy OVERWRITE_OUTPUTS=True."
        )

    users, simulation = source.load_users(
        users_dir,
        sim_yaml,
        selection,
        hp_users_dir,
    )

    candidates = [
        user
        for user in users
        if user.has_hp and not user.has_pv
    ]

    selected = resolve_recipients(
        users,
        recipients,
        count,
        seed,
    )

    selected_keys = {
        user.key
        for user in selected
    }

    # Ezek a háztartások akkor is bekerülnek,
    # ha nincsenek a kézi recipientlistában.
    resize_users = [
        user
        for user in users
        if resize_existing and user.has_hp and user.has_pv
    ]

    resize_keys = {
        user.key
        for user in resize_users
    }

    pv_changed_keys = selected_keys | resize_keys

    frame = read_profiles(
        measurements_csv,
        expected_steps,
    )

    hp_frame = (
        read_profiles(hp_csv, expected_steps)
        if pv_changed_keys and "hp" in target_components
        else None
    )

    if (
        hp_frame is not None
        and "timestamp" in frame
        and "timestamp" in hp_frame
    ):
        measurement_timestamps = pd.DatetimeIndex(
            pd.to_datetime(
                frame["timestamp"],
                errors="raise",
            )
        )

        hp_timestamps = pd.DatetimeIndex(
            pd.to_datetime(
                hp_frame["timestamp"],
                errors="raise",
            )
        )

        if not measurement_timestamps.equals(hp_timestamps):
            raise ValueError(
                "HP és mérési CSV timestamp eltérés"
            )

    # Új PV esetén az eredeti PV-k szolgálnak donorprofilként.
    # Az energiát a CSV-ből számítjuk, nem a YAML size mezőjéből.
    donors = []

    if selected:
        for user in users:
            if not user.has_pv:
                continue

            profile = source.profile_of(
                user.units["pv"]
            )

            values = profile_values(
                frame,
                profile,
                f"PV-donor {user.key}",
            )

            energy = annual_energy(
                values,
                measurements_unit,
                dt,
            )

            if energy > 0:
                donors.append(
                    (user, profile, values, energy)
                )

        if not donors:
            raise ValueError(
                "Nincs pozitív profilú PV-donor "
                "az eredeti userlistában"
            )

    plan = []
    updates = {}
    columns = set(frame.columns)

    # A generált HP-blokkot az új v4 másolatba átemeljük,
    # ha az eredeti Input/Users YAML még nem tartalmazza.
    for user in users:
        original_hp = source.units_of(user.data).get(
            "heat_pump"
        )

        if (
            user.hp_source_path is not None
            and not source.present(original_hp)
        ):
            data = copy.deepcopy(user.data)

            source.units_of(data)["heat_pump"] = (
                copy.deepcopy(user.units["heat_pump"])
            )

            updates[user.key] = data

    for user in users:
        if not user.has_hp:
            continue

        row = dict(
            household=user.key,
            status=(
                "KEPT_EXISTING_PV"
                if user.has_pv
                else "NOT_SELECTED"
            ),
            hp_profile=source.profile_of(
                user.units["heat_pump"]
            ),
            donor_household="",
            donor_profile="",
            new_pv_profile=source.profile_of(
                user.units.get("pv")
            ),
            base_kwh=None,
            hp_kwh=None,
            boiler_kwh=None,
            target_kwh=None,
            donor_pv_kwh=None,
            scale_factor=None,
            new_pv_kwh=None,
            difference_kwh=None,
            difference_pct=None,
        )

        if user.key not in pv_changed_keys:
            plan.append(row)
            continue

        energies = {
            "ue": 0.0,
            "hp": 0.0,
            "ut": 0.0,
        }

        for component in target_components:
            unit_name = (
                "heat_pump"
                if component == "hp"
                else component
            )

            block = user.units.get(unit_name)

            if (
                component == "ut"
                and not source.profile_of(block)
            ):
                continue

            profile = source.profile_of(block)

            profile_frame = (
                hp_frame
                if component == "hp"
                else frame
            )

            values = profile_values(
                profile_frame,
                profile,
                f"{user.key}/{component}",
            )

            unit = (
                hp_unit
                if component == "hp"
                else measurements_unit
            )

            energies[component] = annual_energy(
                values,
                unit,
                dt,
            )

        target = sum(energies.values())

        if target <= 0:
            raise ValueError(
                f"{user.key}: nem pozitív méretezési energia"
            )

        if user.key in resize_keys:
            # Meglévő HP + PV:
            # megtartjuk a saját eredeti PV-profil alakját.
            donor = user

            profile = source.profile_of(
                user.units["pv"]
            )

            values = profile_values(
                frame,
                profile,
                f"Meglévő HP + PV {user.key}",
            )

            energy = annual_energy(
                values,
                measurements_unit,
                dt,
            )

            if energy <= 0:
                raise ValueError(
                    f"{user.key}: a meglévő PV-profil "
                    "energiája nulla"
                )

            factor = min(
                scale_max,
                max(scale_min, target / energy),
            )

        else:
            # Új PV:
            # a skálázási korlát mellett a célenergiához
            # legközelebb kerülő donorprofilt választjuk.
            options = []

            for (
                donor,
                profile,
                values,
                energy,
            ) in donors:
                factor = min(
                    scale_max,
                    max(scale_min, target / energy),
                )

                options.append(
                    (
                        abs(energy * factor - target),
                        abs(factor - 1),
                        donor.key,
                        donor,
                        profile,
                        values,
                        energy,
                        factor,
                    )
                )

            (
                _,
                _,
                _,
                donor,
                profile,
                values,
                energy,
                factor,
            ) = min(
                options,
                key=lambda option: option[:3],
            )

        # Új oszlop készül: az eredeti PV-oszlopot nem írjuk át.
        new_profile = f"pv_v4_{user.key}"
        number = 2

        while new_profile in columns:
            new_profile = f"pv_v4_{user.key}_{number}"
            number += 1

        columns.add(new_profile)

        frame[new_profile] = values * factor

        pv_block = copy.deepcopy(
            donor.units["pv"]
        )

        pv_block["profile"] = new_profile

        if "profile_id" in pv_block:
            pv_block["profile_id"] = new_profile

        # Ebben a projektben size = éves PV-energia, kWh.
        pv_block["size"] = energy * factor

        for field in PV_POWER_FIELDS:
            if field not in pv_block:
                continue

            value = float(pv_block[field])

            if not math.isfinite(value) or value < 0:
                raise ValueError(
                    f"Hibás PV teljesítménymező: {field}"
                )

            pv_block[field] = value * factor

        data = copy.deepcopy(
            updates.get(user.key, user.data)
        )

        source.units_of(data)["pv"] = pv_block

        updates[user.key] = data

        row.update(
            status=(
                "RESIZED_EXISTING_PV"
                if user.key in resize_keys
                else "ADDED_PV"
            ),
            donor_household=donor.key,
            donor_profile=profile,
            new_pv_profile=new_profile,
            base_kwh=energies["ue"],
            hp_kwh=energies["hp"],
            boiler_kwh=energies["ut"],
            target_kwh=target,
            donor_pv_kwh=energy,
            scale_factor=factor,
            new_pv_kwh=energy * factor,
            difference_kwh=energy * factor - target,
            difference_pct=(
                100 * (energy * factor - target) / target
            ),
        )

        plan.append(row)

        print(
            f"{user.key} <- {donor.key}: "
            f"cél={target:.2f} kWh; "
            f"PV={energy * factor:.2f} kWh; "
            f"szorzó={factor:.4f}"
        )

    summary = dict(
        source_users_dir=str(
            Path(users_dir).resolve()
        ),
        source_selection=selection,
        selected_households=len(users),
        pv_selection_mode=(
            "random"
            if recipients is None
            else "manual"
        ),
        requested_new_pv_count=(
            count
            if recipients is None
            else len(selected)
        ),
        requested_manual_households=(
            None
            if recipients is None
            else len(recipients)
        ),
        random_seed=(
            seed
            if recipients is None
            else None
        ),
        pv_with_original_before=sum(
            user.has_pv
            for user in users
        ),
        pv_total_after=(
            sum(user.has_pv for user in users)
            + len(selected)
        ),
        hp_count=sum(
            user.has_hp
            for user in users
        ),
        hp_with_pv_before=sum(
            user.has_hp and user.has_pv
            for user in users
        ),
        hp_without_pv_before=len(candidates),
        new_pv_count=len(selected),
        resized_existing_hp_pv_count=len(resize_users),
        resize_existing_hp_pv=resize_existing,
        hp_with_pv_after=(
            sum(
                user.has_hp and user.has_pv
                for user in users
            )
            + len(selected)
        ),
        target_components=list(target_components),
        pv_scale_min=scale_min,
        pv_scale_max=scale_max,
        hp_profiles_csv=str(
            Path(hp_csv).resolve()
        ),
        measurements_unit=measurements_unit,
        hp_profile_unit=hp_unit,
        expected_steps=expected_steps,
        dt=dt,
        generated_hp_users_dir=str(
            Path(hp_users_dir).resolve()
        ),
        hp_blocks_added=sum(
            not source.present(
                source.units_of(user.data).get("heat_pump")
            )
            and user.hp_source_path is not None
            for user in users
        ),
        original_files_modified=False,
    )

    print(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
    )

    if dry_run:
        print(
            "DRY_RUN=True: egyetlen kimeneti fájl sem készült."
        )
        return plan, summary

    # Előbb minden kimenet ideiglenes helyre készül el.
    # A teljes eredeti Users mappát másoljuk.
    outputs[0].parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with tempfile.TemporaryDirectory(
        prefix=".pv_hp_v4_stage_",
        dir=outputs[0].parent,
    ) as temporary_directory:
        stage = Path(temporary_directory)

        new_users = stage / "users"

        shutil.copytree(
            users_dir,
            new_users,
        )

        for user in users:
            if user.key not in updates:
                continue

            path = new_users / user.path.name

            path.write_text(
                yaml.safe_dump(
                    updates[user.key],
                    allow_unicode=True,
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            # Ellenőrizzük, hogy kizárólag a tervezett
            # HP- és PV-módosítások kerültek a YAML-be.
            check = source.read_yaml(path)

            expected = copy.deepcopy(user.data)

            if user.hp_source_path is not None:
                source.units_of(expected)["heat_pump"] = (
                    copy.deepcopy(user.units["heat_pump"])
                )

            if user.key in pv_changed_keys:
                source.units_of(expected)["pv"] = (
                    copy.deepcopy(
                        source.units_of(
                            updates[user.key]
                        )["pv"]
                    )
                )

            if expected != check:
                raise RuntimeError(
                    "Nem várt eszközváltozás történt: "
                    f"{user.key}"
                )

        new_csv = stage / "measurements.csv"

        frame.to_csv(
            new_csv,
            index=False,
        )

        new_simulation = (
            copy.deepcopy(simulation)
            if simulation is not None
            else {
                "users_list": [
                    user.key
                    for user in users
                ]
            }
        )

        simulation_file = stage / "simulation.yaml"

        simulation_file.write_text(
            yaml.safe_dump(
                new_simulation,
                allow_unicode=True,
                sort_keys=False,
            ),
            encoding="utf-8",
        )

        report_file = stage / "report.csv"

        source.write_csv(
            report_file,
            plan,
        )

        summary_file = stage / "summary.json"

        summary_file.write_text(
            json.dumps(
                summary,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        staged_outputs = [
            new_users,
            new_csv,
            simulation_file,
            report_file,
            summary_file,
        ]

        publish(
            list(zip(staged_outputs, outputs)),
            overwrite,
        )

    print("Kész kimenetek:")

    for path in outputs:
        print(path.resolve())

    return plan, summary


if __name__ == "__main__":
    run()