"""HP-s háztartások és eredeti eszközeik összegyűjtése.

HP-forrás:
    generated_heat_pumps/Users

Eredeti háztartások és eszközök:
    Input/Users

A program kizárólag riportokat készít.
Az eredeti YAML-eket nem módosítja.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import copy
import csv
import json
import math

import yaml


# ============================================================
# BEÁLLÍTÁSOK
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

# Működik a projekt gyökerében és InputTools almappában is.
PROJECT_ROOT = (
    SCRIPT_DIR
    if (SCRIPT_DIR / "Input").is_dir()
    else SCRIPT_DIR.parent
)

INPUT_DIR = PROJECT_ROOT / "Input"

# Eredeti felhasználók.
SOURCE_USERS_DIR = INPUT_DIR / "Users"

SOURCE_SIM_YAML = (
    INPUT_DIR
    / "simulation_config_disaggregated_with_userlist.yaml"
)

SOURCE_MEASUREMENTS_CSV = (
    INPUT_DIR
    / "measurements_disaggregated.csv"
)

# Generált HP-adatok.
GENERATED_HP_DIR = PROJECT_ROOT / "generated_heat_pumps"

SOURCE_HP_USERS_DIR = GENERATED_HP_DIR / "Users"

SOURCE_HP_PROFILES_CSV = (
    GENERATED_HP_DIR
    / "heat_pump_profiles.csv"
)

# "simulation":
#   az eredeti szimulációs users_list tagjait vizsgálja.
#
# "folder":
#   az eredeti SOURCE_USERS_DIR összes háztartását vizsgálja.
#
# Mindkét esetben a generált HP-mappával kapcsolja össze őket.
SOURCE_SELECTION = "simulation"

REPORT_DIR = INPUT_DIR / "hp_original_inventory"

EXCLUDED_USERS = {
    "battery",
    "bess",
    "community",
    "load_battery",
}


@dataclass
class User:
    key: str
    path: Path
    data: dict
    units: dict
    hp_source_path: Path | None = None

    @property
    def has_hp(self):
        return present(
            self.units.get("heat_pump")
        )

    @property
    def has_pv(self):
        block = self.units.get("pv")

        if not isinstance(block, dict):
            return False

        return (
            bool(profile_of(block))
            or number(block.get("size")) > 0
        )


def present(value):
    """Van-e nem üres eszközblokk?"""
    return (
        isinstance(value, dict)
        and bool(value)
    )


def profile_of(block):
    """HP esetén profile_id, más eszköznél általában profile."""
    if not isinstance(block, dict):
        return ""

    value = (
        block.get("profile_id")
        or block.get("profile")
    )

    return (
        ""
        if value is None
        else str(value).strip()
    )


def number(value):
    try:
        value = float(value)

        return (
            value
            if math.isfinite(value)
            else 0.0
        )

    except (ValueError, TypeError):
        return 0.0


def read_yaml(path):
    value = yaml.safe_load(
        Path(path).read_text(
            encoding="utf-8-sig"
        )
    )

    if not isinstance(value, dict):
        raise ValueError(
            f"Nem mapping YAML: {path}"
        )

    return value


def units_of(data):
    """Kezeli a units burkolatú és közvetlen YAML-t is."""
    units = data.get("units", data)

    if not isinstance(units, dict):
        raise ValueError(
            "Hibás units blokk"
        )

    return units


def resolve_user(key, files):
    key = str(key).strip()

    if (
        Path(key).name != key
        or "/" in key
        or "\\" in key
    ):
        raise ValueError(
            "A users_list csak fájlnevet/kódot "
            f"tartalmazhat: {key}"
        )

    stem = Path(key).stem

    matches = [
        path
        for path in files
        if path.stem == stem
    ]

    if len(matches) != 1:
        raise ValueError(
            f"{key}: {len(matches)} "
            "YAML-találat a forrásmappában"
        )

    return matches[0]


def canonical_code(value):
    """Az összekapcsoláshoz használt teljes háztartáskód."""
    return (
        Path(str(value))
        .stem
        .removeprefix("timeseries_")
        .removeprefix("load_")
    )


def load_generated_hp(hp_users_dir):
    """A generált mappából összegyűjti a HP-blokkokat."""
    root = Path(hp_users_dir)

    if not root.is_dir():
        raise FileNotFoundError(
            "Generált HP Users mappa hiányzik: "
            f"{root}"
        )

    records = {}

    files = sorted(
        [
            *root.glob("*.yaml"),
            *root.glob("*.yml"),
        ]
    )

    for path in files:
        if path.stem.lower() in EXCLUDED_USERS:
            continue

        units = units_of(
            read_yaml(path)
        )

        block = units.get("heat_pump")

        if not present(block):
            continue

        key = canonical_code(
            path.stem
        )

        if key in records:
            raise ValueError(
                "Duplikált generált HP-kód: "
                f"{key}"
            )

        records[key] = (
            path,
            copy.deepcopy(block),
        )

    if not records:
        raise ValueError(
            "Nincs nem üres heat_pump blokk "
            f"a generált mappában: {root}"
        )

    return records


def load_users(
    users_dir=SOURCE_USERS_DIR,
    sim_yaml=SOURCE_SIM_YAML,
    selection=SOURCE_SELECTION,
    hp_users_dir=SOURCE_HP_USERS_DIR,
):
    """Eredeti háztartások + hozzájuk kapcsolt generált HP.

    data:
        az eredeti YAML változatlan tartalma.

    units:
        az eredeti eszközök és a generált HP-blokk
        memóriában összekapcsolt változata.
    """

    root = Path(users_dir)

    if not root.is_dir():
        raise FileNotFoundError(
            f"Forrás Users mappa hiányzik: {root}"
        )

    files = sorted(
        [
            *root.glob("*.yaml"),
            *root.glob("*.yml"),
        ],
        key=lambda path: path.name,
    )

    files = [
        path
        for path in files
        if path.stem.lower()
        not in EXCLUDED_USERS
    ]

    if not files:
        raise ValueError(
            f"Nincs háztartási YAML: {root}"
        )

    sim = None

    if selection == "simulation":
        sim = read_yaml(sim_yaml)

        keys = sim.get("users_list")

        if (
            not isinstance(keys, list)
            or not keys
        ):
            raise ValueError(
                "A forrás szimulációs YAML "
                "users_list mezője hiányzik/üres"
            )

        selected = [
            resolve_user(key, files)
            for key in keys
            if Path(str(key)).stem.lower()
            not in EXCLUDED_USERS
        ]

    elif selection == "folder":
        selected = files

    else:
        raise ValueError(
            "SOURCE_SELECTION: "
            "'simulation' vagy 'folder'"
        )

    if (
        len({path.stem for path in selected})
        != len(selected)
    ):
        raise ValueError(
            "Duplikált háztartáskód "
            "a kiválasztásban"
        )

    generated = load_generated_hp(
        hp_users_dir
    )

    users = []

    for path in selected:
        data = read_yaml(path)

        units = copy.deepcopy(
            units_of(data)
        )

        hp_source = None

        match = generated.get(
            canonical_code(path.stem)
        )

        if match is not None:
            hp_source, hp_block = match

            original_hp = units.get(
                "heat_pump"
            )

            if (
                present(original_hp)
                and original_hp != hp_block
            ):
                raise ValueError(
                    f"{path.stem}: eltérő eredeti "
                    "és generált HP-blokk. "
                    "A forrást egyeztesd."
                )

            units["heat_pump"] = (
                copy.deepcopy(hp_block)
            )

        users.append(
            User(
                key=path.stem,
                path=path,
                data=data,
                units=units,
                hp_source_path=hp_source,
            )
        )

    matched = sum(
        user.hp_source_path is not None
        for user in users
    )

    print(
        f"Generált HP-s YAML-ek: {len(generated)}; "
        f"eredeti listához kapcsolható: {matched}"
    )

    if not matched:
        raise ValueError(
            "Egyetlen generált HP-kód sem található "
            "a kiválasztott eredeti listában. "
            "Ellenőrizd a mappát és a fájlnevek "
            "háztartáskódját; nincs véletlen "
            "hozzárendelés."
        )

    if not users:
        raise ValueError(
            "Nincs kiválasztott háztartás"
        )

    return users, sim


def inventory_row(user):
    units = user.units

    hp = units.get("heat_pump") or {}

    params = (
        hp.get("heat_pump_parameters")
        or {}
    )

    house = (
        hp.get("household_parameters")
        or {}
    )

    model = (
        house.get("model_5r2c")
        or {}
    )

    name = units.get("name")

    name = (
        name.get("name", user.key)
        if isinstance(name, dict)
        else (name or user.key)
    )

    devices = [
        key
        for key, value in units.items()
        if key not in {"name", "ue"}
        and present(value)
    ]

    has_ut = (
        present(units.get("ut"))
        and (
            bool(profile_of(units["ut"]))
            or number(
                units["ut"].get("size")
            ) > 0
        )
    )

    return dict(
        household=user.key,
        yaml_file=user.path.name,
        name=str(name),
        hp_source_yaml=str(
            user.hp_source_path or ""
        ),
        hp_present_in_original=present(
            units_of(user.data).get("heat_pump")
        ),
        has_hp=user.has_hp,
        hp_profile=profile_of(hp),
        hp_source_profile=str(
            hp.get("source_profile_id") or ""
        ),
        hp_tariff_in_yaml=str(
            params.get("tariff")
            or hp.get("tariff")
            or ""
        ),
        has_pv=user.has_pv,
        pv_profile=profile_of(
            units.get("pv")
        ),
        pv_size_in_yaml=number(
            (units.get("pv") or {}).get("size")
        ),
        has_hss=present(
            units.get("hss")
        ),
        has_measured_boiler=has_ut,
        boiler_profile=profile_of(
            units.get("ut")
        ),
        has_bess=present(
            units.get("bess")
        ),
        base_profile=profile_of(
            units.get("ue")
        ),
        base_size_in_yaml=number(
            (units.get("ue") or {}).get("size")
        ),
        device_blocks=";".join(devices),
        hp_has_building_parameters=bool(house),
        hp_has_5r2c=bool(model),
        hp_has_equipment_parameters=bool(params),
        needs_pv=(
            user.has_hp
            and not user.has_pv
        ),
    )


def write_csv(path, rows, fields=None):
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if fields is None:
        fields = (
            list(rows[0])
            if rows
            else ["household"]
        )

    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(rows)


def inventory(
    users,
    report_dir,
    hp_users_dir=SOURCE_HP_USERS_DIR,
):
    rows = [
        inventory_row(user)
        for user in users
    ]

    generated = load_generated_hp(
        hp_users_dir
    )

    original_by_code = {
        canonical_code(user.key): user
        for user in users
    }

    matches = []

    for key, (path, block) in generated.items():
        original = original_by_code.get(key)

        matches.append(
            dict(
                generated_household=path.stem,
                generated_yaml=str(path),
                hp_profile=profile_of(block),
                in_original_selected_list=(
                    original is not None
                ),
                original_household=(
                    original.key
                    if original
                    else ""
                ),
            )
        )

    hp_rows = [
        row
        for row in rows
        if row["has_hp"]
    ]

    report_dir = Path(report_dir)

    report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_csv(
        report_dir / "generated_hp_matches.csv",
        matches,
    )

    write_csv(
        report_dir / "all_households.csv",
        rows,
    )

    write_csv(
        report_dir / "hp_households.csv",
        hp_rows,
        list(rows[0]),
    )

    write_csv(
        report_dir / "hp_without_pv.csv",
        [
            row
            for row in hp_rows
            if row["needs_pv"]
        ],
        list(rows[0]),
    )

    keys = [
        row["household"]
        for row in hp_rows
    ]

    (
        report_dir / "hp_households.json"
    ).write_text(
        json.dumps(
            keys,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    (
        report_dir / "hp_households_for_python.txt"
    ).write_text(
        "HP_HOUSEHOLDS = "
        + repr(keys)
        + "\n",
        encoding="utf-8",
    )

    summary = dict(
        households=len(rows),
        hp=len(hp_rows),
        hp_with_pv=sum(
            row["has_pv"]
            for row in hp_rows
        ),
        hp_without_pv=sum(
            row["needs_pv"]
            for row in hp_rows
        ),
        hp_with_hss=sum(
            row["has_hss"]
            for row in hp_rows
        ),
        hp_with_measured_boiler=sum(
            row["has_measured_boiler"]
            for row in hp_rows
        ),
        hp_with_bess=sum(
            row["has_bess"]
            for row in hp_rows
        ),
        generated_hp_total=len(generated),
        generated_hp_outside_selected_list=sum(
            not row["in_original_selected_list"]
            for row in matches
        ),
        hp_matched_from_generated=sum(
            user.hp_source_path is not None
            for user in users
        ),
    )

    (
        report_dir / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
    )

    for row in hp_rows:
        print(
            f"{row['household']}: "
            f"PV={row['has_pv']}, "
            f"HSS={row['has_hss']}, "
            f"bojler={row['has_measured_boiler']}, "
            f"BESS={row['has_bess']}, "
            f"HP-profil={row['hp_profile']}; "
            f"blokkok: {row['device_blocks']}"
        )

    print(
        f"Riportok: {report_dir.resolve()}"
    )

    return rows, summary


def main():
    users, _ = load_users()

    inventory(
        users,
        REPORT_DIR,
    )


if __name__ == "__main__":
    main()