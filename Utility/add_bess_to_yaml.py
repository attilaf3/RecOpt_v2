"""Csak hiányzó BESS-blokkok hozzáadása; az eredeti YAML-szöveg megmarad."""

from pathlib import Path
from datetime import datetime
import copy
import math
import random
import shutil

import numpy as np
import pandas as pd
import yaml


# ============================================================
# BEÁLLÍTÁSOK
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

PROJECT_ROOT = (
    SCRIPT_DIR
    if (SCRIPT_DIR / "Input").is_dir()
    else SCRIPT_DIR.parent
)

USERS_DIR = PROJECT_ROOT / "Input" / "users_v4"

MEASUREMENTS_CSV = (
    PROJECT_ROOT / "Input" / "measurements_disaggregated_v4.csv"
)

DT = 0.25
PROFILE_UNIT = "kWh/step"  # vagy "kW"

FILL_HOURS = 3.0

# None: automatikus kiválasztás az alábbi N_NEW_BESS alapján.
#
# Kézi lista esetén pontos fájlnévtörzseket adj meg:
# RECIPIENTS = [
#     "load_0123456789012345",
#     "load_0987654321098765",
# ]
#
# [] esetén senki nem kap új BESS-t.
RECIPIENTS = ["load_0420144888370690",
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

# None: minden PV-s, BESS nélküli háztartás kap BESS-t.
# Egész szám: ennyi új BESS véletlenszerű kiosztása.
# Kézi recipientlista mellett figyelmen kívül marad.
N_NEW_BESS = None

RANDOM_SEED = 42

# True esetén csak a tervet írja ki, YAML-t nem módosít.
DRY_RUN = False

BESS_PARAMETERS = {
    "eta_bess_in": "0.98",
    "eta_bess_out": "0.96",
    "eta_bess_stor": "0.995",
    "eta_self_discharge": "1.0",
    "soc_bess_max": "0.9",
    "soc_bess_min": "0.1",
}


# ============================================================
# YAML OLVASÁSA ÉS ELLENŐRZÉSE
# ============================================================

def read_user(path):
    raw = path.read_bytes()
    text = raw.decode("utf-8-sig")

    # Aliasoknál egy blokk több helyről is hivatkozható.
    # Ezeket nem módosítjuk automatikusan.
    if any(
        isinstance(event, yaml.events.AliasEvent)
        for event in yaml.parse(text)
    ):
        raise ValueError(
            f"{path.name}: YAML-alias nem támogatott"
        )

    root = yaml.compose(text)

    def validate(node):
        if isinstance(node, yaml.MappingNode):
            keys = [
                key.value
                for key, _ in node.value
            ]

            if len(keys) != len(set(keys)) or "<<" in keys:
                raise ValueError(
                    f"{path.name}: ismétlődő kulcs vagy YAML-merge"
                )

            for _, value in node.value:
                validate(value)

        elif isinstance(node, yaml.SequenceNode):
            for value in node.value:
                validate(value)

    validate(root)

    data = yaml.safe_load(text)

    if not isinstance(data, dict):
        raise ValueError(
            f"{path.name}: a YAML gyökere nem szótár"
        )

    units = data.get("units", data)

    if not isinstance(units, dict):
        raise ValueError(
            f"{path.name}: hibás units blokk"
        )

    node = root

    if "units" in data:
        node = next(
            value
            for key, value in root.value
            if key.value == "units"
        )

    return raw, text, data, units, node


def insert_bess(text, data, node, block):
    """
    Csak az új blokk szövegét szúrja be.
    A meglévő YAML-t nem írja újra yaml.safe_dump-pal.
    """

    if node.flow_style or not node.value:
        raise ValueError(
            "A célhoz nem üres, többsoros YAML-szótár szükséges"
        )

    indent = node.value[0][0].start_mark.column

    newline = "\r\n" if "\r\n" in text else "\n"

    addition = yaml.safe_dump(
        {"bess": block},
        sort_keys=False,
        allow_unicode=True,
    )

    addition = newline.join(
        " " * indent + line
        for line in addition.splitlines()
    ) + newline

    # A célblokk vége: a következő szülőszintű kulcs,
    # vagy a dokumentum vége.
    position = node.end_mark.index

    if position and text[position - 1] not in "\r\n":
        addition = newline + addition

    updated = (
        text[:position]
        + addition
        + text[position:]
    )

    # Ellenőrzés: a tartalomban kizárólag a bess változhatott.
    expected = copy.deepcopy(data)
    expected.get("units", expected)["bess"] = block

    if yaml.safe_load(updated) != expected:
        raise RuntimeError(
            "A beszúrás ellenőrzése sikertelen; nem írunk fájlt"
        )

    return updated


# ============================================================
# FŐPROGRAM
# ============================================================

def main():
    if (
        not math.isfinite(DT)
        or DT <= 0
        or not math.isfinite(FILL_HOURS)
        or FILL_HOURS <= 0
    ):
        raise ValueError(
            "DT és FILL_HOURS pozitív, véges szám legyen"
        )

    if PROFILE_UNIT not in {"kWh/step", "kW"}:
        raise ValueError(
            "PROFILE_UNIT: 'kWh/step' vagy 'kW'"
        )

    files = sorted({
        *USERS_DIR.glob("*.yaml"),
        *USERS_DIR.glob("*.yml"),
    })

    if not files:
        raise FileNotFoundError(
            f"Nincs YAML itt: {USERS_DIR}"
        )

    records = {}
    candidates = []

    for path in files:
        if path.stem in records:
            raise ValueError(
                f"Ismétlődő fájlnévtörzs: {path.stem}"
            )

        record = read_user(path)
        records[path.stem] = (path, record)

        units = record[3]
        pv = units.get("pv")

        if (
            isinstance(pv, dict)
            and (pv.get("profile_id") or pv.get("profile"))
            and "bess" not in units
        ):
            candidates.append(path.stem)

    # --------------------------------------------------------
    # Címzettek kiválasztása
    # --------------------------------------------------------

    if RECIPIENTS is not None:
        selected = []
        seen = set()

        for name in RECIPIENTS:
            key = Path(name).stem

            if key in seen:
                raise ValueError(
                    f"Ismételt címzett: {key}"
                )

            seen.add(key)

            if key not in records:
                raise KeyError(
                    f"Hiányzó háztartás: {key}"
                )

            if "bess" in records[key][1][3]:
                print(
                    f"KIHAGYVA, már van bess kulcs: {key}"
                )
                continue

            if key not in candidates:
                raise ValueError(
                    f"Nincs PV-profil: {key}"
                )

            selected.append(key)

    elif N_NEW_BESS is None:
        selected = candidates

    else:
        if (
            isinstance(N_NEW_BESS, bool)
            or not isinstance(N_NEW_BESS, int)
            or not 0 <= N_NEW_BESS <= len(candidates)
        ):
            raise ValueError(
                f"N_NEW_BESS 0 és {len(candidates)} "
                "közötti egész legyen"
            )

        selected = random.Random(RANDOM_SEED).sample(
            candidates,
            N_NEW_BESS,
        )

    if not selected:
        print(
            "Nincs hozzáadandó BESS; nem változott fájl."
        )
        return

    # --------------------------------------------------------
    # Méretezés az aktuális PV-profilok alapján
    # --------------------------------------------------------

    frame = pd.read_csv(MEASUREMENTS_CSV)
    plans = []

    for key in selected:
        path, record = records[key]
        raw, text, data, units, node = record

        pv = units["pv"]

        profile = str(
            pv.get("profile_id") or pv.get("profile")
        )

        if profile not in frame:
            raise KeyError(
                f"{key}: hiányzó PV-oszlop: {profile}"
            )

        values = pd.to_numeric(
            frame[profile],
            errors="raise",
        ).to_numpy(dtype=float)

        if (
            not values.size
            or not np.isfinite(values).all()
            or (values < 0).any()
        ):
            raise ValueError(
                f"{key}: üres, negatív vagy nem véges PV-profil"
            )

        # Időlépésenkénti energiából átlagos teljesítmény:
        # P[kW] = E[kWh] / dt[h]
        peak = float(values.max()) / (
            DT if PROFILE_UNIT == "kWh/step" else 1.0
        )

        if peak <= 0:
            raise ValueError(
                f"{key}: a PV-csúcs nem pozitív"
            )

        block = dict(BESS_PARAMETERS)

        block["bess_size"] = (
            f"{peak * FILL_HOURS:.12g}"
        )

        # A meglévő modellekben:
        # P_bess,max = bess_size / t_bess_min.
        block["t_bess_min"] = (
            f"{FILL_HOURS:.12g}"
        )

        updated = insert_bess(
            text,
            data,
            node,
            block,
        )

        encoded = updated.encode("utf-8")

        # Az esetleges eredeti UTF-8 BOM is megmarad.
        if raw.startswith(b"\xef\xbb\xbf"):
            encoded = b"\xef\xbb\xbf" + encoded

        plans.append(
            (path, raw, encoded)
        )

        print(
            f"{key}: "
            f"PV-csúcs={peak:.3f} kW; "
            f"BESS={peak * FILL_HOURS:.3f} kWh; "
            f"P_max≈{peak:.3f} kW"
        )

    if DRY_RUN:
        print(
            f"DRY_RUN: {len(plans)} tervezett új BESS, "
            "írás nélkül."
        )
        return

    # --------------------------------------------------------
    # Mentés és írás
    # --------------------------------------------------------

    # Minden kiválasztott háztartás ellenőrzése elkészült.
    # Csak ezután kezdünk fájlt módosítani.
    backup = USERS_DIR.parent / (
        "bess_backup_"
        + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )

    backup.mkdir()

    for path, raw, _ in plans:
        if path.read_bytes() != raw:
            raise RuntimeError(
                f"Közben megváltozott: {path}"
            )

        shutil.copy2(
            path,
            backup / path.name,
        )

    try:
        for path, _, encoded in plans:
            path.write_bytes(encoded)

    except Exception:
        # Írási hiba esetén visszaállítjuk az eredetieket.
        for path, _, _ in plans:
            shutil.copy2(
                backup / path.name,
                path,
            )
        raise

    print(
        f"Kész: {len(plans)} új BESS. "
        f"Mentés: {backup}"
    )


if __name__ == "__main__":
    main()