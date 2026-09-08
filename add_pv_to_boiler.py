from __future__ import annotations

import argparse
import copy
import csv
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


# =============================================================================
# ALAPBEÁLLÍTÁSOK – PyCharmból futtatva ezeket kell módosítani
# =============================================================================
USERS_DIR = Path("Input/Users")
MEASUREMENTS_CSV = Path("Input/measurements_disaggregated.csv")

OUTPUT_USERS_DIR = Path("Input/users_v2")
OUTPUT_MEASUREMENTS_CSV = Path("Input/measurements_disaggregated_v2.csv")
OUTPUT_REPORT_CSV = Path("Input/pv_assignment_v2.csv")

N_NEW_PV_USERS = 10
RANDOM_SEED = 42

# False esetén a program nem ír felül már létező v2 kimenetet.
OVERWRITE_OUTPUTS = True

# A donor PV-profil legfeljebb ekkora mértékben skálázható.
# 0.80–1.20 = legfeljebb -20% / +20%.
PV_SCALE_MIN = 0.80
PV_SCALE_MAX = 1.20

# A bojleres háztartás felismerése:
# - hss blokk, vagy
# - ut blokk érvényes profillal / pozitív mérettel.
# Ha kizárólag HSS-es felhasználókat szeretnél: "hss_only"
BOILER_DEFINITION = "hss_or_ut"  # "hss_only" vagy "hss_or_ut"

# Ha megadod a háztartásokat, pontosan ezek kapnak új PV-t.
# Elfogadott példák ugyanarra a háztartásra:
#   "load_0420144653458813", "0420144653458813", vagy az egyedi "813" végződés.
# None esetén az eredeti, véletlenszerű kiválasztás fut N_NEW_PV_USERS darabbal.
RECIPIENT_HOUSEHOLDS: list[str] | None = ["load_0420144653449093",
"load_0420144888235070",
"load_0420144888295341",
"load_0420144888340898",
"load_0420144888397058",
"load_0420144888460007",
"load_0420144888377089",
"load_0420144888444271",
"load_0420144888481697",
"load_0420144888447785"]


@dataclass
class UserRecord:
    code: str
    path: Path
    data: dict[str, Any]
    units: dict[str, Any]
    has_boiler: bool
    has_pv: bool
    pv_profile: str | None
    ue_annual_kwh: float
    boiler_annual_kwh: float
    pv_annual_kwh: float


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _yaml_files(directory: Path) -> list[Path]:
    files = list(directory.glob("*.yaml")) + list(directory.glob("*.yml"))
    return sorted(files, key=lambda p: p.name.lower())


def _user_code(path: Path, units: dict[str, Any]) -> str:
    """Elsősorban a YAML fájlnév törzsét használja háztartáskódként."""
    return str(path.stem)


def _has_valid_profile(block: Any) -> bool:
    return isinstance(block, dict) and block.get("profile") not in (None, "")


def _has_boiler(units: dict[str, Any], definition: str) -> bool:
    hss = units.get("hss")
    ut = units.get("ut")

    has_hss = isinstance(hss, dict) and bool(hss)
    has_ut = isinstance(ut, dict) and bool(ut) and (
        _has_valid_profile(ut) or _safe_float(ut.get("size"), 0.0) > 0.0
    )

    if definition == "hss_only":
        return has_hss
    if definition == "hss_or_ut":
        return has_hss or has_ut
    raise ValueError(f"Ismeretlen BOILER_DEFINITION: {definition}")


def _has_pv(units: dict[str, Any]) -> bool:
    pv = units.get("pv")
    if not isinstance(pv, dict) or not pv:
        return False
    return _has_valid_profile(pv) or _safe_float(pv.get("size"), 0.0) > 0.0


def load_users(users_dir: Path, boiler_definition: str) -> list[UserRecord]:
    files = _yaml_files(users_dir)
    if not files:
        raise FileNotFoundError(f"Nem található YAML fájl ebben a mappában: {users_dir.resolve()}")

    records: list[UserRecord] = []
    for path in files:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            print(f"[WARN] Nem dict YAML, kihagyva: {path.name}")
            continue

        # Kezeli a 'units:' burkolt és a közvetlen units-struktúrát is.
        units = loaded.get("units", loaded)
        if not isinstance(units, dict):
            print(f"[WARN] Nincs értelmezhető units blokk, kihagyva: {path.name}")
            continue

        pv = units.get("pv") or {}
        pv_profile = pv.get("profile") if isinstance(pv, dict) else None

        records.append(
            UserRecord(
                code=_user_code(path, units),
                path=path,
                data=loaded,
                units=units,
                has_boiler=_has_boiler(units, boiler_definition),
                has_pv=_has_pv(units),
                pv_profile=str(pv_profile) if pv_profile not in (None, "") else None,
                ue_annual_kwh=_safe_float((units.get("ue") or {}).get("size"), 0.0),
                boiler_annual_kwh=_safe_float((units.get("ut") or {}).get("size"), 0.0),
                pv_annual_kwh=_safe_float((units.get("pv") or {}).get("size"), 0.0),
            )
        )

    return records


def select_recipients(
    candidates: list[UserRecord],
    count: int,
    seed: int,
    forced_codes: list[str] | None,
) -> list[UserRecord]:
    if forced_codes is not None:
        requested = [str(code).strip() for code in forced_codes]
        if not requested:
            raise ValueError(
                "A RECIPIENT_HOUSEHOLDS üres lista. Adj meg legalább egy háztartást, "
                "vagy használj None értéket a véletlenszerű módhoz."
            )
        if any(not value for value in requested):
            raise ValueError("A RECIPIENT_HOUSEHOLDS nem tartalmazhat üres azonosítót.")

        selected: list[UserRecord] = []
        unresolved: list[str] = []
        for value in requested:
            stem = Path(value).stem
            without_prefix = stem.removeprefix("timeseries_").removeprefix("load_")
            matches = [
                user for user in candidates
                if user.code == stem
                or user.code.removeprefix("load_") == without_prefix
                or user.code.removeprefix("load_").endswith(without_prefix)
            ]
            if not matches:
                unresolved.append(value)
                continue
            if len(matches) > 1:
                raise ValueError(
                    f"A(z) {value!r} azonosító nem egyértelmű. Találatok: "
                    f"{[user.code for user in matches]}. Adj meg hosszabb azonosítót."
                )
            selected.append(matches[0])

        if unresolved:
            raise ValueError(
                "A megadott háztartások között van olyan, amely nem bojleres, "
                "az eredeti adatokban már PV-s, vagy nem található: "
                f"{unresolved}"
            )
        resolved_codes = [user.code for user in selected]
        if len(set(resolved_codes)) != len(resolved_codes):
            raise ValueError(
                "A RECIPIENT_HOUSEHOLDS ugyanazt a háztartást többször jelöli ki."
            )
        return selected

    if count < 0:
        raise ValueError("N_NEW_PV_USERS nem lehet negatív.")
    if count > len(candidates):
        raise ValueError(
            f"{count} új PV-s felhasználót kértél, de csak {len(candidates)} "
            "bojleres és eredetileg nem PV-s háztartás található."
        )

    rng = random.Random(seed)
    return rng.sample(sorted(candidates, key=lambda u: u.code), count)


def prepare_output_paths(
    users_out: Path,
    measurements_out: Path,
    report_out: Path,
    overwrite: bool,
) -> None:
    existing = [p for p in (users_out, measurements_out, report_out) if p.exists()]
    if existing and not overwrite:
        text = "\n".join(f"  - {p}" for p in existing)
        raise FileExistsError(
            "A következő v3 kimenetek már léteznek, ezért biztonsági okból nem írom felül őket:\n"
            f"{text}\nÁllítsd az OVERWRITE_OUTPUTS értékét True-ra, vagy adj meg más kimeneti nevet."
        )

    if overwrite:
        if users_out.exists():
            shutil.rmtree(users_out)
        for p in (measurements_out, report_out):
            if p.exists():
                p.unlink()

    users_out.parent.mkdir(parents=True, exist_ok=True)
    measurements_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.parent.mkdir(parents=True, exist_ok=True)


def make_unique_profile_name(recipient_code: str, existing_columns: set[str]) -> str:
    base = f"pv_v3_{recipient_code}"
    name = base
    counter = 2
    while name in existing_columns:
        name = f"{base}_{counter}"
        counter += 1
    existing_columns.add(name)
    return name


def assign_pv_profiles(
    users_dir: Path,
    measurements_csv: Path,
    output_users_dir: Path,
    output_measurements_csv: Path,
    output_report_csv: Path,
    count: int,
    seed: int,
    boiler_definition: str,
    forced_recipients: list[str] | None,
    overwrite: bool,
    pv_scale_min: float,
    pv_scale_max: float,
) -> list[dict[str, Any]]:
    if not measurements_csv.exists():
        raise FileNotFoundError(f"Nem található a mérési CSV: {measurements_csv.resolve()}")

    if pv_scale_min <= 0 or pv_scale_max <= 0:
        raise ValueError("A PV-skálázási korlátoknak pozitívnak kell lenniük.")
    if pv_scale_min > pv_scale_max:
        raise ValueError("PV_SCALE_MIN nem lehet nagyobb, mint PV_SCALE_MAX.")
    if users_dir.resolve() == output_users_dir.resolve():
        raise ValueError(
            "Az eredeti users mappa és a kimeneti users mappa nem lehet azonos. "
            "A hibás korábbi PV-k biztonságos javításához mindig az eredeti "
            "Input/Users mappából kell újraépíteni a users_v3 mappát."
        )
    if measurements_csv.resolve() == output_measurements_csv.resolve():
        raise ValueError(
            "Az eredeti és a kimeneti mérési CSV nem lehet azonos."
        )

    users = load_users(users_dir, boiler_definition)
    df = pd.read_csv(measurements_csv)
    if df.empty:
        raise ValueError("A mérési CSV üres.")

    df.columns = [str(c) for c in df.columns]
    columns = set(df.columns)

    donors = [u for u in users if u.has_pv and u.pv_profile in columns]
    candidates = [u for u in users if u.has_boiler and not u.has_pv]

    if not donors:
        pv_without_profile = [u.code for u in users if u.has_pv]
        raise RuntimeError(
            "Nincs olyan eredeti PV-s háztartás, amelynek pv.profile értéke szerepel a CSV oszlopai között. "
            f"PV-s YAML-ok: {pv_without_profile}"
        )

    recipients = select_recipients(candidates, count, seed, forced_recipients)

    prepare_output_paths(
        output_users_dir,
        output_measurements_csv,
        output_report_csv,
        overwrite,
    )

    # A kimenetet minden futáskor az eredeti YAML-okból építjük újra.
    # Emiatt egy korábbi, téves PV-hozzárendelés nem marad bent a users_v3-ban.
    shutil.copytree(users_dir, output_users_dir)

    donor_order = sorted(donors, key=lambda u: u.code)
    report: list[dict[str, Any]] = []

    for recipient in recipients:
        target_annual_kwh = recipient.ue_annual_kwh + recipient.boiler_annual_kwh

        if target_annual_kwh <= 0:
            raise ValueError(
                f"A(z) {recipient.code} háztartás éves UE + bojler fogyasztása nem pozitív."
            )

        donor_options: list[tuple[float, float, UserRecord]] = []

        for candidate_donor in donor_order:
            donor_annual_kwh = candidate_donor.pv_annual_kwh
            if donor_annual_kwh <= 0:
                continue

            ideal_scale = target_annual_kwh / donor_annual_kwh
            applied_scale = min(pv_scale_max, max(pv_scale_min, ideal_scale))
            scaled_annual_kwh = donor_annual_kwh * applied_scale
            difference_kwh = abs(scaled_annual_kwh - target_annual_kwh)

            # Elsődleges rendezés: éves eltérés.
            # Másodlagos rendezés: minél kisebb skálázás legyen szükséges.
            donor_options.append(
                (difference_kwh, abs(applied_scale - 1.0), candidate_donor)
            )

        if not donor_options:
            raise RuntimeError("Nincs pozitív éves termelésű PV-donor.")

        donor_options.sort(key=lambda item: (item[0], item[1], item[2].code))
        _, _, donor = donor_options[0]

        donor_pv = donor.units.get("pv")
        if not isinstance(donor_pv, dict):
            raise RuntimeError(f"Hibás donor PV blokk: {donor.path}")

        donor_profile = str(donor_pv["profile"])
        new_profile = make_unique_profile_name(recipient.code, columns)

        donor_annual_kwh = donor.pv_annual_kwh
        ideal_scale = target_annual_kwh / donor_annual_kwh
        applied_scale = min(pv_scale_max, max(pv_scale_min, ideal_scale))

        # A donor PV-idősora skálázva kerül az új, egyedi CSV-oszlopba.
        donor_series = pd.to_numeric(df[donor_profile], errors="coerce").fillna(0.0)
        df[new_profile] = donor_series * applied_scale

        recipient_copy_path = output_users_dir / recipient.path.relative_to(users_dir)
        copied_data = yaml.safe_load(recipient_copy_path.read_text(encoding="utf-8")) or {}
        copied_units = copied_data.get("units", copied_data)

        # A donor teljes pv blokkját másoljuk, így minden eredeti PV-paraméter megmarad.
        new_pv_block = copy.deepcopy(donor_pv)
        new_pv_block["profile"] = new_profile
        new_pv_block["size"] = donor_annual_kwh * applied_scale
        copied_units["pv"] = new_pv_block

        if "units" in copied_data:
            copied_data["units"] = copied_units
        else:
            copied_data = copied_units

        recipient_copy_path.write_text(
            yaml.safe_dump(
                copied_data,
                allow_unicode=True,
                sort_keys=False,
                default_flow_style=False,
            ),
            encoding="utf-8",
        )

        scaled_pv_annual_kwh = donor_annual_kwh * applied_scale
        difference_kwh = scaled_pv_annual_kwh - target_annual_kwh
        relative_difference_pct = (
            100.0 * difference_kwh / target_annual_kwh
            if target_annual_kwh > 0
            else 0.0
        )

        report.append(
            {
                "recipient_household": recipient.code,
                "recipient_yaml": recipient.path.name,
                "ue_annual_kwh": recipient.ue_annual_kwh,
                "boiler_annual_kwh": recipient.boiler_annual_kwh,
                "target_consumption_kwh": target_annual_kwh,
                "donor_household": donor.code,
                "donor_pv_profile": donor_profile,
                "donor_original_pv_kwh": donor_annual_kwh,
                "pv_scale_factor": applied_scale,
                "scaled_pv_annual_kwh": scaled_pv_annual_kwh,
                "difference_kwh": difference_kwh,
                "relative_difference_pct": relative_difference_pct,
                "new_pv_profile": new_profile,
            }
        )

    # Az eredeti CSV érintetlen marad; csak a v2 fájlt írjuk ki.
    df.to_csv(output_measurements_csv, index=False)

    with output_report_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(report[0].keys()) if report else ["recipient_household"])
        writer.writeheader()
        writer.writerows(report)

    print("\n========== PV-KIOSZTÁS ELKÉSZÜLT ==========")
    print(f"Eredeti YAML-ok száma       : {len(users)}")
    print(f"Eredeti PV-s donorok száma  : {len(donors)}")
    print(f"Bojleres, nem PV-s jelöltek : {len(candidates)}")
    print(f"Új PV-s háztartások száma   : {len(recipients)}")
    print(
        "Kimeneti PV-s háztartások     : "
        f"{sum(user.has_pv for user in users) + len(recipients)} "
        "(eredeti PV-sek + most kijelölt háztartások)"
    )
    print("\nPV-T KAPOTT HÁZTARTÁSOK:")
    for row in report:
        print(
            f"  {row['recipient_household']} <- donor: {row['donor_household']} "
            f"| cél: {row['target_consumption_kwh']:.1f} kWh "
            f"| PV: {row['scaled_pv_annual_kwh']:.1f} kWh "
            f"| skála: {row['pv_scale_factor']:.4f} "
            f"| eltérés: {row['relative_difference_pct']:+.2f}%"
        )

    print("\nKIMENETEK:")
    print(f"  CSV       : {output_measurements_csv}")
    print(f"  Users mappa: {output_users_dir}")
    print(f"  Riport    : {output_report_csv}")
    print("===========================================\n")

    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Meglévő PV-s háztartások profiljaiból PV-t rendel bojleres, "
            "eredetileg nem PV-s felhasználókhoz."
        )
    )
    parser.add_argument("--users-dir", type=Path, default=USERS_DIR)
    parser.add_argument("--measurements", type=Path, default=MEASUREMENTS_CSV)
    parser.add_argument("--output-users-dir", type=Path, default=OUTPUT_USERS_DIR)
    parser.add_argument("--output-measurements", type=Path, default=OUTPUT_MEASUREMENTS_CSV)
    parser.add_argument("--output-report", type=Path, default=OUTPUT_REPORT_CSV)
    parser.add_argument("-n", "--count", type=int, default=N_NEW_PV_USERS)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--boiler-definition",
        choices=("hss_only", "hss_or_ut"),
        default=BOILER_DEFINITION,
    )
    parser.add_argument(
        "--recipients",
        nargs="*",
        default=RECIPIENT_HOUSEHOLDS,
        help=(
            "Konkrét háztartáskódok; felülírja a --count értékét. "
            "Elfogad teljes load_ azonosítót, számazonosítót vagy egyedi végződést."
        ),
    )
    parser.add_argument("--overwrite", action="store_true", default=OVERWRITE_OUTPUTS)
    parser.add_argument("--pv-scale-min", type=float, default=PV_SCALE_MIN)
    parser.add_argument("--pv-scale-max", type=float, default=PV_SCALE_MAX)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    assign_pv_profiles(
        users_dir=args.users_dir,
        measurements_csv=args.measurements,
        output_users_dir=args.output_users_dir,
        output_measurements_csv=args.output_measurements,
        output_report_csv=args.output_report,
        count=args.count,
        seed=args.seed,
        boiler_definition=args.boiler_definition,
        forced_recipients=args.recipients,
        overwrite=args.overwrite,
        pv_scale_min=args.pv_scale_min,
        pv_scale_max=args.pv_scale_max,
    )


if __name__ == "__main__":
    main()
