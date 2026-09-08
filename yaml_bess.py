import argparse
from pathlib import Path
import re
import pandas as pd
import yaml

# ---- Beállítások ----
USERS_DIR = Path(r"Input/users_v2")
MEASUREMENTS_CSV = Path(r"Input/measurements_disaggregated_v2.csv")
DT = 0.25  # A PV-profilok mértékegysége kWh / 15 perc.

# "manual": a MANUAL_BESS_BLOCK értékeivel frissíti a meglévő BESS-blokkokat.
# "pv_peak": a meglévő BESS-nél P_bess = P_pv,peak és
#             E_bess = FILL_HOURS * P_bess.
SIZING_MODE = "pv_peak"
FILL_HOURS = 3.0

MANUAL_BESS_BLOCK = {
    "bess_size": "12.0",
    "eta_bess_in": "0.98",
    "eta_bess_out": "0.96",
    "eta_bess_stor": "0.995",
    "eta_self_discharge": "1.0",
    "soc_bess_max": "0.9",
    "soc_bess_min": "0.1",
    "t_bess_min": "2",
}

def get_last_three_digits_from_filename(filename: str) -> str:
    """
    Kiszedi a fájlnévből az utolsó 3 számjegyet.
    Pl.: load_0420144653458813.yaml -> 813
    """
    stem = Path(filename).stem
    digits = re.findall(r"\d", stem)
    if len(digits) >= 3:
        return "".join(digits[-3:])
    return "N/A"


def _format_number(value: float) -> str:
    """Rövid, YAML-ban is jól olvasható számalak."""
    return f"{float(value):g}"


def get_pv_peak_power_kw(
    pv_dict: dict,
    measurements: pd.DataFrame,
    dt: float,
) -> tuple[float, str]:
    """A pv.profile mért idősorából meghatározza a PV csúcsteljesítményét."""
    if not isinstance(pv_dict, dict):
        raise ValueError("A pv szekciónak YAML-szótárnak kell lennie.")
    if dt <= 0:
        raise ValueError("A dt értékének pozitívnak kell lennie.")
    profile = pv_dict.get("profile")
    if profile in (None, ""):
        raise KeyError("A PV-blokkban nincs profile mező.")
    profile = str(profile)
    if profile not in measurements.columns:
        raise KeyError(f"A PV-profil nem található a mérési CSV-ben: {profile}")

    energy_per_step = pd.to_numeric(
        measurements[profile], errors="coerce"
    ).fillna(0.0).clip(lower=0.0)
    peak_kw = float(energy_per_step.max()) / dt
    if peak_kw <= 0:
        raise ValueError(f"A PV-profil csúcsteljesítménye nem pozitív: {profile}")
    return peak_kw, profile


def build_bess_block(
    pv_dict: dict,
    measurements: pd.DataFrame | None,
    mode: str,
    fill_hours: float,
    dt: float = DT,
) -> tuple[dict, float | None, str | None]:
    """Elkészíti a kézi vagy PV-csúcsteljesítmény-alapú BESS-blokkot."""
    mode = str(mode).strip().lower()
    if mode == "manual":
        return MANUAL_BESS_BLOCK.copy(), None, None
    if mode != "pv_peak":
        raise ValueError("A mode csak 'manual' vagy 'pv_peak' lehet.")
    if fill_hours <= 0:
        raise ValueError("A fill_hours értékének pozitívnak kell lennie.")
    if measurements is None:
        raise ValueError("PV-alapú méretezéshez szükséges a mérési CSV.")

    pv_peak_kw, source_profile = get_pv_peak_power_kw(pv_dict, measurements, dt)
    bess_size_kwh = pv_peak_kw * fill_hours

    block = MANUAL_BESS_BLOCK.copy()
    block["bess_size"] = _format_number(bess_size_kwh)
    # A modellekben P_bess,max = bess_size / t_bess_min, ezért ezzel lesz
    # P_bess,max = P_pv,peak.
    block["t_bess_min"] = _format_number(fill_hours)
    return block, pv_peak_kw, source_profile


def main(
    mode: str = SIZING_MODE,
    fill_hours: float = FILL_HOURS,
    users_dir: Path = USERS_DIR,
    measurements_csv: Path = MEASUREMENTS_CSV,
    dt: float = DT,
):
    mode = str(mode).strip().lower()
    if mode not in {"manual", "pv_peak"}:
        raise ValueError("A mode csak 'manual' vagy 'pv_peak' lehet.")
    if fill_hours <= 0:
        raise ValueError("A fill_hours értékének pozitívnak kell lennie.")
    if dt <= 0:
        raise ValueError("A dt értékének pozitívnak kell lennie.")

    measurements = None
    if mode == "pv_peak":
        if not measurements_csv.exists():
            raise FileNotFoundError(
                f"Nem található a PV-profilokat tartalmazó CSV: {measurements_csv.resolve()}"
            )
        measurements = pd.read_csv(measurements_csv)
        measurements.columns = measurements.columns.map(str)

    yaml_files = sorted(list(users_dir.glob("*.yaml")) + list(users_dir.glob("*.yml")))

    if not yaml_files:
        print(f"Nincs YAML fájl itt: {users_dir.resolve()}")
        return

    count_ue = 0
    count_ut = 0
    count_hss = 0
    count_pv = 0
    count_bess_updated = 0
    count_without_bess = 0
    count_bess_without_pv = 0
    count_auto_sizing_skipped = 0

    pv_file_suffixes = []

    for yaml_file in yaml_files:
        with open(yaml_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            print(f"SKIP (nem dict struktúra): {yaml_file.name}")
            continue

        units = data.get("units", data)

        if not isinstance(units, dict):
            print(f"SKIP (nincs értelmezhető units blokk): {yaml_file.name}")
            continue

        has_ue = "ue" in units
        has_ut = "ut" in units
        has_hss = "hss" in units
        has_pv = "pv" in units
        has_bess = "bess" in units

        if has_ue:
            count_ue += 1
        if has_ut:
            count_ut += 1
        if has_hss:
            count_hss += 1
        if has_pv:
            count_pv += 1
            last3 = get_last_three_digits_from_filename(yaml_file.name)
            pv_file_suffixes.append((yaml_file.name, last3))

        modified = False

        if not has_bess:
            # Ez a segédprogram kizárólag a már kiosztott BESS-ek paramétereit
            # frissíti; új háztartáshoz nem hoz létre BESS-blokkot.
            count_without_bess += 1
        elif not has_pv:
            # PV-csúcsteljesítményből csak PV-s háztartás méretezhető. Manual
            # módban is jelezzük az inkonzisztens YAML-t, és nem írjuk át.
            count_bess_without_pv += 1
            print(f"SKIP (van BESS, de nincs PV): {yaml_file.name}")
        else:
            try:
                bess_block, pv_peak_kw, peak_source = build_bess_block(
                    units["pv"],
                    measurements=measurements,
                    mode=mode,
                    fill_hours=fill_hours,
                    dt=dt,
                )
            except (KeyError, ValueError) as exc:
                count_auto_sizing_skipped += 1
                print(f"SKIP (BESS nem méretezhető): {yaml_file.name} | {exc}")
                continue

            # A meglévő blokk ismeretlen/egyedi mezői megmaradnak, a jelen
            # kódban kezelt BESS-paraméterek viszont az új értékeket kapják.
            updated_bess = dict(units["bess"] or {})
            updated_bess.update(bess_block)
            units["bess"] = updated_bess
            if "units" in data:
                data["units"] = units
            else:
                data = units

            modified = True
            count_bess_updated += 1
            sizing_info = ""
            if mode == "pv_peak":
                sizing_info = (
                    f" | P_pv,peak={pv_peak_kw:g} kW (profil: {peak_source})"
                    f" | P_bess={pv_peak_kw:g} kW"
                    f" | E_bess={float(bess_block['bess_size']):g} kWh"
                    f" | feltöltési idő={fill_hours:g} h"
                )
            print(
                f"BESS FRISSÍTVE: {yaml_file.name}"
                f" | utolsó 3 szám: {get_last_three_digits_from_filename(yaml_file.name)}"
                f"{sizing_info}"
            )

        if modified:
            with open(yaml_file, "w", encoding="utf-8") as f:
                yaml.safe_dump(
                    data,
                    f,
                    allow_unicode=True,
                    sort_keys=False,
                    default_flow_style=False
                )

    print("\n--- ÖSSZESÍTÉS ---")
    print(f"méretezési mód: {mode}")
    if mode == "pv_peak":
        print(f"beállított feltöltési idő: {fill_hours:g} h")
    print(f"YAML fájlok száma: {len(yaml_files)}")
    print(f"ue szekciók száma: {count_ue}")
    print(f"ut szekciók száma: {count_ut}")
    print(f"hss szekciók száma: {count_hss}")
    print(f"pv szekciók száma: {count_pv}")
    print(f"frissített meglévő bess blokkok: {count_bess_updated}")
    print(f"bess nélküli, változatlan YAML-ok: {count_without_bess}")
    if count_bess_without_pv:
        print(f"BESS van, de PV nincs; kihagyva: {count_bess_without_pv}")
    if count_auto_sizing_skipped:
        print(f"méretezési hiba miatt kihagyva: {count_auto_sizing_skipped}")

    print("\nPV-s fájlok és az utolsó 3 számjegyük:")
    for fname, last3 in pv_file_suffixes:
        print(f"  {fname} -> {last3}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Meglévő BESS-blokkok frissítése kézi vagy PV-alapú méretezéssel."
    )
    parser.add_argument(
        "--mode",
        choices=("manual", "pv_peak"),
        default=SIZING_MODE,
        help="manual: fix blokk; pv_peak: PV-csúcsteljesítmény-alapú méretezés",
    )
    parser.add_argument(
        "--fill-hours",
        type=float,
        default=FILL_HOURS,
        help="PV-alapú módban E_bess / P_bess [h], például 2, 3 vagy 4",
    )
    parser.add_argument("--users-dir", type=Path, default=USERS_DIR)
    parser.add_argument("--measurements", type=Path, default=MEASUREMENTS_CSV)
    parser.add_argument(
        "--dt",
        type=float,
        default=DT,
        help="Az idősor időlépése órában; 15 perces adatoknál 0.25",
    )
    args = parser.parse_args()
    main(
        mode=args.mode,
        fill_hours=args.fill_hours,
        users_dir=args.users_dir,
        measurements_csv=args.measurements,
        dt=args.dt,
    )