"""Az 5R2C épületmodellek két időállandójának ábrázolása.

Windows / PyCharm: Shift+F10.

Szükséges csomagok:
    pip install numpy pandas matplotlib pyyaml

Nem szükséges Gurobi vagy optimalizálási eredmény.

Forrás:
    SOURCE = "csv":
        A household_parameters.csv összes épülete.

    SOURCE = "yaml":
        A user YAML-ekben szereplő épületparaméterek.
        A model_5r2c blokk elsőbbséget élvez, ha meg van adva.

Az időállandók a folytonos, kapcsolt épületmodell sajátértékeiből:
    tau_i = -1 / lambda_i

Mértékegységek:
    C: kWh/K
    H: kW/K
    A sajátértékei: 1/h
    időállandók: óra
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


# ============================================================
# BEÁLLÍTÁSOK
# ============================================================

# Ez a fájl közvetlenül a RecOpt_v2 projekt gyökerében legyen,
# az add_to_pv_hp.py és a többi inputmódosító kód mellett.
PROJECT_ROOT = Path(__file__).resolve().parent

PROJECT_ROOT = Path(__file__).resolve().parent

INPUT_DIR = PROJECT_ROOT / "Input"

SOURCE = "yaml"

USERS_DIR = INPUT_DIR / "users_v4"

SIM_YAML = (
    INPUT_DIR
    / "simulation_config_disaggregated_with_userlist_v4.yaml"
)

ONLY_PV_HP = True

OUTPUT_DIR = PROJECT_ROOT / "figures_buildings"

DPI = 220
SAVE_PDF = False

# ============================================================


RAW_FIELDS = (
    "Aref",
    "h",
    "Htr",
    "Hve",
    "U_window",
    "window_total",
)

MODEL_FIELDS = (
    "C_air_kWh_per_K",
    "C_mass_kWh_per_K",
    "H_em_kW_per_K",
    "H_w_kW_per_K",
    "H_ms_kW_per_K",
    "H_is_kW_per_K",
    "H_v_kW_per_K",
)


def positive_values(data, fields, label):
    """Pozitív, véges épületparaméterek ellenőrzése."""

    missing = [name for name in fields if name not in data]

    if missing:
        raise ValueError(
            f"{label}: hiányzó épületparaméterek: "
            + ", ".join(missing)
        )

    try:
        result = np.asarray(
            [data[name] for name in fields],
            dtype=float,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{label}: az épületparaméterek számok legyenek."
        ) from exc

    if not np.isfinite(result).all() or np.any(result <= 0):
        raise ValueError(
            f"{label}: pozitív, véges épületparaméterek szükségesek."
        )

    return result


def continuous_matrix(raw, label):
    """A HP-modellben használt folytonos 5R2C állapotmátrix.

    Állapotvektor:
        x = [T_air, T_mass]

    Az időállandókhoz a homogén rendszer kell:
        dx/dt = A @ x

    A külső hőmérséklet, napsugárzás és fűtés a bemenetek.
    Ezek nem módosítják a konstans paraméterű A mátrix sajátértékeit.
    """

    if raw.get("model_5r2c") is not None:
        model = raw["model_5r2c"]

        if not isinstance(model, dict):
            raise ValueError(
                f"{label}: a model_5r2c blokk nem szótár."
            )

    else:
        area, height, htr, hve, uw, aw = positive_values(
            raw,
            RAW_FIELDS,
            label,
        )

        # Az alapadatokból számított hőátviteli tényezők, W/K.
        hw = uw * aw
        hop = htr - hw
        hms = 9.10 * 3.0 * area
        his = 3.45 * 4.5 * area

        if not 0 < hop < hms:
            raise ValueError(
                f"{label}: hibás hőátviteli adatok.\n"
                "Szükséges: "
                "0 < Htr - U_window*window_total < H_ms\n"
                f"Kapott Htr_op = {hop:.6g} W/K; "
                f"H_ms = {hms:.6g} W/K."
            )

        hem = 1.0 / (1.0 / hop - 1.0 / hms)

        # Ugyanazok a képletek, mint a HPOnly/hp_core.py fájlban.
        model = {
            "C_air_kWh_per_K": (
                5.0 * 0.34 * area * height / 1000.0
            ),
            "C_mass_kWh_per_K": (
                280.0 / 3600.0 * area
            ),
            "H_em_kW_per_K": hem / 1000.0,
            "H_w_kW_per_K": hw / 1000.0,
            "H_ms_kW_per_K": hms / 1000.0,
            "H_is_kW_per_K": his / 1000.0,
            "H_v_kW_per_K": hve / 1000.0,
        }

    ca, cm, hem, hw, hms, his, hv = positive_values(
        model,
        MODEL_FIELDS,
        label,
    )

    denominator = hw + hms + his

    matrix = np.array(
        [
            [
                (
                    his * (his / denominator - 1.0) - hv
                ) / ca,
                his * hms / denominator / ca,
            ],
            [
                hms * his / denominator / cm,
                (
                    hms * (hms / denominator - 1.0) - hem
                ) / cm,
            ],
        ],
        dtype=float,
    )

    return matrix, model


def load_from_csv():
    """Az épületparaméterek betöltése CSV-ből."""

    if not PARAMETERS_CSV.is_file():
        raise FileNotFoundError(
            f"Nem található az épületparaméter-fájl:\n"
            f"{PARAMETERS_CSV}\n\n"
            "Állítsd be a fájl elején az INPUT_DIR vagy "
            "a PARAMETERS_CSV változót."
        )

    data = pd.read_csv(
        PARAMETERS_CSV,
        encoding="utf-8-sig",
    )

    if data.empty:
        raise ValueError(
            f"Üres CSV: {PARAMETERS_CSV}"
        )

    id_column = next(
        (
            column
            for column in (
                "household_key",
                "household",
                "key",
                "profile_id",
                "id",
            )
            if column in data.columns
        ),
        None,
    )

    if id_column is not None:
        # Megőrizzük az azonosítók esetleges vezető nulláit.
        data = pd.read_csv(
            PARAMETERS_CSV,
            encoding="utf-8-sig",
            dtype={id_column: str},
        )
    else:
        print(
            "[INFO] A CSV-ben nincs háztartásazonosító. "
            "Az S01, S02, ... jelölés az adatsorok sorszáma."
        )

    buildings = []

    for position, (_, row) in enumerate(data.iterrows(), start=1):
        if id_column is None:
            label = f"S{position:02d}"
        else:
            if pd.isna(row[id_column]):
                raise ValueError(
                    f"A(z) {position}. adatsorban hiányzik "
                    "az épületazonosító."
                )

            label = str(row[id_column]).strip()

            if not label:
                raise ValueError(
                    f"A(z) {position}. adatsorban üres "
                    "az épületazonosító."
                )

        buildings.append(
            (label, row.to_dict())
        )

    return buildings


def user_yaml_paths():
    """A vizsgálandó user YAML-ek meghatározása."""

    if not USERS_DIR.is_dir():
        raise FileNotFoundError(
            f"Nem található a Users mappa:\n{USERS_DIR}"
        )

    if SIM_YAML is None:
        return sorted(
            {
                *USERS_DIR.glob("*.yaml"),
                *USERS_DIR.glob("*.yml"),
            }
        )

    if not SIM_YAML.is_file():
        raise FileNotFoundError(
            f"Nem található a szimulációs YAML:\n{SIM_YAML}\n\n"
            "Javítsd a SIM_YAML útvonalát, vagy állítsd None-ra "
            "az összes user YAML vizsgálatához."
        )

    config = yaml.safe_load(
        SIM_YAML.read_text(encoding="utf-8-sig")
    ) or {}

    user_list = config.get("users_list")

    if not isinstance(user_list, list):
        raise ValueError(
            f"{SIM_YAML}: users_list lista szükséges."
        )

    paths = []

    for raw_key in user_list:
        key = str(raw_key)

        if key.lower() in {
            "battery",
            "bess",
            "community",
            "load_battery",
        }:
            continue

        candidates = (
            USERS_DIR / key,
            USERS_DIR / f"{key}.yaml",
            USERS_DIR / f"{key}.yml",
        )

        matches = [
            path for path in candidates
            if path.is_file()
        ]

        if len(matches) != 1:
            raise ValueError(
                f"{key}: hiányzó vagy nem egyértelmű "
                f"user YAML a következő mappában:\n{USERS_DIR}"
            )

        paths.append(matches[0])

    return paths


def load_from_yaml():
    """Az épületparaméterek betöltése a user YAML-ekből."""

    buildings = []

    for path in user_yaml_paths():
        config = yaml.safe_load(
            path.read_text(encoding="utf-8-sig")
        ) or {}

        units = config.get("units", config) or {}
        hp = units.get("heat_pump")

        if not hp:
            continue

        if ONLY_PV_HP and not units.get("pv"):
            continue

        raw = hp.get("household_parameters") or {}

        buildings.append(
            (path.stem, raw)
        )

    return buildings


def calculate_time_constants(buildings):
    """A két modális időállandó kiszámítása sajátértékekből."""

    records = []

    for label, raw in buildings:
        matrix, model = continuous_matrix(
            raw,
            label,
        )

        eigenvalues = np.linalg.eigvals(matrix)

        if (
            not np.isfinite(eigenvalues).all()
            or np.max(np.abs(eigenvalues.imag)) > 1e-9
            or np.any(eigenvalues.real >= 0)
        ):
            raise ValueError(
                f"{label}: nem stabil vagy nem valós "
                f"sajátértékek:\n{eigenvalues}"
            )

        # A mátrix egysége 1/h, ezért az eredmény órában adódik.
        taus = np.sort(
            -1.0 / eigenvalues.real
        )

        tau_fast_h = float(taus[0])
        tau_slow_h = float(taus[1])

        record = {
            "building": label,
            **{
                name: raw.get(name, np.nan)
                for name in RAW_FIELDS
            },
            **{
                name: float(model[name])
                for name in MODEL_FIELDS
            },
            "tau_fast_h": tau_fast_h,
            "tau_slow_h": tau_slow_h,
            "tau_fast_min": tau_fast_h * 60.0,
            "tau_slow_days": tau_slow_h / 24.0,
            "lambda_fast_per_h": -1.0 / tau_fast_h,
            "lambda_slow_per_h": -1.0 / tau_slow_h,
        }

        records.append(record)

    if not records:
        raise ValueError(
            "Nincs ábrázolható épület. "
            "Ellenőrizd a forrást, az útvonalakat "
            "és a PV/HP szűrést!"
        )

    data = pd.DataFrame(records)

    if data["building"].duplicated().any():
        raise ValueError(
            "Ismétlődő épületazonosító található."
        )

    return data


def save_figure(fig, filename):
    import os
    import tempfile
    from datetime import datetime

    output = Path(OUTPUT_DIR)
    output.mkdir(parents=True, exist_ok=True)

    formats = ["png"]

    if SAVE_PDF:
        formats.append("pdf")

    try:
        for extension in formats:
            target = output / f"{filename}.{extension}"

            # A Windows alatt a fájlt bezárjuk a lecserélés előtt.
            with tempfile.NamedTemporaryFile(
                mode="w+b",
                prefix="plot_",
                suffix=f".{extension}",
                dir=output,
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)

            try:
                fig.savefig(
                    temporary_path,
                    format=extension,
                    dpi=DPI,
                    bbox_inches="tight",
                )

                try:
                    os.replace(temporary_path, target)

                except OSError as exc:
                    if exc.errno not in (13, 22):
                        raise

                    # Például megnyitott vagy zárolt célfájl esetén.
                    stamp = datetime.now().strftime(
                        "%Y%m%d_%H%M%S_%f"
                    )

                    alternative = output / (
                        f"{filename}_{stamp}.{extension}"
                    )

                    os.replace(
                        temporary_path,
                        alternative,
                    )

                    print(
                        f"[FIGYELEM] Nem sikerült lecserélni: "
                        f"{target}\n"
                        f"Ok: {exc}\n"
                        f"Új néven elmentve: {alternative}"
                    )

                else:
                    print(f"[OK] {target}")

            finally:
                if temporary_path.exists():
                    temporary_path.unlink()

    finally:
        plt.close(fig)

    save_figure(
        fig,
        "idoallandok_epuletenkent",
    )


def plot_scatter(data):
    """A gyors és lassú időállandó kapcsolata."""

    fig, ax = plt.subplots(
        figsize=(9, 6),
        layout="constrained",
    )

    areas = pd.to_numeric(
        data["Aref"],
        errors="coerce",
    )

    if np.isfinite(areas.to_numpy()).all():
        points = ax.scatter(
            data["tau_fast_min"],
            data["tau_slow_h"],
            c=areas,
            cmap="viridis",
            s=70,
            edgecolors="white",
            linewidths=0.7,
        )

        fig.colorbar(
            points,
            ax=ax,
            label="Alapterület [m²]",
        )

    else:
        ax.scatter(
            data["tau_fast_min"],
            data["tau_slow_h"],
            s=70,
            color="#3485A5",
            edgecolors="white",
        )

    if len(data) <= 40:
        for _, row in data.iterrows():
            ax.annotate(
                short_label(row["building"]),
                (
                    row["tau_fast_min"],
                    row["tau_slow_h"],
                ),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
            )

    ax.set_xlabel(
        "Gyors időállandó [perc]"
    )
    ax.set_ylabel(
        "Lassú időállandó [óra]"
    )
    ax.set_title(
        "Gyors és lassú termikus időállandó"
    )

    ax.grid(alpha=0.2)

    save_figure(
        fig,
        "idoallandok_scatter",
    )


def plot_distributions(data):
    """A két időállandó eloszlása."""

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 4.5),
        layout="constrained",
    )

    specifications = (
        (
            "tau_fast_min",
            "Gyors időállandó [perc]",
            "#3485A5",
        ),
        (
            "tau_slow_h",
            "Lassú időállandó [óra]",
            "#D18B40",
        ),
    )

    bins = min(
        10,
        max(1, int(np.sqrt(len(data)))),
    )

    for ax, (column, xlabel, color) in zip(
        axes,
        specifications,
    ):
        values = data[column].to_numpy(dtype=float)
        average = float(values.mean())

        ax.hist(
            values,
            bins=bins,
            color=color,
            edgecolor="white",
        )

        ax.axvline(
            average,
            color="#183A50",
            linestyle="--",
            linewidth=1.5,
            label=f"Átlag: {average:.2f}",
        )

        ax.set_xlabel(xlabel)
        ax.set_ylabel("Épületek száma")
        ax.legend()
        ax.grid(
            axis="y",
            alpha=0.2,
        )
        ax.set_axisbelow(True)

    save_figure(
        fig,
        "idoallandok_eloszlas",
    )

def plot_time_constant_response(data, building=None):
    """A két termikus módus felfutása és kezdeti érintője.

    building=None: az első betöltött épület.
    Egyébként pontos CSV-sorazonosító vagy YAML-fájlnév kiterjesztés nélkül.

    Ezek normalizált modális görbék, nem közvetlenül
    a belső levegő hőmérsékletének lépcsőválaszai.
    """

    if building is None:
        row = data.iloc[0]
    else:
        selected = data.loc[
            data["building"].astype(str).eq(str(building))
        ]

        if len(selected) != 1:
            raise ValueError(
                f"Nem található egyértelmű épület: {building}\n"
                f"Választható: {data['building'].tolist()}"
            )

        row = selected.iloc[0]

    label = str(row["building"])

    specifications = (
        (
            float(row["tau_fast_min"]),
            "perc",
            "Gyors termikus módus",
            "#3485A5",
        ),
        (
            float(row["tau_slow_h"]),
            "óra",
            "Lassú termikus módus",
            "#D18B40",
        ),
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 5),
        layout="constrained",
    )

    value_at_tau = 1.0 - np.exp(-1.0)

    for ax, (tau, unit, title, color) in zip(
        axes,
        specifications,
    ):
        time = np.linspace(0.0, 5.0 * tau, 600)
        response = 1.0 - np.exp(-time / tau)

        # A kezdeti érintő: y = t / tau.
        tangent_time = np.linspace(
            0.0,
            1.08 * tau,
            100,
        )
        tangent = tangent_time / tau

        ax.plot(
            time,
            response,
            color=color,
            linewidth=2.5,
            label="Normalizált felfutás",
        )

        ax.plot(
            tangent_time,
            tangent,
            color="#64748B",
            linestyle="--",
            linewidth=1.6,
            label="Kezdeti érintő",
        )

        ax.axhline(
            1.0,
            color="#183A50",
            linestyle=":",
            linewidth=1.2,
            label="Állandósult érték",
        )

        ax.axvline(
            tau,
            color="#CB5555",
            linestyle="--",
            linewidth=1.2,
            label=f"τ = {tau:.2f} {unit}",
        )

        ax.hlines(
            value_at_tau,
            0.0,
            tau,
            color=color,
            linestyle=":",
            linewidth=1.2,
        )

        # A valódi görbe értéke t = tau esetén.
        ax.scatter(
            [tau],
            [value_at_tau],
            color=color,
            s=65,
            zorder=5,
        )

        # A kezdeti érintő és az állandósult érték metszése.
        ax.scatter(
            [tau],
            [1.0],
            color="#CB5555",
            marker="x",
            s=80,
            linewidths=2,
            zorder=5,
        )

        ax.annotate(
            "63,2%",
            xy=(tau, value_at_tau),
            xytext=(12, -22),
            textcoords="offset points",
            fontsize=10,
        )

        ax.set(
            title=title,
            xlabel=f"Idő [{unit}]",
            ylabel="Normalizált változás",
            xlim=(0.0, 5.0 * tau),
            ylim=(0.0, 1.12),
        )

        ax.grid(alpha=0.2)
        ax.legend(
            loc="lower right",
            fontsize=8,
        )

    fig.suptitle(
        f"{label} – időállandók és kezdeti érintők"
    )

    safe_label = "".join(
        character
        if character.isalnum() or character in "-_."
        else "_"
        for character in label
    )

    save_figure(
        fig,
        f"idoallando_valaszgorbek_{safe_label}",
    )


def main():
    source = str(SOURCE).strip().lower()

    if source == "csv":
        buildings = load_from_csv()

    elif source == "yaml":
        buildings = load_from_yaml()

    else:
        raise ValueError(
            'A SOURCE értéke "csv" vagy "yaml" legyen.'
        )

    if DPI <= 0:
        raise ValueError(
            "A DPI pozitív szám legyen."
        )

    data = calculate_time_constants(buildings)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        OUTPUT_DIR
        / "building_time_constants.csv"
    )

    data.to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"[OK] {csv_path}")

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    plot_by_building(data)
    plot_scatter(data)
    plot_distributions(data)
    plot_time_constant_response(
        data,
        building="load_0420144653433567",
    )

    summary = data[
        [
            "tau_fast_min",
            "tau_slow_h",
        ]
    ].agg(
        [
            "min",
            "mean",
            "max",
        ]
    )

    summary = summary.rename(
        index={
            "min": "Minimum",
            "mean": "Átlag",
            "max": "Maximum",
        },
        columns={
            "tau_fast_min": "Gyors [perc]",
            "tau_slow_h": "Lassú [óra]",
        },
    )

    print()
    print(f"Feldolgozott épületek: {len(data)}")
    print(summary.round(4).to_string())
    print()
    print(f"Eredmények: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()