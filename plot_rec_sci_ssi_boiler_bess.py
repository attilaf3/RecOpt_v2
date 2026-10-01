"""SCI–SSI és villanyszámla összehasonlítás, bojler és tipikus BESS-nap.

Tedd a compare_rec_cases.py mellé, állítsd be a BEÁLLÍTÁSOK részt, majd futtasd:
    python plot_rec_sci_ssi_boiler_bess.py

Az alapadatokat a compare_rec_cases.py CASES listájából olvassa. Az ábrák
külön, írható kimeneti mappába kerülnek; az eredménymappákhoz nem nyúl.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, MultipleLocator
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from compare_rec_cases import (
    CASES, REFERENCE_BY_FAMILY, ALIASES, CaseConfig,
    find_household_summary, normalize_case,
)


# ============================== BEÁLLÍTÁSOK ==============================
# A CASES elérési útjait a compare_rec_cases.py elején állítsd be.
OUTPUT_DIR = Path(__file__).resolve().parent / "results_rec_extra_figures"
HOUSEHOLD = "load_0420144888315704"  # household_key VAGY household/user_name
BOILER_BASELINE_CASE = "0-I"
BOILER_OPT_CASE = "3d-I"
BATTERY_CASE = "3d-I"
BATTERY_HOUSEHOLD: str | None = "load_0420144653458813"  # None: az első aktív BESS-es háztartás
FIRST_DAY = "2021-01-01"           # a profil első napjának naptári dátuma
BOILER_DAY = "2021-06-15"          # mindkét profil ugyanazon napja
BATTERY_DAY: str | None = None    # None: medián napi kilengésű aktív nap
DT_HOURS = 0.25                   # 15 perces profil
SAVE_PDF = False
# Azonos háztartások számláinak közvetlen összehasonlítása a box ploton.
BOXPLOT_CASES = ("0-I", "3d-I", "4d-K-C")
# ========================================================================

EPS = 1e-9
# A négy szín a szcenárió jellegét jelzi. Az eszközök háztartásonként
# átfedhetnek, így egy egész esetet nem lehet egyértelműen PV/Bojler/BESS
# háztartáscsoportként besorolni.
CASE_GROUP = {
    "0-I": "Alapeset",
    "0-K": "Alapeset",
    "1d-I": "Szabályalapú",
    "1d-K": "Szabályalapú",
    "3d-I": "Egyéni optimum",
    "4d-K-C": "Közösségi optimum",
    "4d-K-E": "Közösségi optimum",
}
COLORS = {
    "Alapeset": "#64748B",
    "Szabályalapú": "#3B82A0",
    "Egyéni optimum": "#D18B40",
    "Közösségi optimum": "#3B9474",
}
MARKERS = {"I": "o", "K": "s"}


def _configuration(label: str) -> CaseConfig:
    matches = [case for case in CASES if case.label == label]
    if len(matches) != 1:
        raise ValueError(f"A CASES listában egyetlen '{label}' esetnek kell lennie.")
    return matches[0]


def _raw_summary(case: CaseConfig) -> tuple[Path, pd.DataFrame]:
    source = find_household_summary(case.path)
    return source, pd.read_csv(source, dtype={"household_key": str, "user_key": str})


def _matching_row(frame: pd.DataFrame, identifier: str, case: str) -> pd.Series:
    columns = [col for col in ("household_key", "user_key", "household", "user_name", "name") if col in frame]
    if not columns:
        raise ValueError(f"{case}: nincs háztartásazonosító oszlop.")
    mask = pd.Series(False, index=frame.index)
    for col in columns:
        mask |= frame[col].astype(str).str.strip().eq(str(identifier).strip())
    matched = frame.loc[mask]
    if len(matched) != 1:
        raise ValueError(f"{case}: '{identifier}' azonosítóval {len(matched)} háztartás található.")
    return matched.iloc[0]


def _names_for_household(identifier: str, *cases: CaseConfig) -> list[str]:
    """A YAML household_key és a kimeneti háztartásnév összerendelése."""
    names = [str(identifier).strip()]
    for case in cases:
        try:
            _, raw = _raw_summary(case)
        except FileNotFoundError:
            continue
        rows = raw.loc[
            raw[[c for c in ("household_key", "user_key") if c in raw]]
            .astype(str).eq(str(identifier)).any(axis=1)
        ] if any(c in raw for c in ("household_key", "user_key")) else raw.iloc[0:0]
        if len(rows) == 1:
            for col in ("household", "user_name", "name"):
                if col in rows and pd.notna(rows.iloc[0][col]):
                    names.append(str(rows.iloc[0][col]).strip())
    return list(dict.fromkeys(names))


def _ratio(numerator: pd.Series, denominator: pd.Series) -> float:
    den = float(denominator.sum())
    return float(numerator.sum()) / den if den > EPS else float("nan")


def sci_ssi_points() -> pd.DataFrame:
    """Esetenként egy SCI–SSI pont a közös háztartások összegezett energiáiból."""
    loaded = {}
    for case in CASES:
        try:
            source, raw = _raw_summary(case)
            loaded[case.label] = normalize_case(raw, case, source)
        except FileNotFoundError:
            print(f"[FIGYELEM] SCI–SSI: hiányzó eset kihagyva: {case.label}")
    if not loaded:
        raise FileNotFoundError("Egyik eset household/per_user összesítője sem található.")
    common = set.intersection(*(set(f["household"]) for f in loaded.values()))
    if not common:
        raise ValueError("Nincs közös háztartás a betöltött esetekben.")
    rows = []
    for case in CASES:
        if case.label not in loaded:
            continue
        frame = loaded[case.label]
        frame = frame[frame["household"].isin(common)]
        if frame["pv_kwh"].sum() <= EPS or frame["total_load_kwh"].sum() <= EPS:
            print(f"[FIGYELEM] SCI–SSI: nincs értelmezhető PV vagy fogyasztás: {case.label}")
            continue
        group = CASE_GROUP.get(case.label, "Egyéb eset")
        rows.append({
            "case": case.label, "family": case.family, "category": group,
            "n_households": len(frame),
            "SCI": _ratio(frame["used_pv_kwh"], frame["pv_kwh"]),
            "SSI": _ratio(frame["local_supply_kwh"], frame["total_load_kwh"]),
        })
    if not rows:
        raise ValueError("Nincs olyan eset, amelyhez SCI és SSI számolható.")
    points = pd.DataFrame(rows)
    points.to_csv(OUTPUT_DIR / "sci_ssi_points.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.5, 4.0), layout="constrained")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_axisbelow(True)
    ax.grid(which="major", color="#DCE3EB", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#94A3B8")
    ax.tick_params(colors="#475569", length=0, pad=5, labelsize=8.5)

    x_max = min(1.0, max(0.4, np.ceil((points["SSI"].max() + 0.08) / 0.1) * 0.1))
    for row in points.itertuples(index=False):
        color = COLORS.get(row.category, "#536078")
        marker = MARKERS.get(row.family, "D")
        ax.scatter(row.SSI, row.SCI, s=105, marker=marker, color=color,
                   edgecolor="white", linewidth=1.3, zorder=4)

    ax.set(xlabel="SSI · önellátás", ylabel="SCI · önfogyasztás")
    ax.set_title("SCI–SSI", loc="left", fontsize=12,
                 fontweight="semibold", color="#182B3D", pad=10)
    # A rövidebb vízszintes tartomány jobban elkülöníti a közeli eseteket.
    # A nullát mindkét tengelyen megtartjuk, hogy az arányok olvashatók maradjanak.
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, 1.05)
    ax.xaxis.set_major_locator(MultipleLocator(0.1 if x_max <= 0.6 else 0.2))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    group_handles = [Line2D([], [], marker="o", ls="", markerfacecolor=c,
                            markeredgecolor="white", markersize=9, label=g)
                     for g, c in COLORS.items() if g in set(points["category"])]
    family_handles = [Line2D([], [], marker=marker, ls="", color="#475569",
                             markersize=8, label="Háztartási" if family == "I" else "Közösségi")
                      for family, marker in MARKERS.items() if family in set(points["family"])]
    ax.legend(handles=group_handles + family_handles, loc="upper center",
              bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False, fontsize=8,
              handletextpad=0.35, columnspacing=0.9)
    save(fig, "sci_ssi_pontdiagram")
    return points


def bill_comparison() -> pd.DataFrame:
    """Háztartási éves nettó számlák: átlag és percentilistartomány esetenként."""
    loaded: dict[str, pd.DataFrame] = {}
    for case in CASES:
        try:
            source, raw = _raw_summary(case)
        except FileNotFoundError:
            print(f"[FIGYELEM] Villanyszámla: hiányzó eset kihagyva: {case.label}")
            continue
        if not any(col in raw.columns for col in ALIASES["bill_ft"]):
            print(f"[FIGYELEM] Villanyszámla-oszlop hiányzik: {case.label}")
            continue
        loaded[case.label] = normalize_case(raw, case, source)
    if not loaded:
        raise FileNotFoundError("Nincs használható háztartási villanyszámla-összesítő.")

    common = set.intersection(*(set(f["household"]) for f in loaded.values()))
    if not common:
        raise ValueError("A villanyszámlás esetekben nincs közös háztartás.")
    records = []
    for case in CASES:
        if case.label not in loaded:
            continue
        subset = loaded[case.label].set_index("household").loc[sorted(common), "bill_ft"]
        if not np.isfinite(subset.to_numpy()).all():
            raise ValueError(f"{case.label}: nem véges villanyszámla-érték.")
        q10, q25, median, q75, q90 = np.percentile(subset, [10, 25, 50, 75, 90])
        records.append({
            "case": case.label, "family": case.family,
            "category": CASE_GROUP.get(case.label, "Egyéb eset"),
            "n_households": len(subset), "mean_ft": float(subset.mean()),
            "p10_ft": q10, "p25_ft": q25, "median_ft": median,
            "p75_ft": q75, "p90_ft": q90,
        })
    summary = pd.DataFrame(records)
    for family in summary["family"].unique():
        ref = REFERENCE_BY_FAMILY.get(family)
        reference = summary.loc[summary["case"].eq(ref), "median_ft"]
        if not reference.empty:
            summary.loc[summary["family"].eq(family), "median_change_ft"] = (
                summary.loc[summary["family"].eq(family), "median_ft"] - reference.iloc[0]
            )
    summary.to_csv(OUTPUT_DIR / "villanyszamla_osszehasonlitas.csv", index=False)

    families = [f for f in ("I", "K") if f in set(summary["family"])]
    families.extend(f for f in summary["family"].unique() if f not in families)
    fig, axes = plt.subplots(1, len(families), figsize=(6.5, 4.0),
                             squeeze=False, sharex=True)
    fig.subplots_adjust(left=0.18, right=0.98, top=0.82,
                        bottom=0.26, wspace=0.20)
    fig.patch.set_facecolor("white")
    all_min = min(0.0, float(summary["p10_ft"].min()) / 1000)
    all_max = float(summary["p90_ft"].max()) / 1000
    span = max(1.0, all_max - all_min)
    for ax, family in zip(axes[0], families):
        part = summary.loc[summary["family"].eq(family)].reset_index(drop=True)
        y = np.arange(len(part))
        ax.set_facecolor("#F8FAFC")
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#DCE3EB", lw=0.8)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#94A3B8")
        ax.tick_params(length=0, colors="#475569", pad=5, labelsize=8)
        short_names = {
            "Alapeset": "Alap",
            "Szabályalapú": "Szabályalapú",
            "Egyéni optimum": "Optimalizált",
            "Közösségi optimum": "Optimalizált",
        }
        labels = [short_names.get(category, category) for category in part["category"]]
        ax.set_yticks(y, labels if family == "I" else [""] * len(y))
        ax.invert_yaxis()
        ax.set_title({"I": "Egyéni elszámolás", "K": "Közösségi elszámolás"}.get(family, family),
                     loc="left", color="#182B3D", fontweight="semibold", fontsize=10, pad=9)
        ref = REFERENCE_BY_FAMILY.get(family)
        ref_row = part.loc[part["case"].eq(ref)]
        if not ref_row.empty:
            ax.axvline(float(ref_row.iloc[0]["mean_ft"]) / 1000,
                       color="#A5B1BF", ls=(0, (4, 3)), lw=1.1, zorder=1)
        for index, row in part.iterrows():
            color = COLORS.get(row["category"], "#536078")
            ax.plot([row.p10_ft / 1000, row.p90_ft / 1000], [index, index],
                    color=color, alpha=0.45, lw=2, solid_capstyle="round", zorder=2)
            ax.plot([row.p25_ft / 1000, row.p75_ft / 1000], [index, index],
                    color=color, lw=7, solid_capstyle="round", zorder=3)
            ax.scatter(row.mean_ft / 1000, index, s=65, color=color,
                       edgecolor="white", lw=1.3, zorder=4)
        ax.set_xlim(all_min - span * 0.035, all_max + span * 0.05)

    fig.suptitle("Éves villanyszámlák", fontsize=12,
                 fontweight="semibold", color="#182B3D", ha="left", x=0.18, y=0.96)
    fig.supxlabel("Éves nettó számla [ezer Ft/háztartás]", y=0.15, fontsize=9)
    handles = [
        Line2D([], [], color="#475569", lw=2, marker="o", markersize=7,
               label="Átlag · 10–90% / 25–75%"),
        Line2D([], [], color="#A5B1BF", ls=(0, (4, 3)),
               label="Kategória alapesetének átlaga"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.03), fontsize=8)
    save(fig, "villanyszamla_pont_intervallum")
    plot_total_savings(summary)
    return summary


def plot_total_savings(summary: pd.DataFrame) -> pd.DataFrame:
    """A közös háztartások összesített számlamegtakarítása a saját alapesethez képest."""
    rows = []
    for family, part in summary.groupby("family", sort=False):
        baseline_case = REFERENCE_BY_FAMILY.get(family)
        baseline = part.loc[part["case"].eq(baseline_case)]
        if len(baseline) != 1:
            print(f"[FIGYELEM] Megtakarítás: hiányzik az alapeset: {baseline_case}")
            continue
        base_bill = float(baseline.iloc[0]["mean_ft"] * baseline.iloc[0]["n_households"])
        for row in part.itertuples(index=False):
            total_bill = float(row.mean_ft * row.n_households)
            rows.append({
                "case": row.case, "family": family, "category": row.category,
                "n_households": row.n_households, "baseline_case": baseline_case,
                "total_bill_ft": total_bill, "baseline_bill_ft": base_bill,
                "total_saving_ft": base_bill - total_bill,
            })
    if not rows:
        raise ValueError("A teljes megtakarításhoz nem található alapeset.")
    savings = pd.DataFrame(rows)
    savings.to_csv(OUTPUT_DIR / "villanyszamla_teljes_megtakaritas.csv", index=False)

    families = [f for f in ("I", "K") if f in set(savings["family"])]
    families.extend(f for f in savings["family"].unique() if f not in families)
    fig, axes = plt.subplots(1, len(families), figsize=(6.5, 4.0),
                             squeeze=False, sharex=True)
    fig.subplots_adjust(left=0.18, right=0.97, top=0.81,
                        bottom=0.23, wspace=0.22)
    fig.patch.set_facecolor("white")
    values = savings["total_saving_ft"].to_numpy(dtype=float) / 1e6
    span = max(0.1, float(np.ptp(np.r_[0, values])))
    x_min = min(0.0, float(values.min())) - span * 0.12
    x_max = max(0.0, float(values.max())) + span * 0.16
    names = {
        "Alapeset": "Alap", "Szabályalapú": "Szabályalapú",
        "Egyéni optimum": "Optimalizált", "Közösségi optimum": "Optimalizált",
    }
    for ax, family in zip(axes[0], families):
        part = savings.loc[savings["family"].eq(family)].reset_index(drop=True)
        y = np.arange(len(part))
        ax.set_facecolor("#F8FAFC")
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#DCE3EB", lw=0.8)
        ax.axvline(0, color="#8493A4", lw=1.15, zorder=1)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#94A3B8")
        ax.tick_params(length=0, colors="#475569", pad=5, labelsize=8)
        labels = [names.get(category, category) for category in part["category"]]
        ax.set_yticks(y, labels if family == "I" else [""] * len(y))
        ax.invert_yaxis()
        ax.set_title({"I": "Egyéni elszámolás", "K": "Közösségi elszámolás"}.get(family, family),
                     loc="left", color="#182B3D", fontweight="semibold", fontsize=10, pad=9)
        for index, row in part.iterrows():
            amount = row.total_saving_ft / 1e6
            color = COLORS.get(row.category, "#536078")
            ax.plot([0, amount], [index, index], color=color, lw=2.5,
                    alpha=0.75, solid_capstyle="round", zorder=2)
            ax.scatter(amount, index, s=90, color=color, edgecolor="white",
                       linewidth=1.2, zorder=3)
        ax.set_xlim(x_min, x_max)
    fig.suptitle("Teljes megtakarítás az alapesethez képest", fontsize=12,
                 fontweight="semibold", color="#182B3D", ha="left", x=0.18, y=0.96)
    fig.supxlabel("Éves megtakarítás [millió Ft]", y=0.1, fontsize=9)
    save(fig, "villanyszamla_teljes_megtakaritas")
    return savings


def bill_optimization_boxplot() -> pd.DataFrame:
    """Alapeset és két optimum: éves nettó számla ugyanazon háztartásokra."""
    loaded: dict[str, pd.DataFrame] = {}
    for label in BOXPLOT_CASES:
        case = _configuration(label)
        try:
            source, raw = _raw_summary(case)
        except FileNotFoundError:
            print(f"[FIGYELEM] Box plot: hiányzó eset kihagyva: {label}")
            continue
        if not any(col in raw.columns for col in ALIASES["bill_ft"]):
            print(f"[FIGYELEM] Box plot: villanyszámla-oszlop hiányzik: {label}")
            continue
        normalized = normalize_case(raw, case, source)
        loaded[label] = normalized.set_index("household")[
            ["bill_ft", "has_pv", "has_bess", "has_boiler"]
        ]

    reference = BOXPLOT_CASES[0]
    missing = set(BOXPLOT_CASES) - set(loaded)
    if missing:
        raise ValueError(f"A box plothoz mindhárom eset szükséges; hiányzik: {sorted(missing)}")
    common = sorted(set.intersection(*(set(frame.index) for frame in loaded.values())))
    if not common:
        raise ValueError("A kiválasztott optimumokhoz nincs közös háztartás.")
    paired = pd.DataFrame({label: loaded[label].loc[common, "bill_ft"] for label in loaded},
                          index=common)
    paired.index.name = "household"
    if not np.isfinite(paired.to_numpy()).all():
        raise ValueError("A box plot adatai között hiányzó vagy nem véges számla van.")

    paired.to_csv(OUTPUT_DIR / "villanyszamla_optimumok_haztartasonkent.csv")
    labels = list(paired.columns)
    records = []
    for label in labels:
        amounts = paired[label]
        savings = paired[reference] - amounts
        records.append({
            "case": label,
            "n_households": len(common),
            "median_ft": float(amounts.median()),
            "mean_ft": float(amounts.mean()),
            "p25_ft": float(amounts.quantile(0.25)),
            "p75_ft": float(amounts.quantile(0.75)),
            "median_paired_saving_ft": float(savings.median()),
            "households_with_lower_bill_pct": float((savings > 0).mean() * 100),
        })
    summary = pd.DataFrame(records)
    summary.to_csv(OUTPUT_DIR / "villanyszamla_optimumok_boxplot_osszesito.csv", index=False)

    palette = [COLORS[CASE_GROUP[label]] for label in labels]
    fig, ax = plt.subplots(figsize=(6.5, 4.0), layout="constrained")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#DCE3EB", lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#94A3B8")
    ax.tick_params(colors="#475569", length=0, pad=5, labelsize=8.5)

    data = [paired[label].to_numpy(dtype=float) / 1000 for label in labels]
    artists = ax.boxplot(
        data, positions=np.arange(1, len(labels) + 1), widths=0.48,
        patch_artist=True, whis=1.5, showfliers=True,
        medianprops={"color": "#182B3D", "lw": 2.3},
        whiskerprops={"color": "#627489", "lw": 1.2},
        capprops={"color": "#627489", "lw": 1.2},
        flierprops={"marker": "o", "markerfacecolor": "#718096",
                    "markeredgecolor": "none", "markersize": 3.5, "alpha": 0.4},
    )
    for box, color in zip(artists["boxes"], palette):
        box.set(facecolor=color, edgecolor=color, alpha=0.72, linewidth=1.5)

    # Ugyanazokat a háztartásokat jelöljük mindhárom dobozon. A jelölések
    # átfedhetnek: egy PV-s háztartásnak BESS-e és bojlere is lehet.
    devices = (
        ("has_pv", "PV", "o", "#2563A6", -0.13),
        ("has_bess", "BESS", "s", "#7651A8", 0.0),
        ("has_boiler", "Bojler", "^", "#B94F45", 0.13),
    )
    jitter = np.random.default_rng(42).uniform(-0.025, 0.025, (len(common), len(devices)))
    device_handles = []
    for device_index, (column, name, marker, color, offset) in enumerate(devices):
        mask = np.logical_or.reduce([
            loaded[label].loc[common, column].to_numpy(dtype=bool) for label in labels
        ])
        if not mask.any():
            continue
        device_handles.append(Line2D([], [], ls="", marker=marker, markersize=5,
                                     markerfacecolor=color, markeredgecolor="white",
                                     label=name))
        for position, label in enumerate(labels, start=1):
            ax.scatter(position + offset + jitter[mask, device_index],
                       paired[label].to_numpy(dtype=float)[mask] / 1000,
                       marker=marker, s=21, color=color, alpha=0.85,
                       edgecolor="white", linewidth=0.45, zorder=5)
    if device_handles:
        ax.margins(y=0.12)
        ax.legend(handles=device_handles, loc="upper right", ncol=len(device_handles),
                  frameon=True, facecolor="white", edgecolor="none", fontsize=8,
                  handletextpad=0.25, columnspacing=0.8)
    names = {
        "0-I": "Alapeset",
        "3d-I": "Háztartási szintű\noptimalizálás",
        "4d-K-C": "Közösségi szintű\noptimalizálás",
        "4d-K-E": "Közösségi szintű\noptimalizálás",
    }
    ax.set_xticks(np.arange(1, len(labels) + 1),
                  [names.get(label, label) for label in labels])
    ax.set_ylabel("Éves nettó számla [ezer Ft]")
    ax.set_title("Villanyszámlák", loc="left", fontsize=12,
                 fontweight="semibold", color="#182B3D", pad=10)
    save(fig, "villanyszamla_optimumok_boxplot")
    return summary


def _series_from_matrix(case: CaseConfig, filename: str, identifiers: list[str]) -> pd.Series:
    source, raw = _raw_summary(case)
    path = source.parent / filename
    if not path.exists():
        raise FileNotFoundError(f"Hiányzik: {path}")
    columns = list(pd.read_csv(path, nrows=0).columns)
    for identifier in identifiers:
        if identifier in columns:
            return pd.to_numeric(pd.read_csv(path, usecols=[identifier])[identifier], errors="raise")
    raise KeyError(f"{path}: a kiválasztott háztartás oszlopa hiányzik. Keresett: {identifiers}")


def _series_from_individual(
    case: CaseConfig, household_name: str, columns: str | tuple[str, ...]
) -> pd.Series:
    """Egy háztartás idősorának beolvasása oszlopnév-aliasokkal.

    A régebbi futtatók pl. ``p_elh_kw`` / ``e_bess_kwh`` neveket, az újabbak
    ``p_elh`` / ``e_bess`` neveket is használhatnak. Az első létező alias nyer.
    """
    source, _ = _raw_summary(case)
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in household_name)
    path = source.parent / f"timeseries_{safe}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Hiányzik az egyéni idősor: {path}")

    aliases = (columns,) if isinstance(columns, str) else tuple(columns)
    available = list(pd.read_csv(path, nrows=0).columns)
    for column in aliases:
        if column in available:
            return pd.to_numeric(
                pd.read_csv(path, usecols=[column])[column], errors="raise"
            )
    raise KeyError(
        f"{path}: egyik keresett oszlop sem található: {aliases}. "
        f"Elérhető oszlopok: {available}"
    )


def _series_from_matrix_aliases(
    case: CaseConfig, filenames: tuple[str, ...], identifiers: list[str]
) -> pd.Series:
    """Mátrixos idősor beolvasása több lehetséges fájlnévvel."""
    errors = []
    for filename in filenames:
        try:
            return _series_from_matrix(case, filename, identifiers)
        except (FileNotFoundError, KeyError) as exc:
            errors.append(str(exc))
    raise FileNotFoundError(
        f"{case.label}: egyik idősorfájl sem használható: {filenames}. "
        + " | ".join(errors)
    )


def _daily_slice(series: pd.Series, day: str) -> tuple[pd.DatetimeIndex, np.ndarray]:
    first = pd.Timestamp(FIRST_DAY).normalize()
    date = pd.Timestamp(day).normalize()
    steps = round(24 / DT_HOURS)
    if not np.isclose(steps * DT_HOURS, 24):
        raise ValueError("A DT_HOURS értékének osztania kell a 24 órát.")
    offset = (date - first).days * steps
    if offset < 0 or offset + steps > len(series):
        raise ValueError(f"{day} nincs benne a {len(series)} elemű profilban, kezdőnap: {first.date()}.")
    values = series.iloc[offset:offset + steps].to_numpy(dtype=float)
    clock = pd.date_range(date, periods=steps, freq=pd.Timedelta(hours=DT_HOURS))
    return clock, values


def boiler_day_plot() -> None:
    baseline = _configuration(BOILER_BASELINE_CASE)
    optimized = _configuration(BOILER_OPT_CASE)
    _, opt_raw = _raw_summary(optimized)
    opt_row = _matching_row(opt_raw, HOUSEHOLD, optimized.label)
    opt_name = str(opt_row.get("household", opt_row.get("user_name", HOUSEHOLD)))
    if "hss_optimized" in opt_row and str(opt_row["hss_optimized"]).strip().lower() in ("0", "false", "no"):
        raise ValueError(f"{HOUSEHOLD}: nincs modellezett bojler a(z) {optimized.label} esetben.")
    identifiers = _names_for_household(HOUSEHOLD, optimized, baseline)
    identifiers.insert(0, opt_name)
    measured_kwh = _series_from_matrix_aliases(
        baseline, ("e_boiler.csv", "p_boiler_fixed.csv", "p_elh.csv"), identifiers
    )
    modeled_kw = _series_from_individual(
        optimized, opt_name, ("p_elh_kw", "p_elh", "p_boiler_kw")
    )
    if len(measured_kwh) != len(modeled_kw):
        raise ValueError("A két bojlerprofil eltérő hosszúságú, a napok nem párosíthatók.")
    clock, measured_kw = _daily_slice(measured_kwh, BOILER_DAY)
    _, optimized_kw = _daily_slice(modeled_kw, BOILER_DAY)
    # Az e_boiler.csv energiaprofil [kWh/lépés], a p_* fájlok teljesítményprofilok [kW].
    # A kiválasztott forrás nevétől függetlenül itt az eredeti baseline formátumot
    # csak akkor konvertáljuk, ha e_boiler.csv létezik és ez volt használható.
    baseline_source, _ = _raw_summary(baseline)
    e_boiler_path = baseline_source.parent / "e_boiler.csv"
    if e_boiler_path.exists():
        try:
            e_cols = list(pd.read_csv(e_boiler_path, nrows=0).columns)
            if any(identifier in e_cols for identifier in identifiers):
                measured_kw = measured_kw / DT_HOURS
        except OSError:
            pass
    daily = pd.DataFrame({"time": clock, "B_tarifa_mert_kw": measured_kw,
                          "egyeni_opt_homodell_kw": optimized_kw})
    daily.to_csv(OUTPUT_DIR / "boiler_napi_osszehasonlitas.csv", index=False)
    fig, ax = plt.subplots(figsize=(11, 4.8), constrained_layout=True)
    ax.step(clock, measured_kw, where="post", label=f"{baseline.label}: mért B tarifa", lw=1.8)
    ax.step(clock, optimized_kw, where="post", label=f"{optimized.label}: hőmodell, villamos felvétel", lw=1.8)
    ax.set(xlabel="Idő", ylabel="Bojler villamos teljesítménye [kW]",
           title=f"{HOUSEHOLD} – bojler, {BOILER_DAY} (15 perces profil)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax.set_xlim(clock[0], clock[0] + pd.Timedelta(days=1))
    ax.grid(alpha=0.25)
    ax.legend()
    save(fig, "boiler_mert_B_vs_egyeni_opt")
    print(f"[OK] Bojler: {HOUSEHOLD}, {BOILER_DAY}; "
          f"B={measured_kw.sum()*DT_HOURS:.2f} kWh, "
          f"opt={optimized_kw.sum()*DT_HOURS:.2f} kWh")


def _battery_series(case: CaseConfig, name: str) -> pd.Series:
    if case.family == "I" and case.label not in ("1d-I", "0-I"):
        return _series_from_individual(case, name, ("e_bess_kwh", "e_bess"))
    return _series_from_matrix_aliases(case, ("e_bess.csv", "soc_bess.csv"), [name])


def battery_day_plot() -> None:
    case = _configuration(BATTERY_CASE)
    _, raw = _raw_summary(case)
    if BATTERY_HOUSEHOLD:
        candidates = [_matching_row(raw, BATTERY_HOUSEHOLD, case.label)]
    else:
        if "has_bess" not in raw:
            raise KeyError(f"{case.label}: a household_summary nem tartalmaz has_bess oszlopot.")
        flags = raw["has_bess"].astype(str).str.lower().isin(["1", "true", "yes"])
        candidates = [row for _, row in raw.loc[flags].iterrows()]
    for row in candidates:
        name = str(row.get("household", row.get("user_name", row.get("name", ""))))
        try:
            series = _battery_series(case, name)
        except (FileNotFoundError, KeyError):
            continue
        steps = round(24 / DT_HOURS)
        count = len(series) // steps
        if not count:
            continue
        days = series.to_numpy(dtype=float)[:count * steps].reshape(count, steps)
        swings = np.ptp(days, axis=1)
        active = np.flatnonzero(swings > EPS)
        if len(active):
            break
    else:
        raise ValueError(f"{case.label}: nem található aktív BESS idősor. Állítsd be a BATTERY_HOUSEHOLD értékét.")
    if BATTERY_DAY:
        chosen = pd.Timestamp(BATTERY_DAY).normalize()
    else:
        target = np.median(swings[active])
        index = active[np.argmin(np.abs(swings[active] - target))]
        chosen = pd.Timestamp(FIRST_DAY).normalize() + pd.Timedelta(days=int(index))
    clock, energy = _daily_slice(series, str(chosen.date()))
    pd.DataFrame({"time": clock, "bess_energy_kwh": energy}).to_csv(
        OUTPUT_DIR / "bess_tipikus_nap.csv", index=False)
    fig, ax = plt.subplots(figsize=(11, 4.8), constrained_layout=True)
    ax.step(clock, energy, where="post", color="#9467bd", lw=2)
    ax.fill_between(clock, energy, step="post", color="#9467bd", alpha=0.13)
    ax.set(xlabel="Idő", ylabel="Akkumulátor energiaszintje [kWh]",
           title=f"{case.label} – {name}, {chosen.date()} (napi kilengés: {np.ptp(energy):.2f} kWh)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax.set_xlim(clock[0], clock[0] + pd.Timedelta(days=1))
    ax.grid(alpha=0.25)
    save(fig, "bess_tipikus_energiaszint")
    print(f"[OK] BESS: {name}, {chosen.date()}, eset: {case.label}")


def save(fig, name: str) -> None:
    fig.savefig(OUTPUT_DIR / f"{name}.png", dpi=1200)
    if SAVE_PDF:
        fig.savefig(OUTPUT_DIR / f"{name}.pdf")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for title, draw in (("SCI–SSI", sci_ssi_points), ("villanyszámla", bill_comparison),
                        ("bojler", boiler_day_plot),
                        ("BESS", battery_day_plot)):
        try:
            draw()
        except (ValueError, KeyError, FileNotFoundError) as exc:
            print(f"[FIGYELEM] {title} ábra nem készült el: {exc}")
    print(f"Kimenet: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()


