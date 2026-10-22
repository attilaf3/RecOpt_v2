"""3d-I: egy-egy kiválasztott nap külön villamos, BESS- és bojlerábrákon.

Az OptimizedIndividualScenarios mappába másolható, PyCharm Shift+F10-zel fut.
Csak a fenti BEÁLLÍTÁSOK részt kell változtatni. A profilok 2021-01-01-től
kezdődő, 15 perces teljesítmény-idősorok (a BESS-energia kivételével).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd


# ============================ BEÁLLÍTÁSOK ============================
SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR.parent / "OptimizedScenarios" / "results_3d_I"
FIGURES_DIR = SCRIPT_DIR / "figures_3d_I_professional"
HOUSEHOLD = "0420144653449093"  # háztartáskód vagy household_summary.csv-beli név
# HOUSEHOLD = "0420144888295341"  # háztartáskód vagy household_summary.csv-beli név
# HOUSEHOLD = "0420144888460007"  # háztartáskód vagy household_summary.csv-beli név
FIRST_DAY = "2021-01-01"        # az idősor legelső napja
SEASON_DAYS = (
    ("tel", "2021-01-01"),
    ("tavasz", "2021-04-01"),
    ("nyar", "2021-07-01"),
    ("osz", "2021-10-01"),
)
ONLY_DAY: str | None = None      # pl. "2021-07-15": csak ezt a napot rajzolja
DT = 0.25                       # óra / időlépés
DPI = 1200
SAVE_PDF = False
# =====================================================================

PALETTE = {
    "pv": "#237A8A", "bess": "#3775A9", "grid_a": "#546477",
    "grid_b": "#9981B6", "base": "#657384", "boiler": "#E28A38",
    "boiler_b": "#DDB474", "charge": "#88B5D7", "export": "#B4A45A",
    "temp": "#CC653F", "target": "#9569AA", "sharing": "#48A49C",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.titlesize": 12, "axes.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.axisbelow": True, "grid.color": "#D6DDE3",
    "grid.alpha": 0.65, "grid.linewidth": 0.6,
    "legend.frameon": False, "savefig.facecolor": "white",
})


def _safe_name(value: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in value)


def _household_name(results_dir: Path, choice: str) -> str:
    path = results_dir / "household_summary.csv"
    summary = pd.read_csv(path, dtype={"household_key": str, "household": str})
    if "household" not in summary:
        raise ValueError(f"Hiányzik a 'household' oszlop: {path}")
    mask = summary["household"].astype(str).eq(str(choice).strip())
    if "household_key" in summary:
        mask |= summary["household_key"].astype(str).eq(str(choice).strip())
    matched = summary.loc[mask]
    if len(matched) != 1:
        raise ValueError(f"A háztartás nem található vagy nem egyértelmű: {choice}")
    return str(matched.iloc[0]["household"])


def _column(frame: pd.DataFrame, name: str, *, optional: bool = False) -> np.ndarray:
    if name not in frame:
        if optional:
            return np.zeros(len(frame))
        raise KeyError(f"Hiányzó idősoroszlop: {name}")
    series = pd.to_numeric(frame[name], errors="coerce")
    values = (series.fillna(0) if optional else series).to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"Nem véges adat az idősorban: {name}")
    return values


def _day_chunk(frame: pd.DataFrame, date: str) -> pd.DataFrame:
    steps = round(24 / DT)
    if DT <= 0 or not np.isclose(steps * DT, 24):
        raise ValueError("A DT értékének pozitívnak kell lennie, és osztania kell a 24-et.")
    start = (pd.Timestamp(date).normalize() - pd.Timestamp(FIRST_DAY).normalize()).days * steps
    if start < 0 or start + steps > len(frame):
        raise ValueError(f"{date} kívül esik az idősoron ({len(frame)} sor).")
    return frame.iloc[start:start + steps].reset_index(drop=True)


def _time_axis(ax) -> None:
    ax.set_xlim(0, 24)
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.set_xlabel("Napszak [óra]")
    ax.grid(axis="y")


def _save(fig, filename: str, figures_dir: Path, saved: list[Path]) -> None:
    png = figures_dir / f"{filename}.png"
    fig.savefig(png, dpi=DPI, bbox_inches="tight")
    saved.append(png)
    if SAVE_PDF:
        pdf = figures_dir / f"{filename}.pdf"
        fig.savefig(pdf, bbox_inches="tight")
        saved.append(pdf)
    plt.close(fig)
    print(f"[OK] {png}")


def _balance_figure(chunk: pd.DataFrame, name: str, date: str):
    x = (np.arange(len(chunk)) + 0.5) * DT
    sources = [
        ("PV", _column(chunk, "p_pv_kw"), PALETTE["pv"]),
        ("BESS kisütés", _column(chunk, "p_bess_discharge_kw"), PALETTE["bess"]),
        ("Hálózat A", _column(chunk, "p_grid_import_a_kw"), PALETTE["grid_a"]),
        ("Hálózat B", _column(chunk, "p_grid_import_b_kw"), PALETTE["grid_b"]),
    ]
    sinks = [
        ("Alapfogyasztás", _column(chunk, "p_base_load_kw"), PALETTE["base"]),
        ("Bojler A", _column(chunk, "p_elh_kw"), PALETTE["boiler"]),
        ("Mért bojler B", _column(chunk, "p_fixed_boiler_kw"), PALETTE["boiler_b"]),
        ("BESS töltés", _column(chunk, "p_bess_charge_kw"), PALETTE["charge"]),
        ("PV export", _column(chunk, "p_pv_export_kw"), PALETTE["export"]),
    ]
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    for sign, flows in ((1, sources), (-1, sinks)):
        bottom = np.zeros(len(chunk))
        for label, values, color in flows:
            if np.max(np.abs(values)) <= 1e-9:
                continue
            ax.bar(x, sign * values, bottom=bottom, width=DT * 0.92,
                   color=color, label=label, linewidth=0)
            bottom += sign * values
    ax.axhline(0, color="#25313C", linewidth=0.9)
    ax.set_title(f"{name} | {date} | Villamos energiamérleg")
    ax.set_ylabel("Teljesítmény [kW]")
    _time_axis(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16),
              ncol=4, fontsize=9, columnspacing=1.4)
    return fig


def _bess_figure(chunk: pd.DataFrame, name: str, date: str):
    energy = _column(chunk, "e_bess_kwh")
    charge = _column(chunk, "p_bess_charge_kw")
    discharge = _column(chunk, "p_bess_discharge_kw")
    if max(np.max(np.abs(energy)), np.max(charge), np.max(discharge)) <= 1e-9:
        return None
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    for active, color in ((charge > 1e-6, PALETTE["charge"]),
                          (discharge > 1e-6, PALETTE["bess"])):
        for index in np.flatnonzero(active):
            ax.axvspan(index * DT, (index + 1) * DT,
                       color=color, alpha=0.14, linewidth=0, zorder=0)
    x = np.arange(len(chunk)) * DT
    ax.step(x, energy, where="post", color="#263B69", linewidth=2)
    ax.set_title(f"{name} | {date} | Akkumulátor energiaszintje")
    ax.set_ylabel("Energia [kWh]")
    ax.set_ylim(bottom=0)
    _time_axis(ax)
    ax.legend(handles=[Patch(facecolor=PALETTE["charge"], alpha=0.4, label="Töltés"),
                       Patch(facecolor=PALETTE["bess"], alpha=0.4, label="Kisütés")],
              loc="upper right", ncol=2)
    return fig


def _boiler_figure(chunk: pd.DataFrame, name: str, date: str):
    power_a = _column(chunk, "p_elh_kw")
    power_b = _column(chunk, "p_fixed_boiler_kw")
    temperature = _column(chunk, "t_hss_C", optional=True)
    target = _column(chunk, "boiler_target_C", optional=True)
    modeled = np.max(temperature) > 0
    if max(np.max(power_a), np.max(power_b)) <= 1e-9 and not modeled:
        return None
    rows = 2 if modeled else 1
    fig, axes = plt.subplots(rows, 1, figsize=(7, 4 if modeled else 3.5),
                             sharex=True, layout="constrained")
    if rows == 1:
        axes = [axes]
    x = np.arange(len(chunk)) * DT
    ax = axes[0]
    if np.max(power_a) > 1e-9:
        ax.stairs(power_a, np.arange(len(chunk) + 1) * DT,
                  label="Modellezett bojler – A tarifa", color=PALETTE["boiler"], lw=1.8)
    if np.max(power_b) > 1e-9:
        ax.stairs(power_b, np.arange(len(chunk) + 1) * DT,
                  label="Mért bojler – B tarifa", color=PALETTE["boiler_b"], lw=1.8)
    ax.set_ylabel("Villamos teljesítmény [kW]")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y")
    ax.legend(loc="upper right")
    if modeled:
        ax = axes[1]
        ax.plot(x, temperature, color=PALETTE["temp"], lw=2, label="Tartályhőmérséklet")
        ax.axhline(50, color="#BA514B", ls="--", lw=1,
                   label="50 °C – sávhatári cél")
        if np.max(target) > 0:
            ax.step(x, target, where="post", color=PALETTE["target"],
                    ls=":", lw=1.3, label="Célérték")
        ax.set_ylabel("Hőmérséklet [°C]")
        ax.grid(axis="y")
        ax.legend(loc="upper right", ncol=3, fontsize=9)
    fig.suptitle(f"{name} | {date} | Bojler")
    _time_axis(axes[-1])
    return fig


def plot_individual(out_dir=RESULTS_DIR, household=HOUSEHOLD, *,
                    only_day=ONLY_DAY, figures_dir=FIGURES_DIR) -> list[Path]:
    out = Path(out_dir)
    name = _household_name(out, household)
    path = out / f"timeseries_{_safe_name(name)}.csv"
    data = pd.read_csv(path)
    if data.empty:
        raise ValueError(f"Üres idősor: {path}")
    figures = Path(figures_dir)
    figures.mkdir(parents=True, exist_ok=True)
    days = [("valasztott_nap", only_day)] if only_day else SEASON_DAYS
    saved: list[Path] = []
    for season, date in days:
        chunk = _day_chunk(data, date)
        stem = f"individual_{_safe_name(name)}_{season}_{date}"
        _save(_balance_figure(chunk, name, date), stem + "_merleg", figures, saved)
        for suffix, make_fig in (("bess", _bess_figure), ("bojler", _boiler_figure)):
            fig = make_fig(chunk, name, date)
            if fig is not None:
                _save(fig, stem + "_" + suffix, figures, saved)
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR)
    parser.add_argument("--household", default=HOUSEHOLD)
    parser.add_argument("--day", default=ONLY_DAY,
                        help="YYYY-MM-DD; ha nincs megadva, a SEASON_DAYS napjait rajzolja")
    args = parser.parse_args()
    plot_individual(args.out, args.household, only_day=args.day)


if __name__ == "__main__":
    main()