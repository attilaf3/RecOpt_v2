"""4d-K: egynapos közösségi és választott háztartási ábrák.

Az OptimizedCommunityScenarios mappába másolható, PyCharm Shift+F10-zel fut.
A közösségi mérleget és a megosztást mindig kirajzolja, akkor is, ha
HOUSEHOLD = None. A háztartási ábrákhoz a CSV-k egyéni idősora nem kötelező.
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
RESULTS_NAME = "results_4d-K-C_proportional"  # vagy results_4d-K-E_proportional
RESULTS_DIR = next(
    (p for p in (SCRIPT_DIR.parent / RESULTS_NAME, SCRIPT_DIR / RESULTS_NAME)
     if (p / "community_timeseries.csv").is_file()),
    SCRIPT_DIR.parent / RESULTS_NAME,
)
FIGURES_DIR = SCRIPT_DIR / "figures_4d_K_professional"
HOUSEHOLD: str | None = "0420144653449093"  # None: csak közösségi ábrák
HOUSEHOLD_TITLE = "Vizsgált háztartás"  # a képen nem jelenik meg a háztartáskód
FIRST_DAY = "2021-01-01"
SEASON_DAYS = (
    ("tel", "2021-01-01"),
    ("tavasz", "2021-04-01"),
    ("nyar", "2021-07-01"),
    ("osz", "2021-10-01"),
)
ONLY_DAY: str | None = None  # pl. "2021-07-15"
DT = 0.25
DPI = 1200
SAVE_PDF = False
# =====================================================================

PALETTE = {
    "pv": "#237A8A", "bess": "#3775A9", "grid_a": "#32955D",
    "grid_b": "#9981B6", "base": "#657384", "boiler": "#E28A38",
    "boiler_b": "#DDB474", "charge": "#88B5D7", "export": "#B4A45A",
    "temp": "#27587B", "target": "#9569AA", "sharing": "#48A49C",
    "bess_charge_window": "#D99A4B", "bess_discharge_window": "#5F91BE",
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


def _household_name(out: Path, choice: str) -> str:
    path = out / "household_summary.csv"
    summary = pd.read_csv(path, dtype={"household_key": str, "household": str})
    if "household" not in summary:
        raise ValueError(f"Hiányzó household oszlop: {path}")
    query = str(choice).strip()
    mask = summary["household"].astype(str).eq(query)
    if "household_key" in summary:
        mask |= summary["household_key"].astype(str).eq(query)
    if not mask.any() and query.removeprefix("load_") != query:
        mask = summary["household"].astype(str).eq(query.removeprefix("load_"))
    matched = summary.loc[mask]
    if len(matched) != 1:
        raise ValueError(f"A háztartás nem található vagy nem egyértelmű: {choice}")
    return str(matched.iloc[0]["household"])


def _read_household(out: Path, name: str) -> pd.DataFrame:
    combined = out / f"timeseries_{_safe_name(name)}.csv"
    if combined.is_file():
        return pd.read_csv(combined)
    required = (
        "p_pv", "p_base", "p_elh", "p_boiler_fixed",
        "p_grid_import_a", "p_grid_import_b", "p_pv_export",
        "p_pv_bess", "p_grid_bess", "p_bess_base", "p_bess_boiler",
        "p_shared_in", "p_shared_out", "e_bess",
    )
    optional = ("t_hss_equivalent", "boiler_target_c")
    columns = {}
    for variable in required + optional:
        path = out / f"{variable}.csv"
        if not path.is_file():
            if variable in optional:
                continue
            raise FileNotFoundError(path)
        try:
            columns[variable] = pd.read_csv(path, usecols=[name])[name]
        except ValueError as exc:
            raise ValueError(f"A(z) {name} háztartás nem szerepel itt: {path}") from exc
    return pd.DataFrame(columns)


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


def _save(fig, filename: str, figures: Path, saved: list[Path]) -> None:
    png = figures / f"{filename}.png"
    fig.savefig(png, dpi=DPI, bbox_inches="tight")
    saved.append(png)
    if SAVE_PDF:
        pdf = figures / f"{filename}.pdf"
        fig.savefig(pdf, bbox_inches="tight")
        saved.append(pdf)
    plt.close(fig)
    print(f"[OK] {png}")


def _balance_figure(chunk: pd.DataFrame, title: str, household: bool):
    def c(name: str) -> np.ndarray:
        return _column(chunk, name)
    charge = c("p_pv_bess") + c("p_grid_bess")
    discharge = c("p_bess_base") + c("p_bess_boiler")
    sources = [
        ("PV", c("p_pv"), PALETTE["pv"]),
        ("BESS kisütés", discharge, PALETTE["bess"]),
        ("Hálózat A", c("p_grid_import_a"), PALETTE["grid_a"]),
        ("Hálózat B", c("p_grid_import_b"), PALETTE["grid_b"]),
    ]
    sinks = [
        ("Alapfogyasztás", c("p_base"), PALETTE["base"]),
        ("Bojler A", c("p_elh"), PALETTE["boiler"]),
        ("Mért bojler B", c("p_boiler_fixed"), PALETTE["boiler_b"]),
        ("BESS töltés", charge, PALETTE["charge"]),
        ("PV export", c("p_pv_export"), PALETTE["export"]),
    ]
    if household:
        sources.append(("Megosztásból kapott", c("p_shared_in"), PALETTE["sharing"]))
        sinks.append(("Megosztásba adott", c("p_shared_out"), "#308178"))
    # Közösségi összegzésnél a belső megosztás két oldala kiegyenlíti egymást.
    x = (np.arange(len(chunk)) + 0.5) * DT
    fig, ax = plt.subplots(figsize=(6.5, 5.7), layout="constrained")
    for sign, flows in ((1, sources), (-1, sinks)):
        bottom = np.zeros(len(chunk))
        for label, values, color in flows:
            if np.max(np.abs(values)) <= 1e-9:
                continue
            ax.bar(x, sign * values, bottom=bottom, width=DT * 0.92,
                   color=color, label=label, linewidth=0)
            bottom += sign * values
    ax.axhline(0, color="#25313C", linewidth=0.9)
    ax.set_title(title)
    ax.set_ylabel("Teljesítmény [kW]")
    _time_axis(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16),
              ncol=4, fontsize=9, columnspacing=1.4)
    return fig


def _sharing_figure(chunk: pd.DataFrame, date: str):
    grid = (_column(chunk, "p_grid_import_a")
            + _column(chunk, "p_grid_import_b"))
    shared_in = _column(chunk, "p_shared_in")
    shared_out = _column(chunk, "p_shared_out")
    if not np.allclose(shared_in, shared_out, rtol=1e-5, atol=1e-5):
        raise ValueError(f"{date}: a közösségi p_shared_in és p_shared_out "
                         "összegének időlépésenként egyeznie kell.")
    export = _column(chunk, "p_pv_export")
    edges = np.arange(len(chunk) + 1) * DT
    fig, ax = plt.subplots(figsize=(6.2, 4.2), layout="constrained")
    ax.set_facecolor("white")

    # A mintakép színvilága és finom határvonalai, valódi egymásra halmozással.
    # Az export az utolsó sáv: a felső perem nem a fogyasztási igényt jelenti.
    shared_top = shared_out
    grid_top = shared_top + grid
    export_top = grid_top + export

    for bottom, top, color, opacity in (
        (np.zeros(len(chunk)), shared_top, "#389D68", 0.30),
        (shared_top, grid_top, "#3775A9", 0.20),
        (grid_top, export_top, "#C85357", 0.30),
    ):
        ax.fill_between(edges, np.r_[bottom, bottom[-1]],
                        np.r_[top, top[-1]], step="post",
                        facecolor=color, alpha=opacity, linewidth=0)

    # A két alsó réteg pereme végig látszik.
    ax.step(edges, np.r_[shared_top, shared_top[-1]], where="post",
            color="#389D68", linewidth=1.35)
    ax.step(edges, np.r_[grid_top, grid_top[-1]], where="post",
            color="#3775A9", linewidth=1.35)

    # A piros felső perem csak tényleges exportnál látszik; nincs piros
    # nullavonal. A vízszintes és a kapcsolódó függőleges lépcsőket rajzoljuk.
    active = export > 1e-9
    ax.hlines(export_top[active], edges[:-1][active], edges[1:][active],
              color="#C85357", linewidth=2.3, zorder=4)
    both_active = active[:-1] & active[1:]
    ax.vlines(edges[1:-1][both_active],
              export_top[:-1][both_active], export_top[1:][both_active],
              color="#C85357", linewidth=2.3, zorder=4)
    starts = active & ~np.r_[False, active[:-1]]
    ends = active & ~np.r_[active[1:], False]
    ax.vlines(edges[:-1][starts], grid_top[starts], export_top[starts],
              color="#C85357", linewidth=2.3, zorder=4)
    ax.vlines(edges[1:][ends], grid_top[ends], export_top[ends],
              color="#C85357", linewidth=2.3, zorder=4)

    ax.set_title(f"Közösség | {date} | Megosztás és külső hálózati forgalom")
    ax.set_ylabel("Teljesítmény [kW]")
    ax.set_ylim(bottom=0)
    _time_axis(ax)
    ax.legend(handles=[
        Patch(facecolor="#389D68", alpha=0.55, label="Közösségen belüli megosztás"),
        Patch(facecolor="#3775A9", alpha=0.45, label="Hálózati vételezés (A + B)"),
        Patch(facecolor="#C85357", alpha=0.55, label="Hálózati export"),
    ], loc="upper center", bbox_to_anchor=(0.5, -0.15),
       ncol=3, fontsize=8.3, columnspacing=1.2)
    return fig


def _bess_figure(chunk: pd.DataFrame, name: str, date: str):
    energy = _column(chunk, "e_bess")
    charge = _column(chunk, "p_pv_bess") + _column(chunk, "p_grid_bess")
    discharge = _column(chunk, "p_bess_base") + _column(chunk, "p_bess_boiler")
    if max(np.max(np.abs(energy)), np.max(charge), np.max(discharge)) <= 1e-9:
        return None
    fig, ax = plt.subplots(figsize=(6.5, 3.7), layout="constrained")
    for active, color in ((charge > 1e-6, PALETTE["bess_charge_window"]),
                          (discharge > 1e-6, PALETTE["bess_discharge_window"])):
        for index in np.flatnonzero(active):
            ax.axvspan(index * DT, (index + 1) * DT,
                       color=color, alpha=0.19, linewidth=0, zorder=0)
    ax.step(np.arange(len(chunk)) * DT, energy, where="post", color="#263B69", lw=2)
    ax.set_title(f"{HOUSEHOLD_TITLE} | {date} | Akkumulátor energiaszintje")
    ax.set_ylabel("Energia [kWh]")
    ax.set_ylim(bottom=0)
    _time_axis(ax)
    ax.legend(handles=[Patch(facecolor=PALETTE["bess_charge_window"], alpha=0.36, label="Töltés"),
                       Patch(facecolor=PALETTE["bess_discharge_window"], alpha=0.36, label="Kisütés")],
              loc="upper right", ncol=2)
    return fig


def _boiler_figure(chunk: pd.DataFrame, name: str, date: str):
    power_a = _column(chunk, "p_elh")
    power_b = _column(chunk, "p_boiler_fixed")
    temperature = _column(chunk, "t_hss_equivalent", optional=True)
    target = _column(chunk, "boiler_target_c", optional=True)
    modeled = np.max(temperature) > 0
    if max(np.max(power_a), np.max(power_b)) <= 1e-9 and not modeled:
        return None
    fig, ax_temp = plt.subplots(figsize=(6.2, 4.2), layout="constrained")
    ax_power = ax_temp.twinx()
    edges = np.arange(len(chunk) + 1) * DT
    power_lines = []
    if np.max(power_a) > 1e-9:
        line = ax_power.stairs(power_a, edges, color=PALETTE["boiler"], lw=1.6,
                               label="Modellezett bojler – A tarifa")
        power_lines.append(line)
        ax_power.fill_between(edges, 0, np.r_[power_a, power_a[-1]], step="post",
                              color=PALETTE["boiler"], alpha=0.10, linewidth=0)
    if np.max(power_b) > 1e-9:
        line = ax_power.stairs(power_b, edges, color="#A48244", lw=1.6,
                               label="Mért bojler – B tarifa")
        power_lines.append(line)
        ax_power.fill_between(edges, 0, np.r_[power_b, power_b[-1]], step="post",
                              color="#A48244", alpha=0.10, linewidth=0)
    temperature_lines = []
    if modeled:
        temp_line, = ax_temp.plot(np.arange(len(chunk)) * DT, temperature,
                                  color=PALETTE["temp"], lw=2.2,
                                  label="Tartályhőmérséklet", zorder=5)
        temperature_lines.append(temp_line)
        goal = ax_temp.axhline(50, color="#95A7B9", ls=(0, (4, 3)), lw=1.1,
                                label="50 °C – sávhatári cél")
        temperature_lines.append(goal)
        if np.max(target) > 0 and not np.allclose(target, 50):
            target_line, = ax_temp.step(np.arange(len(chunk)) * DT, target,
                                        where="post", ls=":", lw=1.4,
                                        color=PALETTE["target"], label="Célérték")
            temperature_lines.append(target_line)
        ax_temp.set_ylim(bottom=max(0, min(float(temperature.min()), 50) - 5),
                         top=max(float(temperature.max()), 50) + 5)
    else:
        ax_temp.set_yticks([])
        ax_temp.text(0.02, 0.95, "A mért bojler hőmérséklete nem áll rendelkezésre",
                     transform=ax_temp.transAxes, va="top", fontsize=8.5,
                     color="#657384")
    ax_temp.set_ylabel("Tartályhőmérséklet [°C]", color=PALETTE["temp"])
    ax_power.set_ylabel("Villamos teljesítmény [kW]", color=PALETTE["boiler"])
    ax_power.set_ylim(bottom=0)
    ax_temp.tick_params(axis="y", colors=PALETTE["temp"])
    ax_power.tick_params(axis="y", colors=PALETTE["boiler"])
    ax_power.spines["right"].set_visible(True)
    ax_temp.set_title(f"{HOUSEHOLD_TITLE} | {date} | Bojler")
    _time_axis(ax_temp)
    ax_temp.legend(handles=temperature_lines + power_lines,
                   loc="upper center", bbox_to_anchor=(0.5, -0.15),
                   ncol=2, fontsize=8.5)
    return fig


def plot_community(out_dir=RESULTS_DIR, household=HOUSEHOLD, *,
                   only_day=ONLY_DAY, figures_dir=FIGURES_DIR) -> list[Path]:
    out = Path(out_dir)
    aggregate = pd.read_csv(out / "community_timeseries.csv")
    if aggregate.empty:
        raise ValueError("Üres közösségi idősor.")
    name = _household_name(out, household) if household is not None else None
    individual = _read_household(out, name) if name is not None else None
    if individual is not None and len(individual) != len(aggregate):
        raise ValueError("Eltérő hosszúságú közösségi és háztartási idősor.")
    figures = Path(figures_dir)
    figures.mkdir(parents=True, exist_ok=True)
    days = [("valasztott_nap", only_day)] if only_day else SEASON_DAYS
    saved: list[Path] = []
    for season, date in days:
        chunk = _day_chunk(aggregate, date)
        stem = f"community_{season}_{date}"
        _save(_balance_figure(chunk, f"Közösség | {date} | Villamos energiamérleg", False),
              stem + "_merleg", figures, saved)
        _save(_sharing_figure(chunk, date), stem + "_megosztas", figures, saved)
        if individual is None:
            continue
        part = _day_chunk(individual, date)
        prefix = f"household_{_safe_name(name)}_{season}_{date}"
        _save(_balance_figure(part, f"{HOUSEHOLD_TITLE} | {date} | Villamos energiamérleg", True),
              prefix + "_merleg", figures, saved)
        for suffix, make_fig in (("bess", _bess_figure), ("bojler", _boiler_figure)):
            fig = make_fig(part, name, date)
            if fig is not None:
                _save(fig, prefix + "_" + suffix, figures, saved)
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR)
    parser.add_argument("--household", default=HOUSEHOLD,
                        help="Háztartáskód/-név, vagy 'none' csak a közösségi ábrákhoz")
    parser.add_argument("--day", default=ONLY_DAY,
                        help="YYYY-MM-DD; ha nincs megadva, a SEASON_DAYS napjait rajzolja")
    args = parser.parse_args()
    chosen = None if args.household is None or str(args.household).lower() == "none" else args.household
    plot_community(args.out, chosen, only_day=args.day)


if __name__ == "__main__":
    main()