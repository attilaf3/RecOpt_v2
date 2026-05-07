from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# BEÁLLÍTÁSOK
# =========================
RESULTS_DIR = r"./results_nonopt"      # run_case output mappája
HOUSEHOLD_NAME = "Household_1"         # pontos oszlopnév az e_*.csv fájlokban

WINDOW_DAYS = 3                        # hány nap / évszak
DT = 0.25                              # 15 perc = 0.25 h

DPI = 300
FONTSIZE = 18
FIGSIZE = (22, 8)
# =========================


plt.rcParams.update({"font.size": FONTSIZE})


def _load_matrix(results_dir: Path, filename: str) -> pd.DataFrame:
    path = results_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Hiányzó fájl: {path}")
    return pd.read_csv(path)


def _get_series(df: pd.DataFrame, household_name: str, filename: str) -> np.ndarray:
    if household_name not in df.columns:
        raise KeyError(
            f"A(z) '{household_name}' oszlop nem található ebben a fájlban: {filename}\n"
            f"Elérhető oszlopok: {list(df.columns)}"
        )
    return df[household_name].to_numpy(dtype=float)


def _load_household_series(
    results_dir: Path,
    household_name: str,
) -> dict[str, np.ndarray]:
    """
    Betölti az adott háztartás szükséges e_* idősorait.

    Elvárt fájlok:
      - e_load.csv
      - e_pv.csv
      - e_bess_to_load.csv
      - e_grid_to_load.csv
      - e_inj.csv
      - e_pv_to_bess.csv
    """
    files = {
        "e_load": "e_load.csv",
        "e_pv": "e_pv.csv",
        "e_bess_to_load": "e_bess_to_load.csv",
        "e_grid_to_load": "e_grid_to_load.csv",
        "e_inj": "e_inj.csv",
        "e_pv_to_bess": "e_pv_to_bess.csv",
    }

    out: dict[str, np.ndarray] = {}

    for key, filename in files.items():
        df = _load_matrix(results_dir, filename)
        out[key] = _get_series(df, household_name, filename)

    return out


def _season_windows(dt: float) -> tuple[list[str], list[str], list[int]]:
    """
    Évszak kezdőpontok.

    A korábbi megosztott plot scripted logikáját követi:
      - winter: év eleje
      - spring: 2184. óra
      - summer: 4368. óra
      - autumn: 6552. óra

    15 perces lépésnél:
      step = hour / DT
    """
    t0s_hourly = [0, 2184, 4368, 6552]

    season_keys = ["winter", "spring", "summer", "autumn"]
    season_titles = ["Tél", "Tavasz", "Nyár", "Ősz"]

    t0s = [int(round(hour / dt)) for hour in t0s_hourly]

    return season_keys, season_titles, t0s


def _energy_to_power(e: np.ndarray, dt: float) -> np.ndarray:
    """
    Energia [kWh / időlépés] → teljesítmény [kW].
    """
    return np.asarray(e, dtype=float) / float(dt)


def _plot_one_season(
    results_dir: Path,
    household_name: str,
    season_key: str,
    season_title: str,
    t0: int,
    window_steps: int,
    series: dict[str, np.ndarray],
    dt: float,
) -> Path | None:
    T = len(series["e_load"])
    tf = min(t0 + window_steps, T)

    if tf <= t0 + 1:
        print(f"[WARN] {season_title}: nincs elég adat ehhez az ablakhoz.")
        return None

    # --- szeletelés ---
    sl = slice(t0, tf)

    e_load = series["e_load"][sl]
    e_pv = series["e_pv"][sl]
    e_bess_to_load = series["e_bess_to_load"][sl]
    e_grid_to_load = series["e_grid_to_load"][sl]
    e_inj = series["e_inj"][sl]
    e_pv_to_bess = series["e_pv_to_bess"][sl]

    # --- teljesítmény [kW] ---
    p_load = _energy_to_power(e_load, dt)
    p_pv = _energy_to_power(e_pv, dt)
    p_bess_to_load = _energy_to_power(e_bess_to_load, dt)
    p_grid_to_load = _energy_to_power(e_grid_to_load, dt)
    p_inj = _energy_to_power(e_inj, dt)
    p_pv_to_bess = _energy_to_power(e_pv_to_bess, dt)

    # időtengely: óra az év elejétől
    time_h = np.arange(t0, tf) * dt

    # oszlopszélesség órában
    bar_width = 0.8 * dt

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=FIGSIZE,
        gridspec_kw={"width_ratios": [0.82, 0.18]},
    )

    ax = axes[0]
    ax_leg = axes[1]

    # ======================
    # 0 FELETT: bejövő teljesítmények
    # ======================
    pos_bottom = np.zeros_like(time_h, dtype=float)

    h_pv = ax.bar(
        time_h,
        p_pv,
        width=bar_width,
        bottom=pos_bottom,
        label="PV termelés",
    )
    pos_bottom += p_pv

    h_bess_out = ax.bar(
        time_h,
        p_bess_to_load,
        width=bar_width,
        bottom=pos_bottom,
        label="BESS kisütés",
    )
    pos_bottom += p_bess_to_load

    h_grid = ax.bar(
        time_h,
        p_grid_to_load,
        width=bar_width,
        bottom=pos_bottom,
        label="Hálózati import",
    )

    # ======================
    # 0 ALATT: kifolyó teljesítmények
    # ======================
    neg_bottom = np.zeros_like(time_h, dtype=float)

    h_load = ax.bar(
        time_h,
        -p_load,
        width=bar_width,
        bottom=neg_bottom,
        label="Fogyasztás",
    )
    neg_bottom -= p_load

    h_export = ax.bar(
        time_h,
        -p_inj,
        width=bar_width,
        bottom=neg_bottom,
        label="PV export",
    )
    neg_bottom -= p_inj

    h_bess_in = ax.bar(
        time_h,
        -p_pv_to_bess,
        width=bar_width,
        bottom=neg_bottom,
        label="BESS töltés",
    )

    # --- tengelyek ---
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_title(f"{season_title} – háztartási teljesítménymérleg\n{household_name}")
    ax.set_xlabel("Idő [h]")
    ax.set_ylabel("Teljesítmény [kW]")
    ax.grid(True, alpha=0.3)

    y_pos = p_pv + p_bess_to_load + p_grid_to_load
    y_neg = p_load + p_inj + p_pv_to_bess
    ymax = max(float(np.nanmax(y_pos)), float(np.nanmax(y_neg)), 1e-6)
    ax.set_ylim(-1.15 * ymax, 1.15 * ymax)

    # --- legenda jobb oldalra ---
    handles = [
        h_pv,
        h_bess_out,
        h_grid,
        h_load,
        h_export,
        h_bess_in,
    ]
    labels = [
        "PV termelés",
        "BESS kisütés",
        "Hálózati import",
        "Fogyasztás",
        "PV export",
        "BESS töltés",
    ]

    ax_leg.legend(handles, labels, loc="center", frameon=False)
    ax_leg.axis("off")

    fig.tight_layout()

    out_png = results_dir / f"household_{household_name}_{season_key}_{WINDOW_DAYS}days_power_stack.png"
    fig.savefig(out_png, dpi=DPI)
    plt.close(fig)

    print(f"Mentve: {out_png}")
    return out_png


def plot_household_power_stack_seasons(
    results_dir: str | Path,
    household_name: str,
    window_days: int = 3,
    dt: float = 0.25,
) -> list[Path]:
    """
    4 évszakhoz külön ábrát ment.
    """
    results_dir = Path(results_dir)

    series = _load_household_series(results_dir, household_name)

    steps_per_day = int(round(24 / dt))
    window_steps = int(round(window_days * steps_per_day))

    season_keys, season_titles, t0s = _season_windows(dt)

    saved_paths: list[Path] = []

    for season_key, season_title, t0 in zip(season_keys, season_titles, t0s):
        out_png = _plot_one_season(
            results_dir=results_dir,
            household_name=household_name,
            season_key=season_key,
            season_title=season_title,
            t0=t0,
            window_steps=window_steps,
            series=series,
            dt=dt,
        )

        if out_png is not None:
            saved_paths.append(out_png)

    print(f"[INFO] Kész. Mentett ábrák száma: {len(saved_paths)}")
    return saved_paths


if __name__ == "__main__":
    plot_household_power_stack_seasons(
        results_dir=r"results_base_pv_bess_boiler",
        household_name="0420144888439778",
        window_days=3,
        dt=DT,
    )