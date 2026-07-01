from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# BEÁLLÍTÁSOK
# =========================
RESULTS_DIR = r"./results_nonopt_community_B_proportional"  # run_case output mappája
WINDOW_DAYS = 3                                      # hány nap / évszak
DT = 0.25                                           # 15 perc = 0.25 h
DPI = 300
FONTSIZE = 18
FIGSIZE = (24, 13)
# =========================

plt.rcParams.update({"font.size": FONTSIZE})


def _season_windows(dt: float) -> tuple[list[str], list[str], list[int]]:
    """
    Évszak kezdőpontok:
      - winter: év eleje
      - spring: 2184. óra
      - summer: 4368. óra
      - autumn: 6552. óra
    """
    t0s_hourly = [0, 2184, 4368, 6552]
    season_keys = ["winter", "spring", "summer", "autumn"]
    season_titles = ["Tél", "Tavasz", "Nyár", "Ősz"]
    t0s = [int(round(hour / dt)) for hour in t0s_hourly]
    return season_keys, season_titles, t0s


def _read_community_timeseries(results_dir: Path) -> pd.DataFrame:
    path = results_dir / "community_timeseries.csv"
    if not path.exists():
        raise FileNotFoundError(f"Hiányzó fájl: {path}")
    return pd.read_csv(path)


def _first_existing(df: pd.DataFrame, names: list[str], *, required: bool = True) -> np.ndarray:
    for name in names:
        if name in df.columns:
            return df[name].to_numpy(dtype=float)
    if required:
        raise KeyError(
            "Egyik oszlop sem található: "
            + ", ".join(names)
            + f"\nElérhető oszlopok: {list(df.columns)}"
        )
    return np.zeros(len(df), dtype=float)


def _energy_like_to_power(x: np.ndarray, dt: float, *, source_is_power: bool) -> np.ndarray:
    """
    Nonopt community_timeseries: e_*_total [kWh / lépés] -> kW.
    Opt community_timeseries: p_*_sum [kW] -> kW.
    """
    arr = np.asarray(x, dtype=float)
    return arr if source_is_power else arr / float(dt)


def _load_community_series(results_dir: Path, dt: float) -> dict[str, np.ndarray]:
    df = _read_community_timeseries(results_dir)

    # Opt fájlban p_*_sum oszlopok vannak, nonopt fájlban e_*_total oszlopok.
    source_is_power = any(c.startswith("p_") for c in df.columns)

    def col(names: list[str], required: bool = True) -> np.ndarray:
        return _energy_like_to_power(_first_existing(df, names, required=required), dt, source_is_power=source_is_power)

    series = {
        "p_load": col(["e_load_total", "p_total_load_sum"]),
        "p_base_load": col(["e_ue_total", "e_load_base_total", "p_ue_sum"]),
        "p_boiler": col(["e_boiler_total", "p_el_heater_sum"], required=False),
        "p_hp": col(["e_hp_total", "p_hp_sum"], required=False),
        "p_pv": col(["e_pv_total", "p_pv_sum"]),
        "p_pv_to_load": col(["e_pv_to_load_total", "p_pv_load_sum"], required=False),
        "p_pv_to_bess": col(["e_pv_to_bess_total", "p_pv_bess_sum"], required=False),
        "p_bess_to_load": col(["e_bess_to_load_total", "p_bess_load_sum"], required=False),
        "p_bess_local_to_load": col(["e_bess_local_to_load_total", "p_bess_load_sum"], required=False),
        "p_shared_in": col(["e_shared_in_total", "p_shared_in_sum"], required=False),
        "p_shared_to_base": col(["e_shared_to_base_total", "p_shared_ue_sum"], required=False),
        "p_shared_to_boiler": col(["e_shared_to_boiler_total", "p_shared_elh_sum"], required=False),
        "p_shared_out": col(["e_shared_out_total", "p_shared_out_sum"], required=False),
        "p_grid_to_load": col(["e_grid_to_load_total", "p_grid_load_sum"], required=False),
        "p_grid_to_base": col(["e_grid_to_base_total", "p_grid_ue_sum"], required=False),
        "p_grid_to_boiler": col(["e_grid_to_boiler_total", "p_grid_elh_sum"], required=False),
        "p_grid_to_bess": col(["e_grid_to_bess_total", "p_grid_bess_sum"], required=False),
        "p_grid_import": col(["e_grid_import_total", "p_grid_import_sum"], required=False),
        "p_inj": col(["e_inj_total", "p_grid_export_sum", "e_grid_export_total"], required=False),
        "e_bess": _first_existing(df, ["e_bess_total", "e_bess_sum"], required=False),
    }

    # A külső hálózatos ábrához:
    # P_with = teljes külső import, P_inj = külső export, P_shared = közösségen belüli megosztás.
    if np.nanmax(series["p_grid_import"]) <= 1e-12:
        series["p_grid_import"] = series["p_grid_to_load"] + series["p_grid_to_bess"]

    return series


def _interp_for_smooth(time_h: np.ndarray, y: np.ndarray, multiplier: int = 10) -> tuple[np.ndarray, np.ndarray]:
    if len(time_h) < 2:
        return time_h, y
    n = max(2, len(time_h) * multiplier)
    t_plot = np.linspace(float(time_h[0]), float(time_h[-1]), n)
    y_plot = np.interp(t_plot, time_h, y)
    return t_plot, y_plot


def _plot_one_season(
    results_dir: Path,
    season_key: str,
    season_title: str,
    t0: int,
    window_steps: int,
    series: dict[str, np.ndarray],
    dt: float,
) -> Path | None:
    T = len(series["p_load"])
    tf = min(t0 + window_steps, T)
    if tf <= t0 + 1:
        print(f"[WARN] {season_title}: nincs elég adat ehhez az ablakhoz.")
        return None

    sl = slice(t0, tf)
    time_h = np.arange(t0, tf) * dt
    bar_width = 0.8 * dt

    p_base_load = series["p_base_load"][sl]
    p_boiler = series["p_boiler"][sl]
    p_hp = series["p_hp"][sl]
    p_pv = series["p_pv"][sl]
    p_bess_to_load = series["p_bess_to_load"][sl]
    p_grid_to_base = series["p_grid_to_base"][sl]
    p_grid_to_boiler = series["p_grid_to_boiler"][sl]
    p_grid_to_bess = series["p_grid_to_bess"][sl]
    p_inj = series["p_inj"][sl]
    p_pv_to_bess = series["p_pv_to_bess"][sl]
    p_shared_in = series["p_shared_in"][sl]
    p_shared_to_base = series["p_shared_to_base"][sl]
    p_shared_to_boiler = series["p_shared_to_boiler"][sl]
    p_shared_out = series["p_shared_out"][sl]
    p_grid_import = series["p_grid_import"][sl]
    e_bess = series["e_bess"][sl]

    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=FIGSIZE,
        gridspec_kw={"width_ratios": [0.82, 0.18], "height_ratios": [0.55, 0.45]},
        sharex="col",
    )

    ax_stack = axes[0, 0]
    leg_stack = axes[0, 1]
    ax_c = axes[1, 0]
    leg_c = axes[1, 1]

    # ======================
    # Felső: közösségi stackelt teljesítménymérleg
    # ======================
    pos_bottom = np.zeros_like(time_h, dtype=float)

    h_pv = ax_stack.bar(time_h, p_pv, width=bar_width, bottom=pos_bottom, label="PV termelés")
    pos_bottom += p_pv

    h_bess_out = ax_stack.bar(time_h, p_bess_to_load, width=bar_width, bottom=pos_bottom, label="BESS kisütés")
    pos_bottom += p_bess_to_load

    h_shared_in = ax_stack.bar(time_h, p_shared_in, width=bar_width, bottom=pos_bottom, label="Közösségből vett energia")
    pos_bottom += p_shared_in

    h_grid_base = ax_stack.bar(time_h, p_grid_to_base, width=bar_width, bottom=pos_bottom, label="Hálózati import – A tarifa")
    pos_bottom += p_grid_to_base

    h_grid_boiler = ax_stack.bar(time_h, p_grid_to_boiler, width=bar_width, bottom=pos_bottom, label="Hálózati import – B tarifa / bojler")
    pos_bottom += p_grid_to_boiler

    neg_bottom = np.zeros_like(time_h, dtype=float)

    h_base_load = ax_stack.bar(time_h, -p_base_load, width=bar_width, bottom=neg_bottom, label="Általános fogyasztás")
    neg_bottom -= p_base_load

    h_boiler = ax_stack.bar(time_h, -p_boiler, width=bar_width, bottom=neg_bottom, label="Bojler fogyasztás")
    neg_bottom -= p_boiler

    h_hp = ax_stack.bar(time_h, -p_hp, width=bar_width, bottom=neg_bottom, label="Hőszivattyú fogyasztás")
    neg_bottom -= p_hp

    h_shared_out = ax_stack.bar(time_h, -p_shared_out, width=bar_width, bottom=neg_bottom, label="Megosztott PV energia")
    neg_bottom -= p_shared_out

    h_export = ax_stack.bar(time_h, -p_inj, width=bar_width, bottom=neg_bottom, label="Hálózati export")
    neg_bottom -= p_inj

    h_bess_in_pv = ax_stack.bar(time_h, -p_pv_to_bess, width=bar_width, bottom=neg_bottom, label="BESS töltés PV-ből")
    neg_bottom -= p_pv_to_bess

    h_bess_in_grid = ax_stack.bar(time_h, -p_grid_to_bess, width=bar_width, bottom=neg_bottom, label="BESS töltés hálózatból")

    ax_stack.axhline(0.0, color="black", linewidth=1.0)
    ax_stack.set_title(f"{season_title} – közösségi teljesítménymérleg")
    ax_stack.set_ylabel("Teljesítmény [kW]")
    ax_stack.grid(True, alpha=0.3)

    y_pos = p_pv + p_bess_to_load + p_shared_in + p_grid_to_base + p_grid_to_boiler
    y_neg = p_base_load + p_boiler + p_hp + p_shared_out + p_inj + p_pv_to_bess + p_grid_to_bess
    ymax = max(float(np.nanmax(y_pos)), float(np.nanmax(y_neg)), 1e-6)
    ax_stack.set_ylim(-1.15 * ymax, 1.15 * ymax)

    handles_stack = [
        h_pv,
        h_bess_out,
        h_shared_in,
        h_grid_base,
        h_grid_boiler,
        h_base_load,
        h_boiler,
        h_hp,
        h_shared_out,
        h_export,
        h_bess_in_pv,
        h_bess_in_grid,
    ]
    labels_stack = [h.get_label() for h in handles_stack]
    leg_stack.legend(handles_stack, labels_stack, loc="center", frameon=False)
    leg_stack.axis("off")

    # ======================
    # Alsó: külső hálózat + megosztás görbés plot
    # ======================
    p_with = p_grid_import
    p_shared = np.minimum(p_shared_in, p_shared_out)

    t_plot, p_inj_plot = _interp_for_smooth(time_h, p_inj)
    _, p_with_plot = _interp_for_smooth(time_h, p_with)
    _, p_shared_plot = _interp_for_smooth(time_h, p_shared)

    plot_kw = {"linewidth": 2.0}
    ax_c.plot(t_plot, p_inj_plot, label=r"$P_\mathrm{inj}$", color="tab:red", **plot_kw)
    ax_c.plot(t_plot, p_with_plot, label=r"$P_\mathrm{with}$", color="tab:blue", **plot_kw)
    ax_c.plot(t_plot, p_shared_plot, label=r"$P_\mathrm{shared}$", color="tab:green", **plot_kw)

    ax_c.fill_between(
        t_plot,
        p_shared_plot,
        p_with_plot,
        where=p_with_plot > p_shared_plot,
        label=r"E$_\mathrm{\leftarrow grid}$",
        color="tab:blue",
        alpha=0.3,
    )
    ax_c.fill_between(
        t_plot,
        p_shared_plot,
        p_inj_plot,
        where=p_inj_plot > p_shared_plot,
        label=r"E$_\mathrm{\rightarrow grid}$",
        color="tab:red",
        alpha=0.3,
    )
    ax_c.fill_between(
        t_plot,
        0,
        p_shared_plot,
        where=p_shared_plot > 0,
        label=r"E$_\mathrm{shared}$",
        color="tab:green",
        alpha=0.3,
    )

    ax_c.set_xlabel("Idő [h]")
    ax_c.set_ylabel("Teljesítmény [kW]")
    ax_c.set_title("Közösség és külső hálózat közötti energiacsere", fontsize=FONTSIZE)
    ax_c.grid(True, alpha=0.3)

    hc, lc = ax_c.get_legend_handles_labels()
    leg_c.legend(hc, lc, loc="center", fontsize=FONTSIZE, ncol=1, frameon=False)
    leg_c.axis("off")

    fig.tight_layout()
    out_png = results_dir / f"community_{season_key}_{WINDOW_DAYS}days_power_stack_shared.png"
    fig.savefig(out_png, dpi=DPI)
    plt.close(fig)

    print(f"Mentve: {out_png}")
    return out_png


def plot_community_power_stack_seasons(
    results_dir: str | Path,
    window_days: int = 3,
    dt: float = 0.25,
) -> list[Path]:
    """4 évszakhoz külön közösségi ábrát ment."""
    results_dir = Path(results_dir)
    series = _load_community_series(results_dir, dt=dt)

    steps_per_day = int(round(24 / dt))
    window_steps = int(round(window_days * steps_per_day))
    season_keys, season_titles, t0s = _season_windows(dt)

    saved_paths: list[Path] = []
    for season_key, season_title, t0 in zip(season_keys, season_titles, t0s):
        out_png = _plot_one_season(
            results_dir=results_dir,
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
    plot_community_power_stack_seasons(
        results_dir=RESULTS_DIR,
        window_days=WINDOW_DAYS,
        dt=DT,
    )
