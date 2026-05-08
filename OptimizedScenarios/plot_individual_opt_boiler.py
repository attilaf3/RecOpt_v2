"""
plot_household_boiler_fourpack.py
---------------------------------
1 háztartás bojleres eredményeinek kirajzolása plot_multi stílusban:

Évszakonként külön kép:
  - [0,0] Electric hub (stackelt bar, + befolyók, - kifolyók)
  - [0,1] Electric legenda
  - [1,0] Thermal (T_hss + d_cl)
  - [1,1] Thermal legenda

Forrás: results_individual_opt_boiler/timeseries_<household>.csv
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe


# =========================
# BEÁLLÍTÁSOK
# =========================
RESULTS_DIR = r"results_individual_opt_boiler"
HOUSEHOLD_NAME = "0420144653422463"   # timeseries_<...>.csv-ben a <...> rész
WINDOW_DAYS = 3                       # 2 vagy 3
DT = 0.25                             # 15 perc (óra)

dpi = 300
fontsize = 24
# =========================

plt.rcParams.update({"font.size": fontsize - 2})


def _path_effect(lw: float):
    return [pe.Stroke(linewidth=1.5 * lw, foreground="w"), pe.Normal()]


def _load_timeseries(out_dir: Path, household: str) -> tuple[pd.DataFrame, Path]:
    p = out_dir / f"timeseries_{household}.csv"
    if p.exists():
        return pd.read_csv(p), p

    matches = list(out_dir.glob(f"timeseries_*{household}*.csv"))
    if len(matches) == 1:
        return pd.read_csv(matches[0]), matches[0]

    raise FileNotFoundError(f"Nem találom egyértelműen a timeseries fájlt: {p} (találatok: {len(matches)})")


def plot_household_fourpack_seasons(
    results_dir: str | Path,
    household_name: str,
    window_days: int = 3,
    dt: float = 0.25,
):
    out_dir = Path(results_dir)
    ts, ts_path = _load_timeseries(out_dir, household_name)

    # a run menti ezeket :contentReference[oaicite:2]{index=2}
    needed = [
        "p_pv", "p_ue",
        "p_grid_load", "p_grid_elh",
        "p_pv_grid",
        "p_elh_in",
        "t_hss",
        "d_cl",
    ]
    missing = [c for c in needed if c not in ts.columns]
    if missing:
        raise ValueError(f"Hiányzó oszlop(ok) a timeseries-ben: {missing}")

    # Teljesítmények [kW]
    p_pv = ts["p_pv"].to_numpy(dtype=float)
    p_ue = ts["p_ue"].to_numpy(dtype=float)
    p_grid_load = ts["p_grid_load"].to_numpy(dtype=float)
    p_grid_elh = ts["p_grid_elh"].to_numpy(dtype=float)
    if "p_grid_export" in ts.columns:
        p_pv_grid = ts["p_grid_export"].to_numpy(dtype=float)
    else:
        p_pv_grid = ts["p_pv_grid"].to_numpy(dtype=float)
    p_elh_in = ts["p_elh_in"].to_numpy(dtype=float)

    t_hss = ts["t_hss"].to_numpy(dtype=float)
    d_cl  = ts["d_cl"].to_numpy(dtype=float)

    # Villamos csomópont komponensek
    if "p_grid_import" in ts.columns:
        p_grid_import = ts["p_grid_import"].to_numpy(dtype=float)
    else:
        p_grid_import = p_grid_load + p_grid_elh

    T = len(ts)
    steps_per_day = int(round(24 / dt))
    window_steps = int(round(window_days * steps_per_day))

    # plot_multi-szerű évszak ablakok (órás indexek skálázva 15 percre)
    # plot_multi: t0s=[0,2184,4368,6552], dt=72 óra :contentReference[oaicite:3]{index=3}
    t0s_hourly = [0, 2184, 4368, 6552]
    t0s = [int(t0 * (1 / dt)) for t0 in t0s_hourly]  # 15 perces lépésekre
    season_keys = ["winter", "spring", "summer", "autumn"]
    titles = ["Tél", "Tavasz", "Nyár", "Ősz"]

    bar_kw = dict(width=0.8 * dt)
    plot_kw = dict(lw=3, path_effects=_path_effect(3))

    for season, title, t0 in zip(season_keys, titles, t0s):
        tf = min(t0 + window_steps, T)
        if tf <= t0 + 1:
            continue

        time = np.arange(t0, tf) * dt  # óra az év elejétől

        # szeletek
        pv = p_pv[t0:tf]
        grid_imp = p_grid_import[t0:tf]
        ue = p_ue[t0:tf]
        elh = p_elh_in[t0:tf]
        pv_exp = p_pv_grid[t0:tf]
        temp = t_hss[t0:tf]
        cl = d_cl[t0:tf]

        # ====== 2x2: bal oldalon plotok, jobb oldalon 2 legend axes ======
        fig, axes = plt.subplots(
            nrows=2,
            ncols=2,
            figsize=(20, 12),
            sharex=True,
            gridspec_kw=dict(width_ratios=[0.8, 0.2]),
        )

        ax_e = axes[0, 0]
        leg_e = axes[0, 1]

        ax_t = axes[1, 0]
        leg_t = axes[1, 1]

        # ======================
        # 1) Electric hub (pozitív be, negatív ki)
        # ======================
        bottom = np.zeros_like(time, dtype=float)
        ax_e.bar(time, pv, bottom=bottom, label=r"$PV$", **bar_kw); bottom += pv
        ax_e.bar(time, grid_imp, bottom=bottom, label=r"$Grid\ import$", **bar_kw)

        bottom = np.zeros_like(time, dtype=float)
        ax_e.bar(time, -ue, bottom=bottom, label=r"$Load$", **bar_kw); bottom -= ue
        ax_e.bar(time, -elh, bottom=bottom, label=r"$Boiler\ (opt.)$", **bar_kw); bottom -= elh
        ax_e.bar(time, -pv_exp, bottom=bottom, label=r"$PV\ export$", **bar_kw)

        ax_e.axhline(0.0, color="black", lw=1.0)
        ax_e.set_title(f"{title} – Villamos csomópont", fontsize=fontsize)
        ax_e.set_ylabel("Teljesítmény (kW)", fontsize=fontsize)
        ax_e.grid(True, alpha=0.3)
        ax_e.tick_params(labelsize=fontsize)

        # Szép szimmetrikus y-limit
        y_pos = pv + grid_imp
        y_neg = ue + elh + pv_exp
        ymax = max(float(np.nanmax(y_pos)), float(np.nanmax(y_neg)), 1e-6)
        ax_e.set_ylim(-1.15 * ymax, 1.15 * ymax)

        h_e, l_e = ax_e.get_legend_handles_labels()
        leg_e.legend(h_e, l_e, loc="center", fontsize=fontsize, ncol=1)
        leg_e.axis("off")

        # ======================
        # 2) Thermal: T_hss + d_cl (vékony szürke)
        # ======================
        ln_T, = ax_t.plot(time, temp, label=r"$T_{hss}$", **plot_kw)

        ax_t.set_title(f"{title} – Bojler hőmérséklet + vezérlés", fontsize=fontsize)
        ax_t.set_xlabel("Idő (h)", fontsize=fontsize)
        ax_t.set_ylabel(r"$T_{hss}$ (°C)", fontsize=fontsize)
        ax_t.grid(True, alpha=0.3)
        ax_t.tick_params(labelsize=fontsize)

        ax_t2 = ax_t.twinx()
        ln_cl, = ax_t2.step(time, cl, where="post", color="gray", lw=1.0, alpha=0.9, label=r"$d_{cl}$")
        ax_t2.set_ylim(-0.05, 1.05)
        ax_t2.set_ylabel(r"$d_{cl}$ (-)", fontsize=fontsize, color="gray")
        ax_t2.tick_params(axis="y", labelsize=fontsize - 2, colors="gray")

        # Thermal legend a jobb alsó axes-en
        h1, l1 = ax_t.get_legend_handles_labels()
        h2, l2 = ax_t2.get_legend_handles_labels()
        leg_t.legend(h1 + h2, l1 + l2, loc="center", fontsize=fontsize, ncol=1)
        leg_t.axis("off")

        fig.suptitle(f"Háztartás: {household_name}", fontsize=fontsize + 2)
        fig.tight_layout()
        out_png = out_dir / f"household_{household_name}_{season}_{window_days}days.png"
        plt.savefig(out_png, dpi=dpi)
        plt.close(fig)
        print(f"Mentve: {out_png}")

    print(f"Forrás: {ts_path}")



if __name__ == "__main__":
    plot_household_fourpack_seasons(
        results_dir="results_individual_opt_boiler",
        household_name="0420144888439778",
        window_days=3,
        dt=DT,
    )