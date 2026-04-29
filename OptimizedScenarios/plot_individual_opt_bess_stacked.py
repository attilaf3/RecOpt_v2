"""
plot_household_bess_fourpack.py
--------------------------------
1 háztartás BESS-es eredményeinek kirajzolása a bojleres "fourpack" stílusban:

Évszakonként külön kép:
  - [0,0] Electric hub (stackelt bar, + befolyók, - kifolyók)
  - [0,1] Electric legenda
  - [1,0] SOC + d_bess
  - [1,1] SOC legenda

Forrás: results_individual_opt_bess/timeseries_<household>.csv
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
RESULTS_DIR = r"results_individual_opt_bess"
HOUSEHOLD_NAME = "0420144653422463"   # timeseries_<...>.csv-ben a <...> rész (ha nem pontos, keres fallbackkal)
WINDOW_DAYS = 3                       # 2 vagy 3
DT = 0.25                             # 15 perc (óra) – az ablakok skálázásához kell
P_ARE_KWH_PER_STEP = True             # True: p_* oszlopok kWh/lépés; False: kW

dpi = 300
fontsize = 24
# =========================

plt.rcParams.update({"font.size": fontsize - 2})


def _path_effect(lw: float):
    return [pe.Stroke(linewidth=1.5 * lw, foreground="w"), pe.Normal()]


def _to_kwh_step(x: np.ndarray, dt: float, already_kwh_step: bool) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    return x if already_kwh_step else x * dt


def _load_timeseries(out_dir: Path, household: str) -> tuple[pd.DataFrame, Path]:
    # 1) pontos egyezés
    p = out_dir / f"timeseries_{household}.csv"
    if p.exists():
        return pd.read_csv(p), p

    # 2) safe_name / részleges egyezés (pl. ha névben nincs leading 0 vagy hasonló)
    matches = list(out_dir.glob(f"timeseries_*{household}*.csv"))
    if len(matches) == 1:
        return pd.read_csv(matches[0]), matches[0]

    # 3) ha több találat van, próbáljuk azt, ami a leghosszabb közös prefixet adja
    if len(matches) > 1:
        # egyszerű: a legrövidebb fájlnevet válasszuk (gyakran az a legpontosabb ID)
        matches_sorted = sorted(matches, key=lambda x: len(x.name))
        return pd.read_csv(matches_sorted[0]), matches_sorted[0]

    raise FileNotFoundError(f"Nem találom egyértelműen a timeseries fájlt: {p} (találatok: {len(matches)})")


def plot_household_fourpack_seasons(
    results_dir: str | Path,
    household_name: str,
    window_days: int = 3,
    dt: float = 0.25,
    p_are_kwh_per_step: bool = True,
):
    out_dir = Path(results_dir)
    ts, ts_path = _load_timeseries(out_dir, household_name)

    needed = [
        "p_pv",
        "p_total_load",
        "p_grid_import",
        "p_grid_export",
        "p_bess_in",
        "p_bess_out",
        "e_bess",
        "d_bess",
    ]
    missing = [c for c in needed if c not in ts.columns]
    if missing:
        raise ValueError(f"Hiányzó oszlop(ok) a timeseries-ben: {missing}")

    # Egységesítés kWh/lépésre
    p_pv          = _to_kwh_step(ts["p_pv"].to_numpy(), dt, p_are_kwh_per_step)
    p_total_load  = _to_kwh_step(ts["p_total_load"].to_numpy(), dt, p_are_kwh_per_step)
    p_grid_import = _to_kwh_step(ts["p_grid_import"].to_numpy(), dt, p_are_kwh_per_step)
    p_grid_export = _to_kwh_step(ts["p_grid_export"].to_numpy(), dt, p_are_kwh_per_step)
    p_bess_in     = _to_kwh_step(ts["p_bess_in"].to_numpy(), dt, p_are_kwh_per_step)
    p_bess_out    = _to_kwh_step(ts["p_bess_out"].to_numpy(), dt, p_are_kwh_per_step)

    # SOC és bináris jel (SOC: kWh, d_bess: 0/1)
    soc   = ts["e_bess"].to_numpy(dtype=float)
    d_bess = ts["d_bess"].to_numpy(dtype=float)

    T = len(ts)
    steps_per_day = int(round(24 / dt))
    window_steps = int(round(window_days * steps_per_day))

    # Ugyanaz a szezon-ablak logika, mint a bojleresben (plot_multi stílus)
    t0s_hourly = [0, 2184, 4368, 6552]                 # órás index az évben
    t0s = [int(t0 * (1 / dt)) for t0 in t0s_hourly]    # 15 perces lépésekre
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
        bess_out = p_bess_out[t0:tf]

        load = p_total_load[t0:tf]
        bess_in = p_bess_in[t0:tf]
        pv_exp = p_grid_export[t0:tf]

        soc_s = soc[t0:tf]
        d_s = d_bess[t0:tf]

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

        ax_s = axes[1, 0]
        leg_s = axes[1, 1]

        # ======================
        # 1) Electric hub (pozitív be, negatív ki)
        # Befolyó: grid_import, pv, bess_kisütés
        # Kifolyó: fogyasztás, bess_töltés, pv_export
        # ======================
        bottom = np.zeros_like(time, dtype=float)
        ax_e.bar(time, pv, bottom=bottom, label=r"$PV$", **bar_kw); bottom += pv
        ax_e.bar(time, grid_imp, bottom=bottom, label=r"$Grid\ import$", **bar_kw); bottom += grid_imp
        ax_e.bar(time, bess_out, bottom=bottom, label=r"$BESS\ discharge$", **bar_kw)

        bottom = np.zeros_like(time, dtype=float)
        ax_e.bar(time, -load, bottom=bottom, label=r"$Load$", **bar_kw); bottom -= load
        ax_e.bar(time, -bess_in, bottom=bottom, label=r"$BESS\ charge$", **bar_kw); bottom -= bess_in
        ax_e.bar(time, -pv_exp, bottom=bottom, label=r"$PV\ export$", **bar_kw)

        ax_e.axhline(0.0, color="black", lw=1.0)
        ax_e.set_title(f"{title} – Villamos csomópont", fontsize=fontsize)
        ax_e.set_ylabel("Energia (kWh / 15 perc)" if p_are_kwh_per_step else "Energia (kWh / lépés)", fontsize=fontsize)
        ax_e.grid(True, alpha=0.3)
        ax_e.tick_params(labelsize=fontsize)

        # Szép szimmetrikus y-limit
        y_pos = pv + grid_imp + bess_out
        y_neg = load + bess_in + pv_exp
        ymax = max(float(np.nanmax(y_pos)), float(np.nanmax(y_neg)), 1e-6)
        ax_e.set_ylim(-1.15 * ymax, 1.15 * ymax)

        h_e, l_e = ax_e.get_legend_handles_labels()
        leg_e.legend(h_e, l_e, loc="center", fontsize=fontsize, ncol=1)
        leg_e.axis("off")

        # ======================
        # 2) SOC + d_bess (vékony szürke)
        # ======================
        ln_soc, = ax_s.plot(time, soc_s, label=r"$SOC$", **plot_kw)
        ax_s.set_title(f"{title} – SOC + töltési bináris jel", fontsize=fontsize)
        ax_s.set_xlabel("Idő (h)", fontsize=fontsize)
        ax_s.set_ylabel("SOC (kWh)", fontsize=fontsize)
        ax_s.grid(True, alpha=0.3)
        ax_s.tick_params(labelsize=fontsize)

        ax_s2 = ax_s.twinx()
        ln_db, = ax_s2.step(time, d_s, where="post", color="gray", lw=1.0, alpha=0.9, label=r"$d_{bess}$")
        ax_s2.set_ylim(-0.05, 1.05)
        ax_s2.set_yticks([0, 1])
        ax_s2.set_ylabel(r"$d_{bess}$ (-)", fontsize=fontsize, color="gray")
        ax_s2.tick_params(axis="y", labelsize=fontsize - 2, colors="gray")

        h1, l1 = ax_s.get_legend_handles_labels()
        h2, l2 = ax_s2.get_legend_handles_labels()
        leg_s.legend(h1 + h2, l1 + l2, loc="center", fontsize=fontsize, ncol=1)
        leg_s.axis("off")

        fig.suptitle(f"Háztartás: {household_name}", fontsize=fontsize + 2)
        fig.tight_layout()

        out_png = out_dir / f"household_{household_name}_{season}_{window_days}days.png"
        plt.savefig(out_png, dpi=dpi)
        plt.close(fig)
        print(f"Mentve: {out_png}")

    print(f"Forrás: {ts_path}")


if __name__ == "__main__":
    plot_household_fourpack_seasons(
        results_dir=RESULTS_DIR,
        household_name=HOUSEHOLD_NAME,
        window_days=WINDOW_DAYS,
        dt=DT,
        p_are_kwh_per_step=P_ARE_KWH_PER_STEP,
    )