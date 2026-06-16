"""
plot_household_bess_boiler_sixpack.py
--------------------------------------
1 háztartás kombinált BESS + bojler eredményeinek kirajzolása.

Évszakonként külön kép, 3x2-es elrendezésben:
  - [0,0] Villamos csomópont (stackelt bar, + befolyók, - kifolyók)
  - [0,1] Villamos legend
  - [1,0] BESS SOC + d_bess
  - [1,1] BESS legend
  - [2,0] Bojler T_hss + d_cl
  - [2,1] Bojler legend
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
RESULTS_DIR = r"results_individual_opt_bess_boiler"
HOUSEHOLD_NAME = "0420144653422463"   # timeseries_<...>.csv-ben a <...> rész
WINDOW_DAYS = 3                       # 2 vagy 3
DT = 0.25                             # 15 perc (óra)


DPI = 300
FONTSIZE = 22
# =========================

plt.rcParams.update({"font.size": FONTSIZE - 2})


def _path_effect(lw: float):
    return [pe.Stroke(linewidth=1.5 * lw, foreground="w"), pe.Normal()]


def _load_timeseries(out_dir: Path, household: str) -> tuple[pd.DataFrame, Path]:
    """timeseries fájl betöltése pontos vagy részleges household egyezéssel."""
    p = out_dir / f"timeseries_{household}.csv"
    if p.exists():
        return pd.read_csv(p), p

    matches = list(out_dir.glob(f"timeseries_*{household}*.csv"))
    if len(matches) == 1:
        return pd.read_csv(matches[0]), matches[0]

    if len(matches) > 1:
        matches_sorted = sorted(matches, key=lambda x: len(x.name))
        return pd.read_csv(matches_sorted[0]), matches_sorted[0]

    raise FileNotFoundError(
        f"Nem találom egyértelműen a timeseries fájlt: {p} (találatok: {len(matches)})"
    )


def _col(ts: pd.DataFrame, name: str, default: float = 0.0) -> np.ndarray:
    """Oszlop kiolvasása; ha nincs, konstans nulla vektor."""
    if name in ts.columns:
        return ts[name].to_numpy(dtype=float)
    return np.full(len(ts), float(default), dtype=float)


def _first_existing_col(ts: pd.DataFrame, names: list[str], required: bool = True) -> np.ndarray:
    """Több lehetséges oszlopnév közül az első meglévő kiolvasása."""
    for name in names:
        if name in ts.columns:
            return ts[name].to_numpy(dtype=float)
    if required:
        raise ValueError(f"Hiányzó oszlop. Ezek közül legalább az egyik kellene: {names}")
    return np.zeros(len(ts), dtype=float)


def _season_windows(dt: float):
    """Évszak kezdőpontok 15 perces/osztott időlépésre skálázva."""
    t0s_hourly = [0, 2184, 4368, 6552]
    t0s = [int(t0 * (1 / dt)) for t0 in t0s_hourly]
    season_keys = ["winter", "spring", "summer", "autumn"]
    titles = ["Tél", "Tavasz", "Nyár", "Ősz"]
    return season_keys, titles, t0s


def plot_household_sixpack_seasons(
    results_dir: str | Path,
    household_name: str,
    window_days: int = 3,
    dt: float = 0.25,
):
    out_dir = Path(results_dir)
    ts, ts_path = _load_timeseries(out_dir, household_name)

    # Minimálisan szükséges oszlopok / alternatívák
    required_any = [
        ["p_pv_kw"],
        ["p_ue_kw"],
        ["p_elh_in_kw"],
        ["p_bess_in_kw"],
        ["p_bess_out_kw", "p_bess_load_kw"],
        ["e_bess_kwh"],
        ["d_bess"],
        ["t_hss_C"],
        ["d_cl"],
    ]
    for alternatives in required_any:
        if not any(c in ts.columns for c in alternatives):
            raise ValueError(f"Hiányzó oszlop. Ezek közül legalább az egyik kellene: {alternatives}")

    # -------- Villamos csomópont komponensek, teljesítmények [kW] --------
    p_pv = _first_existing_col(ts, ["p_pv_kw"])
    p_ue = _first_existing_col(ts, ["p_ue_kw"])

    p_elh_in = _first_existing_col(ts, ["p_elh_in_kw"])
    p_bess_in = _first_existing_col(ts, ["p_bess_in_kw"])
    p_bess_out = _first_existing_col(ts, ["p_bess_out_kw", "p_bess_load_kw"])

    # Import: ha van direkt p_grid_import, azt használjuk; ha nincs, splitből rakjuk össze.
    if "p_grid_import_kw" in ts.columns:
        p_grid_import = ts["p_grid_import_kw"].to_numpy(dtype=float)
    else:
        p_grid_import = (
                _col(ts, "p_grid_load_kw")
                + _col(ts, "p_grid_elh_kw")
                + _col(ts, "p_grid_bess_kw")
        )

    # Export: kombinált kódban általában p_grid_export és p_pv_grid is ugyanaz.
    p_pv_export = _first_existing_col(ts, ["p_grid_export_kw", "p_pv_grid_kw"])

    # -------- Állapotok / bináris jelek --------
    soc = _first_existing_col(ts, ["e_bess_kwh"])
    d_bess = _first_existing_col(ts, ["d_bess"])
    t_hss = _first_existing_col(ts, ["t_hss_C"])
    d_cl = _first_existing_col(ts, ["d_cl"])

    T = len(ts)
    steps_per_day = int(round(24 / dt))
    window_steps = int(round(window_days * steps_per_day))

    season_keys, titles, t0s = _season_windows(dt)

    bar_kw = dict(width=0.8 * dt)
    plot_kw = dict(lw=3, path_effects=_path_effect(3))

    saved_paths: list[Path] = []

    for season, title, t0 in zip(season_keys, titles, t0s):
        tf = min(t0 + window_steps, T)
        if tf <= t0 + 1:
            continue

        time = np.arange(t0, tf) * dt  # óra az év elejétől

        # Szeletek
        pv = p_pv[t0:tf]
        grid_imp = p_grid_import[t0:tf]
        bess_out = p_bess_out[t0:tf]

        ue = p_ue[t0:tf]
        elh = p_elh_in[t0:tf]
        bess_in = p_bess_in[t0:tf]
        pv_exp = p_pv_export[t0:tf]

        soc_s = soc[t0:tf]
        d_bess_s = d_bess[t0:tf]
        temp_s = t_hss[t0:tf]
        d_cl_s = d_cl[t0:tf]

        # ====== 3x2: bal oldalon plotok, jobb oldalon legend axes ======
        fig, axes = plt.subplots(
            nrows=3,
            ncols=2,
            figsize=(22, 17),
            sharex=True,
            gridspec_kw=dict(width_ratios=[0.82, 0.18], height_ratios=[1.25, 1.0, 1.0]),
        )

        ax_e, leg_e = axes[0, 0], axes[0, 1]
        ax_s, leg_s = axes[1, 0], axes[1, 1]
        ax_t, leg_t = axes[2, 0], axes[2, 1]

        # ======================
        # 1) Villamos csomópont
        # Befolyó: PV, grid import, BESS kisütés
        # Kifolyó: általános fogyasztás, bojler, BESS töltés, PV export
        # ======================
        bottom = np.zeros_like(time, dtype=float)
        ax_e.bar(time, pv, bottom=bottom, label=r"$PV$", **bar_kw)
        bottom += pv

        ax_e.bar(time, grid_imp, bottom=bottom, label=r"$Grid\ import$", **bar_kw)
        bottom += grid_imp

        ax_e.bar(time, bess_out, bottom=bottom, label=r"$BESS\ discharge$", **bar_kw)

        bottom = np.zeros_like(time, dtype=float)
        ax_e.bar(time, -ue, bottom=bottom, label=r"$Load$", **bar_kw)
        bottom -= ue

        ax_e.bar(time, -elh, bottom=bottom, label=r"$Boiler\ (opt.)$", **bar_kw)
        bottom -= elh

        ax_e.bar(time, -bess_in, bottom=bottom, label=r"$BESS\ charge$", **bar_kw)
        bottom -= bess_in

        ax_e.bar(time, -pv_exp, bottom=bottom, label=r"$PV\ export$", **bar_kw)

        ax_e.axhline(0.0, color="black", lw=1.0)
        ax_e.set_title(f"{title} – Villamos csomópont", fontsize=FONTSIZE)
        ax_e.set_ylabel("Teljesítmény (kW)", fontsize=FONTSIZE)
        ax_e.grid(True, alpha=0.3)
        ax_e.tick_params(labelsize=FONTSIZE)

        y_pos = pv + grid_imp + bess_out
        y_neg = ue + elh + bess_in + pv_exp
        ymax = max(float(np.nanmax(y_pos)), float(np.nanmax(y_neg)), 1e-6)
        ax_e.set_ylim(-1.15 * ymax, 1.15 * ymax)

        h_e, l_e = ax_e.get_legend_handles_labels()
        leg_e.legend(h_e, l_e, loc="center", fontsize=FONTSIZE - 1, ncol=1)
        leg_e.axis("off")

        # ======================
        # 2) SOC + d_bess
        # ======================
        ax_s.plot(time, soc_s, label=r"$SOC$", **plot_kw)
        ax_s.set_title(f"{title} – BESS SOC + töltési bináris jel", fontsize=FONTSIZE)
        ax_s.set_ylabel("SOC (kWh)", fontsize=FONTSIZE)
        ax_s.grid(True, alpha=0.3)
        ax_s.tick_params(labelsize=FONTSIZE)

        ax_s2 = ax_s.twinx()
        ax_s2.step(
            time,
            d_bess_s,
            where="post",
            color="gray",
            lw=1.0,
            alpha=0.9,
            label=r"$d_{bess}$",
        )
        ax_s2.set_ylim(-0.05, 1.05)
        ax_s2.set_yticks([0, 1])
        ax_s2.set_ylabel(r"$d_{bess}$ (-)", fontsize=FONTSIZE, color="gray")
        ax_s2.tick_params(axis="y", labelsize=FONTSIZE - 2, colors="gray")

        h1, l1 = ax_s.get_legend_handles_labels()
        h2, l2 = ax_s2.get_legend_handles_labels()
        leg_s.legend(h1 + h2, l1 + l2, loc="center", fontsize=FONTSIZE, ncol=1)
        leg_s.axis("off")

        # ======================
        # 3) Bojler hőmérséklet + d_cl
        # ======================
        ax_t.plot(time, temp_s, label=r"$T_{hss}$", **plot_kw)
        ax_t.set_title(f"{title} – Bojler hőmérséklet + vezérlés", fontsize=FONTSIZE)
        ax_t.set_xlabel("Idő (h)", fontsize=FONTSIZE)
        ax_t.set_ylabel(r"$T_{hss}$ (°C)", fontsize=FONTSIZE)
        ax_t.grid(True, alpha=0.3)
        ax_t.tick_params(labelsize=FONTSIZE)

        ax_t2 = ax_t.twinx()
        ax_t2.step(
            time,
            d_cl_s,
            where="post",
            color="gray",
            lw=1.0,
            alpha=0.9,
            label=r"$d_{cl}$",
        )
        ax_t2.set_ylim(-0.05, 1.05)
        ax_t2.set_yticks([0, 1])
        ax_t2.set_ylabel(r"$d_{cl}$ (-)", fontsize=FONTSIZE, color="gray")
        ax_t2.tick_params(axis="y", labelsize=FONTSIZE - 2, colors="gray")

        h1, l1 = ax_t.get_legend_handles_labels()
        h2, l2 = ax_t2.get_legend_handles_labels()
        leg_t.legend(h1 + h2, l1 + l2, loc="center", fontsize=FONTSIZE, ncol=1)
        leg_t.axis("off")

        fig.suptitle(f"Háztartás: {household_name}", fontsize=FONTSIZE + 2)
        fig.tight_layout()

        out_png = out_dir / f"household_{household_name}_{season}_{window_days}days_bess_boiler.png"
        plt.savefig(out_png, dpi=DPI)
        plt.close(fig)

        saved_paths.append(out_png)
        print(f"Mentve: {out_png}")

    print(f"Forrás: {ts_path}")
    return saved_paths


if __name__ == "__main__":
    plot_household_sixpack_seasons(
        results_dir=RESULTS_DIR,
        household_name=HOUSEHOLD_NAME,
        window_days=WINDOW_DAYS,
        dt=DT,
    )