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


def _load_optional_household_series(
    results_dir: Path,
    filename: str,
    household_name: str,
) -> np.ndarray | None:
    """
    Opcionális idősor betöltése.

    Ha a fájl nincs jelen, vagy a kiválasztott háztartás nem szerepel benne,
    None értékkel tér vissza. Így a régebbi, bojlermodell nélküli
    eredménymappák is továbbra is ábrázolhatók.
    """
    path = results_dir / filename
    if not path.exists():
        return None

    df = pd.read_csv(path)
    if household_name not in df.columns:
        return None

    return df[household_name].to_numpy(dtype=float)


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
      - e_load.csv              teljes fogyasztás
      - e_base_load.csv         általános fogyasztás
      - e_boiler.csv            bojler fogyasztás
      - e_pv.csv
      - e_bess_to_load.csv
      - e_grid_to_load.csv      teljes hálózati import
      - e_grid_to_base.csv      A tarifás hálózati import
      - e_grid_to_boiler.csv    B tarifás bojlerimport
      - e_inj.csv
      - e_pv_to_bess.csv
      - e_bess.csv
      - d_bess_ch.csv
      - d_bess_dis.csv
    """
    files = {
        "e_load": "e_load.csv",
        "e_base_load": "e_base_load.csv",
        "e_boiler": "e_boiler.csv",
        "e_pv": "e_pv.csv",
        "e_bess_to_load": "e_bess_to_load.csv",
        "e_grid_to_load": "e_grid_to_load.csv",
        "e_grid_to_base": "e_grid_to_base.csv",
        "e_grid_to_boiler": "e_grid_to_boiler.csv",
        "e_inj": "e_inj.csv",
        "e_pv_to_bess": "e_pv_to_bess.csv",
        "e_grid_to_bess": "e_grid_to_bess.csv",
        "e_bess": "e_bess.csv",
        "d_bess_ch": "d_bess_ch.csv",
        "d_bess_dis": "d_bess_dis.csv",
    }

    out: dict[str, np.ndarray] = {}

    for key, filename in files.items():
        df = _load_matrix(results_dir, filename)
        out[key] = _get_series(df, household_name, filename)

    # Dinamikus bojlermodellhez tartozó opcionális idősorok.
    # A t_hss.csv csak akkor értelmezhető az adott háztartásra,
    # ha a hőmérséklet-idősorban van pozitív, véges érték.
    optional_files = {
        "t_hss": "t_hss.csv",
        "d_boiler_available": "d_boiler_available.csv",
        "d_boiler_on": "d_boiler_on.csv",
    }

    for key, filename in optional_files.items():
        values = _load_optional_household_series(
            results_dir,
            filename,
            household_name,
        )
        if values is not None:
            out[key] = values

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
    e_base_load = series["e_base_load"][sl]
    e_boiler = series["e_boiler"][sl]

    e_pv = series["e_pv"][sl]
    e_bess_to_load = series["e_bess_to_load"][sl]

    e_grid_to_load = series["e_grid_to_load"][sl]
    e_grid_to_base = series["e_grid_to_base"][sl]
    e_grid_to_boiler = series["e_grid_to_boiler"][sl]
    e_grid_to_bess = series["e_grid_to_bess"][sl]

    e_inj = series["e_inj"][sl]
    e_pv_to_bess = series["e_pv_to_bess"][sl]

    e_bess = series["e_bess"][sl]
    d_bess_ch = series["d_bess_ch"][sl]
    d_bess_dis = series["d_bess_dis"][sl]


    # --- teljesítmény [kW] ---
    p_load = _energy_to_power(e_load, dt)
    p_base_load = _energy_to_power(e_base_load, dt)
    p_boiler = _energy_to_power(e_boiler, dt)

    p_pv = _energy_to_power(e_pv, dt)
    p_bess_to_load = _energy_to_power(e_bess_to_load, dt)

    p_grid_to_load = _energy_to_power(e_grid_to_load, dt)
    p_grid_to_base = _energy_to_power(e_grid_to_base, dt)
    p_grid_to_boiler = _energy_to_power(e_grid_to_boiler, dt)
    p_grid_to_bess = _energy_to_power(e_grid_to_bess, dt)

    p_inj = _energy_to_power(e_inj, dt)
    p_pv_to_bess = _energy_to_power(e_pv_to_bess, dt)

    # időtengely: óra az év elejétől
    time_h = np.arange(t0, tf) * dt

    # oszlopszélesség órában
    bar_width = 0.8 * dt

    # Az adott háztartásnál akkor tekintjük aktívnak a dinamikus
    # bojlermodellt, ha van értelmezhető t_hss idősor.
    t_hss_full = series.get("t_hss")
    has_boiler_model = (
        t_hss_full is not None
        and len(t_hss_full) == T
        and np.any(np.isfinite(t_hss_full) & (t_hss_full > 0.0))
    )

    if has_boiler_model:
        fig, axes = plt.subplots(
            nrows=3,
            ncols=2,
            figsize=(22, 15),
            gridspec_kw={
                "width_ratios": [0.82, 0.18],
                "height_ratios": [0.50, 0.25, 0.25],
            },
            sharex="col",
        )
    else:
        fig, axes = plt.subplots(
            nrows=2,
            ncols=2,
            figsize=(22, 11),
            gridspec_kw={
                "width_ratios": [0.82, 0.18],
                "height_ratios": [0.5, 0.5],
            },
            sharex="col",
        )

    ax = axes[0, 0]
    ax_leg = axes[0, 1]
    ax_soc = axes[1, 0]
    ax_soc_leg = axes[1, 1]

    if has_boiler_model:
        ax_temp = axes[2, 0]
        ax_temp_leg = axes[2, 1]

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

    h_grid_base = ax.bar(
        time_h,
        p_grid_to_base,
        width=bar_width,
        bottom=pos_bottom,
        label="Hálózati import – A tarifa",
    )
    pos_bottom += p_grid_to_base

    h_grid_boiler = ax.bar(
        time_h,
        p_grid_to_boiler,
        width=bar_width,
        bottom=pos_bottom,
        label="Hálózati import – B tarifa / bojler",
    )
    pos_bottom += p_grid_to_boiler

    # ======================
    # 0 ALATT: kifolyó teljesítmények
    # ======================
    neg_bottom = np.zeros_like(time_h, dtype=float)

    h_base_load = ax.bar(
        time_h,
        -p_base_load,
        width=bar_width,
        bottom=neg_bottom,
        label="Általános fogyasztás",
    )
    neg_bottom -= p_base_load

    h_boiler = ax.bar(
        time_h,
        -p_boiler,
        width=bar_width,
        bottom=neg_bottom,
        label="Bojler fogyasztás",
    )
    neg_bottom -= p_boiler

    h_export = ax.bar(
        time_h,
        -p_inj,
        width=bar_width,
        bottom=neg_bottom,
        label="PV export",
    )
    neg_bottom -= p_inj

    h_bess_in_pv = ax.bar(
        time_h,
        -p_pv_to_bess,
        width=bar_width,
        bottom=neg_bottom,
        label="BESS töltés PV-ből",
    )
    neg_bottom -= p_pv_to_bess

    h_bess_in_grid = ax.bar(
        time_h,
        -p_grid_to_bess,
        width=bar_width,
        bottom=neg_bottom,
        label="BESS töltés hálózatból",
    )

    boiler_is_b_tariff = np.nanmax(e_grid_to_boiler) > 1e-9
    boiler_tariff_label = "B tarifás bojler" if boiler_is_b_tariff else "A tarifás bojler"

    # --- tengelyek ---
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_title(
        f"{season_title} – háztartási teljesítménymérleg\n"
        f"{household_name} – {boiler_tariff_label}"
    )
    ax.set_xlabel("Idő [h]")
    ax.set_ylabel("Teljesítmény [kW]")
    ax.grid(True, alpha=0.3)

    y_pos = p_pv + p_bess_to_load + p_grid_to_base + p_grid_to_boiler
    y_neg = p_base_load + p_boiler + p_inj + p_pv_to_bess + p_grid_to_bess
    ymax = max(float(np.nanmax(y_pos)), float(np.nanmax(y_neg)), 1e-6)
    ax.set_ylim(-1.15 * ymax, 1.15 * ymax)

    # --- legenda jobb oldalra ---
    handles = [
        h_pv,
        h_bess_out,
        h_grid_base,
        h_grid_boiler,
        h_base_load,
        h_boiler,
        h_export,
        h_bess_in_pv,
        h_bess_in_grid,
    ]

    labels = [
        "PV termelés",
        "BESS kisütés",
        "Hálózati import – A tarifa",
        "Hálózati import – B tarifa / bojler",
        "Általános fogyasztás",
        "Bojler fogyasztás",
        "PV export",
        "BESS töltés PV-ből",
        "BESS töltés hálózatból",
    ]

    ax_leg.legend(handles, labels, loc="center", frameon=False)
    ax_leg.axis("off")


    # Alsó
    h_soc, = ax_soc.plot(
        time_h,
        e_bess,
        linewidth=2.0,
        label="BESS SOC",
    )

    ax_ctrl = ax_soc.twinx()

    h_ch, = ax_ctrl.step(
        time_h,
        d_bess_ch,
        where="post",
        linewidth=1.5,
        color="gray",
        alpha=0.8,
        label="Töltés jel",
    )

    h_dis, = ax_ctrl.step(
        time_h,
        d_bess_dis,
        where="post",
        linewidth=1.5,
        color="lightgray",
        alpha=0.9,
        label="Kisütés jel",
    )

    ax_soc.set_ylabel("BESS energiaszint [kWh]")
    ax_soc.set_xlabel("Idő [h]")
    ax_soc.grid(True, alpha=0.3)

    ax_ctrl.set_ylabel("Vezérlőjel [-]")
    ax_ctrl.set_ylim(-0.05, 1.05)
    ax_ctrl.set_yticks([0, 1])

    ax_soc_leg.legend(
        [h_soc, h_ch, h_dis],
        ["BESS SOC [kWh]", "Töltés jel", "Kisütés jel"],
        loc="center",
        frameon=False,
    )
    ax_soc_leg.axis("off")

    # ======================
    # Harmadik panel: HSS-hőmérséklet
    # ======================
    if has_boiler_model:
        t_hss = np.asarray(t_hss_full[sl], dtype=float)

        h_temp, = ax_temp.plot(
            time_h,
            t_hss,
            linewidth=2.0,
            label="HSS hőmérséklet",
        )

        temp_handles = [h_temp]
        temp_labels = ["HSS hőmérséklet [°C]"]

        d_boiler_available_full = series.get("d_boiler_available")
        d_boiler_on_full = series.get("d_boiler_on")

        ax_temp_ctrl = None
        if (
            d_boiler_available_full is not None
            or d_boiler_on_full is not None
        ):
            ax_temp_ctrl = ax_temp.twinx()

        if d_boiler_available_full is not None:
            d_available = np.asarray(
                d_boiler_available_full[sl],
                dtype=float,
            )
            h_available, = ax_temp_ctrl.step(
                time_h,
                d_available,
                where="post",
                linewidth=1.3,
                alpha=0.65,
                label="Bojler engedélyezve",
            )
            temp_handles.append(h_available)
            temp_labels.append("Bojler engedélyezve")

        if d_boiler_on_full is not None:
            d_on = np.asarray(
                d_boiler_on_full[sl],
                dtype=float,
            )
            h_on, = ax_temp_ctrl.step(
                time_h,
                d_on,
                where="post",
                linewidth=1.5,
                alpha=0.85,
                label="Bojler bekapcsolva",
            )
            temp_handles.append(h_on)
            temp_labels.append("Bojler bekapcsolva")

        ax_temp.set_ylabel("Hőmérséklet [°C]")
        ax_temp.set_xlabel("Idő [h]")
        ax_temp.set_title("Bojlertartály hőmérsékletének alakulása")
        ax_temp.grid(True, alpha=0.3)

        finite_temp = t_hss[np.isfinite(t_hss) & (t_hss > 0.0)]
        if finite_temp.size:
            temp_min = float(np.min(finite_temp))
            temp_max = float(np.max(finite_temp))
            margin = max(2.0, 0.10 * max(temp_max - temp_min, 1.0))
            ax_temp.set_ylim(temp_min - margin, temp_max + margin)

        if ax_temp_ctrl is not None:
            ax_temp_ctrl.set_ylabel("Vezérlőjel [-]")
            ax_temp_ctrl.set_ylim(-0.05, 1.05)
            ax_temp_ctrl.set_yticks([0, 1])

        ax_temp_leg.legend(
            temp_handles,
            temp_labels,
            loc="center",
            frameon=False,
        )
        ax_temp_leg.axis("off")

    fig.tight_layout()

    suffix = "power_stack_bess_boiler" if has_boiler_model else "power_stack_bess"
    out_png = (
        results_dir
        / f"household_{household_name}_{season_key}_{WINDOW_DAYS}days_{suffix}.png"
    )
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
        results_dir=r"results_nonopt_individual_B_0.0%bess_boiler_model",
        household_name="0420144888377089",
        window_days=3,
        dt=DT,
    )

    # plot_household_power_stack_seasons(
    #     results_dir=r"results_base_with_bess_A_tariff",
    #     household_name="0420144653458813",
    #     window_days=3,
    #     dt=DT,
    # )

    # plot_household_power_stack_seasons(
    #     results_dir=r"results_base_no_bess_B_tariff",
    #     household_name="0420144888439778",
    #     window_days=3,
    #     dt=DT,
    # )

    # plot_household_power_stack_seasons(
    #     results_dir=r"results_base_no_bess_B_tariff",
    #     household_name="0420144653458813",
    #     window_days=3,
    #     dt=DT,
    # )