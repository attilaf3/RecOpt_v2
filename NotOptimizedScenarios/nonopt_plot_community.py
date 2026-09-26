"""Közösségi nem optimalizált eredmények szezonális ábrázolása."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SEASONS = (
    ("winter", "Tél", 0),
    ("spring", "Tavasz", 2184),
    ("summer", "Nyár", 4368),
    ("autumn", "Ősz", 6552),
)


def _column(frame: pd.DataFrame, name: str, *, optional: bool = False) -> np.ndarray:
    column = f"{name}_total"
    if column in frame:
        return frame[column].to_numpy(dtype=float)
    if optional:
        return np.zeros(len(frame))
    raise KeyError(f"Hiányzó oszlop a community_timeseries.csv fájlban: {column}")


def plot_community_case(
    results_dir: str | Path,
    *,
    case_name: str,
    with_bess: bool,
    window_days: int = 3,
    dt: float = 0.25,
    dpi: int = 300,
) -> list[Path]:
    """Négy szezonális közösségi teljesítménymérleget készít."""
    results_dir = Path(results_dir)
    csv_path = results_dir / "community_timeseries.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Hiányzó eredményfájl: {csv_path}")
    frame = pd.read_csv(csv_path)
    required = [
        "e_base", "e_boiler", "e_pv", "e_shared_in", "e_shared_out",
        "e_grid_import_a", "e_grid_import_b", "e_grid_export",
    ]
    series = {name: _column(frame, name) for name in required}
    for name in ("e_bess_to_load", "e_pv_to_bess", "e_grid_to_bess"):
        series[name] = _column(frame, name, optional=not with_bess)

    steps = int(round(window_days * 24 / dt))
    saved: list[Path] = []
    for season_key, season_title, start_hour in SEASONS:
        t0 = int(round(start_hour / dt))
        tf = min(t0 + steps, len(frame))
        if tf <= t0 + 1:
            print(f"[WARN] {season_title}: nincs elég adat.")
            continue
        sl = slice(t0, tf)
        time = np.arange(t0, tf) * dt
        width = 0.8 * dt
        p = {name: values[sl] / dt for name, values in series.items()}

        fig, axes = plt.subplots(
            2, 2, figsize=(24, 13),
            gridspec_kw={"width_ratios": [0.82, 0.18], "height_ratios": [0.55, 0.45]},
        )
        ax, legend_ax = axes[0]
        positive = [
            ("e_pv", "PV termelés"),
            ("e_bess_to_load", "BESS kisütés"),
            ("e_shared_in", "Közösségből kapott energia"),
            ("e_grid_import_a", "Hálózati import – A tarifa"),
            ("e_grid_import_b", "Hálózati import – B tarifa"),
        ]
        negative = [
            ("e_base", "Általános fogyasztás"),
            ("e_boiler", "Bojler fogyasztás"),
            ("e_shared_out", "Közösségnek átadott energia"),
            ("e_grid_export", "Hálózati export"),
            ("e_pv_to_bess", "BESS töltés PV-ből"),
            ("e_grid_to_bess", "BESS töltés hálózatból"),
        ]
        handles = []
        pos_bottom = np.zeros(tf - t0)
        for name, label in positive:
            if not with_bess and "bess" in name:
                continue
            handle = ax.bar(time, p[name], width=width, bottom=pos_bottom, label=label)
            pos_bottom += p[name]
            handles.append(handle)
        neg_bottom = np.zeros(tf - t0)
        for name, label in negative:
            if not with_bess and "bess" in name:
                continue
            handle = ax.bar(time, -p[name], width=width, bottom=neg_bottom, label=label)
            neg_bottom -= p[name]
            handles.append(handle)
        ax.axhline(0, color="black", linewidth=1)
        ax.set_title(f"{case_name}: {season_title} – közösségi teljesítménymérleg")
        ax.set_ylabel("Teljesítmény [kW]")
        ax.grid(True, alpha=0.3)
        legend_ax.legend(handles, [h.get_label() for h in handles], loc="center", frameon=False)
        legend_ax.axis("off")

        ax_exchange, exchange_legend = axes[1]
        p_inj = p["e_grid_export"]
        p_with = p["e_grid_import_a"] + p["e_grid_import_b"]
        p_shared = np.minimum(p["e_shared_in"], p["e_shared_out"])
        h_inj, = ax_exchange.plot(time, p_inj, color="tab:red", linewidth=2, label=r"$P_\mathrm{inj}$")
        h_with, = ax_exchange.plot(time, p_with, color="tab:blue", linewidth=2, label=r"$P_\mathrm{with}$")
        h_shared, = ax_exchange.plot(time, p_shared, color="tab:green", linewidth=2, label=r"$P_\mathrm{shared}$")
        ax_exchange.fill_between(time, p_shared, p_with, where=p_with > p_shared, color="tab:blue", alpha=0.3)
        ax_exchange.fill_between(time, p_shared, p_inj, where=p_inj > p_shared, color="tab:red", alpha=0.3)
        ax_exchange.fill_between(time, 0, p_shared, where=p_shared > 0, color="tab:green", alpha=0.3)
        ax_exchange.set_title("Közösségi megosztás és külső hálózati energiacsere")
        ax_exchange.set_xlabel("Idő [h]")
        ax_exchange.set_ylabel("Teljesítmény [kW]")
        ax_exchange.grid(True, alpha=0.3)
        exchange_legend.legend([h_inj, h_with, h_shared], [h.get_label() for h in (h_inj, h_with, h_shared)], loc="center", frameon=False)
        exchange_legend.axis("off")

        fig.tight_layout()
        path = results_dir / f"plot_{case_name}_{season_key}_{window_days}days.png"
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        saved.append(path)
        print(f"Mentve: {path}")
    return saved
