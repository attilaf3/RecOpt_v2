"""Egyéni nem optimalizált eredmények szezonális ábrázolása."""

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


def _load(results_dir: Path, name: str, household: str) -> np.ndarray:
    path = results_dir / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Hiányzó eredményfájl: {path}")
    frame = pd.read_csv(path)
    if household not in frame.columns:
        raise KeyError(f"A(z) {household!r} háztartás nem szerepel ebben: {path.name}")
    return frame[household].to_numpy(dtype=float)


def plot_individual_case(
    results_dir: str | Path,
    household: str,
    *,
    case_name: str,
    boiler_tariff: str,
    with_bess: bool,
    with_boiler_model: bool,
    window_days: int = 3,
    dt: float = 0.25,
    dpi: int = 300,
) -> list[Path]:
    """Négy szezonális ábrát készít egy kiválasztott háztartásról."""
    results_dir = Path(results_dir)
    required = [
        "e_base", "e_boiler", "e_pv", "e_grid_import_a",
        "e_grid_import_b", "e_grid_export",
    ]
    if with_bess:
        required += [
            "e_bess_to_load", "e_pv_to_bess", "e_grid_to_bess",
            "e_bess", "d_bess_ch", "d_bess_dis",
        ]
    if with_boiler_model:
        required += ["t_hss", "t_setpoint", "d_enable", "d_on"]
    series = {name: _load(results_dir, name, household) for name in required}

    steps = int(round(window_days * 24 / dt))
    saved: list[Path] = []
    for season_key, season_title, start_hour in SEASONS:
        t0 = int(round(start_hour / dt))
        tf = min(t0 + steps, len(series["e_base"]))
        if tf <= t0 + 1:
            print(f"[WARN] {season_title}: nincs elég adat.")
            continue
        sl = slice(t0, tf)
        time = np.arange(t0, tf) * dt
        width = 0.8 * dt
        power = {name: values[sl] / dt for name, values in series.items() if name.startswith("e_")}

        rows = 1 + int(with_bess) + int(with_boiler_model)
        heights = [0.55] + ([0.225] if with_bess else []) + ([0.225] if with_boiler_model else [])
        fig, axes = plt.subplots(
            rows, 2, figsize=(22, 6 + 4 * rows), squeeze=False,
            gridspec_kw={"width_ratios": [0.82, 0.18], "height_ratios": heights},
        )

        ax, legend_ax = axes[0]
        positive = [
            ("e_pv", "PV termelés"),
            ("e_bess_to_load", "BESS kisütés"),
            ("e_grid_import_a", "Hálózati import – A tarifa"),
            ("e_grid_import_b", "Hálózati import – B tarifa"),
        ]
        negative = [
            ("e_base", "Általános fogyasztás"),
            ("e_boiler", "Bojler fogyasztás"),
            ("e_grid_export", "Hálózati export"),
            ("e_pv_to_bess", "BESS töltés PV-ből"),
            ("e_grid_to_bess", "BESS töltés hálózatból"),
        ]
        handles = []
        pos_bottom = np.zeros(tf - t0)
        for name, label in positive:
            if name not in power:
                continue
            handle = ax.bar(time, power[name], width=width, bottom=pos_bottom, label=label)
            pos_bottom += power[name]
            handles.append(handle)
        neg_bottom = np.zeros(tf - t0)
        for name, label in negative:
            if name not in power:
                continue
            handle = ax.bar(time, -power[name], width=width, bottom=neg_bottom, label=label)
            neg_bottom -= power[name]
            handles.append(handle)
        ax.axhline(0, color="black", linewidth=1)
        ax.set_title(f"{case_name}: {season_title} – {household} – bojler {boiler_tariff} tarifa")
        ax.set_ylabel("Teljesítmény [kW]")
        ax.grid(True, alpha=0.3)
        legend_ax.legend(handles, [h.get_label() for h in handles], loc="center", frameon=False)
        legend_ax.axis("off")

        row = 1
        if with_bess:
            ax_bess, bess_legend = axes[row]
            h_soc, = ax_bess.plot(time, series["e_bess"][sl], linewidth=2, label="BESS energiaszint")
            ax_ctrl = ax_bess.twinx()
            h_ch, = ax_ctrl.step(time, series["d_bess_ch"][sl], where="post", label="Töltés jel")
            h_dis, = ax_ctrl.step(time, series["d_bess_dis"][sl], where="post", label="Kisütés jel")
            ax_bess.set_ylabel("BESS energiaszint [kWh]")
            ax_ctrl.set_ylim(-0.05, 1.05)
            ax_ctrl.set_yticks([0, 1])
            ax_bess.grid(True, alpha=0.3)
            bess_legend.legend([h_soc, h_ch, h_dis], [h.get_label() for h in (h_soc, h_ch, h_dis)], loc="center", frameon=False)
            bess_legend.axis("off")
            row += 1

        if with_boiler_model:
            ax_hss, hss_legend = axes[row]
            h_temp, = ax_hss.plot(
                time, series["t_hss"][sl], linewidth=2,
                label="HSS ekvivalens átlaghőmérséklet",
            )
            h_set, = ax_hss.plot(
                time, series["t_setpoint"][sl], "--", linewidth=1.5,
                label="YAML setpoint",
            )
            ax_ctrl = ax_hss.twinx()
            h_enable, = ax_ctrl.step(time, series["d_enable"][sl], where="post", label="Engedélyezőjel")
            h_on, = ax_ctrl.step(time, series["d_on"][sl], where="post", label="Bojler bekapcsolva")
            ax_hss.set_ylabel("Hőmérséklet [°C]")
            ax_ctrl.set_ylim(-0.05, 1.05)
            ax_ctrl.set_yticks([0, 1])
            ax_hss.grid(True, alpha=0.3)
            hss_legend.legend(
                [h_temp, h_set, h_enable, h_on],
                [h.get_label() for h in (h_temp, h_set, h_enable, h_on)],
                loc="center", frameon=False,
            )
            hss_legend.axis("off")

        axes[-1, 0].set_xlabel("Idő [h]")
        fig.tight_layout()
        path = results_dir / f"plot_{case_name}_{household}_{season_key}_{window_days}days.png"
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        saved.append(path)
        print(f"Mentve: {path}")
    return saved
