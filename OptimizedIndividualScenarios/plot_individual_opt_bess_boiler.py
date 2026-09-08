"""A 3d-I eredmény egy kiválasztott háztartásának szezonális ábrái."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _load(results_dir: Path, household: str) -> tuple[pd.DataFrame, Path]:
    exact = results_dir / f"timeseries_{household}.csv"
    matches = [exact] if exact.exists() else sorted(results_dir.glob(f"timeseries_*{household}*.csv"))
    if len(matches) != 1:
        raise FileNotFoundError(f"A timeseries nem található egyértelműen ({len(matches)} találat).")
    return pd.read_csv(matches[0]), matches[0]


def plot_household(results_dir, household, *, window_days=3, dt=0.25, dpi=200):
    directory = Path(results_dir)
    data, source = _load(directory, household)
    required = {
        "p_pv_kw", "p_base_load_kw", "p_fixed_boiler_kw", "p_grid_import_kw",
        "p_pv_export_kw", "p_bess_charge_kw", "p_bess_discharge_kw",
        "e_bess_kwh", "d_bess_charge", "d_bess_discharge", "p_elh_kw",
        "t_hss_C", "d_cl",
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Hiányzó oszlopok: {', '.join(missing)}")

    starts = [0, int(2184 / dt), int(4368 / dt), int(6552 / dt)]
    seasons = [("winter", "Tél"), ("spring", "Tavasz"), ("summer", "Nyár"), ("autumn", "Ősz")]
    length = int(round(window_days * 24 / dt))
    saved = []
    for (key, title), start in zip(seasons, starts):
        stop = min(start + length, len(data))
        if stop - start < 2:
            continue
        frame = data.iloc[start:stop]
        time = np.arange(stop - start) * dt
        boiler = frame["p_elh_kw"].to_numpy() + frame["p_fixed_boiler_kw"].to_numpy()

        fig, axes = plt.subplots(3, 1, figsize=(16, 13), sharex=True)
        ax = axes[0]
        incoming = [frame.p_pv_kw, frame.p_grid_import_kw, frame.p_bess_discharge_kw]
        outgoing = [frame.p_base_load_kw, boiler, frame.p_bess_charge_kw, frame.p_pv_export_kw]
        bottom = np.zeros(len(frame))
        for values, label in zip(incoming, ["PV", "Hálózati import", "BESS kisütés"]):
            ax.bar(time, values, bottom=bottom, width=0.8 * dt, label=label)
            bottom += np.asarray(values)
        bottom = np.zeros(len(frame))
        for values, label in zip(outgoing, ["Alapfogyasztás", "Bojler", "BESS töltés", "PV-export"]):
            ax.bar(time, -np.asarray(values), bottom=bottom, width=0.8 * dt, label=label)
            bottom -= np.asarray(values)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel("kW")
        ax.set_title(f"{title} – villamos energiamérleg")
        ax.legend(ncol=4, fontsize=9)

        axes[1].plot(time, frame.e_bess_kwh, label="BESS energia (kWh)")
        mode = axes[1].twinx()
        mode.step(time, frame.d_bess_charge, where="post", label="töltés", color="tab:green", alpha=0.7)
        mode.step(time, -frame.d_bess_discharge, where="post", label="kisütés", color="tab:red", alpha=0.7)
        axes[1].set_ylabel("kWh")
        axes[1].set_title("BESS állapot és üzemmód")
        axes[1].legend(loc="upper left")
        mode.legend(loc="upper right")

        axes[2].plot(time, frame.t_hss_C, label="HSS hőmérséklet")
        control = axes[2].twinx()
        control.step(time, frame.d_cl, where="post", color="gray", label="engedélyezőjel")
        axes[2].set_ylabel("°C")
        axes[2].set_xlabel("Ablak kezdete óta eltelt idő (h)")
        axes[2].set_title("Bojler hőállapot és vezérlés")
        axes[2].legend(loc="upper left")
        control.legend(loc="upper right")
        for base in axes:
            base.grid(alpha=0.25)
        fig.suptitle(f"3d-I – {household}")
        fig.tight_layout()
        target = directory / f"plot_3d_I_{household}_{key}_{window_days}days.png"
        fig.savefig(target, dpi=dpi)
        plt.close(fig)
        saved.append(target)
    print(f"Forrás: {source}")
    return saved


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results_3d_I")
    parser.add_argument("--household", required=True)
    parser.add_argument("--window-days", type=int, default=3)
    parser.add_argument("--dt", type=float, default=0.25)
    args = parser.parse_args(argv)
    for path in plot_household(args.results, args.household, window_days=args.window_days, dt=args.dt):
        print(f"Mentve: {path}")


if __name__ == "__main__":
    main()
