import sys
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dpi = 300
fontsize = 18

plt.rcParams.update({
    "font.size": fontsize - 2,
})


def _load_comm(out_dir: Path) -> pd.DataFrame:
    f = out_dir / "community_timeseries.csv"
    if not f.exists():
        raise FileNotFoundError(f"Hiányzik: {f}")
    return pd.read_csv(f)


def _load_mat(out_dir: Path, filename: str) -> np.ndarray | None:
    f = out_dir / filename
    if not f.exists():
        print(f"[WARN] Hiányzik: {f.name} — nullának veszem.")
        return None
    return pd.read_csv(f).to_numpy(dtype=float)


def _sum_mat(out_dir: Path, filename: str, T: int) -> np.ndarray:
    mat = _load_mat(out_dir, filename)
    if mat is None:
        return np.zeros(T)
    return mat.sum(axis=1)


def _safe(col: pd.DataFrame, name: str, T: int) -> np.ndarray:
    if name not in col.columns:
        print(f"[WARN] community_timeseries.csv nem tartalmazza: {name} — nullának veszem.")
        return np.zeros(T)
    return col[name].to_numpy(dtype=float)


def plot_community_node(
    out_dir: str | Path,
    window_len: int = 72,
):
    out_dir = Path(out_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    comm = _load_comm(out_dir)
    T = len(comm)

    # --- community idősorok ---
    p_grid_in = _safe(comm, "p_grid_in", T)
    p_grid_out = _safe(comm, "p_grid_out", T)
    p_bess_in_total = _safe(comm, "p_bess_in_total", T)

    # --- PV oldali bontás ---
    p_pv_load = _sum_mat(out_dir, "p_pv_load.csv", T)
    p_pv_bess = _sum_mat(out_dir, "p_pv_bess.csv", T)
    p_pv_elh = _sum_mat(out_dir, "p_pv_elh.csv", T)
    p_pv_rec = _sum_mat(out_dir, "p_pv_rec.csv", T)
    p_pv_grid = _sum_mat(out_dir, "p_pv_grid.csv", T)

    # Össztermelt PV, mivel p_pv.csv nincs kimentve:
    p_pv_total = p_pv_load + p_pv_bess + p_pv_elh + p_pv_rec + p_pv_grid

    # --- fogyasztási oldali bontás ---
    p_rec_load = _sum_mat(out_dir, "p_rec_load.csv", T)
    p_grid_load = _sum_mat(out_dir, "p_grid_load.csv", T)

    p_rec_elh = _sum_mat(out_dir, "p_rec_elh.csv", T)
    p_grid_elh = _sum_mat(out_dir, "p_grid_elh.csv", T)

    p_grid_bess = _sum_mat(out_dir, "p_grid_bess.csv", T)

    # Saját PV-ből töltött BESS külön hasznos, mert nem külső féltől/hálózatról jön
    p_pv_bess = _sum_mat(out_dir, "p_pv_bess.csv", T)

    # --- évszakok ---
    starts = [0, 2184, 4368, 6552]
    season_names_hu = ["Tél", "Tavasz", "Nyár", "Ősz"]
    season_names_en = ["winter", "spring", "summer", "autumn"]

    bar_kw = dict(width=0.85)

    for t0, title_hu, title_en in zip(starts, season_names_hu, season_names_en):
        tf = min(t0 + window_len, T)
        time = np.arange(t0, tf)

        fig, axes = plt.subplots(
            nrows=1,
            ncols=2,
            figsize=(22, 8),
            gridspec_kw={"width_ratios": [0.78, 0.22]},
        )

        ax = axes[0]
        legend_ax = axes[1]

        # =========================
        # BELÉPŐ ENERGIÁK / FELSŐ OLDAL
        # =========================
        bottom_pos = np.zeros_like(time, dtype=float)

        ax.bar(
            time,
            p_pv_total[t0:tf],
            bottom=bottom_pos,
            label="Össztermelt PV",
            **bar_kw,
        )
        bottom_pos += p_pv_total[t0:tf]

        ax.bar(
            time,
            p_grid_in[t0:tf],
            bottom=bottom_pos,
            label="Külső hálózati import a közösségi csomópontra",
            **bar_kw,
        )
        bottom_pos += p_grid_in[t0:tf]

        # =========================
        # KILÉPŐ ENERGIÁK / ALSÓ OLDAL
        # =========================
        bottom_neg = np.zeros_like(time, dtype=float)

        ax.bar(
            time,
            -p_pv_grid[t0:tf],
            bottom=bottom_neg,
            label="PV export külső hálózatba",
            **bar_kw,
        )
        bottom_neg -= p_pv_grid[t0:tf]

        ax.bar(
            time,
            -p_rec_load[t0:tf],
            bottom=bottom_neg,
            label="Általános fogyasztás megosztott energiából",
            **bar_kw,
        )
        bottom_neg -= p_rec_load[t0:tf]

        ax.bar(
            time,
            -p_grid_load[t0:tf],
            bottom=bottom_neg,
            label="Általános fogyasztás külső hálózatból",
            **bar_kw,
        )
        bottom_neg -= p_grid_load[t0:tf]

        ax.bar(
            time,
            -p_pv_bess[t0:tf],
            bottom=bottom_neg,
            label="BESS töltés saját PV-ből",
            **bar_kw,
        )
        bottom_neg -= p_pv_bess[t0:tf]

        ax.bar(
            time,
            -p_grid_bess[t0:tf],
            bottom=bottom_neg,
            label="BESS töltés külső hálózatból",
            **bar_kw,
        )
        bottom_neg -= p_grid_bess[t0:tf]

        ax.bar(
            time,
            -p_rec_elh[t0:tf],
            bottom=bottom_neg,
            label="Bojler felvétele megosztott energiából",
            **bar_kw,
        )
        bottom_neg -= p_rec_elh[t0:tf]

        ax.bar(
            time,
            -p_grid_elh[t0:tf],
            bottom=bottom_neg,
            label="Bojler felvétele külső hálózatból",
            **bar_kw,
        )
        bottom_neg -= p_grid_elh[t0:tf]

        # opcionális: ha a community_timeseries-ben van p_grid_out, ezt is rárajzoljuk
        # Ez a közösségi mérleg szerinti export, nem feltétlen azonos a p_pv_grid-del,
        # ezért külön, kontúrvonalként jelenik meg.
        ax.plot(
            time,
            -p_grid_out[t0:tf],
            linestyle="--",
            linewidth=2,
            label="Közösségi mérleg szerinti grid export",
        )

        ax.axhline(0, linewidth=1)
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{title_hu} – közösségi villamos csomópont", fontsize=fontsize)
        ax.set_xlabel("Idő [h]", fontsize=fontsize)
        ax.set_ylabel("Energia [kWh / óra]", fontsize=fontsize)

        handles, labels = ax.get_legend_handles_labels()
        legend_ax.legend(handles, labels, loc="center", fontsize=fontsize - 3)
        legend_ax.axis("off")

        fig.tight_layout()
        fig.savefig(fig_dir / f"community_node_{title_en}.png", dpi=dpi)
        plt.close(fig)

    print(f"[OK] Ábrák mentve ide: {fig_dir}")


def main():
    # PyCharm Run gombos futtatás / dupla kattintás esetén
    if len(sys.argv) == 1:
        plot_community_node(
            out_dir=r".\results_disaggregated_hourly",
            window_len=72,   # 48 = 2 nap, 72 = 3 nap
        )
        return

    # Terminálos futtatás esetén továbbra is működik
    ap = argparse.ArgumentParser(description="Disaggregált közösségi csomópont plot")
    ap.add_argument(
        "--out",
        default=r".\results_disaggregated_hourly",
        help="Eredménymappa, ahol a CSV-k vannak.",
    )
    ap.add_argument(
        "--window-len",
        type=int,
        default=72,
        help="Ablakhossz órában. 48 = 2 nap, 72 = 3 nap.",
    )
    args = ap.parse_args()

    plot_community_node(
        out_dir=r".\results_disaggregated_hourly",
        window_len = 72,
    )


if __name__ == "__main__":
    main()