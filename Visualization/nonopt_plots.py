from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

__all__ = [
    "plot_community_energy_balance",
    "plot_household_percentiles_by_group",
    "plot_household_percentiles_by_group_with_global_scurve",
]


GROUP_ORDER = [
    "Nincs PV", "PV", "HP", "PV+HP", "PV+BESS", "PV+BESS+HP",
    "BESS+HP", "BESS+HP+Boiler", "BESS+Boiler", "HP+Boiler",
    "PV+BESS+HP+Boiler", "PV+BESS+Boiler", "PV+HP+Boiler",
]
GROUP_MARKERS = {
    "Nincs PV": "o", "PV": "s", "HP": "D", "PV+HP": "P",
    "PV+BESS": "^", "PV+BESS+HP": "X", "BESS+HP": "H",
    "BESS+HP+Boiler": "*", "BESS+Boiler": "v", "HP+Boiler": "d",
    "PV+BESS+HP+Boiler": "*", "PV+BESS+Boiler": "v", "PV+HP+Boiler": "d",
}


def _ordered_groups(df: pd.DataFrame) -> list[str]:
    groups = list(dict.fromkeys(df["group_label"].astype(str).tolist()))
    return [g for g in GROUP_ORDER if g in groups] + [g for g in groups if g not in GROUP_ORDER]


def _column(frame: pd.DataFrame, *names: str) -> np.ndarray:
    for name in names:
        if name in frame.columns:
            return frame[name].to_numpy(dtype=float)
    return np.zeros(len(frame), dtype=float)


def _datetime_index(frame: pd.DataFrame) -> pd.DatetimeIndex:
    if isinstance(frame.index, pd.DatetimeIndex):
        return pd.DatetimeIndex(frame.index)
    return pd.date_range("2021-01-01", periods=len(frame), freq="h")


def _season_windows(index: pd.DatetimeIndex, length: int) -> list[tuple[str, slice]]:
    if len(index) == 0:
        return []
    length = max(1, min(int(length), len(index)))
    centers = [
        ("Tél", pd.Timestamp(index[0].year, 1, 15, 12)),
        ("Tavasz", pd.Timestamp(index[0].year, 4, 15, 12)),
        ("Nyár", pd.Timestamp(index[0].year, 7, 15, 12)),
        ("Ősz", pd.Timestamp(index[0].year, 10, 15, 12)),
    ]
    windows = []
    for label, center in centers:
        pos = int(index.get_indexer(pd.DatetimeIndex([center]), method="nearest")[0])
        start = max(0, min(pos - length // 2, len(index) - length))
        windows.append((label, slice(start, start + length)))
    return windows


def _stack_signed(ax, x: np.ndarray, components: Sequence[tuple[str, np.ndarray, str]]) -> None:
    positive = np.zeros_like(x, dtype=float)
    negative = np.zeros_like(x, dtype=float)
    for label, values, color in components:
        arr = np.asarray(values, dtype=float)
        pos = np.clip(arr, 0.0, None)
        neg = np.clip(arr, None, 0.0)
        if np.any(pos):
            ax.bar(x, pos, bottom=positive, color=color, width=0.86, label=label)
            positive += pos
        if np.any(neg):
            ax.bar(x, neg, bottom=negative, color=color, width=0.86,
                   label=label if not np.any(pos) else None)
            negative += neg
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    ax.grid(True, axis="y", alpha=0.25)


def _select_households(per_user_df: pd.DataFrame | None, limit: int = 3) -> list[int]:
    if per_user_df is None or per_user_df.empty:
        return []
    selected: list[int] = []
    for group in reversed(GROUP_ORDER):
        matches = per_user_df.index[per_user_df["group_label"] == group].tolist()
        if matches and int(matches[0]) not in selected:
            selected.append(int(matches[0]))
        if len(selected) == limit:
            return selected
    for idx in per_user_df.index:
        if int(idx) not in selected:
            selected.append(int(idx))
        if len(selected) == limit:
            break
    return selected


def plot_community_energy_balance(
    community_ts: pd.DataFrame,
    out_case: str | Path,
    per_user_df: pd.DataFrame | None = None,
    household_ts: Mapping[str, np.ndarray] | None = None,
    user_names: Sequence[str] | None = None,
) -> None:
    """Save seasonal community and sample-household energy-balance plots."""
    if community_ts.empty or not {"e_pv_total", "e_load_total"}.issubset(community_ts.columns):
        return

    out_case = Path(out_case)
    out_case.mkdir(parents=True, exist_ok=True)
    index = _datetime_index(community_ts)
    windows = _season_windows(index, min(96, len(index)))
    has_bess = bool(
        _column(community_ts, "e_bess_to_load_total").sum()
        or _column(community_ts, "e_bess_total").sum()
    )
    ncols = 3 if has_bess else 2
    fig, axes = plt.subplots(len(windows), ncols, figsize=(7.2 * ncols, 4.3 * len(windows)), squeeze=False)
    fig.suptitle("Közösségi energiamérleg szezononkénti bontásban", fontsize=16)
    selected = _select_households(per_user_df)

    for row, (label, window) in enumerate(windows):
        x = np.arange(window.stop - window.start, dtype=float)
        components = [
            ("UE / alap", _column(community_ts, "e_ue_total", "e_load_base_total")[window], "#7f8c8d"),
            ("Boiler", _column(community_ts, "e_boiler_total")[window], "#a65628"),
            ("HP", _column(community_ts, "e_hp_total")[window], "#3d5afe"),
            ("BESS töltés", _column(community_ts, "e_pv_to_bess_total")[window], "#37cdd2"),
            ("Hálózati export", _column(community_ts, "e_inj_total")[window], "#d65a5a"),
            ("PV", -_column(community_ts, "e_pv_total")[window], "#f4c20d"),
            ("Hálózati import", -_column(community_ts, "e_grid_to_load_total", "e_grid_import_total")[window], "#4c78a8"),
            ("BESS kisütés", -_column(community_ts, "e_bess_to_load_total")[window], "#2f8f4e"),
        ]
        _stack_signed(axes[row, 0], x, components)
        axes[row, 0].set_title(f"{label} – közösségi mérleg")
        axes[row, 0].set_ylabel("kWh / időlépés")

        household_col = 1
        if has_bess:
            storage = [
                ("Töltés", _column(community_ts, "e_pv_to_bess_total")[window], "#ff8c00"),
                ("Kisütés", -_column(community_ts, "e_bess_to_load_total")[window], "#5b8cc0"),
            ]
            _stack_signed(axes[row, 1], x, storage)
            axes[row, 1].set_title(f"{label} – villamos tároló")
            household_col = 2

        ax = axes[row, household_col]
        if household_ts and selected:
            grid = np.asarray(household_ts.get("e_grid_to_load", []), dtype=float)
            export = np.asarray(household_ts.get("e_inj", np.zeros_like(grid)), dtype=float)
            if grid.ndim == 2 and export.shape == grid.shape:
                for position, user in enumerate(selected):
                    if user < grid.shape[1]:
                        name = str(user_names[user]) if user_names and user < len(user_names) else f"H{user + 1}"
                        ax.plot(x, grid[window, user] - export[window, user], label=name, linewidth=1.1)
                ax.axhline(0.0, color="black", linewidth=0.8)
                ax.grid(True, alpha=0.25)
                ax.legend()
                ax.set_title(f"{label} – kiválasztott háztartások")
            else:
                ax.set_axis_off()
        else:
            ax.text(0.5, 0.5, "Nincs háztartási idősor", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()

        for axis in axes[row]:
            handles, labels = axis.get_legend_handles_labels()
            if handles and axis.get_legend() is None:
                axis.legend(handles, labels, fontsize=8)
            axis.set_xlabel("Időlépés az ablakban")

    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_case / "community_energy_balance.png", dpi=200)
    plt.close(fig)


def _prepare_percentiles(per_user_df: pd.DataFrame) -> pd.DataFrame:
    required = {"brt_bill_ft", "group_label"}
    if not required.issubset(per_user_df.columns):
        missing = ", ".join(sorted(required - set(per_user_df.columns)))
        raise ValueError(f"Hiányzó percentilis-ábra oszlopok: {missing}")
    df = per_user_df.copy().sort_values("brt_bill_ft").reset_index(drop=True)
    if not df.empty:
        df["percentile"] = 100.0 * (np.arange(len(df)) + 0.5) / len(df)
    return df


def _scatter_percentiles(df: pd.DataFrame) -> None:
    for group in _ordered_groups(df):
        subset = df[df["group_label"] == group]
        plt.scatter(subset["brt_bill_ft"], subset["percentile"], s=28,
                    marker=GROUP_MARKERS.get(group, "o"), alpha=0.8, label=group)


def _finish_percentile_plot(df: pd.DataFrame, path: Path) -> None:
    x = df["brt_bill_ft"].to_numpy(dtype=float)
    if np.ptp(x) > 0:
        plt.xlim(np.percentile(x, 1), np.percentile(x, 99.9))
    plt.xlabel("Éves bruttó villanyszámla [Ft/év]")
    plt.ylabel("Háztartások aránya [%]")
    plt.title("Háztartások villanyszámla szerinti eloszlása")
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_household_percentiles_by_group(per_user_df: pd.DataFrame, out_case: str | Path) -> None:
    df = _prepare_percentiles(per_user_df)
    if df.empty:
        return
    out_case = Path(out_case)
    out_case.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 5.5))
    _scatter_percentiles(df)
    _finish_percentile_plot(df, out_case / "household_bill_percentiles_by_group.png")


def plot_household_percentiles_by_group_with_global_scurve(
    per_user_df: pd.DataFrame,
    out_case: str | Path,
) -> None:
    from scipy.optimize import curve_fit

    df = _prepare_percentiles(per_user_df)
    if df.empty:
        return
    out_case = Path(out_case)
    out_case.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 5.5))
    _scatter_percentiles(df)

    x = df["brt_bill_ft"].to_numpy(dtype=float)
    y = df["percentile"].to_numpy(dtype=float)

    def logistic(values, x0, k):
        return 100.0 / (1.0 + np.exp(-k * (values - x0)))

    if len(df) >= 4 and np.ptp(x) > 0:
        try:
            parameters = curve_fit(logistic, x, y, p0=[np.median(x), 1.0 / max(np.std(x), 1.0)], maxfev=20000)[0]
            x_grid = np.linspace(x.min(), x.max(), 400)
            plt.plot(x_grid, logistic(x_grid, *parameters), color="darkred", linewidth=2,
                     label="Illesztett S-görbe (összes háztartás)")
        except (RuntimeError, ValueError, FloatingPointError):
            pass

    _finish_percentile_plot(df, out_case / "household_bill_percentiles_by_group_global_scurve.png")
