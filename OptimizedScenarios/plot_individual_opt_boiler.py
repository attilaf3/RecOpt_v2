from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def plot_household_percentiles_by_group_with_global_scurve(per_user_df: pd.DataFrame, out_case: Path):
    df = per_user_df.copy()

    if "net_bill_ft" not in df.columns:
        raise ValueError("Hiányzik a 'net_bill_ft' oszlop.")

    if "group_label" not in df.columns:
        raise ValueError("Hiányzik a 'group_label' oszlop.")

    df = df.sort_values("net_bill_ft").reset_index(drop=True)
    n = len(df)

    if n == 0:
        return

    df["percentile"] = 100.0 * (np.arange(n) + 0.5) / n

    marker_map = {
        "Nincs PV": "o",
        "PV": "s",
        "PV+BESS": "^",
    }

    plt.figure(figsize=(9, 5.5))

    # pontok csoportonként
    for group in ["Nincs PV", "PV", "PV+BESS"]:
        sub = df[df["group_label"] == group].copy()
        if len(sub) == 0:
            continue

        plt.scatter(
            sub["net_bill_ft"],
            sub["percentile"],
            s=28,
            marker=marker_map.get(group, "o"),
            alpha=0.8,
            label=group,
        )

    # egyetlen globális S-görbe az összes háztartásra
    x = df["net_bill_ft"].to_numpy(dtype=float)
    y = df["percentile"].to_numpy(dtype=float)

    def logistic(x_, x0, k):
        return 100.0 / (1.0 + np.exp(-k * (x_ - x0)))

    if len(df) >= 4 and np.nanstd(x) > 0:
        x0_init = np.nanmedian(x)
        spread = max(np.nanstd(x), 1.0)
        k_init = 1.0 / spread

        try:
            popt, _ = curve_fit(logistic, x, y, p0=[x0_init, k_init], maxfev=20000)
            x_grid = np.linspace(np.nanmin(x), np.nanmax(x), 400)
            y_fit = logistic(x_grid, *popt)
            plt.plot(
                x_grid,
                y_fit,
                color="darkred",
                linewidth=2,
                label="Illesztett S-görbe (összes háztartás)",
            )
        except Exception:
            pass

    x_lo = np.percentile(df["net_bill_ft"], 0)
    x_hi = np.percentile(df["net_bill_ft"], 100)
    if x_lo == x_hi:
        x_lo = x_lo - 1
        x_hi = x_hi + 1

    plt.xlim(x_lo, x_hi)
    plt.xlabel("Éves nettó villanyszámla [Ft/év]")
    plt.ylabel("Háztartások aránya [%]")
    plt.title("Háztartások villanyszámla szerinti eloszlása")
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_case / "household_bill_percentiles_by_group_global_scurve.png", dpi=200)
    plt.close()


def compute_community_sci_ssi(df: pd.DataFrame) -> dict:
    """
    Közösségi szintű SCI és SSI számítás megosztott energia nélkül.

    SCI = helyben felhasznált PV / összes PV termelés
        = (PV termelés - hálózati export) / PV termelés

    SSI = helyben fedezett villamos energiaigény / összes villamos energiaigény
        = (összes villamos igény - hálózati import) / összes villamos igény

    A bojleres esetben az összes villamos igény:
        általános fogyasztás + bojler villamos input
    """
    required_cols = [
        "pv_gen_kwh",
        "grid_export_kwh",
        "grid_import_total_kwh",
        "load_kwh",
        "boiler_el_input_kwh",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Hiányzó oszlopok az SCI/SSI számításhoz: {missing}")

    total_pv_gen = float(df["pv_gen_kwh"].sum())
    total_grid_export = float(df["grid_export_kwh"].sum())
    total_grid_import = float(df["grid_import_total_kwh"].sum())
    total_load = float(df["load_kwh"].sum())
    total_boiler_el = float(df["boiler_el_input_kwh"].sum())

    total_electric_demand = total_load + total_boiler_el
    total_pv_self_consumed = total_pv_gen - total_grid_export

    sci = total_pv_self_consumed / total_pv_gen if total_pv_gen > 0 else np.nan
    ssi = (total_electric_demand - total_grid_import) / total_electric_demand if total_electric_demand > 0 else np.nan

    return {
        "total_pv_gen_kwh": total_pv_gen,
        "total_grid_export_kwh": total_grid_export,
        "total_grid_import_kwh": total_grid_import,
        "total_load_kwh": total_load,
        "total_boiler_el_input_kwh": total_boiler_el,
        "total_electric_demand_kwh": total_electric_demand,
        "total_pv_self_consumed_kwh": total_pv_self_consumed,
        "SCI": sci,
        "SSI": ssi,
    }


def prepare_per_user_df_for_plot(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # villanyszámla oszlop egységesítése
    if "net_cost_Ft" in out.columns:
        out["net_bill_ft"] = out["net_cost_Ft"]
    elif "net_bill_ft" not in out.columns:
        raise ValueError("Nincs net_cost_Ft vagy net_bill_ft oszlop a bemeneti táblában.")

    # csoportok
    has_pv = out["has_pv"] if "has_pv" in out.columns else pd.Series(0, index=out.index)
    has_bess = out["has_bess"] if "has_bess" in out.columns else pd.Series(0, index=out.index)

    group_label = np.where(
        has_pv.astype(int) == 0,
        "Nincs PV",
        np.where(has_bess.astype(int) == 1, "PV+BESS", "PV")
    )
    out["group_label"] = group_label

    return out


def analyze_boiler_results(results_dir: str | Path):
    out_case = Path(results_dir)

    combined_path = out_case / "household_summary_combined.csv"
    energy_path = out_case / "household_energy_summary.csv"
    finance_path = out_case / "household_finance_summary.csv"

    if combined_path.exists():
        df = pd.read_csv(combined_path)
    elif energy_path.exists() and finance_path.exists():
        energy_df = pd.read_csv(energy_path)
        finance_df = pd.read_csv(finance_path)
        df = energy_df.merge(finance_df, on="household", how="left")
    else:
        raise FileNotFoundError(
            "Nem található sem a household_summary_combined.csv, "
            "sem az energy + finance summary fájl."
        )

    # Plothoz szükséges tábla
    per_user_df = prepare_per_user_df_for_plot(df)

    # S-görbe
    plot_household_percentiles_by_group_with_global_scurve(per_user_df, out_case)

    # SCI / SSI
    community = compute_community_sci_ssi(df)

    # plusz pár egyszerű stat
    community.update({
        "n_households": int(len(df)),
        "mean_net_bill_ft": float(per_user_df["net_bill_ft"].mean()),
        "median_net_bill_ft": float(per_user_df["net_bill_ft"].median()),
        "min_net_bill_ft": float(per_user_df["net_bill_ft"].min()),
        "max_net_bill_ft": float(per_user_df["net_bill_ft"].max()),
    })

    # mentés JSON
    with open(out_case / "community_indicators.json", "w", encoding="utf-8") as f:
        json.dump(community, f, indent=2, ensure_ascii=False)

    # mentés CSV
    pd.DataFrame([community]).to_csv(out_case / "community_indicators.csv", index=False)

    print("Elemzés kész.")
    print(f"SCI = {community['SCI']:.4f}")
    print(f"SSI = {community['SSI']:.4f}")
    print(f"Plot mentve: {out_case / 'household_bill_percentiles_by_group_global_scurve.png'}")
    print(f"Indikátorok mentve: {out_case / 'community_indicators.json'}")

    return community


if __name__ == "__main__":
    analyze_boiler_results("results_individual_opt_boiler")