from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def plot_household_percentiles_by_group_with_global_scurve(per_user_df: pd.DataFrame, out_case: Path, x_col: str):
    df = per_user_df.copy().sort_values(x_col).reset_index(drop=True)
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

    for group in ["Nincs PV", "PV", "PV+BESS"]:
        sub = df[df["group_label"] == group].copy()
        if len(sub) == 0:
            continue

        plt.scatter(
            sub[x_col],
            sub["percentile"],
            s=28,
            marker=marker_map.get(group, "o"),
            alpha=0.8,
            label=group,
        )

    x = df[x_col].to_numpy(dtype=float)
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
            plt.plot(x_grid, y_fit, color="darkred", linewidth=2, label="Illesztett S-görbe (összes)")
        except Exception:
            pass

    x_lo = np.percentile(df[x_col], 1)
    x_hi = np.percentile(df[x_col], 100)
    if x_lo == x_hi:
        x_lo -= 1
        x_hi += 1

    plt.xlim(x_lo, x_hi)
    plt.xlabel("Éves bruttó villanyszámla [Ft/év] (csak vételezés)")
    plt.ylabel("Háztartások aránya [%]")
    plt.title("Háztartások bruttó villanyszámla szerinti eloszlása")
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_case / "household_gross_bill_percentiles_global_scurve.png", dpi=200)
    plt.close()


def compute_per_household_sci_ssi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Háztartásonként:
      SCI = (PV - export) / PV
      SSI = (load - import) / load
    """
    req = ["household", "pv_gen_kwh", "grid_export_kwh", "grid_import_total_kwh", "total_load_kwh"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Hiányzó oszlopok: {missing}")

    out = df[["household"]].copy()
    pv = df["pv_gen_kwh"].astype(float)
    export = df["grid_export_kwh"].astype(float)
    imp = df["grid_import_total_kwh"].astype(float)
    load = df["total_load_kwh"].astype(float)

    out["pv_gen_kwh"] = pv
    out["grid_export_kwh"] = export
    out["grid_import_kwh"] = imp
    out["total_load_kwh"] = load

    out["SCI"] = np.where(pv > 1e-12, (pv - export) / pv, np.nan)
    out["SSI"] = np.where(load > 1e-12, (load - imp) / load, np.nan)

    # clamp (csak esztétika/robosztusság)
    out["SCI"] = out["SCI"].clip(lower=-1.0, upper=1.0)
    out["SSI"] = out["SSI"].clip(lower=-1.0, upper=1.0)

    return out


def compute_community_sci_ssi(df: pd.DataFrame) -> dict:
    pv = float(df["pv_gen_kwh"].sum())
    export = float(df["grid_export_kwh"].sum())
    imp = float(df["grid_import_total_kwh"].sum())
    load = float(df["total_load_kwh"].sum())

    sci = (pv - export) / pv if pv > 1e-12 else np.nan
    ssi = (load - imp) / load if load > 1e-12 else np.nan

    return {
        "SCI": sci,
        "SSI": ssi,
        "total_pv_gen_kwh": pv,
        "total_grid_export_kwh": export,
        "total_grid_import_kwh": imp,
        "total_load_kwh": load,
    }


def analyze_bess_results(results_dir: str | Path):
    out_case = Path(results_dir)
    in_path = out_case / "household_summary.csv"

    if not in_path.exists():
        raise FileNotFoundError(f"Nem található: {in_path}")

    df = pd.read_csv(in_path)

    # --- csoportok a plothoz ---
    # (a run_individual_opt_bess.py-ban has_pv / has_bess már benne van) :contentReference[oaicite:2]{index=2}
    df["group_label"] = np.where(
        df["has_pv"].astype(int) == 0,
        "Nincs PV",
        np.where(df["has_bess"].astype(int) == 1, "PV+BESS", "PV")
    )

    # --- bruttó x-tengely ---
    # bruttó = csak vételezés költsége = grid_cost_Ft
    if "grid_cost_Ft" not in df.columns:
        raise ValueError("Hiányzik a 'grid_cost_Ft' oszlop (bruttó vételezési költség).")
    df["gross_bill_ft"] = df["grid_cost_Ft"].astype(float)

    # --- plot ---
    plot_household_percentiles_by_group_with_global_scurve(df, out_case, x_col="gross_bill_ft")

    # --- SCI/SSI háztartásonként ---
    per_hh = compute_per_household_sci_ssi(df)
    per_hh.to_csv(out_case / "household_sci_ssi.csv", index=False)

    # --- SCI/SSI közösségileg ---
    community = compute_community_sci_ssi(df)
    community.update({
        "n_households": int(len(df)),
        "mean_gross_bill_ft": float(df["gross_bill_ft"].mean()),
        "median_gross_bill_ft": float(df["gross_bill_ft"].median()),
        "min_gross_bill_ft": float(df["gross_bill_ft"].min()),
        "max_gross_bill_ft": float(df["gross_bill_ft"].max()),
    })

    (out_case / "community_indicators.json").write_text(
        json.dumps(community, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    pd.DataFrame([community]).to_csv(out_case / "community_indicators.csv", index=False)

    print("Kész.")
    print(f"Plot: {out_case / 'household_gross_bill_percentiles_global_scurve.png'}")
    print(f"Household SCI/SSI: {out_case / 'household_sci_ssi.csv'}")
    print(f"Community SCI={community['SCI']:.4f}, SSI={community['SSI']:.4f}")

    return community


if __name__ == "__main__":
    analyze_bess_results("results_individual_opt_bess")