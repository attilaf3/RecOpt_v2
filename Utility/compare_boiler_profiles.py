from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import yaml

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


DT = 0.25  # 15 perc [h]
EXPECTED_LEN = 35040

RHO_WATER_KG_PER_L = 1.0
CP_WATER_J_PER_KGK = 4186.0
J_PER_KWH = 3_600_000.0
KWH_PER_L_PER_K = RHO_WATER_KG_PER_L * CP_WATER_J_PER_KGK / J_PER_KWH

EXCLUDE = {"battery", "bess", "community"}


def _safe_name(s: str) -> str:
    return str(s).replace("/", "_").replace("\\", "_").replace(":", "_")


def _keep_15min(v: np.ndarray, name: str = "profil") -> np.ndarray:
    v = np.asarray(v, dtype=float).ravel()
    if v.size != EXPECTED_LEN:
        raise ValueError(f"{name}: a profil hossza {v.size}, de {EXPECTED_LEN} kell.")
    return v


def _energy_profile_kwh_step(v: np.ndarray, name: str = "profil") -> np.ndarray:
    """
    Beolvasott idősor energiaként kezelve: e [kWh / 15 perc].
    """
    return np.maximum(_keep_15min(v, name=name), 0.0)


def _find_user_yaml(roots: Iterable[os.PathLike], key: str) -> Optional[Path]:
    for r in roots:
        r = Path(r)
        for cand in (str(key), f"{key}.yaml"):
            p = r / cand
            if p.exists():
                return p
    return None


def _read_csv_indexed(path: os.PathLike) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df.columns = [str(c) for c in df.columns]
    df = df[[c for c in df.columns if str(c).strip().lower() not in EXCLUDE]]
    return df


def _calc_metrics(meas_kw: np.ndarray, calc_kw: np.ndarray, dt: float) -> dict:
    diff_kw = calc_kw - meas_kw

    meas_kwh = float(np.sum(meas_kw) * dt)
    calc_kwh = float(np.sum(calc_kw) * dt)
    diff_kwh = calc_kwh - meas_kwh

    mae_kw = float(np.mean(np.abs(diff_kw)))
    rmse_kw = float(np.sqrt(np.mean(diff_kw**2)))
    max_abs_kw = float(np.max(np.abs(diff_kw)))

    if np.std(meas_kw) > 1e-12 and np.std(calc_kw) > 1e-12:
        corr = float(np.corrcoef(meas_kw, calc_kw)[0, 1])
    else:
        corr = np.nan

    rel_diff_pct = 100.0 * diff_kwh / meas_kwh if meas_kwh > 1e-12 else np.nan
    ratio_calc_meas = calc_kwh / meas_kwh if meas_kwh > 1e-12 else np.nan

    return {
        "measured_el_kwh": meas_kwh,
        "calc_el_from_liters_kwh": calc_kwh,
        "diff_calc_minus_measured_kwh": diff_kwh,
        "rel_diff_pct": rel_diff_pct,
        "ratio_calc_to_measured": ratio_calc_meas,
        "mae_kw": mae_kw,
        "rmse_kw": rmse_kw,
        "max_abs_diff_kw": max_abs_kw,
        "corr_kw": corr,
        "measured_peak_kw": float(np.max(meas_kw)),
        "calc_peak_kw": float(np.max(calc_kw)),
        "measured_nonzero_steps": int(np.sum(meas_kw > 1e-9)),
        "calc_nonzero_steps": int(np.sum(calc_kw > 1e-9)),
    }


def compare_boiler_profiles(
    sim_yaml: os.PathLike,
    profiles_csv: os.PathLike,
    dhw_profiles_csv: os.PathLike,
    out_dir: os.PathLike,
    max_users: int = 105,
    target_user: str | None = None,
    dt: float = DT,
    save_timeseries: bool = True,
    make_plots: bool = True,
) -> dict:
    sim_yaml = Path(sim_yaml)
    profiles_csv = Path(profiles_csv)
    dhw_profiles_csv = Path(dhw_profiles_csv)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ts_dir = out_dir / "timeseries"
    if save_timeseries:
        ts_dir.mkdir(parents=True, exist_ok=True)

    sim = yaml.safe_load(sim_yaml.read_text(encoding="utf-8")) or {}
    users_all = [u for u in list(sim.get("users_list", [])) if str(u).strip().lower() not in EXCLUDE]

    if target_user is not None:
        users = [u for u in users_all if str(u).strip() == str(target_user).strip()]
        if not users:
            raise RuntimeError(f"A target_user nincs a users_list-ben: {target_user}")
    else:
        users = users_all[: int(max_users)]

    if not users:
        raise RuntimeError("Nincs feldolgozható user a YAML users_list mezőjében.")

    df_meas = _read_csv_indexed(profiles_csv)
    df_liters = _read_csv_indexed(dhw_profiles_csv)

    search_roots = [sim_yaml.parent / "Users", sim_yaml.parent]

    rows: list[dict] = []

    for user_key in users:
        ypath = _find_user_yaml(search_roots, str(user_key))
        if ypath is None:
            print(f"[WARN] User YAML nem található, kihagyom: {user_key}")
            continue

        u = yaml.safe_load(ypath.read_text(encoding="utf-8")) or {}
        units = u.get("units") or {}
        name = (units.get("name") or {}).get("name", str(user_key))

        heater = units.get("ut") or {}
        hss = units.get("hss") or {}

        # Mért villamos bojlerprofil: measurements CSV, ut.profile, kWh/lépés.
        ut_profile = str(heater.get("profile")) if heater.get("profile") is not None else None
        has_measured = bool(ut_profile and ut_profile in df_meas.columns)
        if has_measured:
            e_meas_el_kwh_step = _energy_profile_kwh_step(
                df_meas[ut_profile].to_numpy(),
                name=f"mért bojlerprofil {name} / {ut_profile}",
            )
        else:
            e_meas_el_kwh_step = np.zeros(EXPECTED_LEN, dtype=float)

        # Literes csapolási profil: dhw CSV, hss.profile, liter/lépés.
        dhw_profile = str(hss.get("profile")) if hss.get("profile") is not None else None
        has_liter = bool(dhw_profile and dhw_profile in df_liters.columns)
        if has_liter:
            liters_step = np.maximum(
                _keep_15min(
                    df_liters[dhw_profile].to_numpy(),
                    name=f"literes DHW profil {name} / {dhw_profile}",
                ),
                0.0,
            )
        else:
            liters_step = np.zeros(EXPECTED_LEN, dtype=float)

        T_in = float(hss.get("T_in", 12.0))
        T_out = float(hss.get("T_out", 55.0))
        eta_elh = float(hss.get("eta_elh", 0.95))
        dT = max(T_out - T_in, 0.0)

        # Liter -> termikus energia [kWh/lépés]
        e_calc_th_kwh_step = liters_step * KWH_PER_L_PER_K * dT

        # Termikus energia -> villamos energia [kWh/lépés]
        # Egyszerű direkt összevetés: tartályveszteség és időbeli eltolás nélkül.
        if eta_elh <= 1e-12:
            e_calc_el_kwh_step = np.zeros_like(e_calc_th_kwh_step)
        else:
            e_calc_el_kwh_step = e_calc_th_kwh_step / eta_elh

        p_meas_kw = e_meas_el_kwh_step / dt
        p_calc_kw = e_calc_el_kwh_step / dt
        p_diff_kw = p_calc_kw - p_meas_kw

        metrics = _calc_metrics(p_meas_kw, p_calc_kw, dt=dt)

        row = {
            "user_key": str(user_key),
            "user_name": str(name),
            "yaml_path": str(ypath),
            "ut_profile_measured": ut_profile,
            "hss_profile_liters": dhw_profile,
            "has_measured_ut_profile": int(has_measured),
            "has_liter_dhw_profile": int(has_liter),
            "T_in_C": T_in,
            "T_out_C": T_out,
            "dT_K": dT,
            "eta_elh": eta_elh,
            "liters_total_l": float(np.sum(liters_step)),
            "calc_th_from_liters_kwh": float(np.sum(e_calc_th_kwh_step)),
            **metrics,
        }
        rows.append(row)

        if save_timeseries and (has_measured or has_liter):
            ts = pd.DataFrame({
                "e_measured_el_kwh_step": e_meas_el_kwh_step,
                "p_measured_el_kw": p_meas_kw,
                "liter_dhw_step": liters_step,
                "e_calc_th_from_liters_kwh_step": e_calc_th_kwh_step,
                "e_calc_el_from_liters_kwh_step": e_calc_el_kwh_step,
                "p_calc_el_from_liters_kw": p_calc_kw,
                "p_diff_calc_minus_measured_kw": p_diff_kw,
                "e_diff_calc_minus_measured_kwh_step": e_calc_el_kwh_step - e_meas_el_kwh_step,
            })
            ts.to_csv(ts_dir / f"boiler_compare_{_safe_name(name)}.csv", index=False)

    summary_df = pd.DataFrame(rows)
    if summary_df.empty:
        raise RuntimeError("Nem készült összehasonlítási sor. Ellenőrizd a users_list / YAML / profilneveket.")

    summary_df.to_csv(out_dir / "boiler_profile_comparison_summary.csv", index=False)

    totals = {
        "n_users_processed": int(len(summary_df)),
        "n_users_with_measured": int(summary_df["has_measured_ut_profile"].sum()),
        "n_users_with_liter_profile": int(summary_df["has_liter_dhw_profile"].sum()),
        "n_users_with_both": int(((summary_df["has_measured_ut_profile"] == 1) & (summary_df["has_liter_dhw_profile"] == 1)).sum()),
        "measured_el_kwh_total": float(summary_df["measured_el_kwh"].sum()),
        "calc_el_from_liters_kwh_total": float(summary_df["calc_el_from_liters_kwh"].sum()),
        "diff_calc_minus_measured_kwh_total": float(summary_df["calc_el_from_liters_kwh"].sum() - summary_df["measured_el_kwh"].sum()),
        "out_dir": str(out_dir),
        "dt_h": float(dt),
        "formula": "e_el_calc_kWh = liters * 4186 / 3_600_000 * (T_out - T_in) / eta_elh",
    }
    denom = totals["measured_el_kwh_total"]
    totals["rel_diff_total_pct"] = float(100.0 * totals["diff_calc_minus_measured_kwh_total"] / denom) if denom > 1e-12 else None

    (out_dir / "boiler_profile_comparison_totals.json").write_text(
        json.dumps(totals, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if make_plots and plt is not None:
        _make_plots(summary_df, out_dir)

    print("[INFO] Kész az összehasonlítás.")
    print(json.dumps(totals, indent=2, ensure_ascii=False))
    return totals


def _make_plots(summary_df: pd.DataFrame, out_dir: Path) -> None:
    both = summary_df[
        (summary_df["has_measured_ut_profile"] == 1)
        & (summary_df["has_liter_dhw_profile"] == 1)
    ].copy()

    if both.empty:
        return

    # Scatter: éves mért vs literből számolt villamos energia.
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111)
    x = both["measured_el_kwh"].to_numpy(dtype=float)
    y = both["calc_el_from_liters_kwh"].to_numpy(dtype=float)
    ax.scatter(x, y, s=30, alpha=0.8)
    lim = max(float(np.max(x)), float(np.max(y)), 1.0)
    ax.plot([0, lim], [0, lim], linewidth=1.5, label="1:1")
    ax.set_xlabel("Mért villamos bojlerenergia [kWh/év]")
    ax.set_ylabel("Literből számolt villamos energia [kWh/év]")
    ax.set_title("Mért vs. literprofilból számolt bojlerenergia")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "measured_vs_calc_boiler_energy_scatter.png", dpi=200)
    plt.close(fig)

    # Oszlopdiagram: legnagyobb eltérések.
    top = both.copy()
    top["abs_diff"] = top["diff_calc_minus_measured_kwh"].abs()
    top = top.sort_values("abs_diff", ascending=False).head(30)

    fig = plt.figure(figsize=(11, 6))
    ax = fig.add_subplot(111)
    labels = top["user_name"].astype(str).to_list()
    x_pos = np.arange(len(top))
    width = 0.4
    ax.bar(x_pos - width/2, top["measured_el_kwh"], width=width, label="Mért")
    ax.bar(x_pos + width/2, top["calc_el_from_liters_kwh"], width=width, label="Literből számolt")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=75, ha="right")
    ax.set_ylabel("Villamos energia [kWh/év]")
    ax.set_title("Legnagyobb éves eltérésű bojlerprofilok")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "top_diff_boiler_profiles_bar.png", dpi=200)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Mért villamos bojlerprofilok összevetése literes DHW profilból számolt villamos profillal."
    )
    ap.add_argument("--sim", required=True, help="simulation_config YAML útvonala")
    ap.add_argument("--profiles", required=True, help="mért diszaggregált profilok CSV-je")
    ap.add_argument("--dhw-profiles", required=True, help="literes DHW profilok CSV-je")
    ap.add_argument("--out", default="boiler_profile_comparison", help="kimeneti mappa")
    ap.add_argument("--max-users", type=int, default=105)
    ap.add_argument("--target-user", default=None, help="opcionális: csak egy user_key feldolgozása")
    ap.add_argument("--no-timeseries", action="store_true", help="ne mentsen userenként idősoros CSV-t")
    ap.add_argument("--no-plots", action="store_true", help="ne készítsen ábrákat")
    args = ap.parse_args(argv)

    compare_boiler_profiles(
        sim_yaml=args.sim,
        profiles_csv=args.profiles,
        dhw_profiles_csv=args.dhw_profiles,
        out_dir=args.out,
        max_users=args.max_users,
        target_user=args.target_user,
        save_timeseries=not args.no_timeseries,
        make_plots=not args.no_plots,
    )



if __name__ == "__main__":
    compare_boiler_profiles(
        sim_yaml="Input/simulation_config_disaggregated_with_userlist.yaml",
        profiles_csv="Input/measurements_disaggregated.csv",
        dhw_profiles_csv="Input/dhw_v2.csv",
        out_dir="boiler_profile_comparison",
        max_users=105,
        target_user=None,
        save_timeseries=True,
        make_plots=True,
    )