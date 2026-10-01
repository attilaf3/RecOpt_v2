"""Compare fixed Geo HP profiles with modeled 5R2C HP for the PV cohort.

Run after the 20-household assignment, from the same working directory as
run_0_I.py. The 12 non-PV households keep their fixed Geo profile in BOTH
branches. Uses the project's existing individual and community settlement.
"""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from nonopt_data import DT, N_STEPS, _find_yaml, load_inputs
from nonopt_hp import participation_factor, restricted_mask, simulate_5r2c_hp
from nonopt_simulation import simulate


# ---------- Project paths / user editable settings ----------
SIM_YAML = Path("../Input/simulation_config_disaggregated_with_userlist.yaml")
PROFILES_CSV = Path("../Input/measurements_disaggregated_v3.csv")
DHW_CSV = Path("../Input/dhw_v2.csv")
HP_CSV = Path("../Input/heat_pump_profiles.csv")
QSOL_CSV = Path("../Input/solar_gain_profiles.csv")
WEATHER_CSV = Path("../Input/homerseklet_sugarzas_interpolated_polynomial_fixed.csv")
OUT_DIR = Path("results_hp_comparison")
MAX_USERS = 105
EXPECTED_PV_HP = 8
EXPECTED_HP = 20
TIMESTAMP_TIMEZONE = "local_clock"  # A 2021-es HP-profil órái helyi idő szerintiek.
SHARING_MODE = "proportional"
SUMMER_MONTHS = (6, 7, 8)
# A generált villamosprofilokban naponta 07:00–09:00 és 16:00–18:00 között
# pontosan 2-2 óra nulla HP-fogyasztás van. Mindkét ághoz ezeket használjuk.
MORNING = (7, 9)
EVENING = (16, 18)
# -----------------------------------------------------------


def _inputs():
    solar = pd.read_csv(QSOL_CSV)
    weather = pd.read_csv(WEATHER_CSV)
    hp = pd.read_csv(HP_CSV, usecols=["timestamp"])
    if len(solar) != N_STEPS or len(weather) != N_STEPS or len(hp) != N_STEPS:
        raise ValueError("All solar, weather and HP files must have 35040 rows")
    for frame in (solar, weather):
        if not pd.DatetimeIndex(pd.to_datetime(frame.iloc[:, 0])).equals(
            pd.DatetimeIndex(pd.to_datetime(hp["timestamp"]))
        ):
            raise ValueError("HP, Qsol and weather timestamps do not match")
    if "T" not in weather.columns:
        raise KeyError("Weather T [°C] is missing")
    tout = pd.to_numeric(weather["T"], errors="raise").to_numpy(float)
    if not np.isfinite(tout).all():
        raise ValueError("Outdoor temperature has NaN/Inf")
    timestamps = pd.DatetimeIndex(pd.to_datetime(hp["timestamp"]))
    if TIMESTAMP_TIMEZONE == "UTC":
        local = timestamps.tz_localize("UTC").tz_convert("Europe/Budapest")
    elif TIMESTAMP_TIMEZONE == "local_clock":
        local = timestamps
    else:
        raise ValueError("TIMESTAMP_TIMEZONE must be UTC or local_clock")
    return solar, tout, local


def _model_one(m, hp, qsol, tout, local):
    result = simulate_5r2c_hp(m, hp, qsol, tout, local)
    return (result["electric_kwh"], result["inside_c"],
            result["restricted_critical_steps"],
            result["restricted_emergency_steps"])


def _restricted_mask(local):
    return restricted_mask(local)


def _user_blocks(data):
    config = yaml.safe_load(SIM_YAML.read_text(encoding="utf-8")) or {}
    excluded = {"battery", "bess", "community"}
    user_keys = [str(k) for k in config.get("users_list", [])[:MAX_USERS]
                 if str(k).strip().lower() not in excluded]
    records = []
    for key in user_keys:
        path = _find_yaml([SIM_YAML.parent / "users_v3"], key)
        if path is None:
            continue
        units = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("units") or {}
        records.append((key, units))
    if len(records) != len(data.user_names):
        raise ValueError("The selected YAML order differs from nonopt_data")
    for u, (_, units) in enumerate(records):
        if str((units.get("name") or {}).get("name", records[u][0])) != data.user_names[u]:
            raise ValueError("YAML and data house ordering differs")
    return records


def _summarize_branch(modelled_data, case_name, community, hp_mode):
    result = simulate(
        modelled_data, case_name=case_name, out_dir=OUT_DIR,
        community_settlement=community, rule_based_boiler=False,
        boiler_tariff="B", bess_share_pct=0.0,
        sharing_mode=SHARING_MODE, hp_baseline=True,
        hp_tariff_mode=hp_mode, save_outputs=False,
    )
    return result["per_user"], result["summary"]


def main():
    data = load_inputs(SIM_YAML, PROFILES_CSV, DHW_CSV, MAX_USERS,
                       hp_profiles_csv=HP_CSV)
    solar, tout, local = _inputs()
    records = _user_blocks(data)
    hp_users = [u for u in range(len(records)) if data.e_hp[:, u].sum() > 0]
    pv_hp = [u for u in hp_users if data.e_pv[:, u].sum() > 0]
    if len(hp_users) != EXPECTED_HP or len(pv_hp) != EXPECTED_PV_HP:
        raise ValueError(f"Expected {EXPECTED_HP} HP ({EXPECTED_PV_HP} PV); "
                         f"found {len(hp_users)} HP ({len(pv_hp)} PV)")

    restricted = _restricted_mask(local)
    expected_steps = 365 * 4 * int((MORNING[1]-MORNING[0]) + (EVENING[1]-EVENING[0]))
    if int(restricted.sum()) != expected_steps:
        raise ValueError("A 2+2 órás tiltás nem illeszkedik a 2021-es időindexhez")
    if np.any(data.e_hp[restricted][:, hp_users] > 1e-9):
        raise ValueError("A fix HP profil nem nulla a 07–09 / 16–18 tiltás alatt. "
                         "Ellenőrizd a TIMESTAMP_TIMEZONE beállítást és a CSV-t.")
    # A profil előállításakor a pf már csökkentette a HP villamos energiát.
    # Itt csak ellenőrizzük a nulla részvételű fagyos napokat, nem skálázunk újra.
    pf = participation_factor(local, tout)
    if np.any(data.e_hp[pf == 0][:, hp_users] > 1e-9):
        raise ValueError("A rögzített HP profil nem nulla pf=0 napon; "
                         "ellenőrizd az időzónát és a profilt.")

    modeled = data.e_hp.copy()
    diag_rows, month_rows = [], []
    for u in pv_hp:
        key, units = records[u]
        block = units["heat_pump"]
        model = block["household_parameters"]["model_5r2c"]
        hp = block["heat_pump_parameters"]
        col = str(block["profile_id"])
        if col not in solar:
            raise KeyError(f"Missing Qsol column {col}")
        qsol = pd.to_numeric(solar[col], errors="raise").to_numpy(float)
        electricity, inside, n_critical, n_emergency = _model_one(
            model, hp, qsol, tout, local
        )
        if np.any(electricity[restricted] > 1e-9):
            raise AssertionError("Modelled HP is on during the 2+2 hour ban")
        modeled[:, u] = electricity
        diag_rows.append({"household": key, "min_Ti_C": float(inside.min()),
                          "max_Ti_C": float(inside.max()),
                          "pf_zero_days": int(len(set(local[pf == 0].date))),
                          "profile_restricted_kwh":float(data.e_hp[restricted,u].sum()),
                          "model_restricted_kwh":float(electricity[restricted].sum()),
                          "restricted_critical_steps": n_critical,
                          "restricted_emergency_steps": n_emergency})
        for month in range(1, 13):
            selected = local.month == month
            month_rows.append({"household": key, "month": month,
                               "profile_hp_kwh": float(data.e_hp[selected, u].sum()),
                               "model_hp_kwh": float(electricity[selected].sum())})
    alt = replace(data, e_hp=modeled)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(diag_rows).to_csv(OUT_DIR / "modeled_temperature_checks.csv", index=False)
    pd.DataFrame(month_rows).to_csv(OUT_DIR / "hp_monthly_comparison.csv", index=False)

    all_tables = []
    totals = {}
    for settlement, is_community in (("individual", False), ("community", True)):
        baseline_users, baseline_total = _summarize_branch(data, "HP-profile-Geo", is_community, "GEO")
        modeled_users, modeled_total = _summarize_branch(alt, "HP-5R2C-PV-A", is_community, "PV_A_ELSE_GEO")
        base = baseline_users.set_index("user_name")
        new = modeled_users.set_index("user_name")
        selected_names = [data.user_names[u] for u in hp_users]
        table = pd.DataFrame({"settlement": settlement, "household": selected_names,
                              "has_pv": base.loc[selected_names, "has_pv"].to_numpy(),
                              "profile_tariff": base.loc[selected_names, "hp_tariff"].to_numpy(),
                              "model_tariff": new.loc[selected_names, "hp_tariff"].to_numpy()})
        for source, series in (("profile", base), ("model", new)):
            for key in ("hp_kwh", "grid_import_a_kwh", "grid_import_geo_kwh",
                        "geo_cost_ft", "brt_bill_ft", "SCI", "SSI"):
                table[f"{source}_{key}"] = series.loc[selected_names, key].to_numpy()
        for key in ("hp_kwh", "grid_import_a_kwh", "grid_import_geo_kwh", "brt_bill_ft"):
            table[f"delta_{key}"] = table[f"model_{key}"] - table[f"profile_{key}"]
        all_tables.append(table)
        totals[settlement] = {key: {"profile": float(baseline_total[key]),
                                    "model": float(modeled_total[key]),
                                    "delta": float(modeled_total[key] - baseline_total[key])}
                              for key in ("hp_kwh", "grid_import_a_kwh",
                                          "grid_import_geo_kwh", "brt_bill_ft", "SCI", "SSI")}
        print(f"{settlement}: HP {baseline_total['hp_kwh']:.1f} -> "
              f"{modeled_total['hp_kwh']:.1f} kWh; community bill "
              f"{baseline_total['brt_bill_ft']:.0f} -> {modeled_total['brt_bill_ft']:.0f} Ft")
    result = pd.concat(all_tables, ignore_index=True)
    result.to_csv(OUT_DIR / "hp_profile_vs_model_tariffs.csv", index=False)
    (OUT_DIR / "community_totals.json").write_text(
        json.dumps(totals, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(result[["settlement", "household", "has_pv", "profile_hp_kwh",
                  "model_hp_kwh", "model_tariff", "delta_brt_bill_ft"]].to_string(index=False))
    print(f"Saved comparisons in {OUT_DIR}")


if __name__ == "__main__":
    main()
