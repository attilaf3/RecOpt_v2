"""Generate household/heat-pump YAML files and matching HP profiles.

The generated files are deliberately written to a separate directory.  The
original input files are never modified.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

# Permit both ``python -m HeatPump.yaml_generator`` and direct execution from
# the repository root (``python HeatPump/yaml_generator.py``).
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Utility.configuration import config
from HeatPump.aggregate_hp_profile import make_house_samples
from HeatPump.profile_sampling import perturb_hp_house
from HeatPump.simulate_ata import HOUSES_RAW, load_or_make_inputs, solar_gain_sepsi, tabula_to_5r2c_iso_sepsi, simulate_5r2c


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def generate(
    input_users: Path,
    output_dir: Path,
    profiles_csv: Path,
    share_pct: float = 28.0,
    seed: int | None = 42,
) -> dict[str, Any]:
    """Create enriched YAMLs and one column per HP in the profile CSV."""
    source_files = sorted(input_users.glob("*.yaml")) + sorted(input_users.glob("*.yml"))
    if not source_files:
        raise FileNotFoundError(f"No user YAML files found in {input_users}")

    profiles = pd.read_csv(profiles_csv, index_col=0)
    idx, t_env, irradiance = load_or_make_inputs()
    idx = pd.DatetimeIndex(idx)
    if len(profiles) != len(idx):
        raise ValueError(f"Profile rows ({len(profiles)}) do not match HP weather rows ({len(idx)})")

    output_users = output_dir / "Users"
    output_users.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    n_hp = max(0, min(len(source_files), int(round(len(source_files) * share_pct / 100.0))))
    hp_indices = set(np.sort(rng.choice(len(source_files), size=n_hp, replace=False)).tolist())
    house_samples = make_house_samples(len(source_files), seed)
    hp_profiles: dict[str, np.ndarray] = {}
    manifest: list[dict[str, Any]] = []
    household_rows: list[dict[str, Any]] = []

    for i, source in enumerate(source_files):
        data = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
        user_id = source.stem
        units = data.setdefault("units", {})
        hp_meta: dict[str, Any] | None = None
        if i in hp_indices:
            household = _plain(house_samples[f"house_{i:03d}"])
            household_rows.append({"user_id": user_id, **{k: v for k, v in household.items() if not isinstance(v, dict)}})
            base = dict(HOUSES_RAW[next(iter(HOUSES_RAW))])
            hp_house = perturb_hp_house(base, rng)
            pars = tabula_to_5r2c_iso_sepsi(
                Aref_m2=hp_house["Aref"], h_m=hp_house["h"], Htr_W_per_K=hp_house["Htr"],
                Hve_W_per_K=hp_house["Hve"], U_window_W_m2K=hp_house["U_window"],
                window_total_m2=hp_house["window_total"], Awin_raw=hp_house["Awin_raw"],
            )
            q_sol = solar_gain_sepsi(irradiance, pars.Awin_by_dir_m2, idx)
            result = simulate_5r2c(idx, t_env, q_sol.values, pars, dt_h=0.25)
            profile_id = f"hp_{user_id}"
            hp_profiles[profile_id] = np.asarray(result["P_hp_el"], dtype=float) * 0.25
            hp_meta = {
                "profile_id": profile_id,
                "profile_unit": "kWh/step",
                "household_parameters": household,
                "heat_pump_parameters": _plain(hp_house),
            }
            units["heat_pump"] = hp_meta
        else:
            units.pop("heat_pump", None)
        destination = output_users / source.name
        destination.write_text(yaml.safe_dump(_plain(data), allow_unicode=True, sort_keys=False), encoding="utf-8")
        manifest.append({"user_id": user_id, "heat_pump": hp_meta is not None, "profile_id": hp_meta and hp_meta["profile_id"]})

    hp_df = pd.DataFrame(hp_profiles, index=profiles.index)
    hp_df.index.name = profiles.index.name or "timestamp"
    parsed_index = pd.to_datetime(hp_df.index, errors="coerce")
    if not parsed_index.isna().any():
        hp_df.index = parsed_index
    hp_df.to_csv(output_dir / "heat_pump_profiles.csv")
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(household_rows).to_csv(output_dir / "household_parameters.csv", index=False)
    _plot_outputs(output_dir, hp_df, household_rows, manifest)
    return {"users": len(source_files), "heat_pump_users": n_hp, "share_pct": 100 * n_hp / len(source_files), "output": str(output_dir)}


def _plot_outputs(output_dir: Path, profiles: pd.DataFrame, household_rows: list[dict[str, Any]], manifest: list[dict[str, Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    params = pd.DataFrame(household_rows).select_dtypes(include="number")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    axes = axes.ravel()
    if not profiles.empty:
        plot_profiles = profiles.iloc[:: max(1, len(profiles) // 2000)]
        plot_profiles.iloc[:, : min(5, profiles.shape[1])].plot(ax=axes[0], legend=False, alpha=0.8)
        axes[0].set_ylabel("kWh / 15 min")
        axes[0].set_title("Generated heat-pump profiles (first five)")
        axes[0].tick_params(axis="x", labelrotation=45)
        plot_profiles.sum(axis=1).plot(ax=axes[1], color="tab:red")
        axes[1].set_ylabel("kWh / 15 min")
        axes[1].set_title("Aggregated heat-pump profile")
        axes[1].tick_params(axis="x", labelrotation=45)
        profiles.sum(axis=1).resample("D").sum().plot(ax=axes[2], color="tab:green")
        axes[2].set_ylabel("kWh / day")
        axes[2].set_title("Daily aggregated heat-pump energy")
        axes[2].tick_params(axis="x", labelrotation=45)
        axes[3].imshow(plot_profiles.T.to_numpy(), aspect="auto", interpolation="nearest", cmap="magma")
        axes[3].set_title("Heat-pump profile heatmap")
        axes[3].set_xlabel("Time step")
        axes[3].set_ylabel("Profile")
    else:
        axes[0].text(0.5, 0.5, "No heat pumps selected", ha="center")
        for axis in axes[1:]:
            axis.text(0.5, 0.5, "No heat pumps selected", ha="center")
    if not params.empty:
        params.to_csv(output_dir / "household_parameters.csv", index=False)
    else:
        pd.DataFrame().to_csv(output_dir / "household_parameters.csv", index=False)
    if not params.empty:
        parameter_fig, parameter_ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
        params.drop(columns=[c for c in ["user_id"] if c in params], errors="ignore").boxplot(
            ax=parameter_ax, rot=45
        )
        parameter_ax.set_title("Generated household parameters for heat-pump households")
        parameter_ax.set_ylabel("Value")
        parameter_ax.tick_params(axis="x", labelrotation=45)
        parameter_fig.savefig(output_dir / "heat_pump_household_parameters.png", dpi=150)
        plt.close(parameter_fig)
    fig.savefig(output_dir / "heat_pump_profiles_overview.png", dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-users", type=Path, default=config.getpath("paths", "users_directory"))
    parser.add_argument("--output-dir", type=Path, default=Path(config.getpath("paths", "output_root")) / "generated_heat_pumps")
    parser.add_argument("--profiles-csv", type=Path, default=config.getpath("paths", "profiles_csv"))
    parser.add_argument("--share-pct", type=float, default=config.getfloat("scenario", "heat_pump_share_pct"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(generate(args.input_users, args.output_dir, args.profiles_csv, args.share_pct, args.seed), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
