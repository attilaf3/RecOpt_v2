"""Canonical BESS state indexing; legacy fields retained as explicit aliases."""
import numpy as np


def add_bess_states(timeseries, boundary):
    boundary = np.asarray(boundary, dtype=float)
    if boundary.ndim not in (1, 2) or boundary.shape[0] < 1 or not np.all(np.isfinite(boundary)):
        raise ValueError("BESS boundary must contain finite T+1 states")
    timeseries["e_bess_boundary"] = boundary
    timeseries["e_bess_start"] = boundary[:-1].copy()
    timeseries["e_bess_end"] = boundary[1:].copy()
    # Canonical per-step SOC is always the state at the beginning of the step.
    timeseries["e_bess"] = timeseries["e_bess_start"]
    return timeseries


def add_bess_flows(timeseries, dt):
    """Expose common AC-side BESS flows both in kWh/step and average kW."""
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be positive hours")
    for energy, power in (("e_pv_to_bess", "p_pv_bess"),
                          ("e_grid_to_bess", "p_grid_bess"),
                          ("e_bess_to_load", "p_bess_out")):
        if power in timeseries:
            timeseries[energy] = np.asarray(timeseries[power]) * dt
        elif energy in timeseries:
            timeseries[power] = np.asarray(timeseries[energy]) / dt
        else:
            raise ValueError(f"Missing BESS flow: {energy} or {power}")
    timeseries["e_bess_in"] = timeseries["e_pv_to_bess"] + timeseries["e_grid_to_bess"]
    timeseries["e_bess_out"] = timeseries["e_bess_to_load"]
    timeseries["p_bess_in"] = timeseries["e_bess_in"] / dt
    return timeseries


def save_bess_result(timeseries, directory, dt, user_names):
    """Write the same versioned BESS result package for every runner.

    Existing scenario-specific CSVs remain available. States are kWh, energy
    flows are AC-side kWh/step, powers are average kW during each interval.
    """
    import json
    from pathlib import Path
    import pandas as pd
    data = dict(timeseries)
    add_bess_states(data, data["e_bess_boundary"])
    add_bess_flows(data, dt)
    keys = ("e_bess_boundary", "e_bess_start", "e_bess_end", "e_pv_to_bess",
            "e_grid_to_bess", "e_bess_to_load", "e_bess_in", "e_bess_out",
            "p_pv_bess", "p_grid_bess", "p_bess_in", "p_bess_out")
    arrays = {key: np.asarray(data[key]) for key in keys}
    count = len(arrays["e_bess_start"])
    for key, arr in arrays.items():
        if arr.ndim == 1:
            arr = arr[:, None]
        expected = (count + (key == "e_bess_boundary"), len(user_names))
        if arr.shape != expected or not np.all(np.isfinite(arr)):
            raise ValueError(f"Invalid result shape/values for {key}: expected {expected}")
        arrays[key] = arr
    out = Path(directory) / "bess"
    out.mkdir(parents=True, exist_ok=True)
    for key, arr in arrays.items():
        pd.DataFrame(arr, columns=user_names).to_csv(out/f"{key}.csv", index=False)
    metadata = {"schema_version": 1, "dt_hours": float(dt), "steps": count,
                "users": [str(name) for name in user_names],
                "state_index": "boundary[0..T]; start=boundary[:-1]; end=boundary[1:]",
                "units": {key: ("kW" if key.startswith("p_") else
                    "kWh" if key in keys[:3] else "kWh/step") for key in keys}}
    (out/"schema.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
