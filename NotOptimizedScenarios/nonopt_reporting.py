"""Eredménytáblák és idősorok egységes, redundanciamentes mentése."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def save_results(
    out_dir: str | Path,
    user_names: list[str],
    per_user: pd.DataFrame,
    summary: dict,
    timeseries: dict[str, np.ndarray],
    *,
    sharing: bool,
    bess: bool,
    boiler_diagnostics: dict[str, np.ndarray] | None = None,
    pair_kwh: np.ndarray | None = None,
    pair_payment: np.ndarray | None = None,
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    per_user.to_csv(out / "per_user_summary.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    saved_timeseries = {}
    for name, values in timeseries.items():
        if not bess and ("bess" in name or name.startswith("d_bess")):
            continue
        if not sharing and ("shared" in name or "before_share" in name):
            continue
        pd.DataFrame(values, columns=user_names).to_csv(out / f"{name}.csv", index=False)
        saved_timeseries[name] = values

    if boiler_diagnostics:
        for name, values in boiler_diagnostics.items():
            if name == "dynamic":
                continue
            pd.DataFrame(values, columns=user_names).to_csv(out / f"{name}.csv", index=False)
    if sharing and pair_kwh is not None and pair_payment is not None:
        pd.DataFrame(pair_kwh, index=user_names, columns=user_names).to_csv(out / "shared_pair_kwh.csv")
        pd.DataFrame(pair_payment, index=user_names, columns=user_names).to_csv(out / "shared_pair_payment_ft.csv")

    community = pd.DataFrame({f"{name}_total": values.sum(axis=1) for name, values in saved_timeseries.items()})
    community.to_csv(out / "community_timeseries.csv", index=False)
