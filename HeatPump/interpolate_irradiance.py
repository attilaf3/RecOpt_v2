"""Create a quality-controlled 15-minute facade-irradiance data set.

This replaces the former third-degree polynomial interpolation.  Polynomial
interpolation can overshoot between hourly measurements and create physically
implausible irradiance peaks.  This script uses shape-preserving PCHIP
interpolation instead, preserves all source samples that pass quality control,
retains periods bounded by two zero measurements as zero, and records any
upper-limit clips.

Example
-------
python -m HeatPump.interpolate_irradiance --year 2021

By default, input and output paths come from ``[paths]`` in ``config.ini``:
``heat_pump_solar_raw_csv`` and ``heat_pump_solar_csv``.  Command-line paths
remain available as explicit overrides.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from Utility.configuration import config


DEFAULT_MAX_IRRADIANCE_W_M2 = 1400.0
TARGET_FREQUENCY = "15min"


def read_irradiance_csv(path: str | Path) -> pd.DataFrame:
    """Read a timestamp-indexed irradiance CSV and validate its source samples."""
    data = pd.read_csv(path, index_col=0)
    data.index = pd.to_datetime(data.index, errors="raise")
    data = data.sort_index()

    if data.empty:
        raise ValueError("The input irradiance CSV is empty.")
    if data.index.has_duplicates:
        duplicates = data.index[data.index.duplicated()].unique()[:5].tolist()
        raise ValueError(f"Duplicate timestamps in input, for example: {duplicates}")

    data = data.apply(pd.to_numeric, errors="raise").astype(float)
    if not np.isfinite(data.to_numpy()).all():
        raise ValueError("The input contains NaN or non-finite source measurements.")
    if (data < 0).any().any():
        raise ValueError("The input contains negative source irradiance values.")
    return data


def calendar_index(year: int) -> pd.DatetimeIndex:
    """Return the complete 15-minute index of one Gregorian calendar year."""
    return pd.date_range(
        start=f"{year}-01-01 00:00",
        end=f"{year}-12-31 23:45",
        freq=TARGET_FREQUENCY,
    )


def infer_year(source: pd.DataFrame) -> int:
    years = source.index.year.unique()
    if len(years) != 1:
        raise ValueError("Input spans multiple years; pass --year explicitly after splitting it.")
    return int(years[0])


def zero_bounded_mask(source: pd.Series, target_index: pd.DatetimeIndex) -> pd.Series:
    """Mark target points whose bounding source measurements are both zero."""
    left = source.reindex(target_index, method="ffill")
    right = source.shift(-1).reindex(target_index, method="ffill")
    return ((left == 0.0) & (right == 0.0)).fillna(False)


def interpolate_series(
    source: pd.Series,
    target_index: pd.DatetimeIndex,
    *,
    method: str,
    max_irradiance_w_m2: float | None,
) -> tuple[pd.Series, int]:
    """Interpolate one facade series and return it with the number of clips."""
    target = source.reindex(target_index)
    if target.notna().sum() < 2:
        raise ValueError(f"{source.name}: at least two source samples are required.")

    # PCHIP is shape-preserving: values cannot overshoot neighbouring samples.
    result = target.interpolate(method=method, limit_area="inside")
    result.loc[zero_bounded_mask(source, target_index)] = 0.0

    # Keep source observations exact before applying the explicit QC limit.
    result.loc[source.index] = source
    result = result.clip(lower=0.0)

    clip_count = 0
    if max_irradiance_w_m2 is not None:
        over_limit = result > max_irradiance_w_m2
        clip_count = int(over_limit.sum())
        result = result.clip(upper=max_irradiance_w_m2)

    # A usual hourly input ends at 23:00.  Its final three 15-minute points can
    # only be filled with zero when the final source sample is itself zero.
    tail_mask = result.index > source.index.max()
    if tail_mask.any() and float(source.iloc[-1]) == 0.0:
        result.loc[tail_mask] = 0.0

    if result.isna().any():
        missing = result.index[result.isna()]
        raise ValueError(
            f"{source.name}: {len(missing)} unresolved values remain after interpolation "
            f"(first: {missing[0]}). Check for incomplete source coverage."
        )
    return result, clip_count


def interpolate_irradiance(
    source: pd.DataFrame,
    *,
    year: int,
    method: str = "pchip",
    max_irradiance_w_m2: float | None = DEFAULT_MAX_IRRADIANCE_W_M2,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Return a validated 15-minute irradiance table and per-column clip counts."""
    target_index = calendar_index(year)
    if source.index.min() > target_index[0]:
        raise ValueError(
            "The input does not cover the complete requested year. "
            "Do not silently pad missing periods with zero."
        )
    if (source.index < target_index[0]).any() or (source.index > target_index[-1]).any():
        raise ValueError("The input contains timestamps outside the requested year.")

    output = pd.DataFrame(index=target_index)
    clip_counts: dict[str, int] = {}
    for column in source.columns:
        output[column], clip_counts[column] = interpolate_series(
            source[column], target_index,
            method=method,
            max_irradiance_w_m2=max_irradiance_w_m2,
        )

    # Verify source preservation, except for values explicitly rejected by the
    # requested physical upper limit.
    expected_source = source.clip(lower=0.0, upper=max_irradiance_w_m2)
    if not np.allclose(output.loc[source.index, source.columns].to_numpy(), expected_source.to_numpy()):
        raise AssertionError("Source measurements changed outside the configured QC limit.")
    return output, clip_counts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=Path(config.getpath("paths", "heat_pump_solar_raw_csv")),
        help="Source irradiance CSV; defaults to paths.heat_pump_solar_raw_csv.",
    )
    parser.add_argument(
        "--output", type=Path, default=Path(config.getpath("paths", "heat_pump_solar_csv")),
        help="Destination 15-minute CSV; defaults to paths.heat_pump_solar_csv.",
    )
    parser.add_argument("--year", type=int, help="Calendar year; inferred when input covers one year.")
    parser.add_argument(
        "--method", choices=("pchip", "time"), default="pchip",
        help="Interpolation method; PCHIP is the recommended default.",
    )
    parser.add_argument(
        "--max-irradiance-w-m2", type=float, default=DEFAULT_MAX_IRRADIANCE_W_M2,
        help=("Upper quality-control limit. Default: 1400 W/m2. "
              "Pass a negative value to disable clipping."),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    source = read_irradiance_csv(args.input)
    year = args.year if args.year is not None else infer_year(source)
    upper_limit = None if args.max_irradiance_w_m2 < 0 else args.max_irradiance_w_m2
    if upper_limit is not None and upper_limit <= 0:
        raise ValueError("--max-irradiance-w-m2 must be positive, or negative to disable clipping.")

    output, clip_counts = interpolate_irradiance(
        source,
        year=year,
        method=args.method,
        max_irradiance_w_m2=upper_limit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, date_format="%Y-%m-%d %H:%M")

    print(f"Saved: {args.output} ({output.shape[0]} rows, {output.shape[1]} columns)")
    print(f"Method: {args.method}; upper limit: {upper_limit if upper_limit is not None else 'disabled'} W/m2")
    print("Upper-limit corrections: " + ", ".join(f"{col}={count}" for col, count in clip_counts.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
