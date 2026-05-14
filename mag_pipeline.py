"""
mag_pipeline.py
===============
Aditya-L1 MAG Level-2 NetCDF ingestion pipeline.
Reads .nc files from PRADAN, extracts GSM magnetic field components,
and resamples from 10-second cadence to match SWIS timestamps.

Variable map (from L2_AL1_MAG_YYYYMMDD_V00.nc):
    time     : float64 unix-like timestamps
    Bx_gsm   : X component in GSM frame [nT]
    By_gsm   : Y component in GSM frame [nT]
    Bz_gsm   : Z component in GSM frame [nT] ← key CME indicator
    Bt       : total field magnitude (computed if not present)

Fill value: -9999.0 (from file attributes) → replaced with NaN.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger("mag_pipeline")

FILL_VALUE = -9999.0
FILL_THRESHOLD = -9000.0  # anything below this is treated as fill

MAG_VARS = {
    "time": "time",
    "bx":   "Bx_gsm",
    "by":   "By_gsm",
    "bz":   "Bz_gsm",
}

def parse_mag_nc(filepath: Path) -> pd.DataFrame:
    ds = xr.open_dataset(str(filepath))

    # PRADAN stores time as seconds since 1970-01-01 (Unix epoch)
    raw_time = ds["time"].values.astype(np.float64)
    timestamps = pd.to_datetime(raw_time, unit="s", utc=True).tz_localize(None)

    data = {"time": timestamps}

    for key, varname in MAG_VARS.items():
        if key == "time":
            continue
        try:
            arr = ds[varname].values.astype(np.float64)
            arr[arr < FILL_THRESHOLD] = np.nan
            data[key] = arr
        except KeyError:
            logger.warning("Variable '%s' not found in %s", varname, filepath.name)
            data[key] = np.full(len(timestamps), np.nan)

    ds.close()

    df = pd.DataFrame(data).set_index("time").sort_index()

    # Compute total field magnitude from components
    df["b_mag"] = np.sqrt(df["bx"]**2 + df["by"]**2 + df["bz"]**2)

    logger.info(
        "MAG parsed: %s → %d rows | Bz range: [%.1f, %.1f] nT",
        filepath.name, len(df),
        df["bz"].min(), df["bz"].max()
    )
    return df

def load_mag_directory(directory: str) -> pd.DataFrame:
    all_files = sorted(Path(directory).glob("*.nc"))

    # Only use L2 files
    l2_files = [f for f in all_files if f.name.startswith("L2_")]
    skipped  = len(all_files) - len(l2_files)

    if skipped > 0:
        logger.warning("Skipped %d non-L2 files (use L2_AL1_MAG_* only)", skipped)
    if not l2_files:
        raise FileNotFoundError(f"No L2 MAG .nc files found in: {directory}")

    frames = [parse_mag_nc(f) for f in l2_files]
    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="first")]

    logger.info(
        "MAG total: %d rows from %d files | date range: %s → %s",
        len(df), len(l2_files),
        df.index.min().date(), df.index.max().date()
    )
    return df

def resample_mag_to_swis(
    mag_df: pd.DataFrame,
    swis_index: pd.DatetimeIndex,
    window_seconds: int = 60,
) -> pd.DataFrame:
    """
    Downsample MAG from 10s cadence to SWIS timestamps using vectorized operations.
    """
    # 1. Calculate a rolling median over the time window.
    # We use a time-aware string (e.g., '60s') centered on the timestamp.
    window_str = f"{window_seconds}s"
    rolling_median = mag_df.rolling(window_str, center=True).median()

    # 2. Reindex to exactly match the SWIS timestamps.
    # We grab the nearest calculated median, with a strict tolerance so we
    # don't accidentally map a median to a timestamp hours away during a data gap.
    tolerance = pd.Timedelta(seconds=window_seconds // 2)
    resampled = rolling_median.reindex(swis_index, method='nearest', tolerance=tolerance)

    nan_frac = resampled.isna().mean().mean()
    logger.info(
        "MAG resampled to %d SWIS steps | NaN fraction: %.1f%%",
        len(resampled), nan_frac * 100
    )

    if nan_frac > 0.3:
        logger.warning(
            "High NaN fraction (%.1f%%) after resampling — "
            "check that MAG and SWIS date ranges overlap",
            nan_frac * 100
        )

    return resampled

if __name__ == "__main__":
    # Quick local test — point at your downloaded L2 files
    import sys
    directory = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\ponpo\Downloads"

    df = load_mag_directory(directory)
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nBz stats:\n{df['bz'].describe()}")