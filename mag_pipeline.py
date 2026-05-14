"""
mag_pipeline.py
===============
Aditya-L1 MAG Level-2 NetCDF ingestion pipeline.
Reads .nc files from PRADAN, extracts GSM magnetic field components,
and resamples from 10-second cadence to match SWIS timestamps.
"""
from __future__ import annotations
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger("mag_pipeline")

FILL_THRESHOLD = -9000.0  # anything below this is treated as fill

MAG_VARS = {
    "time": "time",
    "bx":   "Bx_gsm",
    "by":   "By_gsm",
    "bz":   "Bz_gsm",
}

def parse_mag_nc(filepath: Path) -> pd.DataFrame:
    ds = xr.open_dataset(str(filepath))
    raw_time = ds["time"].values.astype(np.float64)
    timestamps = pd.to_datetime(raw_time, unit="s", utc=True).tz_localize(None)
    data = {"time": timestamps}

    for key, varname in MAG_VARS.items():
        if key == "time": continue
        try:
            arr = ds[varname].values.astype(np.float64)
            arr[arr < FILL_THRESHOLD] = np.nan
            data[key] = arr
        except KeyError:
            data[key] = np.full(len(timestamps), np.nan)

    ds.close()
    df = pd.DataFrame(data).set_index("time").sort_index()
    df["b_mag"] = np.sqrt(df["bx"]**2 + df["by"]**2 + df["bz"]**2)
    return df

def load_mag_directory(directory: str) -> pd.DataFrame:
    all_files = sorted(Path(directory).glob("*.nc"))
    l2_files = [f for f in all_files if f.name.startswith("L2_")]
    if not l2_files:
        raise FileNotFoundError(f"No L2 MAG .nc files found in: {directory}")

    frames = [parse_mag_nc(f) for f in l2_files]
    df = pd.concat(frames).sort_index()
    return df[~df.index.duplicated(keep="first")]

def resample_mag_to_swis(mag_df: pd.DataFrame, swis_index: pd.DatetimeIndex, window_seconds: int = 60) -> pd.DataFrame:
    """Downsample MAG from 10s cadence to SWIS timestamps using fast vectorized operations."""
    window_str = f"{window_seconds}s"
    rolling_median = mag_df.rolling(window_str, center=True).median()
    
    tolerance = pd.Timedelta(seconds=window_seconds // 2)
    resampled = rolling_median.reindex(swis_index, method='nearest', tolerance=tolerance)
    
    logger.info("MAG resampled to %d SWIS steps | NaN fraction: %.1f%%", len(resampled), resampled.isna().mean().mean() * 100)
    return resampled