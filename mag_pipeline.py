# mag_pipeline.py
"""
Aditya-L1 MAG Level-2 CDF ingestion pipeline.
Resamples 1-second vector magnetic field data to SWIS cadence.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    import cdflib
    HAS_CDFLIB = True
except ImportError:
    HAS_CDFLIB = False

logger = logging.getLogger("mag_pipeline")

SENTINEL_THRESHOLD = -9e30

# MAG L2 CDF variable names (verify against your actual PRADAN files —
# ISRO may use different variable names across data versions)
MAG_VARMAP = {
    "epoch": "Epoch",          # CDF epoch in TT2000
    "bx":    "BX_RTN",         # Radial-Tangential-Normal frame X [nT]
    "by":    "BY_RTN",         # RTN Y [nT]
    "bz":    "BZ_RTN",         # RTN Z [nT] ← most important for CME
    "b_mag": "BT",             # Total field magnitude sqrt(Bx²+By²+Bz²) [nT]
}
# NOTE: If PRADAN uses GSE frame instead of RTN, rename accordingly.
# For CME detection Bz sign convention: negative Bz = southward = geoeffective.


def parse_mag_cdf(filepath: Path) -> pd.DataFrame:
    """
    Parse a single MAG L2 CDF file → DataFrame at native 1s cadence.
    Sentinel values replaced with NaN.
    """
    if not HAS_CDFLIB:
        raise ImportError("pip install cdflib")

    cdf = cdflib.CDF(str(filepath))
    data: dict[str, np.ndarray] = {}

    raw_epoch = cdf.varget(MAG_VARMAP["epoch"])
    data["time"] = pd.to_datetime(cdflib.cdfepoch.to_datetime(raw_epoch))

    for key, varname in MAG_VARMAP.items():
        if key == "epoch":
            continue
        try:
            arr = cdf.varget(varname).astype(np.float64)
            arr[arr < SENTINEL_THRESHOLD] = np.nan
            data[key] = arr
        except Exception as e:
            logger.warning("MAG var '%s' missing in %s: %s", varname, filepath.name, e)
            data[key] = np.full(len(data["time"]), np.nan)

    df = pd.DataFrame(data).set_index("time").sort_index()
    logger.info("MAG parsed: %s → %d rows at 1s cadence", filepath.name, len(df))
    return df


def load_mag_directory(directory: str) -> pd.DataFrame:
    """Load + concatenate all MAG CDF files in a directory."""
    files = sorted(Path(directory).glob("*.cdf"))
    if not files:
        raise FileNotFoundError(f"No MAG CDF files in: {directory}")
    frames = [parse_mag_cdf(f) for f in files]
    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    logger.info("MAG total: %d rows from %d files", len(df), len(files))
    return df


def resample_mag_to_swis(
    mag_df: pd.DataFrame,
    swis_index: pd.DatetimeIndex,
    window_seconds: int = 60,
) -> pd.DataFrame:
    """
    Downsample MAG from 1s cadence to SWIS timestamps.

    Method: for each SWIS timestamp t, take the median of all MAG samples
    in [t - window/2, t + window/2]. Median is preferred over mean because
    single-point MAG spikes (instrument artefacts) are common and median
    suppresses them without requiring a separate spike-removal pass.

    Parameters
    ----------
    mag_df       : raw MAG DataFrame at 1s cadence
    swis_index   : DatetimeIndex of SWIS observations to align to
    window_seconds : half-window in seconds for aggregation

    Returns
    -------
    DataFrame indexed on swis_index with MAG columns
    """
    half = pd.Timedelta(seconds=window_seconds // 2)
    records = []

    for t in swis_index:
        window = mag_df.loc[t - half : t + half]
        if window.empty:
            records.append({col: np.nan for col in mag_df.columns})
        else:
            records.append(window.median().to_dict())

    resampled = pd.DataFrame(records, index=swis_index)
    nan_frac = resampled.isna().mean().mean()
    logger.info(
        "MAG resampled to %d SWIS steps | NaN fraction: %.1f%%",
        len(resampled), nan_frac * 100
    )
    return resampled