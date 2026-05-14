# feature_engineer.py
"""
Builds the full 14-feature matrix from aligned SWIS + MAG DataFrames.
Drop-in replacement for the engineer_features() call in data_pipeline.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

# ── Existing SWIS features (copied from data_pipeline.py) ──────────────────

def compute_he_h_ratio(df: pd.DataFrame) -> pd.Series:
    he = df["he_flux"].rolling(window=3, min_periods=1, center=True).median()
    h  = df["h_flux"].rolling(window=3, min_periods=1, center=True).median()
    ratio = (he / h.replace(0, np.nan)).clip(0, 1.0)
    ratio.name = "he_h_ratio"
    return ratio


def swis_features(swis_df: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the 6 features from the existing engineer_features()."""
    feat = pd.DataFrame(index=swis_df.index)
    feat["vsw"]        = swis_df["vsw"]
    feat["np"]         = swis_df["np"]
    feat["tp"]         = np.log10(swis_df["tp"].clip(lower=1e3))
    feat["he_h_ratio"] = compute_he_h_ratio(swis_df)
    feat["beta_proxy"] = (swis_df["np"] * swis_df["tp"]) / (swis_df["vsw"] ** 2 + 1e-6)
    feat["dvsw_dt"]    = feat["vsw"].diff().fillna(0.0)
    return feat


# ── New MAG features ────────────────────────────────────────────────────────

def mag_features(mag_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build 8 magnetometer-derived features from aligned MAG data.
    Input columns expected: bx, by, bz, b_mag (all in nT, RTN frame).
    """
    feat = pd.DataFrame(index=mag_df.index)

    bx = mag_df["bx"]
    by = mag_df["by"]
    bz = mag_df["bz"]
    bt = mag_df["b_mag"]

    # 1. Raw Bz — the single most important CME predictor.
    #    Sustained negative Bz means southward IMF → magnetic reconnection
    #    with Earth's magnetosphere → geomagnetic storm.
    feat["bz"] = bz

    # 2. Total field magnitude.
    #    CME sheath: field compression → |B| enhancement.
    #    Magnetic cloud (flux rope core): smooth |B| with slow rotation.
    feat["b_mag"] = bt

    # 3. Clock angle = arctan2(By, Bz) in degrees [-180, 180].
    #    A clock angle near ±180° means Bz is strongly southward.
    #    Rotating clock angle over hours = magnetic flux rope passage.
    feat["clock_angle"] = np.degrees(np.arctan2(by, bz))

    # 4. dBz/dt — rate of change of Bz.
    #    Sharp spike: shock front arrival (abrupt field compression).
    #    Slow drift: flux rope rotation (gradual field rotation over hours).
    feat["dbz_dt"] = bz.diff().fillna(0.0)

    # 5. B vector rotation rate.
    #    At each step, compute the angle between the B vector now and
    #    the B vector one step ago. Rolling sum gives total rotation over
    #    a window — the signature of a passing flux rope.
    #    Formula: angle = arccos(dot(B_t, B_{t-1}) / (|B_t| * |B_{t-1}|))
    bvec = np.column_stack([bx.values, by.values, bz.values])
    dot   = np.sum(bvec[1:] * bvec[:-1], axis=1)
    norm  = np.linalg.norm(bvec[1:], axis=1) * np.linalg.norm(bvec[:-1], axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos_angle = np.clip(dot / (norm + 1e-9), -1, 1)
    rotation_deg = np.degrees(np.arccos(cos_angle))
    rotation_deg = np.concatenate([[0.0], rotation_deg])
    # Rolling 60-step sum ≈ rotation accumulated over last ~1 hour
    feat["b_rotation"] = (
        pd.Series(rotation_deg, index=mag_df.index)
        .rolling(window=60, min_periods=1)
        .sum()
    )

    # 6. Smoothed Bz (low-pass, 121-step Savitzky-Golay ≈ 2-hour window).
    #    CME flux ropes show sustained southward Bz for hours.
    #    Single-step Bz is noisy — the smoothed version separates flux rope
    #    from transient fluctuations and Alfvén waves.
    bz_arr = bz.interpolate(method="linear", limit_direction="both").values
    bz_smooth = savgol_filter(bz_arr, window_length=121, polyorder=3)
    bz_smooth[bz.isna().values] = np.nan
    feat["bz_smoothed"] = bz_smooth

    # 7. Bz persistence — consecutive southward (negative Bz) steps.
    #    A single southward dip is noise. 30+ consecutive southward steps
    #    (~30 min at 1-min cadence) is a real CME magnetic cloud signature.
    #    Implementation: reset counter to 0 whenever Bz > 0.
    southward = (bz < 0).astype(int)
    persistence = []
    count = 0
    for s in southward:
        count = (count + 1) * s   # resets to 0 on northward
        persistence.append(count)
    feat["bz_persistence"] = persistence

    # 8. B elevation angle = arctan(Bz / sqrt(Bx² + By²)) in degrees.
    #    +90° = field points north (away from ecliptic).
    #    -90° = field points south (into ecliptic) = geoeffective.
    #    Captures the 3D orientation of the field, complementing clock angle.
    bxy = np.sqrt(bx**2 + by**2)
    feat["b_elevation"] = np.degrees(np.arctan2(bz, bxy))

    return feat


# ── Master feature builder ──────────────────────────────────────────────────

def build_feature_matrix(
    swis_df: pd.DataFrame,
    mag_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge SWIS and MAG features into the full 14-feature matrix.
    Both DataFrames must share the same DatetimeIndex (after mag resampling).

    Returns
    -------
    DataFrame with 14 columns, same index as input.
    """
    swis_feat = swis_features(swis_df)
    mag_feat  = mag_features(mag_df)

    # Outer join, then forward-fill short gaps, then drop persistent NaN rows
    combined = swis_feat.join(mag_feat, how="outer")
    combined = combined.ffill(limit=12).bfill(limit=12)

    # Critical columns: drop rows where these are still NaN
    critical = ["vsw", "np", "bz", "b_mag"]
    before = len(combined)
    combined.dropna(subset=critical, inplace=True)
    if len(combined) < before:
        print(f"Dropped {before - len(combined)} rows with persistent NaN")

    # Fill remaining non-critical NaN with column median
    for col in combined.columns:
        if combined[col].isna().any():
            combined[col] = combined[col].fillna(combined[col].median())

    print(f"Feature matrix: {combined.shape} | columns: {list(combined.columns)}")
    return combined


FEATURE_NAMES = [
    "vsw", "np", "tp", "he_h_ratio", "beta_proxy", "dvsw_dt",   # SWIS (6)
    "bz", "b_mag", "clock_angle", "dbz_dt",                       # MAG basic (4)
    "b_rotation", "bz_smoothed", "bz_persistence", "b_elevation", # MAG derived (4)
]