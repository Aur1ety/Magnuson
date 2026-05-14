"""
feature_engineer.py
===================
Builds the full 14-feature matrix from aligned SWIS + MAG DataFrames.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import logging

logger = logging.getLogger("feature_engineer")

def compute_he_h_ratio(df: pd.DataFrame) -> pd.Series:
    he = df["he_flux"].rolling(window=3, min_periods=1, center=True).median()
    h  = df["h_flux"].rolling(window=3, min_periods=1, center=True).median()
    ratio = (he / h.replace(0, np.nan)).clip(0, 1.0)
    ratio.name = "he_h_ratio"
    return ratio

def swis_features(swis_df: pd.DataFrame) -> pd.DataFrame:
    feat = pd.DataFrame(index=swis_df.index)
    feat["vsw"]        = swis_df["vsw"]
    feat["np"]         = swis_df["np"]
    feat["tp"]         = np.log10(swis_df["tp"].clip(lower=1e3))
    feat["he_h_ratio"] = compute_he_h_ratio(swis_df)
    feat["beta_proxy"] = (swis_df["np"] * swis_df["tp"]) / (swis_df["vsw"] ** 2 + 1e-6)
    feat["dvsw_dt"]    = feat["vsw"].diff().fillna(0.0)
    return feat

def mag_features(mag_df: pd.DataFrame) -> pd.DataFrame:
    feat = pd.DataFrame(index=mag_df.index)
    bx, by, bz = mag_df["bx"], mag_df["by"], mag_df["bz"]
    
    feat["bz"] = bz
    feat["b_mag"] = mag_df.get("b_mag", np.sqrt(bx**2 + by**2 + bz**2))
    feat["clock_angle"] = np.degrees(np.arctan2(by, bz))
    feat["dbz_dt"] = bz.diff().fillna(0.0)

    # Vectorized B vector rotation rate
    bvec = np.column_stack([bx.values, by.values, bz.values])
    dot = np.sum(bvec[1:] * bvec[:-1], axis=1)
    norm = np.linalg.norm(bvec[1:], axis=1) * np.linalg.norm(bvec[:-1], axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos_angle = np.clip(dot / (norm + 1e-9), -1.0, 1.0)
    
    rotation_deg = np.nan_to_num(np.degrees(np.arccos(cos_angle)), nan=0.0)
    feat["b_rotation"] = pd.Series(np.concatenate([[0.0], rotation_deg]), index=mag_df.index).rolling(window=60, min_periods=1).sum()

    # Smoothed Bz
    bz_arr = bz.interpolate(method="linear", limit_direction="both").values
    bz_smooth = savgol_filter(bz_arr, window_length=121, polyorder=3)
    bz_smooth[bz.isna().values] = np.nan
    feat["bz_smoothed"] = bz_smooth

    # Vectorized Bz persistence
    southward = (bz < 0)
    feat["bz_persistence"] = southward.groupby((~southward).cumsum()).cumsum()

    # B elevation angle
    feat["b_elevation"] = np.degrees(np.arctan2(bz, np.sqrt(bx**2 + by**2)))
    return feat

def build_feature_matrix(swis_df: pd.DataFrame, mag_df: pd.DataFrame) -> pd.DataFrame:
    swis_feat = swis_features(swis_df)
    mag_feat  = mag_features(mag_df)

    combined = swis_feat.join(mag_feat, how="outer").ffill(limit=12).bfill(limit=12)
    combined.dropna(subset=["vsw", "np", "bz", "b_mag"], inplace=True)
    
    for col in combined.columns:
        if combined[col].isna().any():
            combined[col] = combined[col].fillna(combined[col].median())
            
    return combined

FEATURE_NAMES = [
    "vsw", "np", "tp", "he_h_ratio", "beta_proxy", "dvsw_dt",
    "bz", "b_mag", "clock_angle", "dbz_dt",
    "b_rotation", "bz_smoothed", "bz_persistence", "b_elevation"
]