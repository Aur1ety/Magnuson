# CME Detection with Aditya-L1 MAG — Full Implementation Guide

> **Blueprint for:** extending the existing SWIS-based TCN/TCAN pipeline with  
> MAG magnetometer data. Every code block here is meant to slot directly into  
> your existing repo structure. Pass this to Gemini section by section.

---

## Repo structure after this upgrade

```
cme-mag-detector/
├── data_pipeline.py        ← existing (DO NOT REWRITE — extend only)
├── mag_pipeline.py         ← NEW — MAG CDF ingestion + resampling
├── feature_engineer.py     ← NEW — merges SWIS + MAG, builds 14 features
├── label_events.py         ← NEW — ISRO catalog + DONKI cross-ref
├── model_factory.py        ← existing (extend TCN/TCAN, add BzLSTM)
├── ensemble.py             ← NEW — weighted combiner
├── inference_engine.py     ← extend existing CMEInferenceEngine
├── eval.py                 ← NEW — proper metrics suite
├── train.py                ← NEW — unified training script
└── saved_models/
    ├── tcn_v3.pth
    ├── tcan_v3.pth
    └── bzlstm_v1.pth
```

---

## Step 1 — `mag_pipeline.py` (MAG CDF ingestion)

**What it does:** Reads Aditya-L1 MAG Level-2 CDF files from PRADAN,  
extracts Bx/By/Bz/|B|, cleans sentinel values, then resamples from 1-second  
cadence down to SWIS timestamps using a 60-second median (not mean — median  
is more robust to spike artefacts in raw magnetometer data).

**Critical design decision:** Always resample MAG → SWIS timestamps.  
Never upsample SWIS to 1s — that creates synthetic data.

```python
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
```

**How to call it (in your training script):**
```python
from data_pipeline import load_cdf_directory
from mag_pipeline import load_mag_directory, resample_mag_to_swis

swis_raw = load_cdf_directory("/content/swis_data")
mag_raw  = load_mag_directory("/content/mag_data")
mag_aligned = resample_mag_to_swis(mag_raw, swis_index=swis_raw.index)
```

---

## Step 2 — `feature_engineer.py` (14-feature matrix)

**What it does:** Takes the aligned SWIS + MAG DataFrames and produces  
the full 14-feature matrix. Your existing `engineer_features()` produces 6  
features (Vsw, np, log10(Tp), He/H, beta_proxy, dVsw/dt) — this adds 8 more.

**The 14 features explained:**

| # | Feature | Source | Physical meaning |
|---|---------|--------|-----------------|
| 1 | `vsw` | SWIS | Solar wind bulk speed [km/s] |
| 2 | `np` | SWIS | Proton density [cm⁻³] |
| 3 | `tp` | SWIS | log₁₀ proton temperature [K] |
| 4 | `he_h_ratio` | SWIS | He/H flux ratio — CME ejecta tracer |
| 5 | `beta_proxy` | SWIS | Plasma pressure vs magnetic pressure |
| 6 | `dvsw_dt` | SWIS | Speed gradient — shock ramp detector |
| 7 | `bz` | MAG | North-south IMF [nT] — reconnection indicator |
| 8 | `b_mag` | MAG | Total field magnitude [nT] — sheath enhancement |
| 9 | `clock_angle` | MAG | arctan(By/Bz) — flux rope orientation |
| 10 | `dbz_dt` | MAG | Bz rate of change — sharp = shock arrival |
| 11 | `b_rotation` | MAG | Rolling angular rotation of B vector [deg/step] |
| 12 | `bz_smoothed` | MAG | Low-pass Bz — sustained southward field |
| 13 | `bz_persistence` | MAG | Consecutive minutes of southward Bz |
| 14 | `b_elevation` | MAG | arctan(Bz/√(Bx²+By²)) — field latitude angle |

```python
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
```

---

## Step 3 — `label_events.py` (event labelling)

**What it does:** Loads known CME/ICME event times and stamps each row  
in the feature matrix with a binary label. Uses ISRO event reports  
cross-referenced with the NASA DONKI catalog.

```python
# label_events.py
"""
CME event labelling from ISRO / DONKI catalogs.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np
import requests
import logging

logger = logging.getLogger("label_events")


def load_isro_catalog(csv_path: str) -> pd.DataFrame:
    """
    Load ISRO-format CME catalog CSV.
    Expected columns: start_time, end_time (ISO 8601 strings).
    """
    df = pd.read_csv(csv_path, parse_dates=["start_time", "end_time"])
    logger.info("ISRO catalog: %d events", len(df))
    return df


def fetch_donki_cmes(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch CME arrival times from NASA DONKI API.
    Returns DataFrame with columns: start_time, end_time.
    Assumes 48-hour window for each event (adjust from catalog data).
    """
    url = (
        "https://kauai.ccmc.gsfc.nasa.gov/DONKI/WS/get/CMEAnalysis"
        f"?startDate={start_date}&endDate={end_date}&catalog=ALL"
    )
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        events = r.json()
        rows = []
        for e in events:
            if e.get("time21_5") is None:
                continue
            t_start = pd.to_datetime(e["time21_5"])
            rows.append({
                "start_time": t_start,
                "end_time":   t_start + pd.Timedelta(hours=48),
                "speed_kms":  e.get("speed", np.nan),
                "source":     "DONKI",
            })
        logger.info("DONKI: %d CME events fetched", len(rows))
        return pd.DataFrame(rows)
    except Exception as ex:
        logger.warning("DONKI fetch failed: %s — returning empty catalog", ex)
        return pd.DataFrame(columns=["start_time", "end_time"])


def attach_labels(
    feat_df: pd.DataFrame,
    isro_csv: str | None = None,
    donki_start: str | None = None,
    donki_end:   str | None = None,
    lead_hours: int = 2,
) -> pd.DataFrame:
    """
    Stamp CME labels onto feature DataFrame.

    lead_hours: extend label window backwards by this many hours.
                Models the precursor signature that appears before
                the main CME front arrives (useful for early warning).
    """
    df = feat_df.copy()
    df["label"] = 0

    catalogs = []
    if isro_csv and Path(isro_csv).exists():
        catalogs.append(load_isro_catalog(isro_csv))
    if donki_start and donki_end:
        catalogs.append(fetch_donki_cmes(donki_start, donki_end))

    if not catalogs:
        logger.warning("No event catalogs provided — all labels = 0")
        return df

    all_events = pd.concat(catalogs, ignore_index=True)
    lead = pd.Timedelta(hours=lead_hours)

    for _, row in all_events.iterrows():
        mask = (
            (df.index >= row["start_time"] - lead) &
            (df.index <= row["end_time"])
        )
        df.loc[mask, "label"] = 1

    pos_frac = df["label"].mean()
    logger.info(
        "Labels attached: %.2f%% positive (CME)", pos_frac * 100
    )
    return df
```

---

## Step 4 — `model_factory.py` extensions

**What changes:** Your existing TCN and TCAN take `input_dim=8`. Change that  
to `input_dim=14`. Add the new BzLSTM class below the existing models.  
No other changes to model architecture are needed.

```python
# Add to existing model_factory.py — do NOT remove TCN / TCAN

import torch
import torch.nn as nn


class BzLSTM(nn.Module):
    """
    LSTM specialised on the Bz rotation signature of CME flux ropes.
    Uses a longer lookback window than the TCN — the flux rope rotation
    takes 6-12 hours and is a slow, sustained signal that the TCN's
    convolutional receptive field may miss.

    Input:  (batch, seq_len, 14)  — same feature matrix as TCN/TCAN
    Output: (batch, 1)            — P(CME) logit

    Architecture rationale:
    - Two LSTM layers capture long-range temporal dependencies
    - Bz-attention: a learned weight vector emphasises Bz, bz_smoothed,
      bz_persistence columns so the LSTM focuses on the magnetic signal
    - Dropout between layers prevents overfitting on the small CME dataset
    """

    def __init__(
        self,
        input_dim:   int = 14,
        hidden_dim:  int = 128,
        num_layers:  int = 2,
        dropout:     float = 0.3,
        bz_col_indices: list[int] | None = None,
    ):
        super().__init__()
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim

        # Column indices of the MAG-derived Bz features in the feature matrix
        # Default: indices 6,11,12 = bz, bz_smoothed, bz_persistence
        self.bz_cols = bz_col_indices or [6, 11, 12]

        # Input projection with Bz emphasis:
        # double the weight of Bz-related features before the LSTM
        self.input_proj = nn.Linear(input_dim, input_dim)

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, features)

        # Bz emphasis: scale up the Bz-family columns
        x_boosted = x.clone()
        x_boosted[:, :, self.bz_cols] *= 2.0   # learnable alternative: use a mask

        x_proj = self.input_proj(x_boosted)     # (batch, seq_len, features)

        _, (h_n, _) = self.lstm(x_proj)         # h_n: (num_layers, batch, hidden)
        last_hidden  = h_n[-1]                  # (batch, hidden) — top LSTM layer

        logit = self.classifier(last_hidden)    # (batch, 1)
        return logit.squeeze(-1)                # (batch,)


# ── Update input_dim for existing models ────────────────────────────────────
# In your existing TCNModel and TCANModel __init__:
#   Change:  input_dim: int = 8
#   To:      input_dim: int = 14
# Everything else stays the same — the convolutions are feature-agnostic.
```

---

## Step 5 — `ensemble.py`

**What it does:** Loads the three saved models and combines their  
probabilities into a single P(CME) score. Weights are tunable.

```python
# ensemble.py
"""
Weighted ensemble of TCN, TCAN, and BzLSTM.
"""
from __future__ import annotations

import torch
import numpy as np
from pathlib import Path


class CMEEnsemble:
    """
    Weights (default):
      TCN    0.50 — best at plasma shock detection (fast transients)
      TCAN   0.30 — long-range precursor via self-attention
      BzLSTM 0.20 — flux rope Bz rotation over hours

    Tune weights using validation F1 per model on your holdout set.
    """

    WEIGHTS = {"tcn": 0.50, "tcan": 0.30, "bzlstm": 0.20}

    def __init__(
        self,
        tcn_path:    str,
        tcan_path:   str,
        bzlstm_path: str,
        device: str = "cpu",
    ):
        from model_factory import TCNModel, TCANModel, BzLSTM  # your existing imports

        self.device = torch.device(device)

        self.tcn    = self._load(TCNModel(input_dim=14),    tcn_path)
        self.tcan   = self._load(TCANModel(input_dim=14),   tcan_path)
        self.bzlstm = self._load(BzLSTM(input_dim=14),      bzlstm_path)

    def _load(self, model, path):
        if Path(path).exists():
            model.load_state_dict(torch.load(path, map_location=self.device))
        model.to(self.device).eval()
        return model

    @torch.no_grad()
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        x : np.ndarray of shape (N, seq_len, 14)

        Returns
        -------
        np.ndarray of shape (N,) — P(CME) in [0, 1]
        """
        t = torch.tensor(x, dtype=torch.float32, device=self.device)

        p_tcn    = torch.sigmoid(self.tcn(t)).cpu().numpy()
        p_tcan   = torch.sigmoid(self.tcan(t)).cpu().numpy()
        p_bzlstm = torch.sigmoid(self.bzlstm(t)).cpu().numpy()

        w = self.WEIGHTS
        p_ensemble = (
            w["tcn"]    * p_tcn  +
            w["tcan"]   * p_tcan +
            w["bzlstm"] * p_bzlstm
        )
        return p_ensemble

    def predict_latest(self, x: np.ndarray, threshold: float = 0.5) -> dict:
        """Convenience wrapper for inference on the most recent window."""
        p = self.predict(x[-1:])
        score = float(p[0])
        return {
            "p_cme":    score,
            "verdict":  "CME DETECTED" if score >= threshold else "QUIET",
            "threshold": threshold,
        }
```

---

## Step 6 — `train.py` (unified training script)

**What it does:** End-to-end training — loads data, builds features,  
trains all three models with the same loop, saves checkpoints.

```python
# train.py
"""
Unified training script for TCN v3 + TCAN v3 + BzLSTM.
Run: python train.py --swis_dir /content/swis_data --mag_dir /content/mag_data
"""
from __future__ import annotations

import argparse
import logging
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from data_pipeline  import load_cdf_directory, clean_and_impute, build_sequences, split_and_scale
from mag_pipeline   import load_mag_directory, resample_mag_to_swis
from feature_engineer import build_feature_matrix
from label_events   import attach_labels
from model_factory  import TCNModel, TCANModel, BzLSTM

logger = logging.getLogger("train")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def focal_loss(pred, target, alpha=0.75, gamma=2.0):
    """
    Focal loss — down-weights easy negatives (quiet solar wind).
    Critical for CME detection where positives are ~5% of data.
    alpha  : weight for the positive class (CME)
    gamma  : focusing parameter — higher = more focus on hard examples
    """
    bce  = nn.functional.binary_cross_entropy_with_logits(pred, target, reduction="none")
    pt   = torch.exp(-bce)
    loss = alpha * (1 - pt) ** gamma * bce
    return loss.mean()


def train_one_model(model, X_train, y_train, X_val, y_val, epochs=50, lr=1e-3):
    """Generic training loop — same for all three models."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    )
    loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = focal_loss(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        # Validation
        model.eval()
        with torch.no_grad():
            xv = torch.tensor(X_val, dtype=torch.float32, device=device)
            yv = torch.tensor(y_val, dtype=torch.float32, device=device)
            val_loss = focal_loss(model(xv), yv).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0:
            logger.info("Epoch %3d | val_loss=%.4f", epoch, val_loss)

    model.load_state_dict(best_state)
    return model


def main(args):
    # 1. Load raw data
    swis_raw = load_cdf_directory(args.swis_dir)
    mag_raw  = load_mag_directory(args.mag_dir)
    mag_aligned = resample_mag_to_swis(mag_raw, swis_index=swis_raw.index)

    # 2. Build 14-feature matrix
    feat_df = build_feature_matrix(swis_raw, mag_aligned)

    # 3. Attach labels
    feat_df = attach_labels(
        feat_df,
        isro_csv=args.label_csv,
        donki_start=args.start_date,
        donki_end=args.end_date,
    )

    # 4. Clean
    feat_df = clean_and_impute(feat_df)

    # 5. Build sequences + split
    feature_cols = [c for c in feat_df.columns if c != "label"]
    X, y = build_sequences(feat_df, feature_cols=feature_cols)
    splits = split_and_scale(X, y, scaler_path=args.scaler_path)

    Xtr, ytr = splits["X_train"], splits["y_train"]
    Xv,  yv  = splits["X_val"],   splits["y_val"]

    # 6. Train all three models
    models = {
        "tcn":    TCNModel(input_dim=14),
        "tcan":   TCANModel(input_dim=14),
        "bzlstm": BzLSTM(input_dim=14),
    }

    for name, model in models.items():
        logger.info("=== Training %s ===", name.upper())
        trained = train_one_model(model, Xtr, ytr, Xv, yv, epochs=args.epochs)
        save_path = f"saved_models/{name}_v3.pth"
        torch.save(trained.state_dict(), save_path)
        logger.info("Saved → %s", save_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--swis_dir",    default="/content/swis_data")
    p.add_argument("--mag_dir",     default="/content/mag_data")
    p.add_argument("--label_csv",   default=None)
    p.add_argument("--start_date",  default="2024-05-01")
    p.add_argument("--end_date",    default="2026-05-01")
    p.add_argument("--scaler_path", default="scalers/scaler_14feat.pkl")
    p.add_argument("--epochs",      type=int, default=50)
    main(p.parse_args())
```

---

## Step 7 — `eval.py` (evaluation metrics)

**What it does:** Goes beyond simple F1. For space weather you care about  
False Alarm Rate (FAR) and lead time — how early did you flag the CME?

```python
# eval.py
"""
CME detection evaluation metrics.
Beyond F1: FAR, probability of detection (POD), lead time bias.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix
)


def evaluate(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """
    Full evaluation suite for CME binary detection.

    Space weather metrics:
      POD  = Probability of Detection = recall (hit rate)
      FAR  = False Alarm Rate = FP / (FP + TN)  [space weather definition]
      CSI  = Critical Success Index = TP / (TP + FP + FN)
      HSS  = Heidke Skill Score — measures skill vs random chance
    """
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    pod  = tp / (tp + fn + 1e-9)                        # recall
    far  = fp / (fp + tn + 1e-9)                        # space weather FAR
    csi  = tp / (tp + fp + fn + 1e-9)
    hss_num = 2 * (tp * tn - fp * fn)
    hss_den = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    hss  = hss_num / (hss_den + 1e-9)

    return {
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall_pod": pod,
        "far":       far,
        "csi":       csi,
        "hss":       hss,
        "auc_roc":   roc_auc_score(y_true, y_prob) if y_true.sum() > 0 else np.nan,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def print_report(metrics: dict):
    print("\n── CME Detection Evaluation ──────────────────")
    print(f"  F1 Score  : {metrics['f1']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  POD       : {metrics['recall_pod']:.4f}  (probability of detection)")
    print(f"  FAR       : {metrics['far']:.4f}  (false alarm rate)")
    print(f"  CSI       : {metrics['csi']:.4f}  (critical success index)")
    print(f"  HSS       : {metrics['hss']:.4f}  (Heidke skill score)")
    print(f"  AUC-ROC   : {metrics['auc_roc']:.4f}")
    print(f"  Confusion : TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']} TN={metrics['tn']}")
    print("──────────────────────────────────────────────\n")
```

---

## Step 8 — Putting it all together (quick sanity check)

Run this in Colab to verify the full pipeline before handing off to Gemini for gap-filling:

```python
# sanity_check.py — run this first to make sure everything connects

from data_pipeline    import load_cdf_directory
from mag_pipeline     import load_mag_directory, resample_mag_to_swis
from feature_engineer import build_feature_matrix, FEATURE_NAMES
from label_events     import attach_labels
from data_pipeline    import build_sequences, split_and_scale

# With real data:
# swis_raw = load_cdf_directory("/content/swis_data")
# mag_raw  = load_mag_directory("/content/mag_data")

# With synthetic data for pipeline testing:
import pandas as pd
import numpy as np

idx = pd.date_range("2024-05-01", periods=10_000, freq="1min")
rng = np.random.default_rng(0)

swis_raw = pd.DataFrame({
    "vsw": 400 + rng.normal(0, 30, len(idx)),
    "np":  6   + rng.normal(0, 2,  len(idx)),
    "tp":  1e5 * np.exp(rng.normal(0, 0.2, len(idx))),
    "he_flux": 0.3 + rng.normal(0, 0.05, len(idx)),
    "h_flux":  6   + rng.normal(0, 2,    len(idx)),
    "vth": rng.normal(50, 5, len(idx)),
}, index=idx)

mag_raw = pd.DataFrame({
    "bx": rng.normal(0, 5, len(idx)),
    "by": rng.normal(0, 5, len(idx)),
    "bz": rng.normal(0, 5, len(idx)),
    "b_mag": 7 + rng.normal(0, 2, len(idx)),
}, index=idx)

mag_aligned = resample_mag_to_swis(mag_raw, swis_index=swis_raw.index)
feat_df     = build_feature_matrix(swis_raw, mag_aligned)
feat_df["label"] = 0  # skip labelling for sanity check

X, y = build_sequences(feat_df, feature_cols=FEATURE_NAMES)
print(f"X shape: {X.shape}")   # should be (N, 128, 14)
print(f"y shape: {y.shape}")   # should be (N,)
print("All 14 features present:", X.shape[2] == 14)
```

---

## Key gotchas for Gemini to handle

When you hand each section to Gemini, include these notes:

**mag_pipeline.py**
- The actual MAG CDF variable names from PRADAN may differ — check `cdf.cdf_info()["zVariables"]` to list what's actually in your file before hardcoding names.
- Magnetometer data from Aditya-L1 is in RTN (Radial-Tangential-Normal) frame. Bz in RTN is not the same as GSM Bz that space weather forecasters use, but for an in-situ detector at L1 the RTN Bz is still the relevant quantity.

**feature_engineer.py**
- `bz_persistence` uses a Python loop and will be slow for large datasets. Ask Gemini to vectorise it with `groupby` + cumsum tricks.
- The 2× Bz boost in BzLSTM is a hard-coded heuristic. Replace it with a learned attention mask for better results.

**train.py**
- Class imbalance: CMEs are ~5% of data. Focal loss handles this but you may also want to oversample CME windows (`imbalanced-learn` SMOTE) if F1 stays below 0.4.
- Sequence length 128 at 1-min cadence = ~2 hours lookback for TCN/TCAN. BzLSTM should use 512 (8 hours) since flux rope rotation takes that long.

**eval.py**
- The space weather definition of FAR is `FP / (FP + TN)`, not the ML definition `FP / (FP + TP)`. Make sure Gemini uses the right one.
