import os
import logging
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

try:
    import cdflib
    HAS_CDFLIB = True
except ImportError:
    HAS_CDFLIB = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
logger = logging.getLogger("data_pipeline")

SENTINEL_THRESHOLD = -9e30
CDF_VARMAP = {
    "epoch": "Epoch", "vsw": "Proton_Speed", "np": "Proton_Density",
    "tp": "Proton_Temperature", "he_flux": "He_Flux", "h_flux": "H_Flux"
}

SAVGOL_WINDOW, SAVGOL_POLYORD = 11, 3
SEQUENCE_LENGTH, STRIDE = 128, 16

def parse_cdf(filepath: Path) -> pd.DataFrame:
    if not HAS_CDFLIB: raise ImportError("cdflib required")
    cdf = cdflib.CDF(str(filepath))
    data = {"time": cdflib.cdfepoch.to_datetime(cdf.varget(CDF_VARMAP["epoch"]), to_np=True)}
    for key, varname in CDF_VARMAP.items():
        if key == "epoch": continue
        try:
            arr = cdf.varget(varname).astype(np.float64)
            arr[arr < SENTINEL_THRESHOLD] = np.nan
            data[key] = arr
        except: data[key] = np.full(len(data["time"]), np.nan)
    return pd.DataFrame(data).set_index("time").sort_index()

def load_cdf_directory(directory: str) -> pd.DataFrame:
    # RECURSIVE FIX: Digs into Kaggle subfolders
    files = sorted(Path(directory).rglob("*.cdf"))
    if not files: raise FileNotFoundError(f"No CDF files in: {directory}")
    df = pd.concat([parse_cdf(f) for f in files]).sort_index()
    return df[~df.index.duplicated(keep="first")]

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    feat = pd.DataFrame(index=df.index)
    feat["vsw"], feat["np"] = df["vsw"], df["np"]
    feat["tp"] = np.log10(df["tp"].clip(lower=1e3))
    # He/H Ratio - Critical CME Indicator
    he, h = df["he_flux"].rolling(3, center=True).median(), df["h_flux"].rolling(3, center=True).median()
    feat["he_h_ratio"] = (he / h.replace(0, np.nan)).clip(0, 1.0)
    feat["beta_proxy"] = (df["np"] * df["tp"]) / (df["vsw"] ** 2 + 1e-6)
    feat["dvsw_dt"] = feat["vsw"].diff().fillna(0.0)
    return feat

def apply_savgol(df: pd.DataFrame) -> pd.DataFrame:
    smoothed = df.copy()
    for col in [c for c in df.columns if c not in ["dvsw_dt", "he_h_ratio"]]:
        valid = df[col].notna()
        if valid.sum() < SAVGOL_WINDOW: continue
        arr = pd.Series(df[col].values).interpolate(method="linear", limit_direction="both").values
        sm = savgol_filter(arr, window_length=SAVGOL_WINDOW, polyorder=SAVGOL_POLYORD)
        sm[~valid.values] = np.nan
        smoothed[col] = sm
    return smoothed

def attach_labels(df: pd.DataFrame, label_csv: str | None = None) -> pd.DataFrame:
    df["label"] = 0
    if label_csv and Path(label_csv).exists():
        ev = pd.read_csv(label_csv, parse_dates=["start_time", "end_time"])
        for _, r in ev.iterrows():
            df.loc[(df.index >= r["start_time"]) & (df.index <= r["end_time"]), "label"] = 1
    return df

def clean_and_impute(df: pd.DataFrame) -> pd.DataFrame:
    df = df.ffill(limit=12).bfill(limit=12)
    df.dropna(subset=["vsw", "np", "tp"], inplace=True)
    return df.apply(lambda x: x.fillna(x.median()))

def build_sequences(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    cols = [c for c in df.columns if c != "label"]
    X, y = df[cols].values.astype(np.float32), df["label"].values.astype(np.float32)
    Xs, ys = [], []
    for i in range(0, len(df) - SEQUENCE_LENGTH + 1, STRIDE):
        Xs.append(X[i:i+SEQUENCE_LENGTH])
        ys.append(y[i:i+SEQUENCE_LENGTH].max())
    return np.stack(Xs), np.array(ys)

def split_and_scale(X, y, scaler_path="./scaler.pkl"):
    X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=0.15, stratify=y.astype(int), random_state=42)
    X_tr, X_val, y_tr, y_val = train_test_split(X_tv, y_tv, test_size=0.17, stratify=y_tv.astype(int), random_state=42)
    sc = StandardScaler()
    sc.fit(X_tr.reshape(-1, X_tr.shape[2]))
    joblib.dump(sc, scaler_path)
    def s(a): return sc.transform(a.reshape(-1, a.shape[2])).reshape(a.shape)
    return {"X_train": s(X_tr), "y_train": y_tr, "X_val": s(X_val), "y_val": y_val, "X_test": s(X_test), "y_test": y_test, "feature_dim": X_tr.shape[2]}