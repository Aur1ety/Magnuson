"""
label_events.py
===============
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
    if not Path(csv_path).exists(): return pd.DataFrame(columns=["start_time", "end_time"])
    return pd.read_csv(csv_path, parse_dates=["start_time", "end_time"])

def fetch_donki_cmes(start_date: str, end_date: str) -> pd.DataFrame:
    url = f"https://kauai.ccmc.gsfc.nasa.gov/DONKI/WS/get/CMEAnalysis?startDate={start_date}&endDate={end_date}&catalog=ALL"
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        rows = []
        for e in r.json():
            if e.get("time21_5"):
                t_start = pd.to_datetime(e["time21_5"])
                rows.append({"start_time": t_start, "end_time": t_start + pd.Timedelta(hours=48)})
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame(columns=["start_time", "end_time"])

def attach_labels(feat_df: pd.DataFrame, isro_csv: str | None = None, donki_start: str | None = None, donki_end: str | None = None, lead_hours: int = 2) -> pd.DataFrame:
    df = feat_df.copy()
    df["label"] = 0
    catalogs = []
    
    if isro_csv: catalogs.append(load_isro_catalog(isro_csv))
    if donki_start and donki_end: catalogs.append(fetch_donki_cmes(donki_start, donki_end))
    if not catalogs: return df

    all_events = pd.concat(catalogs, ignore_index=True)
    lead = pd.Timedelta(hours=lead_hours)

    for _, row in all_events.iterrows():
        mask = (df.index >= (row["start_time"] - lead)) & (df.index <= row["end_time"])
        df.loc[mask, "label"] = 1
        
    return df