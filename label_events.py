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