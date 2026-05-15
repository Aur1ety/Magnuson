"""
label_events.py
===============
CME/ICME event labelling for the MAG-based detector.

Two label sources:
  1. Local ISRO CSV  -- manually curated events from PRADAN reports
  2. NASA DONKI API  -- automated CME arrival catalog (free, no auth)

Strategy: binary labels (1 = CME/ICME passage, 0 = quiet).
A lead_hours parameter extends the label window backwards so the
model can learn precursor signatures before the main front arrives.

Known events in your training data:
  - May 10-17 2024: G5-class geomagnetic storm (extreme positive)
  - Jul-Aug  2025: quiet baseline (negative class)
  - Jan-Feb  2026: active but moderate (noisy middle class)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger("label_events")


def load_isro_csv(csv_path: str) -> pd.DataFrame:
    """
    Load a manually curated CME event CSV.

    Expected columns:
        start_time  : ISO 8601 string, e.g. "2024-05-10 17:00:00"
        end_time    : ISO 8601 string

    Optional columns (ignored if missing):
        event_type  : e.g. "ICME", "sheath", "flux_rope"
        notes       : free text
    """
    df = pd.read_csv(csv_path, parse_dates=["start_time", "end_time"])
    logger.info("ISRO catalog: %d events loaded from %s", len(df), csv_path)
    return df


def fetch_donki_events(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch CME/ICME arrival times from NASA DONKI REST API.
    No authentication required. Returns empty DataFrame on failure.

    Parameters
    ----------
    start_date : "YYYY-MM-DD"
    end_date   : "YYYY-MM-DD"

    Returns
    -------
    DataFrame with columns [start_time, end_time, speed_kms, source]
    """
    url = (
        "https://kauai.ccmc.gsfc.nasa.gov/DONKI/WS/get/CMEAnalysis"
        f"?startDate={start_date}&endDate={end_date}&catalog=ALL"
    )
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        events = r.json()
    except Exception as ex:
        logger.warning("DONKI fetch failed: %s -- using empty catalog", ex)
        return pd.DataFrame(columns=["start_time", "end_time", "speed_kms"])

    rows = []
    for e in events:
        if e.get("time21_5") is None:
            continue
        t_start = pd.to_datetime(e["time21_5"])
        rows.append({
            "start_time": t_start,
            # CME transit typically lasts 24-48 hrs after leading edge
            "end_time":   t_start + pd.Timedelta(hours=48),
            "speed_kms":  e.get("speed", np.nan),
            "source":     "DONKI",
        })

    df = pd.DataFrame(rows)
    logger.info("DONKI: %d CME events fetched (%s to %s)",
                len(df), start_date, end_date)
    return df


# Known CME/ICME windows in your training data.
# Used as fallback if no label_csv or DONKI provided.
# Dates are approximate -- refine from ISRO event bulletins.
KNOWN_EVENTS = [
    # G5 storm May 2024 -- Bz hit -54nT (confirmed from parsed data)
    {"start_time": "2024-05-10 17:00:00", "end_time": "2024-05-11 20:00:00"},
    # Secondary CME same sequence
    {"start_time": "2024-05-12 00:00:00", "end_time": "2024-05-13 06:00:00"},
    # Jan 2026 storm -- Bz hit -59nT (confirmed from parsed data)
    {"start_time": "2026-01-19 00:00:00", "end_time": "2026-01-20 12:00:00"},
]


def attach_labels(
    feat_df: pd.DataFrame,
    isro_csv:    str | None = None,
    donki_start: str | None = None,
    donki_end:   str | None = None,
    use_known:   bool = True,
    lead_hours:  int  = 2,
) -> pd.DataFrame:
    """
    Stamp binary CME labels onto the feature DataFrame.

    Parameters
    ----------
    feat_df      : feature DataFrame with DatetimeIndex
    isro_csv     : path to local ISRO event CSV (optional)
    donki_start  : DONKI fetch start date "YYYY-MM-DD" (optional)
    donki_end    : DONKI fetch end date   "YYYY-MM-DD" (optional)
    use_known    : if True, always apply KNOWN_EVENTS as a baseline
    lead_hours   : extend label window backwards by this many hours
                   so the model learns precursor signatures

    Returns
    -------
    feat_df with a "label" column added (0 or 1)
    """
    df = feat_df.copy()
    df["label"] = 0

    catalogs = []

    # Source 1: known hardcoded events (always safe to include)
    if use_known:
        catalogs.append(
            pd.DataFrame(KNOWN_EVENTS, columns=["start_time", "end_time"])
            .assign(start_time=lambda x: pd.to_datetime(x["start_time"]),
                    end_time=lambda x:   pd.to_datetime(x["end_time"]))
        )

    # Source 2: local ISRO CSV
    if isro_csv and Path(isro_csv).exists():
        catalogs.append(load_isro_csv(isro_csv))

    # Source 3: NASA DONKI
    if donki_start and donki_end:
        catalogs.append(fetch_donki_events(donki_start, donki_end))

    if not catalogs:
        logger.warning("No event sources -- all labels set to 0")
        return df

    all_events = pd.concat(catalogs, ignore_index=True)
    lead       = pd.Timedelta(hours=lead_hours)

    for _, row in all_events.iterrows():
        mask = (
            (df.index >= row["start_time"] - lead) &
            (df.index <= row["end_time"])
        )
        df.loc[mask, "label"] = 1

    pos_frac = df["label"].mean()
    n_pos    = df["label"].sum()
    logger.info(
        "Labels: %d positive rows (%.2f%% of %d total)",
        n_pos, pos_frac * 100, len(df)
    )

    if pos_frac == 0:
        logger.warning(
            "Zero positive labels -- the event times may not overlap "
            "with your data range. Check KNOWN_EVENTS dates."
        )
    elif pos_frac > 0.4:
        logger.warning(
            "Positive rate %.1f%% seems high -- check event windows "
            "for overlaps or incorrect end times.",
            pos_frac * 100
        )

    return df