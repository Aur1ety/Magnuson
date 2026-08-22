"""Smoke + correctness tests for the data-free core: features, labels, thresholds."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from feature_engineer import MAG_FEATURE_NAMES, build_mag_features
from label_events import attach_labels
from eval import evaluate, find_best_threshold


def _synthetic_mag(n=300, start="2024-05-10 12:00:00"):
    idx = pd.date_range(start, periods=n, freq="1min")
    rng = np.random.default_rng(0)
    bx = rng.normal(2, 1, n)
    by = rng.normal(-1, 1, n)
    bz = rng.normal(-3, 2, n)          # southward-ish
    b_mag = np.sqrt(bx**2 + by**2 + bz**2)
    return pd.DataFrame({"bx": bx, "by": by, "bz": bz, "b_mag": b_mag}, index=idx)


def test_build_mag_features_shape_and_names():
    feats = build_mag_features(_synthetic_mag())
    assert list(feats.columns) == MAG_FEATURE_NAMES
    assert len(feats) == 300
    assert feats.index.equals(_synthetic_mag().index)


def test_build_mag_features_value_ranges():
    feats = build_mag_features(_synthetic_mag())
    assert feats["clock_angle"].between(-180, 180).all()
    assert feats["b_elevation"].between(-90, 90).all()
    assert (feats["b_mag"] >= 0).all()
    # no feature may be entirely NaN
    assert not feats.isna().all().any()


def test_attach_labels_no_sources_gives_zeros():
    feats = build_mag_features(_synthetic_mag(start="1999-01-01"))
    out = attach_labels(feats, use_known=False)
    assert (out["label"] == 0).all()


def test_attach_labels_known_event_overlap():
    # index sits inside the May 2024 G5 storm KNOWN_EVENT
    feats = build_mag_features(_synthetic_mag(start="2024-05-10 18:00:00"))
    out = attach_labels(feats, use_known=True)
    assert out["label"].sum() == len(out)      # fully inside the event window
    # and an index far from any known event stays zero
    quiet = attach_labels(build_mag_features(_synthetic_mag(start="1999-01-01")),
                          use_known=True)
    assert quiet["label"].sum() == 0


def test_find_best_threshold_separable():
    y = np.array([0]*50 + [1]*50)
    p = np.concatenate([np.linspace(0.0, 0.3, 50), np.linspace(0.7, 1.0, 50)])
    thresh, metrics = find_best_threshold(y, p, metric="f1")
    assert 0.3 < thresh < 0.7
    assert metrics["f1"] == 1.0
