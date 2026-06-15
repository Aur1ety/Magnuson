"""
diagnose_missed.py
==================
Finds the window(s) that PatchTransformer missed (FN) and plots all 8 MAG
features across them, with key derived quantities annotated.

Outputs:
  missed_windows_report.txt  -- text summary of the missed window's statistics
  missed_window_N.png        -- one plot per missed window (8-panel feature plot)

Run on Kaggle after train.py has completed:
    !python /kaggle/working/Magnuson/diagnose_missed.py \
        --mag_dir  "/kaggle/input/datasets/aurachan/updated-windows-mag-data" \
        --model_path "saved_models/patchtransformer_mag_v1.pth" \
        --scaler_path "/kaggle/working/scalers/scaler_mag.pkl" \
        --threshold 0.88
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch

from mag_pipeline     import load_mag_directory, resample_mag_to_1min
from feature_engineer import build_mag_features, MAG_FEATURE_NAMES
from label_events     import attach_labels
from model_factory    import PatchTransformer
from train            import build_sequences, SEQUENCE_LENGTH, STRIDE

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("diagnose")

# ── Colours ──────────────────────────────────────────────────────────────────
FEATURE_COLOURS = {
    "bz":             "#e63946",   # red    — most important
    "b_mag":          "#457b9d",   # steel blue
    "clock_angle":    "#f4a261",   # orange
    "dbz_dt":         "#2a9d8f",   # teal
    "b_rotation":     "#8338ec",   # purple
    "bz_smoothed":    "#ff006e",   # pink
    "bz_persistence": "#fb8500",   # amber
    "b_elevation":    "#06d6a0",   # mint
}

# Physical threshold lines to annotate on each panel
THRESHOLDS = {
    "bz":          [("Bz = −10 nT", -10, "--", 0.5),
                    ("Bz = −20 nT", -20, ":",  0.7)],
    "bz_smoothed": [("Bz_sm = −10 nT", -10, "--", 0.5)],
}


def load_model(path, input_dim, seq_len, device):
    model = PatchTransformer(input_dim=input_dim, seq_len=seq_len)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model.to(device)


def predict_proba_single(model, X, device):
    """Return sigmoid probabilities for array X (N, T, F)."""
    with torch.no_grad():
        xb = torch.tensor(X, dtype=torch.float32, device=device)
        return torch.sigmoid(model(xb)).cpu().numpy()


def classify_missed_window(window_raw: np.ndarray,
                           feat_names: list[str]) -> dict:
    """
    Heuristic classification of why a CME window was missed.
    Returns a dict of diagnostics for the report.
    """
    bz_col  = window_raw[:, feat_names.index("bz")]
    mag_col = window_raw[:, feat_names.index("b_mag")]
    rot_col = window_raw[:, feat_names.index("b_rotation")]
    elv_col = window_raw[:, feat_names.index("b_elevation")]

    nan_frac   = np.isnan(window_raw).mean()
    bz_min     = np.nanmin(bz_col)
    mag_max    = np.nanmax(mag_col)
    rot_total  = np.nansum(np.abs(rot_col))
    below_10   = (bz_col < -10).sum()
    below_20   = (bz_col < -20).sum()
    elv_range  = np.nanmax(elv_col) - np.nanmin(elv_col)

    # Classify
    reasons = []
    if nan_frac > 0.15:
        reasons.append(f"DATA GAP: {nan_frac*100:.1f}% NaN")
    if bz_min > -10:
        reasons.append(f"WEAK BZ: min only {bz_min:.1f} nT (never crossed −10 nT)")
    if mag_max < 15:
        reasons.append(f"LOW |B|: max only {mag_max:.1f} nT (weak field enhancement)")
    if rot_total < 5:
        reasons.append(f"NO ROTATION: total b_rotation = {rot_total:.2f} (no flux rope)")
    if not reasons:
        reasons.append("AMBIGUOUS: moderate CME — model uncertain near threshold")

    return {
        "nan_frac":   nan_frac,
        "bz_min":     bz_min,
        "mag_max":    mag_max,
        "rot_total":  rot_total,
        "below_10":   int(below_10),
        "below_20":   int(below_20),
        "elv_range":  elv_range,
        "reasons":    reasons,
    }


def plot_missed_window(window_raw: np.ndarray,
                       window_scaled: np.ndarray,
                       prob: float,
                       feat_names: list[str],
                       diag: dict,
                       window_idx: int,
                       seq_start_min: int,
                       out_path: Path):
    """
    8-panel plot of all MAG features for a missed window.
    Top row: bz, b_mag, clock_angle, dbz_dt
    Bottom row: b_rotation, bz_smoothed, bz_persistence, b_elevation
    """
    n_feat = len(feat_names)
    t      = np.arange(window_raw.shape[0])          # minutes within window
    t_hr   = t / 60.0

    fig = plt.figure(figsize=(18, 10))
    fig.patch.set_facecolor("#0f1117")

    # Title
    fig.suptitle(
        f"Missed CME Window #{window_idx}  |  "
        f"PatchTransformer P={prob:.4f}  |  "
        f"Window starts at t+{seq_start_min} min from dataset origin\n"
        f"Diagnosis: {' | '.join(diag['reasons'])}",
        color="white", fontsize=11, y=0.98,
        fontweight="bold",
    )

    gs = gridspec.GridSpec(2, 4, figure=fig,
                           hspace=0.45, wspace=0.35,
                           left=0.06, right=0.97,
                           top=0.90, bottom=0.08)

    for i, fname in enumerate(feat_names):
        row, col = divmod(i, 4)
        ax = fig.add_subplot(gs[row, col])
        ax.set_facecolor("#1a1d27")

        colour = FEATURE_COLOURS.get(fname, "#aaaaaa")
        vals   = window_raw[:, i]

        ax.plot(t_hr, vals, color=colour, lw=1.4, alpha=0.9)
        ax.fill_between(t_hr, vals, alpha=0.15, color=colour)

        # Threshold lines
        for label, val, ls, alpha in THRESHOLDS.get(fname, []):
            ax.axhline(val, color="white", ls=ls, lw=0.8, alpha=alpha)
            ax.text(t_hr[-1] * 0.02, val, label,
                    color="white", fontsize=6, alpha=0.7, va="bottom")

        # Zero line
        ax.axhline(0, color="#444", lw=0.6)

        ax.set_title(fname, color="white", fontsize=9, pad=4)
        ax.tick_params(colors="white", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")
        ax.set_xlabel("time (hr)", color="#aaa", fontsize=7)

        # Stat annotation
        vmin, vmax = np.nanmin(vals), np.nanmax(vals)
        ax.text(0.98, 0.97,
                f"min={vmin:.2f}\nmax={vmax:.2f}",
                transform=ax.transAxes,
                color="#cccccc", fontsize=6.5, va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.2", fc="#222", ec="none", alpha=0.7))

    # Stats box
    stats_text = (
        f"Bz min:        {diag['bz_min']:.1f} nT\n"
        f"|B| max:       {diag['mag_max']:.1f} nT\n"
        f"Bz < −10 nT:  {diag['below_10']} steps\n"
        f"Bz < −20 nT:  {diag['below_20']} steps\n"
        f"b_rot total:  {diag['rot_total']:.2f}\n"
        f"elv range:    {diag['elv_range']:.2f}\n"
        f"NaN fraction: {diag['nan_frac']*100:.1f}%\n"
        f"Model prob:   {prob:.4f}"
    )
    fig.text(0.01, 0.50, stats_text,
             color="#dddddd", fontsize=8,
             va="center", ha="left",
             fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5",
                       fc="#1a1d27", ec="#444", alpha=0.9),
             transform=fig.transFigure)

    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info("Saved plot: %s", out_path)


def main(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load & process data (same pipeline as train.py) ──────────────────────
    logger.info("Loading MAG data...")
    mag_raw = load_mag_directory(args.mag_dir)
    mag_df  = resample_mag_to_1min(mag_raw)
    mag_df  = mag_df.dropna(how="all")

    feat_df = build_mag_features(mag_df)
    feat_df = attach_labels(
        feat_df,
        isro_csv=args.label_csv,
        donki_start=args.start_date,
        donki_end=args.end_date,
        use_known=True,
    )
    feat_df.dropna(subset=["bz", "b_mag"], inplace=True)

    X_raw, y = build_sequences(feat_df,
                                seq_len=SEQUENCE_LENGTH,
                                stride=STRIDE)
    logger.info("Total sequences: %d | positives: %d", len(y), int(y.sum()))

    # ── Scale using saved scaler ──────────────────────────────────────────────
    scaler  = joblib.load(args.scaler_path)
    n, t, f = X_raw.shape
    X_sc    = scaler.transform(X_raw.reshape(-1, f)).reshape(n, t, f)

    # ── Load model ────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_model(args.model_path,
                        input_dim=len(MAG_FEATURE_NAMES),
                        seq_len=SEQUENCE_LENGTH,
                        device=device)

    # ── Run inference on ALL positive windows ────────────────────────────────
    pos_idx = np.where(y > 0)[0]
    logger.info("Running inference on %d positive windows...", len(pos_idx))

    probs_pos = predict_proba_single(model, X_sc[pos_idx], device)
    missed_mask = probs_pos < args.threshold
    missed_local = np.where(missed_mask)[0]        # indices into pos_idx
    missed_global = pos_idx[missed_local]          # indices into full X

    logger.info(
        "Threshold %.2f | Missed: %d / %d positive windows  (FN rate: %.1f%%)",
        args.threshold, len(missed_global), len(pos_idx),
        100 * len(missed_global) / max(len(pos_idx), 1),
    )

    if len(missed_global) == 0:
        logger.info("No missed windows at this threshold — model caught everything!")
        return

    # ── Report & plots ────────────────────────────────────────────────────────
    report_lines = [
        "=" * 70,
        f"PatchTransformer Missed Window Diagnostic",
        f"Threshold: {args.threshold}",
        f"Missed: {len(missed_global)} / {len(pos_idx)} positive windows",
        "=" * 70,
        "",
    ]

    for rank, (local_i, global_i) in enumerate(
        sorted(zip(missed_local, missed_global),
               key=lambda x: probs_pos[x[0]])   # lowest prob first
    ):
        prob       = float(probs_pos[local_i])
        win_raw    = X_raw[global_i]             # unscaled — physical units
        win_sc     = X_sc[global_i]
        seq_t0_min = global_i * STRIDE           # approx minutes from data start

        diag = classify_missed_window(win_raw, MAG_FEATURE_NAMES)

        report_lines += [
            f"--- Missed window #{rank+1} ---",
            f"  Dataset index   : {global_i}",
            f"  Sequence t0     : ~{seq_t0_min} min from data origin",
            f"  Model prob      : {prob:.4f}  (threshold={args.threshold})",
            f"  Bz minimum      : {diag['bz_min']:.2f} nT",
            f"  |B| maximum     : {diag['mag_max']:.2f} nT",
            f"  Bz < −10 nT     : {diag['below_10']} timesteps",
            f"  Bz < −20 nT     : {diag['below_20']} timesteps",
            f"  b_rotation sum  : {diag['rot_total']:.3f}",
            f"  b_elevation rng : {diag['elv_range']:.3f}",
            f"  NaN fraction    : {diag['nan_frac']*100:.1f}%",
            f"  Diagnosis       : {' | '.join(diag['reasons'])}",
            "",
        ]

        plot_path = out_dir / f"missed_window_{rank+1}_idx{global_i}.png"
        plot_missed_window(
            win_raw, win_sc, prob,
            MAG_FEATURE_NAMES, diag,
            rank + 1, seq_t0_min, plot_path,
        )

    # Also: show the hardest TRUE POSITIVE (lowest prob that was still caught)
    caught_mask  = ~missed_mask
    if caught_mask.any():
        hardest_local  = np.where(caught_mask)[0][np.argmin(probs_pos[caught_mask])]
        hardest_global = pos_idx[hardest_local]
        hardest_prob   = float(probs_pos[hardest_local])

        hardest_raw  = X_raw[hardest_global]
        hardest_diag = classify_missed_window(hardest_raw, MAG_FEATURE_NAMES)

        report_lines += [
            "=" * 70,
            "Hardest TRUE POSITIVE (lowest prob still above threshold)",
            f"  Dataset index : {hardest_global}",
            f"  Model prob    : {hardest_prob:.4f}",
            f"  Bz minimum    : {hardest_diag['bz_min']:.2f} nT",
            f"  |B| maximum   : {hardest_diag['mag_max']:.2f} nT",
            f"  Diagnosis     : {' | '.join(hardest_diag['reasons'])}",
            "",
        ]

        plot_path = out_dir / f"hardest_tp_idx{hardest_global}.png"
        plot_missed_window(
            hardest_raw, X_sc[hardest_global], hardest_prob,
            MAG_FEATURE_NAMES, hardest_diag,
            0, hardest_global * STRIDE, plot_path,
        )
        report_lines.append(f"Hardest TP plot saved: {plot_path}")

    report_text = "\n".join(report_lines)
    report_path = out_dir / "missed_windows_report.txt"
    report_path.write_text(report_text)
    logger.info("Report saved: %s", report_path)
    print("\n" + report_text)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mag_dir",
                   default="/kaggle/input/datasets/aurachan/updated-windows-mag-data")
    p.add_argument("--model_path",
                   default="saved_models/patchtransformer_mag_v1.pth")
    p.add_argument("--scaler_path",
                   default="/kaggle/working/scalers/scaler_mag.pkl")
    p.add_argument("--label_csv",   default=None)
    p.add_argument("--start_date",  default="2024-05-01")
    p.add_argument("--end_date",    default="2026-05-01")
    p.add_argument("--threshold",   type=float, default=0.88)
    p.add_argument("--out_dir",     default="/kaggle/working/diagnostics")
    main(p.parse_args())