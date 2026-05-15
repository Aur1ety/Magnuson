"""
train.py
========
Benchmark training script for all 5 CME detection models.

Models benchmarked:
  1. TCN           - Temporal Convolutional Network
  2. Transformer   - Self-attention encoder
  3. TFT           - Temporal Fusion Transformer
  4. CNNTransformer - CNN + Transformer hybrid
  5. XGBoost       - Gradient boosted trees (non-neural baseline)

Each model is trained independently and evaluated on the same test set.
Best threshold is found automatically for each model.
Final benchmark table printed at the end.

Run on Kaggle:
    !python /kaggle/working/Geomag-Detector/train.py \
        --mag_dir "/kaggle/input/datasets/aurachan/updated-windows-mag-data" \
        --epochs 100
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from mag_pipeline     import load_mag_directory, resample_mag_to_1min
from feature_engineer import build_mag_features, MAG_FEATURE_NAMES
from label_events     import attach_labels
from model_factory    import TCNModel, TransformerModel, TFTModel, CNNTransformer, XGBoostModel
from eval             import evaluate, print_report, find_best_threshold

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger("train")

SEQUENCE_LENGTH = 128
STRIDE          = 16
BATCH_SIZE      = 128


def focal_loss(pred, target, alpha=0.75, gamma=2.0):
    bce  = nn.functional.binary_cross_entropy_with_logits(
        pred, target, reduction="none"
    )
    pt   = torch.exp(-bce)
    return (alpha * (1 - pt) ** gamma * bce).mean()


def build_sequences(df, seq_len=SEQUENCE_LENGTH, stride=STRIDE):
    X_data = df[MAG_FEATURE_NAMES].values.astype(np.float32)
    y_data = df["label"].values.astype(np.float32)
    X_seqs, y_seqs = [], []
    for start in range(0, len(df) - seq_len + 1, stride):
        end = start + seq_len
        X_seqs.append(X_data[start:end])
        y_seqs.append(float(y_data[start:end].max()))
    X = np.stack(X_seqs)
    y = np.array(y_seqs, dtype=np.float32)
    logger.info(
        "Sequences: %d | shape: %s | CME rate: %.2f%%",
        len(X), X.shape, y.mean() * 100
    )
    return X, y


def train_deep_model(
    model, X_train, y_train, X_val, y_val,
    epochs=100, lr=1e-3, patience=20,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)

    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        ),
        batch_size=BATCH_SIZE, shuffle=True,
    )

    best_val  = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            focal_loss(model(xb), yb).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            xv = torch.tensor(X_val, dtype=torch.float32, device=device)
            yv = torch.tensor(y_val, dtype=torch.float32, device=device)
            val_loss = focal_loss(model(xv), yv).item()

        if epoch % 10 == 0:
            logger.info(
                "Epoch %3d | val=%.4f | best=%.4f | patience=%d/%d",
                epoch, val_loss, best_val, no_improve, patience
            )

        if val_loss < best_val:
            best_val   = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

    model.load_state_dict(best_state)
    return model


def predict_proba(model, X, device, batch_size=256):
    model.eval()
    probs = []
    for i in range(0, len(X), batch_size):
        xb = torch.tensor(X[i:i+batch_size], dtype=torch.float32, device=device)
        with torch.no_grad():
            probs.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.concatenate(probs)


def main(args):
    logger.info("Loading MAG data from: %s", args.mag_dir)
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

    X, y = build_sequences(feat_df)

    if y.sum() == 0:
        logger.error("No positive labels — check event windows")
        return

    strat = (y > 0).astype(int)
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=0.15, stratify=strat, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=0.176,
        stratify=(y_tv > 0).astype(int), random_state=42
    )
    logger.info(
        "Split: train=%d | val=%d | test=%d",
        len(y_train), len(y_val), len(y_test)
    )

    n, t, f = X_train.shape
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, f)).reshape(n, t, f)
    X_val   = scaler.transform(X_val.reshape(-1, f)).reshape(X_val.shape)
    X_test  = scaler.transform(X_test.reshape(-1, f)).reshape(X_test.shape)

    Path(args.scaler_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, args.scaler_path)
    logger.info("Scaler saved: %s", args.scaler_path)

    input_dim = len(MAG_FEATURE_NAMES)
    deep_models = {
        "TCN":            TCNModel(input_dim=input_dim),
        "Transformer":    TransformerModel(input_dim=input_dim),
        "TFT":            TFTModel(input_dim=input_dim),
        "CNNTransformer": CNNTransformer(input_dim=input_dim),
    }

    os.makedirs("saved_models", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = {}

    for name, model in deep_models.items():
        logger.info("=" * 50)
        logger.info("Training: %s", name)
        logger.info("=" * 50)

        trained = train_deep_model(
            model, X_train, y_train, X_val, y_val,
            epochs=args.epochs, patience=20,
        )
        save_path = f"saved_models/{name.lower()}_mag_v1.pth"
        torch.save(trained.state_dict(), save_path)

        probs = predict_proba(trained, X_test, device)
        best_thresh, best_metrics = find_best_threshold(y_test, probs, metric="f1")

        logger.info("-- %s @ default threshold 0.50 --", name)
        print_report(evaluate(y_test, probs, threshold=0.50))
        logger.info("-- %s @ best threshold %.2f --", name, best_thresh)
        print_report(best_metrics)

        results[name] = {
            "metrics":    best_metrics,
            "threshold":  best_thresh,
            "probs":      probs,
        }

    logger.info("=" * 50)
    logger.info("Training: XGBoost")
    logger.info("=" * 50)

    try:
        xgb_model = XGBoostModel(
            scale_pos_weight=int((y_train == 0).sum() / (y_train == 1).sum() + 1)
        )
        xgb_model.fit(X_train, y_train, X_val, y_val)
        joblib.dump(xgb_model, "saved_models/xgboost_mag_v1.pkl")

        xgb_probs = xgb_model.predict_proba(X_test)
        best_thresh, best_metrics = find_best_threshold(y_test, xgb_probs, metric="f1")

        logger.info("-- XGBoost @ best threshold %.2f --", best_thresh)
        print_report(best_metrics)

        results["XGBoost"] = {
            "metrics":   best_metrics,
            "threshold": best_thresh,
            "probs":     xgb_probs,
        }

        feat_cols = MAG_FEATURE_NAMES
        stats     = ["mean","std","min","max","range","last","first","trend","skew","kurt"]
        xgb_feat_names = [f"{f}_{s}" for f in feat_cols for s in stats]
        xgb_feat_names += ["bz_min","bz_persist_max","b_rot_total","bz_below_10","bz_below_20"]
        imp = xgb_model.feature_importance(xgb_feat_names)
        logger.info("XGBoost top features:\n%s", imp.to_string())

    except ImportError:
        logger.warning("xgboost not installed -- skipping. Run: pip install xgboost")

    logger.info("\n")
    logger.info("=" * 70)
    logger.info("BENCHMARK RESULTS (at best F1 threshold per model)")
    logger.info("=" * 70)
    logger.info(
        "%-16s %6s %6s %6s %6s %6s %6s %6s",
        "Model", "F1", "Prec", "Recall", "FAR", "CSI", "HSS", "AUC"
    )
    logger.info("-" * 70)

    for name, res in results.items():
        m = res["metrics"]
        logger.info(
            "%-16s %6.3f %6.3f %6.3f  %6.4f %6.3f %6.3f %6.3f",
            name,
            m["f1"], m["precision"], m["recall"],
            m["far"], m["csi"], m["hss"], m["auc_roc"]
        )

    logger.info("=" * 70)

    best_model = max(results, key=lambda k: results[k]["metrics"]["hss"])
    logger.info(
        "Best model by HSS: %s (HSS=%.3f)",
        best_model, results[best_model]["metrics"]["hss"]
    )
    logger.info("Training complete.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mag_dir",     default="/kaggle/input/datasets/aurachan/updated-windows-mag-data")
    p.add_argument("--label_csv",   default=None)
    p.add_argument("--start_date",  default="2024-05-01")
    p.add_argument("--end_date",    default="2026-05-01")
    p.add_argument("--scaler_path", default="/kaggle/working/scalers/scaler_mag.pkl")
    p.add_argument("--epochs",      type=int, default=100)
    main(p.parse_args())