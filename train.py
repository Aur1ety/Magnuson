"""
train.py
========
MAG-only unified training script for TCN + BzLSTM.

Pipeline:
  1. Load MAG L2 NetCDF files
  2. Resample to 1-min cadence
  3. Build 8 MAG features
  4. Attach CME labels
  5. Build sliding-window sequences
  6. Train/val/test split + StandardScaler
  7. Train TCN and BzLSTM with focal loss + early stopping
  8. Save models and scaler
  9. Print test set evaluation

Run on Kaggle:
    !python /kaggle/working/Geomag-Detector/train.py \
        --mag_dir "/kaggle/input/aditya-l1-mag" \
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
from model_factory    import TCNModel, BzLSTM
from eval             import evaluate, print_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger("train")

# ── Constants ─────────────────────────────────────────────────────────────────
SEQUENCE_LENGTH  = 128   # steps -- at 1-min cadence = ~2 hrs of context
STRIDE           = 16    # hop between consecutive windows
EARLY_STOP       = 15    # epochs without improvement before stopping
BATCH_SIZE       = 128


# ── Focal loss ────────────────────────────────────────────────────────────────

def focal_loss(
    pred:   torch.Tensor,
    target: torch.Tensor,
    alpha:  float = 0.75,
    gamma:  float = 2.0,
) -> torch.Tensor:
    """
    Focal loss for class-imbalanced binary classification.

    alpha : weight on the positive (CME) class.
            0.75 means CME events contribute 3x more to the loss
            than quiet windows -- compensates for ~5% CME rate.
    gamma : focusing parameter.
            gamma=2 means the loss for easy correct predictions
            is scaled down by (1-p)^2, forcing the model to focus
            on hard examples (ambiguous CME onset windows).
    """
    bce  = nn.functional.binary_cross_entropy_with_logits(
        pred, target, reduction="none"
    )
    pt   = torch.exp(-bce)
    loss = alpha * (1 - pt) ** gamma * bce
    return loss.mean()


# ── Sequence builder ──────────────────────────────────────────────────────────

def build_sequences(
    df,
    seq_len: int = SEQUENCE_LENGTH,
    stride:  int = STRIDE,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sliding window sequence builder.

    X shape: (N, seq_len, 8)
    y shape: (N,)  -- 1 if ANY step in the window is labelled CME

    'ANY-CME' labelling means the model is penalised for missing
    any window that contains even partial CME coverage. This is
    intentional -- we want early warning, not just detection at peak.
    """
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
        "Sequences built: %d | shape: %s | CME rate: %.2f%%",
        len(X), X.shape, y.mean() * 100
    )
    return X, y


# ── Training loop ─────────────────────────────────────────────────────────────

def train_one_model(
    model:    nn.Module,
    X_train:  np.ndarray,
    y_train:  np.ndarray,
    X_val:    np.ndarray,
    y_val:    np.ndarray,
    epochs:   int   = 100,
    lr:       float = 1e-3,
    patience: int   = EARLY_STOP,
) -> nn.Module:
    """
    Generic training loop with:
      - AdamW optimiser + cosine LR schedule
      - Focal loss
      - Gradient clipping (norm 1.0)
      - Early stopping on validation loss
      - Best-checkpoint restoration

    Returns the model loaded with the best validation checkpoint.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    model  = model.to(device)

    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
    )

    best_val_loss = float("inf")
    best_state    = None
    no_improve    = 0

    for epoch in range(1, epochs + 1):
        # ── Train ──
        model.train()
        train_losses = []
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = focal_loss(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_losses.append(loss.item())
        sched.step()

        # ── Validate ──
        model.eval()
        with torch.no_grad():
            xv = torch.tensor(X_val, dtype=torch.float32, device=device)
            yv = torch.tensor(y_val, dtype=torch.float32, device=device)
            val_loss = focal_loss(model(xv), yv).item()

        if epoch % 10 == 0:
            logger.info(
                "Epoch %3d | train=%.4f | val=%.4f | best=%.4f | patience=%d/%d",
                epoch,
                np.mean(train_losses),
                val_loss,
                best_val_loss,
                no_improve,
                patience,
            )

        # ── Early stopping ──
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {
                k: v.cpu().clone()
                for k, v in model.state_dict().items()
            }
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info(
                    "Early stopping at epoch %d (no improvement for %d epochs)",
                    epoch, patience
                )
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:

    # 1. Load + resample MAG data
    logger.info("Loading MAG data from: %s", args.mag_dir)
    mag_raw = load_mag_directory(args.mag_dir)
    mag_df  = resample_mag_to_1min(mag_raw)
    mag_df  = mag_df.dropna(how='all')

    # 2. Build features
    logger.info("Building features...")
    feat_df = build_mag_features(mag_df)

    # 3. Label
    logger.info("Attaching labels...")
    feat_df = attach_labels(
        feat_df,
        isro_csv=args.label_csv,
        donki_start=args.start_date,
        donki_end=args.end_date,
        use_known=True,
    )

    # 4. Drop rows where key features are still NaN
    before = len(feat_df)
    feat_df.dropna(subset=["bz", "b_mag"], inplace=True)
    logger.info("Dropped %d NaN rows (%d remaining)", before - len(feat_df), len(feat_df))

    # 5. Build sequences
    X, y = build_sequences(feat_df)

    if y.sum() == 0:
        logger.error(
            "No positive labels found. "
            "Check that your data date range covers the known CME events "
            "(May 10-17 2024). Training cannot proceed."
        )
        return

    # 6. Split: 70% train / 15% val / 15% test
    # Stratified to preserve CME rate in each split
    strat = (y > 0).astype(int)
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=0.15, stratify=strat, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=0.176,
        stratify=(y_tv > 0).astype(int),
        random_state=42,
    )
    logger.info(
        "Split: train=%d | val=%d | test=%d",
        len(y_train), len(y_val), len(y_test)
    )

    # 7. Scale (fit on train only -- no data leakage)
    n, t, f = X_train.shape
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, f)).reshape(n, t, f)
    X_val   = scaler.transform(X_val.reshape(-1, f)).reshape(X_val.shape)
    X_test  = scaler.transform(X_test.reshape(-1, f)).reshape(X_test.shape)

    Path(args.scaler_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, args.scaler_path)
    logger.info("Scaler saved: %s", args.scaler_path)

    # 8. Train models
    os.makedirs("saved_models", exist_ok=True)
    input_dim = len(MAG_FEATURE_NAMES)  # 8

    trained_models = {}
    for name, model in [
        ("tcn",    TCNModel(input_dim=input_dim)),
        ("bzlstm", BzLSTM(input_dim=input_dim)),
    ]:
        logger.info("=" * 40)
        logger.info("Training: %s", name.upper())
        logger.info("=" * 40)

        trained = train_one_model(
            model, X_train, y_train, X_val, y_val,
            epochs=args.epochs,
        )
        save_path = f"saved_models/{name}_mag_v1.pth"
        torch.save(trained.state_dict(), save_path)
        logger.info("Saved: %s", save_path)
        trained_models[name] = trained

    # 9. Test set evaluation
    logger.info("=" * 40)
    logger.info("TEST SET EVALUATION")
    logger.info("=" * 40)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for name, model in trained_models.items():
        model.to(device).eval()
        probs = []
        for i in range(0, len(X_test), 256):
            xb = torch.tensor(X_test[i:i+256], dtype=torch.float32, device=device)
            with torch.no_grad():
                probs.append(torch.sigmoid(model(xb)).cpu().numpy())
        probs = np.concatenate(probs)
        logger.info("-- %s --", name.upper())
        print_report(evaluate(y_test, probs))

    logger.info("Training complete.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train MAG-based CME detector")
    p.add_argument(
        "--mag_dir",
        default="/kaggle/input/aditya-l1-mag",
        help="Directory containing L2 MAG .nc files"
    )
    p.add_argument(
        "--label_csv",
        default=None,
        help="Optional path to ISRO CME event CSV"
    )
    p.add_argument(
        "--start_date",
        default="2024-05-01",
        help="DONKI fetch start date YYYY-MM-DD"
    )
    p.add_argument(
        "--end_date",
        default="2026-05-01",
        help="DONKI fetch end date YYYY-MM-DD"
    )
    p.add_argument(
        "--scaler_path",
        default="/kaggle/working/scalers/scaler_mag.pkl",
        help="Path to save the fitted StandardScaler"
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Maximum training epochs (early stopping may end sooner)"
    )
    main(p.parse_args())