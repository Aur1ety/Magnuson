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