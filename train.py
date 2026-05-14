"""
train.py
========
Unified training script for TCN + BzLSTM.
"""
from __future__ import annotations
import argparse
import logging
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# Ensure your data_pipeline is up to date to support these imports
from data_pipeline  import load_cdf_directory, clean_and_impute, build_sequences, split_and_scale
from mag_pipeline   import load_mag_directory, resample_mag_to_swis
from feature_engineer import build_feature_matrix
from label_events   import attach_labels
from model_factory  import TCNModel, TCANModel, BzLSTM

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def focal_loss(pred, target, alpha=0.75, gamma=2.0):
    bce = nn.functional.binary_cross_entropy_with_logits(pred, target, reduction="none")
    return (alpha * (1 - torch.exp(-bce)) ** gamma * bce).mean()

def train_one_model(model, X_train, y_train, X_val, y_val, epochs=50, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, opt = model.to(device), torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)), batch_size=128, shuffle=True)

    best_val_loss, best_state = float("inf"), None

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = focal_loss(model(xb.to(device)), yb.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            val_loss = focal_loss(model(torch.tensor(X_val, dtype=torch.float32, device=device)), torch.tensor(y_val, dtype=torch.float32, device=device)).item()

        if val_loss < best_val_loss:
            best_val_loss, best_state = val_loss, {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        if epoch % 10 == 0: logging.info(f"Epoch {epoch} | Val Loss: {val_loss:.4f}")

    model.load_state_dict(best_state)
    return model

def main(args):
    mag_aligned = resample_mag_to_swis(load_mag_directory(args.mag_dir), swis_index=load_cdf_directory(args.swis_dir).index)
    feat_df = clean_and_impute(attach_labels(build_feature_matrix(load_cdf_directory(args.swis_dir), mag_aligned), args.label_csv, args.start_date, args.end_date))
    
    splits = split_and_scale(*build_sequences(feat_df, feature_cols=[c for c in feat_df.columns if c != "label"]), scaler_path=args.scaler_path)
    Xtr, ytr, Xv, yv = splits["X_train"], splits["y_train"], splits["X_val"], splits["y_val"]

    models = {"tcn": TCNModel(input_dim=14), "bzlstm": BzLSTM(input_dim=14)}
    for name, model in models.items():
        logging.info(f"=== Training {name.upper()} ===")
        torch.save(train_one_model(model, Xtr, ytr, Xv, yv, epochs=args.epochs).state_dict(), f"saved_models/{name}_v3.pth")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--swis_dir", default="/kaggle/input/swis_data")
    p.add_argument("--mag_dir", default="/kaggle/input/mag_data")
    p.add_argument("--label_csv", default=None)
    p.add_argument("--start_date", default="2024-05-01")
    p.add_argument("--end_date", default="2026-05-01")
    p.add_argument("--scaler_path", default="scalers/scaler_14feat.pkl")
    p.add_argument("--epochs", type=int, default=30)
    main(p.parse_args())