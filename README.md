# Magnuson: Storm-Interval Detection from Aditya-L1 MAG Magnetometer Data

Detector of storm-driving intervals in magnetometer data from the Aditya-L1 satellite's MAG instrument, first framed as CME/ICME detection. Uses a PatchTransformer (ViT-style patch attention for time series) on 9 physics-informed magnetic field features.

> **Note on labels.** In practice the training labels so far are the nine hand-typed windows in `label_events.py` (`KNOWN_EVENTS`; dates approximate), each meant to cover a geomagnetic storm. Four of them overlap an ICME in the [Richardson and Cane near-Earth ICME list](https://izw1.caltech.edu/ACE/ASC/DATA/level3/icmetable2.htm), and a fifth (April 2025) covers the shock ahead of an ICME and ends as that ICME begins; one (November 2024) closes before that month's storm started (Kp stayed below 4). The model is therefore best read as a detector of storm-driving intervals rather than of CME passages as such. Note also that `blind_test_runner.py` feeds raw 10-second samples to a model trained on one-minute medians, so its windows and its debounce and bridging steps run on a different time scale from training.

---

## Repository Structure

```
Magnuson/
├── Data/                           # MAG NetCDF + blind test data (partly in the repo, see Data)
├── saved_models/                   # Trained model checkpoints (gitignored)
├── mag_pipeline.py                 # MAG L2 NetCDF ingestion + resampling
├── feature_engineer.py             # 9 physics-informed MAG features
├── label_events.py                 # CME event labeling (ISRO, DONKI, known)
├── model_factory.py                # PatchTransformer + XGBoost + LightGBM
├── train.py                        # Training pipeline (CLI entry point)
├── eval.py                         # Evaluation metrics (ML + space weather)
├── ensemble.py                     # Weighted model ensemble
├── blind_test_runner.py            # Blind test inference + Viterbi filter
├── diagnose_missed.py              # False negative diagnostic tool
├── check_model.py                  # Model input dimension verifier
├── data_pipeline.py                # Original SWIS pipeline (legacy)
├── eval.groovy                     # Legacy eval script
├── magnuson.ipynb                  # Notebook (gitignored)
├── .gitignore
└── README.md
```

---

## Pipeline Architecture

```
MAG L2 NetCDF (10s cadence)
    │
    ▼
mag_pipeline.py  ─── parse, clean fill values, resample to 1-min median
    │
    ▼
feature_engineer.py ─── 9 physics-informed features from Bx/By/Bz/|B|
    │
    ▼
label_events.py   ─── binary labels from ISRO CSV + NASA DONKI + known events
    │
    ▼
train.py          ─── build 128-step sequences → PatchTransformer training
    │
    ▼
model_factory.py  ─── PatchTransformer (primary), XGBoost/LightGBM (baselines)
    │
    ▼
eval.py           ─── F1, POD, FAR, CSI, HSS, AUC-ROC, AUC-PR
```

---

## File Descriptions

### `mag_pipeline.py` — MAG Data Ingestion

Reads Aditya-L1 MAG Level-2 NetCDF files (`L2_AL1_MAG_YYYYMMDD_V00.nc`) from PRADAN.

- Extracts GSM components: `Bx_gsm`, `By_gsm`, `Bz_gsm`
- Computes total field magnitude `|B| = sqrt(Bx² + By² + Bz²)`
- Replaces fill values (`< -9000`) with NaN
- Resamples from native 10-second cadence to 1-minute using median aggregation (robust to spike artefacts)
- Skips L1 files and corrupt files with logging
- Falls back to `h5netcdf` engine if `netcdf4` fails

GSM frame is used because its Z-axis aligns with Earth's magnetic dipole — sustained negative `Bz_gsm` drives geomagnetic storms.

### `feature_engineer.py` — Physics-Informed Features

9 features derived from the raw magnetic field components:

| Feature | Description | Physical Significance |
|---------|-------------|---------------------|
| `bz` | Bz GSM component [nT] | Primary CME indicator — southward Bz drives storms |
| `b_mag` | Total field magnitude [nT] | Sheath compression detection |
| `clock_angle` | `arctan2(By, Bz)` [degrees] | Flux rope orientation |
| `dbz_dt` | First difference of Bz | Shock arrival detector |
| `b_rotation` | Rolling 60-step sum of B-vector angular change | Flux rope passage detection |
| `bz_smoothed` | Savitzky-Golay filtered Bz (window=121, ~2hr) | Sustained southward field |
| `bz_persistence` | Consecutive southward step counter | Rules out noise dips |
| `b_elevation` | `arctan2(Bz, sqrt(Bx²+By²))` [degrees] | 3D field angle |
| `high_b_mag_rotation` | Sheath detector flag (|B| > mean+2σ AND |B| > 10nT AND rotation > 30°) | Catches northward-Bz CMEs |

The `high_b_mag_rotation` feature includes an absolute 10 nT floor to prevent false triggers during quiet solar wind periods.

### `label_events.py` — Event Labeling

Binary labels (1 = labelled event window, 0 = quiet) from three sources:

1. **Local ISRO CSV** — manually curated events from PRADAN reports
2. **NASA DONKI API** — automated CME arrival catalog (`kauai.ccmc.gsfc.nasa.gov/DONKI/WS/get/CMEAnalysis`), no auth required
3. **Known events** — hardcoded catalog covering major storms from May 2024 through Jan 2026 (G5-class storm May 2024 at -54 nT Bz, Jan 2026 storm at -59 nT Bz, etc.)

A `lead_hours` parameter extends labels backward so the model learns precursor signatures before the ICME front arrives. Includes warnings for zero-positive or high-positive-rate label distributions.

### `model_factory.py` — Model Architectures

#### PatchTransformer (primary)

ViT-style patch attention adapted for time series (PatchTST approach):

- Splits 128-step input sequence into 16-step patches → 8 patches
- Linear patch embedding to 64-dim space
- Learned CLS token + positional embedding
- 3-layer TransformerEncoder (4 heads, 256-dim FFN, Pre-LN)
- Classification head: Linear(64 → 32) → GELU → Dropout → Linear(32 → 1)
- Loss: Focal Loss (alpha=0.75, gamma=2.0) for class imbalance

#### XGBoost (baseline)

- 10 statistical features per input channel (mean, std, min, max, range, first, last, trend, skew, kurtosis)
- Plus 5 CME-specific features (Bz min, persistence max, rotation total, counts below -10/-20 nT)
- 300 estimators, max_depth=6, scale_pos_weight for imbalance

#### LightGBM (baseline + SHAP interpretability)

- Same feature extraction as XGBoost
- 500 estimators, early stopping
- SHAP TreeExplainer for feature importance analysis

### `train.py` — Training Pipeline

CLI entry point for the full training workflow:

```
python train.py \
    --mag_dir "/path/to/mag/data" \
    --epochs 100 \
    --baseline  # optionally train XGBoost + LightGBM
```

Steps:
1. Load MAG data from directory (recursive `.nc` search)
2. Resample to 1-minute cadence
3. Build 9 MAG features
4. Attach binary labels from known events + optional ISRO/DONKI
5. Build 128-step sequences with stride 16
6. Train/val/test split (70.5/17.5/15, stratified)
7. Per-channel standardization
8. Train PatchTransformer with AdamW + CosineAnnealingLR + early stopping (patience=20)
9. Optional: train XGBoost and LightGBM baselines
10. Find optimal threshold by F1 sweep (0.1–0.9)

### `eval.py` — Evaluation Metrics

Standard ML metrics plus space weather metrics used by forecasters:

| Metric | Definition | Purpose |
|--------|-----------|---------|
| F1 | Harmonic mean of precision & recall | Overall classifier quality |
| POD | TP / (TP + FN) | Probability of Detection (same as recall) |
| FAR | FP / (FP + TN) | False Alarm Rate (space weather definition) |
| CSI | TP / (TP + FP + FN) | Critical Success Index / Threat Score |
| HSS | Heidke Skill Score | Skill above random chance (0 = no skill, 1 = perfect) |
| AUC-ROC | Area under ROC curve | Threshold-independent rank quality |
| AUC-PR | Area under PR curve | Better for imbalanced classes |

### `ensemble.py` — Weighted Ensemble

Combines TCN and BzLSTM model predictions with tunable weights. Includes:
- `predict()` — batch inference returning per-model and ensemble probabilities
- `predict_latest()` — single-window inference with verdict string
- `update_weights()` — auto-tune weights based on validation F1 scores

### `blind_test_runner.py` — Blind Test Inference

Runs the trained PatchTransformer on held-out blind test windows with:
- Satellite blackout mask (forces probability to 0 when B-field flatlines, indicating sensor dropout)
- Bounded Viterbi decoder with log-probability emissions
- 20-minute debounce filter (suppresses false detections < 20 min)
- 45-minute bridging (fills gaps between CME detections)
- Per-test-directory visualization with probability + HMM state overlay

### `diagnose_missed.py` — False Negative Analysis

For each false negative window, produces:
- Heuristic classification of why the CME was missed (data gap, weak Bz, low |B|, no rotation, ambiguous)
- 8-panel feature plot with physical threshold annotations
- Hardest true positive comparison (lowest-probability correct detection)
- Text report with per-window statistics

---

## Data

MAG L2 NetCDF files at 10-second cadence from Aditya-L1 PRADAN. Data directory contains:

- **Per-month folders** (May 2024 – Feb 2026): training data with known CME events
- **blind test/** : held-out windows for blind evaluation (Oct 2024, Mar 2025, Apr 2026, Sep 2024). The folder named "May 14-17 2024" actually holds 9-13 May 2024, the same files as the training data, so it is not a blind test.

The trained checkpoint (`patchtransformer_mag_v1.pth`), its scaler (`scaler_mag.pkl`) and part of the MAG data are in the repo (they were added before the ignore rules); new data, checkpoints and scalers are gitignored. The MAG L2 files are ISRO Aditya-L1 data from ISSDC PRADAN.

---

## Requirements

```
torch>=2.0
numpy
pandas
xarray
netcdf4 / h5netcdf
scipy
scikit-learn
joblib
requests
matplotlib
tqdm
```

Optional (baselines):
```
xgboost
lightgbm
shap
```

---

## Usage

### Train the model:
```bash
python train.py --mag_dir /path/to/mag/data --epochs 100
```

### Run blind tests:
```bash
python blind_test_runner.py
```

### Diagnose false negatives:
```bash
python diagnose_missed.py \
    --mag_dir /path/to/mag/data \
    --model_path saved_models/patchtransformer_mag_v1.pth \
    --threshold 0.88
```

### Verify model input dimension:
```bash
python check_model.py
```
