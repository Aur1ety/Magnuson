import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import joblib

# 1. Path Management
sys.path.insert(0, os.getcwd())

from model_factory import PatchTransformer
from mag_pipeline import load_mag_directory
from feature_engineer import build_mag_features

# --- CONFIGURATION ---
MODEL_PATH = "patchtransformer_mag_v1.pth"
SCALER_PATH = "scaler_mag.pkl"
THRESHOLD = 0.61  # Optimal F1 threshold from training

# Exact paths matched to your local directory strings
TEST_DIRS = [
    r"C:\Users\ponpo\Documents\Magnuson\Data\blind test\Oct 9_24 - Oct 12_24",
    r"C:\Users\ponpo\Documents\Magnuson\Data\blind test\May 14-17 2024",
    r"C:\Users\ponpo\Documents\Magnuson\Data\blind test\Mar 1-10 2025",
    r"C:\Users\ponpo\Documents\Magnuson\Data\blind test\Apr 1-12 2026",
    r"C:\Users\ponpo\Documents\Magnuson\Data\blind test\Sep 1-10 2024"
]

def apply_viterbi_filter(probs):
    """
    Overhauled Bulletproof Viterbi Filter.
    Cures Emission Dominance using Bounded Direct Log-Likelihood mappings.
    """
    n = len(probs)
    if n == 0: return np.array([])

    # --- 1. TRANSITION MATRIX (LOG SPACE) ---
    T = np.array([
        [1 - 1e-7, 1e-7], 
        [1e-5, 1 - 1e-5]  
    ])
    log_T = np.log(T)
    
    # --- 2. BOUNDED EMISSIONS ---
    eps = 1e-2  
    probs_np = np.array(probs)
    p_cme = np.clip(probs_np, eps, 1 - eps)
    p_quiet = 1.0 - p_cme
    
    log_emission_quiet = np.log(p_quiet)
    log_emission_cme = np.log(p_cme)

    # --- 3. VITERBI PROCESSING ---
    viterbi = np.zeros((2, n))
    backpointer = np.zeros((2, n), dtype=int)
    
    viterbi[0, 0] = np.log(1.0) + log_emission_quiet[0]
    viterbi[1, 0] = np.log(1e-15) + log_emission_cme[0]
    
    for t in range(1, n):
        for s in range(2):
            current_emission = log_emission_cme[t] if s == 1 else log_emission_quiet[t]
            paths = viterbi[:, t-1] + log_T[:, s]
            viterbi[s, t] = current_emission + np.max(paths)
            backpointer[s, t] = np.argmax(paths)
            
    best_path = np.zeros(n, dtype=int)
    best_path[n-1] = np.argmax(viterbi[:, n-1])
    for t in range(n-2, -1, -1):
        best_path[t] = backpointer[best_path[t+1], t+1]
        
    # --- 4. HARD-WIRED SIGNAL SMOOTHING ---
    path_smoothed = best_path.copy()
    
    # Pass A: 20-Minute Debounce
    inside_cme = False
    start_idx = 0
    for i in range(n):
        if path_smoothed[i] == 1 and not inside_cme:
            inside_cme = True
            start_idx = i
        elif path_smoothed[i] == 0 and inside_cme:
            inside_cme = False
            duration = i - start_idx
            if duration < 20:
                path_smoothed[start_idx:i] = 0
    if inside_cme and (n - start_idx) < 20:
        path_smoothed[start_idx:n] = 0

    # Pass B: 45-Minute Bridging
    inside_quiet = False
    q_start_idx = 0
    for i in range(n):
        if path_smoothed[i] == 0 and not inside_quiet:
            inside_quiet = True
            q_start_idx = i
        elif path_smoothed[i] == 1 and inside_quiet:
            inside_quiet = False
            duration = i - q_start_idx
            if duration < 45 and q_start_idx > 0:
                if path_smoothed[q_start_idx - 1] == 1:
                    path_smoothed[q_start_idx:i] = 1
                    
    return path_smoothed

def run_inference():
    print("\n" + "="*70)
    print("🚀 TARGET ACQUIRED: RUNNING VITERBI WITH SATELLITE BLACKOUT MASK!")
    print("="*70 + "\n")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Missing weights file {MODEL_PATH} in current directory.")
        return

    model = PatchTransformer(input_dim=9)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    
    scaler = joblib.load(SCALER_PATH)
    
    for test_dir in TEST_DIRS:
        if not os.path.exists(test_dir):
            print(f"\n⚠️ Skipping Directory: Folder not found locally -> {test_dir}")
            continue
            
        folder_name = os.path.basename(test_dir)
        print(f"\n🔍 Analyzing Mission Data: {folder_name}")
        
        raw_df = load_mag_directory(test_dir)
        features = build_mag_features(raw_df)
        
        feature_cols = [
            'bz', 'b_mag', 'clock_angle', 'dbz_dt', 'b_rotation', 
            'bz_smoothed', 'bz_persistence', 'b_elevation',
            'high_b_mag_rotation'
        ]
        
        X_scaled = scaler.transform(features[feature_cols])
        
        raw_probs = []
        times = []
        win_size = 128
        
        print(f"🏃 Running Transformer inference with Blackout Mask...")
        with torch.no_grad():
            for i in range(win_size, len(X_scaled)):
                
                # --- SATELLITE BLACKOUT MASK ---
                # Checks if the total magnetic field has artificially flatlined
                window_b_mag = features['b_mag'].iloc[i-win_size:i]
                
                if np.std(window_b_mag) < 0.001:
                    prob = 0.0 # Force absolute zero if sensor is dead/NaN-filled
                else:
                    seq = X_scaled[i-win_size:i]
                    seq_tensor = torch.FloatTensor(seq).unsqueeze(0).to(device)
                    output = model(seq_tensor)
                    prob = torch.sigmoid(output).item()
                # -------------------------------

                raw_probs.append(prob)
                times.append(features.index[i])

        print("🛠 Refining detections with Bounded Viterbi Algorithm...")
        clean_states = apply_viterbi_filter(raw_probs)

        plt.figure(figsize=(15, 7))
        plt.plot(times, raw_probs, color='crimson', lw=1, alpha=0.3, label='Transformer Raw Prob')
        plt.fill_between(times, 0, clean_states, color='orange', alpha=0.35, label='CME Detected (HMM)')
        plt.step(times, clean_states, color='black', lw=1.8, label='HMM Hidden State')
        plt.axhline(y=THRESHOLD, color='gray', linestyle=':', alpha=0.5, label='Inference Threshold')

        plt.title(f"Hybrid Transformer-HMM Detection Report: {folder_name}", fontsize=14)
        plt.ylabel("CME Probability / Binary State")
        plt.xlabel("Time (UTC)")
        plt.ylim(-0.05, 1.05)
        plt.grid(True, alpha=0.15)
        plt.legend(loc='upper right')
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    run_inference()