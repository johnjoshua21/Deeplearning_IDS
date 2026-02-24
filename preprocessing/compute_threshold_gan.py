"""
THRESHOLD FINDER — LABELED DATASET
=====================================
- Loads labeled_attack_dataset.csv
- Builds sliding windows (no scaler — data already normalized)
- Computes LSTM reconstruction error for every window
- Dynamically sweeps thresholds based on actual error range
- Recommends best threshold using 4 methods
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import (
    f1_score, accuracy_score,
    precision_score, recall_score,
    roc_curve, auc
)

# ============================================================
# CONFIGURATION
# ============================================================
#DATASET_PATH = "medical_iot_ids/processed/labeled_attack_dataset.csv"
DATASET_PATH = "medical_iot_ids/processed/labeled_attack_dataset_gan.csv"
MODEL_PATH   = "medical_iot_ids/model/lstm_autoencoder.h5"
SCALER_PATH  = "medical_iot_ids/model/scaler.pkl"
WINDOW_SIZE  = 60
COLUMNS      = ['FHR', 'TOCO', 'SpO2', 'RespRate', 'Temp']

print("=" * 60)
print("   THRESHOLD FINDER — LABELED DATASET")
print("=" * 60)

# ============================================================
# STEP 1 — LOAD
# ============================================================
print("\n[1/5] Loading model and dataset...")

model = load_model(MODEL_PATH, compile=False)
df    = pd.read_csv(DATASET_PATH)

print(f"  ✅ Dataset rows : {len(df):,}")
print(f"  Normal  (0)    : {(df['label']==0).sum():,}")
print(f"  Attack  (1)    : {(df['label']==1).sum():,}")

# ============================================================
# STEP 2 — BUILD WINDOWS
# ============================================================
print("\n[2/5] Building sliding windows...")

# Data is already normalized — do NOT apply scaler again
sensor_data = df[COLUMNS].values
labels      = df['label'].values

X_windows = []
y_windows = []

for i in range(len(sensor_data) - WINDOW_SIZE):
    X_windows.append(sensor_data[i : i + WINDOW_SIZE])
    y_windows.append(labels[i + WINDOW_SIZE - 1])

X_windows = np.array(X_windows)
y_windows = np.array(y_windows)

print(f"  Total windows : {len(X_windows):,}")
print(f"  Normal        : {(y_windows==0).sum():,}")
print(f"  Attack        : {(y_windows==1).sum():,}")

# ============================================================
# STEP 3 — RECONSTRUCTION ERRORS
# ============================================================
print("\n[3/5] Computing reconstruction errors...")
print("  (Takes a few minutes...)\n")

y_scores = []
for i, window in enumerate(X_windows):
    if (i+1) % 500 == 0:
        print(f"  Processed {i+1}/{len(X_windows)}...")
    w     = window.reshape(1, WINDOW_SIZE, 5)
    recon = model.predict(w, verbose=0)
    y_scores.append(float(np.mean((w - recon) ** 2)))

y_scores = np.array(y_scores)

normal_errors = y_scores[y_windows == 0]
attack_errors = y_scores[y_windows == 1]

print(f"\n  Normal errors")
print(f"    Mean   : {normal_errors.mean():.6f}")
print(f"    Std    : {normal_errors.std():.6f}")
print(f"    95th % : {np.percentile(normal_errors, 95):.6f}")
print(f"    99th % : {np.percentile(normal_errors, 99):.6f}")
print(f"    Max    : {normal_errors.max():.6f}")

print(f"\n  Attack errors")
print(f"    Mean   : {attack_errors.mean():.6f}")
print(f"    Std    : {attack_errors.std():.6f}")
print(f"    5th %  : {np.percentile(attack_errors, 5):.6f}")
print(f"    Min    : {attack_errors.min():.6f}")
print(f"    Max    : {attack_errors.max():.6f}")

separation = attack_errors.mean() - normal_errors.mean()
print(f"\n  Separation (attack_mean - normal_mean) : {separation:.6f}")
if separation < 0.5:
    print("  ⚠️  Low separation — attacks may not be strong enough")
elif separation > 2.0:
    print("  ✅ Good separation — threshold finder should work well")

# ============================================================
# STEP 4 — SWEEP THRESHOLDS DYNAMICALLY
# ============================================================
print("\n[4/5] Sweeping thresholds...")

# Dynamic range based on actual error distribution
sweep_min = normal_errors.min() * 0.5
sweep_max = attack_errors.max() * 1.1
print(f"  Sweep range : {sweep_min:.4f} → {sweep_max:.4f}")

thresholds = np.round(np.linspace(sweep_min, sweep_max, 5000), 6)

results = []
for t in thresholds:
    y_pred = (y_scores > t).astype(int)

    tn = int(np.sum((y_pred == 0) & (y_windows == 0)))
    fp = int(np.sum((y_pred == 1) & (y_windows == 0)))
    fn = int(np.sum((y_pred == 0) & (y_windows == 1)))
    tp = int(np.sum((y_pred == 1) & (y_windows == 1)))

    acc  = (tp + tn) / len(y_windows)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    fpr  = fp / (fp + tn) if (fp + tn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0

    results.append({
        'Threshold'  : t,
        'Accuracy'   : acc,
        'Precision'  : prec,
        'Recall'     : rec,
        'F1'         : f1,
        'FPR'        : fpr,
        'Specificity': spec,
        'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn
    })

df_results = pd.DataFrame(results)

# ============================================================
# FIND BEST THRESHOLDS
# ============================================================
best_f1_row  = df_results.loc[df_results['F1'].idxmax()]
best_acc_row = df_results.loc[df_results['Accuracy'].idxmax()]

df_low_fpr   = df_results[df_results['FPR'] < 0.10]
best_bal_row = (df_low_fpr.loc[df_low_fpr['F1'].idxmax()]
                if len(df_low_fpr) > 0 else best_f1_row)

fpr_roc, tpr_roc, thresh_roc = roc_curve(y_windows, y_scores)
roc_auc    = auc(fpr_roc, tpr_roc)
youdens_j  = tpr_roc - fpr_roc
best_j_idx = np.argmax(youdens_j)
best_youden_thresh = thresh_roc[best_j_idx]

stat_thresh = normal_errors.mean() + 2.5 * normal_errors.std()

# ============================================================
# PRINT RESULTS
# ============================================================
print("\n" + "=" * 60)
print("         THRESHOLD RECOMMENDATIONS")
print("=" * 60)

print(f"\n  🥇 Best F1-Score:")
print(f"     Threshold   : {best_f1_row['Threshold']:.6f}")
print(f"     F1          : {best_f1_row['F1']:.4f}")
print(f"     Accuracy    : {best_f1_row['Accuracy']:.4f}")
print(f"     Precision   : {best_f1_row['Precision']:.4f}")
print(f"     Recall      : {best_f1_row['Recall']:.4f}")
print(f"     FPR         : {best_f1_row['FPR']:.4f}")

print(f"\n  🥈 Best Accuracy:")
print(f"     Threshold   : {best_acc_row['Threshold']:.6f}")
print(f"     Accuracy    : {best_acc_row['Accuracy']:.4f}")
print(f"     F1          : {best_acc_row['F1']:.4f}")
print(f"     FPR         : {best_acc_row['FPR']:.4f}")

print(f"\n  ⚖️  Best Balanced (F1 with FPR < 10%):")
print(f"     Threshold   : {best_bal_row['Threshold']:.6f}")
print(f"     F1          : {best_bal_row['F1']:.4f}")
print(f"     Accuracy    : {best_bal_row['Accuracy']:.4f}")
print(f"     FPR         : {best_bal_row['FPR']:.4f}")

print(f"\n  📐 Youden's J Statistic (ROC optimal):")
print(f"     Threshold   : {best_youden_thresh:.6f}")
print(f"     TPR         : {tpr_roc[best_j_idx]:.4f}")
print(f"     FPR         : {fpr_roc[best_j_idx]:.4f}")
print(f"     ROC AUC     : {roc_auc:.4f}")

print(f"\n  📊 Statistical (mean_normal + 2.5σ):")
print(f"     Threshold   : {stat_thresh:.6f}")

# Pick best overall — prefer Youden's J if ROC AUC > 0.85, else balanced
if roc_auc > 0.85:
    final_threshold = best_youden_thresh
    method_used     = "Youden's J"
else:
    final_threshold = best_bal_row['Threshold']
    method_used     = "Best Balanced F1"

print(f"\n{'='*60}")
print(f"  ✅ RECOMMENDED THRESHOLD : {final_threshold:.6f}")
print(f"     Method: {method_used}")
print(f"\n  Update in evaluate_with_labeled_dataset.py:")
print(f"     THRESHOLD = {final_threshold:.6f}")
print(f"{'='*60}")

df_results.to_csv(
    'medical_iot_ids/model/threshold_results_gan.csv', index=False
)
print(f"\n  ✅ Saved: medical_iot_ids/model/threshold_results_gan.csv")

# ============================================================
# STEP 5 — PLOTS
# ============================================================
print("\n[5/5] Generating plots...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1 — All metrics vs threshold
ax1 = axes[0, 0]
ax1.plot(df_results['Threshold'], df_results['Accuracy'],   label='Accuracy',    lw=2, color='steelblue')
ax1.plot(df_results['Threshold'], df_results['Precision'],  label='Precision',   lw=2, color='orange')
ax1.plot(df_results['Threshold'], df_results['Recall'],     label='Recall',      lw=2, color='green')
ax1.plot(df_results['Threshold'], df_results['F1'],         label='F1-Score',    lw=2.5, color='red')
ax1.plot(df_results['Threshold'], df_results['Specificity'],label='Specificity', lw=1.5, color='purple', linestyle='--')
ax1.axvline(final_threshold, color='black', lw=2, linestyle='--',
            label=f"Recommended ({final_threshold:.4f})")
ax1.set_xlabel('Threshold', fontsize=12, fontweight='bold')
ax1.set_ylabel('Score',     fontsize=12, fontweight='bold')
ax1.set_title('All Metrics vs Threshold', fontsize=13, fontweight='bold')
ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)

# Plot 2 — F1 zoomed
ax2 = axes[0, 1]
ax2.plot(df_results['Threshold'], df_results['F1'], color='red', lw=2.5, label='F1-Score')
ax2.axvline(best_f1_row['Threshold'],  color='green', lw=2, linestyle='--', label=f"Best F1 ({best_f1_row['Threshold']:.4f})")
ax2.axvline(final_threshold,           color='black', lw=2, linestyle='--', label=f"Recommended ({final_threshold:.4f})")
ax2.axvline(best_youden_thresh,        color='blue',  lw=1.5, linestyle=':', label=f"Youden ({best_youden_thresh:.4f})")
ax2.set_xlabel('Threshold', fontsize=12, fontweight='bold')
ax2.set_ylabel('F1-Score',  fontsize=12, fontweight='bold')
ax2.set_title('F1-Score vs Threshold', fontsize=13, fontweight='bold')
ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

# Plot 3 — Error distribution
ax3 = axes[1, 0]
ax3.hist(normal_errors, bins=80, alpha=0.7, label='Normal', color='green', edgecolor='black', lw=0.3)
ax3.hist(attack_errors, bins=80, alpha=0.7, label='Attack', color='red',   edgecolor='black', lw=0.3)
ax3.axvline(final_threshold, color='black',  lw=2.5, linestyle='--', label=f"Threshold = {final_threshold:.4f}")
ax3.axvline(stat_thresh,     color='purple', lw=1.5, linestyle=':',  label=f"Statistical = {stat_thresh:.4f}")
ax3.set_xlabel('Reconstruction Error', fontsize=12, fontweight='bold')
ax3.set_ylabel('Frequency',            fontsize=12, fontweight='bold')
ax3.set_title('Error Distribution (Normal vs Attack)', fontsize=13, fontweight='bold')
ax3.legend(fontsize=9); ax3.grid(True, alpha=0.3, axis='y')

# Plot 4 — ROC curve
ax4 = axes[1, 1]
ax4.plot(fpr_roc, tpr_roc, color='darkorange', lw=2.5, label=f'ROC (AUC = {roc_auc:.4f})')
ax4.plot([0,1],[0,1], 'navy', lw=1.5, linestyle='--', label='Random')
ax4.scatter(fpr_roc[best_j_idx], tpr_roc[best_j_idx], color='red', s=100, zorder=5,
            label=f"Youden's J ({best_youden_thresh:.4f})")
ax4.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
ax4.set_ylabel('True Positive Rate',  fontsize=12, fontweight='bold')
ax4.set_title('ROC Curve', fontsize=13, fontweight='bold')
ax4.legend(fontsize=9); ax4.grid(True, alpha=0.3)

plt.suptitle('Threshold Optimization — Labeled Dataset',
             fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('medical_iot_ids/model/threshold_analysis_gan.png',
            dpi=300, bbox_inches='tight')
print("  ✅ medical_iot_ids/model/threshold_analysis_gan.png")
plt.show()

print(f"\n{'='*60}")
print(f"  DONE — Use THRESHOLD = {final_threshold:.6f}")
print(f"  in evaluate_with_labeled_dataset.py")
print(f"{'='*60}")