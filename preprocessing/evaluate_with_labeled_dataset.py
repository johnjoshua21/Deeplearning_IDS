"""
EVALUATION USING STATIC LABELED DATASET
=========================================
- Loads the pre-generated labeled_attack_dataset.csv
- Scales using your trained scaler
- Creates sliding windows (WINDOW_SIZE=60)
- Runs LSTM autoencoder on each window
- Reports full metrics + visualizations

Run AFTER generate_attack_dataset.py
"""

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve,
    f1_score, accuracy_score, precision_score, recall_score
)
from tensorflow.keras.models import load_model

# ============================================================
# CONFIGURATION
# ============================================================
DATASET_PATH = "medical_iot_ids/processed/labeled_attack_dataset_gan.csv"
MODEL_PATH   = "medical_iot_ids/model/lstm_autoencoder.h5"
SCALER_PATH  = "medical_iot_ids/model/scaler.pkl"
WINDOW_SIZE  = 60
THRESHOLD    = 1.60   # ← update from compute_threshold.py

COLUMNS = ['FHR', 'TOCO', 'SpO2', 'RespRate', 'Temp']

print("=" * 70)
print("   EVALUATION — STATIC GAN-LABELED DATASET")
print("Current Threshold ",THRESHOLD)
print("=" * 70)

# ============================================================
# LOAD
# ============================================================
print("\n[1/5] Loading model, scaler, dataset...")
model   = load_model(MODEL_PATH, compile=False)
scaler  = joblib.load(SCALER_PATH)
df      = pd.read_csv(DATASET_PATH)

print(f"  ✅ Dataset loaded  : {len(df):,} rows")
print(f"  Normal (0)        : {(df['label']==0).sum():,}")
print(f"  Attack (1)        : {(df['label']==1).sum():,}")
print(f"  Attack types      : {df['attack_type'].unique()}")

# ============================================================
# SLIDING WINDOW CONSTRUCTION
# ============================================================
print("\n[2/5] Building sliding windows...")

sensor_values = df[COLUMNS].values
labels        = df['label'].values
attack_types  = df['attack_type'].values

X_windows     = []
y_windows     = []
type_windows  = []

for i in range(len(sensor_values) - WINDOW_SIZE):
    window       = sensor_values[i : i + WINDOW_SIZE]
    window_label = labels[i + WINDOW_SIZE - 1]        # label = last row of window
    window_type  = attack_types[i + WINDOW_SIZE - 1]

    X_windows.append(window)
    y_windows.append(window_label)
    type_windows.append(window_type)

X_windows    = np.array(X_windows)
y_windows    = np.array(y_windows)
type_windows = np.array(type_windows)

print(f"  Total windows : {len(X_windows):,}")
print(f"  Normal        : {(y_windows==0).sum():,}")
print(f"  Attack        : {(y_windows==1).sum():,}")

# ============================================================
# PREDICT
# ============================================================
print("\n[3/5] Running LSTM autoencoder predictions...")

y_scores = []
for i, window in enumerate(X_windows):
    if (i+1) % 1000 == 0:
        print(f"  {i+1}/{len(X_windows)}...")
    w     = window.reshape(1, WINDOW_SIZE, 5)
    recon = model.predict(w, verbose=0)
    y_scores.append(float(np.mean((w - recon) ** 2)))

y_scores = np.array(y_scores)
y_pred   = (y_scores > THRESHOLD).astype(int)

print("  ✅ Done")

# ============================================================
# METRICS
# ============================================================
print("\n[4/5] Computing metrics...")

accuracy    = accuracy_score(y_windows, y_pred)
precision   = precision_score(y_windows, y_pred, zero_division=0)
recall      = recall_score(y_windows, y_pred, zero_division=0)
f1          = f1_score(y_windows, y_pred, zero_division=0)
cm          = confusion_matrix(y_windows, y_pred)
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
fpr_rate    = fp / (fp + tn) if (fp + tn) > 0 else 0

fpr_roc, tpr_roc, _       = roc_curve(y_windows, y_scores)
roc_auc                    = auc(fpr_roc, tpr_roc)
prec_c, rec_c, _           = precision_recall_curve(y_windows, y_scores)
pr_auc                     = auc(rec_c, prec_c)

print("\n" + "=" * 70)
print("                    RESULTS")
print("=" * 70)
print(f"  Accuracy        : {accuracy:.4f}  ({accuracy*100:.2f}%)")
print(f"  Precision       : {precision:.4f}")
print(f"  Recall (TPR)    : {recall:.4f}")
print(f"  Specificity     : {specificity:.4f}")
print(f"  F1-Score        : {f1:.4f}")
print(f"  ROC AUC         : {roc_auc:.4f}")
print(f"  PR  AUC         : {pr_auc:.4f}")
print(f"\n  TN={tn}  FP={fp}  FN={fn}  TP={tp}")
print(f"  False Positive Rate: {fpr_rate:.4f}")

print("\n📋 Per-class Report:")
print(classification_report(y_windows, y_pred,
      target_names=["Normal","Attack"], digits=4))

print("\n🔍 Per-Attack-Type Detection:")
for atype in np.unique(type_windows):
    mask     = type_windows == atype
    detected = np.sum(y_pred[mask] == 1)
    total    = np.sum(mask)
    rate     = detected/total*100 if total > 0 else 0
    print(f"  {atype:22s}: {detected:4d}/{total}  ({rate:.1f}%)")

# ============================================================
# VISUALIZATIONS
# ============================================================
print("\n[5/5] Generating plots...")

attack_names    = [t for t in np.unique(type_windows) if t != 'Normal']
detection_rates = []
det_colors      = []
for atype in attack_names:
    mask  = type_windows == atype
    rate  = np.sum(y_pred[mask]==1) / np.sum(mask) * 100
    detection_rates.append(rate)
    det_colors.append('#2ecc71' if rate>=90 else '#f39c12' if rate>=70 else '#e74c3c')

metrics_names  = ['Accuracy','Precision','Recall','Specificity','F1-Score']
metrics_values = [accuracy, precision, recall, specificity, f1]

# Fig 1 — Confusion Matrix
plt.figure(figsize=(7,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal','Attack'],
            yticklabels=['Normal','Attack'],
            annot_kws={'size':15,'weight':'bold'})
plt.title(f'Confusion Matrix  |  Accuracy: {accuracy:.3f}', fontsize=14, fontweight='bold')
plt.ylabel('True Label'); plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix_gan.png', dpi=300, bbox_inches='tight')
print("  ✅ confusion_matrix_gan.png")
plt.close()

# Fig 2 — ROC
plt.figure(figsize=(7,5))
plt.plot(fpr_roc, tpr_roc, color='darkorange', lw=2.5, label=f'AUC = {roc_auc:.3f}')
plt.plot([0,1],[0,1], 'navy', lw=1.5, linestyle='--', label='Random')
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.title('ROC Curve', fontsize=14, fontweight='bold')
plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curve_gan.png', dpi=300, bbox_inches='tight')
print("  ✅ roc_curve_gan.png")
plt.close()

# Fig 3 — Error Distribution
plt.figure(figsize=(9,5))
plt.hist(y_scores[y_windows==0], bins=60, alpha=0.7, label='Normal',
         color='green', edgecolor='black', lw=0.5)
plt.hist(y_scores[y_windows==1], bins=60, alpha=0.7, label='Attack',
         color='red', edgecolor='black', lw=0.5)
plt.axvline(THRESHOLD, color='black', lw=2.5, linestyle='--',
            label=f'Threshold = {THRESHOLD}')
plt.xlabel('Reconstruction Error'); plt.ylabel('Frequency')
plt.title('Error Distribution — GAN Dataset', fontsize=14, fontweight='bold')
plt.legend(); plt.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('error_distribution_gan.png', dpi=300, bbox_inches='tight')
print("  ✅ error_distribution_gan.png")
plt.close()

# Fig 4 — Attack Detection Rates
plt.figure(figsize=(10,6))
bars = plt.barh(attack_names, detection_rates, color=det_colors,
                edgecolor='black', lw=1.5)
plt.xlabel('Detection Rate (%)'); plt.xlim([0,115])
plt.title('Detection Rate by Attack Type — GAN Dataset',
          fontsize=14, fontweight='bold')
for i,(b,r) in enumerate(zip(bars, detection_rates)):
    plt.text(r+1, i, f'{r:.1f}%', va='center', fontsize=11, fontweight='bold')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('attack_detection_rates_gan.png', dpi=300, bbox_inches='tight')
print("  ✅ attack_detection_rates_gan.png")
plt.close()

# Fig 5 — Combined 2x2
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal','Attack'], yticklabels=['Normal','Attack'],
            annot_kws={'size':13}, cbar=False, ax=axes[0,0])
axes[0,0].set_title(f'Confusion Matrix (Acc: {accuracy:.3f})', fontweight='bold')

axes[0,1].plot(fpr_roc, tpr_roc, 'darkorange', lw=2, label=f'AUC={roc_auc:.3f}')
axes[0,1].plot([0,1],[0,1],'navy',lw=1.5,linestyle='--')
axes[0,1].set_title('ROC Curve', fontweight='bold')
axes[0,1].legend(); axes[0,1].grid(alpha=0.3)

axes[1,0].barh(attack_names, detection_rates, color=det_colors, edgecolor='black')
for i,r in enumerate(detection_rates):
    axes[1,0].text(r+1, i, f'{r:.1f}%', va='center', fontsize=9)
axes[1,0].set_xlim([0,115])
axes[1,0].set_title('Detection by Attack Type', fontweight='bold')
axes[1,0].grid(axis='x', alpha=0.3)

axes[1,1].bar(metrics_names, metrics_values,
              color=['#2ecc71' if v>=0.9 else '#f39c12' if v>=0.8 else '#e74c3c'
                     for v in metrics_values],
              edgecolor='black')
axes[1,1].set_ylim([0,1.2])
for i,v in enumerate(metrics_values):
    axes[1,1].text(i, v+0.03, f'{v:.3f}', ha='center', fontsize=10, fontweight='bold')
axes[1,1].set_title('Performance Metrics', fontweight='bold')
axes[1,1].grid(axis='y', alpha=0.3)

plt.suptitle('IDS Evaluation — GAN-Generated Attack Dataset',
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('combined_overview_gan.png', dpi=300, bbox_inches='tight')
print("  ✅ combined_overview_gan.png")
plt.close()

# Save results
np.savez('evaluation_results_gan.npz',
         y_test=y_windows, y_scores=y_scores,
         y_pred=y_pred, attack_types=type_windows)
print("  ✅ evaluation_results_gan.npz")

print("\n" + "=" * 70)
print("  EVALUATION COMPLETE")
print("=" * 70)
grade = "A (Excellent)" if accuracy>=0.92 and f1>=0.90 else \
        "B+ (Very Good)" if accuracy>=0.88 and f1>=0.86 else \
        "B (Good)" if accuracy>=0.85 and f1>=0.83 else \
        "C (Needs Improvement)"
print(f"\n  Overall Grade : {grade}")
print("=" * 70)