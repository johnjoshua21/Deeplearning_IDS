"""
COMPLETE IDS EVALUATION SYSTEM
- Uses held-out X_test.npy (never seen by model)
- 7 attack types injected on top of real test windows
- Comprehensive metrics + 8 clean visualizations
- Threshold optimization
"""

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score
)
from tensorflow.keras.models import load_model

# ===========================
# CONFIGURATION
# ===========================
WINDOW_SIZE  = 60

# 💡 UPDATE THIS after running train_lstm.py
# Use the "Suggested threshold" printed at the end of training
THRESHOLD    = 0.30   # ← UPDATE THIS with value from train_lstm.py output

MODEL_PATH   = "medical_iot_ids/model/lstm_autoencoder.h5"
SCALER_PATH  = "medical_iot_ids/model/scaler.pkl"
X_TEST_PATH  = "medical_iot_ids/model/X_test.npy"       # held-out test set from train_lstm.py

print("=" * 80)
print("     COMPLETE IDS EVALUATION — USING HELD-OUT TEST SET")
print("=" * 80)

# ===========================
# LOAD MODEL & DATA
# ===========================
print("\n[1/7] Loading model and test data...")
model  = load_model(MODEL_PATH, compile=False)
scaler = joblib.load(SCALER_PATH)

# Load the held-out test windows (never seen by model during training)
X_test_normal = np.load(X_TEST_PATH)

print(f"  ✅ Model loaded")
print(f"  ✅ Held-out test windows : {X_test_normal.shape}")
print(f"  ✅ Detection threshold   : {THRESHOLD}")

# ===========================
# ATTACK INJECTION FUNCTIONS
# ===========================

def inject_dos_flooding(window):
    """DoS: Random noise + high variance"""
    attack = window.copy()
    attack += np.random.normal(0, 1.5, window.shape)
    return attack

def inject_spoofing(window):
    """Spoofing: Out-of-range extreme values"""
    attack = window.copy()
    for s in np.random.choice(5, size=2, replace=False):
        attack[:, s] = np.random.uniform(3, 6)
    return attack

def inject_mitm(window):
    """MITM: Systematic value manipulation"""
    attack = window.copy()
    attack[:, 0] += np.random.uniform(2.0, 3.5)   # FHR spike
    attack[:, 2] -= np.random.uniform(1.5, 2.5)   # SpO2 drop
    attack[:, 4] += np.random.uniform(1.2, 2.0)   # Temp increase
    return attack

def inject_jamming(window):
    """Jamming: Zeros or constant values"""
    attack = window.copy()
    start      = np.random.randint(0, WINDOW_SIZE - 20)
    jam_length = np.random.randint(10, min(30, WINDOW_SIZE - start))
    attack[start:start + jam_length, :] = np.random.choice([-5, 0, 5], size=(jam_length, 5))
    return attack

def inject_replay(window):
    """Replay: Repeat a segment"""
    attack = window.copy()
    segment      = window[:20]
    attack[20:40] = segment
    attack[40:60] = segment
    return attack

def inject_data_injection(window):
    """False data injection: Random periodic pattern"""
    attack = window.copy()
    for i in range(WINDOW_SIZE):
        if i % 5 == 0:
            attack[i] += np.random.uniform(2, 4, 5)
    return attack

def inject_resource_exhaustion(window):
    """Resource exhaustion: Burst spikes"""
    attack = window.copy()
    for bp in np.random.choice(WINDOW_SIZE, size=15, replace=False):
        attack[bp] += np.random.uniform(2.5, 4.0, 5)
    return attack

attack_functions = {
    'DoS_Flooding'       : inject_dos_flooding,
    'Spoofing'           : inject_spoofing,
    'MITM'               : inject_mitm,
    'Jamming'            : inject_jamming,
    'Replay'             : inject_replay,
    'Data_Injection'     : inject_data_injection,
    'Resource_Exhaustion': inject_resource_exhaustion
}

# ===========================
# BUILD LABELED TEST SET
# ===========================
print("\n[2/7] Building labeled test dataset...")

# How many normal windows we have
n_normal = len(X_test_normal)

# Attacks per type — match normal count for balanced dataset
attacks_per_type = max(50, n_normal // len(attack_functions))

X_eval       = []
y_eval       = []
attack_types = []

# --- NORMAL SAMPLES (from held-out test set) ---
print(f"  Adding {n_normal} normal windows from held-out test set...")
for w in X_test_normal:
    X_eval.append(w)
    y_eval.append(0)
    attack_types.append('Normal')

# --- ATTACK SAMPLES (inject on top of test windows) ---
print(f"  Injecting {attacks_per_type} windows per attack type ({len(attack_functions)} types)...")
for attack_name, attack_func in attack_functions.items():
    for i in range(attacks_per_type):
        idx    = np.random.randint(0, n_normal)
        window = X_test_normal[idx].copy()
        X_eval.append(attack_func(window))
        y_eval.append(1)
        attack_types.append(attack_name)
    print(f"    ✅ {attack_name} — {attacks_per_type} windows injected")

X_eval       = np.array(X_eval)
y_eval       = np.array(y_eval)
attack_types = np.array(attack_types)

print(f"\n  Total samples  : {len(X_eval)}")
print(f"  Normal         : {np.sum(y_eval == 0)}")
print(f"  Attack         : {np.sum(y_eval == 1)}")

# ===========================
# PREDICT
# ===========================
print("\n[3/7] Running predictions...")

y_scores = []
y_pred   = []

for i, window in enumerate(X_eval):
    if (i + 1) % 500 == 0:
        print(f"  Processed {i + 1}/{len(X_eval)}...")

    w     = window.reshape(1, WINDOW_SIZE, 5)
    recon = model.predict(w, verbose=0)
    error = float(np.mean((w - recon) ** 2))

    y_scores.append(error)
    y_pred.append(1 if error > THRESHOLD else 0)

y_scores = np.array(y_scores)
y_pred   = np.array(y_pred)

print("  ✅ Predictions complete")

# ===========================
# METRICS
# ===========================
print("\n[4/7] Calculating metrics...")

accuracy    = accuracy_score(y_eval, y_pred)
cm          = confusion_matrix(y_eval, y_pred)
tn, fp, fn, tp = cm.ravel()

precision   = precision_score(y_eval, y_pred)
recall      = recall_score(y_eval, y_pred)
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
f1          = f1_score(y_eval, y_pred)
fpr_rate    = fp / (fp + tn) if (fp + tn) > 0 else 0

fpr, tpr, _          = roc_curve(y_eval, y_scores)
roc_auc              = auc(fpr, tpr)
prec_curve, rec_curve, _ = precision_recall_curve(y_eval, y_scores)
pr_auc               = auc(rec_curve, prec_curve)

# ===========================
# PRINT RESULTS
# ===========================
print("\n" + "=" * 80)
print("                        EVALUATION RESULTS")
print("=" * 80)

print(f"\n📊 OVERALL PERFORMANCE (Threshold = {THRESHOLD}):")
print(f"   Accuracy        : {accuracy:.4f}  ({accuracy*100:.2f}%)")
print(f"   Precision       : {precision:.4f}")
print(f"   Recall (TPR)    : {recall:.4f}")
print(f"   Specificity     : {specificity:.4f}")
print(f"   F1-Score        : {f1:.4f}")
print(f"   ROC AUC         : {roc_auc:.4f}")
print(f"   PR  AUC         : {pr_auc:.4f}")

print(f"\n📈 CONFUSION MATRIX:")
print(f"   True Negatives  : {tn}")
print(f"   False Positives : {fp}")
print(f"   False Negatives : {fn}")
print(f"   True Positives  : {tp}")
print(f"   False Pos Rate  : {fpr_rate:.4f}")

print("\n📋 CLASSIFICATION REPORT:")
print(classification_report(y_eval, y_pred,
                            target_names=["Normal", "Attack"], digits=4))

print("\n🔍 PER-ATTACK TYPE ANALYSIS:")
for name in ['Normal'] + list(attack_functions.keys()):
    mask     = attack_types == name
    if np.sum(mask) == 0:
        continue
    detected = np.sum(y_pred[mask] == 1)
    total    = np.sum(mask)
    acc      = accuracy_score(y_eval[mask], y_pred[mask])
    print(f"   {name:22s}: {detected:4d}/{total} detected  ({acc*100:.1f}%)")

# ===========================
# SAVE RAW RESULTS
# ===========================
print("\n[5/7] Saving raw results...")
np.savez('evaluation_results.npz',
         y_test=y_eval,
         y_scores=y_scores,
         y_pred=y_pred,
         attack_types=attack_types)
print("  ✅ Saved: evaluation_results.npz")

# ===========================
# VISUALIZATIONS
# ===========================
print("\n[6/7] Generating visualizations...")

plt.style.use('default')

# --- Per-attack detection rates ---
attack_names     = list(attack_functions.keys())
detection_rates  = []
detection_colors = []

for name in attack_names:
    mask     = attack_types == name
    detected = np.sum(y_pred[mask] == 1)
    total    = np.sum(mask)
    rate     = detected / total * 100 if total > 0 else 0
    detection_rates.append(rate)
    detection_colors.append(
        '#2ecc71' if rate >= 90 else '#f39c12' if rate >= 70 else '#e74c3c'
    )

metrics_names  = ['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1-Score']
metrics_values = [accuracy, precision, recall, specificity, f1]
colors_bars    = ['#2ecc71' if v >= 0.90 else '#f39c12' if v >= 0.80 else '#e74c3c'
                  for v in metrics_values]

# ---- Figure 1: Confusion Matrix ----
fig1 = plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'Attack'],
            yticklabels=['Normal', 'Attack'],
            annot_kws={'size': 16, 'weight': 'bold'})
plt.title(f'Confusion Matrix\nAccuracy: {accuracy:.3f} ({accuracy*100:.1f}%)',
          fontsize=16, fontweight='bold', pad=20)
plt.ylabel('True Label',      fontsize=14, fontweight='bold')
plt.xlabel('Predicted Label', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
print("  ✅ confusion_matrix.png")
plt.close()

# ---- Figure 2: ROC Curve ----
fig2 = plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=3, label=f'ROC Curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=14, fontweight='bold')
plt.ylabel('True Positive Rate',  fontsize=14, fontweight='bold')
plt.title('ROC Curve', fontsize=16, fontweight='bold', pad=20)
plt.legend(loc="lower right", fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
print("  ✅ roc_curve.png")
plt.close()

# ---- Figure 3: Precision-Recall Curve ----
fig3 = plt.figure(figsize=(8, 6))
plt.plot(rec_curve, prec_curve, color='blue', lw=3, label=f'PR Curve (AUC = {pr_auc:.3f})')
plt.xlabel('Recall',    fontsize=14, fontweight='bold')
plt.ylabel('Precision', fontsize=14, fontweight='bold')
plt.title('Precision-Recall Curve', fontsize=16, fontweight='bold', pad=20)
plt.legend(loc="lower left", fontsize=12)
plt.grid(True, alpha=0.3)
plt.xlim([0.0, 1.05]); plt.ylim([0.0, 1.05])
plt.tight_layout()
plt.savefig('precision_recall_curve.png', dpi=300, bbox_inches='tight')
print("  ✅ precision_recall_curve.png")
plt.close()

# ---- Figure 4: Error Distribution ----
fig4 = plt.figure(figsize=(10, 6))
plt.hist(y_scores[y_eval == 0], bins=60, alpha=0.7, label='Normal',
         color='green', edgecolor='black', linewidth=0.5)
plt.hist(y_scores[y_eval == 1], bins=60, alpha=0.7, label='Attack',
         color='red',   edgecolor='black', linewidth=0.5)
plt.axvline(THRESHOLD, color='black', linestyle='--', linewidth=3,
            label=f'Threshold = {THRESHOLD}')
plt.xlabel('Reconstruction Error', fontsize=14, fontweight='bold')
plt.ylabel('Frequency',            fontsize=14, fontweight='bold')
plt.title('Reconstruction Error Distribution', fontsize=16, fontweight='bold', pad=20)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('error_distribution.png', dpi=300, bbox_inches='tight')
print("  ✅ error_distribution.png")
plt.close()

# ---- Figure 5: Attack Detection Rates ----
fig5 = plt.figure(figsize=(10, 7))
bars = plt.barh(attack_names, detection_rates, color=detection_colors,
                edgecolor='black', linewidth=1.5)
plt.xlabel('Detection Rate (%)', fontsize=14, fontweight='bold')
plt.title('Attack Detection Rate by Type', fontsize=16, fontweight='bold', pad=20)
plt.xlim([0, 110])
for i, (bar, rate) in enumerate(zip(bars, detection_rates)):
    plt.text(rate + 1, i, f'{rate:.1f}%', va='center', fontsize=12, fontweight='bold')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('attack_detection_rates.png', dpi=300, bbox_inches='tight')
print("  ✅ attack_detection_rates.png")
plt.close()

# ---- Figure 6: Performance Metrics ----
fig6 = plt.figure(figsize=(10, 6))
bars = plt.bar(metrics_names, metrics_values, color=colors_bars,
               edgecolor='black', linewidth=2)
plt.ylim([0, 1.15])
plt.ylabel('Score', fontsize=14, fontweight='bold')
plt.title('Performance Metrics Summary', fontsize=16, fontweight='bold', pad=20)
plt.xticks(fontsize=12, fontweight='bold')
for i, (bar, v) in enumerate(zip(bars, metrics_values)):
    plt.text(i, v + 0.03, f'{v:.3f}\n({v*100:.1f}%)',
             ha='center', fontsize=11, fontweight='bold')
plt.axhline(y=0.9, color='green',  linestyle='--', alpha=0.3, linewidth=2, label='Excellent (≥90%)')
plt.axhline(y=0.8, color='orange', linestyle='--', alpha=0.3, linewidth=2, label='Good (≥80%)')
plt.legend(loc='lower right', fontsize=10)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('performance_metrics.png', dpi=300, bbox_inches='tight')
print("  ✅ performance_metrics.png")
plt.close()

# ---- Figure 7: Combined Overview ----
fig7 = plt.figure(figsize=(16, 12))

ax1 = plt.subplot(2, 2, 1)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'Attack'],
            yticklabels=['Normal', 'Attack'],
            annot_kws={'size': 14}, cbar=False)
plt.title(f'Confusion Matrix (Acc: {accuracy:.3f})', fontsize=14, fontweight='bold')
plt.ylabel('True Label'); plt.xlabel('Predicted Label')

ax2 = plt.subplot(2, 2, 2)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.3f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.title('ROC Curve', fontsize=14, fontweight='bold')
plt.legend(loc="lower right"); plt.grid(alpha=0.3)

ax3 = plt.subplot(2, 2, 3)
plt.barh(attack_names, detection_rates, color=detection_colors, edgecolor='black')
plt.xlabel('Detection Rate (%)'); plt.xlim([0, 110])
plt.title('Detection by Attack Type', fontsize=14, fontweight='bold')
for i, rate in enumerate(detection_rates):
    plt.text(rate + 1, i, f'{rate:.1f}%', va='center', fontsize=10)
plt.grid(axis='x', alpha=0.3)

ax4 = plt.subplot(2, 2, 4)
plt.bar(metrics_names, metrics_values, color=colors_bars, edgecolor='black')
plt.ylim([0, 1.15]); plt.ylabel('Score')
plt.title('Performance Metrics', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=10)
for i, v in enumerate(metrics_values):
    plt.text(i, v + 0.03, f'{v:.3f}', ha='center', fontsize=10, fontweight='bold')
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('combined_overview.png', dpi=300, bbox_inches='tight')
print("  ✅ combined_overview.png")
plt.close()

# ===========================
# THRESHOLD OPTIMIZATION
# ===========================
print("\n[7/7] Threshold optimization...")

results = []
for thresh in np.arange(0.1, 3.0, 0.1):
    y_pred_t = (y_scores > thresh).astype(int)
    results.append({
        'Threshold' : round(thresh, 2),
        'Accuracy'  : accuracy_score(y_eval, y_pred_t),
        'Precision' : precision_score(y_eval, y_pred_t, zero_division=0),
        'Recall'    : recall_score(y_eval, y_pred_t, zero_division=0),
        'F1-Score'  : f1_score(y_eval, y_pred_t, zero_division=0)
    })

df_results = pd.DataFrame(results)

best_f1_row  = df_results.loc[df_results['F1-Score'].idxmax()]
best_acc_row = df_results.loc[df_results['Accuracy'].idxmax()]

print(f"\n✨ Best Thresholds:")
print(f"   F1-Score  → threshold {best_f1_row['Threshold']:.2f}  (F1 = {best_f1_row['F1-Score']:.4f})")
print(f"   Accuracy  → threshold {best_acc_row['Threshold']:.2f}  (Acc = {best_acc_row['Accuracy']:.4f})")

if abs(best_f1_row['Threshold'] - THRESHOLD) > 0.05:
    print(f"\n💡 Suggestion: Update THRESHOLD from {THRESHOLD} → {best_f1_row['Threshold']:.2f}")

# Threshold plot
fig8, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(df_results['Threshold'], df_results['Accuracy'],  'o-', label='Accuracy',  linewidth=2)
ax1.plot(df_results['Threshold'], df_results['Precision'], 's-', label='Precision', linewidth=2)
ax1.plot(df_results['Threshold'], df_results['Recall'],    '^-', label='Recall',    linewidth=2)
ax1.plot(df_results['Threshold'], df_results['F1-Score'],  'd-', label='F1-Score',  linewidth=2)
ax1.axvline(THRESHOLD, color='red', linestyle='--', linewidth=2, label=f'Current ({THRESHOLD})')
ax1.set_xlabel('Threshold', fontsize=12, fontweight='bold')
ax1.set_ylabel('Score',     fontsize=12, fontweight='bold')
ax1.set_title('Performance vs Threshold', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10); ax1.grid(True, alpha=0.3)

colors_t = ['green' if x == df_results['F1-Score'].max() else 'steelblue'
            for x in df_results['F1-Score']]
ax2.bar(range(len(df_results)), df_results['F1-Score'], color=colors_t, edgecolor='black')
ax2.set_xlabel('Threshold Index', fontsize=12, fontweight='bold')
ax2.set_ylabel('F1-Score',        fontsize=12, fontweight='bold')
ax2.set_title('F1-Score by Threshold', fontsize=14, fontweight='bold')
ax2.set_xticks(range(0, len(df_results), 3))
ax2.set_xticklabels([f"{df_results.iloc[i]['Threshold']:.1f}"
                     for i in range(0, len(df_results), 3)])
plt.tight_layout()
plt.savefig('threshold_optimization.png', dpi=300, bbox_inches='tight')
print("  ✅ threshold_optimization.png")
plt.close()

# ===========================
# FINAL SUMMARY
# ===========================
print("\n" + "=" * 80)
print("                    EVALUATION COMPLETE ✓")
print("=" * 80)
print(f"\n  Data source    : Held-out X_test.npy (never seen by model)")
print(f"  Normal windows : {np.sum(y_eval == 0)}")
print(f"  Attack windows : {np.sum(y_eval == 1)}")
print(f"  Attack types   : {len(attack_functions)}")

print("\n📁 Generated Files:")
print("   1. confusion_matrix.png")
print("   2. roc_curve.png")
print("   3. precision_recall_curve.png")
print("   4. error_distribution.png")
print("   5. attack_detection_rates.png")
print("   6. performance_metrics.png")
print("   7. combined_overview.png")
print("   8. threshold_optimization.png")
print("   9. evaluation_results.npz")

print("\n📊 Overall Grade: ", end='')
if accuracy >= 0.92 and f1 >= 0.90:
    print("A  (Excellent)")
elif accuracy >= 0.88 and f1 >= 0.86:
    print("B+ (Very Good)")
elif accuracy >= 0.85 and f1 >= 0.83:
    print("B  (Good)")
else:
    print("C  (Needs Improvement — consider updating THRESHOLD)")

print("\n" + "=" * 80)