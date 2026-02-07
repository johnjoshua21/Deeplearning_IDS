"""
COMPLETE IDS EVALUATION SYSTEM
- Synthetic attack injection
- Comprehensive metrics
- Clean visualizations
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
WINDOW_SIZE = 60
THRESHOLD = 1.20  # ← CHANGE THIS VALUE TO TEST DIFFERENT THRESHOLDS
DATASET_PATH = "medical_iot_ids/processed/final_5sensor_norm.csv"
MODEL_PATH = "medical_iot_ids/model/lstm_autoencoder.h5"
SCALER_PATH = "medical_iot_ids/model/scaler.pkl"

print("=" * 80)
print("     COMPREHENSIVE IDS EVALUATION WITH SYNTHETIC ATTACKS")
print("=" * 80)

# ===========================
# LOAD MODEL & DATA
# ===========================
print("\n[1/7] Loading model and data...")
model = load_model(MODEL_PATH, compile=False)
scaler = joblib.load(SCALER_PATH)
df_norm = pd.read_csv(DATASET_PATH)
data_norm = df_norm.values

print(f"✓ Model loaded")
print(f"✓ Dataset shape: {data_norm.shape}")
print(f"✓ Detection threshold: {THRESHOLD}")


# ===========================
# ATTACK INJECTION FUNCTIONS
# ===========================

def inject_dos_flooding(window):
    """DoS: Random noise + high variance"""
    attack = window.copy()
    noise = np.random.normal(0, 1.5, window.shape)
    attack += noise
    return attack


def inject_spoofing(window):
    """Spoofing: Out-of-range extreme values"""
    attack = window.copy()
    sensors_to_spoof = np.random.choice(5, size=2, replace=False)
    for s in sensors_to_spoof:
        attack[:, s] = np.random.uniform(3, 6)
    return attack


def inject_mitm(window):
    """MITM: Systematic value manipulation"""
    attack = window.copy()
    attack[:, 0] += np.random.uniform(2.0, 3.5)  # FHR spike
    attack[:, 2] -= np.random.uniform(1.5, 2.5)  # SpO2 drop
    attack[:, 4] += np.random.uniform(1.2, 2.0)  # Temp increase
    return attack


def inject_jamming(window):
    """Jamming: Zeros or constant values"""
    attack = window.copy()
    start = np.random.randint(0, WINDOW_SIZE - 20)
    jam_length = np.random.randint(10, min(30, WINDOW_SIZE - start))
    end = start + jam_length
    attack[start:end, :] = np.random.choice([-5, 0, 5], size=(jam_length, 5))
    return attack


def inject_replay(window):
    """Replay: Repeat a segment"""
    attack = window.copy()
    segment = window[:20]
    attack[20:40] = segment
    attack[40:60] = segment
    return attack


def inject_data_injection(window):
    """False data injection: Random patterns"""
    attack = window.copy()
    for i in range(WINDOW_SIZE):
        if i % 5 == 0:
            attack[i] += np.random.uniform(2, 4, 5)
    return attack


def inject_resource_exhaustion(window):
    """Resource exhaustion: Burst patterns"""
    attack = window.copy()
    burst_points = np.random.choice(WINDOW_SIZE, size=15, replace=False)
    for bp in burst_points:
        attack[bp] += np.random.uniform(2.5, 4.0, 5)
    return attack


# ===========================
# CREATE LABELED TEST SET
# ===========================

print("\n[2/7] Creating labeled test dataset...")

X_test = []
y_test = []
attack_types = []

attack_functions = {
    'DoS_Flooding': inject_dos_flooding,
    'Spoofing': inject_spoofing,
    'MITM': inject_mitm,
    'Jamming': inject_jamming,
    'Replay': inject_replay,
    'Data_Injection': inject_data_injection,
    'Resource_Exhaustion': inject_resource_exhaustion
}

# === NORMAL SAMPLES ===
print("Creating 500 normal windows...")
for i in range(1000):
    start = np.random.randint(0, len(data_norm) - WINDOW_SIZE)
    window = data_norm[start:start + WINDOW_SIZE]
    X_test.append(window)
    y_test.append(0)
    attack_types.append('Normal')

# === ATTACK SAMPLES ===
print("Creating 500 attack windows (70 of each type)...")
attacks_per_type = 150

for attack_name, attack_func in attack_functions.items():
    print(f"  Injecting {attack_name}...")
    for i in range(attacks_per_type):
        start = np.random.randint(0, len(data_norm) - WINDOW_SIZE)
        window = data_norm[start:start + WINDOW_SIZE].copy()
        attack_window = attack_func(window)
        X_test.append(attack_window)
        y_test.append(1)
        attack_types.append(attack_name)

X_test = np.array(X_test)
y_test = np.array(y_test)

print(f"\n✓ Test set created:")
print(f"  Total samples: {len(X_test)}")
print(f"  Normal: {np.sum(y_test == 0)}")
print(f"  Attack: {np.sum(y_test == 1)}")

# ===========================
# PREDICT ON TEST SET
# ===========================

print("\n[3/7] Running predictions...")

y_scores = []
y_pred = []

for i, window in enumerate(X_test):
    if (i + 1) % 200 == 0:
        print(f"  Processed {i + 1}/{len(X_test)} windows...")

    w = window.reshape(1, WINDOW_SIZE, 5)
    recon = model.predict(w, verbose=0)
    error = np.mean((w - recon) ** 2)

    y_scores.append(error)
    y_pred.append(1 if error > THRESHOLD else 0)

y_scores = np.array(y_scores)
y_pred = np.array(y_pred)

print("✓ Predictions complete")

# ===========================
# CALCULATE METRICS
# ===========================

print("\n[4/7] Calculating performance metrics...")

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
f1 = f1_score(y_test, y_pred)
fpr_rate = fp / (fp + tn) if (fp + tn) > 0 else 0

fpr, tpr, thresholds = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)

precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_scores)
pr_auc = auc(recall_curve, precision_curve)

print("✓ Metrics calculated")

# ===========================
# DISPLAY RESULTS
# ===========================

print("\n" + "=" * 80)
print("                        EVALUATION RESULTS")
print("=" * 80)

print(f"\n📊 OVERALL PERFORMANCE (Threshold = {THRESHOLD}):")
print(f"   Accuracy:        {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"   Precision:       {precision:.4f}")
print(f"   Recall (TPR):    {recall:.4f}")
print(f"   Specificity:     {specificity:.4f}")
print(f"   F1-Score:        {f1:.4f}")
print(f"   ROC AUC:         {roc_auc:.4f}")
print(f"   PR AUC:          {pr_auc:.4f}")

print(f"\n📈 CONFUSION MATRIX:")
print(f"   True Negatives:  {tn}")
print(f"   False Positives: {fp}")
print(f"   False Negatives: {fn}")
print(f"   True Positives:  {tp}")
print(f"   False Positive Rate: {fpr_rate:.4f}")

print("\n📋 DETAILED CLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred,
                            target_names=["Normal", "Attack"],
                            digits=4))

print("\n🔍 PER-ATTACK TYPE ANALYSIS:")
for attack_name in ['Normal'] + list(attack_functions.keys()):
    mask = np.array(attack_types) == attack_name
    if np.sum(mask) > 0:
        subset_true = y_test[mask]
        subset_pred = y_pred[mask]
        subset_acc = accuracy_score(subset_true, subset_pred)
        detected = np.sum(subset_pred == 1)
        total = len(subset_true)
        print(f"   {attack_name:20s}: {detected}/{total} detected ({subset_acc * 100:.1f}% accuracy)")

# ===========================
# SAVE RESULTS FOR QUICK TESTING
# ===========================

print("\n[5/7] Saving evaluation results...")
np.savez('evaluation_results.npz',
         y_test=y_test,
         y_scores=y_scores,
         y_pred=y_pred,
         attack_types=attack_types)
print("✓ Results saved to: evaluation_results.npz")

# ===========================
# CREATE CLEAN VISUALIZATIONS
# ===========================

print("\n[6/7] Generating visualizations...")

# Set style
plt.style.use('default')
sns.set_palette("husl")

# Calculate per-attack detection rates
attack_names = list(attack_functions.keys())
detection_rates = []
detection_colors = []

for attack_name in attack_names:
    mask = np.array(attack_types) == attack_name
    detected = np.sum(y_pred[mask] == 1)
    total = np.sum(mask)
    rate = detected / total * 100 if total > 0 else 0
    detection_rates.append(rate)

    if rate >= 90:
        detection_colors.append('#2ecc71')
    elif rate >= 70:
        detection_colors.append('#f39c12')
    else:
        detection_colors.append('#e74c3c')

# ========================================================================
# FIGURE 1: Confusion Matrix
# ========================================================================
fig1 = plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'Attack'],
            yticklabels=['Normal', 'Attack'],
            annot_kws={'size': 16, 'weight': 'bold'},
            cbar_kws={'label': 'Count'})
plt.title(f'Confusion Matrix\nAccuracy: {accuracy:.3f} ({accuracy * 100:.1f}%)',
          fontsize=16, fontweight='bold', pad=20)
plt.ylabel('True Label', fontsize=14, fontweight='bold')
plt.xlabel('Predicted Label', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
print("✓ Saved: confusion_matrix.png")
plt.close()

# ========================================================================
# FIGURE 2: ROC Curve
# ========================================================================
fig2 = plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=3,
         label=f'ROC Curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
         label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=14, fontweight='bold')
plt.ylabel('True Positive Rate', fontsize=14, fontweight='bold')
plt.title('ROC Curve', fontsize=16, fontweight='bold', pad=20)
plt.legend(loc="lower right", fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
print("✓ Saved: roc_curve.png")
plt.close()

# ========================================================================
# FIGURE 3: Precision-Recall Curve
# ========================================================================
fig3 = plt.figure(figsize=(8, 6))
plt.plot(recall_curve, precision_curve, color='blue', lw=3,
         label=f'PR Curve (AUC = {pr_auc:.3f})')
plt.xlabel('Recall', fontsize=14, fontweight='bold')
plt.ylabel('Precision', fontsize=14, fontweight='bold')
plt.title('Precision-Recall Curve', fontsize=16, fontweight='bold', pad=20)
plt.legend(loc="lower left", fontsize=12)
plt.grid(True, alpha=0.3)
plt.xlim([0.0, 1.05])
plt.ylim([0.0, 1.05])
plt.tight_layout()
plt.savefig('precision_recall_curve.png', dpi=300, bbox_inches='tight')
print("✓ Saved: precision_recall_curve.png")
plt.close()

# ========================================================================
# FIGURE 4: Error Distribution
# ========================================================================
fig4 = plt.figure(figsize=(10, 6))
plt.hist(y_scores[y_test == 0], bins=60, alpha=0.7, label='Normal',
         color='green', edgecolor='black', linewidth=0.5)
plt.hist(y_scores[y_test == 1], bins=60, alpha=0.7, label='Attack',
         color='red', edgecolor='black', linewidth=0.5)
plt.axvline(THRESHOLD, color='black', linestyle='--', linewidth=3,
            label=f'Threshold = {THRESHOLD}')
plt.xlabel('Reconstruction Error', fontsize=14, fontweight='bold')
plt.ylabel('Frequency', fontsize=14, fontweight='bold')
plt.title('Reconstruction Error Distribution', fontsize=16, fontweight='bold', pad=20)
plt.legend(fontsize=12, loc='upper right')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('error_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved: error_distribution.png")
plt.close()

# ========================================================================
# FIGURE 5: Attack Detection Rates
# ========================================================================
fig5 = plt.figure(figsize=(10, 7))
bars = plt.barh(attack_names, detection_rates, color=detection_colors,
                edgecolor='black', linewidth=1.5)
plt.xlabel('Detection Rate (%)', fontsize=14, fontweight='bold')
plt.title('Attack Detection Rate by Type', fontsize=16, fontweight='bold', pad=20)
plt.xlim([0, 105])

for i, (bar, rate) in enumerate(zip(bars, detection_rates)):
    plt.text(rate + 2, i, f'{rate:.1f}%',
             va='center', fontsize=12, fontweight='bold')

plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('attack_detection_rates.png', dpi=300, bbox_inches='tight')
print("✓ Saved: attack_detection_rates.png")
plt.close()

# ========================================================================
# FIGURE 6: Performance Metrics Summary
# ========================================================================
fig6 = plt.figure(figsize=(10, 6))

metrics_names = ['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1-Score']
metrics_values = [accuracy, precision, recall, specificity, f1]

colors_bars = []
for v in metrics_values:
    if v >= 0.90:
        colors_bars.append('#2ecc71')
    elif v >= 0.80:
        colors_bars.append('#f39c12')
    else:
        colors_bars.append('#e74c3c')

bars = plt.bar(metrics_names, metrics_values, color=colors_bars,
               edgecolor='black', linewidth=2)
plt.ylim([0, 1.1])
plt.ylabel('Score', fontsize=14, fontweight='bold')
plt.title('Performance Metrics Summary', fontsize=16, fontweight='bold', pad=20)
plt.xticks(fontsize=12, fontweight='bold')

for i, (bar, v) in enumerate(zip(bars, metrics_values)):
    plt.text(i, v + 0.03, f'{v:.3f}\n({v * 100:.1f}%)',
             ha='center', fontsize=11, fontweight='bold')

plt.axhline(y=0.9, color='green', linestyle='--', alpha=0.3, linewidth=2, label='Excellent (≥90%)')
plt.axhline(y=0.8, color='orange', linestyle='--', alpha=0.3, linewidth=2, label='Good (≥80%)')

plt.grid(axis='y', alpha=0.3)
plt.legend(loc='lower right', fontsize=10)
plt.tight_layout()
plt.savefig('performance_metrics.png', dpi=300, bbox_inches='tight')
print("✓ Saved: performance_metrics.png")
plt.close()

# ========================================================================
# FIGURE 7: Combined Overview (for presentations)
# ========================================================================
fig7 = plt.figure(figsize=(16, 12))

# Subplot 1: Confusion Matrix
ax1 = plt.subplot(2, 2, 1)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'Attack'],
            yticklabels=['Normal', 'Attack'],
            annot_kws={'size': 14}, cbar=False)
plt.title(f'Confusion Matrix (Acc: {accuracy:.3f})', fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)

# Subplot 2: ROC Curve
ax2 = plt.subplot(2, 2, 2)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.3f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve', fontsize=14, fontweight='bold')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)

# Subplot 3: Attack Detection Rates
ax3 = plt.subplot(2, 2, 3)
plt.barh(attack_names, detection_rates, color=detection_colors, edgecolor='black')
plt.xlabel('Detection Rate (%)', fontsize=12)
plt.title('Detection by Attack Type', fontsize=14, fontweight='bold')
plt.xlim([0, 105])
for i, rate in enumerate(detection_rates):
    plt.text(rate + 2, i, f'{rate:.1f}%', va='center', fontsize=10)
plt.grid(axis='x', alpha=0.3)

# Subplot 4: Performance Metrics
ax4 = plt.subplot(2, 2, 4)
plt.bar(metrics_names, metrics_values, color=colors_bars, edgecolor='black')
plt.ylim([0, 1.1])
plt.ylabel('Score', fontsize=12)
plt.title('Performance Metrics', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=10)
for i, v in enumerate(metrics_values):
    plt.text(i, v + 0.03, f'{v:.3f}', ha='center', fontsize=10, fontweight='bold')
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('combined_overview.png', dpi=300, bbox_inches='tight')
print("✓ Saved: combined_overview.png")
plt.close()

# ===========================
# THRESHOLD OPTIMIZATION
# ===========================

print("\n[7/7] Threshold optimization analysis...")

threshold_list = np.arange(0.1, 2.0, 0.1)
results = []

for thresh in threshold_list:
    y_pred_temp = (y_scores > thresh).astype(int)
    acc = accuracy_score(y_test, y_pred_temp)
    prec = precision_score(y_test, y_pred_temp)
    rec = recall_score(y_test, y_pred_temp)
    f1_temp = f1_score(y_test, y_pred_temp)

    results.append({
        'Threshold': thresh,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1_temp
    })

df_results = pd.DataFrame(results)

print("\n🔧 THRESHOLD OPTIMIZATION:")
print(f"\n✨ Best thresholds:")
print(f"   Accuracy:  {df_results.loc[df_results['Accuracy'].idxmax(), 'Threshold']:.2f} "
      f"({df_results['Accuracy'].max():.4f})")
print(f"   Precision: {df_results.loc[df_results['Precision'].idxmax(), 'Threshold']:.2f} "
      f"({df_results['Precision'].max():.4f})")
print(f"   Recall:    {df_results.loc[df_results['Recall'].idxmax(), 'Threshold']:.2f} "
      f"({df_results['Recall'].max():.4f})")
print(f"   F1-Score:  {df_results.loc[df_results['F1-Score'].idxmax(), 'Threshold']:.2f} "
      f"({df_results['F1-Score'].max():.4f})")

best_f1_threshold = df_results.loc[df_results['F1-Score'].idxmax(), 'Threshold']

if abs(best_f1_threshold - THRESHOLD) > 0.05:
    print(f"\n💡 Suggestion: Consider changing THRESHOLD from {THRESHOLD:.2f} to {best_f1_threshold:.2f}")

# Create threshold comparison plot
fig8 = plt.figure(figsize=(12, 5))

ax1 = plt.subplot(1, 2, 1)
ax1.plot(df_results['Threshold'], df_results['Accuracy'], 'o-', label='Accuracy', linewidth=2)
ax1.plot(df_results['Threshold'], df_results['Precision'], 's-', label='Precision', linewidth=2)
ax1.plot(df_results['Threshold'], df_results['Recall'], '^-', label='Recall', linewidth=2)
ax1.plot(df_results['Threshold'], df_results['F1-Score'], 'd-', label='F1-Score', linewidth=2)
ax1.axvline(THRESHOLD, color='red', linestyle='--', linewidth=2, label=f'Current ({THRESHOLD})')
ax1.set_xlabel('Threshold', fontsize=12, fontweight='bold')
ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
ax1.set_title('Performance vs Threshold', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

ax2 = plt.subplot(1, 2, 2)
colors = ['green' if x == df_results['F1-Score'].max() else 'steelblue'
          for x in df_results['F1-Score']]
ax2.bar(range(len(df_results)), df_results['F1-Score'], color=colors, edgecolor='black')
ax2.set_xlabel('Threshold', fontsize=12, fontweight='bold')
ax2.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
ax2.set_title('F1-Score by Threshold', fontsize=14, fontweight='bold')
ax2.set_xticks(range(0, len(df_results), 3))
ax2.set_xticklabels([f'{df_results.iloc[i]["Threshold"]:.1f}' for i in range(0, len(df_results), 3)])

plt.tight_layout()
plt.savefig('threshold_optimization.png', dpi=300, bbox_inches='tight')
print("✓ Saved: threshold_optimization.png")
plt.close()

# ===========================
# FINAL SUMMARY
# ===========================

print("\n" + "=" * 80)
print("                    EVALUATION COMPLETE! ✓")
print("=" * 80)

print("\n📁 Generated Files:")
print("   1. confusion_matrix.png           - Confusion matrix")
print("   2. roc_curve.png                  - ROC curve")
print("   3. precision_recall_curve.png     - Precision-Recall curve")
print("   4. error_distribution.png         - Error distribution")
print("   5. attack_detection_rates.png     - Per-attack detection rates")
print("   6. performance_metrics.png        - Metrics summary")
print("   7. combined_overview.png          - All-in-one view")
print("   8. threshold_optimization.png     - Threshold analysis")
print("   9. evaluation_results.npz         - Raw results for quick testing")

print("\n📊 Overall Grade: ", end='')
if accuracy >= 0.92 and f1 >= 0.90:
    print("A (Excellent)")
elif accuracy >= 0.88 and f1 >= 0.86:
    print("B+ (Very Good)")
elif accuracy >= 0.85 and f1 >= 0.83:
    print("B (Good)")
else:
    print("C (Needs Improvement)")

print("\n💡 To test different threshold without re-running everything:")
print("   python test_new_threshold.py")

print("\n" + "=" * 80)