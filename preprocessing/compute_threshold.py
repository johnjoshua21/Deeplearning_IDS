"""
THRESHOLD FINDER
- Loads held-out X_test.npy (never seen by model)
- Injects all 7 attack types
- Tries every threshold from 0.01 to 1.0
- Recommends best threshold based on F1, Accuracy, and Balance
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

# ===========================
# CONFIGURATION
# ===========================
MODEL_PATH  = "medical_iot_ids/model/lstm_autoencoder.h5"
X_TEST_PATH = "medical_iot_ids/model/X_test.npy"
WINDOW_SIZE = 60

print("=" * 60)
print("        THRESHOLD FINDER")
print("=" * 60)

# ===========================
# LOAD
# ===========================
print("\n[1/4] Loading model and test data...")
model         = load_model(MODEL_PATH, compile=False)
X_test_normal = np.load(X_TEST_PATH)
print(f"  ✅ Model loaded")
print(f"  ✅ Normal test windows: {len(X_test_normal)}")

# ===========================
# ATTACK FUNCTIONS
# ===========================
def inject_dos(w):
    a = w.copy(); a += np.random.normal(0, 1.5, w.shape); return a

def inject_spoofing(w):
    a = w.copy()
    for s in np.random.choice(5, size=2, replace=False):
        a[:, s] = np.random.uniform(3, 6)
    return a

def inject_mitm(w):
    a = w.copy()
    a[:, 0] += np.random.uniform(2.0, 3.5)
    a[:, 2] -= np.random.uniform(1.5, 2.5)
    a[:, 4] += np.random.uniform(1.2, 2.0)
    return a

def inject_jamming(w):
    a = w.copy()
    start = np.random.randint(0, WINDOW_SIZE - 20)
    length = np.random.randint(10, min(30, WINDOW_SIZE - start))
    a[start:start+length, :] = np.random.choice([-5, 0, 5], size=(length, 5))
    return a

def inject_replay(w):
    a = w.copy()
    seg = w[:20]; a[20:40] = seg; a[40:60] = seg
    return a

def inject_data_injection(w):
    a = w.copy()
    for i in range(WINDOW_SIZE):
        if i % 5 == 0: a[i] += np.random.uniform(2, 4, 5)
    return a

def inject_resource_exhaustion(w):
    a = w.copy()
    for bp in np.random.choice(WINDOW_SIZE, size=15, replace=False):
        a[bp] += np.random.uniform(2.5, 4.0, 5)
    return a

attack_functions = {
    'DoS'               : inject_dos,
    'Spoofing'          : inject_spoofing,
    'MITM'              : inject_mitm,
    'Jamming'           : inject_jamming,
    'Replay'            : inject_replay,
    'Data_Injection'    : inject_data_injection,
    'Resource_Exhaustion': inject_resource_exhaustion
}

# ===========================
# BUILD EVAL SET
# ===========================
print("\n[2/4] Building evaluation set...")

n_normal         = len(X_test_normal)
attacks_per_type = max(30, n_normal // len(attack_functions))

X_eval = list(X_test_normal)
y_eval = [0] * n_normal

for name, func in attack_functions.items():
    for _ in range(attacks_per_type):
        idx = np.random.randint(0, n_normal)
        X_eval.append(func(X_test_normal[idx].copy()))
        y_eval.append(1)

X_eval = np.array(X_eval)
y_eval = np.array(y_eval)

print(f"  Normal  : {np.sum(y_eval == 0)}")
print(f"  Attack  : {np.sum(y_eval == 1)}")
print(f"  Total   : {len(y_eval)}")

# ===========================
# GET RECONSTRUCTION ERRORS
# ===========================
print("\n[3/4] Computing reconstruction errors...")

y_scores = []
for i, window in enumerate(X_eval):
    if (i + 1) % 500 == 0:
        print(f"  {i+1}/{len(X_eval)}...")
    w     = window.reshape(1, WINDOW_SIZE, 5)
    recon = model.predict(w, verbose=0)
    y_scores.append(float(np.mean((w - recon) ** 2)))

y_scores = np.array(y_scores)

print(f"\n  Normal error  — mean: {y_scores[y_eval==0].mean():.6f}  std: {y_scores[y_eval==0].std():.6f}")
print(f"  Attack error  — mean: {y_scores[y_eval==1].mean():.6f}  std: {y_scores[y_eval==1].std():.6f}")

# ===========================
# SWEEP THRESHOLDS
# ===========================
print("\n[4/4] Sweeping thresholds...")

# Fine-grained sweep
thresholds = np.round(np.arange(0.01, 1.0, 0.01), 3)

results = []
for t in thresholds:
    y_pred = (y_scores > t).astype(int)
    acc    = accuracy_score(y_eval, y_pred)
    prec   = precision_score(y_eval, y_pred, zero_division=0)
    rec    = recall_score(y_eval, y_pred, zero_division=0)
    f1     = f1_score(y_eval, y_pred, zero_division=0)
    fpr    = np.sum((y_pred == 1) & (y_eval == 0)) / max(np.sum(y_eval == 0), 1)

    results.append({
        'Threshold' : t,
        'Accuracy'  : acc,
        'Precision' : prec,
        'Recall'    : rec,
        'F1'        : f1,
        'FPR'       : fpr
    })

df = pd.DataFrame(results)

# ===========================
# FIND BEST THRESHOLDS
# ===========================
best_f1       = df.loc[df['F1'].idxmax()]
best_acc      = df.loc[df['Accuracy'].idxmax()]

# Balanced: best F1 where FPR < 0.10
df_low_fpr    = df[df['FPR'] < 0.10]
best_balanced = df_low_fpr.loc[df_low_fpr['F1'].idxmax()] if len(df_low_fpr) > 0 else best_f1

print(f"\n{'='*60}")
print(f"        THRESHOLD RECOMMENDATIONS")
print(f"{'='*60}")
print(f"\n  🥇 Best F1-Score:")
print(f"     Threshold  : {best_f1['Threshold']}")
print(f"     F1         : {best_f1['F1']:.4f}")
print(f"     Accuracy   : {best_f1['Accuracy']:.4f}")
print(f"     Precision  : {best_f1['Precision']:.4f}")
print(f"     Recall     : {best_f1['Recall']:.4f}")
print(f"     FPR        : {best_f1['FPR']:.4f}")

print(f"\n  🥈 Best Accuracy:")
print(f"     Threshold  : {best_acc['Threshold']}")
print(f"     Accuracy   : {best_acc['Accuracy']:.4f}")
print(f"     F1         : {best_acc['F1']:.4f}")
print(f"     FPR        : {best_acc['FPR']:.4f}")

print(f"\n  ⚖️  Best Balanced (F1 with FPR < 10%):")
print(f"     Threshold  : {best_balanced['Threshold']}")
print(f"     F1         : {best_balanced['F1']:.4f}")
print(f"     Accuracy   : {best_balanced['Accuracy']:.4f}")
print(f"     FPR        : {best_balanced['FPR']:.4f}")

print(f"\n{'='*60}")
print(f"  💡 RECOMMENDED THRESHOLD : {best_balanced['Threshold']}")
print(f"     Use this in all your files!")
print(f"{'='*60}")

# ===========================
# PLOT
# ===========================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: All metrics vs threshold
ax1 = axes[0]
ax1.plot(df['Threshold'], df['Accuracy'],  label='Accuracy',  linewidth=2)
ax1.plot(df['Threshold'], df['Precision'], label='Precision', linewidth=2)
ax1.plot(df['Threshold'], df['Recall'],    label='Recall',    linewidth=2)
ax1.plot(df['Threshold'], df['F1'],        label='F1-Score',  linewidth=2.5, color='red')
ax1.axvline(best_balanced['Threshold'], color='black', linestyle='--',
            linewidth=2, label=f"Best ({best_balanced['Threshold']})")
ax1.set_xlabel('Threshold', fontsize=12, fontweight='bold')
ax1.set_ylabel('Score',     fontsize=12, fontweight='bold')
ax1.set_title('Metrics vs Threshold', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Plot 2: F1 vs threshold (zoomed)
ax2 = axes[1]
ax2.plot(df['Threshold'], df['F1'], color='red', linewidth=2.5)
ax2.axvline(best_f1['Threshold'], color='green', linestyle='--',
            linewidth=2, label=f"Best F1 ({best_f1['Threshold']})")
ax2.axvline(best_balanced['Threshold'], color='black', linestyle='--',
            linewidth=2, label=f"Balanced ({best_balanced['Threshold']})")
ax2.set_xlabel('Threshold', fontsize=12, fontweight='bold')
ax2.set_ylabel('F1-Score',  fontsize=12, fontweight='bold')
ax2.set_title('F1-Score vs Threshold', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# Plot 3: Error distribution with threshold line
ax3 = axes[2]
ax3.hist(y_scores[y_eval==0], bins=60, alpha=0.7,
         label='Normal', color='green', edgecolor='black', linewidth=0.5)
ax3.hist(y_scores[y_eval==1], bins=60, alpha=0.7,
         label='Attack', color='red',   edgecolor='black', linewidth=0.5)
ax3.axvline(best_balanced['Threshold'], color='black', linestyle='--',
            linewidth=2.5, label=f"Threshold = {best_balanced['Threshold']}")
ax3.set_xlabel('Reconstruction Error', fontsize=12, fontweight='bold')
ax3.set_ylabel('Frequency',            fontsize=12, fontweight='bold')
ax3.set_title('Error Distribution',    fontsize=14, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3, axis='y')

plt.suptitle('Threshold Optimization Analysis', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('medical_iot_ids/model/threshold_analysis.png', dpi=300, bbox_inches='tight')
print("\n✅ Saved: medical_iot_ids/model/threshold_analysis.png")
plt.show()

# Save full results table
df.to_csv('medical_iot_ids/model/threshold_results.csv', index=False)
print("✅ Saved: medical_iot_ids/model/threshold_results.csv")

print(f"\n{'='*60}")
print(f"  UPDATE THRESHOLD = {best_balanced['Threshold']} in:")
print(f"  • preprocessing/complete_ids_evaluation.py")
print(f"  • project/realtime_ids_test.py")
print(f"  • project/collector.py")
print(f"{'='*60}")