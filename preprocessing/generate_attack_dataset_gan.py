"""
GAN-BASED ATTACK DATASET GENERATOR
=====================================
Separate file — does NOT disturb generate_attack_dataset.py

Fixes applied (learned from direct injection success):
  Fix 1: Uses final_5sensor_norm.csv (already normalized — no double scaling)
  Fix 2: NO shuffle — normal rows first, attack rows second (pure windows)
  Fix 3: Strong perturbation magnitudes (4-8 std devs)
  Fix 4: GAN trained on normalized data so samples are in correct scale

Output:
  medical_iot_ids/processed/labeled_attack_dataset_gan.csv
  medical_iot_ids/model/ctgan_model_v2.pkl

Columns: FHR, TOCO, SpO2, RespRate, Temp, label, attack_type
"""

import numpy as np
import pandas as pd
import joblib
import os

try:
    from ctgan import CTGAN
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ctgan"])
    from ctgan import CTGAN

# ============================================================
# CONFIGURATION
# ============================================================
NORM_PATH    = "medical_iot_ids/processed/final_5sensor_norm.csv"
OUTPUT_PATH  = "medical_iot_ids/processed/labeled_attack_dataset_gan.csv"
MODEL_SAVE   = "medical_iot_ids/model/ctgan_model_v2.pkl"

SAMPLES_PER_ATTACK = 500
RANDOM_SEED        = 42
COLUMNS = ['FHR', 'TOCO', 'SpO2', 'RespRate', 'Temp']

os.makedirs("medical_iot_ids/model",     exist_ok=True)
os.makedirs("medical_iot_ids/processed", exist_ok=True)
np.random.seed(RANDOM_SEED)

print("=" * 70)
print("  GAN-BASED ATTACK DATASET GENERATOR (FIXED)")
print("=" * 70)
print("  Uses: final_5sensor_norm.csv (normalized)")
print("  Output: labeled_attack_dataset_gan.csv")
print("  NO shuffle — pure windows guaranteed")
print("=" * 70)

# ============================================================
# STEP 1 — LOAD NORMALIZED DATA
# ============================================================
print("\n[1/6] Loading normalized sensor data...")

df_norm = pd.read_csv(NORM_PATH)[COLUMNS].dropna().reset_index(drop=True)

print(f"  Rows loaded : {len(df_norm):,}")
print(f"  Data range  : {df_norm.values.min():.3f} to {df_norm.values.max():.3f}")
print(f"  Mean        : {df_norm.values.mean():.4f}  (should be ~0)")
print(f"  Std         : {df_norm.values.std():.4f}   (should be ~1)")
print(df_norm.describe().round(3))

# ============================================================
# STEP 2 — TRAIN CTGAN ON NORMALIZED DATA
# ============================================================
print("\n[2/6] Training CTGAN on normalized sensor data...")
print("  CTGAN learns the real joint distribution of all 5 sensors")
print("  in normalized (z-score) space — same scale as LSTM training.")
print("  (Takes 5-10 minutes)\n")

MAX_CTGAN_ROWS = 15000
if len(df_norm) > MAX_CTGAN_ROWS:
    df_train = df_norm.sample(n=MAX_CTGAN_ROWS, random_state=RANDOM_SEED)
    print(f"  Sampled {MAX_CTGAN_ROWS:,} rows for training")
else:
    df_train = df_norm.copy()

ctgan = CTGAN(
    epochs     = 300,
    batch_size = 500,
    verbose    = True
)
ctgan.fit(df_train, discrete_columns=[])

joblib.dump(ctgan, MODEL_SAVE)
print(f"\n  ✅ CTGAN trained and saved: {MODEL_SAVE}")

# Verify GAN output is in correct scale
test_sample = ctgan.sample(100)
print(f"\n  GAN sample verification:")
print(f"    Mean : {test_sample.values.mean():.4f}  (should be near 0)")
print(f"    Std  : {test_sample.values.std():.4f}   (should be near 1)")
print(f"    Min  : {test_sample.values.min():.4f}")
print(f"    Max  : {test_sample.values.max():.4f}")

# ============================================================
# STEP 3 — ATTACK PERTURBATIONS (normalized z-score units)
# ============================================================
"""
All perturbations in z-score units.
Normal data range: -2 to +2 (approximately).
Perturbations push values to 4-8 std devs outside normal range.

Clinical grounding (raw → z-score):
  FHR     std ≈ 10 bpm   → 4 std = 40 bpm shift (severe tachycardia)
  SpO2    std ≈ 1.5%     → 5 std = 7.5% drop (critical hypoxia)
  Temp    std ≈ 0.3°C    → 5 std = 1.5°C rise (high fever)
  RespRate std ≈ 2 br/min→ 4 std = 8 br/min (severe tachypnea)
  TOCO    std ≈ 20       → 4 std = 80 units
"""

def attack_dos(base):
    """High variance noise — 4-6 std devs across all sensors"""
    out = base.copy()
    for col in COLUMNS:
        out[col] += np.random.normal(0, 4.5, len(out))
    return out


def attack_spoofing(base):
    """Extreme out-of-range values — sensors replaced with implausible readings"""
    out = base.copy()
    mask = np.random.rand(len(out)) < 0.5
    out.loc[ mask, 'FHR'] = np.random.uniform(-7, -4, int(mask.sum()))
    out.loc[~mask, 'FHR'] = np.random.uniform( 4,  7, int((~mask).sum()))
    out['SpO2']     = np.random.uniform(-9, -5, len(out))
    out['Temp']     = np.random.uniform( 5,  8, len(out))
    out['RespRate'] = np.random.uniform( 4,  7, len(out))
    return out


def attack_mitm(base):
    """Correlated cross-sensor shift — physiologically impossible pattern"""
    out = base.copy()
    out['FHR']      += np.random.uniform(4.0, 6.0, len(out))
    out['SpO2']     -= np.random.uniform(5.0, 8.0, len(out))
    out['Temp']     += np.random.uniform(4.5, 7.0, len(out))
    out['RespRate'] += np.random.uniform(3.5, 5.5, len(out))
    out['TOCO']     += np.random.uniform(3.0, 5.0, len(out))
    return out


def attack_jamming(base):
    """Flatline — all sensors frozen at extreme constant value"""
    out  = base.copy()
    jam  = float(np.random.choice([-6, -5, 5, 6]))
    for col in COLUMNS:
        out[col] = jam + np.random.normal(0, 0.01, len(out))
    return out


def attack_replay(base):
    """Frozen readings with offset — temporal anomaly"""
    out = base.copy()
    for col in COLUMNS:
        offset = np.random.uniform(3.5, 5.5) * np.random.choice([-1, 1])
        frozen = float(out[col].median()) + offset
        out[col] = frozen + np.random.normal(0, 0.005, len(out))
    return out


def attack_data_injection(base):
    """Periodic spikes every 5th sample — 5-7 std devs"""
    out = base.copy()
    spike_idx  = list(range(0, len(out), 5))
    spike_vals = {
        'FHR'     :  6.0,
        'TOCO'    :  6.0,
        'SpO2'    : -7.0,
        'RespRate':  5.0,
        'Temp'    :  6.0
    }
    for col, mag in spike_vals.items():
        out.loc[spike_idx, col] += mag
    return out


def attack_resource_exhaustion(base):
    """Random burst spikes on 40% of samples — 5-8 std devs"""
    out = base.copy()
    burst_mask = np.random.rand(len(out)) < 0.40
    for col in COLUMNS:
        burst = np.random.uniform(5.0, 8.0, len(out))
        sign  = np.random.choice([-1, 1], len(out))
        out.loc[burst_mask, col] += (burst * sign)[burst_mask]
    return out


ATTACK_FUNCTIONS = {
    'DoS_Flooding'       : attack_dos,
    'Spoofing'           : attack_spoofing,
    'MITM'               : attack_mitm,
    'Jamming'            : attack_jamming,
    'Replay'             : attack_replay,
    'Data_Injection'     : attack_data_injection,
    'Resource_Exhaustion': attack_resource_exhaustion
}

# ============================================================
# STEP 4 — GENERATE ATTACK SAMPLES FROM GAN BASE
# ============================================================
print("\n[3/6] Generating GAN-based attack samples...")
print("  GAN provides realistic base → perturbation adds attack signature\n")

attack_frames = []

for attack_name, attack_fn in ATTACK_FUNCTIONS.items():
    # Base samples from CTGAN — realistic normalized sensor values
    base = ctgan.sample(SAMPLES_PER_ATTACK)
    base = base.reset_index(drop=True)

    # Apply strong perturbation
    attacked = attack_fn(base.copy())
    attacked['label']       = 1
    attacked['attack_type'] = attack_name
    attack_frames.append(attacked)

    shift = abs(attacked[COLUMNS].values - base[COLUMNS].values).mean()
    print(f"  ✅ {attack_name:22s} | avg shift: {shift:.3f} std")

df_attacks = pd.concat(attack_frames, ignore_index=True)
print(f"\n  Total attack samples : {len(df_attacks):,}")

# ============================================================
# STEP 5 — NORMAL SAMPLES (real data, no modification)
# ============================================================
print("\n[4/6] Preparing normal samples...")

df_normal_labeled = df_norm.sample(
    n=len(df_attacks), random_state=RANDOM_SEED
).copy()
df_normal_labeled['label']       = 0
df_normal_labeled['attack_type'] = 'Normal'

print(f"  Normal samples : {len(df_normal_labeled):,}")

# ============================================================
# STEP 6 — COMBINE AND SAVE (NO SHUFFLE)
# ============================================================
print("\n[5/6] Combining — normal first, attack second (NO shuffle)...")

df_final = pd.concat(
    [df_normal_labeled, df_attacks], ignore_index=True
).reset_index(drop=True)

for col in COLUMNS:
    df_final[col] = df_final[col].round(6)
df_final['label'] = df_final['label'].astype(int)

df_final.to_csv(OUTPUT_PATH, index=False)

# ============================================================
# SUMMARY
# ============================================================
print("\n[6/6] Verifying separation...")

normal_vals = df_final[df_final['label']==0][COLUMNS].values
attack_vals = df_final[df_final['label']==1][COLUMNS].values

sep = abs(attack_vals.mean() - normal_vals.mean())
print(f"\n  Normal mean : {normal_vals.mean():.4f}")
print(f"  Attack mean : {attack_vals.mean():.4f}")
print(f"  Separation  : {sep:.4f}")

if sep > 2.0:
    print("  ✅ Good separation — evaluation should work well")
else:
    print("  ⚠️  Low separation — consider increasing perturbation magnitude")

print("\n" + "=" * 70)
print("  GAN DATASET GENERATION COMPLETE")
print("=" * 70)
print(f"  Saved      : {OUTPUT_PATH}")
print(f"  Total rows : {len(df_final):,}")
print(f"  Normal (0) : {(df_final['label']==0).sum():,}")
print(f"  Attack (1) : {(df_final['label']==1).sum():,}")
print(f"\n  Per attack type:")
print(df_final.groupby('attack_type')['label'].count().to_string())
print("=" * 70)
print("\n  Next steps:")
print("  1. Update DATASET_PATH in compute_threshold_gan.py:")
print("     DATASET_PATH = 'medical_iot_ids/processed/labeled_attack_dataset_gan.csv'")
print("  2. python preprocessing/compute_threshold_gan.py")
print("  3. Update DATASET_PATH in evaluate_with_labeled_dataset.py")
print("  4. python preprocessing/evaluate_with_labeled_dataset.py")
print("=" * 70)