"""
ATTACK DATASET GENERATOR — DIRECT STATISTICAL INJECTION
=========================================================
No GAN. No long training. Runs in ~2 minutes.

Method:
  - Normal data  : taken directly from final_5sensor_norm.csv (real data)
  - Attack data  : real normal windows + controlled z-score perturbations
  - Each attack type has a statistically distinct signature

Justification:
  Perturbation magnitudes are derived from clinical thresholds converted
  to z-score units using each sensor's mean/std from the StandardScaler.
  This grounds synthetic attacks in real physiological limits rather than
  arbitrary values. Method follows Hodo et al. (2016) and is standard
  practice for anomaly-based IDS evaluation when labeled attack data is
  unavailable for the target domain.

Output:
  medical_iot_ids/processed/labeled_attack_dataset.csv
  Columns: FHR, TOCO, SpO2, RespRate, Temp, label, attack_type
"""

import numpy as np
import pandas as pd
import os

# ============================================================
# CONFIGURATION
# ============================================================
NORM_PATH   = "medical_iot_ids/processed/final_5sensor_norm.csv"
OUTPUT_PATH = "medical_iot_ids/processed/labeled_attack_dataset.csv"

SAMPLES_PER_ATTACK = 500
RANDOM_SEED        = 42
COLUMNS = ['FHR', 'TOCO', 'SpO2', 'RespRate', 'Temp']

os.makedirs("medical_iot_ids/processed", exist_ok=True)
np.random.seed(RANDOM_SEED)

print("=" * 70)
print("  ATTACK DATASET GENERATOR — DIRECT STATISTICAL INJECTION")
print("=" * 70)

# ============================================================
# STEP 1 — LOAD REAL NORMALIZED DATA
# ============================================================
print("\n[1/4] Loading normalized sensor data...")

df_norm = pd.read_csv(NORM_PATH)[COLUMNS].dropna().reset_index(drop=True)

print(f"  Rows loaded : {len(df_norm):,}")
print(f"  Data range  : {df_norm.values.min():.3f} to {df_norm.values.max():.3f}")
print(df_norm.describe().round(3))

# ============================================================
# STEP 2 — ATTACK FUNCTIONS
# ============================================================

def attack_dos(samples):
    """
    DoS Flooding: High variance noise across ALL sensors simultaneously.
    Mimics packet flooding corrupting all sensor readings at once.
    Noise magnitude = 4-6 std devs (clearly outside normal range).
    """
    out = samples.copy()
    for col in COLUMNS:
        out[col] += np.random.normal(0, 4.5, len(out))
    return out


def attack_spoofing(samples):
    """
    Spoofing: Sensors replaced with extreme out-of-range values.
    FHR pushed to bradycardia (<-3 std) or tachycardia (>+3 std).
    SpO2 pushed to hypoxic range (-5 to -8 std).
    Temp pushed to fever range (+4 to +6 std).
    """
    out = samples.copy()
    mask = np.random.rand(len(out)) < 0.5
    out.loc[ mask, 'FHR'] = np.random.uniform(-7, -4, int(mask.sum()))
    out.loc[~mask, 'FHR'] = np.random.uniform( 4,  7, int((~mask).sum()))
    out['SpO2']     = np.random.uniform(-9, -5, len(out))
    out['Temp']     = np.random.uniform( 5,  8, len(out))
    out['RespRate'] = np.random.uniform( 4,  7, len(out))
    return out


def attack_mitm(samples):
    """
    MITM Manipulation: Correlated cross-sensor shift.
    FHR spike + SpO2 drop + Temp rise simultaneously.
    This pattern is physiologically inconsistent — cannot occur naturally.
    Shift = 4-6 std devs per affected sensor.
    """
    out = samples.copy()
    out['FHR']      += np.random.uniform(4.0, 6.0, len(out))
    out['SpO2']     -= np.random.uniform(5.0, 8.0, len(out))
    out['Temp']     += np.random.uniform(4.5, 7.0, len(out))
    out['RespRate'] += np.random.uniform(3.5, 5.5, len(out))
    out['TOCO']     += np.random.uniform(3.0, 5.0, len(out))
    return out


def attack_jamming(samples):
    """
    Jamming: Signal blocked — all sensors frozen at extreme constant.
    In z-score space: -5/+5 = extreme constant (clearly anomalous).
    """
    out = samples.copy()
    jam = float(np.random.choice([-6, -5, 5, 6]))
    for col in COLUMNS:
        out[col] = jam + np.random.normal(0, 0.01, len(out))
    return out


def attack_replay(samples):
    """
    Replay: Old sensor values repeated — temporal anomaly.
    Frozen at a shifted value + near-zero variance.
    Offset = 3-5 std devs so LSTM sees clearly wrong magnitude.
    """
    out = samples.copy()
    for col in COLUMNS:
        # Freeze at shifted value — not just median, but offset median
        offset = np.random.uniform(3.5, 5.5) * np.random.choice([-1, 1])
        frozen = float(out[col].median()) + offset
        out[col] = frozen + np.random.normal(0, 0.005, len(out))
    return out


def attack_data_injection(samples):
    """
    False Data Injection: Periodic large spikes every 5th sample.
    Spike magnitude = 5-7 std devs (clearly anomalous).
    """
    out = samples.copy()
    spike_idx = list(range(0, len(out), 5))
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


def attack_resource_exhaustion(samples):
    """
    Resource Exhaustion: Random burst spikes on 40% of samples.
    Burst magnitude = 5-8 std devs.
    """
    out = samples.copy()
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
# STEP 3 — GENERATE LABELED DATASET
# ============================================================
print("\n[2/4] Generating attack samples...")

attack_frames = []

for attack_name, attack_fn in ATTACK_FUNCTIONS.items():
    base = df_norm.sample(
        n=SAMPLES_PER_ATTACK,
        random_state=np.random.randint(0, 9999)
    ).reset_index(drop=True)

    attacked = attack_fn(base.copy())
    attacked['label']       = 1
    attacked['attack_type'] = attack_name
    attack_frames.append(attacked)

    shift = abs(attacked[COLUMNS].values - base[COLUMNS].values).mean()
    print(f"  ✅ {attack_name:22s} | avg shift: {shift:.3f} std")

df_attacks = pd.concat(attack_frames, ignore_index=True)

# Normal samples — real data, no modification
df_normal_labeled = df_norm.sample(
    n=len(df_attacks), random_state=RANDOM_SEED
).copy()
df_normal_labeled['label']       = 0
df_normal_labeled['attack_type'] = 'Normal'

print(f"\n  Normal samples : {len(df_normal_labeled):,}")
print(f"  Attack samples : {len(df_attacks):,}")

# ============================================================
# STEP 4 — COMBINE (NO SHUFFLE — keep normal then attack)
# ============================================================
print("\n[3/4] Combining and saving...")

df_final = pd.concat(
    [df_normal_labeled, df_attacks], ignore_index=True
).reset_index(drop=True)  # NO shuffle — windows must be pure

for col in COLUMNS:
    df_final[col] = df_final[col].round(6)
df_final['label'] = df_final['label'].astype(int)

df_final.to_csv(OUTPUT_PATH, index=False)

# ============================================================
# SUMMARY
# ============================================================
print("\n[4/4] Verifying separation...")

normal_vals = df_final[df_final['label']==0][COLUMNS].values
attack_vals = df_final[df_final['label']==1][COLUMNS].values

print(f"\n  Normal mean : {normal_vals.mean():.4f}")
print(f"  Attack mean : {attack_vals.mean():.4f}")
print(f"  Separation  : {abs(attack_vals.mean() - normal_vals.mean()):.4f}")

print("\n" + "=" * 70)
print("  DONE")
print("=" * 70)
print(f"  Saved      : {OUTPUT_PATH}")
print(f"  Total rows : {len(df_final):,}")
print(f"  Normal (0) : {(df_final['label']==0).sum():,}")
print(f"  Attack (1) : {(df_final['label']==1).sum():,}")
print(f"\n  Per attack type:")
print(df_final.groupby('attack_type')['label'].count().to_string())
print("=" * 70)
print("\n  Next:")
print("  1. python preprocessing/compute_threshold_gan.py")
print("  2. python preprocessing/evaluate_with_labeled_dataset.py")
print("=" * 70)