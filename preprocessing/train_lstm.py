import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ===========================
# PATHS
# ===========================
DATA_PATH       = "medical_iot_ids/processed/X_windows.npy"
MODEL_DIR       = "medical_iot_ids/model"
MODEL_PATH      = os.path.join(MODEL_DIR, "lstm_autoencoder.h5")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "lstm_autoencoder_best.h5")

os.makedirs(MODEL_DIR, exist_ok=True)

# ===========================
# LOAD DATA
# ===========================
X = np.load(DATA_PATH)
print("Loaded data shape:", X.shape)

TIMESTEPS = X.shape[1]   # 60
FEATURES  = X.shape[2]   # 5
N         = len(X)

# ===========================
# 70 / 20 / 10 SPLIT
# ===========================
np.random.seed(42)
indices = np.random.permutation(N)
X = X[indices]

train_end = int(0.70 * N)
val_end   = int(0.90 * N)   # 70 + 20 = 90

X_train = X[:train_end]
X_val   = X[train_end:val_end]
X_test  = X[val_end:]

print(f"\n{'='*60}")
print("        DATASET SPLIT SUMMARY")
print(f"{'='*60}")
print(f"  Total windows  : {N}")
print(f"  Train  (70%)   : {len(X_train)}")
print(f"  Val    (20%)   : {len(X_val)}")
print(f"  Test   (10%)   : {len(X_test)}")
print(f"{'='*60}\n")

# Save test set for later evaluation
np.save(os.path.join(MODEL_DIR, "X_test.npy"), X_test)
print("✅ Test set saved to medical_iot_ids/model/X_test.npy\n")

# ===========================
# BUILD MODEL
# ===========================
model = Sequential([
    LSTM(64, activation="tanh", input_shape=(TIMESTEPS, FEATURES)),
    RepeatVector(TIMESTEPS),
    LSTM(64, activation="tanh", return_sequences=True),
    TimeDistributed(Dense(FEATURES))
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="mse"
)

model.summary()

# ===========================
# CALLBACKS
# ===========================
callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        BEST_MODEL_PATH,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )
]

# ===========================
# TRAIN
# ===========================
print("\n[TRAINING] Starting...")
history = model.fit(
    X_train, X_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_val, X_val),
    callbacks=callbacks,
    shuffle=True
)

# ===========================
# EVALUATE ON TEST SET
# ===========================
print("\n[EVALUATION] Running on held-out test set...")

X_test_pred = model.predict(X_test, verbose=0)
test_errors = np.mean((X_test - X_test_pred) ** 2, axis=(1, 2))

test_loss        = float(np.mean(test_errors))
train_loss_final = history.history['loss'][-1]
val_loss_final   = history.history['val_loss'][-1]

print(f"\n{'='*60}")
print("        FINAL RESULTS")
print(f"{'='*60}")
print(f"  Train Loss (final epoch) : {train_loss_final:.6f}")
print(f"  Val   Loss (final epoch) : {val_loss_final:.6f}")
print(f"  Test  Loss (MSE)         : {test_loss:.6f}")
print(f"  Test  Error  — Mean      : {test_errors.mean():.6f}")
print(f"  Test  Error  — Std       : {test_errors.std():.6f}")
print(f"  Test  Error  — 95th      : {np.percentile(test_errors, 95):.6f}")
print(f"  Test  Error  — 99th      : {np.percentile(test_errors, 99):.6f}")
print(f"{'='*60}")

# ===========================
# SUGGESTED THRESHOLD
# ===========================
suggested_threshold = float(np.mean(test_errors) + 2.5 * np.std(test_errors))
print(f"\n💡 Suggested threshold (mean + 2.5σ): {suggested_threshold:.6f}")
print("   Update THRESHOLD in all config files to this value.\n")

# ===========================
# SAVE FINAL MODEL
# ===========================
model.save(MODEL_PATH)
print(f"✅ Final model saved  : {MODEL_PATH}")
print(f"✅ Best model saved   : {BEST_MODEL_PATH}")

# ===========================
# PLOT TRAINING CURVES
# ===========================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# --- Loss curve ---
ax1 = axes[0]
ax1.plot(history.history['loss'],     label='Train Loss',      linewidth=2, color='steelblue')
ax1.plot(history.history['val_loss'], label='Validation Loss', linewidth=2, color='darkorange', linestyle='--')
best_epoch = int(np.argmin(history.history['val_loss']))
ax1.axvline(best_epoch, color='red', linestyle=':', linewidth=1.5,
            label=f'Best epoch ({best_epoch + 1})')
ax1.set_xlabel('Epoch', fontsize=13, fontweight='bold')
ax1.set_ylabel('MSE Loss', fontsize=13, fontweight='bold')
ax1.set_title('Training & Validation Loss', fontsize=15, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# --- Test error distribution ---
ax2 = axes[1]
ax2.hist(test_errors, bins=50, color='steelblue', edgecolor='black', alpha=0.8)
ax2.axvline(test_errors.mean(), color='green', linewidth=2,
            label=f'Mean = {test_errors.mean():.4f}')
ax2.axvline(suggested_threshold, color='red', linewidth=2, linestyle='--',
            label=f'Threshold = {suggested_threshold:.4f}')
ax2.set_xlabel('Reconstruction Error (MSE)', fontsize=13, fontweight='bold')
ax2.set_ylabel('Frequency', fontsize=13, fontweight='bold')
ax2.set_title('Test Set Error Distribution', fontsize=15, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, 'training_curves.png'), dpi=300, bbox_inches='tight')
print("✅ Training curves saved: medical_iot_ids/model/training_curves.png")
plt.show()

print(f"\n{'='*60}")
print("  TRAINING COMPLETE ✓")
print("  Split: 70% Train | 20% Validation | 10% Test")
print(f"{'='*60}")