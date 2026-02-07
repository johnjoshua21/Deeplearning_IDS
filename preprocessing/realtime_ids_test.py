import numpy as np
import pandas as pd
import joblib
import time
from datetime import datetime
from tensorflow.keras.models import load_model
from collections import deque

# ===========================
# CONFIGURATION
# ===========================
WINDOW_SIZE = 60
THRESHOLD = 1
DATASET_PATH = "medical_iot_ids/processed/final_5sensor_norm.csv"
MODEL_PATH = "medical_iot_ids/model/lstm_autoencoder.h5"
SCALER_PATH = "medical_iot_ids/model/scaler.pkl"

# ===========================
# LOAD MODEL & DATA
# ===========================
print("Loading model and data...")
model = load_model(MODEL_PATH, compile=False)
scaler = joblib.load(SCALER_PATH)
df_norm = pd.read_csv(DATASET_PATH)
data_norm = df_norm.values

print(f"✓ Model loaded")
print(f"✓ Dataset loaded: {data_norm.shape}")
print(f"✓ Threshold: {THRESHOLD}\n")


class RealtimeIDS:
    def __init__(self):
        self.window = deque(maxlen=WINDOW_SIZE)
        self.packet_count = 0
        self.attack_count = 0
        self.normal_count = 0

    def add_packet(self, sensor_values, is_injected_attack=False):
        """Add a new sensor reading (normalized values)"""
        self.window.append(sensor_values)
        self.packet_count += 1

        # Need full window for detection
        if len(self.window) < WINDOW_SIZE:
            print(f"[{self.packet_count}] Collecting data... ({len(self.window)}/{WINDOW_SIZE})")
            return None

        # Prepare window for model
        X = np.array(self.window).reshape(1, WINDOW_SIZE, 5)

        # Get reconstruction
        X_recon = model.predict(X, verbose=0)

        # Calculate error
        error = np.mean((X - X_recon) ** 2)

        # Classify
        is_attack = error > THRESHOLD

        if is_attack:
            self.attack_count += 1
            status = "🚨 ATTACK DETECTED"
            color = "\033[91m"  # Red
        else:
            self.normal_count += 1
            status = "✅ Normal"
            color = "\033[92m"  # Green

        reset = "\033[0m"

        # Display
        print(f"{color}[{self.packet_count}] {status}{reset}")
        print(f"    Error: {error:.6f} | Threshold: {THRESHOLD}")
        print(f"    Values: FHR={sensor_values[0]:.3f}, TOCO={sensor_values[1]:.3f}, "
              f"SpO2={sensor_values[2]:.3f}, RR={sensor_values[3]:.3f}, Temp={sensor_values[4]:.3f}")

        if is_injected_attack:
            print(f"    ⚠️  This WAS an injected attack - Detection: {'CORRECT ✓' if is_attack else 'MISSED ✗'}")

        print(f"    Stats: Normal={self.normal_count}, Attacks={self.attack_count}")
        print("-" * 80)

        return {
            'error': error,
            'is_attack': is_attack,
            'values': sensor_values
        }

    def get_stats(self):
        return {
            'total': self.packet_count,
            'normal': self.normal_count,
            'attacks': self.attack_count,
            'attack_rate': self.attack_count / self.packet_count if self.packet_count > 0 else 0
        }


# ===========================
# TEST SCENARIOS
# ===========================

def test_scenario_1_normal_stream():
    """Test 1: Stream real normal data"""
    print("\n" + "=" * 80)
    print("TEST SCENARIO 1: Real Normal Data Stream")
    print("=" * 80)

    ids = RealtimeIDS()

    # Use random starting point from dataset
    start_idx = np.random.randint(0, len(data_norm) - 200)

    for i in range(start_idx, start_idx + 100):
        sensor_values = data_norm[i]
        ids.add_packet(sensor_values, is_injected_attack=False)
        time.sleep(0.1)  # Simulate real-time delay

    stats = ids.get_stats()
    print(f"\n📊 Final Stats: {stats['normal']} normal, {stats['attacks']} attacks detected")
    print(f"Attack Rate: {stats['attack_rate'] * 100:.2f}%\n")


def test_scenario_2_mixed_attacks():
    """Test 2: Normal data with periodic attack injections"""
    print("\n" + "=" * 80)
    print("TEST SCENARIO 2: Normal Data + Periodic Attack Injections")
    print("=" * 80)

    ids = RealtimeIDS()
    start_idx = np.random.randint(0, len(data_norm) - 200)

    for i in range(100):
        sensor_values = data_norm[start_idx + i].copy()

        # Inject attack every 10 packets
        is_attack = (i % 10 == 0) and (i > 0)

        if is_attack:
            # Simulate MITM attack - manipulate values
            sensor_values[0] += 2.5  # FHR spike
            sensor_values[2] -= 2.0  # SpO2 drop
            sensor_values[4] += 1.8  # Temp spike
            print("\n⚡ INJECTING ATTACK (MITM-style manipulation)")

        ids.add_packet(sensor_values, is_injected_attack=is_attack)
        time.sleep(0.1)

    stats = ids.get_stats()
    print(f"\n📊 Final Stats: {stats['normal']} normal, {stats['attacks']} attacks detected")


def test_scenario_3_spoofing_attack():
    """Test 3: Spoofing - Out of range values"""
    print("\n" + "=" * 80)
    print("TEST SCENARIO 3: Spoofing Attack - Out of Range Values")
    print("=" * 80)

    ids = RealtimeIDS()
    start_idx = np.random.randint(0, len(data_norm) - 200)

    for i in range(100):
        sensor_values = data_norm[start_idx + i].copy()

        # Inject spoofing attack with extreme values
        is_attack = (i >= 40 and i < 60)

        if is_attack:
            # Extreme out-of-range values (normalized)
            sensor_values[0] = 5.0  # Extremely high FHR
            sensor_values[1] = -3.0  # Extremely low TOCO
            sensor_values[2] = -4.0  # Critically low SpO2
            print("\n⚡ INJECTING SPOOFING ATTACK (Extreme values)")

        ids.add_packet(sensor_values, is_injected_attack=is_attack)
        time.sleep(0.1)

    stats = ids.get_stats()
    print(f"\n📊 Final Stats: {stats['normal']} normal, {stats['attacks']} attacks detected")


def test_scenario_4_replay_attack():
    """Test 4: Replay Attack - Repeated old values"""
    print("\n" + "=" * 80)
    print("TEST SCENARIO 4: Replay Attack - Repeating Old Pattern")
    print("=" * 80)

    ids = RealtimeIDS()
    start_idx = np.random.randint(0, len(data_norm) - 200)

    # Capture "old" packet to replay
    replay_packet = data_norm[start_idx].copy()

    for i in range(100):
        is_attack = (i >= 30 and i < 70)

        if is_attack:
            # Keep sending same old values (replay)
            sensor_values = replay_packet
            if i == 30:
                print("\n⚡ STARTING REPLAY ATTACK (Repeating old values)")
        else:
            sensor_values = data_norm[start_idx + i]

        ids.add_packet(sensor_values, is_injected_attack=is_attack)
        time.sleep(0.1)

    stats = ids.get_stats()
    print(f"\n📊 Final Stats: {stats['normal']} normal, {stats['attacks']} attacks detected")


def test_scenario_5_jamming():
    """Test 5: Jamming - Zero/null values"""
    print("\n" + "=" * 80)
    print("TEST SCENARIO 5: Jamming Attack - Zero/Null Values")
    print("=" * 80)

    ids = RealtimeIDS()
    start_idx = np.random.randint(0, len(data_norm) - 200)

    for i in range(100):
        is_attack = (i >= 40 and i < 60)

        if is_attack:
            # Jamming causes all zeros or constant extreme values
            sensor_values = np.array([-5.0, -5.0, -5.0, -5.0, -5.0])
            if i == 40:
                print("\n⚡ STARTING JAMMING ATTACK (Null signals)")
        else:
            sensor_values = data_norm[start_idx + i]

        ids.add_packet(sensor_values, is_injected_attack=is_attack)
        time.sleep(0.1)

    stats = ids.get_stats()
    print(f"\n📊 Final Stats: {stats['normal']} normal, {stats['attacks']} attacks detected")


def test_custom_values():
    """Test with specific values you want to check"""
    print("\n" + "=" * 80)
    print("TEST: Custom Values from Your Dataset")
    print("=" * 80)

    ids = RealtimeIDS()

    # You can specify exact row indices from your normalized dataset
    test_indices = [100, 200, 500, 1000, 2000, 3000]

    print("Testing specific rows from your dataset:\n")

    for idx in test_indices:
        if idx < len(data_norm):
            sensor_values = data_norm[idx]
            print(f"\n>>> Testing Row {idx} from dataset")
            result = ids.add_packet(sensor_values, is_injected_attack=False)
            time.sleep(0.2)


# ===========================
# INTERACTIVE MENU
# ===========================

def main():
    print("\n" + "=" * 80)
    print("         REAL-TIME IDS TESTING with Real Dataset Values")
    print("=" * 80)
    print("\nSelect Test Scenario:\n")
    print("  1. Normal Data Stream (100 packets)")
    print("  2. Mixed Normal + Periodic Attacks")
    print("  3. Spoofing Attack (Extreme Values)")
    print("  4. Replay Attack (Repeated Pattern)")
    print("  5. Jamming Attack (Zero/Null Values)")
    print("  6. Test Specific Dataset Rows")
    print("  7. Run ALL Scenarios")
    print("  8. Exit")
    print("\n" + "=" * 80)

    scenarios = {
        '1': test_scenario_1_normal_stream,
        '2': test_scenario_2_mixed_attacks,
        '3': test_scenario_3_spoofing_attack,
        '4': test_scenario_4_replay_attack,
        '5': test_scenario_5_jamming,
        '6': test_custom_values
    }

    while True:
        try:
            choice = input("\nEnter choice (1-8): ").strip()

            if choice == '8':
                print("\n👋 Exiting...")
                break

            if choice == '7':
                for name, func in scenarios.items():
                    func()
                    time.sleep(2)
                continue

            if choice in scenarios:
                scenarios[choice]()
            else:
                print("❌ Invalid choice. Please enter 1-8.")

        except KeyboardInterrupt:
            print("\n\n👋 Testing stopped by user")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()