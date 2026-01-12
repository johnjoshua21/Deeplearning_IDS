import socket
import time
import random
import json
from datetime import datetime
from config import TARGET_IP, GATEWAY_PORT, SENSOR_RANGES

"""
REALISTIC IoT ATTACKS FOR MEDICAL SENSOR NETWORKS
(Without Attack Labels - IDS Must Detect)

Attack packets now look like normal sensor traffic.
Only the IDS model at the collector can detect anomalies.

This simulates real-world scenarios where attackers:
- Don't announce "I'm attacking!"
- Try to blend in with normal traffic
- Use subtle or obvious anomalies

The LSTM Autoencoder IDS must detect these patterns.
"""


import socket
import time
import random
import json
from datetime import datetime
from config import TARGET_IP, GATEWAY_PORT, SENSOR_RANGES


class IoTAttackInjector:
    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def create_packet(self, sensor_id, sensor_type, value):
        """
        Create packet that looks IDENTICAL to normal sensor traffic
        NO attack labels - IDS must detect purely from patterns
        """
        packet = {
            'sensor_id': sensor_id,
            'sensor_type': sensor_type,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
            'value': value
            # ❌ REMOVED: 'is_attack': 1
            # ❌ REMOVED: 'attack_type': 'DoS'
        }
        return packet

    def send_packet(self, packet):
        """Send packet to gateway"""
        self.sock.sendto(json.dumps(packet).encode(), (TARGET_IP, GATEWAY_PORT))

    # === ATTACK 1: DoS Flooding ===
    def dos_flooding(self, duration=15):
        """Flood with noisy data - IDS should detect unusual patterns"""
        print("\n" + "=" * 70)
        print("🔴 INJECTING: DoS Flooding Attack")
        print("=" * 70)
        print("Pattern: High frequency packets with random noise")
        print("IDS Should Detect: Abnormal value patterns")
        print(f"Duration: {duration}s")
        print("-" * 70)

        sensor_configs = [
            ('S1', 'FHR'),
            ('S2', 'TOCO'),
            ('S3', 'SpO2'),
            ('S4', 'RespRate'),
            ('S5', 'Temp')
        ]

        end_time = time.time() + duration
        count = 0

        while time.time() < end_time:
            sensor_id, sensor_type = random.choice(sensor_configs)
            min_val, max_val = SENSOR_RANGES[sensor_type]
            center = (min_val + max_val) / 2
            noise = random.uniform(-50, 50)
            value = round(center + noise, 2)

            packet = self.create_packet(sensor_id, sensor_type, value)
            self.send_packet(packet)
            count += 1

            if count % 50 == 0:
                print(f"  Sent {count} packets...")

            time.sleep(0.02)

        print(f"✓ Sent {count} attack packets (IDS should detect these)")

    # === ATTACK 2: Spoofing ===
    def spoofing_attack(self, duration=15):
        """Send extreme out-of-range values"""
        print("\n" + "=" * 70)
        print("🔴 INJECTING: Spoofing Attack")
        print("=" * 70)
        print("Pattern: Extreme out-of-range values")
        print("IDS Should Detect: Values far from normal distribution")
        print(f"Duration: {duration}s")
        print("-" * 70)

        sensor_configs = [
            ('S1', 'FHR'),
            ('S2', 'TOCO'),
            ('S3', 'SpO2'),
            ('S4', 'RespRate'),
            ('S5', 'Temp')
        ]

        end_time = time.time() + duration
        count = 0

        while time.time() < end_time:
            sensor_id, sensor_type = random.choice(sensor_configs)
            min_val, max_val = SENSOR_RANGES[sensor_type]

            if random.random() < 0.5:
                value = round(max_val * 1.5, 2)
            else:
                value = round(min_val * 0.5, 2)

            packet = self.create_packet(sensor_id, sensor_type, value)
            self.send_packet(packet)

            print(f"  {sensor_type}: {value} (extreme)")
            count += 1
            time.sleep(1)

        print(f"✓ Sent {count} spoofed packets")

    # === ATTACK 3: MITM ===
    def mitm_data_manipulation(self, duration=15):
        """Subtle manipulation of critical values"""
        print("\n" + "=" * 70)
        print("🔴 INJECTING: MITM Attack")
        print("=" * 70)
        print("Pattern: Subtle value manipulation")
        print("IDS Should Detect: Abnormal temporal patterns")
        print(f"Duration: {duration}s")
        print("-" * 70)

        critical_sensors = [
            ('S1', 'FHR'),
            ('S3', 'SpO2'),
            ('S5', 'Temp')
        ]

        end_time = time.time() + duration
        count = 0

        while time.time() < end_time:
            sensor_id, sensor_type = random.choice(critical_sensors)

            if sensor_type == 'FHR':
                value = round(random.uniform(180, 200), 2)
            elif sensor_type == 'SpO2':
                value = round(random.uniform(88, 93), 2)
            else:
                value = round(random.uniform(38.5, 39.5), 2)

            packet = self.create_packet(sensor_id, sensor_type, value)
            self.send_packet(packet)

            print(f"  {sensor_type}: {value} (manipulated)")
            count += 1
            time.sleep(0.8)

        print(f"✓ Sent {count} manipulated packets")

    # === ATTACK 4: Jamming ===
    def jamming_attack(self, duration=15):
        """Send zeros/null values"""
        print("\n" + "=" * 70)
        print("🔴 INJECTING: Jamming Attack")
        print("=" * 70)
        print("Pattern: Zero/null values (signal loss)")
        print("IDS Should Detect: Flatline patterns")
        print(f"Duration: {duration}s")
        print("-" * 70)

        sensor_configs = [
            ('S1', 'FHR'),
            ('S2', 'TOCO'),
            ('S3', 'SpO2'),
            ('S4', 'RespRate'),
            ('S5', 'Temp')
        ]

        end_time = time.time() + duration
        count = 0

        while time.time() < end_time:
            sensor_id, sensor_type = random.choice(sensor_configs)
            value = random.choice([0, 0, 0, 1, -1])

            packet = self.create_packet(sensor_id, sensor_type, value)
            self.send_packet(packet)

            print(f"  {sensor_type}: {value} (jammed)")
            count += 1
            time.sleep(0.5)

        print(f"✓ Sent {count} jammed packets")

    # === ATTACK 5: Replay ===
    def replay_attack(self, duration=15):
        """Replay same values repeatedly"""
        print("\n" + "=" * 70)
        print("🔴 INJECTING: Replay Attack")
        print("=" * 70)
        print("Pattern: Repeated identical values")
        print("IDS Should Detect: Lack of natural variation")
        print(f"Duration: {duration}s")
        print("-" * 70)

        captured_values = {
            'S1': 145.5,
            'S2': 45.0,
            'S3': 97.0,
            'S4': 18.0,
            'S5': 37.0
        }

        sensor_types = {
            'S1': 'FHR',
            'S2': 'TOCO',
            'S3': 'SpO2',
            'S4': 'RespRate',
            'S5': 'Temp'
        }

        end_time = time.time() + duration
        count = 0

        while time.time() < end_time:
            for sensor_id in ['S1', 'S2', 'S3', 'S4', 'S5']:
                sensor_type = sensor_types[sensor_id]
                value = captured_values[sensor_id]

                packet = self.create_packet(sensor_id, sensor_type, value)
                self.send_packet(packet)

                print(f"  {sensor_type}: {value} (replayed)")
                count += 1
                time.sleep(0.2)

        print(f"✓ Sent {count} replayed packets")

    # === ATTACK 6: Data Injection ===
    def false_data_injection(self, duration=15):
        """Inject random bursts"""
        print("\n" + "=" * 70)
        print("🔴 INJECTING: False Data Injection")
        print("=" * 70)
        print("Pattern: Random value bursts")
        print("IDS Should Detect: Sudden spikes")
        print(f"Duration: {duration}s")
        print("-" * 70)

        sensor_configs = [
            ('S1', 'FHR'),
            ('S2', 'TOCO'),
            ('S3', 'SpO2'),
            ('S4', 'RespRate'),
            ('S5', 'Temp')
        ]

        end_time = time.time() + duration
        count = 0

        while time.time() < end_time:
            sensor_id, sensor_type = random.choice(sensor_configs)
            min_val, max_val = SENSOR_RANGES[sensor_type]
            value = round(random.uniform(min_val * 0.3, max_val * 1.8), 2)

            packet = self.create_packet(sensor_id, sensor_type, value)
            self.send_packet(packet)

            print(f"  {sensor_type}: {value} (injected)")
            count += 1
            time.sleep(0.7)

        print(f"✓ Sent {count} false packets")

    # === ATTACK 7: Resource Exhaustion ===
    def resource_exhaustion(self, duration=15):
        """Burst patterns"""
        print("\n" + "=" * 70)
        print("🔴 INJECTING: Resource Exhaustion")
        print("=" * 70)
        print("Pattern: Rapid bursts")
        print("IDS Should Detect: Unusual burst frequency")
        print(f"Duration: {duration}s")
        print("-" * 70)

        sensor_configs = [
            ('S1', 'FHR'),
            ('S2', 'TOCO'),
            ('S3', 'SpO2'),
            ('S4', 'RespRate'),
            ('S5', 'Temp')
        ]

        end_time = time.time() + duration
        total_count = 0

        while time.time() < end_time:
            print("  🔥 BURST...")
            for _ in range(30):
                sensor_id, sensor_type = random.choice(sensor_configs)
                min_val, max_val = SENSOR_RANGES[sensor_type]
                center = (min_val + max_val) / 2
                value = round(center + random.uniform(-20, 20), 2)

                packet = self.create_packet(sensor_id, sensor_type, value)
                self.send_packet(packet)
                total_count += 1
                time.sleep(0.03)

            print("  💤 Silence...")
            time.sleep(2)

        print(f"✓ Sent {total_count} burst packets")

    def interactive_menu(self):
        """Interactive attack selection"""
        print("\n" + "=" * 70)
        print("     IoT ATTACK INJECTOR - Unsupervised IDS Mode")
        print("=" * 70)
        print("\n⚠️  UNSUPERVISED MODE:")
        print("   • NO attack labels in packets")
        print("   • Attacks look like normal traffic")
        print("   • Only IDS can detect anomalies")
        print("   • Watch collector console for detections")
        print("\n" + "=" * 70)
        print("\nSelect Attack Type:\n")
        print("  1. DoS Flooding")
        print("  2. Spoofing Attack")
        print("  3. MITM Data Manipulation")
        print("  4. Jamming Attack")
        print("  5. Replay Attack")
        print("  6. False Data Injection")
        print("  7. Resource Exhaustion")
        print("  8. Run ALL Attacks")
        print("  9. Exit")
        print("\n" + "=" * 70)

        attacks = {
            '1': ('DoS Flooding', self.dos_flooding),
            '2': ('Spoofing', self.spoofing_attack),
            '3': ('MITM', self.mitm_data_manipulation),
            '4': ('Jamming', self.jamming_attack),
            '5': ('Replay', self.replay_attack),
            '6': ('Data Injection', self.false_data_injection),
            '7': ('Resource Exhaustion', self.resource_exhaustion)
        }

        while True:
            try:
                choice = input("\nEnter attack number (1-9): ").strip()

                if choice == '9':
                    print("\n👋 Exiting...")
                    break

                if choice == '8':
                    print("\n🚨 Running ALL attacks...")
                    for name, attack_func in attacks.values():
                        print(f"\n▶️  {name}")
                        attack_func(duration=10)
                        print("\n⏸️  Waiting 5 seconds...\n")
                        time.sleep(5)
                    print("\n✅ All attacks completed!")
                    continue

                if choice in attacks:
                    name, attack_func = attacks[choice]
                    duration = input(f"Duration (default 15s): ").strip()
                    duration = int(duration) if duration.isdigit() else 15

                    print(f"\n⚡ Launching {name}...")
                    attack_func(duration=duration)
                    print(f"\n✓ Attack finished! Check collector for IDS detections")
                else:
                    print("❌ Invalid choice")

            except KeyboardInterrupt:
                print("\n\n👋 Stopped")
                break
            except Exception as e:
                print(f"❌ Error: {e}")


if __name__ == "__main__":
    print("\n🔴 UNSUPERVISED IDS ATTACK INJECTOR")
    print(f"Target: {TARGET_IP}:{GATEWAY_PORT}\n")

    injector = IoTAttackInjector()
    injector.interactive_menu()

