################################################################################
# FILE 1: project/sensor_node.py (COMPLETE FIXED VERSION)
################################################################################

import socket
import time
import random
import json
import threading
from datetime import datetime
from config import TARGET_IP, GATEWAY_PORT, SENSOR_SEND_INTERVAL, SENSOR_RANGES

UDP_IP = TARGET_IP
UDP_PORT = GATEWAY_PORT


class MedicalSensor:
    def __init__(self, sensor_id, sensor_type):
        self.sensor_id = sensor_id
        self.sensor_type = sensor_type
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.min_val, self.max_val = SENSOR_RANGES[sensor_type]
        self.running = True

    def generate_normal_value(self):
        """Generate realistic sensor value with slight variation"""
        center = (self.min_val + self.max_val) / 2
        variation = (self.max_val - self.min_val) * 0.15
        return round(random.uniform(center - variation, center + variation), 2)

    def create_packet(self):
        """Create a data packet - NO ATTACK LABELS"""
        packet = {
            'sensor_id': self.sensor_id,
            'sensor_type': self.sensor_type,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
            'value': self.generate_normal_value()
            # ❌ REMOVED: 'is_attack': 0
            # ❌ REMOVED: 'attack_type': '-'
        }
        return json.dumps(packet)

    def send_data(self):
        """Continuously send sensor data"""
        try:
            while self.running:
                packet = self.create_packet()
                self.sock.sendto(packet.encode(), (TARGET_IP, GATEWAY_PORT))

                data = json.loads(packet)
                print(f"[{data['timestamp']}] {self.sensor_type} (ID:{self.sensor_id}): {data['value']}")

                time.sleep(SENSOR_SEND_INTERVAL)
        except Exception as e:
            print(f"Error in {self.sensor_type}: {e}")

    def stop(self):
        """Stop sending data"""
        self.running = False
        self.sock.close()


class SensorNetwork:
    def __init__(self):
        self.sensors = []
        self.threads = []

    def create_sensors(self):
        """Create all 5 medical sensors"""
        sensor_configs = [
            ('S1', 'FHR'),
            ('S2', 'TOCO'),
            ('S3', 'SpO2'),
            ('S4', 'RespRate'),
            ('S5', 'Temp')
        ]

        for sensor_id, sensor_type in sensor_configs:
            sensor = MedicalSensor(sensor_id, sensor_type)
            self.sensors.append(sensor)

        print("=" * 70)
        print("         MEDICAL SENSOR NETWORK - ALL NODES ACTIVE")
        print("=" * 70)
        print(f"Network Configuration:")
        print(f"  • Total Sensors: 5")
        print(f"  • Target Gateway: {TARGET_IP}:{GATEWAY_PORT}")
        print(f"  • Send Interval: {SENSOR_SEND_INTERVAL}s")
        print(f"\nSensor Details:")
        for sensor in self.sensors:
            print(f"  [{sensor.sensor_id}] {sensor.sensor_type:10} | Range: {sensor.min_val}-{sensor.max_val}")
        print("=" * 70)
        print("\n🟢 All sensors started! Sending normal data...")
        print("📡 No attack labels - IDS will detect anomalies\n")

    def start_all(self):
        """Start all sensors in separate threads"""
        self.create_sensors()

        for sensor in self.sensors:
            thread = threading.Thread(target=sensor.send_data, daemon=True)
            thread.start()
            self.threads.append(thread)

    def stop_all(self):
        """Stop all sensors"""
        print("\n\n🛑 Stopping all sensors...")
        for sensor in self.sensors:
            sensor.stop()
        print("✓ All sensors stopped")


if __name__ == "__main__":
    network = SensorNetwork()

    try:
        network.start_all()

        # Keep running until user stops
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        network.stop_all()
