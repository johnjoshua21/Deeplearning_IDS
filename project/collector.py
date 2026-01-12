import socket
import csv
import json
import os
import numpy as np
import joblib
from threading import Thread
from collections import deque
from flask import Flask, render_template_string
from tensorflow.keras.models import load_model
from config import HOST_IP, COLLECTOR_PORT, DATA_FOLDER, CSV_FILENAME, DASHBOARD_PORT, DISPLAY_WINDOW, CHART_WINDOW

# ===========================
# IDS MODEL CONFIGURATION
# ===========================
WINDOW_SIZE = 60
THRESHOLD = 1.20
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "medical_iot_ids", "model", "lstm_autoencoder.h5")
SCALER_PATH = os.path.join(BASE_DIR, "..", "medical_iot_ids", "model", "scaler.pkl")

# Load IDS Model
print("=" * 70)
print("Loading IDS Model...")
try:
    model = load_model(MODEL_PATH, compile=False)
    scaler = joblib.load(SCALER_PATH)
    print("✓ LSTM Autoencoder loaded successfully")
    print(f"✓ Threshold: {THRESHOLD}")
    IDS_ENABLED = True
except Exception as load_error:
    print(f"⚠️  Could not load IDS model: {load_error}")
    print("⚠️  Running without IDS detection")
    IDS_ENABLED = False
    model = None
    scaler = None

print("=" * 70)

UDP_IP = HOST_IP

# Create data folder
os.makedirs(DATA_FOLDER, exist_ok=True)

# Store recent packets
recent_packets = deque(maxlen=200)
packet_count = {'normal': 0, 'attack': 0}

# IDS: Sliding window
sensor_windows = {
    'S1': deque(maxlen=WINDOW_SIZE),
    'S2': deque(maxlen=WINDOW_SIZE),
    'S3': deque(maxlen=WINDOW_SIZE),
    'S4': deque(maxlen=WINDOW_SIZE),
    'S5': deque(maxlen=WINDOW_SIZE),
}

# Track IDS statistics
ids_stats = {
    'total_predictions': 0,
    'attacks_detected': 0,
    'normal_detected': 0,
    'last_error': 0.0
}

# Flask app
app = Flask(__name__)

# HTML Template
DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Unsupervised IDS</title>
    <meta http-equiv="refresh" content="2">
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 8px;
        }
        h1 { text-align: center; }
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .stat-card {
            padding: 20px;
            border: 1px solid #ddd;
            border-radius: 6px;
            text-align: center;
        }
        .stat-card h3 { margin: 0 0 10px 0; font-size: 14px; color: #666; }
        .stat-card .number { font-size: 32px; font-weight: bold; }
        table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        th { background: #333; color: white; padding: 12px; }
        td { padding: 10px; border-bottom: 1px solid #eee; }
        .ids-normal { background: #d4edda; color: #155724; padding: 4px 8px; border-radius: 3px; font-weight: bold; }
        .ids-attack { background: #f8d7da; color: #721c24; padding: 4px 8px; border-radius: 3px; font-weight: bold; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🛡️ Unsupervised IDS</h1>
        <div class="stats">
            <div class="stat-card">
                <h3>Total Packets</h3>
                <div class="number">{{ total_packets }}</div>
            </div>
            <div class="stat-card">
                <h3>Normal</h3>
                <div class="number">{{ normal_count }}</div>
            </div>
            <div class="stat-card">
                <h3>Anomalies</h3>
                <div class="number">{{ attack_count }}</div>
            </div>
            <div class="stat-card">
                <h3>Detection Rate</h3>
                <div class="number">{{ detection_rate }}%</div>
            </div>
        </div>
        <h2>Recent Packets</h2>
        <table>
            <thead>
                <tr>
                    <th>Timestamp</th>
                    <th>Sensor</th>
                    <th>Type</th>
                    <th>Value</th>
                    <th>IDS Status</th>
                    <th>Error</th>
                </tr>
            </thead>
            <tbody>
                {% for packet in packets %}
                <tr>
                    <td>{{ packet.timestamp }}</td>
                    <td>{{ packet.sensor_id }}</td>
                    <td>{{ packet.sensor_type }}</td>
                    <td>{{ packet.value }}</td>
                    <td>
                        {% if packet.ids_prediction == 1 %}
                            <span class="ids-attack">ANOMALY</span>
                        {% elif packet.ids_prediction == 0 %}
                            <span class="ids-normal">NORMAL</span>
                        {% else %}
                            BUFFERING
                        {% endif %}
                    </td>
                    <td>{{ '%.4f'|format(packet.ids_error) if packet.ids_error else '-' }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
    </div>
</body>
</html>
"""


@app.route('/')
def dashboard():
    total = packet_count['normal'] + packet_count['attack']
    detection_rate = round((packet_count['attack'] / total * 100) if total > 0 else 0, 1)

    return render_template_string(
        DASHBOARD_HTML,
        packets=list(reversed(recent_packets)),
        total_packets=total,
        normal_count=packet_count['normal'],
        attack_count=packet_count['attack'],
        detection_rate=detection_rate
    )


def udp_receiver():
    """Receive UDP packets and process with IDS"""
    import time

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((HOST_IP, COLLECTOR_PORT))

    csv_path = os.path.join(DATA_FOLDER, CSV_FILENAME)
    file_exists = os.path.exists(csv_path)
    csvfile = open(csv_path, 'a', newline='')
    writer = csv.DictWriter(
        csvfile,
        fieldnames=['timestamp', 'sensor_id', 'sensor_type', 'value', 'ids_prediction', 'ids_error', 'ids_confidence']
    )

    if not file_exists:
        writer.writeheader()

    print("=" * 70)
    print("🛡️  UNSUPERVISED IDS MODE")
    print("=" * 70)
    print(f"✓ Listening on: {HOST_IP}:{COLLECTOR_PORT}")
    print(f"✓ IDS Enabled: {IDS_ENABLED}")
    print("=" * 70)
    print("\n⏳ Collecting data...\n")

    cooldown_seconds = 5
    last_attack_time = 0
    total_packets = 0

    stats = {'normal': 0, 'attacks': 0, 'errors': []}

    try:
        while True:
            data, addr = sock.recvfrom(1024)
            packet = json.loads(data.decode())
            total_packets += 1

            sensor_id = packet['sensor_id']
            sensor_type = packet['sensor_type']
            value = packet['value']

            if sensor_id in sensor_windows:
                sensor_windows[sensor_id].append(value)

            ids_prediction = None
            ids_error = None
            ids_confidence = None

            all_full = all(len(sensor_windows[sid]) == WINDOW_SIZE for sid in sensor_windows)

            if total_packets % 100 == 0:
                print(f"\n🔍 Packet {total_packets}: IDS={IDS_ENABLED}, AllFull={all_full}")

            if IDS_ENABLED and all_full:
                try:
                    window_data = np.array([
                        list(sensor_windows['S1']),
                        list(sensor_windows['S2']),
                        list(sensor_windows['S3']),
                        list(sensor_windows['S4']),
                        list(sensor_windows['S5'])
                    ]).T

                    norm_window = scaler.transform(window_data)
                    model_input = norm_window.reshape(1, WINDOW_SIZE, 5)
                    reconstruction = model.predict(model_input, verbose=0)

                    ids_error = float(np.mean((model_input - reconstruction) ** 2))
                    ids_prediction = 1 if ids_error > THRESHOLD else 0

                    if ids_prediction == 1:
                        ids_confidence = min(100.0, ((ids_error - THRESHOLD) / THRESHOLD) * 100.0)
                    else:
                        ids_confidence = min(100.0, ((THRESHOLD - ids_error) / THRESHOLD) * 100.0)
                    ids_confidence = round(ids_confidence, 2)

                    ids_stats['total_predictions'] += 1
                    ids_stats['last_error'] = ids_error

                    if ids_prediction == 1:
                        ids_stats['attacks_detected'] += 1
                        stats['attacks'] += 1
                        if (time.time() - last_attack_time) > cooldown_seconds:
                            last_attack_time = time.time()
                            for sid in sensor_windows:
                                sensor_windows[sid].clear()
                    else:
                        ids_stats['normal_detected'] += 1
                        stats['normal'] += 1

                    stats['errors'].append(ids_error)

                    if total_packets % 50 == 0:
                        status_text = "🔴 ATTACK" if ids_prediction == 1 else "✅ NORMAL"
                        print(f"{status_text} | Err: {ids_error:.4f} | Thresh: {THRESHOLD}")

                except Exception as prediction_error:
                    print(f"❌ Prediction error: {prediction_error}")

            packet['ids_prediction'] = ids_prediction
            packet['ids_error'] = ids_error
            packet['ids_confidence'] = ids_confidence

            writer.writerow(packet)
            csvfile.flush()
            recent_packets.append(packet)

            if ids_prediction == 1:
                packet_count['attack'] += 1
            elif ids_prediction == 0:
                packet_count['normal'] += 1

            if ids_prediction is None:
                progress = sum(len(w) for w in sensor_windows.values()) / (WINDOW_SIZE * 5) * 100
                print(f"⏳ [{packet['timestamp']}] BUFFERING ({progress:.0f}%) | {sensor_type}: {value}")
            elif ids_prediction == 1:
                print(f"🔴 [{packet['timestamp']}] ANOMALY | {sensor_type}: {value} | Err: {ids_error:.4f}")
            else:
                print(f"✅ [{packet['timestamp']}] NORMAL | {sensor_type}: {value} | Err: {ids_error:.4f}")

            if total_packets % 100 == 0:
                avg_err = np.mean(stats['errors'][-100:]) if stats['errors'] else 0
                print(
                    f"\n📊 Total={total_packets}, Normal={stats['normal']}, Attacks={stats['attacks']}, AvgErr={avg_err:.4f}\n")

    except KeyboardInterrupt:
        print("\n\n🛑 Stopped")
    finally:
        csvfile.close()
        sock.close()


if __name__ == "__main__":
    receiver_thread = Thread(target=udp_receiver, daemon=True)
    receiver_thread.start()
    app.run(host='0.0.0.0', port=DASHBOARD_PORT, debug=False)