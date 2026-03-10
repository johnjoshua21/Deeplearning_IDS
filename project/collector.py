import socket
import json
import time
import numpy as np
import joblib
from threading import Thread
from collections import deque
from flask import Flask, render_template_string, Response
from tensorflow.keras.models import load_model
from config import HOST_IP, COLLECTOR_PORT, SENSOR_RANGES, DASHBOARD_PORT

# ======================================================
# EMAIL ALERT CONFIG  ← 🔧 EDIT THESE BEFORE RUNNING
# ======================================================
EMAIL_ENABLED       = True                          # Set False to disable
SENDER_EMAIL        = "medical.iot.alert@gmail.com"        # Your Gmail address
SENDER_PASSWORD     = "lqkuqjmttndaqdvt"         # Gmail App Password (NOT your login password)
                                                    # Get it: Google Account → Security → App Passwords
RECEIVER_EMAILS = [
    "johnjoshua2118@gmail.com",
    "gobinath.t67@gmail.com",
    "cjoshika1721@gmail.com"
]
# Who receives the alert
SMTP_SERVER         = "smtp.gmail.com"
SMTP_PORT           = 587

EMAIL_COOLDOWN_SEC  = 60   # Min seconds between emails (prevents spam)
                           # Set to 0 to send every attack

# ======================================================
# BLOCKCHAIN
# ======================================================
try:
    from blockchain_logger import BlockchainLogger
    blockchain = BlockchainLogger()
    BLOCKCHAIN_ENABLED = blockchain.enabled
except ImportError:
    print("⚠️  blockchain_logger.py not found - blockchain logging disabled")
    BLOCKCHAIN_ENABLED = False
    blockchain = None

blockchain_stats = {'total_logged': 0, 'last_tx_hash': '-'}

# ======================================================
# CONFIG
# ======================================================
WINDOW_SIZE   = 60
FEATURE_IDS   = ["S1", "S2", "S3", "S4", "S5"]

FEATURE_NAMES = {
    "S1": "FHR",
    "S2": "TOCO",
    "S3": "SpO2",
    "S4": "RespRate",
    "S5": "Temp"
}

EXPECTED_SENSOR_TYPE = FEATURE_NAMES.copy()

MODEL_PATH  = "../medical_iot_ids/model/lstm_autoencoder.h5"
SCALER_PATH = "../medical_iot_ids/model/scaler.pkl"

CALIBRATION_WINDOWS = 120
K_SIGMA             = 2.5

ATTACK_CONFIRMATION  = 3
RECOVERY_CONFIRMATION = 8
MIN_ATTACK_DURATION  = 1.2

# ======================================================
# EMAIL HELPER
# ======================================================
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text       import MIMEText

_last_email_time = 0   # tracks cooldown

def send_attack_email(attack_type, sensors, duration, packets, error, threshold):
    """Send an HTML email alert for a confirmed attack. Non-blocking (runs in thread)."""
    global _last_email_time

    if not EMAIL_ENABLED:
        return

    now = time.time()
    if now - _last_email_time < EMAIL_COOLDOWN_SEC:
        print(f"📧 Email skipped (cooldown {EMAIL_COOLDOWN_SEC}s)")
        return

    def _send():
        global _last_email_time
        try:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

            # ── Subject ──────────────────────────────────────────
            subject = f"🚨 MEDICAL IoT ATTACK DETECTED — {attack_type} [{timestamp}]"

            # ── HTML body ─────────────────────────────────────────
            html = f"""
<!DOCTYPE html>
<html>
<head>
<style>
  body      {{ font-family: Segoe UI, Arial, sans-serif; background:#f0f4f8; margin:0; padding:20px; }}
  .card     {{ background:#ffffff; border-radius:12px; padding:28px 32px; max-width:600px;
               margin:auto; box-shadow:0 4px 20px rgba(0,0,0,0.10); }}
  .banner   {{ background:#c0392b; color:#fff; border-radius:8px; padding:16px 20px;
               margin-bottom:24px; }}
  .banner h2{{ margin:0; font-size:20px; }}
  .banner p {{ margin:4px 0 0; font-size:13px; opacity:0.85; }}
  table     {{ width:100%; border-collapse:collapse; margin-top:16px; }}
  th        {{ background:#1f3864; color:#fff; padding:10px 14px; text-align:left; font-size:13px; }}
  td        {{ padding:10px 14px; font-size:14px; border-bottom:1px solid #e2e8f0; }}
  tr:last-child td {{ border-bottom:none; }}
  tr:nth-child(even) td {{ background:#f8fafc; }}
  .label    {{ font-weight:600; color:#1f3864; width:40%; }}
  .footer   {{ margin-top:24px; font-size:12px; color:#8b9ab0; text-align:center; }}
  .severity {{ display:inline-block; background:#c0392b; color:#fff; padding:3px 12px;
               border-radius:20px; font-size:13px; font-weight:bold; }}
</style>
</head>
<body>
<div class="card">

  <div class="banner">
    <h2>🚨 Intrusion Detected — Medical IoT IDS</h2>
    <p>A confirmed cyberattack was detected on the gynecology sensor network.</p>
  </div>

  <p><span class="severity">HIGH SEVERITY</span></p>

  <table>
    <tr><th colspan="2">Attack Details</th></tr>
    <tr><td class="label">Timestamp</td>      <td>{timestamp}</td></tr>
    <tr><td class="label">Attack Type</td>    <td><b>{attack_type}</b></td></tr>
    <tr><td class="label">Sensors Affected</td><td>{sensors}</td></tr>
    <tr><td class="label">Duration</td>       <td>{duration} seconds</td></tr>
    <tr><td class="label">Packets Flagged</td><td>{packets}</td></tr>
    <tr><td class="label">Reconstruction Error</td><td>{error:.6f}</td></tr>
    <tr><td class="label">Detection Threshold</td> <td>{threshold:.6f}</td></tr>
  </table>

  <p style="margin-top:20px; font-size:13px; color:#c0392b;">
    ⚠️ Please review sensor data immediately and verify patient safety.
  </p>

  <div class="footer">
    This is an automated alert from the Medical IoT Intrusion Detection System.<br>
  </div>
</div>
</body>
</html>
"""
            # ── Plain text fallback ───────────────────────────────
            plain = (
                f"MEDICAL IoT ATTACK ALERT\n"
                f"========================\n"
                f"Timestamp   : {timestamp}\n"
                f"Attack Type : {attack_type}\n"
                f"Sensors     : {sensors}\n"
                f"Duration    : {duration} s\n"
                f"Packets     : {packets}\n"
                f"Error       : {error:.6f}\n"
                f"Threshold   : {threshold:.6f}\n\n"
                f"Please review patient sensor data immediately."
            )

            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"]    = SENDER_EMAIL
            msg["To"] = ", ".join(RECEIVER_EMAILS)
            msg.attach(MIMEText(plain, "plain"))
            msg.attach(MIMEText(html,  "html"))

            with smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=15) as server:
                server.ehlo()
                server.starttls()
                server.login(SENDER_EMAIL, SENDER_PASSWORD)
                server.sendmail(SENDER_EMAIL, RECEIVER_EMAILS, msg.as_string())

            _last_email_time = time.time()


        except smtplib.SMTPAuthenticationError:
            print("❌ Email auth failed — check SENDER_EMAIL and SENDER_PASSWORD (use App Password, not Gmail login)")
        except smtplib.SMTPException as e:
            print(f"❌ SMTP error: {e}")
        except Exception as e:
            print(f"❌ Email error: {e}")

    Thread(target=_send, daemon=True).start()

# ======================================================
# LOAD MODEL  (Python 3.10 + TF 2.12 — plain joblib)
# ======================================================
model  = load_model(MODEL_PATH, compile=False)
scaler = joblib.load(SCALER_PATH)
print("✅ IDS Model Loaded")

# ======================================================
# STATE
# ======================================================
sensor_windows = {sid: deque(maxlen=WINDOW_SIZE) for sid in FEATURE_IDS}
last_value     = {sid: None for sid in FEATURE_IDS}

recent_packets = deque(maxlen=400)
error_history  = deque(maxlen=CALIBRATION_WINDOWS)

CALIBRATION_DONE  = False
THRESHOLD         = None

ATTACK_ACTIVE     = False
ATTACK_START_TIME = None
FIRST_ANOMALY_TIME = None

CONSECUTIVE_ANOMALIES = 0
NORMAL_STREAK         = 0
LAST_DECISION         = "CALIBRATING"

TOTAL            = 0
NORMAL           = 0
INJECTED_ATTACKS = 0
DETECTED_ATTACKS = 0
PENDING_INJECTED = 0

ATTACK_CONFIRMED_IN_SESSION = False

current_attack = {
    "sensors": set(),
    "packets": 0,
    "type_counts": {}
}

last_attack_summary = {
    "type": "-",
    "sensors": "-",
    "duration": "-",
    "packets": 0
}

attack_history = deque(maxlen=6)

# email stats for dashboard
email_stats = {'total_sent': 0, 'last_sent': '-'}

# ======================================================
# HELPERS
# ======================================================
def compute_threshold():
    global THRESHOLD
    THRESHOLD = float(np.mean(error_history) + K_SIGMA * np.std(error_history))


def security_violation(sensor, value, prev, sid):
    lo, hi = SENSOR_RANGES[sensor]
    if EXPECTED_SENSOR_TYPE[sid] != sensor:
        return "Spoofing"
    if value in [0, -1]:
        return "Jamming"
    if value < lo or value > hi:
        return "Spoofing"
    if prev is not None and abs(value - prev) > 0.4 * (hi - lo):
        return "MITM / Manipulation"
    return None


def sensors_all_normal():
    for sid in FEATURE_IDS:
        if not sensor_windows[sid]:
            return False
        v = sensor_windows[sid][-1]
        lo, hi = SENSOR_RANGES[FEATURE_NAMES[sid]]
        if v < lo or v > hi or v in [0, -1]:
            return False
    return True

# ======================================================
# UDP RECEIVER
# ======================================================
def udp_receiver():
    global TOTAL, NORMAL, CALIBRATION_DONE, ATTACK_ACTIVE
    global ATTACK_START_TIME, FIRST_ANOMALY_TIME
    global CONSECUTIVE_ANOMALIES, NORMAL_STREAK
    global LAST_DECISION, INJECTED_ATTACKS, DETECTED_ATTACKS
    global ATTACK_CONFIRMED_IN_SESSION, PENDING_INJECTED, last_attack_summary

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((HOST_IP, COLLECTOR_PORT))
    print("🛡️ IDS Listening...")

    while True:
        try:
            pkt = json.loads(sock.recvfrom(4096)[0].decode())
            pkt["epoch"]     = time.time()
            pkt["timestamp"] = time.strftime("%H:%M:%S")

            # ---------- ATTACK META ----------
            if pkt.get("type") == "ATTACK_META":
                INJECTED_ATTACKS += 1
                PENDING_INJECTED += 1
                continue

            sid   = pkt["sensor_id"]
            stype = pkt["sensor_type"]
            value = pkt["value"]

            if sid not in FEATURE_IDS:
                continue

            TOTAL += 1
            prev = last_value[sid]
            sensor_windows[sid].append(value)

            # ---------- CALIBRATION — window not full yet ----------
            if not all(len(w) == WINDOW_SIZE for w in sensor_windows.values()):
                pkt.update({"ids_status": "CALIBRATING", "attack_type": "-", "ids_error": "-"})
                recent_packets.appendleft(pkt)
                LAST_DECISION = "CALIBRATING"
                last_value[sid] = value
                continue

            # ---------- LSTM INFERENCE ----------
            window = scaler.transform(
                np.array([list(sensor_windows[s]) for s in FEATURE_IDS]).T
            )
            x     = window.reshape(1, WINDOW_SIZE, len(FEATURE_IDS))
            recon = model.predict(x, verbose=0)
            error = float(np.mean((x - recon) ** 2))

            # ---------- CALIBRATION — collecting baseline errors ----------
            if not CALIBRATION_DONE:
                if security_violation(stype, value, prev, sid) is None:
                    error_history.append(error)

                if len(error_history) == CALIBRATION_WINDOWS:
                    compute_threshold()
                    CALIBRATION_DONE = True
                    print(f"✅ Calibration complete | Threshold = {THRESHOLD:.6f}")

                pkt.update({"ids_status": "CALIBRATING", "attack_type": "-", "ids_error": "-"})
                recent_packets.appendleft(pkt)
                last_value[sid] = value
                continue

            # ---------- DETECTION ----------
            violation  = security_violation(stype, value, prev, sid)
            is_anomaly = (error > THRESHOLD) and (violation is not None)

            if is_anomaly:
                if CONSECUTIVE_ANOMALIES == 0:
                    FIRST_ANOMALY_TIME = pkt["epoch"]
                CONSECUTIVE_ANOMALIES += 1
                NORMAL_STREAK = 0
            else:
                NORMAL_STREAK += 1
                CONSECUTIVE_ANOMALIES = 0

            # Attack START
            if CONSECUTIVE_ANOMALIES >= ATTACK_CONFIRMATION and not ATTACK_ACTIVE:
                ATTACK_ACTIVE = True
                ATTACK_START_TIME = FIRST_ANOMALY_TIME
                current_attack["sensors"].clear()
                current_attack["packets"] = 0
                current_attack["type_counts"].clear()

            if ATTACK_ACTIVE and not ATTACK_CONFIRMED_IN_SESSION:
                if pkt["epoch"] - ATTACK_START_TIME >= MIN_ATTACK_DURATION:
                    ATTACK_CONFIRMED_IN_SESSION = True

            if is_anomaly:
                pkt["ids_status"] = "ATTACK"
                pkt["attack_type"] = violation
                current_attack["packets"] += 1
                current_attack["sensors"].add(stype)
                current_attack["type_counts"][violation] = \
                    current_attack["type_counts"].get(violation, 0) + 1
            else:
                pkt["ids_status"] = "NORMAL"
                pkt["attack_type"] = "-"
                NORMAL += 1

            pkt["ids_error"] = round(error, 6)
            recent_packets.appendleft(pkt)
            LAST_DECISION = "ATTACK" if ATTACK_ACTIVE else "NORMAL"

            # ---------- ATTACK END ----------
            if ATTACK_ACTIVE and NORMAL_STREAK >= RECOVERY_CONFIRMATION and sensors_all_normal():
                duration = round(pkt["epoch"] - ATTACK_START_TIME, 1)

                attack_type = max(
                    current_attack["type_counts"],
                    key=current_attack["type_counts"].get
                )
                sensors_affected = ", ".join(sorted(current_attack["sensors"]))

                last_attack_summary = {
                    "type":     attack_type,
                    "sensors":  sensors_affected,
                    "duration": duration,
                    "packets":  current_attack["packets"]
                }

                attack_history.appendleft({
                    "time": time.strftime("%H:%M:%S"),
                    **last_attack_summary
                })

                # ── EMAIL ALERT ──────────────────────────────────
                send_attack_email(
                    attack_type = attack_type,
                    sensors     = sensors_affected,
                    duration    = duration,
                    packets     = current_attack["packets"],
                    error       = error,
                    threshold   = THRESHOLD
                )
                email_stats['total_sent'] += 1
                email_stats['last_sent']   = time.strftime("%H:%M:%S")
                # ─────────────────────────────────────────────────

                # ── BLOCKCHAIN LOG ───────────────────────────────
                if BLOCKCHAIN_ENABLED:
                    tx = blockchain.log_attack(
                        attack_type      = attack_type,
                        sensors_affected = sensors_affected,
                        error_value      = error
                    )
                    if tx:
                        blockchain_stats['total_logged'] += 1
                        blockchain_stats['last_tx_hash'] = tx
                        print(f"🔗 Blockchain TX: {tx[:20]}...")
                # ─────────────────────────────────────────────────

                if PENDING_INJECTED > 0:
                    DETECTED_ATTACKS += 1
                    PENDING_INJECTED -= 1

                ATTACK_ACTIVE               = False
                ATTACK_CONFIRMED_IN_SESSION = False
                CONSECUTIVE_ANOMALIES       = 0
                NORMAL_STREAK               = 0
                FIRST_ANOMALY_TIME          = None

            last_value[sid] = value

        except Exception as e:
            print("❌ Collector error:", e)

# ======================================================
# DASHBOARD
# ======================================================
app  = Flask(__name__)
HTML = """<!DOCTYPE html>
<html>
<head>
<title>Medical IoT IDS</title>
<meta http-equiv="refresh" content="2">
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
body{background:#0e1117;color:#e6edf3;font-family:Segoe UI;padding:20px}
.section{background:#161b22;border-radius:14px;padding:16px;margin-bottom:20px}
.header{display:flex;justify-content:space-between;align-items:center}
.status{padding:10px 24px;border-radius:24px;font-weight:bold}
.status.NORMAL{background:#2ea043;color:black}
.status.ATTACK{background:#f85149;color:black}
.status.CALIBRATING{background:#d29922;color:black}
.kpis{display:grid;grid-template-columns:repeat(5,1fr);gap:15px}
.kpi span{color:#8b949e;font-size:12px}
.kpi p{font-size:22px;font-weight:bold}
.three-col{display:grid;grid-template-columns:1fr 1fr 1fr;gap:20px}
.two-col{display:grid;grid-template-columns:1fr 1fr;gap:20px}
.history{max-height:180px;overflow-y:auto}
.graph-grid{display:grid;grid-template-columns:repeat(3,1fr);
  grid-template-areas:"g1 g2 g3" "g4 g5 .";gap:20px}
.graph{background:#0e1117;padding:12px;border-radius:12px}
.g1{grid-area:g1}.g2{grid-area:g2}.g3{grid-area:g3}.g4{grid-area:g4}.g5{grid-area:g5}
table{width:100%;border-collapse:collapse}
th,td{padding:8px;border-bottom:1px solid #30363d;text-align:center;font-size:13px}
th{color:#8b949e}
tr.NORMAL{color:#2ea043}
tr.CALIBRATING,
tr.CALIBRATING td{
  color:#d29922;
  background:transparent !important;
}
tr.CALIBRATING{color:#d29922;background:#2d210f}
.bar{background:#0d1117;border:1px solid;border-radius:10px;
  padding:9px 16px;margin-bottom:16px;font-size:13px;
  display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.bar a{font-family:monospace;font-size:12px}
.bar-bc{border-color:#1f6feb;color:#58a6ff}
.bar-bc a{color:#79c0ff}
.bar-em{border-color:#2ea043;color:#2ea043}
.bar-off{border-color:#30363d;color:#8b949e}
.btn-csv{
  background:#1f6feb;
  color:#ffffff;
  border:none;
  border-radius:10px;
  padding:8px 18px;
  font-size:13px;
  font-weight:600;
  cursor:pointer;
  display:inline-flex;
  align-items:center;
  gap:6px;
  transition:all 0.2s ease;
  box-shadow:0 0 0 1px rgba(240,246,252,0.1);
}

.btn-csv:hover{
  background:#388bfd;
  transform:translateY(-1px);
  box-shadow:0 4px 10px rgba(0,0,0,0.4);
}

.btn-csv:active{
  transform:translateY(0);
}
</style>
</head>
<body>

<!-- HEADER -->
<div class="section header">
  <h2>🛡️ Medical IoT IDS</h2>
  <div style="display:flex;align-items:center;gap:10px">
   <form action="/download_csv" method="get" style="display:inline">
      <button class="btn-csv" type="submit">⬇ Download CSV</button>
   </form>
    <div class="status {{ decision }}">{{ decision }}</div>
  </div>
</div>

<!-- BLOCKCHAIN BAR -->
{% if blockchain_enabled %}
<div class="bar bar-bc">
  🔗 <b>BLOCKCHAIN ACTIVE</b> &nbsp;·&nbsp; Ethereum Sepolia
  &nbsp;·&nbsp; On-Chain Logs: <b>{{ bc_logged }}</b>
  {% if bc_tx != '-' %}
    &nbsp;·&nbsp; Last TX:
    <a href="https://sepolia.etherscan.io/tx/{{ bc_tx }}" target="_blank">{{ bc_tx[:30] }}...</a>
    <a href="https://sepolia.etherscan.io/tx/{{ bc_tx }}" target="_blank">↗ Etherscan</a>
  {% endif %}
</div>
{% else %}
<div class="bar bar-off">⚠ Blockchain disabled</div>
{% endif %}

<!-- EMAIL BAR -->
{% if email_enabled %}
<div class="bar bar-em">
  📧 <b>EMAIL ALERTS ACTIVE</b> &nbsp;·&nbsp; → {{ receiver }}
  &nbsp;·&nbsp; Sent: <b>{{ em_sent }}</b>
  {% if em_last != '-' %}&nbsp;·&nbsp; Last sent: {{ em_last }}{% endif %}
  &nbsp;·&nbsp; Cooldown: {{ cooldown }}s
</div>
{% else %}
<div class="bar bar-off">📧 Email alerts disabled — set EMAIL_ENABLED = True in collector.py</div>
{% endif %}

<!-- KPIs -->
<div class="section kpis">
  <div class="kpi"><span>Total Packets</span><p>{{ total }}</p></div>
  <div class="kpi"><span>Normal</span><p>{{ normal }}</p></div>
  <div class="kpi"><span>Injected</span><p>{{ injected }}</p></div>
  <div class="kpi"><span>Detected</span><p>{{ detected }}</p></div>
  <div class="kpi"><span>Rate</span><p>{{ rate }}%</p></div>
</div>

<!-- ATTACK SUMMARY + HISTORY -->
<div class="two-col">
  <div class="section">
    <h4>Attack Summary</h4>
    <p><b>Type:</b> {{ summary.type }}</p>
    <p><b>Sensors:</b> {{ summary.sensors }}</p>
    <p><b>Duration:</b> {{ summary.duration }} s</p>
    <p><b>Packets:</b> {{ summary.packets }}</p>
  </div>
  <div class="section">
    <h4>Attack History</h4>
    <div class="history">
    {% for a in history %}
      <p>{{ a.time }} | {{ a.type }} | {{ a.sensors }} | {{ a.duration }} s | {{ a.packets }} packets</p>
    {% endfor %}
    </div>
  </div>
</div>

<!-- SENSOR GRAPHS -->
<div class="section graph-grid">
  <div class="graph g1"><h4>FHR</h4><canvas id="S1"></canvas></div>
  <div class="graph g2"><h4>TOCO</h4><canvas id="S2"></canvas></div>
  <div class="graph g3"><h4>SpO₂</h4><canvas id="S3"></canvas></div>
  <div class="graph g4"><h4>RespRate</h4><canvas id="S4"></canvas></div>
  <div class="graph g5"><h4>Temp</h4><canvas id="S5"></canvas></div>
</div>

<!-- LIVE TABLE -->
<div class="section">
  <h4>Live Sensor Table</h4>
  <div style="max-height:260px;overflow-y:auto">
  <table>
    <tr><th>Time</th><th>Sensor</th><th>Value</th><th>Status</th></tr>
    {% for p in packets %}
    <tr class="{{ p.ids_status }}">
      <td>{{ p.timestamp }}</td>
      <td>{{ p.sensor_type }}</td>
      <td>{{ p.value }}</td>
      <td>{{ p.ids_status }}</td>
    </tr>
    {% endfor %}
  </table>
  </div>
</div>

<script>
const packets={{ packets|tojson }};
["S1","S2","S3","S4","S5"].forEach(id=>{
  const rows=packets.filter(p=>p.sensor_id===id).reverse();
  const ctx=document.getElementById(id);
  if(!ctx)return;
  new Chart(ctx,{type:"line",
    data:{labels:rows.map(p=>p.timestamp),
      datasets:[{data:rows.map(p=>p.value),
        borderColor:"#2ea043",
        pointBackgroundColor:rows.map(p=>
          p.ids_status==="ATTACK"?"#f85149":
          p.ids_status==="CALIBRATING"?"#d29922":"#2ea043"),
        pointRadius:4,tension:0.3}]},
    options:{plugins:{legend:{display:false}},
      scales:{x:{display:false}}}});
});
</script>
</body>
</html>
"""

@app.route("/")
def dashboard():
    rate = round((DETECTED_ATTACKS / INJECTED_ATTACKS) * 100, 2) if INJECTED_ATTACKS else 0
    return render_template_string(
        HTML,
        total    = TOTAL,
        normal   = NORMAL,
        injected = INJECTED_ATTACKS,
        detected = DETECTED_ATTACKS,
        rate     = rate,
        decision = LAST_DECISION,
        packets  = list(recent_packets),
        summary  = last_attack_summary,
        history  = list(attack_history),
        # blockchain
        blockchain_enabled = BLOCKCHAIN_ENABLED,
        bc_logged          = blockchain_stats['total_logged'],
        bc_tx              = blockchain_stats['last_tx_hash'],
        # email
        email_enabled = EMAIL_ENABLED,
        receiver      = RECEIVER_EMAILS,
        em_sent       = email_stats['total_sent'],
        em_last       = email_stats['last_sent'],
        cooldown      = EMAIL_COOLDOWN_SEC,
    )

@app.route("/download_csv")
def download_csv():
    import csv, io
    output = io.StringIO()
    fields = ['timestamp','sensor_id','sensor_type','value',
              'ids_status','attack_type','ids_error']
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction='ignore')
    writer.writeheader()
    for p in reversed(list(recent_packets)):
        writer.writerow(p)
    filename = f"sensor_data_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    return Response(output.getvalue(), mimetype='text/csv',
                    headers={'Content-Disposition': f'attachment; filename={filename}'})

if __name__ == "__main__":
    print(f"📧 Email alerts: {'ENABLED → ' if EMAIL_ENABLED else 'DISABLED'}")
    Thread(target=udp_receiver, daemon=True).start()
    app.run(host="0.0.0.0", port=DASHBOARD_PORT, debug=False)