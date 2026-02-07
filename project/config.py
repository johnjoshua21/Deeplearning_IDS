# Configuration for IDS System - Network Version

# ============================================================================
# NETWORK CONFIGURATION - IMPORTANT!
# ============================================================================
# For Windows (Host) - Where sensors/gateway/collector run:
#   Set HOST_IP to "0.0.0.0" (listen on all interfaces)
#
# For WSL/Attacker - Where attack injector runs:
#   Set TARGET_IP to your Windows IP address (e.g., "192.168.1.100")
#   Find Windows IP: Open PowerShell → run: ipconfig
# ============================================================================

# Network Mode: "localhost" or "network"
NETWORK_MODE = "network"  # Change to "network" for cross-machine attacks

# For HOST (Windows) - Sensors, Gateway, Collector
if NETWORK_MODE == "localhost":
    HOST_IP = "127.0.0.1"
    TARGET_IP = "127.0.0.1"
else:
    # Listen on all interfaces (Windows host)
    HOST_IP = "0.0.0.0"

    # Target IP for attacker (set this to your Windows IP)
    # CHANGE THIS to your Windows IP from ipconfig!
    TARGET_IP = "10.142.19.250"    # ← CHANGE THIS!

# Port Configuration
GATEWAY_PORT = 5005  # Sensors send to Gateway
COLLECTOR_PORT = 5006  # Gateway forwards to Collector
UDP_PORT = GATEWAY_PORT  # For backward compatibility

# Sensor Settings
SENSOR_SEND_INTERVAL = 1.0  # seconds between packets

# Normal ranges for medical sensors
SENSOR_RANGES = {
    'FHR': (110, 160),  # Fetal Heart Rate (bpm)
    'TOCO': (0, 100),  # Uterine contraction intensity
    'SpO2': (95, 100),  # Blood oxygen saturation (%)
    'RespRate': (12, 20),  # Respiratory rate (breaths/min)
    'Temp': (36.5, 37.5)  # Temperature (Celsius)
}

# Data Storage
DATA_FOLDER = "data"
CSV_FILENAME = "sensor_data.csv"

# Dashboard Settings
DASHBOARD_PORT = 8050
DISPLAY_WINDOW = 50  # Number of recent packets to show
CHART_WINDOW = 100  # Number of values to show in graphs


# Helper function to display current configuration
def show_config():
    print("\n" + "=" * 70)
    print("NETWORK CONFIGURATION")
    print("=" * 70)
    print(f"Mode: {NETWORK_MODE}")
    print(f"Host IP (Windows - Listening): {HOST_IP}")
    print(f"Target IP (For Attacker): {TARGET_IP}")
    print(f"Gateway Port: {GATEWAY_PORT}")
    print(f"Collector Port: {COLLECTOR_PORT}")
    print(f"Dashboard Port: {DASHBOARD_PORT}")
    print("=" * 70)

    if NETWORK_MODE == "network":
        print("\n⚠️  IMPORTANT:")
        print(f"   1. Make sure TARGET_IP is set to your Windows IP address")
        print(f"   2. Run sensors/gateway/collector on Windows")
        print(f"   3. Run attack_injector from WSL")
        print(f"   4. Firewall ports {GATEWAY_PORT}, {COLLECTOR_PORT}, {DASHBOARD_PORT} must be open")
        print("=" * 70 + "\n")


if __name__ == "__main__":
    show_config()