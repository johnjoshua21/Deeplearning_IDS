"""
ESP32 Medical IoT Sensor Network - MicroPython Version
Simulates ALL 5 sensors from single ESP32
Sends sensor data to Windows gateway via WiFi
"""

import network
import socket
import time
import ujson
import urandom
from machine import Pin
import _thread

# ============================================================================
# CONFIGURATION - CHANGE THESE!
# ============================================================================
WIFI_SSID = "YOUR_WIFI_NAME"          # ← Change to your WiFi name
WIFI_PASSWORD = "YOUR_WIFI_PASSWORD"  # ← Change to your WiFi password

GATEWAY_IP = "10.142.19.250"  # Your Windows IP (from ipconfig)
GATEWAY_PORT = 5005

SEND_INTERVAL = 1.0     # seconds between packets per sensor

# All 5 medical sensors configuration
SENSOR_CONFIGS = [
    ("S1", "FHR"),       # Fetal Heart Rate
    ("S2", "TOCO"),      # Uterine Contraction
    ("S3", "SpO2"),      # Blood Oxygen
    ("S4", "RespRate"),  # Respiratory Rate
    ("S5", "Temp")       # Temperature
]

# Sensor ranges (min, max)
SENSOR_RANGES = {
    'FHR': (110, 160),
    'TOCO': (0, 100),
    'SpO2': (95, 100),
    'RespRate': (12, 20),
    'Temp': (36.5, 37.5)
}

# ============================================================================
# LED INDICATOR (optional - built-in LED on most ESP32 boards)
# ============================================================================
try:
    led = Pin(2, Pin.OUT)  # GPIO2 is common for built-in LED
except:
    led = None

def blink_led(times=1):
    """Blink LED to show activity"""
    if led:
        for _ in range(times):
            led.on()
            time.sleep(0.05)
            led.off()
            time.sleep(0.05)

# ============================================================================
# WIFI CONNECTION
# ============================================================================
def connect_wifi():
    """Connect to WiFi network"""
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    
    if wlan.isconnected():
        print("Already connected!")
        print_network_info(wlan)
        return wlan
    
    print("\n" + "="*50)
    print("Connecting to WiFi...")
    print("SSID:", WIFI_SSID)
    print("="*50)
    
    wlan.connect(WIFI_SSID, WIFI_PASSWORD)
    
    # Wait for connection (max 15 seconds)
    max_wait = 15
    while max_wait > 0:
        if wlan.isconnected():
            break
        max_wait -= 1
        print(".", end="")
        time.sleep(1)
    
    print()
    
    if wlan.isconnected():
        print("✅ WiFi Connected!")
        print_network_info(wlan)
        blink_led(3)  # Blink 3 times on success
        return wlan
    else:
        print("❌ WiFi Connection Failed!")
        print("Check your SSID and password")
        return None

def print_network_info(wlan):
    """Print network connection details"""
    config = wlan.ifconfig()
    print("\nNetwork Info:")
    print(f"  IP Address: {config[0]}")
    print(f"  Subnet:     {config[1]}")
    print(f"  Gateway:    {config[2]}")
    print(f"  DNS:        {config[3]}")
    print()

# ============================================================================
# SENSOR SIMULATION
# ============================================================================
class MedicalSensor:
    def __init__(self, sensor_id, sensor_type, sock):
        self.sensor_id = sensor_id
        self.sensor_type = sensor_type
        self.min_val, self.max_val = SENSOR_RANGES[sensor_type]
        self.sock = sock  # Share socket among all sensors
        self.running = True
        self.packet_count = 0
    
    def generate_value(self):
        """Generate realistic sensor value"""
        center = (self.min_val + self.max_val) / 2
        variation = (self.max_val - self.min_val) * 0.15
        
        # Generate random value within variation range
        value = center + (urandom.random() - 0.5) * 2 * variation
        
        # Round based on sensor type
        if self.sensor_type in ['SpO2', 'RespRate']:
            return int(value)
        else:
            return round(value, 2)
    
    def get_timestamp(self):
        """Get current timestamp in required format"""
        t = time.localtime()
        return "{:04d}-{:02d}-{:02d} {:02d}:{:02d}:{:02d}.000".format(
            t[0], t[1], t[2], t[3], t[4], t[5]
        )
    
    def create_packet(self):
        """Create sensor data packet"""
        packet = {
            'sensor_id': self.sensor_id,
            'sensor_type': self.sensor_type,
            'timestamp': self.get_timestamp(),
            'value': self.generate_value()
        }
        return packet
    
    def send_packet(self, packet):
        """Send packet to gateway"""
        try:
            json_data = ujson.dumps(packet)
            self.sock.sendto(json_data.encode(), (GATEWAY_IP, GATEWAY_PORT))
            return True
        except Exception as e:
            print(f"[{self.sensor_id}] Send error: {e}")
            return False
    
    def run(self):
        """Main sensor loop - runs in separate thread"""
        print(f"✅ [{self.sensor_id}] {self.sensor_type} started!")
        
        while self.running:
            try:
                # Create and send packet
                packet = self.create_packet()
                success = self.send_packet(packet)
                
                if success:
                    self.packet_count += 1
                    print(f"[{self.sensor_id}] {self.sensor_type:10} = {packet['value']:6} "
                          f"(#{self.packet_count:03d})")
                    blink_led(1)  # Quick blink
                else:
                    print(f"⚠️ [{self.sensor_id}] Send failed")
                
                time.sleep(SEND_INTERVAL)
                
            except Exception as e:
                print(f"❌ [{self.sensor_id}] Error: {e}")
                time.sleep(2)
    
    def stop(self):
        """Stop the sensor"""
        self.running = False

# ============================================================================
# SENSOR NETWORK MANAGER
# ============================================================================
class SensorNetwork:
    def __init__(self):
        self.sensors = []
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.running = True
    
    def create_all_sensors(self):
        """Create all 5 medical sensors"""
        print("\n" + "="*50)
        print("Creating Medical Sensor Network")
        print("="*50)
        
        for sensor_id, sensor_type in SENSOR_CONFIGS:
            sensor = MedicalSensor(sensor_id, sensor_type, self.sock)
            self.sensors.append(sensor)
            min_val, max_val = SENSOR_RANGES[sensor_type]
            print(f"  [{sensor_id}] {sensor_type:10} | Range: {min_val}-{max_val}")
        
        print("="*50)
        print(f"Total Sensors: {len(self.sensors)}")
        print(f"Target: {GATEWAY_IP}:{GATEWAY_PORT}")
        print(f"Interval: {SEND_INTERVAL}s per sensor")
        print("="*50)
    
    def start_all(self):
        """Start all sensors in separate threads"""
        print("\n🟢 Starting all sensors...\n")
        
        # Start each sensor in its own thread
        for sensor in self.sensors:
            try:
                _thread.start_new_thread(sensor.run, ())
                time.sleep(0.2)  # Small delay between thread starts
            except Exception as e:
                print(f"Failed to start {sensor.sensor_id}: {e}")
        
        print("\n📡 All sensors active! Sending data...\n")
    
    def stop_all(self):
        """Stop all sensors"""
        print("\n\n🛑 Stopping all sensors...")
        self.running = False
        for sensor in self.sensors:
            sensor.stop()
        self.sock.close()
        print("✓ All sensors stopped")

# ============================================================================
# MAIN PROGRAM
# ============================================================================
def main():
    print("\n" + "="*50)
    print("ESP32 Medical IoT Sensor Network")
    print("ALL 5 SENSORS ON SINGLE ESP32")
    print("="*50)
    
    # Connect to WiFi
    wlan = connect_wifi()
    
    if not wlan:
        print("Cannot proceed without WiFi. Please check configuration.")
        return
    
    # Create sensor network
    network = SensorNetwork()
    network.create_all_sensors()
    network.start_all()
    
    # Keep running
    try:
        while network.running:
            time.sleep(1)
    except KeyboardInterrupt:
        network.stop_all()

# ============================================================================
# RUN
# ============================================================================
if __name__ == "__main__":
    main()