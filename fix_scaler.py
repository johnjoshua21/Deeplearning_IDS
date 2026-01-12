"""
Recreate scaler with Python 3.7 compatible pickle protocol
Run this once, then restart collector.py
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os

print("=" * 70)
print("RECREATING SCALER WITH PYTHON 3.7 COMPATIBILITY")
print("=" * 70)

# Check if input file exists
input_file = "medical_iot_ids/processed/final_5sensor.csv"
output_file = "medical_iot_ids/model/scaler.pkl"

if not os.path.exists(input_file):
    print(f"❌ ERROR: {input_file} not found!")
    print("   Make sure you're running this from the project root directory")
    exit(1)

print(f"\n✓ Found input file: {input_file}")

# Load data
print("Loading data...")
df = pd.read_csv(input_file)
print(f"✓ Loaded {len(df)} rows with {len(df.columns)} columns")
print(f"  Columns: {list(df.columns)}")

# Create and fit scaler
print("\nFitting StandardScaler...")
scaler = StandardScaler()
scaler.fit(df)
print("✓ Scaler fitted")

# Check scaler stats
print("\nScaler statistics:")
print(f"  Mean: {scaler.mean_}")
print(f"  Std:  {scaler.scale_}")

# Save with protocol 4 (compatible with Python 3.7)
print(f"\nSaving scaler to: {output_file}")
print("  Using pickle protocol 4 (Python 3.7 compatible)")

# Ensure directory exists
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# Save with explicit protocol
joblib.dump(scaler, output_file, protocol=4)
print("✓ Scaler saved successfully!")

# Verify it can be loaded
print("\nVerifying scaler can be loaded...")
try:
    test_scaler = joblib.load(output_file)
    print("✓ Scaler loads successfully!")

    # Test transform
    import numpy as np

    test_data = np.array([[135, 50, 97, 16, 37]])  # Sample sensor values
    normalized = test_scaler.transform(test_data)
    print(f"✓ Transform works! Sample: {test_data[0]} → {normalized[0]}")

except Exception as e:
    print(f"❌ ERROR: Could not load scaler: {e}")
    exit(1)

print("\n" + "=" * 70)
print("SUCCESS! You can now restart collector.py")
print("=" * 70)
print("\nNext steps:")
print("  1. Press Ctrl+C to stop collector.py (if running)")
print("  2. Run: python project/collector.py")
print("  3. You should see: '✓ LSTM Autoencoder loaded successfully'")
print("=" * 70)