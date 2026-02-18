import pandas as pd

print("Loading all sensor CSVs...\n")

fhr_toco = pd.read_csv("medical_iot_ids/processed/fhr_toco_ctu.csv")
spo2     = pd.read_csv("medical_iot_ids/processed/spo2.csv")
resp     = pd.read_csv("medical_iot_ids/processed/resp.csv")
temp     = pd.read_csv("medical_iot_ids/processed/temp.csv")

print(f"  FHR+TOCO : {len(fhr_toco):,} rows")
print(f"  SpO2     : {len(spo2):,} rows")
print(f"  RespRate : {len(resp):,} rows")
print(f"  Temp     : {len(temp):,} rows")

# Use min length so all sensors are aligned
min_len = min(len(fhr_toco), len(spo2), len(resp), len(temp))
print(f"\n  ✅ Using min_len = {min_len:,} rows (all sensors aligned)")

df = pd.concat([
    fhr_toco.iloc[:min_len].reset_index(drop=True),
    spo2.iloc[:min_len].reset_index(drop=True),
    resp.iloc[:min_len].reset_index(drop=True),
    temp.iloc[:min_len].reset_index(drop=True)
], axis=1)

df.columns = ['FHR', 'TOCO', 'SpO2', 'RespRate', 'Temp']

# Drop any remaining NaN rows
before = len(df)
df = df.dropna()
after  = len(df)

print(f"  Dropped {before - after} NaN rows")

df.to_csv("medical_iot_ids/processed/final_5sensor.csv", index=False)

print(f"\n{'='*50}")
print(f"  MERGE SUMMARY")
print(f"{'='*50}")
print(f"  Final rows   : {len(df):,}")
print(f"  Columns      : {list(df.columns)}")
print(f"{'='*50}")
print(df.describe())
print(f"\n✅ Saved: medical_iot_ids/processed/final_5sensor.csv")