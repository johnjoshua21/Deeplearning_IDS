import vitaldb
import pandas as pd
import numpy as np

TARGET_ROWS  = 20000
MAX_CASES    = 200
TRACK        = 'SPO2'

print(f"Loading SpO2 from VitalDB (target: {TARGET_ROWS:,} rows)...\n")

all_data     = []
loaded_cases = []

for case_id in range(1, MAX_CASES + 1):
    try:
        data = vitaldb.load_case(case_id, [TRACK])
        if np.all(np.isnan(data)):
            continue

        df = pd.DataFrame(data, columns=['SpO2'])
        df['SpO2'] = df['SpO2'].interpolate(method='linear', limit_direction='both')
        df['SpO2'] = df['SpO2'].ffill().bfill()
        df = df.dropna()

        if len(df) < 100:
            continue

        all_data.append(df)
        loaded_cases.append(case_id)
        total = sum(len(d) for d in all_data)
        print(f"  ✅ Case {case_id:3d} → {len(df):6,} rows | Total: {total:,}")

        if total >= TARGET_ROWS:
            print(f"\n🎯 Target {TARGET_ROWS:,} reached!")
            break
    except:
        continue

df_final = pd.concat(all_data, ignore_index=True).iloc[:TARGET_ROWS]

print(f"\n{'='*50}")
print(f"  SpO2 FINAL ROWS : {len(df_final):,}")
print(f"  Cases used      : {loaded_cases}")
print(f"  Mean            : {df_final['SpO2'].mean():.2f}")
print(f"  Min / Max       : {df_final['SpO2'].min():.2f} / {df_final['SpO2'].max():.2f}")
print(f"{'='*50}")

df_final.to_csv("medical_iot_ids/processed/spo2.csv", index=False)
print(f"\n✅ Saved: medical_iot_ids/processed/spo2.csv  ({len(df_final):,} rows)")