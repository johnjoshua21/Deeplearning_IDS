import vitaldb
import pandas as pd
import numpy as np

TARGET_ROWS  = 20000
MAX_CASES    = 200
TRACK        = 'RR'          # confirmed working track from find_resp_case.py

print(f"Loading RespRate from VitalDB (target: {TARGET_ROWS:,} rows)...\n")

all_data     = []
loaded_cases = []

for case_id in range(1, MAX_CASES + 1):
    try:
        data = vitaldb.load_case(case_id, [TRACK])
        if np.all(np.isnan(data)):
            continue

        df = pd.DataFrame(data, columns=['RespRate'])
        df['RespRate'] = df['RespRate'].interpolate(method='linear', limit_direction='both')
        df['RespRate'] = df['RespRate'].ffill().bfill()
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

if not all_data:
    print("❌ No RespRate data found. Trying fallback tracks...")
    # Fallback tracks
    for track in ['RESP_RATE', 'ETCO2_RESP', 'RESP']:
        for case_id in range(1, MAX_CASES + 1):
            try:
                data = vitaldb.load_case(case_id, [track])
                if np.all(np.isnan(data)):
                    continue
                df = pd.DataFrame(data, columns=['RespRate'])
                df['RespRate'] = df['RespRate'].interpolate(method='linear', limit_direction='both')
                df['RespRate'] = df['RespRate'].ffill().bfill()
                df = df.dropna()
                if len(df) < 100:
                    continue
                all_data.append(df)
                loaded_cases.append(case_id)
                total = sum(len(d) for d in all_data)
                print(f"  ✅ [{track}] Case {case_id:3d} → {len(df):6,} rows | Total: {total:,}")
                if total >= TARGET_ROWS:
                    break
            except:
                continue
        if all_data:
            break

df_final = pd.concat(all_data, ignore_index=True).iloc[:TARGET_ROWS]

print(f"\n{'='*50}")
print(f"  RespRate FINAL ROWS : {len(df_final):,}")
print(f"  Cases used          : {loaded_cases}")
print(f"  Mean                : {df_final['RespRate'].mean():.2f}")
print(f"  Min / Max           : {df_final['RespRate'].min():.2f} / {df_final['RespRate'].max():.2f}")
print(f"{'='*50}")

df_final.to_csv("medical_iot_ids/processed/resp.csv", index=False)
print(f"\n✅ Saved: medical_iot_ids/processed/resp.csv  ({len(df_final):,} rows)")