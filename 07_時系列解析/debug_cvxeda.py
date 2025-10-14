"""
cvxEDAの問題をデバッグ
"""
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from scipy import signal
import neurokit2 as nk
from src.util.utils3 import load_h5_data, search_by_conditions, get_affective_datas_from_uuids
import time

# データを1つ取得
data_path = "/home/mori/projects/affective-forecast/datas/biometric_data"
uuids = search_by_conditions({"ex-term": "term1"})
datas = get_affective_datas_from_uuids(uuids)
name = datas[0]["filename"]

print(f"Testing with: {name}")

# EDAデータ読み込み
eda_h5 = load_h5_data(f"{data_path}/{name}_eda.h5")
eda_original = np.asarray(eda_h5[0], dtype=np.float32)

print(f"Original EDA shape: {eda_original.shape}, dtype: {eda_original.dtype}")
print(f"Original EDA length: {len(eda_original)}")
print(f"Sample values: {eda_original[:10]}")

# ローパスフィルタ
b, a = signal.butter(4, 0.35, btype='low', fs=4.0)
eda_lowpass = signal.filtfilt(b, a, eda_original)

print(f"\nLowpass EDA shape: {eda_lowpass.shape}, dtype: {eda_lowpass.dtype}")
print(f"Lowpass EDA contiguous: {eda_lowpass.flags['C_CONTIGUOUS']}")

# 3600サンプルに切り取り
eda_short = eda_lowpass[:3600]
print(f"\nShort EDA shape: {eda_short.shape}, dtype: {eda_short.dtype}")

# float32でテスト
print("\n" + "="*60)
print("Testing with float32...")
print("="*60)
eda_f32 = np.ascontiguousarray(eda_short, dtype=np.float32)
start = time.time()
try:
    signals_f32 = nk.eda_phasic(eda_f32, sampling_rate=4.0, method="cvxeda")
    elapsed = time.time() - start
    print(f"SUCCESS! Time: {elapsed:.2f}s")
except Exception as e:
    elapsed = time.time() - start
    print(f"FAILED! Time: {elapsed:.2f}s")
    print(f"Error: {e}")

# float64でテスト
print("\n" + "="*60)
print("Testing with float64...")
print("="*60)
eda_f64 = np.ascontiguousarray(eda_short, dtype=np.float64)
start = time.time()
try:
    signals_f64 = nk.eda_phasic(eda_f64, sampling_rate=4.0, method="cvxeda")
    elapsed = time.time() - start
    print(f"SUCCESS! Time: {elapsed:.2f}s")
except Exception as e:
    elapsed = time.time() - start
    print(f"FAILED! Time: {elapsed:.2f}s")
    print(f"Error: {e}")

print("\n" + "="*60)
print("Conclusion:")
print("="*60)
