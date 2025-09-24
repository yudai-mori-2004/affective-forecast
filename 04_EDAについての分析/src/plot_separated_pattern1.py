import os
from util.utils3 import load_h5_data, search_and_get_filenames
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

# パターン１：閾値 + バンドパスでSCRの直接抽出

def threshold_filter(data, lower_threshold, upper_threshold):
    """閾値フィルタリング：閾値を超えるデータ点を除去し線形補間"""
    filtered_data = data.copy()
    mask = (data < lower_threshold) | (data > upper_threshold)

    # 閾値外のポイントをNaNでマーク
    filtered_data[mask] = np.nan

    # 線形補間
    valid_indices = ~np.isnan(filtered_data)
    if np.sum(valid_indices) < 2:
        return data  # 補間できない場合は元データを返す

    x = np.arange(len(filtered_data))
    filtered_data = np.interp(x, x[valid_indices], filtered_data[valid_indices])

    return filtered_data

def bandpass_filter(data, lowcut, highcut, fs):
    """バンドパスフィルタ"""
    nyquist = fs / 2
    low = lowcut / nyquist
    high = highcut / nyquist

    if low <= 0:
        low = 0.01
    if high >= 1:
        high = 0.99

    b, a = signal.butter(4, [low, high], btype='band')
    filtered_data = signal.filtfilt(b, a, data)
    return filtered_data

if __name__ == "__main__":
    # パス定義
    data_path = f"/home/mori/projects/affective-forecast/datas/biometric_data"
    output_path = f"/home/mori/projects/affective-forecast/plots/EDA分離プロットパターン1"

    # フォルダが存在しない場合は作成
    os.makedirs(output_path, exist_ok=True)

    data_names = search_and_get_filenames({
        "ex-term": "term1"
    })

    # 先頭の10枚の画像のみを比較用のサンプルとする
    data_names.sort(key=lambda x: int(x.split("_")[1]))
    sample_data_names = data_names[:10]

    for i, name in enumerate(sample_data_names):
        eda_file_name = f"{name}_eda.h5"
        eda = load_h5_data(f"{data_path}/{eda_file_name}")

        if eda is not None and eda.shape[1] > 3000:
            x = np.linspace(0, 15, eda.shape[1])
            y = np.asarray(eda[0], dtype=float)

            Fs = 4

            # 1. 閾値フィルタリング (0.1-25 μS)
            y_threshold = threshold_filter(y, 0.1, 25)

            # 2. バンドパスフィルタ (0.05 ~ 0.25Hz)
            y_bandpass = bandpass_filter(y_threshold, 0.05, 0.25, Fs)

            os.makedirs(f"{output_path}/{name}", exist_ok=True)

            plt.figure(figsize=(12, 8))
            plt.plot(x, y, linewidth=0.5, alpha=0.7, label="Original EDA")
            plt.plot(x, y_bandpass, linewidth=0.5, label="SCR (Threshold + Bandpass)")
            plt.xlabel("Time (minutes)")
            plt.ylabel(f"EDA signals (μS)")

            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.title(f"Pattern 1: Threshold + Bandpass (0.1-25μS, {name})")

            plt.tight_layout()
            plt.savefig(f"{output_path}/{name}/pattern1.svg")
            plt.close()

            print(f"Saved Pattern 1 - {i} images")