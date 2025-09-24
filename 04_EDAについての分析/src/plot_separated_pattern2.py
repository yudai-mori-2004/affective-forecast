import os
from util.utils3 import load_h5_data, search_and_get_filenames
import matplotlib.pyplot as plt
import numpy as np
import neurokit2 as nk
from scipy import signal

# パターン２：閾値 + ローパス + cvxEDAでSCRの抽出

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

def lowpass_filter(data, cutoff, fs):
    """ローパスフィルタ"""
    nyquist = fs / 2
    normalized_cutoff = cutoff / nyquist

    if normalized_cutoff >= 1:
        normalized_cutoff = 0.99

    b, a = signal.butter(4, normalized_cutoff, btype='low')
    filtered_data = signal.filtfilt(b, a, data)
    return filtered_data

if __name__ == "__main__":
    # パス定義
    data_path = f"/home/mori/projects/affective-forecast/datas/biometric_data"
    output_path = f"/home/mori/projects/affective-forecast/plots/EDA分離プロットパターン2"

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

            # 2. ローパスフィルタ (0.25Hz以上をカット)
            y_lowpass = lowpass_filter(y_threshold, 0.25, Fs)

            # 3. cvxEDAで分離
            signals = nk.eda_phasic(y_lowpass, sampling_rate=Fs, method="cvxeda")
            tonic = signals['EDA_Tonic'].to_numpy()
            phasic = signals['EDA_Phasic'].to_numpy()

            os.makedirs(f"{output_path}/{name}", exist_ok=True)

            plt.figure(figsize=(12, 8))
            plt.plot(x, y_lowpass, linewidth=0.5, linestyle='--', label="Filtered EDA (Threshold + Lowpass)")
            plt.plot(x, tonic, linewidth=0.5, label="SCL (Tonic)")
            plt.plot(x, phasic, linewidth=0.5, label="SCR (Phasic)")
            plt.xlabel("Time (minutes)")
            plt.ylabel(f"EDA signals (μS)")

            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.title(f"Pattern 2: Threshold + Lowpass + cvxEDA (0.1-25μS, {name})")

            plt.tight_layout()
            plt.savefig(f"{output_path}/{name}/pattern2.svg")
            plt.close()

            print(f"Saved Pattern 2 - {i} images")