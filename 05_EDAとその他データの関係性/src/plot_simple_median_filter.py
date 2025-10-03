import os
from util.utils3 import load_h5_data, search_and_get_filenames
import matplotlib.pyplot as plt
import numpy as np
import neurokit2 as nk
from scipy import signal


def simple_median_filter(data, median_kernel_size):
    """
    メジアンフィルタ：
    波形中の任意のデータ点を、そのデータを含むウィンドウ内の値の中央値で置き換えるフィルタ。
    端点では特別な処理を行わず、カーネルの範囲内のデータの中央値を計算するのみ

    Args:
        data: EDAデータ
        median_kernel_size: カーネルサイズ（奇数）

    Returns:
        filtered_data: 急落&復帰パターンが除去されたデータ
    """
    filtered_data = data.copy()
    n_samples = len(data)

    for target in range(0, n_samples):
        mn = max(0, target - int(median_kernel_size / 2))
        mx = min(target + int(median_kernel_size / 2), n_samples - 1)
        arr = data[mn:mx + 1]
        median = np.median(arr)
        filtered_data[target] = median
    
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
    output_path = f"/home/mori/projects/affective-forecast/plots/EDAフィルタリング"

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

        if eda is not None:
            x = np.linspace(0, 15, eda.shape[1])
            y = np.asarray(eda[0], dtype=float)

            Fs = 4
            median_kernel_size = 5

            y_filtered = simple_median_filter(y, median_kernel_size)

            # y軸範囲を計算
            y_min, y_max = np.min(y), np.max(y)
            y_margin = (y_max - y_min) * 0.05

            os.makedirs(f"{output_path}/{name}", exist_ok=True)

            plt.figure(figsize=(12, 8))
            plt.plot(x, y, label='Original EDA', alpha=0.7)
            plt.plot(x, y_filtered, label=f'Simple Median Filtered (kernel size: {median_kernel_size})', linewidth=2)
            plt.xlabel("Time (minutes)")
            plt.ylabel(f"EDA (μS)")
            plt.ylim(y_min - y_margin, y_max + y_margin)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.title(f"EDA filtered by simple median filter")

            plt.tight_layout()
            plt.savefig(f"{output_path}/{name}/simple_median.svg")
            plt.savefig(f"{output_path}/{name}/simple_median.png")
            plt.close()
