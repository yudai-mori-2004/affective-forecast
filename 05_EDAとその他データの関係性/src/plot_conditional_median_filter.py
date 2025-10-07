import os
from util.utils3 import load_h5_data, search_and_get_filenames
import matplotlib.pyplot as plt
import numpy as np
import neurokit2 as nk
from scipy import signal



def conditional_median_filter(data, median_kernel_size, p):
    """
    条件付きメジアンフィルタ：
    波形中の任意のデータ点が、そのデータを含むウィンドウ内の値の中央値から -p*100% 以上変動している場合、その値をNaNに置き換える。
    端点では特別な処理を行わず、カーネルの範囲内のデータの中央値を計算するのみ

    Args:
        data: EDAデータ
        median_kernel_size: カーネルサイズ（奇数）
        p: 閾値変動率

    Returns:
        filtered_data: 
    """
    filtered_data = data.copy()
    n_samples = len(data)

    for target in range(0, n_samples):
        mn = max(0, target - int(median_kernel_size / 2))
        mx = min(target + int(median_kernel_size / 2), n_samples - 1)
        arr = data[mn:mx + 1]
        median = np.median(arr)
        if median > 0 and np.abs((data[target] / median) - 1.0) > p:
            filtered_data[target] = np.nan

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
            median_kernel_size = 4 * Fs + 1
            p = 0.2

            y_filtered = conditional_median_filter(y, median_kernel_size, p)

            # ±p%ゾーンの可視化のために中央値の配列を計算
            median_values = np.zeros_like(y)
            for target in range(len(y)):
                mn = max(0, target - int(median_kernel_size / 2))
                mx = min(target + int(median_kernel_size / 2), len(y) - 1)
                arr = y[mn:mx + 1]
                median_values[target] = np.median(arr)

            upper_bound = median_values * (1 + p)
            lower_bound = median_values * (1 - p)

            # y軸範囲を計算
            y_min, y_max = np.min(y), np.max(y)
            y_margin = (y_max - y_min) * 0.05

            os.makedirs(f"{output_path}/{name}", exist_ok=True)

            plt.figure(figsize=(12, 8))

            removed_mask = np.isnan(y_filtered)
            removed_indices = np.where(removed_mask)[0]

            for j, idx in enumerate(removed_indices):
                prev_rem = removed_indices[j - 1] if j > 0 else -1
                rem_start = idx - 1 if idx > 0 else -1
                rem_end = idx + 2 if idx + 1 < len(y) else -1

                blue_label = 'Kept' if j == 0 else None
                red_label = 'Removed (drop)' if j == 0 else None

                plt.plot(x[prev_rem + 1:idx], y[prev_rem + 1:idx],
                        color='blue', label=blue_label, linewidth=0.5)

                if rem_start >= 0:
                    plt.plot(x[rem_start:idx + 1], y[rem_start:idx + 1],
                            color='red', linewidth=0.5, label=red_label)

                if rem_end > 0:
                    plt.plot(x[idx:rem_end], y[idx:rem_end],
                            color='red', linewidth=0.5)

            last_idx = removed_indices[-1] if len(removed_indices) > 0 else -1
            last_blue_label = 'Kept' if len(removed_indices) == 0 else None
            plt.plot(x[last_idx + 1:], y[last_idx + 1:],
                    color='blue', label=last_blue_label, linewidth=0.5)

            # ±p%ゾーンのラインをプロット
            plt.plot(x, upper_bound, color='gray', linestyle='--', linewidth=0.5, alpha=0.7, label=f'+{int(p*100)}% zone')
            plt.plot(x, lower_bound, color='gray', linestyle='--', linewidth=0.5, alpha=0.7, label=f'-{int(p*100)}% zone')

            plt.xlabel("Time (minutes)")
            plt.ylabel(f"EDA (μS)")
            plt.ylim(y_min - y_margin, y_max + y_margin)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.title(f"EDA filtered by conditional median filter (kernel size: {median_kernel_size}, p: {p})")

            plt.tight_layout()
            plt.savefig(f"{output_path}/{name}/conditional_median.svg")
            plt.close()
