import os
from util.utils3 import load_h5_data, search_and_get_filenames
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


def temperature_filter(eda_data, temp_data, temp_min, temp_max):
    """
    皮膚温度フィルタ：
    皮膚温度が30~40℃の範囲を逸脱したなら、その区間のデータ点を無効化する。

    Args:
        eda_data: EDAデータ
        temp_data: 皮膚温度データ
        temp_min: 温度下限（30℃が規定値）
        temp_max: 温度上限（40℃が規定値）

    Returns:
        filtered_data: フィルタ済みEDAデータ
    """
    filtered_data = eda_data.copy()
    n_samples = len(eda_data)

    for i in range(n_samples):
        # 皮膚温度が範囲外の場合、そのEDAデータ点を無効化
        if not (temp_min <= temp_data[i] <= temp_max):
            filtered_data[i] = np.nan

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
        temp_file_name = f"{name}_temp.h5"
        eda = load_h5_data(f"{data_path}/{eda_file_name}")
        temp = load_h5_data(f"{data_path}/{temp_file_name}")

        if eda is not None and temp is not None:
            x = np.linspace(0, 15, eda.shape[1])
            y = np.asarray(eda[0], dtype=float)
            temp_data = np.asarray(temp[0], dtype=float)

            temp_min = 30  # ℃
            temp_max = 40  # ℃

            # ローパスフィルタ適用
            b, a = signal.butter(4, 0.35, btype='low')
            y_lowpass = signal.filtfilt(b, a, y)

            # 皮膚温度フィルタ適用
            y_filtered = temperature_filter(y_lowpass, temp_data, temp_min, temp_max)

            # y軸範囲を計算（オリジナルデータ基準）
            y_min, y_max = np.min(y), np.max(y)
            y_margin = (y_max - y_min) * 0.05

            os.makedirs(f"{output_path}/{name}", exist_ok=True)

            plt.figure(figsize=(12, 8))

            removed_mask = np.isnan(y_filtered)
            removed_indices = np.where(removed_mask)[0]

            for j, idx in enumerate(removed_indices):
                prev_rem = removed_indices[j - 1] if j > 0 else -1
                rem_start = idx - 1 if idx > 0 else -1
                rem_end = idx + 2 if idx + 1 < len(y_lowpass) else -1

                blue_label = 'Kept' if j == 0 else None
                red_label = 'Removed (drop)' if j == 0 else None

                plt.plot(x[prev_rem + 1:idx], y_lowpass[prev_rem + 1:idx],
                        color='blue', label=blue_label, linewidth=0.5)

                if rem_start >= 0:
                    plt.plot(x[rem_start:idx + 1], y_lowpass[rem_start:idx + 1],
                            color='red', linewidth=0.5, label=red_label)

                if rem_end > 0:
                    plt.plot(x[idx:rem_end], y_lowpass[idx:rem_end],
                            color='red', linewidth=0.5)

            last_idx = removed_indices[-1] if len(removed_indices) > 0 else -1
            last_blue_label = 'Kept' if len(removed_indices) == 0 else None
            plt.plot(x[last_idx + 1:], y_lowpass[last_idx + 1:],
                    color='blue', label=last_blue_label, linewidth=0.5)

            plt.xlabel("Time (minutes)")
            plt.ylabel(f"EDA (μS)")
            plt.ylim(y_min - y_margin, y_max + y_margin)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.title(f"EDA filtered by temperature filter (range: {temp_min}-{temp_max}℃)")

            plt.tight_layout()
            plt.savefig(f"{output_path}/{name}/temperature.svg")
            plt.savefig(f"{output_path}/{name}/temperature.png")
            plt.close()
