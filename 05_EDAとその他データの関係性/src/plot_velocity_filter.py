import os
from util.utils3 import load_h5_data, search_and_get_filenames
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


def velocity_filter(data, velocity_min, velocity_max, fs):
    """
    変化速度閾値フィルタ：
    隣接するEDAデータ点の値の変化速度が±10μS/secをこえたなら、その両端のデータ点を無効化する。

    Args:
        data: EDAデータ
        velocity_min: EDA変化速度の下限(-10μS/secが規定値)
        velocity_max: EDA変化速度の上限(10μS/secが規定値)
        fs: サンプリング周波数 (Hz)

    Returns:
        filtered_data:
    """
    filtered_data = data.copy()
    n_samples = len(data)
    dt = 1.0 / fs  # サンプル間の時間間隔

    for i in range(n_samples - 1):
        # 隣接する点の変化速度を計算
        velocity = (data[i + 1] - data[i]) / dt

        # 変化速度が閾値を超えた場合、両端を無効化
        if velocity < velocity_min or velocity > velocity_max:
            filtered_data[i] = np.nan
            filtered_data[i + 1] = np.nan

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
            vel_min = -10  # μS/sec
            vel_max = 10   # μS/sec

            # ローパスフィルタ適用
            b, a = signal.butter(4, 0.35, btype='low')
            y_lowpass = signal.filtfilt(b, a, y)

            # 変化速度フィルタ適用
            y_filtered = velocity_filter(y_lowpass, vel_min, vel_max, Fs)

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
            plt.title(f"EDA filtered by velocity filter (min: {vel_min}, max: {vel_max} μS/sec)")

            plt.tight_layout()
            plt.savefig(f"{output_path}/{name}/velocity.svg")
            plt.close()
