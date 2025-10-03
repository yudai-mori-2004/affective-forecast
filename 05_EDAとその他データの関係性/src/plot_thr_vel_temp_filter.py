import os
from util.utils3 import load_h5_data, search_and_get_filenames
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


def threshold_filter(data, threshold_min, threshold_max):
    """閾値フィルタ"""
    mask = np.ones(len(data), dtype=bool)
    for i in range(len(data)):
        if not (threshold_min <= data[i] < threshold_max):
            mask[i] = False
    return mask


def velocity_filter(data, velocity_min, velocity_max, fs):
    """変化速度フィルタ"""
    mask = np.ones(len(data), dtype=bool)
    dt = 1.0 / fs

    for i in range(len(data) - 1):
        velocity = (data[i + 1] - data[i]) / dt
        if velocity < velocity_min or velocity > velocity_max:
            mask[i] = False
            mask[i + 1] = False

    return mask


def temperature_filter(temp_data, temp_min, temp_max):
    """温度フィルタ"""
    mask = np.ones(len(temp_data), dtype=bool)
    for i in range(len(temp_data)):
        if not (temp_min <= temp_data[i] <= temp_max):
            mask[i] = False
    return mask


def apply_margin(removed_mask, margin_seconds, fs):
    """除去点の前後にマージンを追加"""
    margin_mask = np.zeros(len(removed_mask), dtype=bool)
    margin_samples = int(margin_seconds * fs)

    removed_indices = np.where(removed_mask)[0]

    for idx in removed_indices:
        start = max(0, idx - margin_samples)
        end = min(len(removed_mask), idx + margin_samples + 1)
        margin_mask[start:end] = True

    return margin_mask


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

            Fs = 4
            thr_min = 0.05
            thr_max = 60
            vel_min = -10  # μS/sec
            vel_max = 10   # μS/sec
            temp_min = 30  # ℃
            temp_max = 40  # ℃
            margin_seconds = 5

            # ローパスフィルタ適用
            b, a = signal.butter(4, 0.35, btype='low')
            y_lowpass = signal.filtfilt(b, a, y)

            # 各フィルタを適用してマスクを取得
            thr_mask = threshold_filter(y_lowpass, thr_min, thr_max)
            vel_mask = velocity_filter(y_lowpass, vel_min, vel_max, Fs)
            temp_mask = temperature_filter(temp_data, temp_min, temp_max)

            # 全フィルタの統合（ANDを取る）
            combined_mask = thr_mask & vel_mask & temp_mask

            # 除去されたポイント（Falseの部分）
            core_removed_mask = ~combined_mask

            # マージンマスクを適用
            full_margin_mask = apply_margin(core_removed_mask, margin_seconds, Fs)

            # マージンのみで除去された部分（コアでは除去されていないがマージンで除去）
            margin_only_mask = full_margin_mask & ~core_removed_mask

            # y軸範囲を計算（オリジナルデータ基準）
            y_min, y_max = np.min(y), np.max(y)
            y_margin = (y_max - y_min) * 0.05

            os.makedirs(f"{output_path}/{name}", exist_ok=True)

            plt.figure(figsize=(12, 8))

            # 全体を青でプロット（連続線）
            plt.plot(x, y_lowpass, color='blue', label='Kept', linewidth=1, zorder=1)

            # コアフィルタで除去された点の前後の線分を赤でプロット
            core_indices = np.where(core_removed_mask)[0]
            red_label_added = False
            for idx in core_indices:
                label = 'Removed (core)' if not red_label_added else None
                if idx > 0:
                    plt.plot(x[idx-1:idx+1], y_lowpass[idx-1:idx+1], color='red', linewidth=1, zorder=3, label=label)
                    red_label_added = True
                if idx < len(y_lowpass) - 1:
                    plt.plot(x[idx:idx+2], y_lowpass[idx:idx+2], color='red', linewidth=1, zorder=3, label=None)

            # マージンのみで除去された点の前後の線分をオレンジでプロット
            margin_indices = np.where(margin_only_mask)[0]
            orange_label_added = False
            for idx in margin_indices:
                label = 'Removed (margin)' if not orange_label_added else None
                if idx > 0:
                    plt.plot(x[idx-1:idx+1], y_lowpass[idx-1:idx+1], color='orange', linewidth=1, zorder=2, label=label)
                    orange_label_added = True
                if idx < len(y_lowpass) - 1:
                    plt.plot(x[idx:idx+2], y_lowpass[idx:idx+2], color='orange', linewidth=1, zorder=2, label=None)

            plt.xlabel("Time (minutes)")
            plt.ylabel(f"EDA (μS)")
            plt.ylim(y_min - y_margin, y_max + y_margin)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.title(f"EDA filtered by threshold + velocity + temperature (margin: {margin_seconds}s)")

            plt.tight_layout()
            plt.savefig(f"{output_path}/{name}/thr_vel_temp.svg")
            plt.savefig(f"{output_path}/{name}/thr_vel_temp.png")
            plt.close()
