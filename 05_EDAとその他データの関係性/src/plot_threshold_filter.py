import os
from util.utils3 import load_h5_data, search_and_get_filenames
import matplotlib.pyplot as plt
import numpy as np
import neurokit2 as nk
from scipy import signal


def threshold_filter(data, threshold_min, threshold_max):
    """
    閾値フィルタ：
    波形中のデータ点の閾値を 0.05~60μS に定め、これを逸脱するデータ点は除去する

    Args:
        data: EDAデータ
        threshold_min: EDAの閾値(最小)
        threshold_max: EDAの閾値(最大)

    Returns:
        filtered_data: 
    """
    filtered_data = data.copy()
    n_samples = len(data)

    for target in range(0, n_samples):
        if not (threshold_min <= data[target] < threshold_max):
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

            thr_min = 0.05
            thr_max = 60

            # ローパスフィルタ適用
            b, a = signal.butter(4, 0.35, btype='low')
            y_lowpass = signal.filtfilt(b, a, y)

            # 閾値フィルタ適用
            y_filtered = threshold_filter(y_lowpass, thr_min, thr_max)

            # y軸範囲を計算（オリジナルデータ基準）
            y_min, y_max = np.min(y), np.max(y)
            y_margin = (y_max - y_min) * 0.05

            os.makedirs(f"{output_path}/{name}", exist_ok=True)

            plt.figure(figsize=(12, 8))

            # 除去された部分（NaN）のマスクを取得
            removed_mask = np.isnan(y_filtered)

            # 全体を青でプロット（連続線）
            plt.plot(x, y_lowpass, color='blue', label='Kept', linewidth=1, zorder=1)

            # 除去された点の前後の線分を赤でプロット
            removed_indices = np.where(removed_mask)[0]
            red_label_added = False
            for idx in removed_indices:
                label = 'Removed (drop)' if not red_label_added else None
                # 前の点から除去点までの線分
                if idx > 0:
                    plt.plot(x[idx-1:idx+1], y_lowpass[idx-1:idx+1], color='red', linewidth=1, zorder=2, label=label)
                    red_label_added = True
                # 除去点から次の点までの線分
                if idx < len(y_lowpass) - 1:
                    label = None  # 2つ目以降はラベルなし
                    plt.plot(x[idx:idx+2], y_lowpass[idx:idx+2], color='red', linewidth=1, zorder=2, label=label)

            plt.xlabel("Time (minutes)")
            plt.ylabel(f"EDA (μS)")
            plt.ylim(y_min - y_margin, y_max + y_margin)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.title(f"EDA filtered by threshold filter (min: {thr_min}, max: {thr_max})")

            plt.tight_layout()
            plt.savefig(f"{output_path}/{name}/threshold.svg")
            plt.close()
