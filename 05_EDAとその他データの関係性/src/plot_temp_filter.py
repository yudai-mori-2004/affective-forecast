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
            plt.title(f"EDA filtered by temperature filter (range: {temp_min}-{temp_max}℃)")

            plt.tight_layout()
            plt.savefig(f"{output_path}/{name}/temperature.svg")
            plt.savefig(f"{output_path}/{name}/temperature.png")
            plt.close()
