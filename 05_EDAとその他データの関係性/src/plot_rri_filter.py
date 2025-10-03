import os
from util.utils3 import load_h5_data, search_and_get_filenames
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


def rri_filter(eda_data, rri_data, delta, min_samples):
    """
    RRI（心拍）フィルタ：
    心拍数データの差分の差分の値がΔ以下である区間が一定サンプル数以上続く場合、
    その区間とその前後のマージンのEDAデータ点を無効化する。

    Args:
        eda_data: EDAデータ
        rri_data: 心拍数データ
        delta: 差分の差分の閾値
        min_samples: 除去対象とする最小連続サンプル数

    Returns:
        filtered_data: フィルタ済みEDAデータ
    """
    filtered_data = eda_data.copy()
    n_samples = len(eda_data)

    # 心拍数データの差分の差分を計算
    rri_diff = np.diff(np.diff(rri_data))

    # 差分の差分の値がΔ以下のマスクを作成
    low_variation_mask = np.abs(rri_diff) <= delta

    # 除去マスクを元のサイズで作成
    remove_mask = np.zeros(n_samples, dtype=bool)

    # 連続するTrue区間を検出し、min_samples以上の区間のみ除去
    i = 0
    while i < len(low_variation_mask):
        if low_variation_mask[i]:
            # 連続区間の開始
            start = i
            while i < len(low_variation_mask) and low_variation_mask[i]:
                i += 1
            # 連続区間の終了（iは次のFalseまたは終端）
            end = i

            # 連続区間の長さをチェック
            if end - start >= min_samples:
                # 2階差分のインデックスstartは、元データのstart, start+1, start+2に対応
                # 2階差分のインデックスend-1は、元データのend-1, end, end+1に対応
                for j in range(start, min(end + 2, n_samples)):
                    remove_mask[j] = True
        else:
            i += 1

    # マスクに基づいてEDAデータを無効化
    filtered_data[remove_mask] = np.nan

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
        rri_file_name = f"{name}_rri.h5"
        eda = load_h5_data(f"{data_path}/{eda_file_name}")
        rri = load_h5_data(f"{data_path}/{rri_file_name}")

        if eda is not None and rri is not None:
            x = np.linspace(0, 15, eda.shape[1])
            y = np.asarray(eda[0], dtype=float)
            rri_data = np.asarray(rri[0], dtype=float)

            Fs = 4
            delta = 0.0001  # 心拍数差分の差分の閾値
            min_duration_seconds = 3.0  # 除去対象とする最小連続時間（秒）
            min_samples = int(min_duration_seconds * Fs)  # サンプル数に変換

            # ローパスフィルタ適用
            b, a = signal.butter(4, 0.35, btype='low')
            y_lowpass = signal.filtfilt(b, a, y)

            # RRIフィルタ適用
            y_filtered = rri_filter(y_lowpass, rri_data, delta, min_samples)

            # y軸範囲を計算（オリジナルデータ基準）
            y_min, y_max = np.min(y), np.max(y)
            y_margin = (y_max - y_min) * 0.05

            os.makedirs(f"{output_path}/{name}", exist_ok=True)

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16))

            # 除去された部分（NaN）のマスクを取得
            removed_mask = np.isnan(y_filtered)

            # 上段: EDAデータのプロット
            # 全体を青でプロット（連続線）
            ax1.plot(x, y_lowpass, color='blue', label='Kept', linewidth=1, zorder=1)

            # 除去された点の前後の線分を赤でプロット
            removed_indices = np.where(removed_mask)[0]
            red_label_added = False
            for idx in removed_indices:
                label = 'Removed (drop)' if not red_label_added else None
                # 前の点から除去点までの線分
                if idx > 0:
                    ax1.plot(x[idx-1:idx+1], y_lowpass[idx-1:idx+1], color='red', linewidth=1, zorder=2, label=label)
                    red_label_added = True
                # 除去点から次の点までの線分
                if idx < len(y_lowpass) - 1:
                    label = None  # 2つ目以降はラベルなし
                    ax1.plot(x[idx:idx+2], y_lowpass[idx:idx+2], color='red', linewidth=1, zorder=2, label=label)

            ax1.set_xlabel("Time (minutes)")
            ax1.set_ylabel(f"EDA (μS)")
            ax1.set_ylim(y_min - y_margin, y_max + y_margin)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_title(f"EDA filtered by RRI filter (delta: {delta})")

            # 下段: RRIデータのプロット
            x_rri = np.linspace(0, 15, len(rri_data))
            ax2.plot(x_rri, rri_data, color='green', linewidth=1)
            ax2.set_xlabel("Time (minutes)")
            ax2.set_ylabel("RRI (ms)")
            ax2.grid(True, alpha=0.3)
            ax2.set_title("RRI Signal")

            plt.tight_layout()
            plt.savefig(f"{output_path}/{name}/rri.svg")
            plt.savefig(f"{output_path}/{name}/rri.png")
            plt.close()
