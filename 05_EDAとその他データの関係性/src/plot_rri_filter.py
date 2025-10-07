import os
from util.utils3 import load_h5_data, search_and_get_filenames
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


def rri_filter(eda_data, rri_data, threshold, min_samples, lowpass_cutoff):
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

    # 心拍数データをスムージングして、差分の差分を計算
    b, a = signal.butter(4, lowpass_cutoff, btype='low')
    rri_data_smoothed = signal.filtfilt(b, a, rri_data)
    rri_data_smoothed_diff = np.diff(rri_data_smoothed)
    rri_data_smoothed_diff_diff = np.diff(rri_data_smoothed_diff)

    # rri_data_smoothed_diff_diffの値がΔ以下のマスクを作成
    low_variation_mask = np.abs(rri_data_smoothed_diff_diff) <= threshold

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

            Fs = 4
            threshold = 0.00015  # 心拍数差分の差分の閾値
            min_duration_seconds = 10.0  # 除去対象とする最小連続時間（秒）
            min_samples = int(min_duration_seconds * Fs)  # サンプル数に変換

            # EDAにローパスフィルタ適用
            b, a = signal.butter(4, 0.35, btype='low')
            y_lowpass = signal.filtfilt(b, a, y)

            # RRIデータの処理
            rri_lowpass_cutoff = 0.1
            rri_data = np.asarray(rri[0], dtype=float)
            rri_data_diff = np.diff(rri_data)
            rri_data_diff_diff = np.diff(rri_data_diff)

            b, a = signal.butter(4, rri_lowpass_cutoff, btype='low')
            rri_data_smoothed = signal.filtfilt(b, a, rri_data)
            rri_data_smoothed_diff = np.diff(rri_data_smoothed)
            rri_data_smoothed_diff_diff = np.diff(rri_data_smoothed_diff)

            # RRIフィルタ適用
            y_filtered = rri_filter(y_lowpass, rri_data, threshold, min_samples, rri_lowpass_cutoff)

            # y軸範囲を計算（オリジナルデータ基準）
            y_min, y_max = np.min(y), np.max(y)
            y_margin = (y_max - y_min) * 0.05

            os.makedirs(f"{output_path}/{name}", exist_ok=True)

            fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7) = plt.subplots(7, 1, figsize=(12, 8*7))

            # 除去された部分（NaN）のマスクを取得
            removed_mask = np.isnan(y_filtered)

            removed_indices = np.where(removed_mask)[0]

            for j, idx in enumerate(removed_indices):
                prev_rem = removed_indices[j - 1] if j > 0 else -1
                rem_start = idx - 1 if idx > 0 else -1
                rem_end = idx + 2 if idx + 1 < len(y_lowpass) else -1

                blue_label = 'Kept' if j == 0 else None
                red_label = 'Removed (drop)' if j == 0 else None

                ax1.plot(x[prev_rem + 1:idx], y_lowpass[prev_rem + 1:idx],
                        color='blue', label=blue_label, linewidth=0.5)

                if rem_start >= 0:
                    ax1.plot(x[rem_start:idx + 1], y_lowpass[rem_start:idx + 1],
                            color='red', linewidth=0.5, label=red_label)

                if rem_end > 0:
                    ax1.plot(x[idx:rem_end], y_lowpass[idx:rem_end],
                            color='red', linewidth=0.5)

            last_idx = removed_indices[-1] if len(removed_indices) > 0 else -1
            last_blue_label = 'Kept' if len(removed_indices) == 0 else None
            ax1.plot(x[last_idx + 1:], y_lowpass[last_idx + 1:],
                    color='blue', label=last_blue_label, linewidth=0.5)
            
            ax1.set_xlabel("Time (minutes)")
            ax1.set_ylabel(f"EDA (μS)")
            ax1.set_ylim(y_min - y_margin, y_max + y_margin)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_title(f"EDA filtered by RRI filter (delta: {threshold})")

            # 下段: RRIデータのプロット
            x_rri = np.linspace(0, 15, len(rri_data))
            ax2.plot(x_rri, rri_data, color='green', linewidth=1)
            ax2.set_xlabel("Time (minutes)")
            ax2.set_ylabel("RRI (ms)")
            ax2.grid(True, alpha=0.3)
            ax2.set_title("RRI Signal")

            x_rri_diff = np.linspace(0, 15, len(rri_data)-1)
            ax3.plot(x_rri_diff, rri_data_diff, color='green', linewidth=1)
            ax3.set_xlabel("Time (minutes)")
            ax3.set_ylabel("RRI (ms)")
            ax3.grid(True, alpha=0.3)
            ax3.set_title("RRI Signal Diff")

            x_rri_diff_diff = np.linspace(0, 15, len(rri_data)-2)
            ax4.plot(x_rri_diff_diff, rri_data_diff_diff, color='green', linewidth=1)
            ax4.hlines([-threshold, threshold], 0, 15, colors='blue', linestyles='dashed')
            ax4.set_xlabel("Time (minutes)")
            ax4.set_ylabel("RRI (ms)")
            ax4.grid(True, alpha=0.3)
            ax4.set_title(f"RRI Signal Diff of Diff (threshold: {threshold})")

            ax5.plot(x_rri, rri_data_smoothed, color='purple', linewidth=1)
            ax5.set_xlabel("Time (minutes)")
            ax5.set_ylabel("RRI (ms)")
            ax5.grid(True, alpha=0.3)
            ax5.set_title(f"RRI Signal Lowpassed (lowpass cutoff: {rri_lowpass_cutoff})")

            ax6.plot(x_rri_diff, rri_data_smoothed_diff, color='purple', linewidth=1)
            ax6.set_xlabel("Time (minutes)")
            ax6.set_ylabel("RRI (ms)")
            ax6.grid(True, alpha=0.3)
            ax6.set_title(f"RRI Signal Lowpassed Diff (lowpass cutoff: {rri_lowpass_cutoff})")

            ax7.plot(x_rri_diff_diff, rri_data_smoothed_diff_diff, color='purple', linewidth=1)
            ax7.hlines([-threshold, threshold], 0, 15, colors='blue', linestyles='dashed')
            ax7.set_xlabel("Time (minutes)")
            ax7.set_ylabel("RRI (ms)")
            ax7.grid(True, alpha=0.3)
            ax7.set_title(f"RRI Signal Lowpassed Diff of Diff (lowpass cutoff: {rri_lowpass_cutoff}, threshold: {threshold})")

            plt.tight_layout()
            plt.savefig(f"{output_path}/{name}/rri.svg")
            plt.close()
