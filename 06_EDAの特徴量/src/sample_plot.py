import os
from util.utils3 import load_h5_data, search_and_get_filenames, get_affective_datas_from_uuids, search_by_conditions
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.ndimage import median_filter


if __name__ == "__main__":
    data_path = f"/home/mori/projects/affective-forecast/datas/biometric_data"
    output_path = f"/home/mori/projects/affective-forecast/plots/EDAフィルタリングサンプル"
    os.makedirs(output_path, exist_ok=True)

    uuids = search_by_conditions({
        "ex-term": "term1"
    })
    datas = get_affective_datas_from_uuids(uuids)

    datas.sort(key=lambda x: int(x["filename"].split("_")[1]))
    data_names = [d["filename"] for d in datas]

    # 最初の10個のみ
    sample_data_names = data_names[:10]

    Fs = 4.0

    # パターン
    median_windows = [1, 5, 17, 33, 61]
    median_thresholds = [0.05, 0.15, 0.30, 0.45, 0.60]
    crop_samples = [int(20*Fs), int(40*Fs), int(60*Fs), int(120*Fs), int(180*Fs), int(300*Fs), int(600*Fs), 3599]

    for i, name in enumerate(sample_data_names):
        print(f"Processing {i+1}/{len(sample_data_names)}: {name}")
        eda_file_name = f"{name}_eda.h5"
        temp_file_name = f"{name}_temp.h5"
        eda_h5 = load_h5_data(f"{data_path}/{eda_file_name}")
        temp_h5 = load_h5_data(f"{data_path}/{temp_file_name}")

        if eda_h5 is not None and temp_h5 is not None and eda_h5.shape[1] >= 3599 and eda_h5.shape[1] == temp_h5.shape[1]:
            eda = np.asarray(eda_h5[0], dtype=float)
            temp = np.asarray(temp_h5[0], dtype=float)

            # threshold
            thr_mask = (eda < 0.05) | (eda > 60)

            # lowpass
            b, a = signal.butter(4, 0.35, btype='low', fs=Fs)
            eda_lowpass = signal.filtfilt(b, a, eda)

            # velocity
            dt = 1.0 / Fs
            v = np.diff(eda) / dt
            vel_mask = (v < -10.0) | (v > 10.0)
            vel_mask = np.concatenate([[False], vel_mask]) | np.concatenate([vel_mask, [False]])

            # temperature
            temp_mask = (temp < 30) | (temp > 40)

            # データごとのフォルダ作成
            os.makedirs(f"{output_path}/{name}", exist_ok=True)

            for width in median_windows:
                for thres in median_thresholds:
                    # median
                    med = median_filter(eda, size=width, mode='nearest')
                    med_mask = np.abs((eda / med) - 1.0) > thres

                    length = len(eda_lowpass)
                    for sample in crop_samples:
                        # 解析ユニットのデータ
                        end_idx = length
                        start_idx = max(0, length - sample)
                        wave_unit = eda_lowpass[start_idx:end_idx]

                        # マスクの適用（各マスクを解析範囲に合わせて切り出し）
                        thr_mask_unit = thr_mask[start_idx:end_idx]
                        med_mask_unit = med_mask[start_idx:end_idx]
                        vel_mask_unit = vel_mask[start_idx:end_idx]
                        temp_mask_unit = temp_mask[start_idx:end_idx]

                        # 全マスクを統合
                        combined_mask = thr_mask_unit | med_mask_unit | vel_mask_unit | temp_mask_unit

                        # マスクを適用してフィルタリング済みデータを作成
                        filtered_wave = wave_unit.copy()
                        filtered_wave[combined_mask] = np.nan

                        # 時間軸（解析範囲の時間）
                        x_unit = np.linspace(0, len(wave_unit) / Fs / 60, len(wave_unit))  # 分単位

                        # 除去率の計算
                        total_samples = len(wave_unit)
                        nan_ratio = np.sum(combined_mask) / total_samples if total_samples > 0 else 1.0
                        thr_nan_ratio = np.sum(thr_mask_unit) / total_samples if total_samples > 0 else 1.0
                        med_nan_ratio = np.sum(med_mask_unit) / total_samples if total_samples > 0 else 1.0
                        vel_nan_ratio = np.sum(vel_mask_unit) / total_samples if total_samples > 0 else 1.0
                        temp_nan_ratio = np.sum(temp_mask_unit) / total_samples if total_samples > 0 else 1.0

                        # プロット
                        plt.figure(figsize=(12, 8))

                        # 除去された箇所のインデックス
                        removed_mask = combined_mask
                        removed_indices = np.where(removed_mask)[0]

                        # 除去箇所ごとにプロット
                        if len(removed_indices) > 0:
                            for j, idx in enumerate(removed_indices):
                                prev_rem = removed_indices[j - 1] if j > 0 else -1
                                rem_start = idx - 1 if idx > 0 else -1
                                rem_end = idx + 2 if idx + 1 < len(wave_unit) else -1

                                blue_label = 'Kept' if j == 0 else None
                                red_label = 'Removed' if j == 0 else None

                                # 保持された部分（青）
                                plt.plot(x_unit[prev_rem + 1:idx], wave_unit[prev_rem + 1:idx],
                                        color='blue', label=blue_label, linewidth=0.5)

                                # 除去された部分の接続（赤）
                                if rem_start >= 0:
                                    plt.plot(x_unit[rem_start:idx + 1], wave_unit[rem_start:idx + 1],
                                            color='red', linewidth=0.5, label=red_label)

                                if rem_end > 0:
                                    plt.plot(x_unit[idx:rem_end], wave_unit[idx:rem_end],
                                            color='red', linewidth=0.5)

                            # 最後の保持部分
                            last_idx = removed_indices[-1]
                            plt.plot(x_unit[last_idx + 1:], wave_unit[last_idx + 1:],
                                    color='blue', linewidth=0.5)
                        else:
                            # 除去がない場合は全て青
                            plt.plot(x_unit, wave_unit, color='blue', label='Kept', linewidth=0.5)

                        # y軸範囲を計算
                        y_min, y_max = np.nanmin(wave_unit), np.nanmax(wave_unit)
                        y_margin = (y_max - y_min) * 0.05

                        plt.xlabel("Time (minutes)")
                        plt.ylabel("EDA (μS)")
                        plt.ylim(y_min - y_margin, y_max + y_margin)
                        plt.legend()
                        plt.grid(True, alpha=0.3)

                        title = f"EDA Filtering: {name}\n"
                        title += f"Window={width}, Threshold={thres}, Sample={sample} ({sample/Fs:.1f}s)\n"
                        title += f"Remove: Total={nan_ratio:.2%}, Thr={thr_nan_ratio:.2%}, Med={med_nan_ratio:.2%}, "
                        title += f"Vel={vel_nan_ratio:.2%}, Temp={temp_nan_ratio:.2%}"
                        plt.title(title, fontsize=10)

                        plt.tight_layout()

                        # ファイル名
                        filename = f"w{width}_t{thres}_s{sample}.svg"
                        plt.savefig(f"{output_path}/{name}/{filename}")
                        plt.close()

    print(f"\nAll plots saved to {output_path}")
