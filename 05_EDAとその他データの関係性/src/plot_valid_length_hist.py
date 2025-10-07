import os
from util.utils3 import load_h5_data, search_and_get_filenames
import matplotlib.pyplot as plt
import numpy as np
from filter_definition import rri_filter, conditional_median_filter
from scipy import signal


if __name__ == "__main__":

    data_path = f"/home/mori/projects/affective-forecast/datas/biometric_data"
    output_path = f"/home/mori/projects/affective-forecast/plots/EDAフィルタリング"
    os.makedirs(output_path, exist_ok=True)

    data_names = search_and_get_filenames({
        "ex-term": "term1"
    })

    data_names.sort(key=lambda x: int(x.split("_")[1]))

    # パラメータ設定
    Fs = 4
    eda_lowpass_cutoff = 0.35
    rri_lowpass_cutoff = 0.1
    rri_threshold = 0.00015
    min_duration_seconds = 10.0
    min_samples = int(min_duration_seconds * Fs)
    median_kernel_size = 4 * Fs + 1
    p = 0.2

    # 各データの最大有効連続時間を記録するリスト
    max_valid_durations = []

    for i, name in enumerate(data_names):
        eda_file_name = f"{name}_eda.h5"
        rri_file_name = f"{name}_rri.h5"
        eda = load_h5_data(f"{data_path}/{eda_file_name}")
        rri = load_h5_data(f"{data_path}/{rri_file_name}")

        if eda is not None and rri is not None:
            # EDAデータの取得とローパスフィルタ適用
            eda_data = np.asarray(eda[0], dtype=float)
            b, a = signal.butter(4, eda_lowpass_cutoff, btype='low')
            eda_lowpass = signal.filtfilt(b, a, eda_data)

            # RRIデータの取得とローパスフィルタ適用
            rri_data = np.asarray(rri[0], dtype=float)
            b, a = signal.butter(4, rri_lowpass_cutoff, btype='low')
            rri_lowpass = signal.filtfilt(b, a, rri_data)

            # rri_filterを適用
            eda_rri_filtered = rri_filter(eda_lowpass, rri_lowpass, rri_threshold, min_samples)
            rri_mask = np.isnan(eda_rri_filtered)

            # conditional_median_filterを適用
            eda_median_filtered = conditional_median_filter(eda_lowpass, median_kernel_size, p)
            median_mask = np.isnan(eda_median_filtered)

            # 2つのマスクをORで結合（True = 除去）
            combined_mask = rri_mask | median_mask

            # 有効データマスク（False = 有効）
            valid_mask = ~combined_mask

            # 有効データの連続区間を検出し、最大長を計算
            max_valid_length = 0
            current_length = 0

            for is_valid in valid_mask:
                if is_valid:
                    current_length += 1
                    max_valid_length = max(max_valid_length, current_length)
                else:
                    current_length = 0

            # サンプル数を秒に変換
            max_valid_duration_seconds = max_valid_length / Fs
            max_valid_durations.append(max_valid_duration_seconds)

            print(f"{name}: {max_valid_duration_seconds:.2f} seconds")

    # 統計情報を計算
    mean_val = np.mean(max_valid_durations)
    median_val = np.median(max_valid_durations)
    min_val = np.min(max_valid_durations)
    max_val = np.max(max_valid_durations)
    n_samples = len(max_valid_durations)

    # ヒストグラムを作成
    plt.figure(figsize=(10, 6))
    bin_width = 20
    bins = np.arange(0, max(max_valid_durations) + bin_width, bin_width)
    plt.hist(max_valid_durations, bins=bins, edgecolor='black', alpha=0.7)
    plt.xlabel("Maximum Valid Continuous Duration (seconds)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Maximum Valid Continuous Data Duration")
    plt.grid(True, alpha=0.3)

    # 統計情報をテキストボックスで表示
    stats_text = f'N = {n_samples}\nMean = {mean_val:.2f} s\nMedian = {median_val:.2f} s\nMin = {min_val:.2f} s\nMax = {max_val:.2f} s'
    plt.text(0.98, 0.97, stats_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(f"{output_path}/max_valid_duration_histogram.svg")
    plt.close()

    print(f"\nHistogram saved to {output_path}/max_valid_duration_histogram.svg")
    print(f"Total samples: {len(max_valid_durations)}")
    print(f"Mean: {np.mean(max_valid_durations):.2f} seconds")
    print(f"Median: {np.median(max_valid_durations):.2f} seconds")
    print(f"Min: {np.min(max_valid_durations):.2f} seconds")
    print(f"Max: {np.max(max_valid_durations):.2f} seconds")
