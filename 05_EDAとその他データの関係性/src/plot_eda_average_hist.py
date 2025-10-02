import os
from util.utils3 import load_h5_data, search_and_get_filenames
import matplotlib.pyplot as plt
import numpy as np
import neurokit2 as nk
from scipy import signal
from scipy import stats

# 被験者ごとの各計測の平均EDAをヒストグラムにプロットしてみる

if __name__ == "__main__":
    # パス定義
    data_path = f"/home/mori/projects/affective-forecast/datas/biometric_data"
    output_path = f"/home/mori/projects/affective-forecast/plots/各被験者のEDA平均値の分布"

    # フォルダが存在しない場合は作成
    os.makedirs(output_path, exist_ok=True)

    # Term1について、全員分プロットする S01 ~ S57
    subjects = [f"S{i:02d}" for i in range(1, 58)]

    medians_of_averages = []
    averages_of_averages = []
    modes_of_averages = []

    for subject in subjects:
        data_names = search_and_get_filenames({
            "ex-term": "term1",
            "ID": subject
        })

        data_names.sort(key=lambda x: int(x.split("_")[1]))
        sample_data_names = data_names[:]

        averages = []
        for i, name in enumerate(sample_data_names):
            eda_file_name = f"{name}_eda.h5"
            eda = load_h5_data(f"{data_path}/{eda_file_name}")
            if eda is None:
                continue
            averages.append(np.mean(eda[0]))

        if len(averages) > 0:
            medians_of_averages.append(np.median(averages))
            averages_of_averages.append(np.mean(averages))

            # ヒストグラムのビンに基づいた最頻値を計算
            max_val = max(10, int(np.max(averages)) + 1)
            bin_edges = np.arange(0, max_val + 1, 1.0)
            counts, _ = np.histogram(averages, bins=bin_edges)
            max_count_index = np.argmax(counts)
            # 最頻ビンの中央値を最頻値とする
            mode_value = bin_edges[max_count_index] + 0.5
            modes_of_averages.append(mode_value)

        if len(averages) > 0:
            plt.figure(figsize=(12, 8))

            # 0.0から1.0刻みでビンを設定
            max_val = max(10, int(np.max(averages)) + 1)  # 最低10まで、データの最大値+1まで
            bin_edges = np.arange(0, max_val + 1, 1.0)  # 0.0, 1.0, 2.0, ...

            plt.hist(averages, bins=bin_edges, color='blue', alpha=0.7, edgecolor='black')
            plt.xlabel("EDA Average Value (μS)")
            plt.ylabel("Frequency")
            plt.title(f"Distribution of EDA Average Values for {subject} (n={len(averages)})")
            plt.grid(True, alpha=0.3)

            # 統計情報を追加
            plt.axvline(np.mean(averages), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(averages):.3f}')
            plt.axvline(np.median(averages), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(averages):.3f}')
            plt.legend()

            plt.tight_layout()
            plt.savefig(f"{output_path}/{subject}_eda_average_hist.png")
            plt.close()

            print(f"Saved histogram for {subject} - {len(averages)} measurements")

    # 全被験者の中央値の分布
    if len(medians_of_averages) > 0:
        plt.figure(figsize=(12, 8))
        max_val = max(10, int(np.max(medians_of_averages)) + 1)
        bin_edges = np.arange(0, max_val + 1, 1.0)
        plt.hist(medians_of_averages, bins=bin_edges, color='green', alpha=0.7, edgecolor='black')
        plt.xlabel("EDA Median Value (μS)")
        plt.ylabel("Number of Subjects")
        plt.title(f"Distribution of EDA Median Values Across All Subjects (n={len(medians_of_averages)})")
        plt.grid(True, alpha=0.3)
        plt.gca().yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        plt.axvline(np.mean(medians_of_averages), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(medians_of_averages):.3f}')
        plt.axvline(np.median(medians_of_averages), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(medians_of_averages):.3f}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_path}/all_subjects_medians_hist.png")
        plt.close()
        print("Saved medians distribution histogram")

    # 全被験者の平均値の分布
    if len(averages_of_averages) > 0:
        plt.figure(figsize=(12, 8))
        max_val = max(10, int(np.max(averages_of_averages)) + 1)
        bin_edges = np.arange(0, max_val + 1, 1.0)
        plt.hist(averages_of_averages, bins=bin_edges, color='red', alpha=0.7, edgecolor='black')
        plt.xlabel("EDA Mean Value (μS)")
        plt.ylabel("Number of Subjects")
        plt.title(f"Distribution of EDA Mean Values Across All Subjects (n={len(averages_of_averages)})")
        plt.grid(True, alpha=0.3)
        plt.gca().yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        plt.axvline(np.mean(averages_of_averages), color='blue', linestyle='--', linewidth=2, label=f'Mean: {np.mean(averages_of_averages):.3f}')
        plt.axvline(np.median(averages_of_averages), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(averages_of_averages):.3f}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_path}/all_subjects_means_hist.png")
        plt.close()
        print("Saved means distribution histogram")

    # 全被験者の最頻値の分布
    if len(modes_of_averages) > 0:
        plt.figure(figsize=(12, 8))
        max_val = max(10, int(np.max(modes_of_averages)) + 1)
        bin_edges = np.arange(0, max_val + 1, 1.0)
        plt.hist(modes_of_averages, bins=bin_edges, color='purple', alpha=0.7, edgecolor='black')
        plt.xlabel("EDA Mode Value (μS)")
        plt.ylabel("Number of Subjects")
        plt.title(f"Distribution of EDA Mode Values Across All Subjects (n={len(modes_of_averages)})")
        plt.grid(True, alpha=0.3)
        plt.gca().yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_path}/all_subjects_modes_hist.png")
        plt.close()
        print("Saved modes distribution histogram")