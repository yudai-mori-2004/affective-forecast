import os
from util.utils3 import load_h5_data, search_and_get_filenames
import matplotlib.pyplot as plt
import numpy as np
import neurokit2 as nk
from scipy import signal

# パターン０： 生 + cvxEDAでSCRの抽出

if __name__ == "__main__":
    # パス定義
    data_path = f"/home/mori/projects/affective-forecast/datas/biometric_data"
    output_path = f"/home/mori/projects/affective-forecast/plots/EDA分離プロットパターン0"

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

        if eda is not None and eda.shape[1] > 3000:
            x = np.linspace(0, 15, eda.shape[1])
            y = np.asarray(eda[0], dtype=float)

            Fs = 4

            # 1. cvxEDAで分離
            signals = nk.eda_phasic(y, sampling_rate=Fs, method="cvxeda")
            tonic = signals['EDA_Tonic'].to_numpy()
            phasic = signals['EDA_Phasic'].to_numpy()

            os.makedirs(f"{output_path}/{name}", exist_ok=True)

            plt.figure(figsize=(12, 8))
            plt.plot(x, y, linewidth=0.5, linestyle='--', label="Raw EDA")
            plt.plot(x, tonic, linewidth=0.5, label="SCL (Tonic)")
            plt.plot(x, phasic, linewidth=0.5, label="SCR (Phasic)")
            plt.xlabel("Time (minutes)")
            plt.ylabel(f"EDA signals (μS)")

            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.title(f"Pattern 0: Raw + cvxEDA ({name})")

            plt.tight_layout()
            plt.savefig(f"{output_path}/{name}/pattern0.svg")
            plt.close()

            print(f"Saved Pattern 0 - {i} images")