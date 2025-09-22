import os
from util.utils3 import load_h5_data, search_and_get_filenames
import matplotlib.pyplot as plt
import numpy as np
import neurokit2 as nk # https://neuropsychology.github.io/NeuroKit/functions/eda.html
from scipy.ndimage import gaussian_filter1d

# ガウシアンフィルタによって高周波のノイズ除去を行ったものをプロット

if __name__ == "__main__":
    # パス定義
    data_path = f"/home/mori/projects/affective-forecast/datas/biometric_data"
    output_path = f"/home/mori/projects/affective-forecast/plots/EDA分離プロット10サンプル比較"
    
    # フォルダが存在しない場合は作成
    os.makedirs(output_path, exist_ok=True)

    data_names = search_and_get_filenames({
        "ex-term": "term1"
    })
    
    # 先頭の10枚の画像のみを比較用のサンプルとする
    data_names.sort(key=lambda x: int(x.split("_")[1]))
    sample_data_names = data_names[:10]

    for sigma_sec in [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0]:
        for i, name in enumerate(sample_data_names):
            eda_file_name = f"{name}_eda.h5"
            eda = load_h5_data(f"{data_path}/{eda_file_name}")

            if eda is not None and eda.shape[1] > 3000:
                x = np.linspace(0, 15, eda.shape[1])
                y = np.asarray(eda[0], dtype=float)

                # 4HzのEDA信号に,40サンプル幅(10秒間), 標準偏差sigma_sec秒のガウシアンフィルタをかける 
                Fs = 4
                sigma_samp = sigma_sec * Fs  # サンプル単位でσを計算 (=1.6)
                y_smooth = gaussian_filter1d(y, sigma=sigma_samp, mode="nearest") # 端の処理は、端点の値を繰り返す処理

                signals = nk.eda_phasic(y_smooth, sampling_rate=Fs, method="cvxeda") # 分解だけcvxEDA
                tonic  = signals['EDA_Tonic'].to_numpy()
                phasic = signals['EDA_Phasic'].to_numpy()

                os.makedirs(f"{output_path}/{name}", exist_ok=True)

                plt.plot(x, y_smooth, linewidth=0.5, linestyle='--', label="Filtered EDA")
                plt.plot(x, tonic, linewidth=0.5, label="Tonic")
                plt.plot(x, phasic, linewidth=0.5, label="Phasic")
                plt.xlabel("Time (minutes)")
                plt.ylabel(f"Gaussian filtered EDA (μS)")

                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.title(f"{name} Separated EDA plot")
                
                plt.tight_layout()
                plt.savefig(f"{output_path}/{name}/sigma_{sigma_sec}s.svg")
                plt.close()

                print(f"Saved {i} images")
        