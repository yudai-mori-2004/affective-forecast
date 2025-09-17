import os
from util.utils3 import load_h5_data, search_and_get_filenames
import matplotlib.pyplot as plt
import numpy as np
import neurokit2 as nk
from scipy.ndimage import gaussian_filter1d

# ガウシアンフィルタによって高周波のノイズ除去を行ったものをプロット

if __name__ == "__main__":
    
    # パス定義
    data_path = f"/home/mori/projects/affective-forecast/datas/biometric_data"
    output_path = f"/home/mori/projects/affective-forecast/plots/GaussianFilteredEDAのプロット"
    
    # フォルダが存在しない場合は作成
    os.makedirs(output_path, exist_ok=True)

    data_names = search_and_get_filenames({
        "ex-term": "term1"
    })

    for i, name in enumerate(data_names):
        eda_file_name = f"{name}_eda.h5"
        eda = load_h5_data(f"{data_path}/{eda_file_name}")

        if eda is not None and eda.shape[1] > 3000:
            x = np.linspace(0, 15, eda.shape[1])
            y = eda[0]

            # 4HzのEDA信号に,40サンプル幅(10秒間), 標準偏差0.4秒のガウシアンフィルタをかける 
            Fs = 4
            sigma_sec = 0.4
            sigma_samp = sigma_sec * Fs  # サンプル単位でσを計算 (=1.6)
            y_smooth = gaussian_filter1d(y, sigma=sigma_samp, mode="nearest") # 端の処理は、端点の値を繰り返す処理

            plt.plot(x, y_smooth, linewidth=1.5)
            plt.xlabel("Time (minutes)")
            plt.ylabel(f"Gaussian filtered EDA (μS)")

            plt.grid(True, alpha=0.3)
            plt.title(f"{name} Gaussian filtered EDA plot")
            
            plt.tight_layout()
            plt.savefig(f"{output_path}/{name}_gaussian_filtered_eda_plot.png")
            plt.close()

            print(f"Saved {i} images")
    