import os
from util.utils3 import load_h5_data, search_and_get_filenames
import matplotlib.pyplot as plt
import numpy as np

# RRIについて、感情価（Valence）が 4（最高）と -4（最低）のものをすべて図示してみる。
# 実際の波形データを観察してこの2群の性質に違いがないか確かめる

if __name__ == "__main__":
    
    # パス定義
    data_path = f"/home/mori/projects/affective-forecast/datas/biometric_data"
    output_path = f"/home/mori/projects/affective-forecast/plots/EDAの生データプロット"
    
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

            plt.plot(x, y, linewidth=1.5)
            plt.xlabel("Time (minutes)")
            plt.ylabel(f"EDA (μS)")

            plt.grid(True, alpha=0.3)
            plt.title(f"{name} EDA plot")
            
            plt.tight_layout()
            plt.savefig(f"{output_path}/{name}_eda_plot.png")
            plt.close()

            print(f"Saved {i} images")
    