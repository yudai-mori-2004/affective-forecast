import os
from util.utils3 import load_h5_data, search_and_get_filenames
import matplotlib.pyplot as plt
import numpy as np
import neurokit2 as nk
from scipy import signal

# 被験者ごとの各計測波形を同時にプロットしてみる

if __name__ == "__main__":
    # パス定義
    data_path = f"/home/mori/projects/affective-forecast/datas/biometric_data"
    output_path = f"/home/mori/projects/affective-forecast/plots/生波形の一枚プロット"

    # フォルダが存在しない場合は作成
    os.makedirs(output_path, exist_ok=True)

    # 全員分プロットすると多すぎるため、被験者 S01 ~ S03 までを選ぶ
    subjects = [f"S{i:02d}" for i in range(4, 11)]

    for subject in subjects:
        data_names = search_and_get_filenames({
            "ex-term": "term1",
            "ID": subject
        })

        data_names.sort(key=lambda x: int(x.split("_")[1]))
        sample_data_names = data_names[:]
        os.makedirs(f"{output_path}/{subject}", exist_ok=True)

        for i, name in enumerate(sample_data_names):
            eda_file_name = f"{name}_eda.h5"
            temp_file_name = f"{name}_temp.h5"
            rri_file_name = f"{name}_rri.h5"
            act_file_name = f"{name}_act.h5"

            eda = load_h5_data(f"{data_path}/{eda_file_name}")
            temp = load_h5_data(f"{data_path}/{temp_file_name}")
            rri = load_h5_data(f"{data_path}/{rri_file_name}")
            act = load_h5_data(f"{data_path}/{act_file_name}")

            fig, axes = plt.subplots(13, 1, figsize=(12, 40), sharex=True)

            if eda is None or temp is None or rri is None or act is None:
                continue

            if eda is not None:
                x = np.linspace(0, 15, eda.shape[1])
                y = np.asarray(eda[0], dtype=float)
                axes[0].plot(x, y, linewidth=0.5, color='blue')
                axes[0].set_ylabel("EDA (μS)")
                axes[0].grid(True, alpha=0.3)
                axes[0].set_title(f"EDA Signal ({name})")

            if temp is not None:
                x = np.linspace(0, 15, temp.shape[1])
                y = np.asarray(temp[0], dtype=float)
                axes[1].plot(x, y, linewidth=0.5, color='red')
                axes[1].set_ylabel("Temperature (°C)")
                axes[1].grid(True, alpha=0.3)
                axes[1].set_title(f"Temperature Signal ({name})")

            if rri is not None:
                x = np.linspace(0, 15, rri.shape[1])
                y = np.asarray(rri[0], dtype=float)
                axes[2].plot(x, y, linewidth=0.5, color='green')
                axes[2].set_ylabel("RRI (s)")
                axes[2].grid(True, alpha=0.3)
                axes[2].set_title(f"RRI Signal ({name})")

            if act is not None:
                x = np.linspace(0, 15, act.shape[1])

                # act[0]: x軸方向の生加速度
                y = np.asarray(act[0], dtype=float)
                axes[3].plot(x, y, linewidth=0.5, color='red')
                axes[3].set_ylabel("Raw X accel (G)")
                axes[3].grid(True, alpha=0.3)
                axes[3].set_title(f"Raw X Acceleration ({name})")

                # act[1]: y軸方向の生加速度
                y = np.asarray(act[1], dtype=float)
                axes[4].plot(x, y, linewidth=0.5, color='green')
                axes[4].set_ylabel("Raw Y accel (G)")
                axes[4].grid(True, alpha=0.3)
                axes[4].set_title(f"Raw Y Acceleration ({name})")

                # act[2]: z軸方向の生加速度
                y = np.asarray(act[2], dtype=float)
                axes[5].plot(x, y, linewidth=0.5, color='blue')
                axes[5].set_ylabel("Raw Z accel (G)")
                axes[5].grid(True, alpha=0.3)
                axes[5].set_title(f"Raw Z Acceleration ({name})")

                # act[3]: 身体運動ベクトルの大きさ
                y = np.asarray(act[3], dtype=float)
                axes[6].plot(x, y, linewidth=0.5, color='black')
                axes[6].set_ylabel("Body motion magnitude (G)")
                axes[6].grid(True, alpha=0.3)
                axes[6].set_title(f"Body Motion Magnitude ({name})")

                # act[4]: x軸方向の加速度の重力成分
                y = np.asarray(act[4], dtype=float)
                axes[7].plot(x, y, linewidth=0.5, color='cyan')
                axes[7].set_ylabel("Gravity X (G)")
                axes[7].grid(True, alpha=0.3)
                axes[7].set_title(f"Gravity X Component ({name})")

                # act[5]: y軸方向の加速度の重力成分
                y = np.asarray(act[5], dtype=float)
                axes[8].plot(x, y, linewidth=0.5, color='magenta')
                axes[8].set_ylabel("Gravity Y (G)")
                axes[8].grid(True, alpha=0.3)
                axes[8].set_title(f"Gravity Y Component ({name})")

                # act[6]: z軸方向の加速度の重力成分
                y = np.asarray(act[6], dtype=float)
                axes[9].plot(x, y, linewidth=0.5, color='yellow')
                axes[9].set_ylabel("Gravity Z (G)")
                axes[9].grid(True, alpha=0.3)
                axes[9].set_title(f"Gravity Z Component ({name})")

                # act[7]: x軸方向の身体運動成分
                y = np.asarray(act[7], dtype=float)
                axes[10].plot(x, y, linewidth=0.5, color='orange')
                axes[10].set_ylabel("Body motion X (G)")
                axes[10].grid(True, alpha=0.3)
                axes[10].set_title(f"Body Motion X ({name})")

                # act[8]: y軸方向の身体運動成分
                y = np.asarray(act[8], dtype=float)
                axes[11].plot(x, y, linewidth=0.5, color='purple')
                axes[11].set_ylabel("Body motion Y (G)")
                axes[11].grid(True, alpha=0.3)
                axes[11].set_title(f"Body Motion Y ({name})")

                # act[9]: z軸方向の身体運動成分
                y = np.asarray(act[9], dtype=float)
                axes[12].plot(x, y, linewidth=0.5, color='brown')
                axes[12].set_ylabel("Body motion Z (G)")
                axes[12].grid(True, alpha=0.3)
                axes[12].set_title(f"Body Motion Z ({name})")

            axes[12].set_xlabel("Time (minutes)")

            plt.tight_layout()
            plt.savefig(f"{output_path}/{subject}/all_signals_{name}.svg")
            plt.close()

            print(f"Saved all_signals - {i} images")