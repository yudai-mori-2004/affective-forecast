import os
from util.utils3 import load_h5_data, search_and_get_filenames
import matplotlib.pyplot as plt
import numpy as np
import neurokit2 as nk
from scipy import signal

# パターン５：急落直後に復帰するものを除外 + ローパス + cvxEDAでSCRの抽出

def detect_and_remove_drop_recovery(data, fs, drop_threshold=-1.0, recovery_time_max=5.0, recovery_ratio=0.8):
    """
    急落&復帰パターンを検出して除去する

    Args:
        data: EDAデータ
        fs: サンプリング周波数
        drop_threshold: 急落判定の閾値 (μS/s)
        recovery_time_max: 復帰判定の最大時間 (秒)
        recovery_ratio: 復帰判定の閾値比率

    Returns:
        filtered_data: 急落&復帰パターンが除去されたデータ
    """
    filtered_data = data.copy()
    n_samples = len(data)

    # 除去するポイントをマーク
    remove_mask = np.zeros(n_samples, dtype=bool)

    # 異なるΔtで急落を探索 (1サンプル～5秒分)
    for delta_t_samples in range(1, int(fs * 5) + 1):
        delta_t_sec = delta_t_samples / fs

        for t in range(n_samples - delta_t_samples):
            if remove_mask[t]:  # 既に除去対象になっている場合はスキップ
                continue

            y_base = data[t]
            y_min = data[t + delta_t_samples]
            delta_y = y_min - y_base

            # 急落判定
            velocity = delta_y / delta_t_sec
            if velocity < drop_threshold:
                # 復帰判定の探索範囲
                recovery_samples = int(recovery_time_max * fs)
                recovery_end = min(t + delta_t_samples + recovery_samples, n_samples)

                # 復帰判定
                recovery_threshold = y_min + recovery_ratio * (y_base - y_min)
                recovery_found = False

                for recovery_t in range(t + delta_t_samples, recovery_end):
                    if data[recovery_t] > recovery_threshold:
                        # 急落&復帰パターンを検出
                        remove_mask[t:recovery_t+1] = True
                        recovery_found = True
                        break

    # 除去対象ポイントをNaNでマーク
    filtered_data[remove_mask] = np.nan

    # 線形補間
    valid_indices = ~np.isnan(filtered_data)
    if np.sum(valid_indices) < 2:
        return data  # 補間できない場合は元データを返す

    x = np.arange(len(filtered_data))
    filtered_data = np.interp(x, x[valid_indices], filtered_data[valid_indices])

    return filtered_data

def lowpass_filter(data, cutoff, fs):
    """ローパスフィルタ"""
    nyquist = fs / 2
    normalized_cutoff = cutoff / nyquist

    if normalized_cutoff >= 1:
        normalized_cutoff = 0.99

    b, a = signal.butter(4, normalized_cutoff, btype='low')
    filtered_data = signal.filtfilt(b, a, data)
    return filtered_data

if __name__ == "__main__":
    # パス定義
    data_path = f"/home/mori/projects/affective-forecast/datas/biometric_data"
    output_path = f"/home/mori/projects/affective-forecast/plots/EDA分離プロットパターン5"

    # フォルダが存在しない場合は作成
    os.makedirs(output_path, exist_ok=True)

    data_names = search_and_get_filenames({
        "ex-term": "term1"
    })

    # 先頭の10枚の画像のみを比較用のサンプルとする
    data_names.sort(key=lambda x: int(x.split("_")[1]))
    sample_data_names = data_names[:10]

    # recovery_time_maxの比較パターン
    recovery_times = [5, 10, 20, 40]

    for recovery_time in recovery_times:
        for i, name in enumerate(sample_data_names):
            eda_file_name = f"{name}_eda.h5"
            eda = load_h5_data(f"{data_path}/{eda_file_name}")

            if eda is not None and eda.shape[1] > 3000:
                x = np.linspace(0, 15, eda.shape[1])
                y = np.asarray(eda[0], dtype=float)

                Fs = 4

                # 1. 急落&復帰パターンの除去 (recovery_time_maxを変更)
                y_drop_filtered = detect_and_remove_drop_recovery(y, Fs, recovery_time_max=recovery_time)

                # 2. ローパスフィルタ (0.25Hz以上をカット)
                y_lowpass = lowpass_filter(y_drop_filtered, 0.25, Fs)

                # 3. cvxEDAで分離
                signals = nk.eda_phasic(y_lowpass, sampling_rate=Fs, method="cvxeda")
                tonic = signals['EDA_Tonic'].to_numpy()
                phasic = signals['EDA_Phasic'].to_numpy()

                os.makedirs(f"{output_path}/{name}", exist_ok=True)

                plt.figure(figsize=(12, 8))
                plt.plot(x, y_lowpass, linewidth=0.5, linestyle='--', label="Filtered EDA (Drop Recovery + Lowpass)")
                plt.plot(x, tonic, linewidth=0.5, label="SCL (Tonic)")
                plt.plot(x, phasic, linewidth=0.5, label="SCR (Phasic)")
                plt.xlabel("Time (minutes)")
                plt.ylabel(f"EDA signals (μS)")

                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.title(f"Pattern 5: Drop Recovery Detection + Lowpass + cvxEDA (recovery_time={recovery_time}s, {name})")

                plt.tight_layout()
                plt.savefig(f"{output_path}/{name}/pattern5_recovery_{recovery_time}s.svg")
                plt.close()

                print(f"Saved Pattern 5 - recovery_time={recovery_time}s - {i} images")