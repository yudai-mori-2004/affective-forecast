import os
from util.utils3 import load_h5_data, search_and_get_filenames
import matplotlib.pyplot as plt
import numpy as np
import neurokit2 as nk
from scipy import signal

# パターン６：相対変化による急落直後復帰検出 + ローパス + cvxEDAでSCRの抽出

def detect_and_remove_drop_recovery_relative(data, fs, dt_max=5.0, recovery_time_max=5.0, median_window=60.0, r_drop=0.80, r_rec=0.95):
    """
    相対変化による急落&復帰パターンを検出して除去する

    Args:
        data: EDAデータ
        fs: サンプリング周波数
        dt_max: 急落判定の最大時間幅 (秒)
        recovery_time_max: 復帰判定の最大時間 (秒)
        median_window: 中央値計算で遡る時間幅 (秒)
        r_drop: 急落率の閾値 (例: 0.80 = 20%下落)
        r_rec: 復帰率の閾値 (例: 0.95 = 95%まで復帰)

    Returns:
        filtered_data: 急落&復帰パターンが除去されたデータ
        median_points: 中央値計算タイミング (時刻t)
        drop_points: 急落検出タイミング (時刻t+Δt)
        recovery_points: 復帰検出タイミング (時刻t+Δt+ΔT)
    """
    filtered_data = data.copy()
    n_samples = len(data)

    # 検出ポイントを記録
    median_points = []
    drop_points = []
    recovery_points = []

    # 時間幅をサンプル数に変換
    dt_max_samples = int(dt_max * fs)
    recovery_max_samples = int(recovery_time_max * fs)
    median_window_samples = int(median_window * fs)

    # 全ての候補区間を先に列挙
    intervals = []

    # 異なるΔtで急落を探索
    for delta_t_samples in range(1, dt_max_samples + 1):
        for t in range(median_window_samples, n_samples - delta_t_samples):
            # 時刻tでの中央値m(t)を計算 (元データから)
            start_median = max(0, t - median_window_samples)
            m_t = np.median(data[start_median:t+1])

            if m_t <= 0:  # ゼロ除算回避
                continue

            # 急落判定: r(t,t+Δt) = y(t+Δt) / m(t) < r_drop
            y_drop = data[t + delta_t_samples]
            r_drop_ratio = y_drop / m_t

            if r_drop_ratio < r_drop:
                # 復帰判定の探索
                recovery_end = min(t + delta_t_samples + recovery_max_samples, n_samples)

                for recovery_t in range(t + delta_t_samples, recovery_end):
                    # r(t,t+Δt+ΔT) = y(t+Δt+ΔT) / m(t) > r_rec
                    y_recovery = data[recovery_t]
                    r_recovery_ratio = y_recovery / m_t

                    if r_recovery_ratio > r_rec:
                        # 急落&復帰パターンを検出 - 候補区間として記録
                        intervals.append((t, recovery_t))

                        # 検出ポイントを記録
                        median_points.append(t)
                        drop_points.append(t + delta_t_samples)
                        recovery_points.append(recovery_t)
                        break

    # 区間を結合（順序に依存しない処理）
    intervals.sort()
    merged_intervals = []
    for start, end in intervals:
        if not merged_intervals or start > merged_intervals[-1][1] + 1:
            merged_intervals.append([start, end])
        else:
            merged_intervals[-1][1] = max(merged_intervals[-1][1], end)

    # 結合された区間を一括でマスク
    remove_mask = np.zeros(n_samples, dtype=bool)
    for start, end in merged_intervals:
        remove_mask[start:end+1] = True

    # 除去対象ポイントをNaNでマーク
    filtered_data[remove_mask] = np.nan

    # 線形補間
    valid_indices = ~np.isnan(filtered_data)
    if np.sum(valid_indices) < 2:
        return data, median_points, drop_points, recovery_points  # 補間できない場合は元データを返す

    x = np.arange(len(filtered_data))
    filtered_data = np.interp(x, x[valid_indices], filtered_data[valid_indices])

    return filtered_data, median_points, drop_points, recovery_points

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
    output_path = f"/home/mori/projects/affective-forecast/plots/EDA分離プロットパターン6"

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

                # 1. 相対変化による急落&復帰パターンの除去
                y_drop_filtered, median_points, drop_points, recovery_points = detect_and_remove_drop_recovery_relative(y, Fs, recovery_time_max=recovery_time)

                # 2. ローパスフィルタ (0.25Hz以上をカット)
                y_lowpass = lowpass_filter(y_drop_filtered, 0.25, Fs)

                # 3. cvxEDAで分離
                signals = nk.eda_phasic(y_lowpass, sampling_rate=Fs, method="cvxeda")
                tonic = signals['EDA_Tonic'].to_numpy()
                phasic = signals['EDA_Phasic'].to_numpy()

                os.makedirs(f"{output_path}/{name}", exist_ok=True)

                plt.figure(figsize=(12, 8))
                plt.plot(x, y_lowpass, linewidth=0.5, linestyle='--', label="Filtered EDA (Relative Drop Recovery + Lowpass)")
                plt.plot(x, tonic, linewidth=0.5, label="SCL (Tonic)")
                plt.plot(x, phasic, linewidth=0.5, label="SCR (Phasic)")

                # 検出ポイントをプロット（実際のEDA値で縦軸に意味を持たせる）
                if median_points:
                    median_times = [mp / Fs / 60 for mp in median_points]  # 分単位に変換
                    # 中央値を計算して表示
                    median_values = []
                    for mp in median_points:
                        start_median = max(0, mp - int(30.0 * Fs))  # 30秒遡る
                        m_t = np.median(y[start_median:mp+1])
                        median_values.append(m_t)
                    plt.scatter(median_times, median_values, color='green', s=20, alpha=0.7, label="Median Values m(t)", zorder=5)

                if drop_points:
                    drop_times = [dp / Fs / 60 for dp in drop_points]  # 分単位に変換
                    drop_values = [y[dp] for dp in drop_points]  # 急落時の実際のEDA値
                    plt.scatter(drop_times, drop_values, color='red', s=20, alpha=0.7, label="Drop Values y(t+Δt)", zorder=5)

                if recovery_points:
                    recovery_times = [rp / Fs / 60 for rp in recovery_points]  # 分単位に変換
                    recovery_values = [y[rp] for rp in recovery_points]  # 復帰時の実際のEDA値
                    plt.scatter(recovery_times, recovery_values, color='blue', s=20, alpha=0.7, label="Recovery Values y(t+Δt+ΔT)", zorder=5)

                plt.xlabel("Time (minutes)")
                plt.ylabel(f"EDA signals (μS)")

                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.title(f"Pattern 6: Relative Drop Recovery Detection + Lowpass + cvxEDA (recovery_time={recovery_time}s, {name})")

                plt.tight_layout()
                plt.savefig(f"{output_path}/{name}/pattern6_recovery_{recovery_time}s.svg")
                plt.close()

                print(f"Saved Pattern 6 - recovery_time={recovery_time}s - {i} images")