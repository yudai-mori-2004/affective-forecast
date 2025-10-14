import os
from util.utils3 import load_h5_data, search_and_get_filenames, get_affective_datas_from_uuids, search_by_conditions
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.ndimage import median_filter as scipy_median_filter
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter
import neurokit2 as nk

def median_filter(x, size):
    """カスタムメディアンフィルタ（NaN対応）"""
    assert size > 0 and size % 2 == 1
    x = np.asarray(x, dtype=float)
    n = len(x)
    h = size // 2
    out = np.empty(n, dtype=float)

    for i in range(n):
        w = x[max(0, i - h):min(n, i + h + 1)]
        nan_cnt = np.isnan(w).sum()
        if nan_cnt * 2 > len(w):     # 過半数がNaNならNaN
            out[i] = np.nan
        else:
            out[i] = np.nanmedian(w)
    return out

if __name__ == "__main__":

    data_path = f"/home/mori/projects/affective-forecast/datas/biometric_data"
    output_path = f"/home/mori/projects/affective-forecast/06_EDAの特徴量/videos"
    os.makedirs(output_path, exist_ok=True)

    uuids = search_by_conditions({
        "ex-term": "term1"
    })
    datas = get_affective_datas_from_uuids(uuids)

    datas.sort(key=lambda x: int(x["filename"].split("_")[1]))
    data_names = [d["filename"] for d in datas]

    Fs = 4.0

    # パターン
    median_windows = [1, 5, 17, 33, 61]
    median_thresholds = [0.05, 0.15, 0.30, 0.45, 0.60]

    # 処理するデータ数を指定
    num_data = 10
    selected_data_names = data_names[:num_data]

    for data_idx, name in enumerate(selected_data_names):
        print(f"\n{'='*60}")
        print(f"Processing {data_idx+1}/{len(selected_data_names)}: {name}")
        print(f"{'='*60}")

        eda_file_name = f"{name}_eda.h5"
        eda_h5 = load_h5_data(f"{data_path}/{eda_file_name}")

        if eda_h5 is not None and eda_h5.shape[1] >= 3599:
            # 元のEDAデータ
            eda_original = np.asarray(eda_h5[0], dtype=float)

            # threshold filter
            extend_sec = 5
            extend_samples = int(Fs * extend_sec)
            thr_mask = (eda_original < 0.05) | (eda_original > 60)
            thr_mask = np.convolve(thr_mask.astype(int), np.ones(2*extend_samples+1, dtype=int), mode='same') > 0

            # lowpass filter (sample_plot.pyと同様)
            b, a = signal.butter(4, 0.35, btype='low', fs=Fs)
            eda_lowpass = signal.filtfilt(b, a, eda_original)

            # 時間軸（分単位）
            x_time = np.linspace(0, len(eda_lowpass) / Fs / 60, len(eda_lowpass))

            # y軸範囲を固定（0～オリジナルデータの最大値）
            y_max_fixed = np.nanmax(eda_original)
            y_min_fixed = 0

            # すべての組み合わせを生成
            param_combinations = []
            for width in median_windows:
                for thres in median_thresholds:
                    param_combinations.append((width, thres))

            print(f"Total frames: {len(param_combinations)}")

            # プロット用のフィギュアを作成
            fig, ax = plt.subplots(figsize=(12, 8))

            def update_plot(frame_idx):
                """各フレームでプロットを更新"""
                ax.clear()

                width, thres = param_combinations[frame_idx]

                # median filter適用
                med = scipy_median_filter(eda_original, size=width, mode='nearest')
                med_mask = np.abs((eda_original / med) - 1.0) > thres

                # 全マスクを統合
                combined_mask = thr_mask | med_mask

                # 除去率の計算
                total_samples = len(eda_lowpass)
                nan_ratio = np.sum(combined_mask) / total_samples
                thr_nan_ratio = np.sum(thr_mask) / total_samples
                med_nan_ratio = np.sum(med_mask) / total_samples

                # 除去部分を直線補間したEDAを作成
                eda_interpolated = eda_lowpass.copy()
                if np.any(combined_mask):
                    # マスク部分にNaNを設定
                    eda_interpolated[combined_mask] = np.nan
                    # 直線補間（NaN部分を補間）
                    valid_indices = np.where(~combined_mask)[0]
                    if len(valid_indices) > 1:
                        eda_interpolated = np.interp(
                            np.arange(len(eda_interpolated)),
                            valid_indices,
                            eda_lowpass[valid_indices]
                        )
                    else:
                        # 有効な点が少なすぎる場合はそのまま
                        eda_interpolated = eda_lowpass.copy()

                # 補間後のEDAに対してcvxEDAでトニック成分を抽出
                try:
                    signals = nk.eda_phasic(eda_interpolated, sampling_rate=Fs, method="cvxeda")
                    tonic = signals['EDA_Tonic'].to_numpy()
                    phasic = signals['EDA_Phasic'].to_numpy()
                except:
                    # cvxEDAが失敗した場合は元のデータを使用
                    tonic = eda_interpolated
                    phasic = np.zeros_like(eda_interpolated)

                # トニック成分を灰色でプロット（背景として）
                ax.plot(x_time, tonic, color='gray', linestyle="--", linewidth=1.5, alpha=0.7, label='Tonic (cvxEDA)', zorder=1)

                # Phasic成分を緑色でプロット
                ax.plot(x_time, phasic, color='green', linestyle=":", linewidth=1.0, alpha=0.6, label='Phasic (cvxEDA)', zorder=1)

                # プロット（sample_plot.pyと同じスタイル）
                removed_mask = combined_mask
                removed_indices = np.where(removed_mask)[0]

                if len(removed_indices) > 0:
                    for j, idx in enumerate(removed_indices):
                        prev_rem = removed_indices[j - 1] if j > 0 else -1
                        rem_start = idx - 1 if idx > 0 else -1
                        rem_end = idx + 2 if idx + 1 < len(eda_lowpass) else -1

                        blue_label = 'Kept' if j == 0 else None
                        red_label = 'Removed' if j == 0 else None

                        # 保持された部分（青）
                        ax.plot(x_time[prev_rem + 1:idx], eda_lowpass[prev_rem + 1:idx],
                                color='blue', label=blue_label, linewidth=0.5, zorder=2)

                        # 除去された部分の接続（赤）
                        if rem_start >= 0:
                            ax.plot(x_time[rem_start:idx + 1], eda_lowpass[rem_start:idx + 1],
                                    color='red', linewidth=0.5, label=red_label, zorder=2)

                        if rem_end > 0:
                            ax.plot(x_time[idx:rem_end], eda_lowpass[idx:rem_end],
                                    color='red', linewidth=0.5, zorder=2)

                    # 最後の保持部分
                    last_idx = removed_indices[-1]
                    ax.plot(x_time[last_idx + 1:], eda_lowpass[last_idx + 1:],
                            color='blue', linewidth=0.5, zorder=2)
                else:
                    # 除去がない場合は全て青
                    ax.plot(x_time, eda_lowpass, color='blue', label='Kept', linewidth=0.5, zorder=2)

                # y軸範囲を固定値で設定
                y_margin = (y_max_fixed - y_min_fixed) * 0.05

                ax.set_xlabel("Time (minutes)", fontsize=12)
                ax.set_ylabel("EDA (μS)", fontsize=12)
                ax.set_ylim(y_min_fixed - y_margin, y_max_fixed + y_margin)
                ax.legend(loc='upper right')
                ax.grid(True, alpha=0.3)

                # タイトル
                title = f"EDA Filtering: {name}\n"
                title += f"Median Window={width}, Threshold={thres:.2f}\n"
                title += f"Remove: Total={nan_ratio:.2%}, Thr={thr_nan_ratio:.2%}, Med={med_nan_ratio:.2%}"
                ax.set_title(title, fontsize=11, fontweight='bold')

                print(f"Frame {frame_idx + 1}/{len(param_combinations)}: Window={width}, Threshold={thres:.2f}")

            # アニメーション作成
            anim = animation.FuncAnimation(
                fig,
                update_plot,
                frames=len(param_combinations),
                interval=500,  # 各フレーム500ms
                repeat=True
            )

            # MP4として保存（利用可能なwriterを使用）
            output_file = f"{output_path}/{name}_filter_improved_animation.mp4"

            # 利用可能なwriterを確認
            try:
                writer = FFMpegWriter(fps=2, metadata=dict(artist='Me'), bitrate=1800)
                print(f"\nUsing FFMpegWriter...")
            except:
                from matplotlib.animation import PillowWriter
                writer = PillowWriter(fps=2, metadata=dict(artist='Me'))
                print(f"\nFFmpeg not found. Using PillowWriter...")

            print(f"Saving video to {output_file}...")
            anim.save(output_file, writer=writer)
            plt.close()

            print(f"\nVideo saved successfully: {output_file}")
        else:
            print(f"Error: Could not load EDA data for {name}")






