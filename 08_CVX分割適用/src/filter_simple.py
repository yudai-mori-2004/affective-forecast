import os
from util.utils3 import load_h5_data, search_and_get_filenames, get_affective_datas_from_uuids, search_by_conditions
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import neurokit2 as nk
from neurokit2.eda.eda_phasic import _eda_phasic_cvxeda
from scipy import signal
from scipy.ndimage import median_filter
from scipy.stats import skew, kurtosis, iqr
from sklearn.linear_model import LinearRegression


def median_filter(x, size):
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


def plot_cvx_eda(eda, cvx_segments, name, median_window, median_threshold, output_path, Fs):
    """
    CVXEDAで分離したtonic/phasic成分をプロット

    Parameters:
    -----------
    eda : array
        フィルタリング済みのEDAデータ（NaNを含む）
    cvx_segments : list of dict
        CVXEDA処理結果のリスト [{'start': int, 'end': int, 'tonic': array, 'phasic': array}, ...]
    name : str
        データ名
    median_window : int
        メジアンフィルタのウィンドウサイズ
    median_threshold : float
        メジアンフィルタの閾値
    output_path : str
        出力先のパス
    Fs : float
        サンプリング周波数
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # 時間軸（分単位）
    x = np.linspace(0, len(eda) / Fs / 60, len(eda))

    # 上段：元のEDAとTonic成分を重ねて表示
    original_plotted = False
    tonic_plotted = False
    for seg in cvx_segments:
        start, end = seg['start'], seg['end']

        # 元のEDA（緑）
        original_label = 'Original EDA' if not original_plotted else None
        axes[0].plot(x[start:end], eda[start:end],
                    color='green', label=original_label, linewidth=0.5, alpha=0.7)
        original_plotted = True

        # Tonic成分（青）
        tonic_label = 'Tonic' if not tonic_plotted else None
        axes[0].plot(x[start:end], seg['tonic'],
                    color='blue', label=tonic_label, linewidth=0.5)
        tonic_plotted = True

    axes[0].set_ylabel("EDA (μS)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(f"Original EDA & Tonic: {name}")

    # 下段：Phasic成分
    phasic_plotted = False
    for seg in cvx_segments:
        start, end = seg['start'], seg['end']
        label = 'Phasic' if not phasic_plotted else None
        axes[1].plot(x[start:end], seg['phasic'],
                    color='orange', label=label, linewidth=0.5)
        phasic_plotted = True
    axes[1].set_ylabel("Phasic (μS)")
    axes[1].set_xlabel("Time (minutes)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f"CVXEDA Decomposition\nWindow={median_window}, Threshold={median_threshold}", fontsize=10)
    plt.tight_layout()

    # ファイル名
    filename = f"{name}_cvxeda.svg"
    plt.savefig(f"{output_path}/{filename}")
    plt.close()


def plot_filtered_eda(eda, continuous_segments, name, median_window, median_threshold, output_path, Fs):
    """
    フィルタリングされたEDAデータをプロット

    Parameters:
    -----------
    eda : array
        フィルタリング済みのEDAデータ（NaNを含む）
    continuous_segments : list of tuple
        20秒以上の連続区間のリスト [(start, end), ...]
    name : str
        データ名
    median_window : int
        メジアンフィルタのウィンドウサイズ
    median_threshold : float
        メジアンフィルタの閾値
    output_path : str
        出力先のパス
    Fs : float
        サンプリング周波数
    """
    plt.figure(figsize=(12, 8))

    # 時間軸（分単位）
    x = np.linspace(0, len(eda) / Fs / 60, len(eda))

    # 緑でプロットする部分のマスクを作成
    green_mask = np.zeros(len(eda), dtype=bool)
    for start, end in continuous_segments:
        green_mask[start:end] = True

    # 緑の区間をプロット（20秒以上の連続区間）
    green_plotted = False
    endpoint_plotted = False
    for start, end in continuous_segments:
        label = 'Valid (≥20s)' if not green_plotted else None
        plt.plot(x[start:end], eda[start:end],
                color='green', label=label, linewidth=0.5)
        green_plotted = True

        # 端点をドットで表示
        endpoint_label = 'Endpoints' if not endpoint_plotted else None
        plt.plot([x[start], x[end-1]], [eda[start], eda[end-1]],
                'o', color='darkgreen', markersize=4, label=endpoint_label)
        endpoint_plotted = True

    # 赤の区間をプロット（緑以外の全て）
    red_mask = ~green_mask
    red_indices = np.where(red_mask)[0]

    if len(red_indices) > 0:
        # 連続する赤の区間ごとにプロット
        diff_red = np.diff(np.concatenate([[red_indices[0]-2], red_indices, [red_indices[-1]+2]]))
        breaks = np.where(diff_red > 1)[0]

        red_plotted = False
        for i in range(len(breaks) - 1):
            seg_start = red_indices[breaks[i]] if i > 0 else red_indices[0]
            seg_end = red_indices[breaks[i+1] - 1] if breaks[i+1] < len(red_indices) else red_indices[-1]

            label = 'Removed or short' if not red_plotted else None
            # plt.plot(x[seg_start:seg_end+1], eda[seg_start:seg_end+1],
            #         color='red', label=label, linewidth=0.5)
            red_plotted = True

    # y軸範囲を計算
    y_min, y_max = np.nanmin(eda), np.nanmax(eda)
    y_margin = (y_max - y_min) * 0.05

    plt.xlabel("Time (minutes)")
    plt.ylabel("EDA (μS)")
    plt.ylim(0, y_max + y_margin)
    plt.legend()
    plt.grid(True, alpha=0.3)

    title = f"EDA Filtering: {name}\n"
    title += f"Window={median_window}, Threshold={median_threshold}"
    plt.title(title, fontsize=10)

    plt.tight_layout()

    # ファイル名
    filename = f"{name}_filtered.svg"
    plt.savefig(f"{output_path}/{filename}")
    plt.close()

if __name__ == "__main__":

    data_path = f"/home/mori/projects/affective-forecast/datas/biometric_data"
    uuids = search_by_conditions({
        "ex-term": "term1",
        "ID": "S01"
    })
    datas = get_affective_datas_from_uuids(uuids)

    datas.sort(key=lambda x: int(x["filename"].split("_")[1]))
    data_names = [d["filename"] for d in datas]

    Fs = 4.0

    # medianフィルターの値は、依然決めた(33, 0.15)とする
    median_window = 33
    median_threshold = 0.15

    # 全ての特徴量を格納するリスト
    features_list = []

    for i, name in enumerate(data_names):
        print(f"Processing {i+1}/{len(data_names)}: {name}")
        eda_file_name = f"{name}_eda.h5"
        eda_h5 = load_h5_data(f"{data_path}/{eda_file_name}")

        if eda_h5 is not None and eda_h5.shape[1] >= 3599:
            x = np.linspace(0, 15, eda_h5.shape[1])
            eda = np.asarray(eda_h5[0], dtype=float)

            # 値が0.05~60から外れるものをnanに設定し、さらに、その各エラー点の前後5秒間もnanとしておく
            extend_sec = 5
            extend_samples = int(Fs * extend_sec)
            thr_mask = (eda < 0.05) | (eda > 60)
            thr_mask = np.convolve(thr_mask.astype(int), np.ones(2*extend_samples+1, dtype=int), mode='same') > 0
            eda[thr_mask] = np.nan
            

            # nanを考慮したメジアンフィルタを適用（過半数がnanならその点はnanとなる）
            med = median_filter(eda, size=median_window)
            ratio = eda / med
            med_mask = (np.abs(ratio - 1.0) > median_threshold) | (np.isnan(ratio))
            eda[med_mask] = np.nan


            # 最小連続区間を30秒と定義する
            min_length = int(30 * Fs)
            valid_mask = ~np.isnan(eda)
            diff = np.diff(np.concatenate([[False], valid_mask, [False]]).astype(int))
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0]
            continuous_segments = []
            for start, end in zip(starts, ends):
                if end - start >= min_length:
                    continuous_segments.append((start, end))

            # プロット
            output_path = f"/home/mori/projects/affective-forecast/08_CVX分割適用/plots/split"
            os.makedirs(output_path, exist_ok=True)
            plot_filtered_eda(eda, continuous_segments, name, median_window, median_threshold, output_path, Fs)

            cvx_segments = []
            for start, end in continuous_segments:
                segment_data = eda[start:end]
                # プライベート関数を直接呼び出してパラメータを調整
                # gamma: tonic成分の滑らかさ（デフォルト1e-2 → 大きくするとより滑らか）
                # delta_knot: スプラインのノット間隔（デフォルト10 → 大きくするとより滑らか）
                tonic, phasic = _eda_phasic_cvxeda(
                    segment_data,
                    sampling_rate=Fs,
                )
                cvx_segments.append({
                    'start': start,
                    'end': end,
                    'tonic': tonic,
                    'phasic': phasic
                })

            output_path = f"/home/mori/projects/affective-forecast/08_CVX分割適用/plots/cvx"
            os.makedirs(output_path, exist_ok=True)
            plot_cvx_eda(eda, cvx_segments, name, median_window, median_threshold, output_path, Fs)






