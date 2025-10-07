from util.utils3 import load_h5_data, search_and_get_filenames
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

def rri_filter(eda_data, rri_data, threshold, min_samples):
    """
    RRI（心拍）フィルタ：
    心拍数データの差分の差分の値がΔ以下である区間が一定サンプル数以上続く場合、
    その区間とその前後のマージンのEDAデータ点を無効化する。

    Args:
        eda_data: EDAデータ
        rri_data: 心拍数データ（ローパス処理済み）
        threshold: 差分の差分の閾値
        min_samples: 除去対象とする最小連続サンプル数

    Returns:
        filtered_data: フィルタ済みEDAデータ
    """
    filtered_data = eda_data.copy()
    n_samples = len(eda_data)

    # 心拍数データの差分の差分を計算
    rri_diff = np.diff(rri_data)
    rri_diff_diff = np.diff(rri_diff)

    # rri_diff_diffの値がΔ以下のマスクを作成
    low_variation_mask = np.abs(rri_diff_diff) <= threshold

    # 除去マスクを元のサイズで作成
    remove_mask = np.zeros(n_samples, dtype=bool)

    # 連続するTrue区間を検出し、min_samples以上の区間のみ除去
    i = 0
    while i < len(low_variation_mask):
        if low_variation_mask[i]:
            # 連続区間の開始
            start = i
            while i < len(low_variation_mask) and low_variation_mask[i]:
                i += 1
            # 連続区間の終了（iは次のFalseまたは終端）
            end = i

            # 連続区間の長さをチェック
            if end - start >= min_samples:
                # 2階差分のインデックスstartは、元データのstart, start+1, start+2に対応
                # 2階差分のインデックスend-1は、元データのend-1, end, end+1に対応
                for j in range(start, min(end + 2, n_samples)):
                    remove_mask[j] = True
        else:
            i += 1

    # マスクに基づいてEDAデータを無効化
    filtered_data[remove_mask] = np.nan

    return filtered_data

def temperature_filter(eda_data, temp_data, temp_min, temp_max):
    """
    皮膚温度フィルタ：
    皮膚温度が30~40℃の範囲を逸脱したなら、その区間のデータ点を無効化する。

    Args:
        eda_data: EDAデータ
        temp_data: 皮膚温度データ
        temp_min: 温度下限（30℃が規定値）
        temp_max: 温度上限（40℃が規定値）

    Returns:
        filtered_data: フィルタ済みEDAデータ
    """
    filtered_data = eda_data.copy()
    n_samples = len(eda_data)

    for i in range(n_samples):
        # 皮膚温度が範囲外の場合、そのEDAデータ点を無効化
        if not (temp_min <= temp_data[i] <= temp_max):
            filtered_data[i] = np.nan

    return filtered_data

def threshold_filter(eda_data, threshold_min, threshold_max):
    """
    閾値フィルタ：
    波形中のデータ点の閾値を 0.05~60μS に定め、これを逸脱するデータ点は除去する

    Args:
        eda_data: EDAデータ
        threshold_min: EDAの閾値(最小)
        threshold_max: EDAの閾値(最大)

    Returns:
        filtered_data: フィルタ済みEDAデータ
    """
    filtered_data = eda_data.copy()
    n_samples = len(eda_data)

    for target in range(0, n_samples):
        if not (threshold_min <= eda_data[target] < threshold_max):
            filtered_data[target] = np.nan
    return filtered_data

def velocity_filter(eda_data, velocity_min, velocity_max, fs):
    """
    変化速度閾値フィルタ：
    隣接するEDAデータ点の値の変化速度が±10μS/secをこえたなら、その両端のデータ点を無効化する。

    Args:
        eda_data: EDAデータ
        velocity_min: EDA変化速度の下限(-10μS/secが規定値)
        velocity_max: EDA変化速度の上限(10μS/secが規定値)
        fs: サンプリング周波数 (Hz)

    Returns:
        filtered_data: フィルタ済みEDAデータ
    """
    filtered_data = eda_data.copy()
    n_samples = len(eda_data)
    dt = 1.0 / fs  # サンプル間の時間間隔

    for i in range(n_samples - 1):
        # 隣接する点の変化速度を計算
        velocity = (eda_data[i + 1] - eda_data[i]) / dt

        # 変化速度が閾値を超えた場合、両端を無効化
        if velocity < velocity_min or velocity > velocity_max:
            filtered_data[i] = np.nan
            filtered_data[i + 1] = np.nan

    return filtered_data


def conditional_median_filter(eda_data, median_kernel_size, p):
    """
    条件付きメジアンフィルタ：
    波形中の任意のデータ点が、そのデータを含むウィンドウ内の値の中央値から -p*100% 以上変動している場合、その値をNaNに置き換える。
    端点では特別な処理を行わず、カーネルの範囲内のデータの中央値を計算するのみ

    Args:
        data: EDAデータ
        median_kernel_size: カーネルサイズ（奇数）
        p: 閾値変動率

    Returns:
        filtered_data: 
    """
    filtered_data = eda_data.copy()
    n_samples = len(eda_data)

    for target in range(0, n_samples):
        mn = max(0, target - int(median_kernel_size / 2))
        mx = min(target + int(median_kernel_size / 2), n_samples - 1)
        arr = eda_data[mn:mx + 1]
        median = np.median(arr)
        if median > 0 and np.abs((eda_data[target] / median) - 1.0) > p:
            filtered_data[target] = np.nan

    return filtered_data