import os
from util.utils3 import load_h5_data, search_and_get_filenames, get_affective_datas_from_uuids, search_by_conditions
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import neurokit2 as nk
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

if __name__ == "__main__":

    data_path = f"/home/mori/projects/affective-forecast/datas/biometric_data"
    output_path = f"/home/mori/projects/affective-forecast/06_EDAの特徴量/feature"
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

    # 全ての特徴量を格納するリスト
    features_list = []

    for i, name in enumerate(data_names):
        print(f"Processing {i+1}/{len(data_names)}: {name}")
        eda_file_name = f"{name}_eda.h5"
        eda_h5 = load_h5_data(f"{data_path}/{eda_file_name}")

        if eda_h5 is not None and eda_h5.shape[1] >= 3599:
            x = np.linspace(0, 15, eda_h5.shape[1])
            eda = np.asarray(eda_h5[0], dtype=float)

            # threshold
            extend_sec = 5
            extend_samples = int(Fs * extend_sec)
            thr_mask = (eda < 0.05) | (eda > 60)
            thr_mask = np.convolve(thr_mask.astype(int), np.ones(2*extend_samples+1, dtype=int), mode='same') > 0
            eda[thr_mask] = np.nan
            

            for width in median_windows:
                for thres in median_thresholds:
                    # median
                    med = median_filter(eda, size=width)
                    ratio = eda / med
                    med_mask = (np.abs(ratio - 1.0) > thres) | (np.isnan(ratio))
                    eda[thr_mask] = np.nan

                    eda_interpolated = eda.copy()
                    combined_mask = thr_mask | med_mask

                    if np.any(combined_mask):
                        # マスク部分にNaNを設定
                        eda_interpolated[combined_mask] = np.nan
                        # 直線補間（NaN部分を補間）
                        valid_indices = np.where(~combined_mask)[0]
                        if len(valid_indices) > 1:
                            eda_interpolated = np.interp(
                                np.arange(len(eda_interpolated)),
                                valid_indices,
                                eda[valid_indices]
                            )
                        else:
                            eda_interpolated = eda.copy()



                      


    print(f"Saved to {output_file}")






