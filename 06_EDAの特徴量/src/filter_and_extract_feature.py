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
    crop_samples = [int(20*Fs), int(40*Fs), int(60*Fs), int(120*Fs), int(180*Fs), int(300*Fs), int(600*Fs), 3599]

    # 全ての特徴量を格納するリスト
    features_list = []

    for i, name in enumerate(data_names):
        print(f"Processing {i+1}/{len(data_names)}: {name}")
        eda_file_name = f"{name}_eda.h5"
        temp_file_name = f"{name}_temp.h5"
        eda_h5 = load_h5_data(f"{data_path}/{eda_file_name}")
        temp_h5 = load_h5_data(f"{data_path}/{temp_file_name}")

        if eda_h5 is not None and temp_h5 is not None and eda_h5.shape[1] >= 3599 and eda_h5.shape[1] == temp_h5.shape[1]:
            x = np.linspace(0, 15, eda_h5.shape[1])
            eda = np.asarray(eda_h5[0], dtype=float)
            temp = np.asarray(temp_h5[0], dtype=float)

            # threshold
            thr_mask = (eda < 0.05) | (eda > 60)

            # lowpass
            b, a = signal.butter(4, 0.35, btype='low', fs=Fs)
            eda_lowpass = signal.filtfilt(b, a, eda)

            # velocity
            dt = 1.0 / Fs
            v = np.diff(eda) / dt
            vel_mask = (v < -10.0) | (v > 10.0)
            vel_mask = np.concatenate([[False], vel_mask]) | np.concatenate([vel_mask, [False]])

            # temperature
            temp_mask = (temp < 30) | (temp > 40)

            for width in median_windows:
                for thres in median_thresholds:
                    # median
                    med = median_filter(eda, size=width, mode='nearest')
                    med_mask = np.abs((eda / med) - 1.0) > thres

                    length = len(eda_lowpass)
                    for sample in crop_samples:
                        # 解析ユニットのデータ
                        end_idx = length
                        start_idx = max(0, length - sample)
                        wave_unit = eda_lowpass[start_idx:end_idx]

                        # マスクの適用（各マスクを解析範囲に合わせて切り出し）
                        thr_mask_unit = thr_mask[start_idx:end_idx]
                        med_mask_unit = med_mask[start_idx:end_idx]
                        vel_mask_unit = vel_mask[start_idx:end_idx]
                        temp_mask_unit = temp_mask[start_idx:end_idx]

                        # 全マスクを統合
                        combined_mask = thr_mask_unit | med_mask_unit | vel_mask_unit | temp_mask_unit

                        # マスクを適用してフィルタリング済みデータを作成
                        filtered_wave = wave_unit.copy()
                        filtered_wave[combined_mask] = np.nan

                        # 有効なデータのみを抽出
                        valid_wave = filtered_wave[~np.isnan(filtered_wave)]

                        # 特徴量を格納する辞書
                        feature = {}

                        # メタデータ
                        feature["name"] = name
                        feature["median_width"] = width
                        feature["median_threshold"] = thres
                        feature["crop_sample"] = sample

                        # ノイズ除去に関する情報
                        total_samples = len(wave_unit)
                        feature["nan_ratio"] = np.sum(combined_mask) / total_samples if total_samples > 0 else 1.0
                        feature["thr_nan_ratio"] = np.sum(thr_mask_unit) / total_samples if total_samples > 0 else 1.0
                        feature["med_nan_ratio"] = np.sum(med_mask_unit) / total_samples if total_samples > 0 else 1.0
                        feature["vel_nan_ratio"] = np.sum(vel_mask_unit) / total_samples if total_samples > 0 else 1.0
                        feature["temp_nan_ratio"] = np.sum(temp_mask_unit) / total_samples if total_samples > 0 else 1.0

                        # 有効データがない場合は全てNaN
                        if len(valid_wave) == 0:
                            feature["raw_mean"] = np.nan
                            feature["raw_median"] = np.nan
                            feature["raw_std"] = np.nan
                            feature["raw_min"] = np.nan
                            feature["raw_max"] = np.nan
                            feature["raw_range"] = np.nan
                            feature["raw_iqr"] = np.nan
                            feature["raw_mad"] = np.nan
                            feature["raw_skew"] = np.nan
                            feature["raw_kurtosis"] = np.nan
                            feature["rms"] = np.nan
                            feature["D1_mean"] = np.nan
                            feature["D1_std"] = np.nan
                            feature["D2_mean"] = np.nan
                            feature["D2_std"] = np.nan
                            feature["line_length"] = np.nan
                            feature["zero_cross_rate"] = np.nan
                            feature["pos_neg_ratio"] = np.nan
                            feature["ac1"] = np.nan
                            feature["ac_decay"] = np.nan
                            feature["entropy_shannon"] = np.nan
                            feature["entropy_sample"] = np.nan
                            feature["entropy_perm"] = np.nan
                            feature["trend_slope"] = np.nan
                            feature["trend_r2"] = np.nan
                            feature["detrended_rms"] = np.nan
                        else:
                            # 生データのスケールや変動の大きさ、形状に関する基本的な情報
                            feature["raw_mean"] = np.mean(valid_wave)
                            feature["raw_median"] = np.median(valid_wave)
                            feature["raw_std"] = np.std(valid_wave, ddof=1) if len(valid_wave) > 1 else np.nan
                            feature["raw_min"] = np.min(valid_wave)
                            feature["raw_max"] = np.max(valid_wave)
                            feature["raw_range"] = feature["raw_max"] - feature["raw_min"]
                            feature["raw_iqr"] = iqr(valid_wave) if len(valid_wave) > 1 else np.nan
                            feature["raw_mad"] = np.median(np.abs(valid_wave - feature["raw_median"]))
                            feature["raw_skew"] = skew(valid_wave) if len(valid_wave) > 2 else np.nan
                            feature["raw_kurtosis"] = kurtosis(valid_wave) if len(valid_wave) > 3 else np.nan
                            feature["rms"] = np.sqrt(np.mean(valid_wave ** 2))

                            # 差分・二階差分に関する情報
                            D1 = np.diff(valid_wave)
                            D2 = np.diff(D1)
                            feature["D1_mean"] = np.mean(D1) if len(D1) > 0 else np.nan
                            feature["D1_std"] = np.std(D1, ddof=1) if len(D1) > 1 else np.nan
                            feature["D2_mean"] = np.mean(D2) if len(D2) > 0 else np.nan
                            feature["D2_std"] = np.std(D2, ddof=1) if len(D2) > 1 else np.nan
                            feature["line_length"] = np.sum(np.abs(D1)) if len(D1) > 0 else np.nan

                            # ゼロクロス率
                            if len(D1) > 1:
                                zero_crossings = np.sum(D1[:-1] * D1[1:] < 0)
                                feature["zero_cross_rate"] = zero_crossings / len(D1)
                            else:
                                feature["zero_cross_rate"] = np.nan

                            # 正負勾配の比率
                            pos_count = np.sum(D1 > 0) if len(D1) > 0 else 0
                            neg_count = np.sum(D1 < 0) if len(D1) > 0 else 0
                            feature["pos_neg_ratio"] = pos_count / neg_count if neg_count > 0 else np.nan

                            # 自己相関
                            try:
                                if len(valid_wave) > 2:
                                    autocorr = np.correlate(valid_wave - np.mean(valid_wave),
                                                           valid_wave - np.mean(valid_wave),
                                                           mode='full')
                                    autocorr = autocorr[len(autocorr)//2:]
                                    if autocorr[0] != 0:
                                        autocorr = autocorr / autocorr[0]
                                        feature["ac1"] = autocorr[1] if len(autocorr) > 1 else np.nan
                                        # 1/e に減衰するτを探す
                                        decay_threshold = 1.0 / np.e
                                        decay_idx = np.where(autocorr < decay_threshold)[0]
                                        feature["ac_decay"] = decay_idx[0] if len(decay_idx) > 0 else len(autocorr)
                                    else:
                                        feature["ac1"] = np.nan
                                        feature["ac_decay"] = np.nan
                                else:
                                    feature["ac1"] = np.nan
                                    feature["ac_decay"] = np.nan
                            except Exception as e:
                                feature["ac1"] = np.nan
                                feature["ac_decay"] = np.nan

                            # エントロピー情報
                            try:
                                # シャノンエントロピー（ヒストグラムベース）
                                hist, _ = np.histogram(valid_wave, bins=min(50, len(valid_wave)//2), density=True)
                                hist = hist[hist > 0]
                                feature["entropy_shannon"] = -np.sum(hist * np.log2(hist + 1e-10))
                            except Exception as e:
                                feature["entropy_shannon"] = np.nan

                            try:
                                # サンプルエントロピー（neurokitを使用）
                                feature["entropy_sample"] = nk.entropy_sample(valid_wave, delay=1, dimension=2)[0]
                            except Exception as e:
                                feature["entropy_sample"] = np.nan

                            try:
                                # 順列エントロピー
                                feature["entropy_perm"] = nk.entropy_permutation(valid_wave, delay=1, dimension=3)[0]
                            except Exception as e:
                                feature["entropy_perm"] = np.nan

                            # 線形回帰に関する特徴量
                            try:
                                if len(valid_wave) > 2:
                                    X = np.arange(len(valid_wave)).reshape(-1, 1)
                                    y = valid_wave.reshape(-1, 1)
                                    reg = LinearRegression()
                                    reg.fit(X, y)

                                    feature["trend_slope"] = reg.coef_[0][0]
                                    feature["trend_r2"] = reg.score(X, y)

                                    # トレンド除去後のRMS
                                    y_pred = reg.predict(X)
                                    detrended = y - y_pred
                                    feature["detrended_rms"] = np.sqrt(np.mean(detrended ** 2))
                                else:
                                    feature["trend_slope"] = np.nan
                                    feature["trend_r2"] = np.nan
                                    feature["detrended_rms"] = np.nan
                            except Exception as e:
                                feature["trend_slope"] = np.nan
                                feature["trend_r2"] = np.nan
                                feature["detrended_rms"] = np.nan

                        # リストに追加
                        features_list.append(feature)

    # DataFrameに変換して保存
    print(f"\nSaving {len(features_list)} features to CSV...")
    df = pd.DataFrame(features_list)
    output_file = f"{output_path}/eda_features.csv"
    df.to_csv(output_file, index=False)
    print(f"Saved to {output_file}")
    print(f"Total features: {len(df)}")
    print(f"Feature columns: {len(df.columns)}")






