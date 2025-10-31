import os
from util.utils3 import load_h5_data, get_affective_datas_from_uuids, search_by_conditions
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import neurokit2 as nk
from neurokit2.eda.eda_phasic import _eda_phasic_cvxeda
from scipy import signal
from scipy.ndimage import median_filter
from scipy.stats import skew, kurtosis, iqr
from sklearn.linear_model import LinearRegression
from scipy.integrate import trapezoid
import warnings

def median_filter(x, size):
    assert size > 0 and size % 2 == 1
    x = np.asarray(x, dtype=float)
    n = len(x)
    h = size // 2
    out = np.empty(n, dtype=float)

    for i in range(n):
        w = x[max(0, i - h):min(n, i + h + 1)]
        nan_cnt = np.isnan(w).sum()
        if nan_cnt * 2 > len(w):  # 過半数がNaNならNaN
            out[i] = np.nan
        else:
            out[i] = np.nanmedian(w)
    return out


def plot_cvx_eda(eda, cvx_segment, name, median_window, median_threshold, output_path, Fs):
    """
    CVXEDAで分離したtonic/phasic成分をプロット
    (この関数は特徴量計算には直接関係しないため、内容は変更していません)
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    x = np.linspace(0, len(eda) / Fs / 60, len(eda))

    def get_ylim(data):
        max_val = np.nanmax(data)
        if max_val <= 1: return 1
        elif max_val <= 2: return 2
        elif max_val <= 5: return 5
        elif max_val <= 10: return 10
        elif max_val <= 15: return 15
        elif max_val <= 20: return 20
        elif max_val <= 30: return 30
        else: return int(np.ceil(max_val / 10) * 10)

    original_plotted = False
    tonic_plotted = False
    original_label = 'Original EDA' if not original_plotted else None
    axes[0].plot(x, eda,
                 color='green', label=original_label, linewidth=0.5, alpha=0.7)
    original_plotted = True
    tonic_label = 'Tonic' if not tonic_plotted else None
    axes[0].plot(x, cvx_segment['tonic'],
                 color='blue', label=tonic_label, linewidth=0.5)
    tonic_plotted = True
    ylim_original = get_ylim(np.concatenate([eda, cvx_segment['tonic']]))
    axes[0].set_ylabel("EDA (μS)")
    axes[0].set_ylim(0, ylim_original)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(f"Original EDA & Tonic: {name}")

    phasic_plotted = False
    label = 'Phasic' if not phasic_plotted else None
    axes[1].plot(x, cvx_segment['phasic'],
                 color='orange', label=label, linewidth=0.5)
    phasic_plotted = True
    ylim_phasic = get_ylim(cvx_segment['phasic'])
    axes[1].set_ylabel("Phasic (μS)")
    axes[1].set_xlabel("Time (minutes)")
    axes[1].set_ylim(0, ylim_phasic)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title(f"Phasic: {name}")
    plt.tight_layout()
    filename = f"{name}_cvxeda.svg"
    plt.savefig(f"{output_path}/{filename}")
    plt.close()

### <<< 追加: 周波数帯域パワー計算用のヘルパー関数
def get_band_power(freqs, psd, f_min, f_max):
    """PSDから指定した周波数帯域のパワー（面積）を計算する"""
    idx_band = np.where((freqs >= f_min) & (freqs < f_max))[0]
    if len(idx_band) == 0:
        return 0
    # 積分（面積を計算）
    return trapezoid(psd[idx_band], freqs[idx_band])


if __name__ == "__main__":
    
    # RuntimeWarningを非表示（0除算やNaNのmeanなど）
    warnings.filterwarnings('ignore', category=RuntimeWarning) ### <<< 追加

    data_path = f"/home/mori/projects/affective-forecast/datas/biometric_data"
    uuids = search_by_conditions({
        "ex-term": "term1",
    })
    datas = get_affective_datas_from_uuids(uuids)

    datas.sort(key=lambda x: int(x["filename"].split("_")[1]))
    data_names = [d["filename"] for d in datas]

    Fs = 4.0

    # medianフィルターの値は、以前決めた(33, 0.15)とする
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

            # NaN部分を線形補間で埋める（時系列を保持）
            valid_indices = ~np.isnan(eda)
            if np.sum(valid_indices) < 2:
                print("All nan")
                continue
            x_range = np.arange(len(eda))
            x_valid = x_range[valid_indices]    # 有効なデータのインデックス
            eda_valid = eda[valid_indices]  # 有効なデータの値
            eda = np.interp(x_range, x_valid, eda_valid)  # 全てのxに対して補間

            # 全体に対してCVXEDAを1回だけ適用（境界効果を避けるため）
            print(f"   Applying CVXEDA to full signal...")
            tonic_full, phasic_full = _eda_phasic_cvxeda(
                eda,
                sampling_rate=Fs,
            )

            # 各時間長でセグメントを切り出して特徴量計算
            lengthes = [900, 840, 780, 720, 660, 600, 540, 480, 420, 360, 300, 240, 180, 120, 60]
            for l in lengthes:
                sid = 3600 - int(4*l) ### <<< 変更 (4*lがfloatになる可能性を避ける)
                eda_seg = eda[sid:]
                phasic_seg = phasic_full[sid:]
                tonic_seg = tonic_full[sid:]

                feature = {}
                feature["name"] = name
                feature["length_seconds"] = l
                
                # 1階・2階微分
                D1 = np.diff(eda_seg)
                D2 = np.diff(D1)

                # --- 時間領域 ---
                feature["Mean"] = np.mean(eda_seg)
                feature["SD"] = np.std(eda_seg, ddof=1)

                feature["D1M"] = np.mean(D1) if len(D1) > 0 else np.nan
                feature["D2M"] = np.mean(D2) if len(D2) > 0 else np.nan

                feature["D1SD"] = np.std(D1, ddof=1) if len(D1) > 0 else np.nan
                feature["D2SD"] = np.std(D2, ddof=1) if len(D2) > 0 else np.nan

                feature["EDL"] = np.mean(tonic_seg)
                
                feature["DR"] = np.max(eda_seg) - np.min(eda_seg)
                feature["RMS"] = np.sqrt(np.mean(eda_seg ** 2))
                
                if feature["RMS"] == 0:
                    feature["PMRMSR"] = np.nan
                else:
                    feature["PMRMSR"] = np.max(eda_seg) / feature["RMS"]
                    
                feature["RSSL"] = np.sqrt(np.sum(eda_seg ** 2))

                # --- Phasic/Peak関連 (nk.eda_peaks) ---
                try:
                    # Phasic成分の平均と標準偏差
                    feature["PHVM"] = np.mean(phasic_seg)
                    feature["PHVSD"] = np.std(phasic_seg, ddof=1)
                    
                    # Phasicセグメントからピーク情報を抽出
                    peaks_idx, info = nk.eda_peaks(phasic_seg, sampling_rate=Fs, method="neurokit", amplitude_min=0.1)

                    rise_times = info.get('SCR_RiseTime', [])
                    recovery_times = info.get('SCR_RecoveryTime', [])
                    amplitudes = info.get('SCR_Amplitude', [])

                    # ピークが1つ以上検出された場合
                    if len(amplitudes) > 0:
                        # Phasic成分の総立ち上がり/立ち下がり時間
                        feature["SRT"] = np.sum(rise_times)
                        feature["SFT"] = np.sum(recovery_times) # SFT (Sum Fall Time) を RecoveryTime で代用
                        
                        # 上昇速度 (Amplitude / RiseTime)
                        rise_rates = np.divide(amplitudes, rise_times, where=np.array(rise_times) > 0)
                        rise_rates_finite = rise_rates[np.isfinite(rise_rates)]
                        feature["RM"] = np.mean(rise_rates_finite) if len(rise_rates_finite) > 0 else 0.0

                        # 上昇速度の標準偏差
                        feature["RRSTD"] = np.std(rise_rates_finite, ddof=1) if len(rise_rates_finite) > 1 else 0.0
                        
                        # 下降速度 (Amplitude / RecoveryTime)
                        decay_rates = np.divide(amplitudes, recovery_times, where=np.array(recovery_times) > 0)
                        decay_rates_finite = decay_rates[np.isfinite(decay_rates)]
                        feature["DCRM"] = np.mean(decay_rates_finite) if len(decay_rates_finite) > 0 else 0.0
                        
                        # 下降速度の標準偏差
                        feature["DCRSD"] = np.std(decay_rates_finite, ddof=1) if len(decay_rates_finite) > 1 else 0.0

                        # PNN50 (ピーク間隔が50msより大きいものの数)
                        if len(peaks_idx) > 1:
                            peak_intervals_sec = np.diff(peaks_idx) / Fs
                            feature["PNN50"] = np.sum(peak_intervals_sec > 0.050)
                        else:
                            feature["PNN50"] = 0
                    
                    # ピークが検出されなかった場合
                    else:
                        feature["SRT"] = 0.0
                        feature["SFT"] = 0.0
                        feature["RM"] = 0.0
                        feature["RRSTD"] = 0.0
                        feature["DCRM"] = 0.0
                        feature["DCRSD"] = 0.0
                        feature["PNN50"] = 0

                except Exception as e:
                    print(f"    Error in Phasic/Peak features: {e}")
                    feature["PHVM"], feature["PHVSD"], feature["SRT"], feature["SFT"] = np.nan, np.nan, np.nan, np.nan
                    feature["RM"], feature["RRSTD"], feature["DCRM"], feature["DCRSD"], feature["PNN50"] = np.nan, np.nan, np.nan, np.nan, np.nan


                # --- 形態的特徴 ---
                try:
                    # AL (アーク長)
                    dt = 1.0 / Fs
                    dy = D1
                    feature["AL"] = np.sum(np.sqrt(dt**2 + dy**2))
                    
                    # IN (積分面積)
                    feature["IN"] = trapezoid(eda_seg, dx=dt)
                    
                    # AP (正規化された平均パワー -> 平均二乗)
                    feature["AP"] = np.mean(eda_seg ** 2)
                    
                    # IL (周囲長対面積比)
                    feature["IL"] = feature["AL"] / feature["IN"] if feature["IN"] != 0 else np.nan
                    
                    # EL (エネルギー対周囲長比)
                    energy = np.sum(eda_seg ** 2)
                    feature["EL"] = energy / feature["AL"] if feature["AL"] != 0 else np.nan
                    
                    # EN (エントロピー -> サンプルエントロピー)
                    en_result = nk.entropy_sample(eda_seg)
                    feature["EN"] = en_result[0] if isinstance(en_result, tuple) else en_result

                except Exception as e:
                    print(f"    Error in Morphological features: {e}")
                    feature["AL"], feature["IN"], feature["AP"], feature["IL"], feature["EL"], feature["EN"] = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan


                # --- 統計的特徴 ---
                feature["MedVal"] = np.median(eda_seg)
                feature["Var"] = np.var(eda_seg, ddof=1)
                feature["KU"] = kurtosis(eda_seg)
                feature["SKU"] = skew(eda_seg)
                # MO (モーメント) -> 5次の中心積率を例として計算
                feature["MO"] = np.mean((eda_seg - feature["Mean"])**5)


                # --- 周波数領域 (PSD) ---
                # Phasic成分のパワースペクトル密度(PSD)をWelch法で計算
                try:
                    # セグメントが短すぎる場合や定数値の場合にwelchが失敗することがある
                    # nperseg: 窓長。信号長より長くはできない。ここでは最大60秒(240点)または信号長
                    nperseg = min(len(phasic_seg), int(Fs * 60)) 
                    if nperseg < 2: # 窓長が短すぎる
                        raise ValueError("Segment too short for PSD")
                        
                    freqs, psd = signal.welch(phasic_seg, fs=Fs, nperseg=nperseg, window='hann')
                    
                    # 論文 [113, 126] に基づく帯域パワー
                    p_band1 = get_band_power(freqs, psd, 0.02, 0.25)
                    p_band2 = get_band_power(freqs, psd, 0.25, 0.40)
                    p_band3 = get_band_power(freqs, psd, 0.40, 1.0)

                    # 各バンドのパワーを個別に保存
                    feature["PSD_Band1"] = p_band1  # 0.02-0.25 Hz
                    feature["PSD_Band2"] = p_band2  # 0.25-0.40 Hz
                    feature["PSD_Band3"] = p_band3  # 0.40-1.0 Hz

                    # SP (Spectral Power) -> 論文[26, 59]の帯域の合計パワー
                    feature["SP"] = get_band_power(freqs, psd, 0.1, 0.4)

                    # SSP (Sum Spectral Power) -> PSD帯域の合計パワー
                    feature["SSP"] = p_band1 + p_band2 + p_band3

                    # MSSP (Mean Spectral Power) -> PSD全体の平均
                    feature["MSSP"] = np.mean(psd)

                except Exception as e:
                    print(f"    Error in Frequency features: {e}")
                    feature["PSD_Band1"], feature["PSD_Band2"], feature["PSD_Band3"] = np.nan, np.nan, np.nan
                    feature["SP"], feature["SSP"], feature["MSSP"] = np.nan, np.nan, np.nan


                # --- 時間周波数領域 (STFT) ---
                try:
                    # STFTを計算 (窓長10秒 = 40点)
                    nperseg_stft = min(len(eda_seg), int(Fs * 10))
                    if nperseg_stft < 2:
                        raise ValueError("Segment too short for STFT")
                        
                    f_stft, t_stft, Zxx = signal.stft(eda_seg, fs=Fs, nperseg=nperseg_stft, window='hann')
                    
                    # パワースペクトログラム
                    Sxx = np.abs(Zxx)**2
                    
                    # TFEnergy (時間周波数エネルギー)
                    feature["TFEnergy"] = np.sum(Sxx)

                    # TFFlux (時間周波数フラックス)
                    # 各時間スライスで正規化 (合計が0のスライスでの0除算を回避)
                    slice_sums = np.sum(Sxx, axis=0, keepdims=True)
                    norm_Sxx = np.divide(Sxx, slice_sums, where=slice_sums > 0)
                    # 時間フレーム間の差の二乗和平方根
                    flux = np.sqrt(np.sum(np.diff(norm_Sxx, axis=1)**2, axis=0))
                    feature["TFFlux"] = np.mean(flux[np.isfinite(flux)]) if len(flux) > 0 else 0.0
                    
                    # TFFlatness (時間周波数フラットネス)
                    # 0の対数を避けるために微小量(1e-10)を加算
                    gmean = np.exp(np.mean(np.log(Sxx + 1e-10), axis=0))
                    amean = np.mean(Sxx, axis=0)
                    flatness = np.divide(gmean, amean, where=amean > 0)
                    feature["TFFlatness"] = np.mean(flatness[np.isfinite(flatness)]) if len(flatness) > 0 else 0.0
                    
                    # Eshannon (シャノンエントロピー)
                    # `nk.entropy_shannon` は確率分布を期待するため、信号に直接適用するのは厳密には異なる
                    # ここでは論文[80]の文脈を考慮し、信号の複雑さとして nk.entropy_shannon を使う
                    shannon_entropy_tuple = nk.entropy_shannon(eda_seg)
                    feature["Eshannon"] = shannon_entropy_tuple[0] if isinstance(shannon_entropy_tuple, tuple) else shannon_entropy_tuple
                    
                    # Elog (対数エントロピー)
                    if feature["Eshannon"] is not np.nan and feature["Eshannon"] > 0:
                        feature["Elog"] = np.log(feature["Eshannon"])
                    else:
                        feature["Elog"] = 0.0

                except Exception as e:
                    print(f"    Error in Time-Frequency features: {e}")
                    feature["TFFlux"], feature["TFFlatness"], feature["TFEnergy"] = np.nan, np.nan, np.nan
                    feature["Eshannon"], feature["Elog"] = np.nan, np.nan


                features_list.append(feature)

    # 特徴量をCSVに保存
    print(f"\nSaving {len(features_list)} features to CSV...")
    df = pd.DataFrame(features_list)
    output_dir = "/home/mori/projects/affective-forecast/08_CVX分割適用/features"
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/eda_features_completed.csv"
    df.to_csv(output_file, index=False)
    print(f"Saved to {output_file}")
    print(f"Total features: {len(df)}")
    print(f"Feature columns: {len(df.columns)}")