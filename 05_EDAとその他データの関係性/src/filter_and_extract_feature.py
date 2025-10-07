import os
from util.utils3 import load_h5_data, search_and_get_filenames, get_affective_datas_from_uuids, search_by_conditions
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import neurokit2 as nk
from filter_definition import rri_filter, conditional_median_filter
from scipy import signal
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

def write_arousal_correlation(output_path, df):
    # 特徴量とarousalの相関分析（CSV読み込み後に実行）
    feature_cols_temp = [col for col in df.columns if col not in ["filename", "arousal", "arousal_label"]]
    arousal_correlations = []
    print(df.columns)
    for col in feature_cols_temp:
        corr = df[col].corr(df['arousal'])
        arousal_correlations.append({'feature': col, 'correlation': corr})

    corr_df = pd.DataFrame(arousal_correlations)
    corr_df['abs_correlation'] = corr_df['correlation'].abs()
    corr_df = corr_df.sort_values('abs_correlation', ascending=False)

    # テキストファイルに保存
    correlation_log_path = f"{output_path}/Arousal/arousal_feature_correlation.txt"
    with open(correlation_log_path, "w", encoding="utf-8") as f:
        f.write("特徴量とArousalの相関分析\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"総特徴量数: {len(feature_cols_temp)}\n")
        f.write(f"{'特徴量':<25} {'相関係数':>15} {'絶対値':>12}\n")
        f.write("-" * 70 + "\n")
        for _, row in corr_df.iterrows():
            f.write(f"{row['feature']:<25} {row['correlation']:>15.6f} {row['abs_correlation']:>12.6f}\n")
        f.write("\n" + "=" * 70 + "\n")
        f.write(f"最も強い相関: {corr_df.iloc[0]['feature']} ({corr_df.iloc[0]['correlation']:.6f})\n")
        f.write(f"最も弱い相関: {corr_df.iloc[-1]['feature']} ({corr_df.iloc[-1]['correlation']:.6f})\n")
    print(f"Feature correlation log saved to {correlation_log_path}")


    # 生データを散布図に保存
    scatter_plot_dir = f"{output_path}/Arousal/生の散布図"
    os.makedirs(scatter_plot_dir, exist_ok=True)
    for col in feature_cols_temp:
        plt.figure(figsize=(8, 6))
        # ジッターを追加（Arousal軸）
        jitter = np.random.normal(0, 0.1, size=len(df))
        plt.scatter(df[col], df['arousal'] + jitter, alpha=0.5, s=10)
        plt.xlabel(col)
        plt.ylabel('Arousal')
        corr = df[col].corr(df['arousal'])
        plt.title(f'r={corr:.3f}')
        plt.tight_layout()
        plt.savefig(f"{scatter_plot_dir}/{col}.svg")
        plt.close()
    print(f"Scatter plots saved to {scatter_plot_dir}")


    # リッジ回帰による予測スコアを特徴量ごとに可視化
    feature_cols = [col for col in df.columns if col not in ["filename", "arousal", "valence", "arousal_label"]]
    X = df[feature_cols].values
    y = df["arousal"].values
    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    ridge = Ridge().fit(X_train_scaled, y_train)
    print("Training set score: {:.2f}".format(ridge.score(X_train_scaled, y_train)))
    print("Test set score: {:.2f}".format(ridge.score(X_test_scaled, y_test)))



def write_valence_correlation(output_path, df):
    # 特徴量とvalenceの相関分析（CSV読み込み後に実行）
    feature_cols_temp = [col for col in df.columns if col not in ["filename", "arousal", "arousal_label"]]
    valence_correlations = []
    print(df.columns)
    for col in feature_cols_temp:
        corr = df[col].corr(df['valence'])
        valence_correlations.append({'feature': col, 'correlation': corr})

    corr_df = pd.DataFrame(valence_correlations)
    corr_df['abs_correlation'] = corr_df['correlation'].abs()
    corr_df = corr_df.sort_values('abs_correlation', ascending=False)

    # 相関分析結果をテキストファイルに保存
    correlation_log_path = f"{output_path}/valence_feature_correlation.txt"
    with open(correlation_log_path, "w", encoding="utf-8") as f:
        f.write("特徴量とValenceの相関分析\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"総特徴量数: {len(feature_cols_temp)}\n")

        f.write(f"{'特徴量':<25} {'相関係数':>15} {'絶対値':>12}\n")
        f.write("-" * 70 + "\n")
        for _, row in corr_df.iterrows():
            f.write(f"{row['feature']:<25} {row['correlation']:>15.6f} {row['abs_correlation']:>12.6f}\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write(f"最も強い相関: {corr_df.iloc[0]['feature']} ({corr_df.iloc[0]['correlation']:.6f})\n")
        f.write(f"最も弱い相関: {corr_df.iloc[-1]['feature']} ({corr_df.iloc[-1]['correlation']:.6f})\n")

    print(f"Feature correlation log saved to {correlation_log_path}")

if __name__ == "__main__":

    data_path = f"/home/mori/projects/affective-forecast/datas/biometric_data"
    output_path = f"/home/mori/projects/affective-forecast/plots/EDA特徴量と覚醒度・感情価の相関"
    os.makedirs(f"{output_path}/Arousal", exist_ok=True)
    os.makedirs(f"{output_path}/Valence", exist_ok=True)

    # 特徴量CSVのパス
    feature_csv_path = f"{output_path}/features.csv"

    # 既存のCSVがあるかチェック
    if os.path.exists(feature_csv_path):
        print(f"Loading existing features from {feature_csv_path}")
        df = pd.read_csv(feature_csv_path)
    else:
        print("Extracting features...")

        uuids = search_by_conditions({
            "ex-term": "term1"
        })
        datas = get_affective_datas_from_uuids(uuids)

        # ソートしてから辞書に変換（ファイル名をキーにする）
        datas.sort(key=lambda x: int(x["filename"].split("_")[1]))
        data_names = [d["filename"] for d in datas]
        arousal_dict = {d["filename"]: d["arousal"] for d in datas}
        valence_dict = {d["filename"]: d["valence"] for d in datas}

        # パラメータ設定
        valid_eda_seconds = 30 # 有効なデータのうち、最後尾の30秒区間をEDA特徴量抽出に使う（15minの時点が、被験者が感情入力をしたタイミングのため）

        Fs = 4
        eda_lowpass_cutoff = 0.35
        rri_lowpass_cutoff = 0.1
        rri_threshold = 0.00015
        min_duration_seconds = 10.0
        min_samples = int(min_duration_seconds * Fs)
        median_kernel_size = 4 * Fs + 1
        p = 0.2

        # 各データの最大有効連続時間を記録するリスト
        max_valid_durations = []

        features = []
        excluded_data = []  # 除外されたデータを記録

        for i, name in enumerate(data_names):
            eda_file_name = f"{name}_eda.h5"
            rri_file_name = f"{name}_rri.h5"
            eda = load_h5_data(f"{data_path}/{eda_file_name}")
            rri = load_h5_data(f"{data_path}/{rri_file_name}")

            if eda is not None and rri is not None:
                # EDAデータの取得とローパスフィルタ適用
                eda_data = np.asarray(eda[0], dtype=float)
                b, a = signal.butter(4, eda_lowpass_cutoff, btype='low')
                eda_lowpass = signal.filtfilt(b, a, eda_data)

                # RRIデータの取得とローパスフィルタ適用
                rri_data = np.asarray(rri[0], dtype=float)
                b, a = signal.butter(4, rri_lowpass_cutoff, btype='low')
                rri_lowpass = signal.filtfilt(b, a, rri_data)

                # rri_filterを適用
                eda_rri_filtered = rri_filter(eda_lowpass, rri_lowpass, rri_threshold, min_samples)
                rri_mask = np.isnan(eda_rri_filtered)

                # conditional_median_filterを適用
                eda_median_filtered = conditional_median_filter(eda_lowpass, median_kernel_size, p)
                median_mask = np.isnan(eda_median_filtered)

                # 2つのマスクをORで結合（True = 除去）
                combined_mask = rri_mask | median_mask

                # 有効データマスク（False = 有効）
                valid_mask = ~combined_mask

                current_count = 0
                eda_range = []
                importance = 0

                # validなvalid_eda_seconds秒間の連続データのうち、最も末尾に近いものを取得する
                # 各連続データの最後のサンプルの末尾からの距離をimportanceという変数で保持する
                for i, valid in enumerate(valid_mask[::-1]):
                    if valid:
                        current_count = current_count+1
                        if current_count >= valid_eda_seconds*Fs:
                            start_idx = len(valid_mask) - i - 1
                            eda_range = eda_data[start_idx:start_idx + valid_eda_seconds*Fs]
                            importance = i - current_count + 1
                            break
                    else:
                        current_count = 0
                        continue

                if len(eda_range)==0:#有効なデータがなかった時
                    excluded_data.append({
                        "filename": name,
                        "reason": "有効な連続データ不足",
                        "detail": f"{valid_eda_seconds}秒間の有効な連続データが見つかりませんでした"
                    })
                    continue
                if np.sum(eda_range)<0.01:#data1593のようなゼロのみのデータを除外
                    excluded_data.append({
                        "filename": name,
                        "reason": "ゼロのみのデータ",
                        "detail": f"np.sum(eda_range) = {np.sum(eda_range):.6f} < 0.01"
                    })
                    continue
                if eda.shape[1] < 3000:#data2711のようなデータを除外するため
                    excluded_data.append({
                        "filename": name,
                        "reason": "データ長不足",
                        "detail": f"eda.shape[1] = {eda.shape[1]} < 3000"
                    })
                    continue

                signals = nk.eda_phasic(eda_range, sampling_rate=Fs, method="cvxeda")
                signals_peaks, info_peaks = nk.eda_peaks(eda_range, sampling_rate=Fs)

                tonic = signals['EDA_Tonic'].to_numpy()
                phasic = signals['EDA_Phasic'].to_numpy()

                # rawの統計
                raw_mean = float(np.mean(eda_range))
                raw_std  = float(np.std(eda_range, ddof=1))
                rms      = float(np.sqrt(np.mean(eda_range**2)))

                # 差分の統計
                d1 = np.diff(eda_range)
                d2 = np.diff(d1) if d1.size > 1 else np.array([0.0])
                D1_mean = float(np.mean(d1)) if d1.size else 0.0
                D1_std  = float(np.std(d1, ddof=1)) if d1.size > 1 else 0.0
                D2_mean = float(np.mean(d2)) if d2.size else 0.0
                D2_std  = float(np.std(d2, ddof=1)) if d2.size > 1 else 0.0

                # トニック(EDL)の統計
                edl_mean = float(np.mean(tonic))
                edl_std  = float(np.std(tonic, ddof=1))

                # Phasic成分の統計
                phasic_mean = float(np.mean(phasic))
                phasic_std  = float(np.std(phasic, ddof=1))

                # SCR由来の特徴（info_peaksから取得）
                scr_risetime = info_peaks["SCR_RiseTime"]
                scr_amp      = info_peaks["SCR_Amplitude"]
                # scr_recovery = info_peaks["SCR_RecoveryTime"]
                scr_count    = len(info_peaks["SCR_Peaks"])

                # SCR統計値
                sum_rize_time = float(np.nansum(scr_risetime))
                scr_amp_mean  = float(np.nanmean(scr_amp)) if len(scr_amp) > 0 else 0.0
                scr_amp_max   = float(np.nanmax(scr_amp)) if len(scr_amp) > 0 else 0.0
                scr_amp_min   = float(np.nanmin(scr_amp)) if len(scr_amp) > 0 else 0.0
                # scr_recovery_mean = float(np.nanmean(scr_recovery)) if len(scr_recovery) > 0 else 0.0

                # 代入
                feature_object = {}
                feature_object["raw_mean"]          = raw_mean
                feature_object["raw_std"]           = raw_std
                feature_object["rms"]               = rms
                feature_object["D1_mean"]           = D1_mean
                feature_object["D1_std"]            = D1_std
                feature_object["D2_mean"]           = D2_mean
                feature_object["D2_std"]            = D2_std
                feature_object["edl_mean"]          = edl_mean
                feature_object["edl_std"]           = edl_std
                feature_object["phasic_mean"]       = phasic_mean
                feature_object["phasic_std"]        = phasic_std
                feature_object["scr_count"]         = scr_count
                feature_object["sum_rize_time"]     = sum_rize_time
                feature_object["scr_amp_mean"]      = scr_amp_mean
                feature_object["scr_amp_max"]       = scr_amp_max
                feature_object["scr_amp_min"]       = scr_amp_min
                # feature_object["scr_recovery_mean"] = scr_recovery_mean
                feature_object["importance"]        = int(importance)
                feature_object["filename"]          = name
                feature_object["arousal"]           = arousal_dict[name]
                feature_object["valence"]           = valence_dict[name]

                features.append(feature_object)
                print(f"created features array length:{len(features)}")


        df = pd.DataFrame(features)
        # arousalの正負ラベル（正: 1, 負: 0）
        df["arousal_label"] = (df["arousal"] > 0).astype(int)
        df["valence_label"] = (df["valence"] > 0).astype(int)
        # CSVとして保存
        df.to_csv(feature_csv_path, index=False)
        print(f"Features saved to {feature_csv_path}")
        # 除外データをテキストファイルに保存
        excluded_log_path = f"{output_path}/excluded_data.txt"
        with open(excluded_log_path, "w", encoding="utf-8") as f:
            f.write("除外データ一覧\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"総除外データ数: {len(excluded_data)}\n\n")
            for i, item in enumerate(excluded_data, 1):
                f.write(f"[{i}] {item['filename']}\n")
                f.write(f"    理由: {item['reason']}\n")
                f.write(f"    詳細: {item['detail']}\n")
                f.write("\n")
        print(f"Excluded data log saved to {excluded_log_path}")


    write_arousal_correlation(output_path, df)

    # # 特徴量列を取得
    # feature_cols = [col for col in df.columns if col not in ["filename", "arousal", "valence", "arousal_label"]]

    # X = df[feature_cols].values
    # y = df["arousal_label"].values

    # # train/test split
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # # 標準化
    # scaler = StandardScaler()
    # X_train_scaled = scaler.fit_transform(X_train)
    # X_test_scaled = scaler.transform(X_test)

    # # SVC学習
    # svc = SVC(C=10,gamma=0.5)
    # svc.fit(X_train_scaled, y_train)

    # print(f"\nTotal samples: {len(df)}")
    # print(f"Positive arousal: {(y == 1).sum()}, Negative arousal: {(y == 0).sum()}")
    # print(f"Accuracy on training set: {svc.score(X_train_scaled, y_train):.2f}")
    # print(f"Accuracy on test set: {svc.score(X_test_scaled, y_test):.2f}")
