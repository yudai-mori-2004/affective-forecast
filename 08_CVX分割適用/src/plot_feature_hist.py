import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    # 特徴量CSVを読み込み（再計算後のクリーンなCSV）
    feature_file = "/home/mori/projects/affective-forecast/08_CVX分割適用/features/eda_features_completed.csv"
    df = pd.read_csv(feature_file)

    print(f"Loaded {len(df)} feature rows")
    print(f"Columns: {len(df.columns)} columns")

    # メタデータ以外の全カラムを数値型に変換
    meta_columns = ['name', 'length_seconds']
    for col in df.columns:
        if col not in meta_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            valid_count = df[col].notna().sum()
            print(f"  {col}: {valid_count}/{len(df)} valid values")

    # プロットする特徴量（メタデータ以外）
    feature_columns = [col for col in df.columns if col not in meta_columns]

    # 各特徴量についてヒストグラムを作成
    lengthes = [900, 840, 780, 720, 660, 600, 540, 480, 420, 360, 300, 240, 180, 120, 60]

    for feature_name in feature_columns:
        # 出力先ディレクトリ（特徴量ごと）
        output_dir = f"/home/mori/projects/affective-forecast/08_CVX分割適用/plots/raw_feature_hist/{feature_name}"
        os.makedirs(output_dir, exist_ok=True)

        # この特徴量の全データ（全長さ）から範囲を計算
        all_data = df[feature_name].replace([np.inf, -np.inf], np.nan).dropna()
        if len(all_data) == 0:
            print(f"Skipping {feature_name}: no valid data")
            continue

        for l in lengthes:
            # ヒストグラム描画（NaN、inf、-infを除外）
            data = df[df['length_seconds'] == l][feature_name].replace([np.inf, -np.inf], np.nan).dropna()

            if len(data) == 0:
                continue

            plt.figure(figsize=(10, 6))

            # この長さのデータに最適化された範囲を計算
            data_min, data_max = data.min(), data.max()

            # データが定数の場合
            if data_min == data_max:
                data_range = (data_min - 0.5, data_max + 0.5)
            else:
                data_margin = (data_max - data_min) * 0.05
                data_range = (data_min - data_margin, data_max + data_margin)

            # ビン数を適応的に設定（Sturgesの公式）
            n_bins = max(10, min(50, int(np.ceil(np.log2(len(data)) + 1))))

            # ビンエッジを計算
            bin_edges = np.linspace(data_range[0], data_range[1], n_bins + 1)

            plt.hist(data, bins=bin_edges, color='blue', alpha=0.7, edgecolor='black')

            plt.xlabel(feature_name)
            plt.ylabel('Frequency')
            plt.title(f'Histogram of raw_{feature_name} (Length={l}s, n={len(data)})')
            plt.xlim(data_range)

            # x軸の目盛りをビンの境界に合わせる（最大10個まで表示）
            if n_bins <= 10:
                plt.xticks(bin_edges)
            else:
                # ビンが多い場合は、適当な間隔で目盛りを表示
                tick_step = max(1, n_bins // 10)
                plt.xticks(bin_edges[::tick_step])

            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()

            # 保存（ファイル名で長さを区別）
            output_file = f"{output_dir}/raw_length_{l}.png"
            plt.savefig(output_file, dpi=100)

            plt.close()

        print(f"Saved histograms for {feature_name}")

    print("\nAll histograms completed!")
