import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":
    # 特徴量CSVを読み込み
    feature_file = "/home/mori/projects/affective-forecast/08_CVX分割適用/features/eda_features_completed.csv"
    df = pd.read_csv(feature_file)

    print(f"Loaded {len(df)} feature rows")
    print(f"Columns: {len(df.columns)} columns")

    # メタデータ以外の全カラムを数値型に変換
    meta_columns = ['name', 'length_seconds']
    for col in df.columns:
        if col not in meta_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 特徴量カラム（メタデータ以外）
    feature_columns = [col for col in df.columns if col not in meta_columns]

    # 出力先ディレクトリ（plotsフォルダ配下に変更）
    csv_output_dir = "/home/mori/projects/affective-forecast/08_CVX分割適用/plots/raw_feature_correlation"
    heatmap_output_dir = "/home/mori/projects/affective-forecast/08_CVX分割適用/plots/raw_feature_correlation_heatmap"
    os.makedirs(csv_output_dir, exist_ok=True)
    os.makedirs(heatmap_output_dir, exist_ok=True)

    # 各時間長ごとに相関行列を計算
    lengthes = [900, 840, 780, 720, 660, 600, 540, 480, 420, 360, 300, 240, 180, 120, 60]

    for l in lengthes:
        print(f"\nProcessing correlation for length={l}s...")

        # この時間長のデータを抽出
        data_subset = df[df['length_seconds'] == l][feature_columns]

        # 相関行列を計算（ピアソン相関）
        # NaNを含む列は自動的に除外される
        correlation_matrix = data_subset.corr(method='pearson')

        # CSVに保存
        csv_file = f"{csv_output_dir}/raw_correlation_length_{l}s.csv"
        correlation_matrix.to_csv(csv_file)

        # ヒートマップを作成
        plt.figure(figsize=(16, 14))
        sns.heatmap(correlation_matrix, cmap='coolwarm', center=0, vmin=-1, vmax=1,
                    square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title(f'Raw Feature Correlation Heatmap (Length={l}s, n={len(data_subset)})', fontsize=14)
        plt.tight_layout()
        heatmap_file = f"{heatmap_output_dir}/raw_correlation_heatmap_{l}s.png"
        plt.savefig(heatmap_file, dpi=150)
        plt.close()

        print(f"  CSV saved to {csv_file}")
        print(f"  Heatmap saved to {heatmap_file}")
        print(f"  Matrix size: {correlation_matrix.shape[0]}x{correlation_matrix.shape[1]}")
        print(f"  Sample count: {len(data_subset)}")

        # 統計情報を出力
        # 対角要素を除いた相関係数の統計
        corr_values = correlation_matrix.values
        # 上三角のみ取得（対角除く）
        upper_triangle = corr_values[np.triu_indices_from(corr_values, k=1)]
        valid_corrs = upper_triangle[~np.isnan(upper_triangle)]

        if len(valid_corrs) > 0:
            print(f"  Correlation stats:")
            print(f"    Mean: {np.mean(valid_corrs):.4f}")
            print(f"    Std:  {np.std(valid_corrs):.4f}")
            print(f"    Min:  {np.min(valid_corrs):.4f}")
            print(f"    Max:  {np.max(valid_corrs):.4f}")

            # 高相関ペアを表示（|r| > 0.9）
            high_corr_pairs = []
            for i in range(len(correlation_matrix)):
                for j in range(i+1, len(correlation_matrix)):
                    corr_val = correlation_matrix.iloc[i, j]
                    if not np.isnan(corr_val) and abs(corr_val) > 0.9:
                        high_corr_pairs.append((
                            correlation_matrix.index[i],
                            correlation_matrix.columns[j],
                            corr_val
                        ))

            if high_corr_pairs:
                print(f"  High correlation pairs (|r| > 0.9): {len(high_corr_pairs)}")
                for feat1, feat2, corr in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True)[:5]:
                    print(f"    {feat1} <-> {feat2}: {corr:.4f}")

    print("\n=== All correlation matrices saved! ===")
    print(f"CSV output directory: {csv_output_dir}")
    print(f"Heatmap output directory: {heatmap_output_dir}")
