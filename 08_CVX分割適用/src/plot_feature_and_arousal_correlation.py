import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from util.utils3 import search_by_conditions, get_affective_datas_from_uuids

if __name__ == "__main__":
    # 特徴量CSVを読み込み
    feature_file = "/home/mori/projects/affective-forecast/08_CVX分割適用/features/eda_features_completed.csv"
    df = pd.read_csv(feature_file)

    print(f"Loaded {len(df)} feature rows")
    print(f"Columns: {len(df.columns)} columns")

    # Arousalデータを取得
    print("\nLoading arousal data...")
    data_path = "/home/mori/projects/affective-forecast/datas/biometric_data"
    uuids = search_by_conditions({
        "ex-term": "term1",
    })
    datas = get_affective_datas_from_uuids(uuids)

    # データ名でソート（特徴量と同じ順序）
    datas.sort(key=lambda x: int(x["filename"].split("_")[1]))
    data_names = [d["filename"] for d in datas]

    # filename -> arousal のマッピングを作成
    filename_to_arousal = {d["filename"]: d["arousal"] for d in datas}

    print(f"Loaded arousal data for {len(datas)} files")
    print(f"Arousal range: {min(filename_to_arousal.values())} - {max(filename_to_arousal.values())}")

    # 特徴量に Arousal を追加
    df["Arousal"] = df["name"].map(filename_to_arousal)

    # マッピングできなかった行をチェック
    missing_arousal = df["Arousal"].isna().sum()
    if missing_arousal > 0:
        print(f"Warning: {missing_arousal} rows have missing arousal values")
        # Arousalが無い行を除外
        df = df[df["Arousal"].notna()]
        print(f"After filtering: {len(df)} rows")

    # メタデータ以外の全カラムを数値型に変換
    meta_columns = ['name', 'length_seconds']
    for col in df.columns:
        if col not in meta_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 特徴量カラム（メタデータ以外、Arousalを含む）
    feature_columns = [col for col in df.columns if col not in meta_columns]

    print(f"\nFeature columns (including Arousal): {len(feature_columns)}")

    # 出力先ディレクトリ
    csv_output_dir = "/home/mori/projects/affective-forecast/08_CVX分割適用/plots/raw_feature_and_arousal_correlation"
    heatmap_output_dir = "/home/mori/projects/affective-forecast/08_CVX分割適用/plots/raw_feature_and_arousal_correlation_heatmap"
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
        plt.figure(figsize=(18, 16))
        sns.heatmap(correlation_matrix, cmap='coolwarm', center=0, vmin=-1, vmax=1,
                    square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                    annot=False)  # 特徴量が多いのでannotationは無し
        plt.title(f'Raw Feature & Arousal Correlation Heatmap (Length={l}s, n={len(data_subset)})', fontsize=14)
        plt.tight_layout()
        heatmap_file = f"{heatmap_output_dir}/raw_correlation_heatmap_{l}s.png"
        plt.savefig(heatmap_file, dpi=150)
        plt.close()

        print(f"  CSV saved to {csv_file}")
        print(f"  Heatmap saved to {heatmap_file}")
        print(f"  Matrix size: {correlation_matrix.shape[0]}x{correlation_matrix.shape[1]}")
        print(f"  Sample count: {len(data_subset)}")

        # Arousalとの相関を抽出して表示
        if "Arousal" in correlation_matrix.columns:
            arousal_corr = correlation_matrix["Arousal"].drop("Arousal", errors='ignore')
            arousal_corr_sorted = arousal_corr.abs().sort_values(ascending=False)

            print(f"\n  Top 10 features correlated with Arousal:")
            for i, (feat, corr_abs) in enumerate(arousal_corr_sorted.head(10).items()):
                corr_val = arousal_corr[feat]
                if not np.isnan(corr_val):
                    print(f"    {i+1}. {feat}: {corr_val:.4f}")

            # 相関が弱い特徴量も表示
            print(f"\n  Bottom 5 features (weakest correlation with Arousal):")
            for i, (feat, corr_abs) in enumerate(arousal_corr_sorted.tail(5).items()):
                corr_val = arousal_corr[feat]
                if not np.isnan(corr_val):
                    print(f"    {feat}: {corr_val:.4f}")

    print("\n=== All correlation matrices with Arousal saved! ===")
    print(f"CSV output directory: {csv_output_dir}")
    print(f"Heatmap output directory: {heatmap_output_dir}")
