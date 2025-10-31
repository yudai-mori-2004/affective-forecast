import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    # 特徴量CSVを読み込み
    feature_file = "/home/mori/projects/affective-forecast/08_CVX分割適用/features/eda_features.csv"
    df = pd.read_csv(feature_file)

    print(f"Loaded {len(df)} feature rows")
    print(f"Columns: {df.columns.tolist()}")

    # 出力先ディレクトリ
    output_dir = "/home/mori/projects/affective-forecast/08_CVX分割適用/plots/feature_scatter"
    os.makedirs(output_dir, exist_ok=True)

    # プロットする特徴量（メタデータ以外）
    meta_columns = ['name', 'length_seconds']
    feature_columns = [col for col in df.columns if col not in meta_columns]

    # 1. 横軸：length_seconds の散布図
    print("\n=== Plotting vs Length ===")
    output_dir_length = f"{output_dir}/vs_length"
    os.makedirs(output_dir_length, exist_ok=True)

    for feature_name in feature_columns:
        print(f"Plotting {feature_name} vs length...")

        plt.figure(figsize=(10, 6))

        plt.scatter(df['length_seconds'],
                   df[feature_name],
                   c='blue',
                   marker='o',
                   alpha=0.6,
                   s=30)

        plt.xlabel('Length (seconds)')
        plt.ylabel(feature_name)
        plt.title(f'{feature_name} vs Length')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        output_file = f"{output_dir_length}/{feature_name}.png"
        plt.savefig(output_file, dpi=100)
        plt.close()

    print(f"Saved {len(feature_columns)} scatter plots to {output_dir_length}")

    # 2. 横軸：name（被験者）の散布図
    print("\n=== Plotting vs Name ===")
    output_dir_name = f"{output_dir}/vs_name"
    os.makedirs(output_dir_name, exist_ok=True)

    # nameをカテゴリカル変数として扱う（整数部分でソート）
    unique_names = df['name'].unique()
    # data_{整数}_E4 形式から整数を抽出してソート
    unique_names_sorted = sorted(unique_names, key=lambda x: int(x.split('_')[1]))
    name_to_idx = {name: i for i, name in enumerate(unique_names_sorted)}
    df['name_idx'] = df['name'].map(name_to_idx)

    for feature_name in feature_columns:
        print(f"Plotting {feature_name} vs name...")

        plt.figure(figsize=(12, 6))

        plt.scatter(df['name_idx'],
                   df[feature_name],
                   c='green',
                   marker='o',
                   alpha=0.6,
                   s=30)

        plt.xlabel('Data Name (Subject)')
        plt.ylabel(feature_name)
        plt.title(f'{feature_name} vs Data Name')
        plt.xticks(range(len(unique_names_sorted)), unique_names_sorted, rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        output_file = f"{output_dir_name}/{feature_name}.png"
        plt.savefig(output_file, dpi=100)
        plt.close()

    print(f"Saved {len(feature_columns)} scatter plots to {output_dir_name}")
    print(f"\n=== Total plots: {len(feature_columns) * 2} ===")
