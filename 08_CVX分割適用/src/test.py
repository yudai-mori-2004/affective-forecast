import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    # 特徴量CSVを読み込み
    feature_file = "/home/mori/projects/affective-forecast/08_CVX分割適用/features/eda_features_completed.csv"
    df = pd.read_csv(feature_file)

    # メタデータ以外の全カラムを数値型に変換（ENなどのタプル文字列も処理）
    meta_columns = ['name', 'length_seconds']
    for col in df.columns:
        if col not in meta_columns:
            # 文字列として読み込まれたタプル形式のデータを処理
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # プロットする特徴量（メタデータ以外）
    meta_columns = ['name', 'length_seconds']
    feature_columns = ["MSSP"]

    # 各特徴量についてヒストグラムを作成
    lengthes = [900, 840, 780, 720, 660, 600, 540, 480, 420, 360, 300, 240, 180, 120, 60]

    for feature_name in feature_columns:
        # この特徴量の全データ（全長さ）から範囲を計算
        all_data = df[feature_name].replace([np.inf, -np.inf], np.nan).dropna()
        if len(all_data) == 0:
            print(f"Skipping {feature_name}: no valid data")
            continue

        pp=0
        ppp=0
        for l in lengthes:
            data = df[df['length_seconds'] == l][feature_name].replace([np.inf, -np.inf], np.nan).dropna()
            print(np.mean(data))






