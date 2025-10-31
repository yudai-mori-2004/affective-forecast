import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
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

    # データ名でソート
    datas.sort(key=lambda x: int(x["filename"].split("_")[1]))

    # filename -> arousal のマッピングを作成
    filename_to_arousal = {d["filename"]: d["arousal"] for d in datas}

    print(f"Loaded arousal data for {len(datas)} files")
    print(f"Arousal range: {min(filename_to_arousal.values())} - {max(filename_to_arousal.values())}")

    # 特徴量に Arousal を追加
    df["Arousal"] = df["name"].map(filename_to_arousal)

    # Arousalが無い行を除外
    missing_arousal = df["Arousal"].isna().sum()
    if missing_arousal > 0:
        print(f"Warning: {missing_arousal} rows have missing arousal values")
        df = df[df["Arousal"].notna()]
        print(f"After filtering: {len(df)} rows")

    # メタデータ以外の全カラムを数値型に変換
    meta_columns = ['name', 'length_seconds']
    for col in df.columns:
        if col not in meta_columns and col != 'Arousal':
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 特徴量カラム（メタデータとArousal以外）
    feature_columns = [col for col in df.columns if col not in meta_columns + ['Arousal']]

    print(f"\nFeature columns: {len(feature_columns)}")

    # 出力先ディレクトリ
    output_dir = "/home/mori/projects/affective-forecast/08_CVX分割適用/plots/arousal_prediction_rf"
    os.makedirs(output_dir, exist_ok=True)

    # 各時間長ごとにランダムフォレストを訓練
    lengthes = [900, 840, 780, 720, 660, 600, 540, 480, 420, 360, 300, 240, 180, 120, 60]

    results = []

    for l in lengthes:
        print(f"\n{'='*60}")
        print(f"Training Random Forest for length={l}s...")
        print(f"{'='*60}")

        # この時間長のデータを抽出
        data_subset = df[df['length_seconds'] == l].copy()

        # 特徴量とターゲット
        X = data_subset[feature_columns]
        y = data_subset['Arousal']

        # NaNを含む列を除外
        valid_features = X.columns[~X.isna().all()]
        X = X[valid_features].fillna(X[valid_features].median())

        print(f"  Valid features: {len(valid_features)}")
        print(f"  Sample count: {len(X)}")

        if len(X) < 10:
            print(f"  Skipping: Not enough samples")
            continue

        # Train/Test分割（80/20）
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        print(f"  Train samples: {len(X_train)}")
        print(f"  Test samples: {len(X_test)}")

        # ランダムフォレストモデル
        rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )

        # 訓練
        rf.fit(X_train, y_train)

        # 予測
        y_train_pred = rf.predict(X_train)
        y_test_pred = rf.predict(X_test)

        # 評価指標
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        train_rmse = np.sqrt(train_mse)
        test_rmse = np.sqrt(test_mse)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)

        print(f"\n  Results:")
        print(f"    Train RMSE: {train_rmse:.4f}")
        print(f"    Test RMSE:  {test_rmse:.4f}")
        print(f"    Train MAE:  {train_mae:.4f}")
        print(f"    Test MAE:   {test_mae:.4f}")
        print(f"    Train R²:   {train_r2:.4f}")
        print(f"    Test R²:    {test_r2:.4f}")

        # 交差検証スコア
        cv_scores = cross_val_score(rf, X, y, cv=5, scoring='r2', n_jobs=-1)
        print(f"    CV R² (5-fold): {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        # 結果を保存
        results.append({
            'length_seconds': l,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'cv_r2_mean': cv_scores.mean(),
            'cv_r2_std': cv_scores.std(),
            'n_samples': len(X),
            'n_features': len(valid_features)
        })

        # Feature Importance（上位20個）
        feature_importance = pd.DataFrame({
            'feature': valid_features,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)

        print(f"\n  Top 10 Important Features:")
        for i, row in feature_importance.head(10).iterrows():
            print(f"    {row['feature']}: {row['importance']:.4f}")

        # Feature Importanceをプロット
        plt.figure(figsize=(10, 8))
        top_features = feature_importance.head(20)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title(f'Top 20 Feature Importances (Length={l}s, Test R²={test_r2:.4f})')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/feature_importance_{l}s.png", dpi=150)
        plt.close()

        # 予測 vs 実測プロット
        plt.figure(figsize=(10, 5))

        # Train
        plt.subplot(1, 2, 1)
        plt.scatter(y_train, y_train_pred, alpha=0.5, s=20)
        plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
        plt.xlabel('Actual Arousal')
        plt.ylabel('Predicted Arousal')
        plt.title(f'Train (R²={train_r2:.4f})')
        plt.grid(True, alpha=0.3)

        # Test
        plt.subplot(1, 2, 2)
        plt.scatter(y_test, y_test_pred, alpha=0.5, s=20, color='green')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Arousal')
        plt.ylabel('Predicted Arousal')
        plt.title(f'Test (R²={test_r2:.4f})')
        plt.grid(True, alpha=0.3)

        plt.suptitle(f'Arousal Prediction (Length={l}s)', fontsize=14)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/prediction_{l}s.png", dpi=150)
        plt.close()

    # 全結果をCSVに保存
    results_df = pd.DataFrame(results)
    results_file = f"{output_dir}/rf_results_summary.csv"
    results_df.to_csv(results_file, index=False)
    print(f"\n{'='*60}")
    print(f"Results summary saved to {results_file}")
    print(f"{'='*60}")

    # 結果サマリーをプロット
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Test R²
    axes[0, 0].plot(results_df['length_seconds'], results_df['test_r2'], 'o-', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('Length (seconds)')
    axes[0, 0].set_ylabel('Test R²')
    axes[0, 0].set_title('Test R² vs Length')
    axes[0, 0].grid(True, alpha=0.3)

    # Test RMSE
    axes[0, 1].plot(results_df['length_seconds'], results_df['test_rmse'], 'o-', linewidth=2, markersize=8, color='orange')
    axes[0, 1].set_xlabel('Length (seconds)')
    axes[0, 1].set_ylabel('Test RMSE')
    axes[0, 1].set_title('Test RMSE vs Length')
    axes[0, 1].grid(True, alpha=0.3)

    # CV R²
    axes[1, 0].errorbar(results_df['length_seconds'], results_df['cv_r2_mean'],
                        yerr=results_df['cv_r2_std'], fmt='o-', linewidth=2, markersize=8, color='green')
    axes[1, 0].set_xlabel('Length (seconds)')
    axes[1, 0].set_ylabel('CV R² (mean ± std)')
    axes[1, 0].set_title('Cross-Validation R² vs Length')
    axes[1, 0].grid(True, alpha=0.3)

    # Sample count
    axes[1, 1].plot(results_df['length_seconds'], results_df['n_samples'], 'o-', linewidth=2, markersize=8, color='red')
    axes[1, 1].set_xlabel('Length (seconds)')
    axes[1, 1].set_ylabel('Number of Samples')
    axes[1, 1].set_title('Sample Count vs Length')
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle('Random Forest Performance Summary', fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/rf_summary.png", dpi=150)
    plt.close()

    print(f"\nAll plots saved to {output_dir}")
    print("\nBest performing length (by Test R²):")
    best_idx = results_df['test_r2'].idxmax()
    best_result = results_df.loc[best_idx]
    print(f"  Length: {best_result['length_seconds']}s")
    print(f"  Test R²: {best_result['test_r2']:.4f}")
    print(f"  Test RMSE: {best_result['test_rmse']:.4f}")
    print(f"  CV R²: {best_result['cv_r2_mean']:.4f} (+/- {best_result['cv_r2_std']:.4f})")
