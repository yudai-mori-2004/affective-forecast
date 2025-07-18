import os
import json
import numpy as np
import matplotlib         # GUI を使わず画像ファイルを保存するためのバックエンド固定
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from util.utils3 import (
    create_index_files,    # 参考コードに合わせたユーティリティ（ここでは未使用）
    is_time_of_day_match,  # 指定時間帯かどうかを判定
    get_filenames_from_uuids,
    search_and_get_filenames
)

# -----------------------------------------------------------------------------
#   1) 3 時間ごとに valence と arousal の全測定値を収集し，
#      平均 (μ) と標準偏差 (σ) を計算して print で出力する。
#   2) 折れ線グラフに平均値を描き，塗りつぶし部分を ±1σ（μ−σ 〜 μ+σ）の範囲とする。
#   3) valence と arousal を別々の PNG ファイルとして保存する（計 2 枚）。
# -----------------------------------------------------------------------------

if __name__ == "__main__":

    # ---------- 設定 ----------
    INDEX_FILE = "/home/mori/projects/affective-forecast/datas/index/datetime.json"
    OUTPUT_DIR = "/home/mori/projects/affective-forecast/plots/survey_9"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 3 時間おきの時間帯
    TIME_RANGES = [[0, 3], [3, 6], [6, 9], [9, 12],
                   [12, 15], [15, 18], [18, 21], [21, 24]]

    # ---------- インデックス読み込み ----------
    with open(INDEX_FILE, "r", encoding="utf-8") as f:
        field_index = json.load(f)

    # 結果格納用コンテナ
    time_labels   = []
    stats_valence = {"mean": [], "std": []}
    stats_arousal = {"mean": [], "std": []}

    # ---------- 各時間帯ループ ----------
    for t_start, t_end in TIME_RANGES:

        # 時間帯ラベル
        label = f"{t_start:02d}:00-{t_end:02d}:00"
        time_labels.append(label)

        # 対象 UUID
        uuids = [
            rec["uuid"]
            for rec in field_index
            if is_time_of_day_match(rec["field_value"], [t_start, t_end])
        ]
        files_in_range = set(get_filenames_from_uuids(uuids))

        # 値リスト
        vals_valence, vals_arousal = [], []

        # valence × arousal の 9×9 グリッドを網羅
        for v in range(-4, 5):
            for a in range(-4, 5):
                cond = {"valence": v, "arousal": a}
                files = set(search_and_get_filenames(cond)) & files_in_range
                n = len(files)
                vals_valence.extend([v] * n)
                vals_arousal.extend([a] * n)

        # 統計計算
        def mean_std(arr):
            if not arr:
                return 0.0, 0.0
            return float(np.mean(arr)), float(np.std(arr))

        μ_v, σ_v = mean_std(vals_valence)
        μ_a, σ_a = mean_std(vals_arousal)

        stats_valence["mean"].append(μ_v)
        stats_valence["std"].append(σ_v)
        stats_arousal["mean"].append(μ_a)
        stats_arousal["std"].append(σ_a)

        # 結果表示
        print(
            f"{label:>11} | "
            f"Valence μ={μ_v:.3f} ±{σ_v:.3f} (n={len(vals_valence)}), "
            f"Arousal μ={μ_a:.3f} ±{σ_a:.3f} (n={len(vals_arousal)})"
        )

    # ---------- 可視化 ----------
    def plot_mean_std(stats, title, color, outfile):
        """平均値を折れ線で描き，±1σ を塗りつぶしで示す"""
        x     = np.arange(len(time_labels))
        mean  = np.array(stats["mean"])
        std   = np.array(stats["std"])
        upper = mean + std
        lower = mean - std

        plt.figure(figsize=(8, 6))

        # ±1σ を塗りつぶし
        plt.fill_between(x, lower, upper, color=color, alpha=0.20,
                         label="Mean ±1σ")

        # 平均値折れ線
        plt.plot(x, mean, '-o', color=color, linewidth=2,
                 markersize=8, label="Mean")

        # 体裁
        plt.xticks(x, time_labels, rotation=45)
        plt.xlabel("Time Range")
        plt.ylabel("Score")
        plt.title(title)
        plt.grid(alpha=0.3)
        plt.axhline(0, color="k", linewidth=0.8, alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, outfile))
        plt.close()

    # Valence グラフ
    plot_mean_std(
        stats_valence,
        "Valence: Mean ±1σ by Time of Day",
        "blue",
        "valence_mean_std.png"
    )

    # Arousal グラフ
    plot_mean_std(
        stats_arousal,
        "Arousal: Mean ±1σ by Time of Day",
        "red",
        "arousal_mean_std.png"
    )

    print(f"\nPNG files saved in: {OUTPUT_DIR}")
