#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
survey_10.py
-----------------------------------------------
1. index ディレクトリから uuid→{term, study_id, valence, arousal, datetime} を突合し、
   ・subject_id  (term+study_id)
   ・time_bin    (3時間幅ラベル)
   ・valence
   ・arousal
   のロング形式 DataFrame を構築して CSV に書き出す。
2. Rscript run_lmm.R <in_csv> <out_json> を呼び出し、
   time_bin を固定効果、subject_id をランダム切片とした
   LMM (lme4) を推定。
3. R の出力 JSON を読み取り、Mean ± residual SD を
   Valence / Arousal 別に PNG で描画・保存する。
"""

import os
import json
import subprocess
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from util.utils3 import get_hour_from_unix_time, is_time_of_day_match

# ---------- パス設定 ----------
BASE_DIR   = "/home/mori/projects/affective-forecast"
INDEX_DIR  = os.path.join(BASE_DIR, "datas/index")
OUTPUT_DIR = os.path.join(BASE_DIR, "plots/survey_10")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CSV_PATH   = os.path.join(OUTPUT_DIR, "va_long.csv")
JSON_PATH  = os.path.join(OUTPUT_DIR, "va_lmm.json")

# # ---------- インデックス読み込み ----------
# with open(os.path.join(INDEX_DIR, "datetime.json"), "r", encoding="utf-8") as f:
#     dt_index = json.load(f)
# with open(os.path.join(INDEX_DIR, "ex-term.json"), "r", encoding="utf-8") as f:
#     term_index = json.load(f)
# with open(os.path.join(INDEX_DIR, "study_id.json"), "r", encoding="utf-8") as f:
#     stud_index = json.load(f)
# with open(os.path.join(INDEX_DIR, "valence.json"), "r", encoding="utf-8") as f:
#     val_index = json.load(f)
# with open(os.path.join(INDEX_DIR, "arousal.json"), "r", encoding="utf-8") as f:
#     aro_index = json.load(f)

# # uuid → 値 の dict を作成
# term_dict   = {d["uuid"]: d["field_value"] for d in term_index}
# stud_dict   = {d["uuid"]: d["field_value"] for d in stud_index}
# val_dict    = {d["uuid"]: d["field_value"] for d in val_index}
# aro_dict    = {d["uuid"]: d["field_value"] for d in aro_index}
# dt_dict = {
#     d["uuid"]: get_hour_from_unix_time(d["field_value"])
#     for d in dt_index
# }

# # ---------- ロング形式データ生成 ----------
# rows = []
# for uuid, dt in dt_dict.items():
#     term      = term_dict[uuid]
#     study_id  = stud_dict[uuid]
#     valence   = val_dict[uuid]
#     arousal   = aro_dict[uuid]

#     subject_id = f"{term}{study_id}"

#     hr = dt
#     bin_start = hr - (hr % 3)
#     bin_end   = (bin_start + 3) % 24
#     time_bin  = f"{bin_start:02d}:00-{bin_end:02d}:00"

#     rows.append((subject_id, time_bin, valence, arousal))

# df = pd.DataFrame(rows, columns=["subject_id", "time_bin", "valence", "arousal"])
# df["time_bin"] = pd.Categorical(df["time_bin"],
#                                 categories=sorted(df["time_bin"].unique()),
#                                 ordered=True)

# # ---------- CSV 出力 ----------
# df.to_csv(CSV_PATH, index=False)
# print(f"[Python] CSV written: {CSV_PATH} ({len(df)} rows)")

# # ---------- Rscript 呼び出し ----------
# cmd = ["Rscript", os.path.join(os.path.dirname(__file__), "run_lmm.R"),
#        CSV_PATH, JSON_PATH]
# try:
#     subprocess.check_call(cmd)
# except subprocess.CalledProcessError as e:
#     print("Rscript execution failed:")
#     print(e)
#     raise SystemExit(1)

# ---------- JSON 読み込み ----------
with open(JSON_PATH, "r", encoding="utf-8") as f:
    res = json.load(f)

time_labels = res["time_bins"]
x = np.arange(len(time_labels))

def plot_mean_sd(res_key: str, color: str, outfile: str):
    mean = np.array(res[res_key]["mean"])
    sd   = np.array(res[res_key]["sd"])
    plt.figure(figsize=(8,6))
    plt.fill_between(x, mean - sd, mean + sd, color=color, alpha=0.2,
                     label="Residual SD (±1σ)")
    plt.plot(x, mean, '-o', color=color, markersize=8, label="Mean (LMM)")
    plt.xticks(x, time_labels, rotation=45)
    plt.xlabel("Time Range"); plt.ylabel("Score")
    plt.title(f"{res_key.capitalize()}  •  Mean ±1σ (LMM, subject removed)")
    plt.grid(alpha=.3); plt.axhline(0, color="k", alpha=.4)
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, outfile))
    plt.close()

plot_mean_sd("valence", "blue", "valence_lmm.png")
plot_mean_sd("arousal", "red",  "arousal_lmm.png")

print(f"[Python] Plots saved under: {OUTPUT_DIR}")
