#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
survey_10.py
-----------------------------------------------

"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime

# ---------- パス設定 ----------
BASE_DIR   = "/home/mori/projects/affective-forecast"
INDEX_DIR  = os.path.join(BASE_DIR, "datas/index")
OUTPUT_DIR = os.path.join(BASE_DIR, "plots/survey_10")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CSV_PATH1   = os.path.join(OUTPUT_DIR, "affective_by_discrete_time.csv")
CSV_PATH2   = os.path.join(OUTPUT_DIR, "affective_by_continuous_time.csv")

def get_hour_minute_from_unix_time(unix_time):
    dt = datetime.fromtimestamp(unix_time)
    return dt.hour, dt.minute

# ---------- インデックス読み込み ----------
with open(os.path.join(INDEX_DIR, "datetime.json"), "r", encoding="utf-8") as f:
    dt_index = json.load(f)
with open(os.path.join(INDEX_DIR, "ex-term.json"), "r", encoding="utf-8") as f:
    term_index = json.load(f)
with open(os.path.join(INDEX_DIR, "study_id.json"), "r", encoding="utf-8") as f:
    stud_index = json.load(f)
with open(os.path.join(INDEX_DIR, "valence.json"), "r", encoding="utf-8") as f:
    val_index = json.load(f)
with open(os.path.join(INDEX_DIR, "arousal.json"), "r", encoding="utf-8") as f:
    aro_index = json.load(f)
with open(os.path.join(INDEX_DIR, "gender.json"), "r", encoding="utf-8") as f:
    gender_index = json.load(f)
with open(os.path.join(INDEX_DIR, "age.json"), "r", encoding="utf-8") as f:
    age_index = json.load(f)
with open(os.path.join(INDEX_DIR, "mean_optimism.json"), "r", encoding="utf-8") as f:
    pessim_index = json.load(f)
with open(os.path.join(INDEX_DIR, "mean_pessimism.json"), "r", encoding="utf-8") as f:
    optim_index = json.load(f)

# uuid → 値 の dict を作成
term_dict   = {d["uuid"]: d["field_value"] for d in term_index}
stud_dict   = {d["uuid"]: d["field_value"] for d in stud_index}
val_dict    = {d["uuid"]: d["field_value"] for d in val_index}
aro_dict    = {d["uuid"]: d["field_value"] for d in aro_index}
gender_dict = {d["uuid"]: d["field_value"] for d in gender_index}
age_dict    = {d["uuid"]: round(d["field_value"], -1) for d in age_index}
pessim_dict = {d["uuid"]: round(d["field_value"]) for d in pessim_index}
optim_dict  = {d["uuid"]: round(d["field_value"]) for d in optim_index}
dt_dict = {
    d["uuid"]: get_hour_minute_from_unix_time(d["field_value"])
    for d in dt_index
}

rows_discrete = []
rows_continuous = []

for uuid, (hour, minute) in dt_dict.items():
    term      = term_dict[uuid]
    study_id  = stud_dict[uuid]
    valence   = val_dict[uuid]
    arousal   = aro_dict[uuid]
    gender    = gender_dict[uuid]
    age       = age_dict[uuid]
    optim     = optim_dict[uuid]
    pessim    = pessim_dict[uuid]

    subject_id = f"{term}{study_id}"

    bin_start = hour - (hour % 3)
    bin_end   = (bin_start + 3) % 24
    time_bin  = f"{bin_start:02d}:00-{bin_end:02d}:00"
    
    time_continuous = (hour + minute / 60.0) / 24.0

    rows_discrete.append((subject_id, time_bin, valence, arousal, gender, age, optim, pessim))
    rows_continuous.append((subject_id, time_continuous, valence, arousal , gender, age, optim, pessim))

df_discrete = pd.DataFrame(rows_discrete, columns=["subject_id", "time_bin", "valence", "arousal", "gender", "age", "mean_optimism", "mean_pessimism"])
df_discrete["time_bin"] = pd.Categorical(df_discrete["time_bin"],categories=sorted(df_discrete["time_bin"].unique()),
                                ordered=True)
df_discrete["gender"] = pd.Categorical(df_discrete["gender"],categories=sorted(df_discrete["gender"].unique()),
                                ordered=True)
df_discrete["age"] = pd.Categorical(df_discrete["age"],categories=sorted(df_discrete["age"].unique()),
                                ordered=True)
df_discrete["mean_optimism"] = pd.Categorical(df_discrete["mean_optimism"],categories=sorted(df_discrete["mean_optimism"].unique()),
                                ordered=True)
df_discrete["mean_pessimism"] = pd.Categorical(df_discrete["mean_pessimism"],categories=sorted(df_discrete["mean_pessimism"].unique()),
                                ordered=True)

df_continuous = pd.DataFrame(rows_continuous, columns=["subject_id", "time_continuous", "valence", "arousal", "gender", "age", "mean_optimism", "mean_pessimism"])

df_discrete.to_csv(CSV_PATH1, index=False)
print(f"[Python] CSV written: {CSV_PATH1} ({len(df_discrete)} rows)")

df_continuous.to_csv(CSV_PATH2, index=False)
print(f"[Python] CSV written: {CSV_PATH2} ({len(df_continuous)} rows)")