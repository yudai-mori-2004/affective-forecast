import os
import numpy as np
import matplotlib.pyplot as plt
import json
import datetime
import hashlib
from typing import Union, Optional, Dict, List

# 各被験者が何回計測を行ったか、サンプル数をヒストグラムで表示

if __name__ == "__main__":

    from util.utils3 import search_and_get_filenames, draw_hist, object_to_json

    sample = {}

    for term in ["term1", "term2", "term3"]:
        for study_id in range(0,200):
            search_conditions = {
                "ex-term": term,
                "study_id": study_id,
            }
            filenames = search_and_get_filenames(search_conditions)
            count = len(filenames)
            
            if f"{term}" not in sample:
                sample[f"{term}"] = {}
            
            sample[f"{term}"][f"{term}{study_id}"] = count
    
    # allのデータを作成（term1~3の合計）
    sample["all"] = {}
    for study_id in range(0, 200):
        total_count = 0
        for term in ["term1", "term2", "term3"]:
            sample["all"][f"{term}{study_id}"] = sample[term][f"{term}{study_id}"]
    
    
    # ファイル名を取得（拡張子なし）
    file_name = os.path.splitext(os.path.basename(__file__))[0]
    # 出力パスを定義
    output_path = f"/home/mori/projects/affective-forecast/plots"
    # フォルダが存在しない場合は作成
    os.makedirs(output_path, exist_ok=True)

    
    # 各termのヒストグラムを作成
    for term in ["term1", "term2", "term3", "all"]:
        values = [count for count in sample[term].values() if count != 0]
        
        # タイトルとラベルを分かりやすく設定
        if term == "all":
            title = "Distribution of Total Measurements per Subject (All Terms)"
        else:
            title = f"Distribution of Measurements per Subject ({term.upper()})"
        
        draw_hist(
            data=values, 
            bins=20, 
            x_label="Number of measurements", 
            y_label="Number of subjects", 
            title=title,
            save_at=f"{output_path}/{file_name}/{term}_hist"
        )

        values = [d for d in sample[term].values()]
        print(f"{term} Ave. ", np.mean(values))
        print(f"{term} Std. ",np.std(values))
