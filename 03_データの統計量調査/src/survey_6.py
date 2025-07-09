from util.utils3 import create_index_files
import os
import numpy as np
import matplotlib.pyplot as plt
import json
import datetime
import hashlib
from typing import Union, Optional, Dict, List

# valence と arousal スコアを xy 座標上で可視化

if __name__ == "__main__":

    from util.utils3 import search_and_get_filenames

    affection = {}

    for valence in [-4,-3,-2,-1,0,1,2,3,4]:
        for arousal in [-4,-3,-2,-1,0,1,2,3,4]:
            search_conditions = {
                "valence": valence,
                "arousal": arousal,
            }
            filenames = search_and_get_filenames(search_conditions)
            count = len(filenames)
            
            if f"{valence}" not in affection:
                affection[f"{valence}"] = {}
            
            affection[f"{valence}"][f"{arousal}"] = count
    
    print(affection)
    
    # matplotlib で可視化
    valence_values = []
    arousal_values = []
    counts = []
    
    for valence_str, arousal_dict in affection.items():
        for arousal_str, count in arousal_dict.items():
            valence_values.append(int(valence_str))
            arousal_values.append(int(arousal_str))
            counts.append(count)
    
    file_name = os.path.splitext(os.path.basename(__file__))[0]
    output_path = f"/home/mori/projects/affective-forecast/plots/{file_name}"
    os.makedirs(output_path, exist_ok=True)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(valence_values, arousal_values, s=counts, alpha=0.6, c=counts, cmap='viridis')
    plt.colorbar(scatter, label='Number of samples')
    plt.xlabel('Valence')
    plt.ylabel('Arousal')
    plt.title('Valence vs Arousal Distribution across All Data')
    plt.grid(True, alpha=0.3)
    
    # 軸の範囲を設定
    plt.xlim(-4.5, 4.5)
    plt.ylim(-4.5, 4.5)
    
    # 原点に線を追加
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # 各点にカウント数を表示
    for i, (v, a, c) in enumerate(zip(valence_values, arousal_values, counts)):
        if c > 0:  # カウントが0より大きい場合のみ表示
            plt.annotate(str(c), (v, a), xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f"{output_path}/affection_distribution")
    plt.close()
# python 03_データの統計量調査/src/survey_6.py
