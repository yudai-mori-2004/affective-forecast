from util.utils3 import create_index_files, is_time_of_day_match, get_filenames_from_uuids
import os
import numpy as np
import matplotlib.pyplot as plt
import json
import datetime
import hashlib
from typing import Union, Optional, Dict, List

# 一日の時間帯ごとに、valence と arousal スコアを xy 座標上で可視化（割合表示）

if __name__ == "__main__":

    from util.utils3 import search_and_get_filenames

    f = open("/home/mori/projects/affective-forecast/datas/index/datetime.json", 'r', encoding='utf-8')
    field_index = json.load(f)
    
    time_ranges = [[0,3],[3,6],[6,9],[9,12],[12,15],[15,18],[18,21],[21,24]]

    for time_range in time_ranges:

        uuid_in_time_range = [dic["uuid"] for dic in field_index if is_time_of_day_match(dic["field_value"],time_range)]
        filenames_in_time_range = get_filenames_from_uuids(uuid_in_time_range)

        affection = {}

        for valence in [-4,-3,-2,-1,0,1,2,3,4]:
            for arousal in [-4,-3,-2,-1,0,1,2,3,4]:
                search_conditions = {
                    "valence": valence,
                    "arousal": arousal,
                }
                filenames = search_and_get_filenames(search_conditions)
                filenames = set(filenames).intersection(set(filenames_in_time_range))
                count = len(filenames)
                
                if f"{valence}" not in affection:
                    affection[f"{valence}"] = {}
                
                affection[f"{valence}"][f"{arousal}"] = count
        
        # 時間帯の総サンプル数を計算
        total_samples = len(filenames_in_time_range)
        
        # matplotlib で可視化
        valence_values = []
        arousal_values = []
        proportions = []
        
        for valence_str, arousal_dict in affection.items():
            for arousal_str, count in arousal_dict.items():
                valence_values.append(int(valence_str))
                arousal_values.append(int(arousal_str))
                # 割合として計算
                proportion = count / total_samples if total_samples > 0 else 0
                proportions.append(proportion)
        
        file_name = os.path.splitext(os.path.basename(__file__))[0]
        output_path = f"/home/mori/projects/affective-forecast/plots/{file_name}"
        os.makedirs(output_path, exist_ok=True)

        plt.figure(figsize=(10, 8))
        # 点のサイズを最適化（最大割合が適切な大きさになるように調整）
        max_proportion = max(proportions) if proportions else 0.01
        size_scale = 300 / max_proportion if max_proportion > 0 else 300  # 最大サイズを300に設定
        sizes = [p * size_scale for p in proportions]
        scatter = plt.scatter(valence_values, arousal_values, s=sizes, alpha=0.6, c=proportions, cmap='viridis')
        plt.colorbar(scatter, label='Proportion')
        plt.xlabel('Valence')
        plt.ylabel('Arousal')
        plt.title(f'Valence vs Arousal Distribution ({time_range[0]}:00-{time_range[1]}:00)')
        plt.grid(True, alpha=0.3)
        
        # 軸の範囲を設定
        plt.xlim(-4.5, 4.5)
        plt.ylim(-4.5, 4.5)
        
        # 原点に線を追加
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        
        # 各点に割合を表示
        for i, (v, a, p) in enumerate(zip(valence_values, arousal_values, proportions)):
            if p > 0:  # 割合が0より大きい場合のみ表示
                plt.annotate(f'{p:.3f}', (v, a), xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(f"{output_path}/affection_distribution_{time_range[0]}_{time_range[1]}")
        plt.close()
# python 03_データの統計量調査/src/survey_8.py
