import os
import numpy as np
import matplotlib.pyplot as plt
import json
import datetime
from typing import Union, Optional, Dict, List

# 全計測データについて、計測時刻がいつに集中しているか調べるためのヒストグラムを描画

if __name__ == "__main__":

    from util.utils3 import search_and_get_filenames, draw_hist, object_to_json, is_time_of_day_match, get_filenames_from_uuids

    # datetime.jsonファイルから時刻データを読み込み
    f = open("/home/mori/projects/affective-forecast/datas/index/datetime.json", 'r', encoding='utf-8')
    field_index = json.load(f)
    
    # 各ファイルのUUIDを取得してdatetime.jsonから時刻を抽出
    uuid_filename_mapping_path = "/home/mori/projects/affective-forecast/datas/index/uuid_filename_mapping.json"
    with open(uuid_filename_mapping_path, 'r') as f:
        uuid_filename_mapping = json.load(f)
    
    # ファイル名からUUIDを取得するための辞書
    filename_to_uuid = {entry["filename"]: entry["uuid"] for entry in uuid_filename_mapping}
    
    # 各termの時刻データを格納する辞書
    times_by_term = {}
    
    # 各termのファイルについて時刻データを取得
    for term in ["term1", "term2", "term3"]:
        search_conditions = {
            "ex-term": term,
        }
        filenames = search_and_get_filenames(search_conditions)
        
        times_by_term[term] = []
        
        # 各ファイルの時刻を取得
        for filename in filenames:
            if filename in filename_to_uuid:
                uuid = filename_to_uuid[filename]
                # datetime.jsonからunix timestampを取得
                for dt_entry in field_index:
                    if dt_entry["uuid"] == uuid:
                        unix_timestamp = dt_entry["field_value"]
                        # Unix timestampから時刻を取得
                        dt = datetime.datetime.fromtimestamp(unix_timestamp)
                        hour = dt.hour
                        times_by_term[term].append(hour)
                        break
    
    # allのデータを作成（term1~3の合計）
    times_by_term["all"] = []
    for term in ["term1", "term2", "term3"]:
        times_by_term["all"].extend(times_by_term[term])
    
    print(f"Measurements by term: {[(term, len(times)) for term, times in times_by_term.items()]}")
    
    # ファイル名を取得（拡張子なし）
    file_name = os.path.splitext(os.path.basename(__file__))[0]
    # 出力パスを定義
    output_path = f"/home/mori/projects/affective-forecast/plots"
    # フォルダが存在しない場合は作成
    os.makedirs(output_path, exist_ok=True)
    
    # 各termの時間帯別ヒストグラムを作成
    for term in ["term1", "term2", "term3", "all"]:
        times = times_by_term[term]
        
        # タイトルとラベルを分かりやすく設定
        if term == "all":
            title = "Measurement counts by hour of day (All Terms)"
        else:
            title = f"Measurement counts by hour of day ({term.upper()})"
        
        draw_hist(
            data=times, 
            bins=24,  # 24時間分
            x_label="Hour of Day", 
            y_label="Measurement Count", 
            title=title,
            save_at=f"{output_path}/{file_name}/{term}_hourly_hist"
        )

        print(f"{term} Ave. ", np.mean(times))
        print(f"{term} Std. ",np.std(times))
