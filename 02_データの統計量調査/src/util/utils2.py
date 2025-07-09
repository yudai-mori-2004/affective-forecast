#!/usr/bin/env python3
"""
データ操作ユーティリティ
"""

import os
import h5py
import numpy as np
import csv
from datetime import datetime
import matplotlib.pyplot as plt
import json
from typing import Union, Optional, Dict, List


data_info = {
    "act": {
        "name": "Activity", 
        "unit": "g", 
        "color": "blue",
        "descriptions": [
            "Raw acceleration in x-axis (32Hz*15min)",
            "Raw acceleration in y-axis (32Hz*15min)", 
            "Raw acceleration in z-axis (32Hz*15min)",
            "Body movement vector magnitude (32Hz*15min)",
            "Gravity component of x-axis acceleration (32Hz*15min)",
            "Gravity component of y-axis acceleration (32Hz*15min)",
            "Gravity component of z-axis acceleration (32Hz*15min)",
            "Body movement component in x-axis (32Hz*15min)",
            "Body movement component in y-axis (32Hz*15min)",
            "Body movement component in z-axis (32Hz*15min)"
        ]
    },
    "eda": {
        "name": "Electrodermal Activity", 
        "unit": "μS", 
        "color": "green",
        "descriptions": [
            "Skin potential data before emotion measurement (4Hz*15min)"
        ]
    },
    "rri": {
        "name": "RR Interval", 
        "unit": "seconds", 
        "color": "red",
        "descriptions": [
            "Heart rate interval data before emotion measurement (4Hz*15min)"
        ]
    },
    "temp": {
        "name": "Temperature", 
        "unit": "°C", 
        "color": "orange",
        "descriptions": [
            "Skin temperature data before emotion measurement (4Hz*15min)"
        ]
    }
}


def load_h5_data(file_path) -> np.ndarray:
    """HDF5ファイルからデータを読み込み、numpy配列として返す"""
    # ファイル存在確認
    if not os.path.exists(file_path):
        return None
    
    if not file_path.endswith('.h5'):
        return None
    
    try:
        f = h5py.File(file_path, "r")
        dataset_name = list(f.keys())[0]
        return np.array(f[dataset_name])
    except Exception as e:
        return None



def load_timestamp() -> List[Dict[str, str]]:
    """timestampファイルを読み込み、辞書のリストとして返す"""
    
    path = "/home/mori/projects/affective-forecast/datas/meta_data/timestamp.csv"

    if not os.path.exists(path):
        return None
    
    try:
        data = []
        csvfile = open(path, 'r', encoding='utf-8', newline='')
        reader = csv.DictReader(csvfile)         
        for row in reader:
            data.append(dict(row))
        return data
    except Exception as e:
        return None


def parse_queries(queries):

    if queries is None:
        return None

    grouped_queries = {}
    for query in queries:
        prefix = query[0]
        if prefix not in grouped_queries:
            grouped_queries[prefix] = []
        grouped_queries[prefix].append(query)
    
    result = {}
    for prefix, query_list in grouped_queries.items():
        values = []
        for query in query_list:
            range_str = query[1:]
            if '~' in range_str:
                start, end = range_str.split('~')
                values.append([int(start), int(end)])
            else:
                values.append([int(range_str)]*2)
        result[prefix] = values
    return result


def search_data_by_query(data_kind, act_kind, queries):
    """
    条件に一致するデータを検索して統合データのリストを返す
    クエリ指定は 数値 or 範囲

    a: arausal の値が条件を満たすデータをすべて返す
    v: valence の値が条件を満たすデータをすべて返す
    c: 全データ通しての 計測回数 が条件を満たす被験者のデータをすべて返す
    t: datetime の値が条件を満たすデータをすべて返す
    s: 15分間のうち、指定した時間(分単位) のデータを返す
    
    """

    timestamp_data = load_timestamp()

    data_kinds = ["act", "eda", "rri", "temp"] 
    act_kinds = ["accx", "accy", "accz", "VI", "gmx", "gmy", "gmz", "bmx", "bmy", "bmz"]

    integrated_data = []

    # クエリを解析
    all_query_values = parse_queries(queries)
    print(all_query_values)

    fetch_file_indexes = [int(x["index"]) - 1 for x in timestamp_data]

    c_query = all_query_values.get('c', None)
    if c_query is not None:
        sub_list = [f"{x["ex-term"]}_{x["ID"]}" for x in timestamp_data]
        sub_counts = {}
        for sub in sub_list:
            sub_counts[sub] = sub_counts.get(sub, 0) + 1
        result = []
        for q in c_query:
            filtered_sub_list = [sub for sub, count in sub_counts.items() if q[0] <= count <= q[1]]
            filtered_indexes = [index for index in fetch_file_indexes if f"{timestamp_data[index]["ex-term"]}_{timestamp_data[index]["ID"]}" in filtered_sub_list]
            result.extend(filtered_indexes)
        fetch_file_indexes = list(set(result))


    # 各クエリについても同じように検索処理を書く
    v_query = all_query_values.get('v', None)
    if v_query is not None:
        result = []
        for q in v_query:
            filtered_indexes = [index for index in fetch_file_indexes if q[0] <= int(timestamp_data[index]["valence"]) <= q[1]]
            result.extend(filtered_indexes)
        fetch_file_indexes = list(set(result))

    a_query = all_query_values.get('a', None)
    if a_query is not None:
        result = []
        for q in a_query:
            filtered_indexes = [index for index in fetch_file_indexes if q[0] <= int(timestamp_data[index]["arousal"]) <= q[1]]
            result.extend(filtered_indexes)
        fetch_file_indexes = list(set(result))

    t_query = all_query_values.get('t', None)
    if t_query is not None:
        result = []
        for q in t_query:
            filtered_indexes = []
            for index in fetch_file_indexes:
                dt_str = timestamp_data[index]["datetime"].replace("'", "")
                dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
                hour = dt.hour 
                if q[0] <= hour < q[1]:
                    filtered_indexes.append(index)
            result.extend(filtered_indexes)
        fetch_file_indexes = list(set(result))

    print(fetch_file_indexes)
    
    for index in fetch_file_indexes:
        wave_datas = load_h5_data(f"/home/mori/projects/affective-forecast/datas/biometric_data/data_{index + 1}_E4_{data_kind}.h5")
        field_name = f"{data_kind}{"_" if act_kind is not None else ""}{act_kind}"
        wave_data = None if wave_datas is None else wave_datas[0] if act_kind is None else wave_datas[act_kinds.index(act_kind)]

        integrated_data.append({
            "subID": timestamp_data[index]["ID"],
            "date": timestamp_data[index]["datetime"],
            "valence": timestamp_data[index]["valence"],
            "arousal": timestamp_data[index]["arousal"],
            "exist_data": (
                wave_datas is not None
            ),
            field_name: wave_data,
        })
    return integrated_data



def plot_15minutes_waves(datas, data_kind, data_row=0, data_range=None, save_at=None):
    """横軸:15分間 縦軸:指定した生体データの値 で波形を描画する"""

    plt.figure(figsize=(12, 8))

    for data in datas:
        if not data["exist_data"]:
            print("データが存在しません")
            plt.close()
            return False
        
        # データ種別に応じた詳細情報を設定
        
        info = data_info.get(data_kind, {"name": data_kind.upper(), "unit": "", "color": "blue", "descriptions": ["Unknown data"]})
        
        # data_rowに対応する説明を取得
        description = info["descriptions"][data_row] if data_row < len(info["descriptions"]) else f"Row {data_row} data"
        
        x = np.linspace(0, 15, data[f"{data_kind}_data"].shape[1])
        y = data[f"{data_kind}_data"][data_row]
        
        plt.plot(x, y, color=info["color"], linewidth=1.5)
        plt.ylim(data_range[0], data_range[1])
        plt.xlabel("Time (minutes)")
        plt.ylabel(f"{info['name']} ({info['unit']})")
        plt.grid(True, alpha=0.3)
        plt.title(f"{info['name']} Waveform - Subject {data['subID']} ({data['date']})\nValence: {data['valence']}, Arousal: {data['arousal']}")
        
        # データの説明をグラフ下部に追加
        plt.figtext(0.1, 0.02, f"Data: {description}", fontsize=10, style='italic')
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1)  # 下部の余白を調整

    if save_at is not None:
        plt.savefig(save_at)
        plt.close()
    else:
        plt.show()

    return True

def draw_hist(data, bins, x_label=None, y_label=None, title=None, description=None, save_at=None):
    """横軸:データの値 縦軸:度数 でヒストグラムを描画する"""
    
    plt.figure(figsize=(12, 8))
    plt.hist(data, bins=bins)
    plt.xlabel(f"{x_label}")
    plt.ylabel(f"{y_label}")
    plt.title(f"Histgram - {title}")

    # データの説明をグラフ下部に追加
    if description is not None:
        plt.figtext(0.1, 0.02, f"Description: {description}", fontsize=10, style='italic')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)

    if save_at is not None:
        plt.savefig(save_at)
    else:
        plt.show()

    return True

def object_to_json(obj, save_at=None):
    """オブジェクトをファイルに保存"""
    # ディレクトリが存在しない場合は作成
    os.makedirs(os.path.dirname(save_at), exist_ok=True)
    
    with open(save_at, "w", encoding="utf-8") as f: 
        json.dump(obj, f, indent=2, ensure_ascii=False)
    print(f"Object saved to {save_at}")

def object_from_json(load_from=None):
    """JSONファイルからオブジェクトを読み込み"""
    # ファイルが存在するかチェック
    if not os.path.exists(load_from):
        raise FileNotFoundError(f"File not found: {load_from}")
    
    with open(load_from, "r", encoding="utf-8") as f:
        obj = json.load(f)
    print(f"Object loaded from {load_from}")
    return obj