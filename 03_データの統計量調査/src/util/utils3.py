#!/usr/bin/env python3
"""
データ操作ユーティリティ
"""

import os
import h5py
import numpy as np
import csv
import datetime
import matplotlib.pyplot as plt
import json
import hashlib
import bisect
from typing import Union, Optional, Dict, List

def to_unix_time(str_time):
    s = str_time.strip("'")  # -> "2022-10-10 09:02:49"
    # datetime オブジェクトにパース
    dt = datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
    # ローカルタイム系での Unix タイムスタンプを取得（float）
    ts = dt.timestamp()
    # 整数に
    return int(ts)


def get_hour_from_unix_time(unix_time: int) -> int:
    """
    Unix時刻から時間（0-23）を取得
    
    Args:
        unix_time: Unix時刻（整数）
    
    Returns:
        int: 時間（0-23）
    """
    dt = datetime.datetime.fromtimestamp(unix_time)
    return dt.hour


def is_time_of_day_match(unix_time: int, hour_condition) -> bool:
    """
    Unix時刻が指定された時間帯条件を満たすかどうかを判定
    
    Args:
        unix_time: Unix時刻（整数）
        hour_condition: 時間条件
            - int: 特定の時間（例: 9 -> 9時台）
            - list: 時間範囲（例: [9, 17] -> 9時以上17時以下）
    
    Returns:
        bool: 条件を満たす場合True
    
    Examples:
        is_time_of_day_match(1665370969, 9)        # 9時台かどうか
        is_time_of_day_match(1665370969, [9, 17])  # 9時～17時の間かどうか
    """
    hour = get_hour_from_unix_time(unix_time)
    
    if isinstance(hour_condition, list) and len(hour_condition) == 2:
        # 範囲指定
        start_hour, end_hour = hour_condition
        return start_hour <= hour <= end_hour
    else:
        # 完全一致
        return hour == hour_condition

def create_index_files() -> List[Dict[str, str]]:
    # ----------メタデータ読み込み：データが正常に読み込まれることを確認済み----------
    timestamp_path = "/home/mori/projects/affective-forecast/datas/meta_data/timestamp.csv"
    timestamp_data = []
    timestamp_csvfile = open(timestamp_path, 'r', encoding='utf-8', newline='')
    timestamp_reader = csv.DictReader(timestamp_csvfile)         
    for row in timestamp_reader:
        timestamp_data.append(dict(row))
    for timestamp_dict in timestamp_data:
        timestamp_dict["index"] = int(timestamp_dict["index"])
        timestamp_dict["datetime"] = to_unix_time(timestamp_dict["datetime"])
        timestamp_dict["valence"] = int(timestamp_dict["valence"])
        timestamp_dict["arousal"] = int(timestamp_dict["arousal"])

    # print(timestamp_data[0])
    # print("-------------------------------------")

    # Load individual term data
    individual_term_data = {}
    for term_num in [1, 2, 3]:
        path = f"/home/mori/projects/affective-forecast/datas/meta_data/individual_term{term_num}.csv"
        data = []
        with open(path, 'r', encoding='shift_jis', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                data.append(dict(row))
        individual_term_data[f"term{term_num}"] = data
    # ----------メタデータ読み込み：データが正常に読み込まれることを確認済み----------


    # ----------timestampデータに個人データを統合------------
    columns_mapping = {
        "mean_optimism": "【平均値】楽観（奇数行）",
        "mean_pessimism": "【平均値】悲観（偶数行）",
        "study_id": "研究ID番号",
        "device_mybeat": "myBeat(デバイス)",
        "device_e4": "E4（リストバンド）",
        "gender": "性別",
        "age": "年齢",
        "total_score_optimism": "【総得点】楽観（奇数行）",
        "total_score_pessimism": "【総得点】悲観（偶数行）",
        "expect_positive_future": "自分の将来は，良いことが起こると思う",
        "expect_negative_future": "私の将来は，暗いと思う",
        "expect_happy_future": "将来，幸せになれると思う",
        "imagine_unwanted_future_self": "望ましくない，未来の自分の姿ばかりを想像する",
        "positive_future_outlook": "私は将来に対して，前向きに考えている",
        "expect_everything_goes_wrong": "何もかもが悪い方向にしか進まないだろうと思う",
        "feel_future_is_blessed": "自分の将来は，恵まれていると思う",
        "expect_wishes_unfulfilled": "私の望みは叶わないと思う",
        "future_expectations": "自分の将来に期待がもてる",
        "despair_about_future": "自分の将来に絶望している",
        "expect_good_life_ahead": "これからの人生は良いものになるだろうと思う",
        "imagine_failure_when_planning": "何かを計画する時，失敗している自分の姿が頭に浮かぶ",
        "expect_positive_if_uncertain": "結果が予想できない時は，良い方向に期待する",
        "imagine_all_failures": "何をしても，うまくいかないことばかりを想像する",
        "expect_more_good_than_bad": "私には，悪いことよりも良いことが起こると思う",
        "believe_goals_unachievable": "結局，自分の目標は達成できないだろう",
        "expect_success_when_starting": "何かに取りかかる時は，成功するだろうと考える",
        "imagine_bad_future": "今後のことを考えると，悪いことばかりが頭に浮かぶ",
        "look_forward_to_future": "自分の将来を楽しみにしている",
        "expect_failure_when_starting": "何かに取りかかる時は，失敗するだろうと考える",
    }


    for term in ["term1", "term2", "term3"]:
        for indiv in individual_term_data[term]:
            ID = int(indiv["研究ID番号"])
            for timestamp_dict in timestamp_data:
                timestamp_term = str(timestamp_dict["ex-term"])
                timestamp_ID_str = str(timestamp_dict["ID"])
                timestamp_ID = int(timestamp_ID_str.replace("S",""))
                if timestamp_term == term and timestamp_ID == ID:
                    timestamp_dict["mean_optimism"] = float(indiv["【平均値】楽観（奇数行）"])
                    timestamp_dict["mean_pessimism"] = float(indiv["【平均値】悲観（偶数行）"])
                    timestamp_dict["study_id"] = int(indiv["研究ID番号"])
                    timestamp_dict["device_mybeat"] = int(indiv["myBeat(デバイス)"])
                    timestamp_dict["device_e4"] = indiv["E4（リストバンド）"]
                    timestamp_dict["gender"] = int(indiv["性別"].split(":")[0])
                    timestamp_dict["age"] = int(indiv["年齢"])
                    timestamp_dict["total_score_optimism"] = int(indiv["【総得点】楽観（奇数行）"])
                    timestamp_dict["total_score_pessimism"] = int(indiv["【総得点】悲観（偶数行）"])
                    timestamp_dict["expect_positive_future"] = int(indiv["自分の将来は，良いことが起こると思う"].split(":")[0])
                    timestamp_dict["expect_negative_future"] = int(indiv["私の将来は，暗いと思う"].split(":")[0])
                    timestamp_dict["expect_happy_future"] = int(indiv["将来，幸せになれると思う"].split(":")[0])
                    timestamp_dict["imagine_unwanted_future_self"] = int(indiv["望ましくない，未来の自分の姿ばかりを想像する"].split(":")[0])
                    timestamp_dict["positive_future_outlook"] = int(indiv["私は将来に対して，前向きに考えている"].split(":")[0])
                    timestamp_dict["expect_everything_goes_wrong"] = int(indiv["何もかもが悪い方向にしか進まないだろうと思う"].split(":")[0])
                    timestamp_dict["feel_future_is_blessed"] = int(indiv["自分の将来は，恵まれていると思う"].split(":")[0])
                    timestamp_dict["expect_wishes_unfulfilled"] = int(indiv["私の望みは叶わないと思う"].split(":")[0])
                    timestamp_dict["future_expectations"] = int(indiv["自分の将来に期待がもてる"].split(":")[0])
                    timestamp_dict["despair_about_future"] = int(indiv["自分の将来に絶望している"].split(":")[0])
                    timestamp_dict["expect_good_life_ahead"] = int(indiv["これからの人生は良いものになるだろうと思う"].split(":")[0])
                    timestamp_dict["imagine_failure_when_planning"] = int(indiv["何かを計画する時，失敗している自分の姿が頭に浮かぶ"].split(":")[0])
                    timestamp_dict["expect_positive_if_uncertain"] = int(indiv["結果が予想できない時は，良い方向に期待する"].split(":")[0])
                    timestamp_dict["imagine_all_failures"] = int(indiv["何をしても，うまくいかないことばかりを想像する"].split(":")[0])
                    timestamp_dict["expect_more_good_than_bad"] = int(indiv["私には，悪いことよりも良いことが起こると思う"].split(":")[0])
                    timestamp_dict["believe_goals_unachievable"] = int(indiv["結局，自分の目標は達成できないだろう"].split(":")[0])
                    timestamp_dict["expect_success_when_starting"] = int(indiv["何かに取りかかる時は，成功するだろうと考える"].split(":")[0])
                    timestamp_dict["imagine_bad_future"] = int(indiv["今後のことを考えると，悪いことばかりが頭に浮かぶ"].split(":")[0])
                    timestamp_dict["look_forward_to_future"] = int(indiv["自分の将来を楽しみにしている"].split(":")[0])
                    timestamp_dict["expect_failure_when_starting"] = int(indiv["何かに取りかかる時は，失敗するだろうと考える"].split(":")[0])
    
    # 以下のコードで、全timestamp_dataの29057件すべての辞書に対して、すべてのフィールドが完備されていることを確認済み
    # また、実際にtimestamp_dataの中から無作為に5件抽出し、それらが元データと一致していることを目視でも確認済み

    # keynum = {}
    # for timestamp_dict in timestamp_data:
    #     for k,v in timestamp_dict.items():
    #         if k not in keynum:
    #             keynum[k] = 1
    #         else:
    #             keynum[k] = keynum[k] + 1
    # print(keynum)


    # ----------timestampデータに個人データを統合------------


    # ----------各フィールドごとにソート済みインデックスを作成------------
    # uuid: timestamp_dataのハッシュ値,  field_value: ソート基準となるフィールドの値
    uuid_list = [hashlib.sha256(json.dumps(s, sort_keys=True).encode()).hexdigest() for s in timestamp_data]
    fields = [field for field, _ in timestamp_data[0].items()]
    indexes = {}
    
    # Create index directory if it doesn't exist
    index_dir = "/home/mori/projects/affective-forecast/datas/index"
    os.makedirs(index_dir, exist_ok=True)
    
    for field in fields:
        index = []
        for i, _ in enumerate(timestamp_data):
            index.append({"uuid": uuid_list[i], "field": field, "field_value": timestamp_data[i][field]})
        
        # Sort by field_value
        index.sort(key=lambda x: x["field_value"])
        indexes[field] = index
        
        # Save to file
        output_path = os.path.join(index_dir, f"{field}.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(index, f, ensure_ascii=False, indent=2)
    # ----------各フィールドごとにソート済みインデックスを作成------------


    # ----------uuidでソート済みの計測データファイル名リストを作成------------
    uuid_filename_list = []
    for i, timestamp_dict in enumerate(timestamp_data):
        uuid = uuid_list[i]
        # Generate filename from timestamp data (same pattern as utils2.py line 328)
        index = timestamp_dict["index"]
        filename = f"data_{index}_E4"
        
        uuid_filename_list.append({
            "uuid": uuid,
            "filename": filename
        })
    
    # Sort by uuid for fast lookup
    uuid_filename_list.sort(key=lambda x: x["uuid"])
    
    # Save uuid-sorted filename list
    uuid_filename_path = os.path.join(index_dir, "uuid_filename_mapping.json")
    with open(uuid_filename_path, 'w', encoding='utf-8') as f:
        json.dump(uuid_filename_list, f, ensure_ascii=False, indent=2)
    
    print(f"Successfully created index files")
    # ----------uuidでソート済みの計測データファイル名リストを作成------------


def search_by_conditions(search_conditions: Dict) -> List[str]:
    """
    検索条件に基づいてマッチするuuidのリストを返す
    
    Args:
        search_conditions: 検索条件の辞書
            - フィールド名: 値 (完全一致)
            - フィールド名: [最小値, 最大値] (範囲指定)
    
    Returns:
        List[str]: マッチするuuidのリスト
    
    Example:
        search_conditions = {
            "valence": [1, 3],      # valenceが1以上3以下
            "arousal": 2,           # arousalが2
            "gender": 1             # genderが1
        }
    """
    index_dir = "/home/mori/projects/affective-forecast/datas/index"
    
    if not search_conditions:
        return []
    
    # すべての条件でフィルタリング
    matching_uuids = None
    
    for field_name, condition in search_conditions.items():
        # インデックスファイルを読み込み
        index_path = os.path.join(index_dir, f"{field_name}.json")
        if not os.path.exists(index_path):
            print(f"Warning: Index file not found for field '{field_name}'")
            continue
            
        with open(index_path, 'r', encoding='utf-8') as f:
            field_index = json.load(f)
        
        # 条件に応じてフィルタリング
        field_matching_uuids = set()
        
        if isinstance(condition, list) and len(condition) == 2:
            # 範囲指定
            min_val, max_val = condition
            for entry in field_index:
                if min_val <= entry["field_value"] <= max_val:
                    field_matching_uuids.add(entry["uuid"])
        else:
            # 完全一致
            for entry in field_index:
                if entry["field_value"] == condition:
                    field_matching_uuids.add(entry["uuid"])
        
        # 積集合を取る（AND条件）
        if matching_uuids is None:
            matching_uuids = field_matching_uuids
        else:
            matching_uuids = matching_uuids.intersection(field_matching_uuids)
    
    return list(matching_uuids) if matching_uuids else []


def get_filenames_from_uuids(uuids: List[str]) -> List[str]:
    """
    uuidのリストからファイル名のリストを取得
    
    Args:
        uuids: uuidのリスト
    
    Returns:
        List[str]: ファイル名のリスト
    """
    index_dir = "/home/mori/projects/affective-forecast/datas/index"
    uuid_filename_path = os.path.join(index_dir, "uuid_filename_mapping.json")
    
    if not os.path.exists(uuid_filename_path):
        print("Error: UUID-filename mapping not found")
        return []
    
    with open(uuid_filename_path, 'r', encoding='utf-8') as f:
        uuid_filename_list = json.load(f)
    
    # uuidでソート済みなのでバイナリサーチで高速検索
    filenames = []
    uuid_to_filename = {entry["uuid"]: entry["filename"] for entry in uuid_filename_list}
    
    for uuid in uuids:
        if uuid in uuid_to_filename:
            filenames.append(uuid_to_filename[uuid])
        else:
            print(f"Warning: UUID not found: {uuid}")
    
    return filenames




def search_and_get_filenames(search_conditions: Dict) -> List[str]:
    """
    検索条件に基づいてマッチするファイル名のリストを取得
    
    Args:
        search_conditions: 検索条件の辞書
    
    Returns:
        List[str]: マッチするファイル名のリスト
    
    Example:
        search_conditions = {
            "valence": [1, 3],      # valenceが1以上3以下
            "arousal": 2,           # arousalが2
            "gender": 1             # genderが1
        }
        filenames = search_and_get_filenames(search_conditions)
    """
    uuids = search_by_conditions(search_conditions)
    return get_filenames_from_uuids(uuids)



def load_h5_data(file_path: str, dataset_name: Optional[str] = None) -> np.ndarray:
    """
    HDF5ファイルからデータを読み込み、numpy配列として返す
    
    Args:
        file_path (str): HDF5ファイルのパス
        dataset_name (str, optional): データセット名。指定しない場合は最初のデータセットを使用
        
    Returns:
        np.ndarray: n次元numpy配列
        
    Raises:
        FileNotFoundError: ファイルが存在しない場合
        KeyError: 指定されたデータセット名が存在しない場合
        ValueError: ファイルが無効な場合
    """
    
    # ファイル存在確認
    if not os.path.exists(file_path):
        return None
    
    if not file_path.endswith('.h5'):
        return None
    
    try:
        with h5py.File(file_path, "r") as f:
            # データセット名が指定されていない場合は最初のデータセットを使用
            if dataset_name is None:
                if len(f.keys()) == 0:
                    raise ValueError(f"HDF5ファイルにデータセットが見つかりません: {file_path}")
                dataset_name = list(f.keys())[0]
            
            # データセット存在確認
            if dataset_name not in f:
                available_datasets = list(f.keys())
                raise KeyError(f"データセット '{dataset_name}' が見つかりません。利用可能なデータセット: {available_datasets}")
            
            # データ読み込み
            dataset = f[dataset_name]
            data = dataset[()]  # 全データを一度に読み込み
            
            # numpy配列として返す
            return np.array(data)
            
    except Exception as e:
        return None


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