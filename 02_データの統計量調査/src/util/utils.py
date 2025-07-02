#!/usr/bin/env python3
"""
HDF5ファイル読み込みユーティリティ
"""

import os
import h5py
import numpy as np
import csv
import matplotlib.pyplot as plt
from typing import Union, Optional, Dict, List


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



def load_csv_data(file_path: str, encoding: str = 'utf-8') -> List[Dict[str, str]]:
    """
    CSVファイルを読み込み、辞書のリストとして返す
    
    Args:
        file_path (str): CSVファイルのパス
        encoding (str): ファイルエンコーディング（デフォルト: utf-8）
        
    Returns:
        List[Dict[str, str]]: カラムヘッダーをキーとする辞書のリスト
        
    Raises:
        FileNotFoundError: ファイルが存在しない場合
        ValueError: ファイルが無効またはエンコーディングエラーが発生した場合
    """
    
    # ファイル存在確認
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CSVファイルが見つかりません: {file_path}")
    
    if not file_path.endswith('.csv'):
        raise ValueError(f"CSVファイルではありません: {file_path}")
    
    try:
        data = []
        with open(file_path, 'r', encoding=encoding, newline='') as csvfile:
            # DictReaderを使用してヘッダーを自動処理
            reader = csv.DictReader(csvfile)
            
            # ファイルにヘッダーがあるかチェック
            if reader.fieldnames is None:
                raise ValueError(f"CSVファイルにヘッダーがありません: {file_path}")
            
            # すべての行を読み込み
            for row in reader:
                data.append(dict(row))
                
        return data
        
    except UnicodeDecodeError as e:
        raise ValueError(f"CSVファイルのエンコーディングエラー: {file_path}。別のエンコーディングを試してください。エラー: {str(e)}")
    except Exception as e:
        raise ValueError(f"CSVファイルの読み込みに失敗しました: {file_path}、エラー: {str(e)}")


def search_data(subID=None, date=None, valence=None, arousal=None):
    """条件に一致するデータを検索して統合データのリストを返す"""
    DATA_PATH = "/home/mori/projects/affective-forecast/datas"
    timestamp_path = "meta_data/timestamp.csv"
    timestamp_data = load_csv_data(f"{DATA_PATH}/{timestamp_path}")
    
    integrated_data = []
    
    for v in timestamp_data:
        _subID = v["ID"]
        _date = v["datetime"]
        _valence = int(v["valence"])
        _arousal = int(v["arousal"])
        
        if (subID is not None and _subID != subID) or (date is not None and _date != date) or (valence is not None and _valence != valence) or (arousal is not None and _arousal != arousal):
            continue
            
        act, eda, rri, temp = tuple(f"data_{v['index']}_E4_{kind}.h5" for kind in ["act", "eda", "rri", "temp"])
        act_data, eda_data, rri_data, temp_data = tuple(load_h5_data(f"{DATA_PATH}/biometric_data/{fn}") for fn in [act, eda, rri, temp])
        
        integrated_data.append({
            "subID": _subID,
            "date": _date,
            "valence": _valence,
            "arousal": _arousal,
            "exist_data": (
                act_data is not None and
                eda_data is not None and
                rri_data is not None and
                temp_data is not None
            ),
            "act_data": act_data,
            "eda_data": eda_data,
            "rri_data": rri_data,
            "temp_data": temp_data
        })
    
    return integrated_data


def plot_wave(data, data_kind, data_row=0, save_at=None):
    """バイオメトリックデータの波形を描画する"""
    if not data["exist_data"]:
        print("データが存在しません")
        return False
    
    # データ種別に応じた詳細情報を設定
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
    
    info = data_info.get(data_kind, {"name": data_kind.upper(), "unit": "", "color": "blue", "descriptions": ["Unknown data"]})
    
    # data_rowに対応する説明を取得
    description = info["descriptions"][data_row] if data_row < len(info["descriptions"]) else f"Row {data_row} data"
    
    x = np.linspace(0, 15, data[f"{data_kind}_data"].shape[1])
    y = data[f"{data_kind}_data"][data_row]
    
    plt.figure(figsize=(12, 8))
    plt.plot(x, y, color=info["color"], linewidth=1.5)
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
    else:
        plt.show()

    return True

