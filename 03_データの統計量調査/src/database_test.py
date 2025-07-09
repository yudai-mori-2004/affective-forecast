from util.utils3 import create_index_files
import os
import numpy as np
import matplotlib.pyplot as plt
import json
import datetime
import hashlib
from typing import Union, Optional, Dict, List


if __name__ == "__main__":
    # インデックスファイルを作成
    create_index_files()
    
    # テストコード
    from util.utils3 import search_and_get_filenames
    
    # テスト1: 単一フィールドでの完全一致検索
    print("=== テスト1: valence=1の検索 ===")
    search_conditions = {"valence": 1}
    filenames = search_and_get_filenames(search_conditions)
    print(f"マッチした件数: {len(filenames)}")
    print(f"最初の5件: {filenames[:5]}")
    
    # テスト2: 範囲指定検索
    print("\n=== テスト2: valence範囲[1,3]の検索 ===")
    search_conditions = {"valence": [1, 3]}
    filenames = search_and_get_filenames(search_conditions)
    print(f"マッチした件数: {len(filenames)}")
    print(f"最初の5件: {filenames[:5]}")
    
    # テスト3: 複数条件のAND検索
    print("\n=== テスト3: valence=2 AND arousal=1の検索 ===")
    search_conditions = {"valence": 2, "arousal": 1}
    filenames = search_and_get_filenames(search_conditions)
    print(f"マッチした件数: {len(filenames)}")
    print(f"最初の5件: {filenames[:5]}")
    
    # テスト4: 複雑な条件
    print("\n=== テスト4: 複雑な条件の検索 ===")
    search_conditions = {
        "valence": 1,
        "arousal": -1,
        "gender": 1,
        "ID": "S117"
    }
    filenames = search_and_get_filenames(search_conditions)
    print(f"マッチした件数: {len(filenames)}")
    print(f"最初の5件: {filenames[:5]}")
    
    print("\n=== テスト完了 ===")
# python 03_データの統計量調査/src/a.py
