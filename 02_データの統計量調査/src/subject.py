#!/usr/bin/env python3
from util.utils import search_data, plot_waves

if __name__ == "__main__":
    # 使用例: 特定の被験者のデータを検索して描画
    data_list = search_data(subID="S01")
    print(f"検索結果: {len(data_list)}件のデータが見つかりました")
    
    # データが存在する最初の項目を描画
    for i, data in enumerate(data_list):
        if data["exist_data"]:
           # print(f"[{i}]番目のデータを描画します")
            plot_waves([data], "eda", 0, save_at=f"/home/mori/projects/affective-forecast/plots/Day02/eda/plt_{data["date"]}")
            
    else:
        print("描画可能なデータが見つかりませんでした")