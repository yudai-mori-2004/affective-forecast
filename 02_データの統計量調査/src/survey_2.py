import os
from util.utils import search_data, draw_hist, object_to_json, plot_15minutes_waves

# RRIについて、感情価（Valence）が 4（最高）と -4（最低）のものをすべて図示してみる。
# 実際の波形データを観察してこの2群の性質に違いがないか確かめる

if __name__ == "__main__":
    
    # ファイル名を取得（拡張子なし）
    file_name = os.path.splitext(os.path.basename(__file__))[0]
    
    # 出力パスを定義
    output_path_max = f"/home/mori/projects/affective-forecast/plots/{file_name}/max_valence"
    output_path_min = f"/home/mori/projects/affective-forecast/plots/{file_name}/min_valence"
    
    # フォルダが存在しない場合は作成
    os.makedirs(output_path_max, exist_ok=True)
    os.makedirs(output_path_min, exist_ok=True)

    max_valence = search_data(valence=4, data_kinds=["rri"])[:]
    for i, max in enumerate(max_valence):
        plot_15minutes_waves(
            datas=[max],
            data_kind="rri", 
            data_row=0, 
            data_range=[0.30,1.30],
            save_at=f"{output_path_max}/{max["subID"]}-rri-{max["date"]}")
        print(f"Process: max: {i}/{len(max_valence)}")
    
    min_valence = search_data(valence=-4, data_kinds=["rri"])[:]
    for i, min in enumerate(min_valence):
        plot_15minutes_waves(
            datas=[min],
            data_kind="rri", 
            data_row=0, 
            data_range=[0.30,1.30],
            save_at=f"{output_path_min}/{min["subID"]}-rri-{min["date"]}")
        print(f"Process: min: {i + 1}/{len(min_valence)}")
