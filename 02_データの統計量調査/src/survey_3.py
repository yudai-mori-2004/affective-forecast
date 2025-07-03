import os
from util.utils import search_data, draw_hist, object_to_json, plot_15minutes_waves

# 全計測データについて、感情価（Valence）が 4（最高）と -4（最低）のものをすべて図示してみる。
# 実際の波形データを観察してこの2群の性質に違いがないか確かめる

if __name__ == "__main__":
    
    # ファイル名を取得（拡張子なし）
    file_name = os.path.splitext(os.path.basename(__file__))[0]
    
    data_kinds = ["act", "eda", "rri", "temp"] 
    act_kinds = ["accx", "accy", "accz", "VI", "gmx", "gmy", "gmz", "bmx", "bmy", "bmz"]

    for k, kind in enumerate(data_kinds):

        if kind == "act":
            for j, act in enumerate(act_kinds):
                # 出力パスを定義
                output_path_max = f"/home/mori/projects/affective-forecast/plots/{file_name}/{kind}_{act}/max_valence"
                output_path_min = f"/home/mori/projects/affective-forecast/plots/{file_name}/{kind}_{act}/min_valence"
                
                # フォルダが存在しない場合は作成
                os.makedirs(output_path_max, exist_ok=True)
                os.makedirs(output_path_min, exist_ok=True)

                max_valence = search_data(valence=4, data_kinds=[kind])[:]

                data_range = [min([min(values[f"{kind}_data"][act_kinds.index(act)]) for values in max_valence if values["exist_data"]]), 
                                max([max(values[f"{kind}_data"][act_kinds.index(act)]) for values in max_valence  if values["exist_data"]])]

                for i, m in enumerate(max_valence):
                    plot_15minutes_waves(
                        datas=[m],
                        data_kind=kind, 
                        data_row=act_kinds.index(act), 
                        data_range=data_range,
                        save_at=f"{output_path_max}/{m["subID"]}-{kind}_{act}-{m["date"]}")
                    print(f"Process: {kind}_{act} max: {i + 1}/{len(max_valence)}")
                
                min_valence = search_data(valence=-4, data_kinds=[kind])[:]

                data_range = [min([min(values[f"{kind}_data"][act_kinds.index(act)]) for values in min_valence if values["exist_data"]]), 
                                max([max(values[f"{kind}_data"][act_kinds.index(act)]) for values in min_valence  if values["exist_data"]])]

                for i, n in enumerate(min_valence):
                    plot_15minutes_waves(
                        datas=[n],
                        data_kind=kind, 
                        data_row=act_kinds.index(act), 
                        data_range=data_range,
                        save_at=f"{output_path_min}/{n["subID"]}-{kind}_{act}-{n["date"]}")
                    print(f"Process: {kind}_{act} min: {i + 1}/{len(min_valence)}")
        else: 
            # 出力パスを定義
            output_path_max = f"/home/mori/projects/affective-forecast/plots/{file_name}/{kind}/max_valence"
            output_path_min = f"/home/mori/projects/affective-forecast/plots/{file_name}/{kind}/min_valence"
            
            # フォルダが存在しない場合は作成
            os.makedirs(output_path_max, exist_ok=True)
            os.makedirs(output_path_min, exist_ok=True)

            max_valence = search_data(valence=4, data_kinds=[kind])[:]

            data_range = [min([min(values[f"{kind}_data"][0]) for values in max_valence if values["exist_data"]]), 
                            max([max(values[f"{kind}_data"][0]) for values in max_valence  if values["exist_data"]])]

            for i, m in enumerate(max_valence):
                plot_15minutes_waves(
                    datas=[m],
                    data_kind=kind, 
                    data_row=0, 
                    data_range=data_range,
                    save_at=f"{output_path_max}/{m["subID"]}-{kind}-{m["date"]}")
                print(f"Process: {kind} max: {i + 1}/{len(max_valence)}")
            
            min_valence = search_data(valence=-4, data_kinds=[kind])[:]

            data_range = [min([min(values[f"{kind}_data"][0]) for values in min_valence if values["exist_data"]]), 
                            max([max(values[f"{kind}_data"][0]) for values in min_valence  if values["exist_data"]])]

            for i, n in enumerate(min_valence):
                plot_15minutes_waves(
                    datas=[n],
                    data_kind=kind, 
                    data_row=0, 
                    data_range=data_range,
                    save_at=f"{output_path_min}/{n["subID"]}-{kind}-{n["date"]}")
                print(f"Process: {kind} min: {i + 1}/{len(min_valence)}")
