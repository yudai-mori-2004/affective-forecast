import os
from util.utils import search_data, draw_hist, object_to_json

# S01~S119までの合計 119 名に対して、各被験者が何回計測を行ったかをヒストグラムで表示

if __name__ == "__main__":

    start = 1
    end = 119

    nums=[]
    
    # ファイル名を取得（拡張子なし）
    file_name = os.path.splitext(os.path.basename(__file__))[0]
    
    # 出力パスを定義
    output_path = f"/home/mori/projects/affective-forecast/plots"
    
    # フォルダが存在しない場合は作成
    os.makedirs(output_path, exist_ok=True)

    for i in range(start, end + 1):
        ID = ""
        if i < 10:
            ID = f"S0{i}"
        else:
            ID = f"S{i}"
            
        data_list = search_data(subID = ID)
        nums.append({
            ID: len(data_list)
        })
        print(f"Progress: {i}/{end - start + 1}")

    values = [list(d.values())[0] for d in nums]

    draw_hist(
        data=values, 
        bins=20, 
        x_label="Number of measurements", 
        y_label="Number of subjects", 
        title="Histgram - Number of subjects per measurements count",
        save_at=f"{output_path}/{file_name}/hist"
        )
    
    object_to_json(nums, f"{output_path}/{file_name}/obj.json")