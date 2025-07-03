import os
from datetime import datetime
from util.utils import load_csv_data, search_data, draw_hist, object_to_json, plot_15minutes_waves

# 一度メタデータを可能な限り収集して一覧として表示できるようにしておく
# 1. 被験者ベース：全被験者数、データが存在する被験者数、全被験者間で各データの最小・最大・分散、各被験者の各データの有無、各被験者の各データの最小・最大・分散
# 2. データタイプベース：

if __name__ == "__main__":
    
    DATA_PATH = "/home/mori/projects/affective-forecast/datas"
    timestamp_path = "meta_data/timestamp.csv"
    timestamp_data = load_csv_data(f"{DATA_PATH}/{timestamp_path}")
    print(timestamp_data)

    times = []
    
    file_name = os.path.splitext(os.path.basename(__file__))[0]
    output_path = f"/home/mori/projects/affective-forecast/plots/{file_name}"
    os.makedirs(output_path, exist_ok=True)

    for i, tmp in enumerate(timestamp_data):
        time_str = tmp["datetime"].replace("\'","")  # シングルクォートを除去
        
        # datetime文字列をパース
        dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
        
        # 時刻部分を取得（0-23の整数）
        hour = dt.hour
        times.append(hour)
        

    draw_hist(
        data=times, 
        bins=24,  # 24時間分
        x_label="Hour of Day", 
        y_label="Measurement Count", 
        title="Histogram - Measurement counts by hour of day",
        save_at=f"{output_path}/hist_hourly"
    )