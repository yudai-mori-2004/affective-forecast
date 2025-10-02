import os
from util.utils3 import load_h5_data, search_and_get_filenames
import pandas as pd
import numpy as np

# EDAの生データをCSV形式でエクスポート

if __name__ == "__main__":
    
    # パス定義
    data_path = f"/home/mori/projects/affective-forecast/datas/biometric_data"
    output_path = f"/home/mori/projects/affective-forecast/plots/EDAの生データCSV"
    
    # フォルダが存在しない場合は作成
    os.makedirs(output_path, exist_ok=True)

    data_names = search_and_get_filenames({
        "ex-term": "term1"
    })

    for i, name in enumerate(data_names):
        eda_file_name = f"{name}_eda.h5"
        eda = load_h5_data(f"{data_path}/{eda_file_name}")

        if eda is not None:
            time = np.linspace(0, 15, eda.shape[1])
            eda_values = eda[0]
            
            df = pd.DataFrame({
                'time_minutes': time,
                'eda_microsiemens': eda_values
            })
            
            csv_filename = f"{output_path}/{name}_eda_raw.csv"
            df.to_csv(csv_filename, index=False)

            print(f"Saved {name} EDA data to CSV ({i+1}/{len(data_names)})")
    