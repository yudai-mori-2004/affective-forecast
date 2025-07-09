import os
from datetime import datetime
import numpy as np
from itertools import combinations, product
import math
import pandas as pd
from scipy import stats

import csv
from util.utils import load_csv_data, search_data, draw_hist, object_to_json, plot_15minutes_waves, load_h5_data

# 複数のラベリングパターンを用いた「計測データ」の基本統計分析
# 「感情データ」に対しても同様のことを次回行う
# 
# ラベリング次元:
# 1. Data type: [act_accx, act_accy, eda, rri, temp, ...]
# 2. Subject: [0~100 sessions, 100~200sessions, ... , 500~600 sessions]  
# 3. Session: [valence -4, -3, ..., 3, 4] [arousal -4, -3, ..., 3, 4] [0:00~3:00, 3:00~6:00, ..., 21:00~24:00]
# 4. Value: [0~5min, 5~10min, 10~15min] 
#
# 実行する基本統計分析:
# maximum, minimum, range, mean, median, variance, standard deviation, autocorrelation, trend
# 
# 分析レベル:
# 1. データタイプ別に、2,3,4の条件付けはなしで統計分析 （10回）
# 2. データタイプ別に、2のラベリングパターンは使い,3,4の条件付けはなしで統計分析　（10*(1+6)=130回）
# 3. データタイプ別に、3のラベリングパターンは使い,2,4の条件付けはなしで統計分析 （10*(1+9+9+8+(9*9+9*8+9*8))=2520回
# 4. データタイプ別に、4のラベリングパターンは使い,2,3の条件付けはなしで統計分析 （10*(1+3)=40回）
#  ↑ 今回はここまで。
#
# 5. データタイプ別に、2,3のラベリングパターンは使い,4の条件付けはなしで統計分析
# 6. データタイプ別に、2,4のラベリングパターンは使い,3の条件付けはなしで統計分析
# 7. データタイプ別に、3,4のラベリングパターンは使い,2の条件付けはなしで統計分析
# 8. データタイプ別に、2,3,4のラベリングパターンすべてを使い統計分析
#
# 例：3の場合の回数計算は、(データタイプ数*(制限なし + valence9パターン + arrousal9パターン + 時間8パターン + (valence9パターン*arrousal9パターン + valence9パターン*時間8パターン + arrousal9パターン*時間8パターン)))という意味です
#
# 出力：各ラベリングパターンの組み合わせに対する統計サマリーテーブル


# グローバルキャッシュ
_h5_cache = {}

def search_data_by_query(data_kind, act_kind, queries):
    # """条件に一致するデータを検索して統合データのリストを返す"""
    print(f"[DEBUG] search_data_by_query called with: data_kind={data_kind}, act_kind={act_kind}, queries={queries}")
    DATA_PATH = "/home/mori/projects/affective-forecast/datas"
    timestamp_path = "meta_data/timestamp.csv"
    
    print(f"[DEBUG] Loading timestamp data from: {DATA_PATH}/{timestamp_path}")
    timestamp_data = load_csv_data(f"{DATA_PATH}/{timestamp_path}")
    print(f"[DEBUG] Loaded {len(timestamp_data)} timestamp records")

    data_kinds = ["act", "eda", "rri", "temp"] 
    act_kinds = ["accx", "accy", "accz", "VI", "gmx", "gmy", "gmz", "bmx", "bmy", "bmz"]

    integrated_data = []

    # クエリを解析
    print(f"[DEBUG] Parsing queries: {queries}")
    all_query_values = parse_queries(queries)
    print(f"[DEBUG] Parsed query values: {all_query_values}")

    # 解析結果から値を取得
    session_count_range = all_query_values.get('c', [[0, 1000]])[0]  # デフォルト値
    valence_values = all_query_values.get('v', None)
    arousal_values = all_query_values.get('a', None)
    time_values = all_query_values.get('t', None)
    segment_values = all_query_values.get('s', None)
    
    print(f"[DEBUG] Filter conditions - Session count: {session_count_range}, Valence: {valence_values}, Arousal: {arousal_values}, Time: {time_values}, Segment: {segment_values}")

    # 被験者IDごとのセッション数を計算
    subID_list = [x["ID"] for x in timestamp_data]
    subID_counts = {}
    for subID in subID_list:
        subID_counts[subID] = subID_counts.get(subID, 0) + 1
    
    print(f"[DEBUG] Found {len(subID_counts)} unique subjects")
    print(f"[DEBUG] Subject session counts: {list(subID_counts.items())[:5]}...")  # 最初の5件のみ表示

    # セッション数による被験者フィルタリング
    filtered_subIDs = [subID for subID, count in subID_counts.items() 
                        if session_count_range[0] <= count < session_count_range[1]]
    print(f"[DEBUG] Filtered subjects count: {len(filtered_subIDs)}")

    # 条件に一致するデータをフィルタリング
    print(f"[DEBUG] Starting data filtering...")
    processed_count = 0
    matched_count = 0
    MAX_RECORDS = 100  # メモリ不足を防ぐための上限
    
    for data in timestamp_data:
        processed_count += 1
        
        # 上限に達したら処理を停止
        if matched_count >= MAX_RECORDS:
            print(f"[DEBUG] Reached maximum record limit ({MAX_RECORDS}), stopping data loading")
            break
        
        # 被験者条件チェック
        if data["ID"] not in filtered_subIDs:
            continue
            
        # 感情価条件チェック
        if valence_values and int(data["valence"]) not in valence_values:
            continue
            
        # 覚醒度条件チェック
        if arousal_values and int(data["arousal"]) not in arousal_values:
            continue
            
        # 時刻条件チェック
        if time_values:
            dt = datetime.strptime(data["datetime"].replace("'", ""), "%Y-%m-%d %H:%M:%S")
            hour = dt.hour
            time_match = False
            for time_range in time_values:
                if isinstance(time_range, list):
                    if time_range[0] <= hour < time_range[1]:
                        time_match = True
                        break
                else:
                    if hour == time_range:
                        time_match = True
                        break
            if not time_match:
                continue
        
        # 条件に一致するデータを追加（キャッシュ使用）
        file_key = f"{data['index']}_{data_kind}"
        if file_key not in _h5_cache:
            _h5_cache[file_key] = load_h5_data(f"{DATA_PATH}/biometric_data/data_{data['index']}_E4_{data_kind}.h5")
        
        wave_datas = _h5_cache[file_key]
        field_name = f"{data_kind}{"_" if act_kind is not None else ""}{act_kind}"
        wave_data = None if wave_datas is None else wave_datas[0] if act_kind == "" else wave_datas[act_kinds.index(act_kind)]

        if wave_data is not None and segment_values is not None:
            print(f"[DEBUG] Applying segment filter: {segment_values}")
            window = [len(wave_data)/segment_values[0], len(wave_data)/segment_values[1]]
            wave_data = wave_data[window[0]:window[1]]
            print(f"[DEBUG] Segmented data length: {len(wave_data) if wave_data is not None else 0}")
        
        integrated_data.append({
            "subID": data["ID"],
            "date": data["datetime"],
            "valence": data["valence"],
            "arousal": data["arousal"],
            "exist_data": (
                wave_datas is not None
            ),
            field_name: wave_data,
        })
        matched_count += 1
        
        if matched_count % 10 == 0:
            print(f"[DEBUG] Matched {matched_count} data records so far")

    print(f"[DEBUG] Filtering completed. Total matched: {matched_count}/{len(timestamp_data)}")
    if matched_count >= MAX_RECORDS:
        print(f"[DEBUG] WARNING: Data was limited to {MAX_RECORDS} records to prevent memory issues")
    print(f"[DEBUG] Valid data count: {len([d for d in integrated_data if d['exist_data']])}")
    print(f"[DEBUG] Invalid data count: {len([d for d in integrated_data if not d['exist_data']])}")
    
    # キャッシュサイズ監視
    if len(_h5_cache) > 100:
        print(f"[WARNING] H5 cache size: {len(_h5_cache)} files. Consider clearing cache.")
    
    return integrated_data







def calculate_statistics(data):
    """
    analyze_dataに対して基本統計分析を実行
    
    基本統計量（平均、分散、最大最小、レンジ、中央値、標準偏差）：
    - 全時系列データを平坦化して統計計算
    
    自己相関・トレンド分析：
    - 各時系列で個別に計算し、サンプル間で平均化
    """
    
    # dataは時系列データのリスト（各要素が1つのセッションの時系列）
    if not data:
        return None
    
    # 基本統計量用：全データを平坦化
    flattened_data = []
    for session_data in data:
        if isinstance(session_data, (list, np.ndarray)) and len(session_data) > 0:
            if isinstance(session_data, np.ndarray):
                flattened_data.extend(session_data.flatten().tolist())
            else:
                flattened_data.extend(session_data)
    
    if not flattened_data:
        return None
    
    y_flat = np.array(flattened_data)
    
    # 基本統計量
    stats_result = {
        "sample_count": len(y_flat),
        'maximum': np.max(y_flat),
        'minimum': np.min(y_flat),
        'range': np.max(y_flat) - np.min(y_flat),
        'mean': np.mean(y_flat),
        'median': np.median(y_flat),
        'variance': np.var(y_flat),
        'std': np.std(y_flat),
    }
    
    # 自己相関・トレンド分析用：各セッションで個別計算してから平均化
    autocorr_results = []
    trend_results = []
    
    for session_data in data:
        if isinstance(session_data, (list, np.ndarray)) and len(session_data) > 0:
            y_session = np.array(session_data)
            
            # 各セッションの自己相関
            autocorr = calculate_autocorrelation(y_session, max_lag=50)
            autocorr_results.append(autocorr)
            
            # 各セッションのトレンド
            trend_result = calculate_trend(y_session)
            trend_results.append(trend_result)
    
    # 自己相関の平均化
    if autocorr_results:
        # 各ラグでの自己相関を平均化
        max_lag = min(len(autocorr) for autocorr in autocorr_results)
        avg_autocorr = [np.mean([autocorr[lag] for autocorr in autocorr_results]) for lag in range(max_lag)]
        
        stats_result['autocorr_lag1'] = avg_autocorr[1] if len(avg_autocorr) > 1 else 0
        stats_result['autocorr_lag5'] = avg_autocorr[5] if len(avg_autocorr) > 5 else 0
        stats_result['autocorr_lag10'] = avg_autocorr[10] if len(avg_autocorr) > 10 else 0
    else:
        stats_result['autocorr_lag1'] = 0
        stats_result['autocorr_lag5'] = 0
        stats_result['autocorr_lag10'] = 0
    
    # トレンド分析の平均化
    if trend_results:
        stats_result['trend_slope'] = np.mean([t['slope'] for t in trend_results])
        stats_result['trend_r_squared'] = np.mean([t['r_squared'] for t in trend_results])
        stats_result['trend_p_value'] = np.mean([t['p_value'] for t in trend_results])
    else:
        stats_result['trend_slope'] = 0
        stats_result['trend_r_squared'] = 0
        stats_result['trend_p_value'] = 1
    
    return stats_result


def calculate_autocorrelation(data, max_lag=50):
    """自己相関を計算"""
    autocorr = []
    for lag in range(max_lag + 1):
        if lag == 0:
            autocorr.append(1.0)
        elif lag >= len(data):
            autocorr.append(0.0)
        else:
            corr = np.corrcoef(data[:-lag], data[lag:])[0, 1]
            autocorr.append(corr if not np.isnan(corr) else 0.0)
    return autocorr

def calculate_trend(data):
    """トレンド分析"""
    x = np.arange(len(data))
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, data)
    
    return {
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_value**2,
        'p_value': p_value,
        'std_err': std_err
    }

def create_analysis_label(data_kind, act_kind, using_queries):
    """分析ラベルを作成（可読性重視）"""
    label_parts = []
    
    # データタイプ
    if act_kind:
        label_parts.append(f"{data_kind}_{act_kind}")
    else:
        label_parts.append(data_kind)
    
    # 条件ラベルを可読性重視で変換
    conditions = []
    
    # using_queriesを解析
    if using_queries:
        for query in using_queries:
            if query.startswith('c'):
                range_str = query[1:]
                if '~' in range_str:
                    start, end = range_str.split('~')
                    conditions.append(f"Sessions({start}-{end})")
                else:
                    conditions.append(f"Sessions({range_str})")
            elif query.startswith('v'):
                val = query[1:]
                if val == '0':
                    conditions.append("Valence(Neutral)")
                elif val.startswith('-'):
                    conditions.append(f"Valence(Negative{val})")
                else:
                    conditions.append(f"Valence(Positive{val})")
            elif query.startswith('a'):
                val = query[1:]
                if val == '0':
                    conditions.append("Arousal(Neutral)")
                elif val.startswith('-'):
                    conditions.append(f"Arousal(Low{val})")
                else:
                    conditions.append(f"Arousal(High{val})")
            elif query.startswith('t'):
                time_str = query[1:]
                if '~' in time_str:
                    start_time, end_time = time_str.split('~')
                    start_h = start_time.split(':')[0]
                    end_h = end_time.split(':')[0]
                    conditions.append(f"Time({start_h}h-{end_h}h)")
            elif query.startswith('s'):
                segment_str = query[1:]
                if '~' in segment_str:
                    start, end = segment_str.split('~')
                    conditions.append(f"Segment({start}-{end}min)")
                else:
                    conditions.append(f"Segment({segment_str}min)")
    
    # 条件がない場合
    if conditions:
        label_parts.append(" + ".join(conditions))
    else:
        label_parts.append("NoCondition")
    
    return " | ".join(label_parts)


def save_statistics_to_csv(all_statistics, file_name):
    """統計結果をCSVファイルに保存"""
    if not all_statistics:
        return
    
    # 出力パスを定義
    output_path = f"/home/mori/projects/affective-forecast/plots/{file_name}"
    os.makedirs(output_path, exist_ok=True)
    
    # CSVファイルのパス
    csv_file_path = f"{output_path}/statistics_summary.csv"
    
    # CSVのヘッダー（基本情報を含む）
    basic_info_keys = ['data_count', 'valid_data_count', 'invalid_data_count', 'data_availability_rate', 'unique_subjects', 'date_range_start', 'date_range_end']
    stats_keys = list(all_statistics[0]['stats'].keys()) if all_statistics[0]['stats'] else []
    
    fieldnames = ['analysis_label', 'data_kind', 'act_kind'] + basic_info_keys + stats_keys
    
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for stat_data in all_statistics:
            row = {
                'analysis_label': stat_data['label'],
                'data_kind': stat_data['data_kind'],
                'act_kind': stat_data['act_kind'],
                **stat_data['basic_info']
            }
            
            if stat_data['stats']:
                row.update(stat_data['stats'])
            
            writer.writerow(row)
    
    print(f"Statistics saved to: {csv_file_path}")


def generate_combinations(sets):
    """
    各セットから 0 個〜全てのセットを選び，選んだセットの要素を
    1 つずつ組み合わせたリストを返す。
    """
    result = [[]]  # まず 0 個選ぶパターンとして空リスト
    n = len(sets)
    for k in range(1, n+1):
        # どのセットを k 個選ぶかのインデックスの組み合わせ
        for idxs in combinations(range(n), k):
            # 選んだセット同士の直積
            for prod in product(*(sets[i] for i in idxs)):
                result.append(list(prod))
    return result




# メイン処理に追加する部分
def process_statistics_analysis():
    """統計分析のメイン処理"""
    file_name = os.path.splitext(os.path.basename(__file__))[0]
    
    data_kinds = ["act", "eda", "rri", "temp"] 
    act_kinds = ["accx", "accy", "accz", "VI", "gmx", "gmy", "gmz", "bmx", "bmy", "bmz"]

    subject_queries_set = [["c0~100", "c100~200", "c200~300", "c300~400", "c400~500", "c500~600"]]
    session_queries_set = [["v-4", "v-3", "v-2", "v-1", "v0", "v1", "v2", "v3", "v4"], 
                      ["a-4", "a-3", "a-2", "a-1", "a0", "a1", "a2", "a3", "a4"], 
                      ["t0:00~3:00", "t3:00~6:00", "t6:00~9:00", "t9:00~12:00", "t12:00~15:00", "t15:00~18:00", "t18:00~21:00", "t21:00~24:00"]]
    value_queries_set = [["s0~5", "s5~10", "s10~15"]]

    all_statistics = []
    batch_size = 1
    
    # CSVファイルの初期化
    output_path = f"/home/mori/projects/affective-forecast/plots/{file_name}"
    os.makedirs(output_path, exist_ok=True)
    csv_file_path = f"{output_path}/statistics_summary.csv"
    csv_initialized = False
    
    # 処理カウンター
    total_processed = 0
    successful_analyses = 0
    failed_analyses = 0

    for using_queries_set in [subject_queries_set, session_queries_set, value_queries_set]:
        for k, kind in enumerate(data_kinds):
            if kind == "act":

                for j, act in enumerate(act_kinds):
                    queries_pattern = generate_combinations(using_queries_set)

                    for using_queries in queries_pattern:
                        total_processed += 1
                        # データ取得
                        analyze_data = search_data_by_query(kind, act, using_queries)
                        
                        field_name = f"{kind}{"_" if act != "" else ""}{act}"
                        stats_result = calculate_statistics([d[field_name] for d in analyze_data if d["exist_data"]])
                        
                        print(f"[DEBUG] Processing {kind}_{act} with {len(using_queries)} conditions: {len(analyze_data)} samples")
                        
                        # ラベル作成
                        label = create_analysis_label(kind, act, using_queries)
                        
                        # 基本情報を追加
                        basic_info = {
                            'data_count': len(analyze_data),
                            'valid_data_count': len([d for d in analyze_data if d.get('exist_data', False)]),
                            'invalid_data_count': len([d for d in analyze_data if not d.get('exist_data', False)]),
                            'data_availability_rate': len([d for d in analyze_data if d.get('exist_data', False)]) / len(analyze_data) * 100 if analyze_data else 0,
                            'unique_subjects': len(set(d.get('subID', '') for d in analyze_data)),
                            'date_range_start': min([d.get('date', '') for d in analyze_data]) if analyze_data else '',
                            'date_range_end': max([d.get('date', '') for d in analyze_data]) if analyze_data else ''
                        }
                        
                        # 結果保存
                        if stats_result:
                            successful_analyses += 1
                            all_statistics.append({
                                'label': label,
                                'data_kind': kind,
                                'act_kind': act,
                                'basic_info': basic_info,
                                'stats': stats_result
                            })
                        else:
                            failed_analyses += 1
                            all_statistics.append({
                                'label': label,
                                'data_kind': kind,
                                'act_kind': act,
                                'basic_info': basic_info,
                                'stats': None
                            })
                        
                        print(f"Processed ({total_processed}): {label} - Data: {basic_info['data_count']}, Valid: {basic_info['valid_data_count']}")
                        
                        # バッチサイズに達したらCSVに書き込み
                        if len(all_statistics) >= batch_size:
                            write_batch_to_csv(all_statistics, csv_file_path, csv_initialized)
                            csv_initialized = True
                            all_statistics = []  # バッチをクリア
                            print(f"Batch written to CSV (Total processed: {total_processed})")
                                    
            else:
                # act_kind ではなく、act="" として処理
                queries_pattern = generate_combinations(using_queries_set)
                
                for j, using_queries in enumerate(queries_pattern):
                    total_processed += 1
                    # データ取得
                    analyze_data = search_data_by_query(kind, "", using_queries)

                    # フィールド名は act がないぶん kind だけ
                    field_name = kind

                    # 統計計算
                    stats_result = calculate_statistics([d[field_name] for d in analyze_data if d["exist_data"]])
                    
                    print(f"[DEBUG] Processing {kind} with {len(using_queries)} conditions: {len(analyze_data)} samples")
                    
                    label = create_analysis_label(kind,"",using_queries)

                    # 基本情報を追加
                    basic_info = {
                        'data_count': len(analyze_data),
                        'valid_data_count': len([d for d in analyze_data if d.get('exist_data', False)]),
                        'invalid_data_count': len([d for d in analyze_data if not d.get('exist_data', False)]),
                        'data_availability_rate': (
                            len([d for d in analyze_data if d.get('exist_data', False)]) / len(analyze_data) * 100
                            if analyze_data else 0
                        ),
                        'unique_subjects': len(set(d.get('subID', '') for d in analyze_data)),
                        'date_range_start': min((d.get('date','') for d in analyze_data), default=''),
                        'date_range_end':   max((d.get('date','') for d in analyze_data), default='')
                    }

                    # 結果保存
                    if stats_result:
                        successful_analyses += 1
                        all_statistics.append({
                            'label':       label,
                            'data_kind':   kind,
                            'act_kind':    "",
                            'basic_info':  basic_info,
                            'stats':       stats_result
                        })
                    else:
                        failed_analyses += 1
                        all_statistics.append({
                            'label':       label,
                            'data_kind':   kind,
                            'act_kind':    "",
                            'basic_info':  basic_info,
                            'stats':       None
                        })

                    print(
                        f"Processed ({total_processed}): {label} - "
                        f"Data: {basic_info['data_count']}, "
                        f"Valid: {basic_info['valid_data_count']}"
                    )

                    # バッチサイズに達したら CSV に書き込み
                    if len(all_statistics) >= batch_size:
                        write_batch_to_csv(all_statistics, csv_file_path, csv_initialized)
                        csv_initialized = True
                        all_statistics = []  # バッチをクリア
                        print(f"Batch written to CSV (Total processed: {total_processed})")


    # 残りのデータをCSVに書き込み
    if all_statistics:
        write_batch_to_csv(all_statistics, csv_file_path, csv_initialized)
        print(f"Final batch written to CSV")
    
    # 基本情報サマリー
    print(f"\n=== Analysis Summary ===")
    print(f"Total processed: {total_processed}")
    print(f"Successful analyses: {successful_analyses}")
    print(f"Failed analyses: {failed_analyses}")
    print(f"Success rate: {successful_analyses/total_processed*100:.1f}%")
    print(f"Results saved to: {csv_file_path}")



def write_batch_to_csv(batch_statistics, csv_file_path, file_exists):
    """バッチをCSVファイルに書き込み"""
    if not batch_statistics:
        return
    
    # ヘッダー情報を取得
    basic_info_keys = ['data_count', 'valid_data_count', 'invalid_data_count', 'data_availability_rate', 'unique_subjects', 'date_range_start', 'date_range_end']
    stats_keys = list(batch_statistics[0]['stats'].keys()) if batch_statistics[0]['stats'] else []
    fieldnames = ['analysis_label', 'data_kind', 'act_kind'] + basic_info_keys + stats_keys
    
    # ファイルモードを決定
    mode = 'a' if file_exists else 'w'
    
    with open(csv_file_path, mode, newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # 初回のみヘッダーを書き込み
        if not file_exists:
            writer.writeheader()
        
        for stat_data in batch_statistics:
            row = {
                'analysis_label': stat_data['label'],
                'data_kind': stat_data['data_kind'],
                'act_kind': stat_data['act_kind'],
                **stat_data['basic_info']
            }
            
            if stat_data['stats']:
                row.update(stat_data['stats'])
            
            writer.writerow(row)

if __name__ == "__main__":
   #process_statistics_analysis()
   print(search_data_by_query("rri","",["c700~1000"]))