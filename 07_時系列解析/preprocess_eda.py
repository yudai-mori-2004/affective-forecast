"""
EDAデータの前処理スクリプト

以下のデータを事前計算してh5ファイルとして保存：
1. eda_lowpass: ローパスフィルタ適用済みEDA
2. eda_masked: マスク適用EDA（アーチファクト除去）
3. scl: SCL（Tonic成分、cvxEDA）
4. scr: SCR（Phasic成分、cvxEDA）
"""
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import h5py
import numpy as np
from scipy import signal
from scipy.ndimage import median_filter as scipy_median_filter
from tqdm import tqdm
from src.util.utils3 import load_h5_data, search_by_conditions, get_affective_datas_from_uuids

try:
    import neurokit2 as nk
    HAS_NEUROKIT = True
except ImportError:
    HAS_NEUROKIT = False
    print("Warning: neurokit2 not installed. SCL/SCR will not be generated.")


def apply_eda_mask(eda_original: np.ndarray, Fs: float = 4.0,
                   median_window: int = 17, median_threshold: float = 0.30) -> np.ndarray:
    """EDAデータにマスクを適用"""
    # threshold filter
    extend_sec = 5
    extend_samples = int(Fs * extend_sec)
    thr_mask = (eda_original < 0.05) | (eda_original > 60)
    thr_mask = np.convolve(thr_mask.astype(int), np.ones(2*extend_samples+1, dtype=int), mode='same') > 0

    # median filter
    med = scipy_median_filter(eda_original, size=median_window, mode='nearest')
    med_mask = np.abs((eda_original / (med + 1e-8)) - 1.0) > median_threshold

    # 全マスクを統合
    combined_mask = thr_mask | med_mask

    # マスク部分をNaNに設定
    eda_masked = eda_original.copy()
    eda_masked[combined_mask] = np.nan

    return eda_masked


def extract_scl_scr(eda: np.ndarray, Fs: float = 4.0, verbose: bool = False):
    """EDAからSCL（Tonic）とSCR（Phasic）を分離"""
    if not HAS_NEUROKIT:
        return None, None

    try:
        # float64に変換（cvxEDAはfloat64を要求）
        eda = np.asarray(eda, dtype=np.float64)

        if verbose:
            print(f"  EDA shape: {eda.shape}, dtype: {eda.dtype}, contiguous: {eda.flags['C_CONTIGUOUS']}")

        # NaNを補間
        if np.any(np.isnan(eda)):
            valid_indices = np.where(~np.isnan(eda))[0]
            if len(valid_indices) > 1:
                eda_interpolated = np.interp(
                    np.arange(len(eda)),
                    valid_indices,
                    eda[valid_indices]
                )
            else:
                eda_interpolated = np.nan_to_num(eda, nan=0.0)
        else:
            eda_interpolated = eda

        # 配列をC連続にする
        eda_interpolated = np.ascontiguousarray(eda_interpolated, dtype=np.float64)

        if verbose:
            print(f"  Interpolated shape: {eda_interpolated.shape}, dtype: {eda_interpolated.dtype}")

        # cvxEDAでトニック・フェージック成分を抽出
        signals = nk.eda_phasic(eda_interpolated, sampling_rate=Fs, method="cvxeda")
        tonic = signals['EDA_Tonic'].to_numpy()
        phasic = signals['EDA_Phasic'].to_numpy()
        return tonic, phasic
    except Exception as e:
        if verbose:
            import traceback
            print(f"Error in cvxEDA: {e}")
            traceback.print_exc()
        else:
            print(f"Error in cvxEDA: {e}")
        return None, None


def preprocess_eda_data(
    data_path: str = "/home/mori/projects/affective-forecast/datas/biometric_data",
    search_conditions: dict = None,
    Fs: float = 4.0,
    median_window: int = 17,
    median_threshold: float = 0.30,
    overwrite: bool = False
):
    """
    全EDAデータを前処理して保存

    Args:
        data_path: データディレクトリのパス
        search_conditions: データ検索条件
        Fs: サンプリング周波数
        median_window: マスク用メディアンフィルタ窓幅
        median_threshold: マスク用閾値
        overwrite: 既存ファイルを上書きするか
    """
    if search_conditions is None:
        search_conditions = {"ex-term": "term1"}

    # データリストを取得
    print("Loading data list...")
    uuids = search_by_conditions(search_conditions)
    datas = get_affective_datas_from_uuids(uuids)
    datas.sort(key=lambda x: int(x["filename"].split("_")[1]))

    print(f"Total files to process: {len(datas)}")

    # ローパスフィルタ係数を事前計算
    b, a = signal.butter(4, 0.35, btype='low', fs=Fs)

    # 統計情報
    stats = {
        'total': len(datas),
        'success': 0,
        'skipped': 0,
        'failed': 0
    }

    # 各データファイルを処理
    for data_info in tqdm(datas, desc="Processing EDA data"):
        name = data_info["filename"]

        # 既存ファイルチェック
        output_files = [
            f"{data_path}/{name}_eda_lowpass.h5",
            f"{data_path}/{name}_eda_masked.h5",
        ]
        if HAS_NEUROKIT:
            output_files.extend([
                f"{data_path}/{name}_scl.h5",
                f"{data_path}/{name}_scr.h5",
            ])

        if not overwrite and all(os.path.exists(f) for f in output_files):
            stats['skipped'] += 1
            continue

        # 元のEDAデータを読み込み
        try:
            eda_h5 = load_h5_data(f"{data_path}/{name}_eda.h5")
            if eda_h5 is None or eda_h5.shape[1] < 3599:
                print(f"\nSkipping {name}: Invalid EDA data")
                stats['failed'] += 1
                continue

            # float64で読み込む（cvxEDAに必要）
            eda_original = np.asarray(eda_h5[0], dtype=np.float64)

        except Exception as e:
            print(f"\nError loading {name}: {e}")
            stats['failed'] += 1
            continue

        try:
            # 1. ローパスフィルタ適用
            eda_lowpass = signal.filtfilt(b, a, eda_original).astype(np.float32)

            # 保存
            with h5py.File(f"{data_path}/{name}_eda_lowpass.h5", 'w') as f:
                f.create_dataset('eda_lowpass', data=eda_lowpass.reshape(1, -1), compression='gzip')

            # 2. マスク適用EDA
            eda_masked = apply_eda_mask(eda_original, Fs, median_window, median_threshold).astype(np.float32)

            # 保存
            with h5py.File(f"{data_path}/{name}_eda_masked.h5", 'w') as f:
                f.create_dataset('eda_masked', data=eda_masked.reshape(1, -1), compression='gzip')

            # 3. SCL/SCR（neurokit2がある場合のみ）
            if HAS_NEUROKIT:
                scl, scr = extract_scl_scr(eda_lowpass, Fs)

                if scl is not None and scr is not None:
                    # SCL保存
                    with h5py.File(f"{data_path}/{name}_scl.h5", 'w') as f:
                        f.create_dataset('scl', data=scl.astype(np.float32).reshape(1, -1), compression='gzip')

                    # SCR保存
                    with h5py.File(f"{data_path}/{name}_scr.h5", 'w') as f:
                        f.create_dataset('scr', data=scr.astype(np.float32).reshape(1, -1), compression='gzip')

            stats['success'] += 1

        except Exception as e:
            print(f"\nError processing {name}: {e}")
            stats['failed'] += 1

    # 結果サマリー
    print("\n" + "="*60)
    print("Preprocessing completed!")
    print("="*60)
    print(f"Total files: {stats['total']}")
    print(f"Successfully processed: {stats['success']}")
    print(f"Skipped (already exists): {stats['skipped']}")
    print(f"Failed: {stats['failed']}")
    print("="*60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess EDA data")
    parser.add_argument("--data_path", type=str,
                        default="/home/mori/projects/affective-forecast/datas/biometric_data",
                        help="Data directory path")
    parser.add_argument("--term", type=str, default="term1", help="Data term (term1/term2/term3)")
    parser.add_argument("--median_window", type=int, default=17, help="Median filter window size")
    parser.add_argument("--median_threshold", type=float, default=0.30, help="Median filter threshold")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")

    args = parser.parse_args()

    preprocess_eda_data(
        data_path=args.data_path,
        search_conditions={"ex-term": args.term},
        median_window=args.median_window,
        median_threshold=args.median_threshold,
        overwrite=args.overwrite
    )
