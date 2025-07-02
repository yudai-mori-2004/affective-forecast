#!/usr/bin/env python3
import os
import h5py

def list_and_preview_h5_files(directory="sample", preview_shape=(2, 5)):
    """
    指定ディレクトリ内のすべての .h5 ファイルを走査し、
    各データセットの形状と冒頭の一部データを表示します。
    """
    for fname in sorted(os.listdir(directory)):
        if not fname.endswith(".h5"):
            continue
        filepath = os.path.join(directory, fname)
        print(f"\n=== File: {fname} ===")
        with h5py.File(filepath, "r") as f:
            for dset_name, dset in f.items():
                data = dset[()]
                print(f"- Dataset '{dset_name}': shape={data.shape}, dtype={data.dtype}")
                if data.ndim >= 5:
                    snippet = data[:preview_shape[0], :preview_shape[1]]
                else:
                    snippet = data[:preview_shape[1]]
                print("  Sample data:\n", snippet)



def inspect_h5(directory="sample"):
    for fname in sorted(os.listdir(directory)):
        if not fname.endswith(".h5"): continue
        path = os.path.join(directory, fname)
        with h5py.File(path, "r") as f:
            print(f"\n=== {fname} ===")
            # キー一覧
            print("datasets:", list(f.keys()))
            dset = f["E4"]
            print(f"shape={dset.shape}, dtype={dset.dtype}")
            # 属性一覧
            for k,v in dset.attrs.items():
                print(f"  attr {k}: {v}")


if __name__ == "__main__":
    list_and_preview_h5_files()
