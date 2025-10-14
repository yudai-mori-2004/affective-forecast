"""
PyTorch Dataset for multimodal biometric data
"""
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, List, Tuple
from scipy import signal
from scipy.ndimage import median_filter as scipy_median_filter
from src.util.utils3 import search_by_conditions, get_affective_datas_from_uuids, load_h5_data

try:
    import neurokit2 as nk
    HAS_NEUROKIT = True
except ImportError:
    HAS_NEUROKIT = False


class BiometricDataset(Dataset):
    """
    マルチモーダル生体データのDataset

    データ構成:
    - EDA: 4Hz × 15分 = 3600サンプル
    - 加速度: 32Hz × 15分 × 3軸 = 28800サンプル × 3
    - 皮膚温度: 4Hz × 15分 = 3600サンプル
    - RRI: 4Hz × 15分 = 3600サンプル

    ターゲット:
    - Arousal: -4 ~ 4 (9段階)
    - Valence: -4 ~ 4 (9段階)
    """

    def __init__(
        self,
        search_conditions: Dict = None,
        data_path: str = "/home/mori/projects/affective-forecast/datas/biometric_data",
        target_type: str = "both",  # "arousal", "valence", or "both"
        normalize: bool = True,
        # EDA関連（独立したフラグ）
        use_eda: bool = False,           # 生のEDA
        use_eda_masked: bool = False,    # マスク適用EDA（NaN含む）
        use_scr: bool = False,           # SCR（Phasic）
        use_scl: bool = False,           # SCL（Tonic）
        # 加速度関連（独立したフラグ）
        use_acc_x: bool = False,         # 加速度X軸
        use_acc_y: bool = False,         # 加速度Y軸
        use_acc_z: bool = False,         # 加速度Z軸
        use_acc_mag: bool = True,        # 加速度magnitude（acc[3]）
        # その他
        use_temp: bool = True,           # 温度
        use_rri: bool = True,            # RRI
        # フィルタパラメータ（eda_masked用）
        eda_median_window: int = 17,
        eda_median_threshold: float = 0.30
    ):
        """
        Args:
            search_conditions: データ検索条件
            data_path: データディレクトリのパス
            target_type: 予測対象 ("arousal", "valence", "both")
            normalize: データを正規化するかどうか
            use_eda: 生のEDAデータを使用
            use_eda_masked: マスク適用EDAデータを使用
            use_scr: SCR（Phasic成分）を使用
            use_scl: SCL（Tonic成分）を使用
            use_acc_x: 加速度X軸を使用
            use_acc_y: 加速度Y軸を使用
            use_acc_z: 加速度Z軸を使用
            use_acc_mag: 加速度magnitude（acc[3]）を使用
            use_temp: 温度データを使用
            use_rri: RRIデータを使用
            eda_median_window: EDAマスク用のメディアンフィルタ窓幅
            eda_median_threshold: EDAマスク用の閾値
        """
        self.data_path = data_path
        self.target_type = target_type
        self.normalize = normalize

        # EDA関連
        self.use_eda = use_eda
        self.use_eda_masked = use_eda_masked
        self.use_scr = use_scr
        self.use_scl = use_scl

        # 加速度関連
        self.use_acc_x = use_acc_x
        self.use_acc_y = use_acc_y
        self.use_acc_z = use_acc_z
        self.use_acc_mag = use_acc_mag

        # その他
        self.use_temp = use_temp
        self.use_rri = use_rri

        # フィルタパラメータ
        self.eda_median_window = eda_median_window
        self.eda_median_threshold = eda_median_threshold
        self.Fs = 4.0  # サンプリング周波数

        # デフォルトの検索条件
        if search_conditions is None:
            search_conditions = {"ex-term": "term1"}

        # データリストを取得
        uuids = search_by_conditions(search_conditions)
        self.datas = get_affective_datas_from_uuids(uuids)
        self.datas.sort(key=lambda x: int(x["filename"].split("_")[1]))

        # 有効なデータのみをフィルタリング
        self.valid_indices = []
        for idx in range(len(self.datas)):
            if self._check_data_validity(idx):
                self.valid_indices.append(idx)

        print(f"Total samples: {len(self.datas)}, Valid samples: {len(self.valid_indices)}")

        # 入力チャンネル数を計算（単純な加算）
        self.n_channels = 0
        if use_eda:
            self.n_channels += 1
        if use_eda_masked:
            self.n_channels += 1
        if use_scr:
            self.n_channels += 1
        if use_scl:
            self.n_channels += 1
        if use_acc_x:
            self.n_channels += 1
        if use_acc_y:
            self.n_channels += 1
        if use_acc_z:
            self.n_channels += 1
        if use_acc_mag:
            self.n_channels += 1
        if use_temp:
            self.n_channels += 1
        if use_rri:
            self.n_channels += 1

        # サンプル長（4Hzに統一）
        self.sample_length = 3600  # 4Hz × 15分

    def _check_data_validity(self, idx: int) -> bool:
        """データが有効かチェック"""
        data_info = self.datas[idx]
        name = data_info["filename"]

        # EDA関連のチェック（前処理済みファイルまたは元ファイル）
        if self.use_eda:
            # eda_lowpass.h5 または eda.h5 の存在確認
            eda_h5 = load_h5_data(f"{self.data_path}/{name}_eda_lowpass.h5")
            if eda_h5 is None:
                eda_h5 = load_h5_data(f"{self.data_path}/{name}_eda.h5")
            if eda_h5 is None or eda_h5.shape[1] < 3599:
                return False

        if self.use_eda_masked:
            # eda_masked.h5 または eda.h5 の存在確認
            eda_masked_h5 = load_h5_data(f"{self.data_path}/{name}_eda_masked.h5")
            if eda_masked_h5 is None:
                eda_h5 = load_h5_data(f"{self.data_path}/{name}_eda.h5")
                if eda_h5 is None or eda_h5.shape[1] < 3599:
                    return False

        if self.use_scl:
            # scl.h5の存在確認（必須）
            scl_h5 = load_h5_data(f"{self.data_path}/{name}_scl.h5")
            if scl_h5 is None or scl_h5.shape[1] < 3599:
                return False

        if self.use_scr:
            # scr.h5の存在確認（必須）
            scr_h5 = load_h5_data(f"{self.data_path}/{name}_scr.h5")
            if scr_h5 is None or scr_h5.shape[1] < 3599:
                return False

        # 加速度関連のチェック
        if self.use_acc_x or self.use_acc_y or self.use_acc_z or self.use_acc_mag:
            act_h5 = load_h5_data(f"{self.data_path}/{name}_act.h5")
            if act_h5 is None or act_h5.shape[1] < 28799:
                return False
            # magnitudeを使う場合はacc[3]の存在をチェック
            if self.use_acc_mag and act_h5.shape[0] < 4:
                return False

        # 温度のチェック
        if self.use_temp:
            temp_h5 = load_h5_data(f"{self.data_path}/{name}_temp.h5")
            if temp_h5 is None or temp_h5.shape[1] < 3599:
                return False

        # RRIのチェック
        if self.use_rri:
            rri_h5 = load_h5_data(f"{self.data_path}/{name}_rri.h5")
            if rri_h5 is None or rri_h5.shape[1] < 3599:
                return False

        return True

    def _apply_eda_mask(self, eda_original: np.ndarray) -> np.ndarray:
        """EDAデータにマスクを適用（filter_improve.pyと同様の処理）"""
        # threshold filter
        extend_sec = 5
        extend_samples = int(self.Fs * extend_sec)
        thr_mask = (eda_original < 0.05) | (eda_original > 60)
        thr_mask = np.convolve(thr_mask.astype(int), np.ones(2*extend_samples+1, dtype=int), mode='same') > 0

        # median filter
        med = scipy_median_filter(eda_original, size=self.eda_median_window, mode='nearest')
        med_mask = np.abs((eda_original / (med + 1e-8)) - 1.0) > self.eda_median_threshold

        # 全マスクを統合
        combined_mask = thr_mask | med_mask

        # マスク部分をNaNに設定
        eda_masked = eda_original.copy()
        eda_masked[combined_mask] = np.nan

        return eda_masked

    def _extract_scl_scr(self, eda: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """EDAからSCL（Tonic）とSCR（Phasic）を分離"""
        if not HAS_NEUROKIT:
            # neurokit2がない場合は元のEDAを返す
            return eda, np.zeros_like(eda)

        try:
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

            # cvxEDAでトニック・フェージック成分を抽出
            signals = nk.eda_phasic(eda_interpolated, sampling_rate=self.Fs, method="cvxeda")
            tonic = signals['EDA_Tonic'].to_numpy()
            phasic = signals['EDA_Phasic'].to_numpy()
            return tonic, phasic
        except:
            # 失敗した場合は元のEDAを返す
            return eda, np.zeros_like(eda)

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            data: (n_channels, sample_length) のTensor
            target: (1,) or (2,) のTensor (arousal, valence, or both)
        """
        # 有効なインデックスを取得
        real_idx = self.valid_indices[idx]
        data_info = self.datas[real_idx]
        name = data_info["filename"]

        # チャンネルリスト
        channels = []

        # ===== EDAデータ =====
        # 生のEDA（ローパスフィルタ済み）
        if self.use_eda:
            eda_h5 = load_h5_data(f"{self.data_path}/{name}_eda_lowpass.h5")
            if eda_h5 is not None:
                eda = np.asarray(eda_h5[0][:3600], dtype=np.float32)
                if len(eda) < 3600:
                    eda = np.pad(eda, (0, 3600 - len(eda)), mode='constant')
                channels.append(eda)
            else:
                # フォールバック：生データから計算
                eda_h5 = load_h5_data(f"{self.data_path}/{name}_eda.h5")
                eda_original = np.asarray(eda_h5[0][:3600], dtype=np.float32)
                if len(eda_original) < 3600:
                    eda_original = np.pad(eda_original, (0, 3600 - len(eda_original)), mode='constant')
                b, a = signal.butter(4, 0.35, btype='low', fs=self.Fs)
                eda = signal.filtfilt(b, a, eda_original)
                channels.append(eda)

        # マスク適用EDA
        if self.use_eda_masked:
            eda_masked_h5 = load_h5_data(f"{self.data_path}/{name}_eda_masked.h5")
            if eda_masked_h5 is not None:
                eda_masked = np.asarray(eda_masked_h5[0][:3600], dtype=np.float32)
                if len(eda_masked) < 3600:
                    eda_masked = np.pad(eda_masked, (0, 3600 - len(eda_masked)), mode='constant')
                channels.append(eda_masked)
            else:
                # フォールバック：リアルタイム計算
                eda_h5 = load_h5_data(f"{self.data_path}/{name}_eda.h5")
                eda_original = np.asarray(eda_h5[0][:3600], dtype=np.float32)
                if len(eda_original) < 3600:
                    eda_original = np.pad(eda_original, (0, 3600 - len(eda_original)), mode='constant')
                eda_masked = self._apply_eda_mask(eda_original)
                channels.append(eda_masked)

        # SCL（Tonic成分）
        if self.use_scl:
            scl_h5 = load_h5_data(f"{self.data_path}/{name}_scl.h5")
            if scl_h5 is not None:
                scl = np.asarray(scl_h5[0][:3600], dtype=np.float32)
                if len(scl) < 3600:
                    scl = np.pad(scl, (0, 3600 - len(scl)), mode='constant')
                channels.append(scl)
            else:
                print(f"Warning: {name}_scl.h5 not found. Run preprocess_eda.py first.")
                channels.append(np.zeros(3600, dtype=np.float32))

        # SCR（Phasic成分）
        if self.use_scr:
            scr_h5 = load_h5_data(f"{self.data_path}/{name}_scr.h5")
            if scr_h5 is not None:
                scr = np.asarray(scr_h5[0][:3600], dtype=np.float32)
                if len(scr) < 3600:
                    scr = np.pad(scr, (0, 3600 - len(scr)), mode='constant')
                channels.append(scr)
            else:
                print(f"Warning: {name}_scr.h5 not found. Run preprocess_eda.py first.")
                channels.append(np.zeros(3600, dtype=np.float32))

        # ===== 加速度データ =====
        if self.use_acc_x or self.use_acc_y or self.use_acc_z or self.use_acc_mag:
            act_h5 = load_h5_data(f"{self.data_path}/{name}_act.h5")

            # X軸
            if self.use_acc_x:
                acc_x_full = np.asarray(act_h5[0][:28800], dtype=np.float32)
                acc_x = acc_x_full[::8][:3600]  # 32Hz -> 4Hz
                if len(acc_x) < 3600:
                    acc_x = np.pad(acc_x, (0, 3600 - len(acc_x)), mode='constant')
                channels.append(acc_x)

            # Y軸
            if self.use_acc_y:
                acc_y_full = np.asarray(act_h5[1][:28800], dtype=np.float32)
                acc_y = acc_y_full[::8][:3600]
                if len(acc_y) < 3600:
                    acc_y = np.pad(acc_y, (0, 3600 - len(acc_y)), mode='constant')
                channels.append(acc_y)

            # Z軸
            if self.use_acc_z:
                acc_z_full = np.asarray(act_h5[2][:28800], dtype=np.float32)
                acc_z = acc_z_full[::8][:3600]
                if len(acc_z) < 3600:
                    acc_z = np.pad(acc_z, (0, 3600 - len(acc_z)), mode='constant')
                channels.append(acc_z)

            # Magnitude
            if self.use_acc_mag:
                acc_mag_full = np.asarray(act_h5[3][:28800], dtype=np.float32)
                acc_mag = acc_mag_full[::8][:3600]
                if len(acc_mag) < 3600:
                    acc_mag = np.pad(acc_mag, (0, 3600 - len(acc_mag)), mode='constant')
                channels.append(acc_mag)

        # ===== 温度データ =====
        if self.use_temp:
            temp_h5 = load_h5_data(f"{self.data_path}/{name}_temp.h5")
            temp = np.asarray(temp_h5[0][:3600], dtype=np.float32)
            if len(temp) < 3600:
                temp = np.pad(temp, (0, 3600 - len(temp)), mode='constant')
            channels.append(temp)

        # ===== RRIデータ =====
        if self.use_rri:
            rri_h5 = load_h5_data(f"{self.data_path}/{name}_rri.h5")
            rri = np.asarray(rri_h5[0][:3600], dtype=np.float32)
            if len(rri) < 3600:
                rri = np.pad(rri, (0, 3600 - len(rri)), mode='constant')
            channels.append(rri)

        # チャンネルを結合
        data = np.stack(channels, axis=0)  # (n_channels, sample_length)

        # 正規化
        if self.normalize:
            for i in range(data.shape[0]):
                mean = np.nanmean(data[i])
                std = np.nanstd(data[i])
                if std > 0:
                    data[i] = (data[i] - mean) / std
                # NaNを0で埋める
                data[i] = np.nan_to_num(data[i], nan=0.0)

        # ターゲット
        arousal = data_info["arousal"]
        valence = data_info["valence"]

        # ターゲットを[-4, 4]から[0, 8]にシフト
        arousal_class = arousal + 4
        valence_class = valence + 4

        if self.target_type == "arousal":
            target = np.array([arousal_class], dtype=np.int64)
        elif self.target_type == "valence":
            target = np.array([valence_class], dtype=np.int64)
        else:  # both
            target = np.array([arousal_class, valence_class], dtype=np.int64)

        # Tensorに変換
        data_tensor = torch.from_numpy(data).float()
        target_tensor = torch.from_numpy(target).long()

        return data_tensor, target_tensor

    def get_data_info(self, idx: int) -> Dict:
        """データ情報を取得"""
        real_idx = self.valid_indices[idx]
        return self.datas[real_idx]


def create_dataloaders(
    batch_size: int = 32,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    search_conditions: Dict = None,
    **dataset_kwargs
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    DataLoaderを作成

    Args:
        batch_size: バッチサイズ
        train_ratio: 訓練データの割合
        val_ratio: 検証データの割合
        test_ratio: テストデータの割合
        search_conditions: データ検索条件
        **dataset_kwargs: Datasetに渡す追加引数

    Returns:
        train_loader, val_loader, test_loader
    """
    # Dataset作成
    dataset = BiometricDataset(search_conditions=search_conditions, **dataset_kwargs)

    # データ分割
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    # DataLoader作成
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # テスト
    dataset = BiometricDataset(
        search_conditions={"ex-term": "term1"},
        target_type="both",
        use_eda=True,
        use_acc_mag=True,
        use_temp=True,
        use_rri=True
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"Input channels: {dataset.n_channels}")
    print(f"Sample length: {dataset.sample_length}")

    # サンプル取得
    data, target = dataset[0]
    print(f"Data shape: {data.shape}")
    print(f"Target shape: {target.shape}")
    print(f"Target: {target}")
