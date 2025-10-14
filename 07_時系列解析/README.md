# 1D CNN for Affective Prediction

マルチモーダル生体データ（EDA、加速度、皮膚温度、RRI）を用いた感情予測（Arousal/Valence）のための1D CNNモデル

## データ構成

- **EDA（皮膚電気活動）**: 4Hz × 15分 = 3600サンプル
- **加速度**: 32Hz × 15分 × 3軸 = 28800サンプル × 3軸
- **皮膚温度**: 4Hz × 15分 = 3600サンプル
- **RRI（心拍間隔）**: 4Hz × 15分 = 3600サンプル

**ターゲット:**
- Arousal: -4 ~ 4 (9段階)
- Valence: -4 ~ 4 (9段階)

## ファイル構成

```
07_時系列解析/
├── dataset.py          # PyTorch Dataset
├── train.py            # 訓練スクリプト
├── models/
│   ├── __init__.py
│   └── cnn1d.py        # 1D CNNモデル定義
├── checkpoints/        # モデルチェックポイント保存先
├── logs/               # TensorBoardログ
└── README.md
```

## モデル一覧

1. **SimpleCNN1D**: シンプルな4層CNN
2. **ResCNN1D**: ResNetスタイルのCNN（Residual接続）
3. **DeepCNN1D**: より深い6層CNN
4. **MultiTaskCNN1D**: ArousalとValenceを同時予測するマルチタスク学習モデル

## 環境構築

必要なパッケージ：
```bash
# PyTorch
torch>=2.0.0
torchvision>=0.15.0

# その他
numpy
h5py
tqdm
tensorboard
```

## データの種類

実装では以下のデータを**独立して選択**できます：

### **EDA（皮膚電気活動）関連**
- **生のEDA (`use_eda`)**: ローパスフィルタ適用後の生データ
- **マスク適用EDA (`use_eda_masked`)**: アーチファクト除去マスク適用（NaN含む）
- **SCL (`use_scl`)**: トニック成分（cvxEDA、低周波）→ 覚醒度・緊張度
- **SCR (`use_scr`)**: フェージック成分（cvxEDA、高周波）→ 急激な感情反応

### **加速度関連**
- **X軸 (`use_acc_x`)**: 加速度X軸（4Hzダウンサンプリング）
- **Y軸 (`use_acc_y`)**: 加速度Y軸（4Hzダウンサンプリング）
- **Z軸 (`use_acc_z`)**: 加速度Z軸（4Hzダウンサンプリング）
- **Magnitude (`use_acc_mag`)**: 身体運動量 (デフォルト)

### **その他**
- **温度 (`use_temp`)**: 皮膚温度
- **RRI (`use_rri`)**: 心拍間隔

## 使い方

### 前処理（推奨）

**SCL/SCRやマスク適用EDAを使う場合は、必ず前処理を実行してください。**

前処理により訓練速度が**10-100倍高速化**します。

```bash
# term1のデータを前処理
python preprocess_eda.py --term term1

# term2も前処理する場合
python preprocess_eda.py --term term2

# 上書きモード（既存ファイルも再計算）
python preprocess_eda.py --term term1 --overwrite
```

前処理により以下のファイルが生成されます：
- `{name}_eda_lowpass.h5`: ローパスフィルタ適用済みEDA
- `{name}_eda_masked.h5`: マスク適用EDA
- `{name}_scl.h5`: SCL（Tonic成分）
- `{name}_scr.h5`: SCR（Phasic成分）

### 基本的な訓練

```bash
python train.py --model simple --target_type arousal --epochs 100
```

### パラメータ一覧

**データ関連:**
- `--term`: データのterm（term1/term2/term3）デフォルト: term1
- `--target_type`: 予測対象（arousal/valence/both）デフォルト: arousal

**EDA関連:**
- `--use_eda`: 生のEDAデータを使用（0/1）デフォルト: 0
- `--use_eda_masked`: マスク適用EDAデータを使用（0/1）デフォルト: 0
- `--use_scr`: SCR（Phasic成分）を使用（0/1）デフォルト: 0
- `--use_scl`: SCL（Tonic成分）を使用（0/1）デフォルト: 0

**加速度関連:**
- `--use_acc_x`: 加速度X軸を使用（0/1）デフォルト: 0
- `--use_acc_y`: 加速度Y軸を使用（0/1）デフォルト: 0
- `--use_acc_z`: 加速度Z軸を使用（0/1）デフォルト: 0
- `--use_acc_mag`: 加速度magnitude（身体運動量）を使用（0/1）デフォルト: 1

**その他:**
- `--use_temp`: 温度データを使用（0/1）デフォルト: 1
- `--use_rri`: RRIデータを使用（0/1）デフォルト: 1

**モデル関連:**
- `--model`: モデルタイプ（simple/resnet/deep/multitask）デフォルト: simple
- `--dropout`: Dropout率 デフォルト: 0.5

**訓練関連:**
- `--batch_size`: バッチサイズ デフォルト: 32
- `--epochs`: エポック数 デフォルト: 100
- `--lr`: 学習率 デフォルト: 0.001
- `--weight_decay`: Weight decay デフォルト: 1e-5
- `--patience`: 早期停止のpatience デフォルト: 15

**その他:**
- `--save_dir`: チェックポイント保存先 デフォルト: ./checkpoints
- `--log_dir`: TensorBoardログ保存先 デフォルト: ./logs

### 実行例

**1. デフォルト設定（acc_mag + temp + rri）**
```bash
python train.py --model resnet --target_type arousal
```

**2. 生のEDAデータを使用**
```bash
python train.py \
    --model resnet \
    --target_type arousal \
    --use_eda 1 \
    --use_acc_mag 1 \
    --use_temp 1 \
    --use_rri 1
```

**3. SCL/SCRを使用（感情認識に効果的）**
```bash
python train.py \
    --model resnet \
    --target_type arousal \
    --use_scl 1 \
    --use_scr 1 \
    --use_acc_mag 1 \
    --use_temp 1 \
    --use_rri 1
```

**4. マスク適用EDAを使用**
```bash
python train.py \
    --model resnet \
    --target_type arousal \
    --use_eda_masked 1 \
    --use_acc_mag 1 \
    --use_temp 1 \
    --use_rri 1
```

**5. 加速度3軸を使用**
```bash
python train.py \
    --model resnet \
    --target_type arousal \
    --use_acc_x 1 \
    --use_acc_y 1 \
    --use_acc_z 1 \
    --use_temp 1 \
    --use_rri 1
```

**6. マルチタスク学習（ArousalとValence同時予測）**
```bash
python train.py \
    --model multitask \
    --use_eda 1 \
    --use_acc_mag 1 \
    --use_temp 1 \
    --use_rri 1
```

## TensorBoardでの可視化

訓練中の損失やAccuracyをTensorBoardで確認できます：

```bash
tensorboard --logdir ./logs
```

ブラウザで `http://localhost:6006` にアクセス

## 出力

訓練完了後、以下のファイルが保存されます：

```
checkpoints/{model}_{timestamp}/
├── best_model.pth      # ベストモデル
├── final_model.pth     # 最終モデル
├── config.json         # 訓練時の設定
└── results.json        # テスト結果
```

**results.json 例:**
```json
{
  "best_val_loss": 1.234,
  "test_loss": 1.345,
  "test_acc": 45.67
}
```

## データセットの動作確認

```bash
python dataset.py
```

## モデルの動作確認

```bash
cd models
python cnn1d.py
```

## Tips

1. **バッチサイズ調整**: GPUメモリに応じて調整（32, 64, 128など）
2. **学習率調整**: 収束しない場合は0.0001や0.01を試す
3. **Dropout調整**: 過学習する場合は0.6-0.7に増やす
4. **データ選択**: 特定のモダリティのみで実験可能（例: EDAのみ）
5. **早期停止**: patienceを調整して訓練時間を短縮

## トラブルシューティング

**Q: CUDAメモリエラーが出る**
A: バッチサイズを小さくする（16や8など）

**Q: 精度が低い**
A: 以下を試す：
- エポック数を増やす
- 学習率を調整
- データ正規化の確認
- モデルを変更（ResNet, Deepなど）

**Q: 訓練が進まない**
A: 学習率を下げる（0.0001など）

## 今後の改善案

- [ ] Data Augmentation（時系列用）
- [ ] Attention機構の追加
- [ ] Transformer-based モデル
- [ ] クラス不均衡への対応（Focal Loss等）
- [ ] K-fold Cross Validation
- [ ] 回帰タスクとしての実装
