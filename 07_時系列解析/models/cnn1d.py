"""
1D CNN models for time-series biometric data
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN1D(nn.Module):
    """シンプルな1D CNNモデル"""

    def __init__(
        self,
        in_channels: int = 6,  # EDA(1) + ACC(3) + TEMP(1) + RRI(1)
        num_classes: int = 9,  # -4 ~ 4 の9段階
        dropout: float = 0.5
    ):
        super(SimpleCNN1D, self).__init__()

        # 畳み込み層
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm1d(128)

        self.conv3 = nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2)
        self.bn3 = nn.BatchNorm1d(256)

        self.conv4 = nn.Conv1d(256, 512, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm1d(512)

        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Fully connected layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # x: (batch, in_channels, seq_len)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))

        # Global pooling
        x = self.global_pool(x)  # (batch, 512, 1)
        x = x.squeeze(-1)  # (batch, 512)

        # FC layers
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class ResBlock1D(nn.Module):
    """1D Residual Block"""

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResBlock1D, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out)

        return out


class ResCNN1D(nn.Module):
    """ResNetスタイルの1D CNNモデル"""

    def __init__(
        self,
        in_channels: int = 6,
        num_classes: int = 9,
        dropout: float = 0.5
    ):
        super(ResCNN1D, self).__init__()

        # 初期畳み込み
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # Residual blocks
        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)

        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # FC layers
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

        layers = []
        layers.append(ResBlock1D(in_channels, out_channels, stride, downsample))
        for _ in range(1, blocks):
            layers.append(ResBlock1D(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        # x: (batch, in_channels, seq_len)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Global pooling
        x = self.global_pool(x)  # (batch, 512, 1)
        x = x.squeeze(-1)  # (batch, 512)

        # FC layer
        x = self.dropout(x)
        x = self.fc(x)

        return x


class MultiTaskCNN1D(nn.Module):
    """マルチタスク学習版（ArousalとValenceを同時予測）"""

    def __init__(
        self,
        in_channels: int = 6,
        num_classes: int = 9,
        dropout: float = 0.5,
        use_resnet: bool = False
    ):
        super(MultiTaskCNN1D, self).__init__()

        # 共有特徴抽出層
        if use_resnet:
            self.backbone = ResCNN1D(in_channels, num_classes, dropout)
            # 最後のFC層を削除
            self.backbone.fc = nn.Identity()
            feature_dim = 512
        else:
            self.backbone = SimpleCNN1D(in_channels, num_classes, dropout)
            # 最後のFC層を削除
            self.backbone.fc2 = nn.Identity()
            feature_dim = 256

        # タスク固有のヘッド
        self.arousal_head = nn.Linear(feature_dim, num_classes)
        self.valence_head = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        # 共有特徴抽出
        features = self.backbone(x)

        # 各タスクの予測
        arousal_out = self.arousal_head(features)
        valence_out = self.valence_head(features)

        return arousal_out, valence_out


class DeepCNN1D(nn.Module):
    """より深い1D CNNモデル"""

    def __init__(
        self,
        in_channels: int = 6,
        num_classes: int = 9,
        dropout: float = 0.5
    ):
        super(DeepCNN1D, self).__init__()

        # 畳み込み層
        self.conv_blocks = nn.ModuleList([
            # Block 1
            nn.Sequential(
                nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
            ),
            # Block 2
            nn.Sequential(
                nn.Conv1d(64, 128, kernel_size=5, padding=2),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Conv1d(128, 128, kernel_size=5, padding=2),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2, stride=2)
            ),
            # Block 3
            nn.Sequential(
                nn.Conv1d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Conv1d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2, stride=2)
            ),
            # Block 4
            nn.Sequential(
                nn.Conv1d(256, 512, kernel_size=3, padding=1),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Conv1d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool1d(1)
            )
        ])

        # FC layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        # 畳み込み
        for block in self.conv_blocks:
            x = block(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # FC layers
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x


def get_model(model_name: str, **kwargs):
    """
    モデルを取得

    Args:
        model_name: "simple", "resnet", "deep", "multitask"
        **kwargs: モデルに渡す引数

    Returns:
        model: PyTorchモデル
    """
    models = {
        "simple": SimpleCNN1D,
        "resnet": ResCNN1D,
        "deep": DeepCNN1D,
        "multitask": MultiTaskCNN1D
    }

    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")

    return models[model_name](**kwargs)


if __name__ == "__main__":
    # モデルのテスト
    batch_size = 4
    in_channels = 6
    seq_len = 3600

    # ダミーデータ
    x = torch.randn(batch_size, in_channels, seq_len)

    print("=" * 60)
    print("SimpleCNN1D")
    print("=" * 60)
    model = SimpleCNN1D(in_channels=in_channels, num_classes=9)
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("\n" + "=" * 60)
    print("ResCNN1D")
    print("=" * 60)
    model = ResCNN1D(in_channels=in_channels, num_classes=9)
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("\n" + "=" * 60)
    print("MultiTaskCNN1D")
    print("=" * 60)
    model = MultiTaskCNN1D(in_channels=in_channels, num_classes=9)
    arousal_out, valence_out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Arousal output shape: {arousal_out.shape}")
    print(f"Valence output shape: {valence_out.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
