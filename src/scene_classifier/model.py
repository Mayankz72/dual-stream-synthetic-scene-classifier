import torch.nn as nn


class ResBlock(nn.Module):
    """Two 3x3 conv-BN-ReLU layers with an identity skip connection."""

    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.net(x))


class SEBlock(nn.Module):
    """Squeeze-and-excitation channel attention (reduction ratio r)."""

    def __init__(self, channels: int, r: int = 4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // r),
            nn.ReLU(inplace=True),
            nn.Linear(channels // r, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(x).unsqueeze(-1).unsqueeze(-1)


class ShapeNet(nn.Module):
    """Four-stage residual + SE-attention CNN encoder with a binary logit head.

    Shared backbone for both the RGB stream and the edge-feature stream;
    trained from scratch (no pre-trained weights, per competition rules).
    """

    def __init__(self, in_channels: int = 3):
        super().__init__()

        def down_block(ci, co):
            return nn.Sequential(
                ResBlock(ci),
                SEBlock(ci),
                nn.Conv2d(ci, co, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(co),
                nn.ReLU(inplace=True),
            )

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.d1 = down_block(32, 64)
        self.d2 = down_block(64, 128)
        self.d3 = down_block(128, 256)
        self.d4 = down_block(256, 512)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        x = self.d4(x)
        x = self.pool(x)
        return self.head(x)
