"""Kiến trúc mô hình và hàm nạp trọng số."""
import timm
import torch
import torch.nn as nn
from safetensors.torch import load_file

from .config import MODEL_SPECS, WEIGHTS_DIR


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.relu(out + identity)


class ResNet50Custom(nn.Module):
    """ResNet50 tự cài đặt, train từ đầu (xem research/notebooks/ResNet50.ipynb)."""

    def __init__(self, num_classes):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.layer1 = self._make_layer(64, 3)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)

    def _make_layer(self, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * Bottleneck.expansion, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels * Bottleneck.expansion),
            )
        layers = [Bottleneck(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels * Bottleneck.expansion
        layers += [Bottleneck(self.in_channels, out_channels) for _ in range(1, blocks)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.fc(torch.flatten(self.avgpool(x), 1))


def build_model(key: str, num_classes: int) -> nn.Module:
    spec = MODEL_SPECS[key]
    if spec["arch"] == "resnet50_custom":
        model = ResNet50Custom(num_classes)
    else:
        model = timm.create_model(spec["arch"], pretrained=False, num_classes=num_classes)

    path = WEIGHTS_DIR / spec["weights"]
    if not path.exists():
        raise FileNotFoundError(f"Thiếu trọng số {path} — chạy: python scripts/download_weights.py")
    # File lưu fp16; load_state_dict copy vào tham số fp32 của model (CPU chạy fp32 nhanh hơn).
    model.load_state_dict(load_file(str(path)), strict=True)
    return model.eval()
