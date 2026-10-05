"""Cấu hình tập trung: đường dẫn, đặc tả mô hình, biến môi trường."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_DIR = Path(os.getenv("WEIGHTS_DIR", ROOT / "weights"))
CLASS_NAMES_PATH = ROOT / "data" / "class_names.json"
EXAMPLES_DIR = ROOT / "examples"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# preprocess PHẢI khớp val_transforms lúc train (xem research/evaluate_models.py):
#   "resize224"         = Resize((224, 224))            (ConvNeXt, Swin)
#   "resize256_crop224" = Resize(256) + CenterCrop(224) (ResNet50)
# test_top1: Top-1 trên tập test (2.193 ảnh, 206 lớp) — research/eval_out/summary_test.csv
MODEL_SPECS = {
    "convnext": {
        "display": "ConvNeXt-Base",
        "arch": "convnext_base.fb_in22k_ft_in1k",
        "weights": "convnext_base.safetensors",
        "preprocess": "resize224",
        "test_top1": 92.61,
        "color": "#7c3aed",
    },
    "swin": {
        "display": "Swin-Base",
        "arch": "swin_base_patch4_window7_224.ms_in22k_ft_in1k",
        "weights": "swin_base.safetensors",
        "preprocess": "resize224",
        "test_top1": 90.10,
        "color": "#0369a1",
    },
    "resnet50": {
        "display": "ResNet50 (custom)",
        "arch": "resnet50_custom",
        "weights": "resnet50_custom.safetensors",
        "preprocess": "resize256_crop224",
        "test_top1": 65.75,
        "color": "#dc2626",
    },
}

# Soft-voting (trung bình xác suất) ConvNeXt + Swin: 93.21% Top-1 trên test,
# cao hơn từng mô hình đơn lẻ và cao hơn bỏ phiếu đa số 3 mô hình (92.43%).
DEFAULT_ENSEMBLE = "convnext,swin"


def _csv_env(name: str, default: str) -> list[str]:
    return [s.strip() for s in os.getenv(name, default).split(",") if s.strip()]


# Biến môi trường (đặt trên Railway → Variables)
ENABLED_MODELS = [m for m in _csv_env("ENABLED_MODELS", ",".join(MODEL_SPECS)) if m in MODEL_SPECS]
ENSEMBLE_MODELS = [m for m in _csv_env("ENSEMBLE_MODELS", DEFAULT_ENSEMBLE) if m in ENABLED_MODELS]
TORCH_THREADS = int(os.getenv("TORCH_THREADS", "0"))  # 0 = để PyTorch tự chọn
TOP_K = int(os.getenv("TOP_K", "3"))
MAX_DOWNLOAD_MB = float(os.getenv("MAX_DOWNLOAD_MB", "15"))
