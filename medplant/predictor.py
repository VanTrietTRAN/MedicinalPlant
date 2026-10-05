"""Tiền xử lý ảnh, suy luận đa mô hình và kết hợp (soft-voting)."""
import io
import json
import logging
import time
from dataclasses import dataclass

import httpx
import torch
from PIL import Image, ImageOps
from torchvision import transforms

from . import config
from .models import build_model

log = logging.getLogger(__name__)

_NORMALIZE = transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD)
PREPROCESS = {
    "resize224": transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(), _NORMALIZE,
    ]),
    "resize256_crop224": transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), _NORMALIZE,
    ]),
}


@dataclass
class ModelResult:
    key: str
    display: str
    color: str
    topk: list[tuple[str, float]]  # [(tên lớp, xác suất 0..1)]


@dataclass
class Prediction:
    models: list[ModelResult]
    ensemble: list[tuple[str, float]]
    ensemble_members: list[str]
    elapsed_ms: float


class Predictor:
    def __init__(self):
        if config.TORCH_THREADS > 0:
            torch.set_num_threads(config.TORCH_THREADS)
        self.class_names = json.loads(config.CLASS_NAMES_PATH.read_text(encoding="utf-8"))
        self.models = {}
        for key in config.ENABLED_MODELS:
            t0 = time.perf_counter()
            self.models[key] = build_model(key, len(self.class_names))
            log.info("Đã nạp %s (%.1fs)", config.MODEL_SPECS[key]["display"], time.perf_counter() - t0)
        if not self.models:
            raise RuntimeError("ENABLED_MODELS rỗng — cần ít nhất một mô hình.")
        self.ensemble_members = config.ENSEMBLE_MODELS or list(self.models)
        self._warmup()

    def _warmup(self):
        """Chạy 1 lượt giả để request đầu tiên của người dùng không bị chậm."""
        self.predict(Image.new("RGB", (256, 256), (90, 140, 60)))

    @torch.inference_mode()
    def predict(self, img: Image.Image) -> Prediction:
        t0 = time.perf_counter()
        img = ImageOps.exif_transpose(img).convert("RGB")
        # Mỗi kiểu tiền xử lý chỉ tính 1 lần, dùng chung cho các mô hình cùng kiểu
        inputs = {}
        probs = {}
        for key, model in self.models.items():
            pp = config.MODEL_SPECS[key]["preprocess"]
            if pp not in inputs:
                inputs[pp] = PREPROCESS[pp](img).unsqueeze(0)
            probs[key] = model(inputs[pp]).softmax(dim=1)[0]

        results = [
            ModelResult(key, config.MODEL_SPECS[key]["display"], config.MODEL_SPECS[key]["color"],
                        self._topk(p))
            for key, p in probs.items()
        ]
        ens = torch.stack([probs[k] for k in self.ensemble_members]).mean(dim=0)
        return Prediction(results, self._topk(ens), self.ensemble_members,
                          (time.perf_counter() - t0) * 1000)

    def _topk(self, p: torch.Tensor) -> list[tuple[str, float]]:
        vals, idx = p.topk(config.TOP_K)
        return [(self.class_names[i], v) for v, i in zip(vals.tolist(), idx.tolist())]


def fetch_image(url: str) -> Image.Image:
    """Tải ảnh từ URL (http/https), giới hạn dung lượng để tránh làm nghẽn server."""
    url = url.strip()
    if not url.lower().startswith(("http://", "https://")):
        raise ValueError("Link phải bắt đầu bằng http:// hoặc https://")
    limit = int(config.MAX_DOWNLOAD_MB * 1024 * 1024)
    headers = {"User-Agent": "MedicinalPlantDemo/1.0 (+https://github.com/VanTrietTRAN/MedicinalPlant)"}
    with httpx.stream("GET", url, headers=headers, timeout=15, follow_redirects=True) as r:
        r.raise_for_status()
        buf = bytearray()
        for chunk in r.iter_bytes():
            buf.extend(chunk)
            if len(buf) > limit:
                raise ValueError(f"Ảnh vượt quá {config.MAX_DOWNLOAD_MB:g} MB")
    img = Image.open(io.BytesIO(buf))
    img.load()
    return img
