"""Chuyển checkpoint .pth (fp32, pickle) sang .safetensors fp16 cho bản deploy.

- Dung lượng giảm ~50% (≈800MB -> ≈400MB), load không cần pickle.
- Lúc chạy, trọng số được nâng lại fp32 nên kết quả gần như trùng khớp.
- Sinh weights/manifest.json (sha256 + kích thước) để bước build kiểm tra toàn vẹn.

Cách dùng:
    python scripts/convert_weights.py --src "<thư mục chứa .pth>"
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from medplant.config import MODEL_SPECS, WEIGHTS_DIR  # noqa: E402

# Tên file .pth gốc (sinh ra từ các notebook train) -> key trong MODEL_SPECS
SOURCE_FILES = {
    "convnext": "best_model_convnext.pth",
    "swin": "best_swin_base_model.pth",
    "resnet50": "best_model_resnet50.pth",
}


def sha256sum(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Thư mục chứa các file .pth gốc")
    args = ap.parse_args()
    src = Path(args.src)

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for key, fname in SOURCE_FILES.items():
        sd = torch.load(src / fname, map_location="cpu", weights_only=True)
        sd = sd.get("state_dict", sd)
        sd = {k.removeprefix("module."): v for k, v in sd.items()}
        # Chỉ hạ fp16 cho tensor số thực; giữ nguyên int (vd. num_batches_tracked)
        sd = {k: (v.half() if v.is_floating_point() else v).contiguous() for k, v in sd.items()}

        out = WEIGHTS_DIR / MODEL_SPECS[key]["weights"]
        save_file(sd, str(out))
        manifest[out.name] = {"sha256": sha256sum(out), "size": out.stat().st_size}
        print(f"{fname:28s} -> {out.name:28s} {out.stat().st_size / 1e6:7.1f} MB")

    (WEIGHTS_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Đã ghi weights/manifest.json")


if __name__ == "__main__":
    main()
