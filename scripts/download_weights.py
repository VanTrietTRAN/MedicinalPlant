"""Đảm bảo weights/*.safetensors có mặt và đúng sha256 (theo weights/manifest.json).

Trọng số lưu trên GitHub bằng Git LFS. Khi Railway build, mã nguồn thường chỉ chứa
file "con trỏ" LFS (vài trăm byte), nên script này tải file thật về từ WEIGHTS_BASE_URL.

    WEIGHTS_BASE_URL  mặc định: kho LFS của repo GitHub (nhánh main)
                      có thể đổi sang link GitHub Release / Hugging Face / S3...
"""
import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_DIR = Path(os.getenv("WEIGHTS_DIR", ROOT / "weights"))
BASE_URL = os.getenv(
    "WEIGHTS_BASE_URL",
    "https://media.githubusercontent.com/media/VanTrietTRAN/MedicinalPlant/main/weights",
).rstrip("/")


def sha256sum(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path):
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "medplant-build"})
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total and done % (50 << 20) < (1 << 20):
                print(f"    {done / 1e6:6.0f} / {total / 1e6:.0f} MB", flush=True)
    tmp.replace(dest)


def main() -> int:
    manifest = json.loads((WEIGHTS_DIR / "manifest.json").read_text())
    for name, meta in manifest.items():
        path = WEIGHTS_DIR / name
        if path.exists() and path.stat().st_size == meta["size"] and sha256sum(path) == meta["sha256"]:
            print(f"[ok]   {name}")
            continue
        url = f"{BASE_URL}/{name}"
        print(f"[tải]  {name} <- {url}", flush=True)
        download(url, path)
        if sha256sum(path) != meta["sha256"]:
            print(f"[LỖI]  sha256 không khớp cho {name}", file=sys.stderr)
            return 1
        print(f"[ok]   {name} ({meta['size'] / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
