#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
evaluate_models.py — Đánh giá đầy đủ 3 mô hình phân loại 206 loài dược liệu.

Sinh ra: Top-1, Top-5, macro-P/R/F1, weighted-F1, CI 95% (bootstrap),
McNemar p-value từng cặp, per-class P/R/F1, cặp loài nhầm nhiều nhất,
confusion matrix, và cache logits.

Các stage:
    --stage all      (mặc định) infer + metrics
    --stage infer    chỉ chạy model, cache logits vào logits_<split>.npz
    --stage metrics  chỉ tính lại metrics từ cache (không cần load model)
    --stage leak     kiểm tra rò rỉ dữ liệu train<->val/test (KHÔNG cần load model)

Ví dụ:
    python evaluate_models.py --stage leak
    python evaluate_models.py
    python evaluate_models.py --stage metrics

QUAN TRỌNG — các thông số dưới đây đã được đối chiếu trực tiếp với 3 notebook
huấn luyện (swin_transformer.ipynb, convnext.ipynb, ResNet50.ipynb).
Đừng sửa nếu không có lý do; sai transform => số vô nghĩa.
"""

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

# Console Windows mặc định cp1252 -> vỡ chữ tiếng Việt. Ép UTF-8 cho stdout/stderr.
for _s in ("stdout", "stderr"):
    _f = getattr(sys, _s, None)
    if _f is not None and hasattr(_f, "reconfigure"):
        try:
            _f.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

# ============================================================================
# CONFIG — SỬA Ở ĐÂY
# ============================================================================
CONFIG = {
    # Thư mục gốc của bộ dữ liệu đã chia, chứa 3 thư mục con train/ val/ test/
    # (bộ "medical-split-fn" tải từ Kaggle: chanhhieudo/fn-ds-split)
    "data_root": r"D:\datasets\medical-split-fn",

    # Đánh giá cả tập val (VẤN ĐỀ 2: điều tra chênh lệch val ~92% vs test 82.79%)
    "eval_val_too": True,

    "device": "auto",          # "auto" | "cpu" | "cuda"
    "batch_size": 32,
    "num_workers": 2,          # Windows: giữ thấp; forward trên CPU mới là nút cổ chai
    "out_dir": "eval_out",

    "bootstrap_n": 2000,       # số lần lặp bootstrap cho CI 95%
    "seed": 42,

    "n_confused_pairs": 40,    # số cặp nhầm lẫn xuất khẩu ra CSV
    "near_dup_hamming": 5,     # ngưỡng Hamming (trên dHash 64-bit) coi là ảnh trùng lặp
}

# Đường dẫn checkpoint — mặc định nằm cùng thư mục với file script này.
_HERE = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------------------
# ĐẶC TẢ MÔ HÌNH
#
# eval_tf: PHẢI khớp val_transforms lúc train.
#   "resize224"      = Resize((224,224)) + CenterCrop(224)   -> CenterCrop là no-op
#                      (ConvNeXt, Swin — xem convnext.ipynb / swin_transformer.ipynb)
#   "resize256_crop224" = Resize(256) + CenterCrop(224)
#                      (ResNet50 — xem ResNet50.ipynb; KHÁC hai model kia!)
#
# label_smoothing: dùng để tái tạo đúng cột "Loss" trong Bảng III.
# ----------------------------------------------------------------------------
MODEL_SPECS = {
    "ResNet50": {
        "ckpt": os.path.join(_HERE, "best_model_resnet50.pth"),
        "kind": "custom_resnet50",
        "eval_tf": "resize256_crop224",
        "label_smoothing": 0.0,
    },
    "ConvNeXt-Base": {
        "ckpt": os.path.join(_HERE, "best_model_convnext.pth"),
        "kind": "timm",
        "timm_name": "convnext_base.fb_in22k_ft_in1k",
        "eval_tf": "resize224",
        "label_smoothing": 0.1,
    },
    "Swin-Base": {
        "ckpt": os.path.join(_HERE, "best_swin_base_model.pth"),
        "kind": "timm",
        "timm_name": "swin_base_patch4_window7_224.ms_in22k_ft_in1k",
        "eval_tf": "resize224",
        "label_smoothing": 0.1,
    },
}

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".jfif", ".ppm")

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


# ============================================================================
# DUYỆT DỮ LIỆU  (khớp torchvision.datasets.DatasetFolder)
# ============================================================================
def find_classes(directory):
    """Giống DatasetFolder.find_classes: sắp xếp alphabet."""
    classes = sorted(e.name for e in os.scandir(directory) if e.is_dir())
    if not classes:
        raise RuntimeError(f"Không tìm thấy thư mục lớp nào trong: {directory}")
    return classes, {c: i for i, c in enumerate(classes)}


def make_index(split_dir):
    """Trả về (samples, classes). samples = [(path, label), ...] thứ tự cố định."""
    classes, class_to_idx = find_classes(split_dir)
    samples = []
    for cls in classes:
        cdir = os.path.join(split_dir, cls)
        for root, _, fnames in sorted(os.walk(cdir, followlinks=True)):
            for fname in sorted(fnames):
                if fname.lower().endswith(IMG_EXTS):
                    samples.append((os.path.join(root, fname), class_to_idx[cls]))
    if not samples:
        raise RuntimeError(f"Không tìm thấy ảnh nào trong: {split_dir}")
    return samples, classes


# ============================================================================
# MODEL
# ============================================================================
def _build_torch_bits():
    """Import torch muộn để --stage metrics / leak chạy được khi không có torch."""
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms
    from PIL import Image, ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    return torch, nn, Dataset, DataLoader, transforms, Image


def build_custom_resnet50(nn, num_classes):
    """Bản sao chính xác kiến trúc ResNet50 tự cài trong ResNet50.ipynb."""

    class Bottleneck(nn.Module):
        expansion = 4

        def __init__(self, in_channels, out_channels, stride=1, downsample=None):
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                                   stride=stride, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion,
                                   kernel_size=1, bias=False)
            self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
            self.relu = nn.ReLU(inplace=True)
            self.downsample = downsample

        def forward(self, x):
            identity = x
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.relu(self.bn2(self.conv2(out)))
            out = self.bn3(self.conv3(out))
            if self.downsample is not None:
                identity = self.downsample(x)
            out += identity
            return self.relu(out)

    class ResNet50(nn.Module):
        def __init__(self, num_classes=1000):
            super().__init__()
            self.in_channels = 64
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.layer1 = self._make_layer(Bottleneck, 64, 3, stride=1)
            self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2)
            self.layer3 = self._make_layer(Bottleneck, 256, 6, stride=2)
            self.layer4 = self._make_layer(Bottleneck, 512, 3, stride=2)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)

        def _make_layer(self, block, out_channels, blocks, stride):
            downsample = None
            if stride != 1 or self.in_channels != out_channels * block.expansion:
                downsample = nn.Sequential(
                    nn.Conv2d(self.in_channels, out_channels * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels * block.expansion),
                )
            layers = [block(self.in_channels, out_channels, stride, downsample)]
            self.in_channels = out_channels * block.expansion
            for _ in range(1, blocks):
                layers.append(block(self.in_channels, out_channels))
            return nn.Sequential(*layers)

        def forward(self, x):
            x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
            x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
            x = self.avgpool(x)
            return self.fc(torch_flatten(x))

    return ResNet50(num_classes=num_classes)


def torch_flatten(x):
    import torch
    return torch.flatten(x, 1)


def build_model(name, spec, num_classes, torch, nn):
    if spec["kind"] == "timm":
        import timm
        model = timm.create_model(spec["timm_name"], pretrained=False,
                                  num_classes=num_classes)
    elif spec["kind"] == "custom_resnet50":
        model = build_custom_resnet50(nn, num_classes)
    else:
        raise ValueError(spec["kind"])

    sd = torch.load(spec["ckpt"], map_location="cpu", weights_only=True)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}

    res = model.load_state_dict(sd, strict=False)
    miss, unexp = list(res.missing_keys), list(res.unexpected_keys)
    print(f"    load_state_dict: missing={len(miss)} unexpected={len(unexp)}")
    if miss:
        print(f"      MISSING (10 đầu): {miss[:10]}")
    if unexp:
        print(f"      UNEXPECTED (10 đầu): {unexp[:10]}")
    if miss or unexp:
        print("      !! CẢNH BÁO: state_dict không khớp hoàn toàn -> kiểm tra lại kiến trúc.")
    return model


class EvalDS:
    """Dataset ở mức module để pickle được cho DataLoader worker trên Windows (spawn)."""

    def __init__(self, samples, tf):
        self.samples, self.tf = samples, tf

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        from PIL import Image, ImageFile
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        path, label = self.samples[i]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), (0, 0, 0))
        return self.tf(img), label


def build_transform(kind, transforms):
    if kind == "resize224":
        # ConvNeXt / Swin: Resize((224,224)) rồi CenterCrop(224) (crop là no-op)
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    if kind == "resize256_crop224":
        # ResNet50: Resize(256) (cạnh ngắn) rồi CenterCrop(224)
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    raise ValueError(kind)


# ============================================================================
# INFERENCE
# ============================================================================
def run_inference(split, samples, classes, out_dir, cfg):
    torch, nn, Dataset, DataLoader, transforms, Image = _build_torch_bits()

    if cfg["device"] == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg["device"])
    print(f"\n[{split}] device = {device}  |  {len(samples)} ảnh  |  {len(classes)} lớp")

    labels = np.array([s[1] for s in samples], dtype=np.int64)
    store = {"labels": labels, "classes": np.array(classes, dtype=object),
             "paths": np.array([s[0] for s in samples], dtype=object)}

    for name, spec in MODEL_SPECS.items():
        if not os.path.isfile(spec["ckpt"]):
            print(f"  [!] Bỏ qua {name}: không thấy checkpoint {spec['ckpt']}")
            continue
        print(f"\n  --> {name}  (transform={spec['eval_tf']})")
        model = build_model(name, spec, len(classes), torch, nn).to(device).eval()

        tf = build_transform(spec["eval_tf"], transforms)
        loader = DataLoader(EvalDS(samples, tf), batch_size=cfg["batch_size"],
                            shuffle=False, num_workers=cfg["num_workers"],
                            pin_memory=(device.type == "cuda"))

        logits_all = np.zeros((len(samples), len(classes)), dtype=np.float32)
        pos, n_done = 0, 0
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(device, non_blocking=True)
                if device.type == "cuda":
                    with torch.autocast(device_type="cuda"):
                        out = model(x)
                else:
                    out = model(x)
                out = out.float().cpu().numpy()
                logits_all[pos:pos + len(out)] = out
                pos += len(out)
                n_done += len(out)
                if n_done % (cfg["batch_size"] * 10) == 0 or n_done == len(samples):
                    print(f"      {n_done}/{len(samples)}", end="\r", flush=True)
        print(f"      {len(samples)}/{len(samples)}  xong.")

        top1 = float((logits_all.argmax(1) == labels).mean())
        print(f"      Top-1 = {top1 * 100:.2f}%")
        store[f"logits::{name}"] = logits_all
        del model

    path = os.path.join(out_dir, f"logits_{split}.npz")
    np.savez_compressed(path, **store)
    print(f"\n  Đã lưu cache: {path}")
    return path


# ============================================================================
# METRICS
# ============================================================================
def softmax_ce(logits, labels, label_smoothing=0.0):
    z = logits - logits.max(axis=1, keepdims=True)
    logsumexp = np.log(np.exp(z).sum(axis=1)) + 0.0
    logp = z - logsumexp[:, None]
    n, C = logits.shape
    nll = -logp[np.arange(n), labels]
    if label_smoothing > 0:
        smooth = -logp.mean(axis=1)
        return float(((1 - label_smoothing) * nll + label_smoothing * smooth).mean())
    return float(nll.mean())


def prf_per_class(y_true, y_pred, n_classes):
    hit = (y_true == y_pred)
    tp = np.bincount(y_true[hit], minlength=n_classes).astype(float)
    pred_cnt = np.bincount(y_pred, minlength=n_classes).astype(float)
    support = np.bincount(y_true, minlength=n_classes).astype(float)
    fp = pred_cnt - tp
    fn = support - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        rec = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
    return prec, rec, f1, support


def macro_f1_from_preds(y_true, y_pred, n_classes):
    _, _, f1, _ = prf_per_class(y_true, y_pred, n_classes)
    return float(f1.mean())


def bootstrap_ci(y_true, y_pred, top5_hit, n_classes, B, seed):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    t1, t5, mf1 = np.empty(B), np.empty(B), np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        yt, yp = y_true[idx], y_pred[idx]
        t1[b] = (yt == yp).mean()
        t5[b] = top5_hit[idx].mean()
        mf1[b] = macro_f1_from_preds(yt, yp, n_classes)
    q = lambda a: (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))
    return q(t1), q(t5), q(mf1)


def mcnemar(correct_a, correct_b):
    """Trả về (b, c, p_value, method). b = A đúng/B sai, c = A sai/B đúng."""
    from scipy import stats
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    n = b + c
    if n == 0:
        return b, c, 1.0, "identical"
    if n < 25:
        p = float(stats.binomtest(b, n, 0.5).pvalue)
        return b, c, p, "exact binomial"
    chi2 = (abs(b - c) - 1.0) ** 2 / n
    p = float(stats.chi2.sf(chi2, 1))
    return b, c, p, "chi2 + continuity correction"


def genus(cls_name):
    return cls_name.split("_")[0]


def compute_metrics(split, out_dir, cfg):
    path = os.path.join(out_dir, f"logits_{split}.npz")
    if not os.path.isfile(path):
        print(f"[!] Không thấy {path} — chạy --stage infer trước.")
        return None
    d = np.load(path, allow_pickle=True)
    labels = d["labels"]
    classes = list(d["classes"])
    n_classes = len(classes)
    model_names = [k.split("::", 1)[1] for k in d.files if k.startswith("logits::")]

    print(f"\n{'=' * 78}\nSPLIT = {split}   |   {len(labels)} ảnh   |   {n_classes} lớp")
    print(f"3 lớp đầu : {classes[:3]}")
    print(f"3 lớp cuối: {classes[-3:]}")
    print("=" * 78)

    summary_rows = []
    correct_map = {}

    for name in model_names:
        logits = d[f"logits::{name}"]
        pred = logits.argmax(1)
        correct = (pred == labels)
        correct_map[name] = correct

        order5 = np.argsort(-logits, axis=1)[:, :5]
        top5_hit = (order5 == labels[:, None]).any(axis=1)

        prec, rec, f1, support = prf_per_class(labels, pred, n_classes)
        w = support / support.sum()

        ls = MODEL_SPECS.get(name, {}).get("label_smoothing", 0.0)
        row = {
            "model": name,
            "n_samples": len(labels),
            "top1": float(correct.mean()),
            "top5": float(top5_hit.mean()),
            "macro_precision": float(prec.mean()),
            "macro_recall": float(rec.mean()),
            "macro_f1": float(f1.mean()),
            "weighted_precision": float((prec * w).sum()),
            "weighted_recall": float((rec * w).sum()),
            "weighted_f1": float((f1 * w).sum()),
            "loss_ce_train_cfg": softmax_ce(logits, labels, ls),
            "loss_ce_plain": softmax_ce(logits, labels, 0.0),
            "label_smoothing_used": ls,
        }

        (t1lo, t1hi), (t5lo, t5hi), (flo, fhi) = bootstrap_ci(
            labels, pred, top5_hit, n_classes, cfg["bootstrap_n"], cfg["seed"])
        row.update({"top1_ci_lo": t1lo, "top1_ci_hi": t1hi,
                    "top5_ci_lo": t5lo, "top5_ci_hi": t5hi,
                    "macro_f1_ci_lo": flo, "macro_f1_ci_hi": fhi})
        summary_rows.append(row)

        print(f"\n[{name}]")
        print(f"  Top-1        : {row['top1']*100:6.2f}%   CI95 [{t1lo*100:.2f}, {t1hi*100:.2f}]")
        print(f"  Top-5        : {row['top5']*100:6.2f}%   CI95 [{t5lo*100:.2f}, {t5hi*100:.2f}]")
        print(f"  macro-P/R/F1 : {row['macro_precision']*100:.2f} / "
              f"{row['macro_recall']*100:.2f} / {row['macro_f1']*100:.2f}"
              f"   CI95(F1) [{flo*100:.2f}, {fhi*100:.2f}]")
        print(f"  weighted-F1  : {row['weighted_f1']*100:6.2f}%")
        gap = row["weighted_f1"] - row["macro_f1"]
        print(f"  weighted-macro gap: {gap*100:+.2f} điểm "
              f"({'lớp ít ảnh đang bị bỏ rơi' if gap > 0.02 else 'ổn'})")
        print(f"  Loss (LS={ls})  : {row['loss_ce_train_cfg']:.4f}   (CE thuần: {row['loss_ce_plain']:.4f})")

        # per-class CSV
        pc = os.path.join(out_dir, f"per_class_{split}_{name.replace('/', '-')}.csv")
        with open(pc, "w", newline="", encoding="utf-8-sig") as f:
            wcsv = csv.writer(f)
            wcsv.writerow(["class", "precision", "recall", "f1", "support"])
            for i in np.argsort(f1):
                wcsv.writerow([classes[i], f"{prec[i]:.4f}", f"{rec[i]:.4f}",
                               f"{f1[i]:.4f}", int(support[i])])

        # confusion matrix
        cm = np.zeros((n_classes, n_classes), dtype=np.int32)
        np.add.at(cm, (labels, pred), 1)
        np.save(os.path.join(out_dir, f"confmat_{split}_{name.replace('/', '-')}.npy"), cm)

    # ---------------- summary CSV ----------------
    scsv = os.path.join(out_dir, f"summary_{split}.csv")
    fields = list(summary_rows[0].keys())
    with open(scsv, "w", newline="", encoding="utf-8-sig") as f:
        wcsv = csv.DictWriter(f, fieldnames=fields)
        wcsv.writeheader()
        for r in summary_rows:
            wcsv.writerow(r)
    print(f"\n  -> {scsv}")

    # ---------------- McNemar ----------------
    mrows = []
    print(f"\n{'-' * 78}\nMcNemar (kiểm định cặp, split={split})\n{'-' * 78}")
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            a, b_ = model_names[i], model_names[j]
            nb, nc, p, meth = mcnemar(correct_map[a], correct_map[b_])
            verdict = ("KHÁC BIỆT có ý nghĩa thống kê (p<0.05)" if p < 0.05
                       else "KHÔNG khác biệt có ý nghĩa (p>=0.05) -> phải viết 'tương đương'")
            mrows.append({"model_a": a, "model_b": b_,
                          "a_correct_b_wrong": nb, "a_wrong_b_correct": nc,
                          "p_value": p, "method": meth, "verdict": verdict})
            print(f"  {a} vs {b_}: b={nb} c={nc}  p={p:.4g}  [{meth}]\n      -> {verdict}")
    mcsv = os.path.join(out_dir, f"mcnemar_{split}.csv")
    with open(mcsv, "w", newline="", encoding="utf-8-sig") as f:
        wcsv = csv.DictWriter(f, fieldnames=list(mrows[0].keys()))
        wcsv.writeheader()
        for r in mrows:
            wcsv.writerow(r)
    print(f"  -> {mcsv}")

    # ---------------- accuracy trên phần "sạch" (đã loại ảnh rò rỉ) ----------------
    dpath = os.path.join(out_dir, f"dupmask_{split}.npz")
    if os.path.isfile(dpath):
        dm = np.load(dpath, allow_pickle=True)
        near = dm["is_near_dup"]
        if len(near) == len(labels):
            clean = ~near
            print(f"\n{'-' * 78}\nAccuracy sau khi LOẠI ảnh rò rỉ (gần trùng với train), split={split}\n{'-' * 78}")
            print(f"  Giữ lại {int(clean.sum())}/{len(labels)} ảnh "
                  f"({clean.mean()*100:.1f}%); loại {int(near.sum())} ảnh rò rỉ.")
            print(f"{'Model':<16}{'Top-1 (all)':>13}{'Top-1 (sạch)':>15}"
                  f"{'Top-1 (rò rỉ)':>15}{'chênh':>9}")
            clean_rows = []
            for name in model_names:
                pred = d[f"logits::{name}"].argmax(1)
                ok = (pred == labels)
                a_all = float(ok.mean())
                a_cln = float(ok[clean].mean()) if clean.any() else float("nan")
                a_dup = float(ok[near].mean()) if near.any() else float("nan")
                print(f"{name:<16}{a_all*100:>12.2f}%{a_cln*100:>14.2f}%"
                      f"{a_dup*100:>14.2f}%{(a_all-a_cln)*100:>+8.2f}")
                clean_rows.append({"model": name, "top1_all": a_all,
                                   "top1_clean": a_cln, "top1_leaked": a_dup,
                                   "inflation_points": (a_all - a_cln) * 100,
                                   "n_clean": int(clean.sum()), "n_leaked": int(near.sum())})
            ccsv2 = os.path.join(out_dir, f"clean_subset_{split}.csv")
            with open(ccsv2, "w", newline="", encoding="utf-8-sig") as f:
                wcsv = csv.DictWriter(f, fieldnames=list(clean_rows[0].keys()))
                wcsv.writeheader()
                for r in clean_rows:
                    wcsv.writerow(r)
            print(f"  -> {ccsv2}")

    # ---------------- cặp nhầm lẫn ----------------
    best = max(summary_rows, key=lambda r: r["top1"])["model"]
    logits = d[f"logits::{best}"]
    pred = logits.argmax(1)
    pairs = Counter()
    for t, p in zip(labels, pred):
        if t != p:
            pairs[(int(t), int(p))] += 1
    ccsv = os.path.join(out_dir, f"confused_pairs_{split}.csv")
    with open(ccsv, "w", newline="", encoding="utf-8-sig") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["model", "true_class", "predicted_class", "count",
                       "support_true", "pct_of_true_class", "same_genus"])
        support = np.bincount(labels, minlength=n_classes)
        for (t, p), cnt in pairs.most_common(cfg["n_confused_pairs"]):
            wcsv.writerow([best, classes[t], classes[p], cnt, int(support[t]),
                           f"{cnt / max(support[t], 1) * 100:.1f}",
                           "YES" if genus(classes[t]) == genus(classes[p]) else "no"])
    print(f"\n{'-' * 78}\n15 cặp nhầm nhiều nhất (model tốt nhất: {best})\n{'-' * 78}")
    same_g = 0
    for (t, p), cnt in pairs.most_common(15):
        sg = genus(classes[t]) == genus(classes[p])
        same_g += sg
        print(f"  {cnt:3d}x  {classes[t]}  ->  {classes[p]}" + ("   [CÙNG CHI]" if sg else ""))
    total_sg = sum(1 for (t, p) in pairs if genus(classes[t]) == genus(classes[p]))
    print(f"\n  Trong top-15: {same_g}/15 cặp cùng chi (genus).")
    print(f"  Toàn bộ: {total_sg}/{len(pairs)} cặp nhầm là cùng chi "
          f"({total_sg / max(len(pairs), 1) * 100:.1f}%) -> bằng chứng fine-grained.")
    print(f"  -> {ccsv}")

    # ---------------- dòng dán vào bài báo ----------------
    print(f"\n{'=' * 78}\nDÁN VÀO BẢNG III (split={split})\n{'=' * 78}")
    print(f"{'Model':<16}{'Top-1':>9}{'Top-5':>9}{'macro-P':>10}{'macro-R':>10}"
          f"{'macro-F1':>11}{'w-F1':>9}{'Loss':>8}")
    for r in summary_rows:
        print(f"{r['model']:<16}{r['top1']*100:>8.2f}%{r['top5']*100:>8.2f}%"
              f"{r['macro_precision']*100:>9.2f}%{r['macro_recall']*100:>9.2f}%"
              f"{r['macro_f1']*100:>10.2f}%{r['weighted_f1']*100:>8.2f}%"
              f"{r['loss_ce_train_cfg']:>8.2f}")
    print("\nLaTeX:")
    for r in summary_rows:
        print(f"{r['model']} & {r['top1']*100:.2f} & {r['top5']*100:.2f} & "
              f"{r['macro_precision']*100:.2f} & {r['macro_recall']*100:.2f} & "
              f"{r['macro_f1']*100:.2f} & {r['weighted_f1']*100:.2f} & "
              f"{r['loss_ce_train_cfg']:.2f} \\\\")

    return summary_rows


# ============================================================================
# VẤN ĐỀ 2 — DÒ RÒ RỈ DỮ LIỆU
# ============================================================================
def dhash64(path, Image):
    """dHash 64-bit: resize 9x8 grayscale, so sánh pixel liền kề theo hàng."""
    try:
        img = Image.open(path).convert("L").resize((9, 8), Image.BILINEAR)
    except Exception:
        return None
    a = np.asarray(img, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return np.packbits(bits).view(np.uint64)[0]


def hamming_matrix_min(qa, qb, chunk=512):
    """Với mỗi phần tử của qa, tìm Hamming distance nhỏ nhất tới qb."""
    best = np.full(len(qa), 64, dtype=np.uint8)
    arg = np.full(len(qa), -1, dtype=np.int64)
    for i in range(0, len(qa), chunk):
        blk = qa[i:i + chunk]
        x = np.bitwise_xor(blk[:, None], qb[None, :])
        dist = np.bitwise_count(x).astype(np.uint8)
        j = dist.argmin(axis=1)
        best[i:i + chunk] = dist[np.arange(len(blk)), j]
        arg[i:i + chunk] = j
    return best, arg


def parse_seq_index(fname):
    """'Abutilon_indicum_57.jpg' -> 57. Crawler đánh số tuần tự theo observation."""
    stem = os.path.splitext(os.path.basename(fname))[0]
    tail = stem.rsplit("_", 1)[-1]
    return int(tail) if tail.isdigit() else None


def run_leak_check(cfg, out_dir):
    _, _, _, _, _, Image = _build_torch_bits()
    root = cfg["data_root"]
    splits = {}
    for s in ("train", "val", "test"):
        sd = os.path.join(root, s)
        if os.path.isdir(sd):
            splits[s] = make_index(sd)[0]
    if "train" not in splits:
        print("[!] Không thấy thư mục train/ — bỏ qua kiểm tra rò rỉ.")
        return

    print(f"\n{'=' * 78}\nVẤN ĐỀ 2 — KIỂM TRA RÒ RỈ DỮ LIỆU\n{'=' * 78}")
    for s, sm in splits.items():
        print(f"  {s}: {len(sm)} ảnh")

    print("\n[1/2] Phân tích chỉ số tuần tự trong tên file")
    print("      (crawler lưu {loài}_{n}.jpg; ảnh cùng 1 observation có n liền nhau)")
    owner = {}
    for s, sm in splits.items():
        for p, lab in sm:
            n = parse_seq_index(p)
            if n is not None:
                owner[(lab, n)] = s
    adjacent_cross = Counter()
    adjacent_total = 0
    for (lab, n), s in owner.items():
        s2 = owner.get((lab, n + 1))
        if s2 is None:
            continue
        adjacent_total += 1
        if s2 != s:
            adjacent_cross[tuple(sorted((s, s2)))] += 1
    print(f"      Cặp ảnh có chỉ số liền kề: {adjacent_total}")
    for k, v in sorted(adjacent_cross.items()):
        pct = v / max(adjacent_total, 1) * 100
        print(f"        {k[0]:<5} <-> {k[1]:<5}: {v:5d} cặp liền kề bị tách ({pct:.1f}%)")
    if adjacent_total and sum(adjacent_cross.values()) / adjacent_total > 0.1:
        print("      => Chia theo ẢNH, không theo observation. Ảnh gần trùng bị tách"
              " qua nhiều tập.")
    else:
        print("      => Ít cặp liền kề bị tách; có vẻ chia theo khối.")

    print(f"\n[2/2] Dò ảnh trùng/gần trùng bằng dHash (ngưỡng Hamming <= {cfg['near_dup_hamming']})")
    hashes, labs = {}, {}
    for s, sm in splits.items():
        print(f"      băm {s} ...", end="", flush=True)
        hs, ls = [], []
        for p, lab in sm:
            h = dhash64(p, Image)
            if h is not None:
                hs.append(h); ls.append(lab)
        hashes[s] = np.array(hs, dtype=np.uint64)
        labs[s] = np.array(ls, dtype=np.int64)
        print(f" {len(hs)} ảnh")

    rows = []
    thr = cfg["near_dup_hamming"]
    for s in ("val", "test"):
        if s not in hashes:
            continue
        dist, arg = hamming_matrix_min(hashes[s], hashes["train"])
        exact = int((dist == 0).sum())
        near = int((dist <= thr).sum())
        same_cls = int(((dist <= thr) & (labs["train"][arg] == labs[s])).sum())
        n = len(dist)
        print(f"\n      {s} vs train:")
        print(f"        trùng khít  (d=0)   : {exact:5d} / {n}  ({exact/n*100:.2f}%)")
        print(f"        gần trùng   (d<={thr}) : {near:5d} / {n}  ({near/n*100:.2f}%)")
        print(f"          trong đó cùng lớp : {same_cls:5d}  ({same_cls/n*100:.2f}%)")
        rows.append({"split": s, "n": n, "exact_dup_vs_train": exact,
                     "near_dup_vs_train": near, "near_dup_same_class": same_cls,
                     "exact_pct": exact / n * 100, "near_pct": near / n * 100})

        # Lưu mask để mục metrics đo được accuracy trên phần "sạch" (không rò rỉ).
        np.savez_compressed(
            os.path.join(out_dir, f"dupmask_{s}.npz"),
            min_hamming=dist,
            is_exact_dup=(dist == 0),
            is_near_dup=(dist <= thr),
            paths=np.array([p for p, _ in splits[s]], dtype=object),
        )

    lcsv = os.path.join(out_dir, "leakage_report.csv")
    if rows:
        with open(lcsv, "w", newline="", encoding="utf-8-sig") as f:
            wcsv = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wcsv.writeheader()
            for r in rows:
                wcsv.writerow(r)
        print(f"\n  -> {lcsv}")
        if len(rows) == 2:
            gap = rows[0]["near_pct"] - rows[1]["near_pct"]
            print(f"\n  KẾT LUẬN: val có {gap:+.2f} điểm % ảnh gần-trùng với train nhiều hơn test.")
            if gap > 3:
                print("    => Đây chính là lý do val cao hơn test. Val bị 'thổi phồng' "
                      "do rò rỉ observation.\n"
                      "    => Trong bài báo: chỉ báo cáo số trên TEST, và nêu rõ hạn chế "
                      "chia theo ảnh thay vì theo observation.")
            else:
                print("    => Rò rỉ không đủ giải thích chênh lệch; "
                      "nghi vấn khác biệt checkpoint/transform lúc vẽ biểu đồ.")


# ============================================================================
# MAIN
# ============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["all", "infer", "metrics", "leak"])
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    args = ap.parse_args()

    cfg = dict(CONFIG)
    if args.data_root:
        cfg["data_root"] = args.data_root
    if args.device:
        cfg["device"] = args.device
    if args.out_dir:
        cfg["out_dir"] = args.out_dir
    if args.batch_size:
        cfg["batch_size"] = args.batch_size
    if args.num_workers is not None:
        cfg["num_workers"] = args.num_workers

    out_dir = os.path.abspath(cfg["out_dir"])
    os.makedirs(out_dir, exist_ok=True)

    splits = ["test"] + (["val"] if cfg["eval_val_too"] else [])

    if args.stage == "leak":
        run_leak_check(cfg, out_dir)
        return

    if args.stage in ("all", "infer"):
        if not os.path.isdir(cfg["data_root"]):
            print(f"[LỖI] Không thấy data_root: {cfg['data_root']}\n"
                  f"       Sửa CONFIG['data_root'] hoặc dùng --data-root.")
            sys.exit(1)
        for split in splits:
            sd = os.path.join(cfg["data_root"], split)
            if not os.path.isdir(sd):
                print(f"[!] Bỏ qua split '{split}': không thấy {sd}")
                continue
            samples, classes = make_index(sd)
            print(f"\n[{split}] {len(samples)} ảnh, {len(classes)} lớp")
            print(f"  3 lớp đầu : {classes[:3]}")
            print(f"  3 lớp cuối: {classes[-3:]}")
            if len(classes) != 206:
                print(f"  !! CẢNH BÁO: {len(classes)} lớp, kỳ vọng 206.")
            run_inference(split, samples, classes, out_dir, cfg)

    if args.stage in ("all", "metrics"):
        results = {}
        for split in splits:
            r = compute_metrics(split, out_dir, cfg)
            if r:
                results[split] = r
        # VẤN ĐỀ 2: đối chiếu val vs test
        if "val" in results and "test" in results:
            print(f"\n{'=' * 78}\nVẤN ĐỀ 2 — ĐỐI CHIẾU VAL vs TEST (cùng code, cùng checkpoint)\n{'=' * 78}")
            print(f"{'Model':<16}{'val Top-1':>12}{'test Top-1':>12}{'chênh':>10}")
            vmap = {r["model"]: r for r in results["val"]}
            for r in results["test"]:
                v = vmap.get(r["model"])
                if not v:
                    continue
                gap = (v["top1"] - r["top1"]) * 100
                flag = "  <-- BẤT THƯỜNG" if gap > 5 else ""
                print(f"{r['model']:<16}{v['top1']*100:>11.2f}%{r['top1']*100:>11.2f}%"
                      f"{gap:>+9.2f}{flag}")
            print("\nNếu chênh vẫn lớn -> chạy: python evaluate_models.py --stage leak")

    print("\nXong.")


if __name__ == "__main__":
    main()
