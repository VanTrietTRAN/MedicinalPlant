# -*- coding: utf-8 -*-
"""
CI bootstrap phân tầng (stratified) cho Bảng III.

Bootstrap thường lấy mẫu lại toàn bộ 2193 ảnh. Với support 8–16 ảnh/lớp, một
resample có thể làm vài lớp biến mất hoặc tụt còn 1–2 ảnh; macro-F1 lấy trung
bình đều trên 206 lớp nên bị kéo xuống, và CI lệch hẳn về phía dưới điểm ước
lượng. Bootstrap phân tầng lấy lại đúng số ảnh của từng lớp nên giữ nguyên
support và không lớp nào biến mất.

Chạy:  python bootstrap_ci_stratified.py
Đọc:   eval_out/logits_test.npz  (labels + logits của cả ba model)
Ghi:   eval_out/ci_stratified_test.csv
"""

import csv
import os

import numpy as np
from sklearn.metrics import f1_score

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "eval_out")
NPZ = os.path.join(OUT_DIR, "logits_test.npz")
OUT_CSV = os.path.join(OUT_DIR, "ci_stratified_test.csv")

N_ITERS = 2000
SEED = 42


def stratified_ci(labels, preds, n_classes, n_iters=N_ITERS, seed=SEED):
    """CI phân tầng — resample trong từng lớp, giữ nguyên support."""
    rng = np.random.default_rng(seed)
    idx_by_class = [np.where(labels == c)[0] for c in range(n_classes)]
    lab = list(range(n_classes))
    f1_vals, acc_vals = [], []
    for _ in range(n_iters):
        idx = np.concatenate([rng.choice(ix, len(ix), replace=True)
                              for ix in idx_by_class if len(ix)])
        f1_vals.append(f1_score(labels[idx], preds[idx], average="macro",
                                labels=lab, zero_division=0))
        acc_vals.append((labels[idx] == preds[idx]).mean())
    return np.array(f1_vals), np.array(acc_vals)


def plain_ci(labels, preds, n_classes, n_iters=N_ITERS, seed=SEED):
    """Bootstrap thường — resample toàn bộ tập test, để đối chiếu."""
    rng = np.random.default_rng(seed)
    n = len(labels)
    lab = list(range(n_classes))
    f1_vals, acc_vals = [], []
    for _ in range(n_iters):
        idx = rng.integers(0, n, n)
        f1_vals.append(f1_score(labels[idx], preds[idx], average="macro",
                                labels=lab, zero_division=0))
        acc_vals.append((labels[idx] == preds[idx]).mean())
    return np.array(f1_vals), np.array(acc_vals)


def pct(v):
    return np.percentile(v, 2.5) * 100, np.percentile(v, 97.5) * 100


def main():
    z = np.load(NPZ, allow_pickle=True)
    labels = z["labels"]
    n_classes = len(z["classes"])
    supports = np.bincount(labels, minlength=n_classes)
    print("test: %d ảnh / %d lớp | support min %d, max %d, trung vị %d"
          % (len(labels), n_classes, supports.min(), supports.max(),
             int(np.median(supports))))
    print("số lớp có support < 10: %d\n" % int((supports < 10).sum()))

    models = ["ResNet50", "Swin-Base", "ConvNeXt-Base"]
    rows = []
    for name in models:
        preds = z["logits::%s" % name].argmax(1)
        f1_hat = f1_score(labels, preds, average="macro",
                          labels=list(range(n_classes)), zero_division=0) * 100
        acc_hat = (labels == preds).mean() * 100

        f1_s, acc_s = stratified_ci(labels, preds, n_classes)
        f1_p, acc_p = plain_ci(labels, preds, n_classes)

        s_lo, s_hi = pct(f1_s)
        p_lo, p_hi = pct(f1_p)
        as_lo, as_hi = pct(acc_s)
        ap_lo, ap_hi = pct(acc_p)

        print("== %s" % name)
        print("   macro-F1  điểm %.2f" % f1_hat)
        print("      stratified [%.2f, %.2f]  rộng %.2f  lệch tâm %+.2f"
              % (s_lo, s_hi, s_hi - s_lo, (s_lo + s_hi) / 2 - f1_hat))
        print("      thường     [%.2f, %.2f]  rộng %.2f  lệch tâm %+.2f"
              % (p_lo, p_hi, p_hi - p_lo, (p_lo + p_hi) / 2 - f1_hat))
        print("   top-1     điểm %.2f" % acc_hat)
        print("      stratified [%.2f, %.2f]  rộng %.2f"
              % (as_lo, as_hi, as_hi - as_lo))
        print("      thường     [%.2f, %.2f]  rộng %.2f\n"
              % (ap_lo, ap_hi, ap_hi - ap_lo))

        rows.append({
            "model": name, "macro_f1": round(f1_hat, 2),
            "f1_strat_lo": round(s_lo, 2), "f1_strat_hi": round(s_hi, 2),
            "f1_plain_lo": round(p_lo, 2), "f1_plain_hi": round(p_hi, 2),
            "top1": round(acc_hat, 2),
            "top1_strat_lo": round(as_lo, 2), "top1_strat_hi": round(as_hi, 2),
            "top1_plain_lo": round(ap_lo, 2), "top1_plain_hi": round(ap_hi, 2),
        })

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("ghi:", OUT_CSV)


if __name__ == "__main__":
    main()
