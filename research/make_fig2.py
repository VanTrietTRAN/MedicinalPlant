# -*- coding: utf-8 -*-
"""
Dựng lại Fig. 2 của bài báo FAIR 2026.

Hình cũ ghép 3 ảnh chụp màn hình từ báo cáo môn học. Panel ConvNeXt trong hình đó
đến từ *run Mixup* (best val 85.16) chứ không phải checkpoint được báo cáo trong
Bảng III (best_model_convnext.pth, val 92.37) — hai run khác nhau, nên hình mâu
thuẫn với bảng. Panel Swin cũng vẽ toàn bộ 52 epoch trong khi checkpoint được
dùng là bản lưu ở epoch 13.

Hình mới chỉ chứa dữ liệu truy vết được về đúng các checkpoint trong Bảng III:
  (a) đường học của ResNet-50 — đọc trực tiếp từ log trong ResNet50.ipynb, và
      checkpoint của nó tái lập chính xác con số test 65.75% trong bảng;
  (b) F1 theo từng lớp (đã sắp xếp) của cả ba model, tính từ eval_out/, tức là
      từ đúng ba file .pth mà Bảng III báo cáo.

Chạy:  python make_fig2.py
Xuất:  eval_out/fig2_training_and_perclass.png
"""

import csv
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT_DIR = os.path.join(HERE, "eval_out")
OUT_PNG = os.path.join(OUT_DIR, "fig2_training_and_perclass.png")

# Bản để nộp: đặt cạnh file .docx cho dễ tìm. PDF là bản vector — IEEE ưu tiên
# vector, và đây cũng là bản để gửi kèm nếu ban tổ chức yêu cầu source figure.
EXPORT_PNG = os.path.join(ROOT, "Fig2_training_and_perclass.png")
EXPORT_PDF = os.path.join(ROOT, "Fig2_training_and_perclass.pdf")

# Bảng màu categorical đã validate (slot 1/2/3). Kèm kiểu nét khác nhau để hình
# vẫn đọc được khi in trắng đen — IEEE proceedings thường in grayscale.
C_BLUE = "#2a78d6"
C_ORANGE = "#eb6834"
C_AQUA = "#1baf7a"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#d8d8d4"


def resnet_history(nb_path):
    """Đọc train/val accuracy từng epoch ra khỏi output của ResNet50.ipynb."""
    nb = json.load(open(nb_path, encoding="utf-8"))
    text = ""
    for cell in nb["cells"]:
        for out in cell.get("outputs", []):
            if "text" in out:
                text += "".join(out["text"])
    train, val = [], []
    for line in (l.strip() for l in text.splitlines()):
        m = re.match(r"^train Loss: [\d.]+ Acc: ([\d.]+)$", line)
        if m:
            train.append(float(m.group(1)))
        m = re.match(r"^val Loss: [\d.]+ Acc: ([\d.]+)$", line)
        if m:
            val.append(float(m.group(1)))
    assert len(train) == len(val) and train, "không đọc được history của ResNet-50"
    return train, val


def per_class_f1(model_name):
    """F1 từng lớp trên tập test, sắp xếp giảm dần."""
    path = os.path.join(OUT_DIR, "per_class_test_%s.csv" % model_name)
    with open(path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    return sorted((float(r["f1"]) for r in rows), reverse=True)


def style_axes(ax):
    ax.grid(True, color=GRID, linewidth=0.5, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(colors=INK_2, length=2, width=0.6, labelsize=6)


def main():
    train, val = resnet_history(os.path.join(HERE, "ResNet50.ipynb"))
    epochs = range(1, len(train) + 1)

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 7,
        "axes.titlesize": 7.5,
        "axes.labelsize": 7,
        "legend.fontsize": 6.2,
        "figure.dpi": 400,
        "savefig.dpi": 400,
    })

    # Giữ hình thấp: nó chiếm nguyên chiều ngang 2 cột, cao thêm là bài tràn sang
    # trang thứ 6. 1.72 in ở đây ~ 1.80 in sau khi đặt vào Word.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 1.72))

    # ---- (a) ResNet-50: khoảng cách train/val ---------------------------------
    ax1.plot(epochs, train, color=C_BLUE, linewidth=1.4, linestyle="-",
             label="Training")
    ax1.plot(epochs, val, color=C_ORANGE, linewidth=1.4, linestyle="--",
             label="Validation")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("(a) ResNet-50 (from scratch)", color=INK, pad=4)
    ax1.set_xlim(0, len(train))
    ax1.set_ylim(0, 1.0)
    ax1.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    style_axes(ax1)

    # Chú thích khoảng cách train–val ở epoch cuối: đây là luận điểm của panel.
    last = len(train)
    ax1.annotate(
        "", xy=(last, train[-1]), xytext=(last, val[-1]),
        arrowprops=dict(arrowstyle="<->", color=INK_2, linewidth=0.7,
                        shrinkA=0, shrinkB=0),
    )
    ax1.text(last - 4, (train[-1] + val[-1]) / 2, "gap\n%.2f" % (train[-1] - val[-1]),
             ha="right", va="center", color=INK_2, fontsize=6, linespacing=1.1)
    ax1.legend(loc="lower right", frameon=False, handlelength=1.8,
               borderaxespad=0.3, labelcolor=INK_2)

    # ---- (b) F1 theo lớp của đúng ba checkpoint trong Bảng III ----------------
    series = [
        ("ConvNeXt-Base", C_BLUE, "-"),
        ("Swin-Base", C_ORANGE, "--"),
        ("ResNet50", C_AQUA, ":"),
    ]
    label_of = {"ResNet50": "ResNet-50"}
    for name, color, ls in series:
        f1 = per_class_f1(name)
        ax2.plot(range(1, len(f1) + 1), f1, color=color, linewidth=1.4,
                 linestyle=ls, label=label_of.get(name, name))
    ax2.set_xlabel("Class rank (sorted by F1)")
    ax2.set_ylabel("Per-class F1")
    ax2.set_title("(b) Per-class F1 on the test set", color=INK, pad=4)
    ax2.set_xlim(1, 206)
    ax2.set_ylim(0, 1.02)
    ax2.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax2.set_xticks([1, 50, 100, 150, 206])
    style_axes(ax2)
    ax2.legend(loc="lower left", frameon=False, handlelength=1.8,
               borderaxespad=0.3, labelcolor=INK_2)

    fig.tight_layout(pad=0.35, w_pad=1.6)
    for path in (OUT_PNG, EXPORT_PNG, EXPORT_PDF):
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        print("saved:", path)

    # Vài con số để viết caption / phần thảo luận cho khớp.
    for name, _, _ in series:
        f1 = per_class_f1(name)
        n = len(f1)
        print("%-14s median F1 %.3f | classes F1<0.5: %d | F1=0: %d | F1>=0.9: %d"
              % (label_of.get(name, name), f1[n // 2],
                 sum(1 for v in f1 if v < 0.5), sum(1 for v in f1 if v == 0),
                 sum(1 for v in f1 if v >= 0.9)))
    print("ResNet-50 final train %.4f / val %.4f (epoch %d)"
          % (train[-1], val[-1], len(train)))


if __name__ == "__main__":
    main()
