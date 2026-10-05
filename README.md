# Nhận diện cây dược liệu Việt Nam (FAIR 2026)

**Demo trực tuyến:** https://web-production-68ba3.up.railway.app

Web demo nhận diện **206 loài cây dược liệu Việt Nam** từ ảnh, dùng ba mô hình học sâu
ConvNeXt-Base, Swin-Base và ResNet50 (tự cài đặt, train từ đầu). Kết luận cuối cùng là
**soft-voting** (trung bình xác suất) của ConvNeXt-Base + Swin-Base.

| Mô hình | Top-1 (test, 2.193 ảnh) | Top-5 |
|---|---|---|
| ResNet50 (custom, từ đầu) | 65,75% | 83,58% |
| Swin-Base | 90,10% | 96,58% |
| ConvNeXt-Base | 92,61% | 97,49% |
| **Soft-voting ConvNeXt + Swin** (dùng trong demo) | **93,21%** | — |

Số liệu từ [research/eval_out/summary_test.csv](research/eval_out/summary_test.csv).

## Cấu trúc

```
├── app.py                 # Giao diện web Gradio (entry point)
├── medplant/
│   ├── config.py          # Đặc tả mô hình, biến môi trường
│   ├── models.py          # Kiến trúc + nạp trọng số
│   └── predictor.py       # Tiền xử lý, suy luận, soft-voting, tải ảnh từ URL
├── assets/style.css       # Giao diện
├── data/class_names.json  # 206 tên lớp (thứ tự khớp lúc train)
├── data/vi_names.json     # Tên tiếng Việt, tên khác, trang nguồn cho 206 loài
├── weights/               # *.safetensors fp16 (Git LFS) + manifest.json (sha256)
├── scripts/
│   ├── convert_weights.py # .pth fp32 -> .safetensors fp16
│   └── download_weights.py# Tải/kiểm tra trọng số (dùng khi build Docker)
├── examples/              # Ảnh mẫu (tuỳ chọn) — app tự hiển thị
├── research/              # Notebook train, script đánh giá, kết quả thống kê
├── Dockerfile
├── .railway/railway.ts    # Cấu hình hạ tầng Railway (IaC); package.json chỉ phục vụ file này
└── DEPLOY.md              # Hạ tầng Railway, cách cập nhật, xử lý sự cố
```

## Tên tiếng Việt

App hiển thị song ngữ: tên tiếng Việt (tên chính) và tên khoa học. Tên Việt của cả 206 loài lấy từ
[tracuuduoclieu.vn](https://tracuuduoclieu.vn) — cùng nguồn dự án đã crawl khi xây dựng bộ dữ liệu
(`research/notebooks/crawl.ipynb`). Mỗi mục trong `data/vi_names.json` ghi URL trang nguồn, và trang đó
đều ghi đúng tên khoa học của lớp (6 loài được ghi dưới tên đồng nghĩa, có trường `note` giải thích).
Trên giao diện, thẻ kết luận có link **Tra cứu dược liệu** tới trang nguồn.

## Chạy trên máy

```bash
git lfs install && git clone https://github.com/VanTrietTRAN/MedicinalPlant.git
cd MedicinalPlant
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python scripts/download_weights.py   # bỏ qua được nếu git lfs đã tải đủ trọng số
python app.py                        # mở http://localhost:7860
```

## Biến môi trường

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `PORT` | `7860` | Cổng HTTP (Railway tự cấp) |
| `ENABLED_MODELS` | `convnext,swin,resnet50` | Mô hình được nạp. Bớt mô hình để giảm RAM |
| `ENSEMBLE_MODELS` | `convnext,swin` | Mô hình tham gia soft-voting |
| `TOP_K` | `3` | Số dự đoán hiển thị mỗi mô hình |
| `TORCH_THREADS` | `0` (tự động) | Số luồng CPU cho PyTorch |
| `MAX_DOWNLOAD_MB` | `15` | Giới hạn dung lượng ảnh tải từ link |
| `WEIGHTS_BASE_URL` | kho LFS của repo | Nguồn tải trọng số lúc build |

## Tối ưu so với notebook gốc (`codewebxla.ipynb`)

- Bỏ phụ thuộc đường dẫn Kaggle và file CSV ngoài; tên lớp lấy đúng thứ tự `ImageFolder` lúc train.
- Sửa tiền xử lý ResNet50 cho khớp lúc train (`Resize(256) + CenterCrop(224)`).
- Kết luận bằng soft-voting ConvNeXt + Swin (93,21%) thay cho bỏ phiếu đa số 3 mô hình (92,43%).
- Trọng số fp16 `.safetensors`: 798MB → 397MB, xác suất lệch < 3·10⁻⁴, nhãn trùng 100%.
- PyTorch bản CPU trong Docker; mỗi kiểu tiền xử lý chỉ tính một lần; warm-up khi khởi động.
- Hỗ trợ tải ảnh lên / webcam / clipboard / link, hiển thị top-3, chế độ tối, giao diện mobile.

## Tái lập kết quả nghiên cứu

Notebook train ở [research/notebooks/](research/notebooks/), script đánh giá ở [research/](research/).
Bộ dữ liệu đã chia: Kaggle `chanhhieudo/fn-ds-split`. Các script đánh giá cần checkpoint `.pth` gốc
(không đưa lên git) đặt cạnh script; sửa `CONFIG["data_root"]` trong `evaluate_models.py`.

> Kết quả chỉ mang tính tham khảo cho mục đích nghiên cứu, không thay thế chuyên gia khi thu hái
> hay sử dụng dược liệu.
