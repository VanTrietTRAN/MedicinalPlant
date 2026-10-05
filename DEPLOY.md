# Kế hoạch deploy lên Railway

Mục tiêu: đưa web demo nhận diện cây dược liệu lên một URL công khai để trình diễn.

## 1. Kiến trúc

```
GitHub: VanTrietTRAN/MedicinalPlant (nhánh main)
  ├─ mã nguồn app + Dockerfile + railway.json
  └─ weights/*.safetensors (Git LFS, ~397MB)
          │  git push  → Railway tự build lại
          ▼
Railway build (Dockerfile)
  1. python:3.12-slim + PyTorch CPU
  2. pip install -r requirements.txt
  3. scripts/download_weights.py  → tải 3 file trọng số, kiểm tra sha256
  4. COPY mã nguồn
          ▼
Railway container: python app.py  → Gradio lắng nghe 0.0.0.0:$PORT
          ▼
Domain công khai: https://<tên-service>.up.railway.app
```

Vì sao tải trọng số lúc build: GitHub giới hạn 100MB/file nên trọng số nằm trong Git LFS, còn mã nguồn
Railway kéo về chỉ chứa file "con trỏ" LFS. Script tải file thật từ `media.githubusercontent.com`
và kiểm tra sha256 theo `weights/manifest.json`, nên build hỏng ngay nếu file sai.

## 2. Yêu cầu tài nguyên

| Cấu hình `ENABLED_MODELS` | RAM khi chạy (đo thực tế) | Gợi ý |
|---|---|---|
| `convnext,swin,resnet50` (mặc định) | ~1,4GB | Cấp ≥ 2GB RAM |
| `convnext,swin` | ~1,1GB | Giữ nguyên độ chính xác 93,21% |
| `convnext` | ~0,6GB | Dùng khi gói Railway giới hạn RAM thấp |

- Image Docker ~1,5GB (PyTorch CPU + Gradio + trọng số).
- Thời gian suy luận: ~0,5s/ảnh cho cả 3 mô hình trên CPU 12 luồng; trên vCPU của Railway dự kiến 1–3s.
- Gói Trial của Railway có thể không đủ RAM cho cả 3 mô hình; nên dùng gói Hobby. Hãy kiểm tra
  hạn mức hiện tại tại https://railway.com/pricing.

## 3. Các bước deploy

### Bước 1 — Mã nguồn trên GitHub (đã thực hiện)
Repo `https://github.com/VanTrietTRAN/MedicinalPlant` đã có toàn bộ mã nguồn và trọng số LFS.
Kiểm tra trên GitHub: mở `weights/convnext_base.safetensors` phải thấy nhãn **"Stored with Git LFS"**.

### Bước 2 — Tạo project trên Railway
1. Đăng nhập https://railway.com bằng tài khoản GitHub.
2. **New Project → Deploy from GitHub repo** → cấp quyền cho Railway đọc repo → chọn `VanTrietTRAN/MedicinalPlant`.
3. Railway đọc `railway.json` và build bằng `Dockerfile` (không cần cấu hình build/start command).

### Bước 3 — Biến môi trường (tuỳ chọn)
Tab **Variables** của service. Không bắt buộc đặt gì; chỉ đặt khi cần thay đổi mặc định:

| Biến | Khi nào dùng |
|---|---|
| `ENABLED_MODELS=convnext,swin` | Muốn giảm RAM (xem bảng mục 2) |
| `TORCH_THREADS=2` | Đặt bằng số vCPU được cấp nếu thấy CPU bị tranh chấp |
| `WEIGHTS_BASE_URL=...` | Chuyển nguồn trọng số sang GitHub Release (xem mục 5) |

Không đặt `PORT` — Railway tự cấp và app tự đọc.

### Bước 4 — Mở domain công khai
Tab **Settings → Networking → Generate Domain**. Nếu Railway hỏi cổng, chọn cổng mà Railway gợi ý
(chính là `$PORT` app đang lắng nghe; xem dòng `Running on local URL: http://0.0.0.0:<port>` trong Deploy Logs).

### Bước 5 — Kiểm tra
1. **Build Logs**: thấy `[tải] convnext_base.safetensors ...` rồi `[ok]` cho cả 3 file.
2. **Deploy Logs**: thấy `Đã nạp ConvNeXt-Base`, `Đã nạp Swin-Base`, `Đã nạp ResNet50 (custom)`, rồi
   `Running on local URL`. Healthcheck `/` (timeout 300s) chuyển sang xanh.
3. Mở domain, tải một ảnh lá cây (vd. *Ginkgo biloba*) → bấm **Dự đoán** → có kết luận và 3 thẻ mô hình.
4. Dán thử một link ảnh → ảnh hiện lên ô bên trái, kết quả hiện bên phải.

## 4. Cập nhật về sau

```bash
git add -A && git commit -m "..." && git push
```
Railway tự build lại khi file trong `watchPatterns` của `railway.json` thay đổi (mã app, trọng số,
Dockerfile...). Sửa `research/` hay tài liệu sẽ **không** kích hoạt build.

Thay trọng số mới: đặt file `.pth` vào một thư mục rồi chạy
`python scripts/convert_weights.py --src <thư mục>` (sinh lại `.safetensors` + `manifest.json`), commit, push.

Thêm ảnh mẫu: chép ảnh `.jpg/.png` (có quyền chia sẻ) vào `examples/`, commit, push.

## 5. Xử lý sự cố

| Triệu chứng | Nguyên nhân | Cách xử lý |
|---|---|---|
| Build lỗi `HTTP Error 404` / `sha256 không khớp` khi tải trọng số | File LFS chưa được push, hoặc hết hạn mức băng thông Git LFS của GitHub (mỗi lần build tải ~400MB) | Kiểm tra mục 3/Bước 1. Hoặc chuyển sang GitHub Release: tạo release tag `weights-v1`, upload 3 file `.safetensors`, đặt `WEIGHTS_BASE_URL=https://github.com/VanTrietTRAN/MedicinalPlant/releases/download/weights-v1` rồi Redeploy |
| Container bị kill, log `Out of memory` / exit 137 | Thiếu RAM | Tăng giới hạn RAM của service hoặc đặt `ENABLED_MODELS=convnext,swin` |
| Healthcheck thất bại | App chưa nạp xong mô hình trong 300s, hoặc lỗi lúc khởi động | Xem Deploy Logs; tăng `healthcheckTimeout` trong `railway.json` |
| Dự đoán chậm (>5s) | vCPU ít/bị chia sẻ | Đặt `TORCH_THREADS` bằng số vCPU; bớt mô hình |
| "Không tải được ảnh từ link" | Link không phải ảnh trực tiếp, trang chặn bot, hoặc ảnh > 15MB | Dùng link ảnh trực tiếp (đuôi .jpg/.png) hoặc tải ảnh lên |

## 6. Chi phí cho demo

- Railway tính theo RAM/CPU sử dụng theo thời gian. Ngoài buổi demo có thể bật **App Sleeping**
  (Settings → Serverless) để service ngủ khi không có truy cập; lần truy cập đầu sẽ chờ ~20–40s để nạp mô hình.
- Hoặc tạm dừng bằng **Remove deployment** sau buổi demo; khi cần chỉ việc **Redeploy**.

## 7. Checklist trước buổi demo

- [ ] Build và deploy xanh, domain mở được
- [ ] Thử 3–5 ảnh thật của các loài trong danh sách `data/class_names.json`
- [ ] Thử trên điện thoại (giao diện mobile, chụp ảnh bằng webcam)
- [ ] Tắt App Sleeping trong buổi demo để tránh chờ khởi động
