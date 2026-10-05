# Deploy lên Railway

**Bản demo đang chạy:** https://web-production-68ba3.up.railway.app

| Thành phần | Giá trị |
|---|---|
| Workspace / gói | haha Hehe's Projects — Railway **Pro** |
| Project | `medicinal-plant` (`bfbff7ea-5330-48e6-9d22-ddf5885885d9`) |
| Environment | `production` |
| Service | `web` — nguồn GitHub `VanTrietTRAN/MedicinalPlant`, nhánh `main` |
| Region | Singapore (`asia-southeast1-eqsg3a`), 1 replica |
| Build | `Dockerfile` (PyTorch CPU + tải trọng số từ Git LFS, kiểm tra sha256) |
| Cổng | `PORT=8080`, domain trỏ vào 8080 |
| Healthcheck | `GET /`, timeout 300s |
| Cấu hình hạ tầng | [.railway/railway.ts](.railway/railway.ts) (Infrastructure as Code) |

## 1. Kiến trúc

```
git push (nhánh main)
   │   chỉ build lại khi đổi: app.py, medplant/, assets/, data/, examples/,
   │   weights/, scripts/, requirements.txt, Dockerfile, .railway/
   ▼
Railway build (Dockerfile)
   1. python:3.12-slim + PyTorch CPU 2.13
   2. pip install -r requirements.txt
   3. scripts/download_weights.py → tải 3 file .safetensors (~397MB) từ
      media.githubusercontent.com (Git LFS), kiểm tra sha256 theo weights/manifest.json
   4. COPY mã nguồn → chạy bằng user không phải root
   ▼
Container (Singapore): python app.py → Gradio 0.0.0.0:8080
   ▼
https://web-production-68ba3.up.railway.app
```

Vì sao tải trọng số lúc build: GitHub giới hạn 100MB/file nên trọng số nằm trong Git LFS, mà mã nguồn
Railway kéo về chỉ chứa file "con trỏ" LFS. `.dockerignore` loại các file con trỏ này để chúng không ghi đè
lên file thật đã tải.

## 2. Số liệu thực tế trên Railway

Đo ngày 05/10/2026, region Singapore:
- Khởi động container: nạp 3 mô hình 2,5s, sẵn sàng phục vụ sau ~3s.
- Suy luận cả 3 mô hình: ~340ms/ảnh. Một lượt dự đoán trên trình duyệt (gồm upload): ~2s.
- Tải trang: ~0,9s. RAM: ~1,4GB với 3 mô hình, gói Pro dư sức chứa.
- Đã test end-to-end trên domain công khai: dán link ảnh, tải ảnh lên (giao diện mobile), bấm dự đoán khi chưa có ảnh.

## 3. Cập nhật bản demo

Sửa code → commit → push là Railway tự build và deploy lại (không có downtime: bản cũ chạy đến khi
bản mới qua healthcheck):

```bash
git add -A && git commit -m "..." && git push
```

- **Thay trọng số:** `python scripts/convert_weights.py --src <thư mục chứa .pth>` → commit → push.
- **Thêm ảnh mẫu:** chép ảnh `.jpg/.png` (có quyền chia sẻ) vào `examples/` → commit → push.
- **Đổi biến môi trường** (`ENABLED_MODELS`, `TOP_K`, ...): thêm vào `env` trong `.railway/railway.ts`
  rồi chạy `railway config apply` (xem mục 4), hoặc sửa trong Dashboard → service `web` → Variables.

## 4. Quản lý hạ tầng bằng CLI

Cài một lần: `npm i -g @railway/cli`, `npm install` (trong repo), `railway login`, `railway link`
(chọn project `medicinal-plant`, service `web`).

| Việc | Lệnh |
|---|---|
| Xem trạng thái | `railway status` |
| Log build / log chạy | `railway logs --build` / `railway logs` |
| Danh sách deployment | `railway deployment list` |
| Xem trước thay đổi cấu hình | `railway config plan` |
| Áp dụng cấu hình | `railway config apply` |
| Deploy lại bản hiện tại | `railway redeploy` |
| Mở dashboard | `railway open` |

> **Windows + Git Bash:** `railway config plan/apply` báo nhầm "requires Railway CLI 5.42.1 or newer"
> vì SDK đọc biến `$_` của Git Bash. Hãy chạy trong PowerShell:
> `$env:_ = "$(npm root -g)\@railway\cli\bin\railway.exe"; railway config plan`

> **Region:** IaC engine hiện chưa nhận ra thay đổi region khi số replica không đổi, nên region được đặt
> thêm bằng API (`serviceInstanceUpdate.multiRegionConfig`). Nếu muốn đổi region, hãy đổi trong
> Dashboard → service `web` → Settings → Regions, rồi sửa `replicas` trong `.railway/railway.ts` cho khớp.

## 5. Xử lý sự cố

| Triệu chứng | Nguyên nhân | Cách xử lý |
|---|---|---|
| Build lỗi `HTTP Error 404` / `sha256 không khớp` khi tải trọng số | File LFS chưa push, hoặc hết hạn mức băng thông Git LFS của GitHub (mỗi lần build tải ~400MB, Docker cache giúp bỏ qua khi `manifest.json` không đổi) | Tạo GitHub Release tag `weights-v1`, upload 3 file `.safetensors`, đặt biến `WEIGHTS_BASE_URL=https://github.com/VanTrietTRAN/MedicinalPlant/releases/download/weights-v1`, rồi `railway redeploy` |
| Container bị kill, exit 137 | Thiếu RAM | Đặt `ENABLED_MODELS=convnext,swin` (~1,1GB, vẫn 93,21%) |
| Healthcheck thất bại | Chưa nạp xong mô hình trong 300s, hoặc lỗi khởi động | `railway logs`; tăng `healthcheckTimeout` trong `.railway/railway.ts` |
| 502 khi mở domain | Cổng domain ≠ cổng app | Domain phải trỏ 8080 và biến `PORT=8080` (đã cố định trong IaC) |
| "Không tải được ảnh từ link" | Link không phải ảnh trực tiếp, trang chặn bot, ảnh > 15MB | Dùng link ảnh trực tiếp hoặc tải ảnh lên |

## 6. Chi phí

Gói Pro tính theo RAM/CPU dùng thực tế; service chạy liên tục với ~1,4GB RAM. Ngoài thời gian demo có thể:
- Bật **App Sleeping** (đặt `sleepApplication: true` trong `deploy` của `.railway/railway.ts` → `railway config apply`):
  service ngủ khi không có truy cập; lần mở đầu tiên phải chờ container khởi động lại (vài giây đến ~1 phút).
- Hoặc tạm gỡ bằng `railway down`, khi cần thì `railway redeploy`.

## 7. Checklist trước buổi demo

- [x] Build và deploy xanh, domain mở được
- [x] Dự đoán qua link ảnh và tải ảnh lên trên bản deploy
- [ ] Thử 3–5 ảnh thật của các loài trong `data/class_names.json`
- [ ] Thử trên điện thoại (giao diện mobile, chụp ảnh bằng camera)
- [ ] Nếu đã bật App Sleeping: tắt đi hoặc mở trang trước buổi demo 1–2 phút
