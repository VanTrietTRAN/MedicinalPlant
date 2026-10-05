FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    GRADIO_ANALYTICS_ENABLED=False \
    HF_HUB_DISABLE_TELEMETRY=1

WORKDIR /app
RUN useradd --create-home --uid 1000 appuser

# 1) PyTorch bản CPU (~200MB thay vì ~2.5GB bản CUDA) — layer ít thay đổi, cache lâu
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.13.0 torchvision==0.28.0

# 2) Thư viện còn lại
COPY requirements.txt .
RUN pip install -r requirements.txt

# 3) Trọng số (~400MB): chỉ tải lại khi manifest.json đổi.
#    Railway tự truyền biến môi trường cùng tên vào ARG lúc build.
ARG WEIGHTS_BASE_URL=https://media.githubusercontent.com/media/VanTrietTRAN/MedicinalPlant/main/weights
COPY scripts/download_weights.py scripts/
COPY weights/manifest.json weights/
RUN python scripts/download_weights.py

# 4) Mã nguồn app (weights/*.safetensors đã bị loại trong .dockerignore để không ghi đè)
COPY . .

USER appuser
EXPOSE 7860
CMD ["python", "app.py"]
