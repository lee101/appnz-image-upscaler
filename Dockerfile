FROM python:3.11-slim
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 curl && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir "numpy<2" opencv-python-headless==4.10.0.84 basicsr==1.4.2 realesrgan==0.3.0 gfpgan==1.3.8 fastapi==0.115.6 uvicorn==0.34.0
# basicsr 1.4.2 imports torchvision.transforms.functional_tensor, removed in torchvision 0.16
RUN sed -i 's/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms.functional import rgb_to_grayscale/' \
    /usr/local/lib/python3.11/site-packages/basicsr/data/degradations.py
RUN mkdir -p /models /app/gfpgan/weights && \
    curl -fsSL -o /models/RealESRGAN_x4plus.pth https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth && \
    curl -fsSL -o /models/GFPGANv1.4.pth https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth && \
    curl -fsSL -o /app/gfpgan/weights/detection_Resnet50_Final.pth https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth && \
    curl -fsSL -o /app/gfpgan/weights/parsing_parsenet.pth https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth
WORKDIR /app
COPY server.py /app/server.py
ENV PORT=5000 PYTHONUNBUFFERED=1 MODEL_DIR=/models
EXPOSE 5000
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD curl -fsS "http://localhost:${PORT:-5000}/healthz" || exit 1
CMD ["python", "server.py"]
