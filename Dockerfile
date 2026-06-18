FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y ffmpeg wget && rm -rf /var/lib/apt/lists/*

# torchvision 0.16.2 es compatible con basicsr (evita el bug de functional_tensor de >=0.17)
RUN pip install --no-cache-dir \
      runpod boto3 opencv-python-headless numpy \
      torchvision==0.16.2 basicsr realesrgan

# Pesos horneados (no se descargan en runtime, no hay network volume)
RUN mkdir -p /models && \
    wget -q -O /models/RealESRGAN_x4plus.pth \
      https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth && \
    wget -q -O /models/RealESRGAN_x2plus.pth \
      https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth

COPY handler.py /handler.py
CMD ["python", "-u", "/handler.py"]
