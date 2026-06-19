FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y ffmpeg wget && rm -rf /var/lib/apt/lists/*

# NO reinstalar torch/torchvision: usa los del base (par compatible).
# numpy<2 es la clave: numpy 2 rompe torch.from_numpy ("Numpy is not available").
RUN pip install --no-cache-dir \
      runpod boto3 opencv-python-headless "numpy<2" \
      basicsr realesrgan

# basicsr importa un modulo de torchvision que se removio en 0.17 -> parchear.
RUN python - <<'EOF'
import pathlib
p = pathlib.Path("/usr/local/lib/python3.10/dist-packages/basicsr/data/degradations.py")
if p.exists():
    s = p.read_text()
    s = s.replace("from torchvision.transforms.functional_tensor import rgb_to_grayscale",
                  "from torchvision.transforms.functional import rgb_to_grayscale")
    p.write_text(s)
    print("patched degradations.py")
EOF

RUN mkdir -p /models && \
    wget -q -O /models/RealESRGAN_x4plus.pth https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth && \
    wget -q -O /models/RealESRGAN_x2plus.pth https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth

COPY handler.py /handler.py
CMD ["python", "-u", "/handler.py"]
