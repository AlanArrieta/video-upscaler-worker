import os, subprocess, tempfile, shutil, uuid, glob
import cv2, torch, runpod, boto3
from botocore.config import Config
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

MODELS = "/models"
# destino -> (lado largo en px, escala del modelo a usar)
TARGETS = {"1080": (1920, 2), "2k": (2560, 2), "4k": (3840, 4)}

def r2():
    return boto3.client(
        "s3",
        endpoint_url=os.environ["BUCKET_ENDPOINT_URL"],
        aws_access_key_id=os.environ["BUCKET_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["BUCKET_SECRET_ACCESS_KEY"],
        config=Config(signature_version="s3v4"), region_name="auto",
    )
BUCKET = os.environ["BUCKET_NAME"]

def sh(cmd): subprocess.run(cmd, check=True)

def probe_fps(path):
    out = subprocess.check_output(
        ["ffprobe","-v","0","-select_streams","v:0","-show_entries",
         "stream=r_frame_rate","-of","default=nokey=1:noprint_wrappers=1", path]
    ).decode().strip()
    n, d = out.split("/"); return float(n)/float(d)

def make_upsampler(scale):
    block_scale = 4 if scale == 4 else 2
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                    num_block=23, num_grow_ch=32, scale=block_scale)
    path = f"{MODELS}/RealESRGAN_x{block_scale}plus.pth"
    return RealESRGANer(scale=block_scale, model_path=path, model=model,
                        tile=512, tile_pad=10, pre_pad=0, half=True,
                        device="cuda")   # tile=512 mantiene VRAM baja: hasta 4K cabe en 16 GB

def even(x): return int(round(x/2)*2)

def target_dims(w, h, long_target):
    f = long_target / max(w, h)
    return even(w*f), even(h*f)

def handler(event):
    inp = event["input"]
    video_url = inp["video_url"]
    target = str(inp.get("target", "1080")).lower()
    if target not in TARGETS:
        return {"error": f"target invalido: {target}"}
    long_target, scale = TARGETS[target]

    work = tempfile.mkdtemp()
    src = f"{work}/in.mp4"; frames = f"{work}/f"; up = f"{work}/u"; out = f"{work}/out.mp4"
    os.makedirs(frames); os.makedirs(up)
    try:
        # 1) bajar master desde R2
        sh(["wget","-q","-O",src, video_url])
        fps = probe_fps(src)

        # 2) extraer frames (PNG sin perdida)
        sh(["ffmpeg","-y","-loglevel","error","-i",src,
            "-qscale:v","1","-qmin","1",f"{frames}/%08d.png"])
        flist = sorted(glob.glob(f"{frames}/*.png"))
        if not flist: return {"error": "0 frames extraidos"}

        # 3) upscalear cada frame con Real-ESRGAN
        ups = make_upsampler(scale)
        h0, w0 = cv2.imread(flist[0]).shape[:2]
        for fp in flist:
            img = cv2.imread(fp, cv2.IMREAD_COLOR)
            o, _ = ups.enhance(img, outscale=scale)
            cv2.imwrite(f"{up}/{os.path.basename(fp)}", o)

        # 4) downscale lanczos al destino exacto + re-encode H.264
        nw, nh = target_dims(w0, h0, long_target)
        sh(["ffmpeg","-y","-loglevel","error","-framerate",str(fps),
            "-i",f"{up}/%08d.png",
            "-vf",f"scale={nw}:{nh}:flags=lanczos",
            "-c:v","libx264","-pix_fmt","yuv420p","-crf","16",
            "-r",str(fps), out])

        # 5) subir resultado a R2 y devolver URL firmada (24 h)
        key = f"upscaled/{target}_{uuid.uuid4().hex}.mp4"
        cli = r2()
        cli.upload_file(out, BUCKET, key, ExtraArgs={"ContentType":"video/mp4"})
        url = cli.generate_presigned_url(
            "get_object", Params={"Bucket":BUCKET,"Key":key}, ExpiresIn=86400)

        return {"video_url": url, "key": key, "target": target,
                "width": nw, "height": nh, "frames": len(flist), "fps": fps}
    finally:
        shutil.rmtree(work, ignore_errors=True)

runpod.serverless.start({"handler": handler})
