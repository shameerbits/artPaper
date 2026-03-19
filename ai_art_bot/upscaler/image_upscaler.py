import os
from datetime import datetime, timezone
from pathlib import Path

import replicate
import requests

from utils.config import UPSCALED_DIR, get_settings
from utils.logger import logger


UPSCALER_BACKEND = os.getenv("UPSCALER_BACKEND", "realesrgan").strip().lower()
REALESRGAN_MODEL_NAME = os.getenv("REALESRGAN_MODEL_NAME", "RealESRGAN_x4plus")
REALESRGAN_OUTSCALE = int(os.getenv("REALESRGAN_OUTSCALE", "4"))
REALESRGAN_WEIGHTS_DIR = Path(os.getenv("REALESRGAN_WEIGHTS_DIR", str(Path(UPSCALED_DIR).parent / "weights")))
REPLICATE_UPSCALER_MODEL = os.getenv("REPLICATE_UPSCALER_MODEL", "nightmareai/real-esrgan")
REPLICATE_UPSCALER_SCALE = int(os.getenv("REPLICATE_UPSCALER_SCALE", "4"))

REALESRGAN_MODEL_CONFIG = {
    "RealESRGAN_x4plus": {
        "weights": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "netscale": 4,
        "num_block": 23,
    },
    "RealESRNet_x4plus": {
        "weights": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRNet_x4plus.pth",
        "netscale": 4,
        "num_block": 23,
    },
    "RealESRGAN_x4plus_anime_6B": {
        "weights": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        "netscale": 4,
        "num_block": 6,
    },
}


def _build_output_path() -> Path:
    filename = f"upscaled_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.png"
    return Path(UPSCALED_DIR) / filename


def _download_to_output(output_url: str) -> str:
    response = requests.get(output_url, timeout=120)
    response.raise_for_status()
    output_path = _build_output_path()
    output_path.write_bytes(response.content)
    logger.info(f"Upscaled image saved to {output_path}")
    return str(output_path)


def _run_replicate_model(image_path: str, model: str) -> str:
    client = replicate.Client(api_token=get_settings().replicate_api_token)
    with open(image_path, "rb") as image_file:
        result = client.run(
            model,
            input={"image": image_file, "scale": REPLICATE_UPSCALER_SCALE},
        )
    output_url = result[0] if isinstance(result, list) else result
    return _download_to_output(output_url)


def _ensure_realesrgan_weights(model_name: str) -> tuple[Path, int, int]:
    config = REALESRGAN_MODEL_CONFIG.get(model_name)
    if config is None:
        supported = ", ".join(sorted(REALESRGAN_MODEL_CONFIG))
        raise RuntimeError(f"Unsupported REALESRGAN_MODEL_NAME '{model_name}'. Supported: {supported}")

    REALESRGAN_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    weights_path = REALESRGAN_WEIGHTS_DIR / f"{model_name}.pth"
    if not weights_path.exists():
        logger.info(f"Downloading Real-ESRGAN weights for {model_name} to {weights_path}")
        response = requests.get(config["weights"], timeout=300)
        response.raise_for_status()
        weights_path.write_bytes(response.content)

    return weights_path, int(config["netscale"]), int(config["num_block"])


def _upscale_with_realesrgan(image_path: str) -> str:
    try:
        import cv2
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
    except Exception as exc:
        raise RuntimeError(
            "Local Real-ESRGAN dependencies are missing. Install with: "
            "pip install realesrgan opencv-python-headless"
        ) from exc

    weights_path, netscale, num_block = _ensure_realesrgan_weights(REALESRGAN_MODEL_NAME)
    logger.info(f"Upscaling with local Real-ESRGAN model: {REALESRGAN_MODEL_NAME}")

    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=num_block,
        num_grow_ch=32,
        scale=netscale,
    )
    upsampler = RealESRGANer(
        scale=netscale,
        model_path=str(weights_path),
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=False,
    )
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read image for upscaling: {image_path}")

    output, _ = upsampler.enhance(image, outscale=REALESRGAN_OUTSCALE)
    output_path = _build_output_path()
    if not cv2.imwrite(str(output_path), output):
        raise RuntimeError(f"Failed to write upscaled image: {output_path}")

    logger.info(f"Upscaled image saved to {output_path}")
    return str(output_path)


def _upscale_with_replicate(image_path: str) -> str:
    logger.info(f"Upscaling with Replicate model: {REPLICATE_UPSCALER_MODEL}")
    return _run_replicate_model(image_path, REPLICATE_UPSCALER_MODEL)


def upscale_image(image_path: str) -> str:
    if UPSCALER_BACKEND == "realesrgan":
        return _upscale_with_realesrgan(image_path)
    if UPSCALER_BACKEND == "realesrgan_local":
        return _upscale_with_realesrgan(image_path)
    if UPSCALER_BACKEND == "replicate":
        return _upscale_with_replicate(image_path)
    raise RuntimeError(
        f"Unsupported UPSCALER_BACKEND '{UPSCALER_BACKEND}'. Use 'realesrgan', 'realesrgan_local', or 'replicate'."
    )