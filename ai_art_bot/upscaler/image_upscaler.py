import os
from math import sqrt
from datetime import datetime, timezone
from pathlib import Path

import replicate
import requests

from upscaler.accelerated_upscaler import upscale_with_directml
from utils.config import UPSCALED_DIR, WEIGHTS_DIR, get_settings
from utils.logger import logger


UPSCALER_BACKEND = os.getenv("UPSCALER_BACKEND", "realesrgan").strip().lower()
REALESRGAN_MODEL_NAME = os.getenv("REALESRGAN_MODEL_NAME", "RealESRGAN_x4plus")
REALESRGAN_OUTSCALE = int(os.getenv("REALESRGAN_OUTSCALE", "4"))
REALESRGAN_WEIGHTS_DIR = Path(os.getenv("REALESRGAN_WEIGHTS_DIR", str(WEIGHTS_DIR)))
REALESRGAN_TILE = max(int(os.getenv("REALESRGAN_TILE", "256")), 0)
REALESRGAN_TILE_PAD = max(int(os.getenv("REALESRGAN_TILE_PAD", "10")), 0)
REALESRGAN_PRE_PAD = max(int(os.getenv("REALESRGAN_PRE_PAD", "0")), 0)
REALESRGAN_MAX_INPUT_SIDE = max(int(os.getenv("REALESRGAN_MAX_INPUT_SIDE", "0")), 0)
REALESRGAN_MAX_INPUT_PIXELS = max(int(os.getenv("REALESRGAN_MAX_INPUT_PIXELS", "0")), 0)
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


def _is_memory_error(exc: Exception) -> bool:
    message = str(exc).lower()
    memory_markers = (
        "defaultcpuallocator",
        "not enough memory",
        "out of memory",
    )
    return any(marker in message for marker in memory_markers)


def _iter_tile_candidates(base_tile: int) -> list[int]:
    candidates: list[int] = [base_tile]
    for tile in (512, 256, 192, 128, 96, 64):
        if tile not in candidates:
            candidates.append(tile)
    return candidates


def _downscale_if_needed(image, cv2):
    if image is None:
        return image

    height, width = image.shape[:2]
    scale = 1.0

    if REALESRGAN_MAX_INPUT_SIDE > 0:
        longest_side = max(height, width)
        if longest_side > REALESRGAN_MAX_INPUT_SIDE:
            scale = min(scale, REALESRGAN_MAX_INPUT_SIDE / float(longest_side))

    if REALESRGAN_MAX_INPUT_PIXELS > 0:
        pixels = height * width
        if pixels > REALESRGAN_MAX_INPUT_PIXELS:
            scale = min(scale, sqrt(REALESRGAN_MAX_INPUT_PIXELS / float(pixels)))

    if scale >= 1.0:
        return image

    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))
    logger.warning(
        f"Downscaling input before upscaling to avoid OOM: {width}x{height} -> {new_width}x{new_height}"
    )
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)


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
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read image for upscaling: {image_path}")
    image = _downscale_if_needed(image, cv2)

    tile_candidates = _iter_tile_candidates(REALESRGAN_TILE)
    last_error: Exception | None = None
    output = None

    for tile in tile_candidates:
        try:
            logger.info(
                f"Running Real-ESRGAN with tile={tile}, tile_pad={REALESRGAN_TILE_PAD}, pre_pad={REALESRGAN_PRE_PAD}"
            )
            upsampler = RealESRGANer(
                scale=netscale,
                model_path=str(weights_path),
                model=model,
                tile=tile,
                tile_pad=REALESRGAN_TILE_PAD,
                pre_pad=REALESRGAN_PRE_PAD,
                half=False,
            )
            output, _ = upsampler.enhance(image, outscale=REALESRGAN_OUTSCALE)
            break
        except Exception as exc:
            last_error = exc
            if _is_memory_error(exc) and tile != tile_candidates[-1]:
                logger.warning(
                    f"Upscaling failed due to memory pressure with tile={tile}; retrying with smaller tile. Error: {exc}"
                )
                continue
            raise

    if output is None:
        raise RuntimeError(f"Upscaling failed after trying multiple tile sizes. Last error: {last_error}")

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
    if UPSCALER_BACKEND == "directml":
        return upscale_with_directml(image_path)
    if UPSCALER_BACKEND == "openvino":
        logger.warning("UPSCALER_BACKEND=openvino is deprecated; using directml backend")
        return upscale_with_directml(image_path)
    raise RuntimeError(
        "Unsupported UPSCALER_BACKEND "
        f"'{UPSCALER_BACKEND}'. Use 'realesrgan', 'realesrgan_local', 'replicate', or 'directml'."
    )