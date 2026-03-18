import os
from datetime import datetime, UTC
from pathlib import Path

import replicate
import requests

from utils.config import UPSCALED_DIR, get_settings
from utils.logger import logger


UPSCALER_BACKEND = os.getenv("UPSCALER_BACKEND", "realesrgan").strip().lower()
REALESRGAN_REPLICATE_MODEL = os.getenv("REALESRGAN_REPLICATE_MODEL", "nightmareai/real-esrgan")
REPLICATE_UPSCALER_MODEL = os.getenv("REPLICATE_UPSCALER_MODEL", "nightmareai/real-esrgan")
REPLICATE_UPSCALER_SCALE = int(os.getenv("REPLICATE_UPSCALER_SCALE", "4"))


def _download_to_output(output_url: str) -> str:
    response = requests.get(output_url, timeout=120)
    response.raise_for_status()
    filename = f"upscaled_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.png"
    output_path = Path(UPSCALED_DIR) / filename
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


def _upscale_with_realesrgan(image_path: str) -> str:
    logger.info(f"Upscaling with Real-ESRGAN model: {REALESRGAN_REPLICATE_MODEL}")
    return _run_replicate_model(image_path, REALESRGAN_REPLICATE_MODEL)


def _upscale_with_replicate(image_path: str) -> str:
    logger.info(f"Upscaling with Replicate model: {REPLICATE_UPSCALER_MODEL}")
    return _run_replicate_model(image_path, REPLICATE_UPSCALER_MODEL)


def upscale_image(image_path: str) -> str:
    if UPSCALER_BACKEND == "realesrgan":
        return _upscale_with_realesrgan(image_path)
    if UPSCALER_BACKEND == "replicate":
        return _upscale_with_replicate(image_path)
    raise RuntimeError(
        f"Unsupported UPSCALER_BACKEND '{UPSCALER_BACKEND}'. Use 'realesrgan' or 'replicate'."
    )