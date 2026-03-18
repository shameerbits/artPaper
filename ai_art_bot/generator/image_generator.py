import base64
import os
from datetime import datetime, UTC
from io import BytesIO
from pathlib import Path

from openai import OpenAI
from PIL import Image

from utils.config import GENERATED_DIR, get_settings
from utils.logger import logger


OPENAI_IMAGE_SIZE = os.getenv("OPENAI_IMAGE_SIZE", "1024x1536")
WALLPAPER_WIDTH = int(os.getenv("WALLPAPER_WIDTH", "1080"))
WALLPAPER_HEIGHT = int(os.getenv("WALLPAPER_HEIGHT", "1920"))


def _to_mobile_wallpaper(image_bytes: bytes, width: int = WALLPAPER_WIDTH, height: int = WALLPAPER_HEIGHT) -> bytes:
    """Center-crop and resize to an exact mobile wallpaper aspect ratio (default 9:16)."""
    target_ratio = width / height

    with Image.open(BytesIO(image_bytes)) as image:
        image = image.convert("RGB")
        src_w, src_h = image.size
        src_ratio = src_w / src_h

        if src_ratio > target_ratio:
            # Source is wider than target: crop left/right.
            crop_w = int(src_h * target_ratio)
            left = (src_w - crop_w) // 2
            box = (left, 0, left + crop_w, src_h)
        else:
            # Source is taller than target: crop top/bottom.
            crop_h = int(src_w / target_ratio)
            top = (src_h - crop_h) // 2
            box = (0, top, src_w, top + crop_h)

        wallpaper = image.crop(box).resize((width, height), Image.Resampling.LANCZOS)
        output = BytesIO()
        wallpaper.save(output, format="PNG")
        return output.getvalue()


def generate_image(prompt: str) -> str:
    client = OpenAI(api_key=get_settings().openai_api_key)
    response = client.images.generate(
        model="gpt-image-1",
        prompt=prompt,
        size=OPENAI_IMAGE_SIZE,
    )
    image_base64 = response.data[0].b64_json
    image_bytes = base64.b64decode(image_base64)
    wallpaper_bytes = _to_mobile_wallpaper(image_bytes)
    filename = f"generated_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.png"
    output_path = Path(GENERATED_DIR) / filename
    output_path.write_bytes(wallpaper_bytes)
    logger.info(f"Generated image saved to {output_path}")
    return str(output_path)