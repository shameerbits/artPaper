from datetime import datetime, UTC
from pathlib import Path

import replicate
import requests

from utils.config import UPSCALED_DIR, get_settings
from utils.logger import logger


def upscale_image(image_path: str) -> str:
    client = replicate.Client(api_token=get_settings().replicate_api_token)
    with open(image_path, "rb") as image_file:
        result = client.run(
            "nightmareai/real-esrgan",
            input={"image": image_file, "scale": 4},
        )
    output_url = result[0] if isinstance(result, list) else result
    response = requests.get(output_url, timeout=120)
    response.raise_for_status()
    filename = f"upscaled_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.png"
    output_path = Path(UPSCALED_DIR) / filename
    output_path.write_bytes(response.content)
    logger.info(f"Upscaled image saved to {output_path}")
    return str(output_path)