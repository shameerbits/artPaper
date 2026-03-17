import base64
from datetime import datetime, UTC
from pathlib import Path

from openai import OpenAI

from utils.config import GENERATED_DIR, get_settings
from utils.logger import logger


def generate_image(prompt: str) -> str:
    client = OpenAI(api_key=get_settings().openai_api_key)
    response = client.images.generate(
        model="gpt-image-1",
        prompt=prompt,
        size="1024x1792",
    )
    image_base64 = response.data[0].b64_json
    image_bytes = base64.b64decode(image_base64)
    filename = f"generated_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.png"
    output_path = Path(GENERATED_DIR) / filename
    output_path.write_bytes(image_bytes)
    logger.info(f"Generated image saved to {output_path}")
    return str(output_path)