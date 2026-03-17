import time
from collections.abc import Callable
from typing import Any

from database.db import init_db, mark_uploaded, save_image_record
from generator.image_generator import generate_image
from scraper.midjourney_scraper import get_random_prompt
from uploader.deviantart_upload import upload_image
from upscaler.image_upscaler import upscale_image
from utils.logger import logger


class PipelineRunner:
    def __init__(self) -> None:
        init_db()

    def _retry(self, label: str, func: Callable[..., Any], *args: Any, attempts: int = 3, **kwargs: Any) -> Any:
        for attempt in range(1, attempts + 1):
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                logger.warning(f"{label} failed on attempt {attempt}/{attempts}: {exc}")
                if attempt == attempts:
                    raise
                time.sleep(min(5 * attempt, 15))
        raise RuntimeError(f"Retry loop ended unexpectedly for {label}")

    def run_once(self, mode: str = "full") -> dict:
        prompt = self._retry("prompt selection", get_random_prompt)
        image_path = self._retry("image generation", generate_image, prompt)

        if mode == "generate_only":
            record_id = save_image_record(prompt=prompt, image_path=image_path, upscaled_path=None, uploaded=False)
            return {"id": record_id, "prompt": prompt, "image_path": image_path, "uploaded": False}

        upscaled_path = self._retry("image upscale", upscale_image, image_path)
        record_id = save_image_record(
            prompt=prompt,
            image_path=image_path,
            upscaled_path=upscaled_path,
            uploaded=False,
        )
        upload_payload = self._retry("image upload", upload_image, upscaled_path, prompt)
        mark_uploaded(record_id, uploaded=True, upscaled_path=upscaled_path)
        result = {
            "id": record_id,
            "prompt": prompt,
            "image_path": image_path,
            "upscaled_path": upscaled_path,
            "uploaded": True,
            "upload": upload_payload,
        }
        logger.info(f"Pipeline completed for record {record_id}")
        return result

    def run_loop(self, interval_minutes: int = 60) -> None:
        logger.info(f"Starting loop with {interval_minutes}-minute interval")
        while True:
            try:
                self.run_once(mode="full")
            except Exception as exc:
                logger.exception(f"Pipeline run failed: {exc}")
            time.sleep(max(interval_minutes, 1) * 60)