import time
import os
import importlib
from collections.abc import Callable
from typing import Any

from database.db import (
    enqueue_task,
    get_task,
    init_db,
    list_tasks,
    mark_uploaded,
    task_settings,
    update_task_result,
    update_task_status,
    save_image_record,
)
from generator.image_generator import generate_image
from scraper.prompt_provider import convert_to_natural_prompt, get_random_prompt
from uploader.deviantart_upload import upload_image
from upscaler.image_upscaler import upscale_image
from generator import image_generator as image_generator_module
from upscaler import image_upscaler as image_upscaler_module
from uploader import deviantart_upload as deviantart_upload_module
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

    def _normalize_prompt(self, prompt: str, prompt_mode: str) -> str:
        if prompt_mode == "reformat":
            normalized = convert_to_natural_prompt(prompt)
            return normalized or prompt
        return prompt

    def _apply_task_settings(self, settings: dict[str, Any]) -> None:
        if not settings:
            return
        for key, value in settings.items():
            if value is None:
                continue
            os.environ[str(key)] = str(value)

    def enqueue_manual_task(
        self,
        *,
        prompt: str,
        prompt_mode: str,
        pipeline_mode: str,
        settings: dict[str, Any] | None = None,
        source_image_path: str | None = None,
        source_upscaled_path: str | None = None,
    ) -> int:
        if not prompt.strip():
            raise RuntimeError("Manual prompt cannot be empty")
        return enqueue_task(
            prompt=prompt.strip(),
            prompt_mode=prompt_mode,
            pipeline_mode=pipeline_mode,
            settings=settings,
            source_image_path=source_image_path,
            source_upscaled_path=source_upscaled_path,
        )

    def get_queue(self, limit: int = 100, status: str | None = None) -> list[dict]:
        return list_tasks(limit=limit, status=status)

    def process_task(self, task_id: int) -> dict:
        task = get_task(task_id)
        if task is None:
            raise RuntimeError(f"Task {task_id} not found")

        update_task_status(task_id, "running")
        try:
            result = self._execute_task(task)
            update_task_status(task_id, "success")
            return {"task_id": task_id, "status": "success", **result}
        except Exception as exc:
            update_task_status(task_id, "failure", error_message=str(exc))
            logger.exception(f"Task {task_id} failed: {exc}")
            return {"task_id": task_id, "status": "failure", "error": str(exc)}

    def process_next_queued_task(self) -> dict:
        queued = self.get_queue(limit=1, status="queued")
        if not queued:
            return {"status": "no_info", "message": "No queued tasks available"}
        task_id = int(queued[0]["id"])
        return self.process_task(task_id)

    def _execute_task(self, task: dict) -> dict:
        settings = task_settings(task)
        self._apply_task_settings(settings)
        importlib.reload(image_generator_module)
        importlib.reload(image_upscaler_module)
        importlib.reload(deviantart_upload_module)

        prompt_mode = (task.get("prompt_mode") or "as_is").strip().lower()
        pipeline_mode = (task.get("pipeline_mode") or "full").strip().lower()
        prompt = self._normalize_prompt(str(task.get("prompt") or ""), prompt_mode)

        image_path = str(task.get("source_image_path") or "").strip() or None
        upscaled_path = str(task.get("source_upscaled_path") or "").strip() or None
        upload_payload: dict | None = None

        if pipeline_mode in {"generate_only", "generate_upscale", "full"}:
            image_path = self._retry("image generation", image_generator_module.generate_image, prompt)
            update_task_result(task_id=int(task["id"]), image_path=image_path)

        if pipeline_mode in {"upscale_only", "generate_upscale", "upscale_upload", "full"}:
            if not image_path:
                raise RuntimeError("Upscale step requires an image path")
            upscaled_path = self._retry("image upscale", image_upscaler_module.upscale_image, image_path)
            update_task_result(task_id=int(task["id"]), upscaled_path=upscaled_path)

        if pipeline_mode in {"upload_only", "upscale_upload", "full"}:
            upload_source = upscaled_path or image_path
            if not upload_source:
                raise RuntimeError("Upload step requires an image path")
            upload_payload = self._retry("image upload", deviantart_upload_module.upload_image, upload_source, prompt)
            update_task_result(task_id=int(task["id"]), upload_payload=upload_payload)

        uploaded = bool(upload_payload)
        record_image_path = image_path or upscaled_path
        if not record_image_path:
            raise RuntimeError("Task completed without a source image path")
        record_id = save_image_record(
            prompt=prompt,
            image_path=record_image_path,
            upscaled_path=upscaled_path,
            uploaded=uploaded,
        )
        if uploaded:
            mark_uploaded(record_id, uploaded=True, upscaled_path=upscaled_path)

        result = {
            "id": record_id,
            "prompt": prompt,
            "image_path": image_path,
            "upscaled_path": upscaled_path,
            "uploaded": uploaded,
            "upload": upload_payload,
        }
        logger.info(f"Task {task['id']} completed and saved as image record {record_id}")
        return result

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