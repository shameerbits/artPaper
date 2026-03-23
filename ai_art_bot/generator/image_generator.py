import base64
import os
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

from openai import OpenAI
from PIL import Image

from utils.config import GENERATED_DIR, MODELS_DIR, get_settings
from utils.logger import logger


OPENAI_IMAGE_SIZE = os.getenv("OPENAI_IMAGE_SIZE", "1024x1536")
WALLPAPER_WIDTH = int(os.getenv("WALLPAPER_WIDTH", "1080"))
WALLPAPER_HEIGHT = int(os.getenv("WALLPAPER_HEIGHT", "1920"))
LOCAL_IMAGE_WIDTH = int(os.getenv("LOCAL_IMAGE_WIDTH", str(WALLPAPER_WIDTH)))
LOCAL_IMAGE_HEIGHT = int(os.getenv("LOCAL_IMAGE_HEIGHT", str(WALLPAPER_HEIGHT)))
LOCAL_NUM_INFERENCE_STEPS = int(os.getenv("LOCAL_NUM_INFERENCE_STEPS", "24"))
LOCAL_GUIDANCE_SCALE = float(os.getenv("LOCAL_GUIDANCE_SCALE", "7.0"))
LOCAL_SEED = int(os.getenv("LOCAL_SEED", "-1"))
LOCAL_DIRECTML_FALLBACK_TO_CPU = (
    os.getenv("LOCAL_DIRECTML_FALLBACK_TO_CPU", "true").strip().lower() in {"1", "true", "yes"}
)
LOCAL_MODEL_USE_DIRECTML = (
    os.getenv("LOCAL_MODEL_USE_DIRECTML", os.getenv("LOCAL_MODEL_USE_OPENVINO", "false"))
    .strip()
    .lower()
    in {"1", "true", "yes"}
)
DIRECTML_DEVICE_ID = max(int(os.getenv("DIRECTML_DEVICE_ID", "0")), 0)

_LOCAL_PIPELINE: Any | None = None
_LOCAL_PIPELINE_SOURCE: str | None = None
_LOCAL_PIPELINE_USE_DIRECTML: bool | None = None


def _cast_scheduler_tensors_to_float32(scheduler: Any) -> None:
    """Best-effort cast of known scheduler runtime tensors to float32."""
    try:
        import torch
    except Exception:
        return

    tensor_attrs = (
        "timesteps",
        "sigmas",
        "init_noise_sigma",
        "alphas_cumprod",
        "sqrt_alphas_cumprod",
        "sqrt_one_minus_alphas_cumprod",
        "sigmas_up",
        "sigmas_down",
        "lambda_t",
    )

    for name in tensor_attrs:
        value = getattr(scheduler, name, None)
        if isinstance(value, torch.Tensor):
            setattr(scheduler, name, value.to(dtype=torch.float32))
            continue

        if isinstance(value, list):
            converted = [item.to(dtype=torch.float32) if isinstance(item, torch.Tensor) else item for item in value]
            setattr(scheduler, name, converted)
            continue

        if isinstance(value, tuple):
            converted = tuple(item.to(dtype=torch.float32) if isinstance(item, torch.Tensor) else item for item in value)
            setattr(scheduler, name, converted)


def _patch_scheduler_for_directml_float32(pipeline: Any) -> None:
    """Ensure scheduler-generated runtime tensors stay float32 on DirectML."""
    scheduler = getattr(pipeline, "scheduler", None)
    if scheduler is None or getattr(scheduler, "_dml_float32_patched", False):
        return

    original_set_timesteps = getattr(scheduler, "set_timesteps", None)
    if callable(original_set_timesteps):
        def _set_timesteps_float32(*args: Any, **kwargs: Any) -> Any:
            result = original_set_timesteps(*args, **kwargs)
            _cast_scheduler_tensors_to_float32(scheduler)
            return result

        scheduler.set_timesteps = _set_timesteps_float32  # type: ignore[method-assign]

    _cast_scheduler_tensors_to_float32(scheduler)
    setattr(scheduler, "_dml_float32_patched", True)


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


def _save_image_as_wallpaper(image: Image.Image) -> str:
    output = BytesIO()
    image.convert("RGB").save(output, format="PNG")
    wallpaper_bytes = _to_mobile_wallpaper(output.getvalue())
    filename = f"generated_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.png"
    output_path = Path(GENERATED_DIR) / filename
    output_path.write_bytes(wallpaper_bytes)
    logger.info(f"Generated image saved to {output_path}")
    return str(output_path)


def _resolve_model_source() -> str:
    settings = get_settings()
    local_model_path = settings.local_model_path.strip()
    if local_model_path:
        path = Path(local_model_path).expanduser()
        if path.exists():
            return str(path)
        logger.warning(
            f"LOCAL_MODEL_PATH does not exist: {local_model_path}. Falling back to LOCAL_MODEL_ID."
        )
    return settings.local_model_id.strip()


def _load_local_pipeline(model_source: str, use_directml: bool) -> Any:
    global _LOCAL_PIPELINE, _LOCAL_PIPELINE_SOURCE, _LOCAL_PIPELINE_USE_DIRECTML
    if (
        _LOCAL_PIPELINE is not None
        and _LOCAL_PIPELINE_SOURCE == model_source
        and _LOCAL_PIPELINE_USE_DIRECTML == use_directml
    ):
        return _LOCAL_PIPELINE

    try:
        import torch
        from diffusers import AutoPipelineForText2Image, DPMSolverMultistepScheduler
    except Exception as exc:
        raise RuntimeError(
            "Local SD generation requires `diffusers`, `transformers`, `accelerate`, and `safetensors`. "
            "Install optional local generation dependencies first."
        ) from exc

    pipeline = AutoPipelineForText2Image.from_pretrained(
        model_source,
        torch_dtype=torch.float32,
        safety_checker=None,
    )
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)

    if use_directml:
        try:
            import torch_directml
        except Exception as exc:
            raise RuntimeError(
                "DirectML local generation requires `torch-directml`. "
                "Install with: pip install torch-directml"
            ) from exc

        dml_device = torch_directml.device(DIRECTML_DEVICE_ID)
        pipeline = pipeline.to(dml_device)
        _patch_scheduler_for_directml_float32(pipeline)
    else:
        pipeline = pipeline.to("cpu")

    _LOCAL_PIPELINE = pipeline
    _LOCAL_PIPELINE_SOURCE = model_source
    _LOCAL_PIPELINE_USE_DIRECTML = use_directml
    logger.info(f"Loaded local model source={model_source} backend={'directml' if use_directml else 'diffusers'}")
    return pipeline


def _generate_image_openai(prompt: str) -> str:
    client = OpenAI(api_key=get_settings().openai_api_key)
    response = client.images.generate(
        model="gpt-image-1",
        prompt=prompt,
        size=OPENAI_IMAGE_SIZE,
    )
    image_base64 = response.data[0].b64_json
    image_bytes = base64.b64decode(image_base64)
    wallpaper_bytes = _to_mobile_wallpaper(image_bytes)
    filename = f"generated_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.png"
    output_path = Path(GENERATED_DIR) / filename
    output_path.write_bytes(wallpaper_bytes)
    logger.info(f"Generated image saved to {output_path}")
    return str(output_path)


def _generate_image_local(prompt: str) -> str:
    model_source = _resolve_model_source()
    if not model_source:
        raise RuntimeError("No local model configured. Set LOCAL_MODEL_PATH or LOCAL_MODEL_ID.")

    pipeline = _load_local_pipeline(model_source=model_source, use_directml=LOCAL_MODEL_USE_DIRECTML)

    generation_kwargs: dict[str, Any] = {
        "prompt": prompt,
        "num_inference_steps": LOCAL_NUM_INFERENCE_STEPS,
        "guidance_scale": LOCAL_GUIDANCE_SCALE,
        "width": LOCAL_IMAGE_WIDTH,
        "height": LOCAL_IMAGE_HEIGHT,
    }

    if LOCAL_SEED >= 0 and not LOCAL_MODEL_USE_DIRECTML:
        try:
            import torch

            generation_kwargs["generator"] = torch.Generator(device="cpu").manual_seed(LOCAL_SEED)
        except Exception:
            logger.warning("Failed to set LOCAL_SEED generator; continuing without deterministic seed")
    elif LOCAL_SEED >= 0 and LOCAL_MODEL_USE_DIRECTML:
        logger.warning("LOCAL_SEED is ignored for DirectML local generation on this backend")

    if LOCAL_MODEL_USE_DIRECTML:
        try:
            import torch

            default_dtype = torch.get_default_dtype()
            torch.set_default_dtype(torch.float32)
            try:
                result = pipeline(**generation_kwargs)
            finally:
                torch.set_default_dtype(default_dtype)
        except Exception as exc:
            error_text = str(exc)
            if (
                LOCAL_DIRECTML_FALLBACK_TO_CPU
                and "does not support Double (Float64) operations" in error_text
            ):
                logger.warning(
                    "DirectML failed due to Float64 limitation. Falling back to CPU generation for this run."
                )
                cpu_pipeline = _load_local_pipeline(model_source=model_source, use_directml=False)
                result = cpu_pipeline(**generation_kwargs)
            else:
                raise
    else:
        result = pipeline(**generation_kwargs)

    image = result.images[0]
    return _save_image_as_wallpaper(image)


def download_local_model(model_id: str | None = None, local_dir: str | None = None) -> str:
    target_model = (model_id or get_settings().local_model_id).strip()
    if not target_model:
        raise RuntimeError("No model id provided. Pass --model-id or set LOCAL_MODEL_ID.")

    destination = Path(local_dir).expanduser().resolve() if local_dir else MODELS_DIR
    destination.mkdir(parents=True, exist_ok=True)
    model_slug = target_model.split("/")[-1]
    target_path = destination / model_slug

    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        raise RuntimeError("Model download requires `huggingface_hub`. Install optional local generation dependencies.") from exc

    logger.info(f"Downloading model {target_model} to {target_path}")
    snapshot_download(
        repo_id=target_model,
        local_dir=str(target_path),
        local_dir_use_symlinks=False,
    )
    logger.info(f"Model downloaded successfully: {target_path}")
    return str(target_path)


def generate_image(prompt: str) -> str:
    backend = get_settings().image_backend.strip().lower()
    if backend == "openai":
        return _generate_image_openai(prompt)
    if backend in {"local", "local_sd", "sd15"}:
        return _generate_image_local(prompt)
    raise RuntimeError(f"Unsupported IMAGE_BACKEND: {backend}")