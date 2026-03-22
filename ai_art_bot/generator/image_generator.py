import base64
import os
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

from openai import OpenAI
from PIL import Image

from utils.config import GENERATED_DIR, get_settings
from utils.logger import logger


OPENAI_IMAGE_SIZE = os.getenv("OPENAI_IMAGE_SIZE", "1024x1536")
WALLPAPER_WIDTH = int(os.getenv("WALLPAPER_WIDTH", "1080"))
WALLPAPER_HEIGHT = int(os.getenv("WALLPAPER_HEIGHT", "1920"))
LOCAL_IMAGE_WIDTH = int(os.getenv("LOCAL_IMAGE_WIDTH", str(WALLPAPER_WIDTH)))
LOCAL_IMAGE_HEIGHT = int(os.getenv("LOCAL_IMAGE_HEIGHT", str(WALLPAPER_HEIGHT)))
LOCAL_NUM_INFERENCE_STEPS = int(os.getenv("LOCAL_NUM_INFERENCE_STEPS", "24"))
LOCAL_GUIDANCE_SCALE = float(os.getenv("LOCAL_GUIDANCE_SCALE", "7.0"))
LOCAL_SEED = int(os.getenv("LOCAL_SEED", "-1"))
LOCAL_MODEL_USE_OPENVINO = os.getenv("LOCAL_MODEL_USE_OPENVINO", "false").strip().lower() in {"1", "true", "yes"}
OPENVINO_DEVICE = os.getenv("OPENVINO_DEVICE", "GPU")
OPENVINO_OOM_AUTO_RETRY = os.getenv("OPENVINO_OOM_AUTO_RETRY", "true").strip().lower() in {"1", "true", "yes"}
OPENVINO_GPU_FALLBACK_TO_CPU = os.getenv("OPENVINO_GPU_FALLBACK_TO_CPU", "true").strip().lower() in {"1", "true", "yes"}
OPENVINO_EXPORT_CACHE_DIR = Path(
    os.getenv("OPENVINO_EXPORT_CACHE_DIR", str(Path(GENERATED_DIR).parent / "models" / "openvino_cache"))
)

_LOCAL_PIPELINE: Any | None = None
_LOCAL_PIPELINE_SOURCE: str | None = None
_LOCAL_PIPELINE_USE_OPENVINO: bool | None = None


def _safe_model_slug(model_source: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in model_source)


def _is_openvino_export_dir(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    return any(path.rglob("openvino_model.xml"))


def _is_openvino_memory_error(exc: Exception) -> bool:
    message = str(exc).lower()
    markers = (
        "exceeded max size of memory object allocation",
        "exceed_allocatable_mem_size",
        "not enough memory",
        "out of memory",
    )
    return any(marker in message for marker in markers)


def _is_openvino_gpu_runtime_error(exc: Exception) -> bool:
    message = str(exc).lower()
    markers = (
        "clflush, error code: -5",
        "clwaitforevents, error code: -14",
        "ocl_stream.cpp",
        "intel_gpu",
    )
    return any(marker in message for marker in markers)


def _align_to_64(value: int) -> int:
    return max(64, (value // 64) * 64)


def _openvino_size_candidates(width: int, height: int) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = [(width, height)]
    for scale in (0.85, 0.75, 0.67, 0.5):
        resized = (_align_to_64(int(width * scale)), _align_to_64(int(height * scale)))
        if resized not in candidates:
            candidates.append(resized)
    return candidates


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
    return settings.local_model_path.strip() or settings.local_model_id.strip()


def _load_local_pipeline(model_source: str, use_openvino: bool) -> Any:
    global _LOCAL_PIPELINE, _LOCAL_PIPELINE_SOURCE, _LOCAL_PIPELINE_USE_OPENVINO
    if (
        _LOCAL_PIPELINE is not None
        and _LOCAL_PIPELINE_SOURCE == model_source
        and _LOCAL_PIPELINE_USE_OPENVINO == use_openvino
    ):
        return _LOCAL_PIPELINE

    if use_openvino:
        openvino_pipeline_cls = None
        try:
            from optimum.intel.openvino import OVDiffusionPipeline

            openvino_pipeline_cls = OVDiffusionPipeline
        except Exception:
            try:
                from optimum.intel.openvino import OVStableDiffusionPipeline

                openvino_pipeline_cls = OVStableDiffusionPipeline
            except Exception as exc:
                raise RuntimeError(
                    "OpenVINO local generation requires optimum-intel with OpenVINO diffusion pipelines. "
                    "Install compatible `optimum-intel[openvino]` and OpenVINO runtime packages."
                ) from exc

        if openvino_pipeline_cls is None:
            raise RuntimeError(
                "Failed to resolve an OpenVINO diffusion pipeline class from optimum-intel."
            )

        source_path = Path(model_source)
        cached_export_dir = OPENVINO_EXPORT_CACHE_DIR / _safe_model_slug(model_source)
        load_source = model_source
        export_model = True

        if source_path.exists():
            load_source = str(source_path)
            export_model = not _is_openvino_export_dir(source_path)
        elif _is_openvino_export_dir(cached_export_dir):
            load_source = str(cached_export_dir)
            export_model = False

        ov_kwargs: dict[str, Any] = {"export": export_model, "compile": False}
        if export_model:
            cached_export_dir.mkdir(parents=True, exist_ok=True)
            ov_kwargs["model_save_dir"] = str(cached_export_dir)

        try:
            pipeline = openvino_pipeline_cls.from_pretrained(load_source, **ov_kwargs)
        except TypeError:
            # Backward compatibility for older optimum-intel versions.
            ov_kwargs.pop("model_save_dir", None)
            pipeline = openvino_pipeline_cls.from_pretrained(load_source, **ov_kwargs)

        if export_model and not _is_openvino_export_dir(cached_export_dir):
            try:
                pipeline.save_pretrained(str(cached_export_dir))
                logger.info(f"Persisted OpenVINO export cache to {cached_export_dir}")
            except Exception as exc:
                logger.warning(f"Failed to persist OpenVINO export cache at {cached_export_dir}: {exc}")

        # If cache exists, always reload from cache to avoid tempfile-backed exports on Windows.
        if export_model and _is_openvino_export_dir(cached_export_dir):
            try:
                pipeline = openvino_pipeline_cls.from_pretrained(
                    str(cached_export_dir),
                    export=False,
                    compile=False,
                )
                logger.info(f"Reloaded OpenVINO pipeline from cache: {cached_export_dir}")
            except Exception as exc:
                logger.warning(f"Failed to reload OpenVINO pipeline from cache {cached_export_dir}: {exc}")

        pipeline.to(OPENVINO_DEVICE)
    else:
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
        pipeline = pipeline.to("cpu")

    _LOCAL_PIPELINE = pipeline
    _LOCAL_PIPELINE_SOURCE = model_source
    _LOCAL_PIPELINE_USE_OPENVINO = use_openvino
    logger.info(f"Loaded local model source={model_source} backend={'openvino' if use_openvino else 'diffusers'}")
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

    pipeline = _load_local_pipeline(model_source=model_source, use_openvino=LOCAL_MODEL_USE_OPENVINO)

    generation_kwargs: dict[str, Any] = {
        "prompt": prompt,
        "num_inference_steps": LOCAL_NUM_INFERENCE_STEPS,
        "guidance_scale": LOCAL_GUIDANCE_SCALE,
        "width": LOCAL_IMAGE_WIDTH,
        "height": LOCAL_IMAGE_HEIGHT,
    }

    if LOCAL_SEED >= 0 and not LOCAL_MODEL_USE_OPENVINO:
        try:
            import torch

            generation_kwargs["generator"] = torch.Generator(device="cpu").manual_seed(LOCAL_SEED)
        except Exception:
            logger.warning("Failed to set LOCAL_SEED generator; continuing without deterministic seed")

    if LOCAL_MODEL_USE_OPENVINO and OPENVINO_OOM_AUTO_RETRY:
        size_candidates = _openvino_size_candidates(LOCAL_IMAGE_WIDTH, LOCAL_IMAGE_HEIGHT)
        last_error: Exception | None = None
        result = None
        did_cpu_fallback = False
        for width, height in size_candidates:
            try:
                if width != LOCAL_IMAGE_WIDTH or height != LOCAL_IMAGE_HEIGHT:
                    logger.warning(
                        f"Retrying OpenVINO generation with reduced size to avoid GPU OOM: {width}x{height}"
                    )
                generation_kwargs["width"] = width
                generation_kwargs["height"] = height
                result = pipeline(**generation_kwargs)
                break
            except Exception as exc:
                last_error = exc
                if (
                    OPENVINO_GPU_FALLBACK_TO_CPU
                    and not did_cpu_fallback
                    and str(OPENVINO_DEVICE).upper().startswith("GPU")
                    and _is_openvino_gpu_runtime_error(exc)
                ):
                    logger.warning(
                        "OpenVINO GPU runtime failure detected; retrying on CPU backend for stability."
                    )
                    pipeline.to("CPU")
                    did_cpu_fallback = True
                    continue
                if _is_openvino_memory_error(exc) and (width, height) != size_candidates[-1]:
                    continue
                raise

        if result is None:
            raise RuntimeError(f"OpenVINO generation failed after OOM retries. Last error: {last_error}")
    else:
        result = pipeline(**generation_kwargs)

    image = result.images[0]
    return _save_image_as_wallpaper(image)


def download_local_model(model_id: str | None = None, local_dir: str | None = None) -> str:
    target_model = (model_id or get_settings().local_model_id).strip()
    if not target_model:
        raise RuntimeError("No model id provided. Pass --model-id or set LOCAL_MODEL_ID.")

    destination = Path(local_dir).expanduser().resolve() if local_dir else (Path(__file__).resolve().parents[1] / "models")
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