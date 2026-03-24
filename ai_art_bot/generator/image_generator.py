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
LOCAL_SAMPLER = os.getenv("LOCAL_SAMPLER", "euler_a").strip().lower()
LOCAL_NEGATIVE_PROMPT = os.getenv("LOCAL_NEGATIVE_PROMPT", "").strip()
LOCAL_SEED = int(os.getenv("LOCAL_SEED", "-1"))
LOCAL_ENABLE_LONG_PROMPTS = (
    os.getenv("LOCAL_ENABLE_LONG_PROMPTS", "true").strip().lower() in {"1", "true", "yes"}
)
LOCAL_DIRECTML_FALLBACK_TO_CPU = (
    os.getenv("LOCAL_DIRECTML_FALLBACK_TO_CPU", "true").strip().lower() in {"1", "true", "yes"}
)
LOCAL_MODEL_USE_DIRECTML = (
    os.getenv("LOCAL_MODEL_USE_DIRECTML", "false")
    .strip()
    .lower()
    in {"1", "true", "yes"}
)
LOCAL_ENABLE_ATTENTION_SLICING = (
    os.getenv("LOCAL_ENABLE_ATTENTION_SLICING", "false").strip().lower() in {"1", "true", "yes"}
)
LOCAL_ENABLE_VAE_TILING = (
    os.getenv("LOCAL_ENABLE_VAE_TILING", "false").strip().lower() in {"1", "true", "yes"}
)
LOCAL_AUTO_MEMORY_OPTIMIZATION = (
    os.getenv("LOCAL_AUTO_MEMORY_OPTIMIZATION", "false").strip().lower() in {"1", "true", "yes"}
)
LOCAL_DIRECTML_PREFER_FLOAT32 = (
    os.getenv("LOCAL_DIRECTML_PREFER_FLOAT32", "true").strip().lower() in {"1", "true", "yes"}
)
DIRECTML_DEVICE_ID = max(int(os.getenv("DIRECTML_DEVICE_ID", "0")), 0)

_LOCAL_PIPELINE: Any | None = None
_LOCAL_PIPELINE_SOURCE: str | None = None
_LOCAL_PIPELINE_USE_DIRECTML: bool | None = None
_DIRECTML_CAUSAL_MASK_PATCHED = False

_XFORMERS_AVAILABLE = False
try:
    import xformers
    _XFORMERS_AVAILABLE = True
except ImportError:
    _XFORMERS_AVAILABLE = False



def _patch_transformers_causal_mask_for_directml() -> None:
    """Patch Transformers causal mask creation to avoid Float64 ops on DirectML backends."""
    global _DIRECTML_CAUSAL_MASK_PATCHED
    if _DIRECTML_CAUSAL_MASK_PATCHED:
        return

    try:
        import torch
        from transformers.modeling_attn_mask_utils import AttentionMaskConverter
    except Exception:
        return

    original_make_causal_mask = getattr(AttentionMaskConverter, "_make_causal_mask", None)
    if original_make_causal_mask is None:
        return

    def _make_causal_mask_directml_safe(
        input_ids_shape: Any,
        dtype: Any,
        device: Any,
        past_key_values_length: int = 0,
        sliding_window: int | None = None,
    ) -> Any:
        bsz, tgt_len = input_ids_shape
        cpu_device = torch.device("cpu")
        min_value = torch.finfo(dtype).min

        # Build on CPU first to avoid DirectML kernels that may implicitly request Float64.
        mask = torch.full((tgt_len, tgt_len), min_value, dtype=dtype, device=cpu_device)
        mask_cond = torch.arange(mask.size(-1), device=cpu_device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)

        if past_key_values_length > 0:
            prefix = torch.zeros((tgt_len, past_key_values_length), dtype=dtype, device=cpu_device)
            mask = torch.cat([prefix, mask], dim=-1)

        if sliding_window is not None:
            diagonal = past_key_values_length - sliding_window + 1
            context_mask = torch.triu(torch.ones_like(mask, dtype=torch.bool), diagonal=diagonal)
            mask.masked_fill_(context_mask, min_value)

        mask = mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)

        if device is not None:
            try:
                target_device = torch.device(device)
            except Exception:
                target_device = device
            if str(target_device) != "cpu":
                mask = mask.to(device=target_device, dtype=dtype)
        return mask

    setattr(AttentionMaskConverter, "_make_causal_mask", staticmethod(_make_causal_mask_directml_safe))
    _DIRECTML_CAUSAL_MASK_PATCHED = True
    logger.info("Applied DirectML-safe patch for transformers causal attention mask")


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


def _get_available_vram() -> int | None:
    """Estimate available VRAM in MB. Returns None if unable to detect."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory // (1024 ** 2)
    except Exception:
        pass

    if LOCAL_MODEL_USE_DIRECTML:
        try:
            import torch_directml
            # DirectML doesn't expose VRAM directly; assume Conservative estimate for guidance.
            return 2048  # Conservative fallback
        except Exception:
            pass

    return None


def _apply_memory_optimizations(pipeline: Any, enable_attention_slicing: bool = False, 
                                 enable_vae_tiling: bool = False, auto_optimize: bool = False) -> None:
    """Apply memory optimization techniques to reduce pipeline VRAM usage.
    
    Args:
        pipeline: The diffusers pipeline to optimize
        enable_attention_slicing: Slice attention computation (reduces memory ~20%, slower ~20%)
        enable_vae_tiling: Process VAE in tiles (reduces memory ~15%, slower ~5-10%)
        auto_optimize: Automatically enable both if VRAM < 4GB
    """
    should_optimize = enable_attention_slicing or enable_vae_tiling or auto_optimize
    if not should_optimize:
        return

    available_vram = None
    if auto_optimize:
        available_vram = _get_available_vram()
        if available_vram is not None and available_vram >= 4096:
            # Sufficient VRAM, skip optimizations
            logger.info(f"Auto memory optimization skipped: available VRAM {available_vram}MB >= 4GB threshold")
            return
        if available_vram is not None:
            logger.info(f"Auto memory optimization enabled: available VRAM {available_vram}MB < 4GB")
        enable_attention_slicing = True
        enable_vae_tiling = True

    # Enable attention slicing for memory savings
    if enable_attention_slicing:
        try:
            pipeline.enable_attention_slicing()
            logger.info("Enabled attention slicing (memory-optimized, ~20% slower)")
        except Exception as exc:
            logger.warning(f"Failed to enable attention slicing: {exc}")

    # Enable VAE tiling to process decode/encode in chunks
    if enable_vae_tiling:
        try:
            pipeline.enable_vae_tiling()
            logger.info("Enabled VAE tiling (memory-optimized, ~5-10% slower)")
        except Exception as exc:
            logger.warning(f"Failed to enable VAE tiling: {exc}")

    # Try to enable xformers optimizations (independent of memory state)
    _enable_xformers_memory_efficient_attention(pipeline)





def _get_available_vram() -> int | None:
    """Estimate available VRAM in MB. Returns None if unable to detect."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory // (1024 ** 2)
    except Exception:
        pass

    if LOCAL_MODEL_USE_DIRECTML:
        try:
            import torch_directml
            # DirectML doesn't expose VRAM directly; assume conservative estimate for guidance.
            return 2048  # Conservative fallback
        except Exception:
            pass

    return None


def _get_optimal_dtype(use_directml: bool) -> Any:
    """Determine optimal tensor dtype based on hardware capabilities.
    
    Returns float16 for memory efficiency on CUDA/DirectML, 
    but may fall back to float32 if needed.
    """
    try:
        import torch
        # DirectML can run float16, but many Windows drivers are more stable with float32.
        if use_directml:
            return torch.float32 if LOCAL_DIRECTML_PREFER_FLOAT32 else torch.float16
        if torch.cuda.is_available():
            return torch.float16
        return torch.float32
    except Exception:
        return None


def _should_fallback_from_directml(error_text: str) -> bool:
    lowered = error_text.lower()
    fallback_markers = (
        "does not support double (float64) operations",
        "gpu device instance has been suspended",
        "getdeviceremovedreason",
        "device removed",
        "dxgi_error_device_removed",
        "dxgi_error_device_hung",
        "out of memory",
    )
    return any(marker in lowered for marker in fallback_markers)


def _enable_xformers_memory_efficient_attention(pipeline: Any) -> bool:
    """Enable xformers memory-efficient attention if available.
    
    xformers provides ~20% speedup and ~10% memory savings via optimized attention kernels.
    Returns True if successfully enabled, False otherwise.
    """
    if not _XFORMERS_AVAILABLE:
        return False
    
    try:
        pipeline.enable_xformers_memory_efficient_attention()
        logger.info("Enabled xformers memory-efficient attention (~20% faster, ~10% less memory)")
        return True
    except Exception as exc:
        logger.debug(f"Could not enable xformers attention: {exc}")
        return False


def _enable_model_cpu_offload(pipeline: Any) -> bool:
    """Enable CPU offloading to reduce peak VRAM usage.
    
    Components are moved to CPU and only brought to GPU when needed, reducing max VRAM.
    Tradeoff: ~10-15% slower execution but ~40-50% VRAM reduction.
    Returns True if successfully enabled, False otherwise.
    """
    try:
        pipeline.enable_model_cpu_offload()
        logger.info("Enabled model CPU offloading (40-50% VRAM reduction, ~10-15% slower)")
        return True
    except Exception as exc:
        logger.debug(f"Could not enable model CPU offload: {exc}")
        return False


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
        from diffusers import (
            AutoPipelineForText2Image,
            DDIMScheduler,
            DPMSolverMultistepScheduler,
            EulerAncestralDiscreteScheduler,
        )
    except Exception as exc:
        raise RuntimeError(
            "Local SD generation requires `diffusers`, `transformers`, `accelerate`, and `safetensors`. "
            "Install optional local generation dependencies first."
        ) from exc

    optimal_dtype = _get_optimal_dtype(use_directml)
    logger.info(f"Loading model with dtype={optimal_dtype}")

    # Load pipeline with memory optimizations
    pipeline = AutoPipelineForText2Image.from_pretrained(
        model_source,
        torch_dtype=optimal_dtype or torch.float32,
        low_cpu_mem_usage=True,  # Avoid duplicate weights during loading
        use_safetensors=True,     # Use safetensors for faster, safer loading
        safety_checker=None,      # Remove unnecessary component
    )
    sampler_aliases = {
        "euler_a": "euler_a",
        "euler_ancestral": "euler_a",
        "euler_ancestral_discrete": "euler_a",
        "ddim": "ddim",
        "dpm": "dpm",
        "dpm_solver": "dpm",
        "dpm_solver_multistep": "dpm",
    }
    scheduler_key = sampler_aliases.get(LOCAL_SAMPLER, "euler_a")
    scheduler_map = {
        "euler_a": EulerAncestralDiscreteScheduler,
        "ddim": DDIMScheduler,
        "dpm": DPMSolverMultistepScheduler,
    }
    scheduler_class = scheduler_map[scheduler_key]
    pipeline.scheduler = scheduler_class.from_config(pipeline.scheduler.config)
    logger.info(
        f"Using local SD sampler: requested='{LOCAL_SAMPLER}' resolved='{scheduler_key}' class={scheduler_class.__name__}"
    )

    if use_directml:
        try:
            import torch_directml
        except Exception as exc:
            raise RuntimeError(
                "DirectML local generation requires `torch-directml`. "
                "Install with: pip install torch-directml"
            ) from exc

        _patch_transformers_causal_mask_for_directml()
        dml_device = torch_directml.device(DIRECTML_DEVICE_ID)
        pipeline = pipeline.to(dml_device)
        _patch_scheduler_for_directml_float32(pipeline)
    else:
        pipeline = pipeline.to("cpu")

    # Enable xformers memory-efficient attention (if available, independent of device)
    _enable_xformers_memory_efficient_attention(pipeline)

    # Enable model CPU offloading for CUDA backend (reduces peak memory by 40-50%)
    # Note: Disable on DirectML due to compatibility; use attention slicing instead
    if not use_directml and _XFORMERS_AVAILABLE:
        _enable_model_cpu_offload(pipeline)

    # Apply memory optimizations (before storing in global)
    _apply_memory_optimizations(
        pipeline,
        enable_attention_slicing=LOCAL_ENABLE_ATTENTION_SLICING,
        enable_vae_tiling=LOCAL_ENABLE_VAE_TILING,
        auto_optimize=LOCAL_AUTO_MEMORY_OPTIMIZATION,
    )

    _LOCAL_PIPELINE = pipeline
    _LOCAL_PIPELINE_SOURCE = model_source
    _LOCAL_PIPELINE_USE_DIRECTML = use_directml
    logger.info(
        f"Loaded local model source={model_source} "
        f"backend={'directml' if use_directml else 'cuda' if torch.cuda.is_available() else 'cpu'} "
        f"dtype={optimal_dtype or torch.float32}"
    )
    return pipeline


def _encode_prompt_chunks(pipeline: Any, prompt: str, *, target_chunk_count: int | None = None) -> tuple[Any, int]:
    """Encode arbitrarily long prompt by chunking into CLIP-sized windows and concatenating embeddings."""
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("Long prompt encoding requires torch") from exc

    tokenizer = getattr(pipeline, "tokenizer", None)
    text_encoder = getattr(pipeline, "text_encoder", None)
    if tokenizer is None or text_encoder is None:
        raise RuntimeError("Pipeline does not expose tokenizer/text_encoder for long prompt encoding")

    max_length = int(getattr(tokenizer, "model_max_length", 77))
    if max_length < 3:
        raise RuntimeError(f"Invalid tokenizer model_max_length: {max_length}")

    bos_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.cls_token_id
    eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.sep_token_id
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_token_id
    if bos_token_id is None or eos_token_id is None or pad_token_id is None:
        raise RuntimeError("Tokenizer is missing BOS/EOS/PAD token ids for long prompt encoding")

    tokenized = tokenizer(
        prompt,
        add_special_tokens=False,
        truncation=False,
        return_attention_mask=False,
    )
    token_ids = list(tokenized["input_ids"])
    payload_len = max_length - 2
    chunks = [token_ids[idx : idx + payload_len] for idx in range(0, len(token_ids), payload_len)]
    if not chunks:
        chunks = [[]]

    if target_chunk_count is not None and target_chunk_count > len(chunks):
        chunks.extend([[] for _ in range(target_chunk_count - len(chunks))])

    device = getattr(text_encoder, "device", None)
    if device is None:
        device = getattr(pipeline, "_execution_device", None) or getattr(pipeline, "device", "cpu")

    encoded_chunks: list[Any] = []
    for chunk in chunks:
        ids = [bos_token_id] + chunk + [eos_token_id]
        attention_mask_values = [1] * len(ids)
        if len(ids) < max_length:
            pad_count = max_length - len(ids)
            ids.extend([pad_token_id] * pad_count)
            attention_mask_values.extend([0] * pad_count)

        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        attention_mask = torch.tensor([attention_mask_values], dtype=torch.long, device=device)
        with torch.no_grad():
            chunk_embeds = text_encoder(input_ids, attention_mask=attention_mask)[0]
        encoded_chunks.append(chunk_embeds)

    prompt_embeds = torch.cat(encoded_chunks, dim=1)
    return prompt_embeds, len(chunks)


def _inject_long_prompt_embeddings(pipeline: Any, generation_kwargs: dict[str, Any]) -> None:
    """Switch prompt input to prompt_embeds to support prompts longer than tokenizer max length."""
    if not LOCAL_ENABLE_LONG_PROMPTS:
        return
    if "prompt" not in generation_kwargs or "prompt_embeds" in generation_kwargs:
        return
    if hasattr(pipeline, "text_encoder_2"):
        # SDXL-like pipelines need additional pooled embeddings handling; keep default behavior.
        return

    prompt = str(generation_kwargs.get("prompt") or "")
    if not prompt.strip():
        return

    try:
        prompt_embeds, chunk_count = _encode_prompt_chunks(pipeline, prompt)
        if chunk_count <= 1:
            return

        negative_prompt = str(generation_kwargs.get("negative_prompt") or "")
        negative_prompt_embeds, _ = _encode_prompt_chunks(
            pipeline,
            negative_prompt,
            target_chunk_count=chunk_count,
        )

        generation_kwargs.pop("prompt", None)
        generation_kwargs.pop("negative_prompt", None)
        generation_kwargs["prompt_embeds"] = prompt_embeds
        if float(generation_kwargs.get("guidance_scale", 1.0)) > 1.0:
            generation_kwargs["negative_prompt_embeds"] = negative_prompt_embeds

        logger.info(
            f"Long prompt enabled: encoded prompt in {chunk_count} CLIP chunks ({chunk_count * 77} token slots)"
        )
    except Exception as exc:
        logger.warning(f"Long prompt embedding fallback failed; using default tokenizer truncation: {exc}")


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
    if LOCAL_NEGATIVE_PROMPT:
        generation_kwargs["negative_prompt"] = LOCAL_NEGATIVE_PROMPT
    _inject_long_prompt_embeddings(pipeline, generation_kwargs)

    if LOCAL_SEED >= 0 and not LOCAL_MODEL_USE_DIRECTML:
        try:
            import torch

            generation_kwargs["generator"] = torch.Generator(device="cpu").manual_seed(LOCAL_SEED)
        except Exception:
            logger.warning("Failed to set LOCAL_SEED generator; continuing without deterministic seed")
    elif LOCAL_SEED >= 0 and LOCAL_MODEL_USE_DIRECTML:
        logger.warning("LOCAL_SEED is ignored for DirectML local generation on this backend")

    # Generate image; DirectML may require CPU fallback if device is removed/suspended.
    try:
        result = pipeline(**generation_kwargs)
    except Exception as exc:
        error_text = str(exc)
        if (
            LOCAL_DIRECTML_FALLBACK_TO_CPU
            and LOCAL_MODEL_USE_DIRECTML
            and _should_fallback_from_directml(error_text)
        ):
            logger.warning(
                "DirectML failed (device removed/suspended or unsupported op). Falling back to CPU generation for this run."
            )
            cpu_pipeline = _load_local_pipeline(model_source=model_source, use_directml=False)
            result = cpu_pipeline(**generation_kwargs)
        else:
            raise

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