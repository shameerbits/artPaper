import os
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import requests

from utils.config import ACCEL_DEFAULT_ONNX_MODEL_PATH, MODELS_DIR, UPSCALED_DIR
from utils.logger import logger


ACCEL_ONNX_MODEL_PATH = os.getenv("ACCEL_ONNX_MODEL_PATH", str(ACCEL_DEFAULT_ONNX_MODEL_PATH)).strip()
ACCEL_ONNX_MODEL_URL = os.getenv("ACCEL_ONNX_MODEL_URL", "").strip()
ACCEL_MODEL_CACHE_DIR = Path(
    os.getenv("ACCEL_MODEL_CACHE_DIR", str(MODELS_DIR))
)
ACCEL_DEFAULT_MODEL_PATH = Path(os.getenv("ACCEL_DEFAULT_MODEL_PATH", str(ACCEL_DEFAULT_ONNX_MODEL_PATH)))
ACCEL_TILE = max(int(os.getenv("ACCEL_TILE", "0")), 0)
ACCEL_TILE_PAD = max(int(os.getenv("ACCEL_TILE_PAD", "8")), 0)
OPENVINO_DEVICE = os.getenv("OPENVINO_DEVICE", "GPU_FP32").strip() or "GPU_FP32"
DIRECTML_DEVICE_ID = max(int(os.getenv("DIRECTML_DEVICE_ID", "0")), 0)


def _build_output_path(prefix: str) -> Path:
    filename = f"{prefix}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.png"
    return Path(UPSCALED_DIR) / filename


def _resolve_model_path() -> Path:
    if ACCEL_ONNX_MODEL_PATH:
        model_path = Path(ACCEL_ONNX_MODEL_PATH)
        if model_path.exists():
            return model_path
        logger.warning(f"ACCEL_ONNX_MODEL_PATH does not exist: {model_path}. Trying fallback options.")

    if ACCEL_DEFAULT_MODEL_PATH.exists():
        logger.info(f"Using default ONNX model path: {ACCEL_DEFAULT_MODEL_PATH}")
        return ACCEL_DEFAULT_MODEL_PATH

    if not ACCEL_ONNX_MODEL_URL:
        raise RuntimeError(
            "No ONNX model configured for accelerated upscaler. "
            "Provide ACCEL_ONNX_MODEL_PATH, ACCEL_ONNX_MODEL_URL, "
            "or place model at the default path: "
            f"{ACCEL_DEFAULT_MODEL_PATH}"
        )

    ACCEL_MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    filename = Path(ACCEL_ONNX_MODEL_URL.split("?")[0]).name or "upscaler.onnx"
    model_path = ACCEL_MODEL_CACHE_DIR / filename
    if not model_path.exists():
        logger.info(f"Downloading ONNX upscaler model to {model_path}")
        response = requests.get(ACCEL_ONNX_MODEL_URL, timeout=300)
        response.raise_for_status()
        model_path.write_bytes(response.content)
    return model_path


def _create_session(model_path: Path, provider: str):
    try:
        import onnxruntime as ort
    except Exception as exc:
        if provider == "DmlExecutionProvider":
            install_hint = "pip install onnxruntime-directml"
        else:
            install_hint = "pip install onnxruntime-openvino"
        raise RuntimeError(
            "onnxruntime is not installed for accelerated upscaling. Install with: "
            f"{install_hint}"
        ) from exc

    available = ort.get_available_providers()
    if provider not in available:
        raise RuntimeError(
            f"Requested provider '{provider}' is not available. Available providers: {available}."
        )

    provider_stack: list = [provider, "CPUExecutionProvider"]
    if provider == "OpenVINOExecutionProvider":
        provider_stack = [("OpenVINOExecutionProvider", {"device_type": OPENVINO_DEVICE}), "CPUExecutionProvider"]
    if provider == "DmlExecutionProvider":
        provider_stack = [("DmlExecutionProvider", {"device_id": DIRECTML_DEVICE_ID}), "CPUExecutionProvider"]

    return ort.InferenceSession(str(model_path), providers=provider_stack)


def _prepare_input(image_bgr: np.ndarray) -> np.ndarray:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_float = image_rgb.astype(np.float32) / 255.0
    return np.transpose(image_float, (2, 0, 1))[None, ...]


def _fixed_input_hw(session) -> tuple[int, int] | None:
    model_input = session.get_inputs()[0]
    shape = getattr(model_input, "shape", None)
    if not isinstance(shape, (list, tuple)) or len(shape) != 4:
        return None

    in_h = shape[2]
    in_w = shape[3]
    if isinstance(in_h, int) and isinstance(in_w, int) and in_h > 0 and in_w > 0:
        return in_h, in_w
    return None


def _run_model(session, image_bgr: np.ndarray) -> np.ndarray:
    input_tensor = _prepare_input(image_bgr)
    model_input = session.get_inputs()[0]
    input_type = model_input.type

    if "float16" in input_type:
        input_tensor = input_tensor.astype(np.float16)
    elif "float" not in input_type:
        raise RuntimeError(f"Unsupported ONNX input dtype: {input_type}. Expected float32/float16.")

    output = session.run(None, {model_input.name: input_tensor})[0]
    if output.ndim == 4:
        output = output[0]

    if output.ndim == 3 and output.shape[0] in {1, 3}:
        output = np.transpose(output, (1, 2, 0))

    if output.ndim != 3 or output.shape[2] not in {1, 3}:
        raise RuntimeError(f"Unexpected ONNX output shape: {output.shape}")

    output = np.clip(output, 0.0, 1.0)
    if output.shape[2] == 1:
        output = np.repeat(output, 3, axis=2)

    output_uint8 = (output * 255.0 + 0.5).astype(np.uint8)
    return cv2.cvtColor(output_uint8, cv2.COLOR_RGB2BGR)


def _run_model_tiled(
    session,
    image_bgr: np.ndarray,
    tile_h: int,
    tile_w: int,
    tile_pad: int,
    fixed_hw: tuple[int, int] | None = None,
) -> np.ndarray:
    height, width = image_bgr.shape[:2]
    out_image = None
    scale_x = 0.0
    scale_y = 0.0

    for y in range(0, height, tile_h):
        for x in range(0, width, tile_w):
            in_x0 = max(x - tile_pad, 0)
            in_y0 = max(y - tile_pad, 0)
            in_x1 = min(x + tile_w + tile_pad, width)
            in_y1 = min(y + tile_h + tile_pad, height)

            patch = image_bgr[in_y0:in_y1, in_x0:in_x1]
            patch_h, patch_w = patch.shape[:2]
            model_in_h = patch_h
            model_in_w = patch_w

            if fixed_hw is not None:
                model_in_h, model_in_w = fixed_hw
                if patch_h > model_in_h or patch_w > model_in_w:
                    raise RuntimeError(
                        f"Patch size {patch_w}x{patch_h} exceeds fixed model input {model_in_w}x{model_in_h}."
                    )
                if patch_h != model_in_h or patch_w != model_in_w:
                    pad_bottom = model_in_h - patch_h
                    pad_right = model_in_w - patch_w
                    patch = cv2.copyMakeBorder(
                        patch,
                        0,
                        pad_bottom,
                        0,
                        pad_right,
                        cv2.BORDER_REFLECT_101,
                    )

            patch_out = _run_model(session, patch)

            if scale_x == 0.0 or scale_y == 0.0:
                scale_x = patch_out.shape[1] / float(model_in_w)
                scale_y = patch_out.shape[0] / float(model_in_h)
                out_h = int(round(height * scale_y))
                out_w = int(round(width * scale_x))
                out_image = np.zeros((out_h, out_w, 3), dtype=np.uint8)

            patch_inner_x0 = x - in_x0
            patch_inner_y0 = y - in_y0
            patch_inner_x1 = patch_inner_x0 + min(tile_w, width - x)
            patch_inner_y1 = patch_inner_y0 + min(tile_h, height - y)

            out_inner_x0 = int(round(patch_inner_x0 * scale_x))
            out_inner_y0 = int(round(patch_inner_y0 * scale_y))
            out_inner_x1 = int(round(patch_inner_x1 * scale_x))
            out_inner_y1 = int(round(patch_inner_y1 * scale_y))

            cropped = patch_out[out_inner_y0:out_inner_y1, out_inner_x0:out_inner_x1]
            dest_x0 = int(round(x * scale_x))
            dest_y0 = int(round(y * scale_y))
            dest_x1 = dest_x0 + cropped.shape[1]
            dest_y1 = dest_y0 + cropped.shape[0]
            out_image[dest_y0:dest_y1, dest_x0:dest_x1] = cropped

    if out_image is None:
        raise RuntimeError("Failed to produce tiled output.")
    return out_image


def _upscale_with_provider(image_path: str, provider: str, output_prefix: str) -> str:
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read image for accelerated upscaling: {image_path}")

    model_path = _resolve_model_path()
    session = _create_session(model_path, provider)
    logger.info(f"Accelerated ONNX upscaling with provider={provider}, model={model_path}")

    fixed_hw = _fixed_input_hw(session)
    if fixed_hw is not None:
        logger.info(f"Detected fixed ONNX input size: {fixed_hw[1]}x{fixed_hw[0]}")

    effective_tile_h = ACCEL_TILE
    effective_tile_w = ACCEL_TILE
    effective_tile_pad = ACCEL_TILE_PAD

    if fixed_hw is not None:
        fixed_h, fixed_w = fixed_hw
        effective_tile_h = fixed_h
        effective_tile_w = fixed_w
        effective_tile_pad = 0
        logger.info("Using fixed-input tiled inference for ONNX model.")

    if fixed_hw is not None or ACCEL_TILE > 0:
        output = _run_model_tiled(
            session,
            image,
            tile_h=effective_tile_h,
            tile_w=effective_tile_w,
            tile_pad=effective_tile_pad,
            fixed_hw=fixed_hw,
        )
    else:
        output = _run_model(session, image)

    output_path = _build_output_path(output_prefix)
    if not cv2.imwrite(str(output_path), output):
        raise RuntimeError(f"Failed to write accelerated upscaled image: {output_path}")

    logger.info(f"Upscaled image saved to {output_path}")
    return str(output_path)


def upscale_with_directml(image_path: str) -> str:
    return _upscale_with_provider(image_path, provider="DmlExecutionProvider", output_prefix="upscaled_directml")


def upscale_with_openvino(image_path: str) -> str:
    return _upscale_with_provider(
        image_path,
        provider="OpenVINOExecutionProvider",
        output_prefix="upscaled_openvino",
    )