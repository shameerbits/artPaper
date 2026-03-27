import argparse
import json
import os
import sys
from typing import Any, TYPE_CHECKING

import requests
from utils.config import (
    SECRET_ENV_KEYS,
    WEB_CONFIG_KEYS,
    apply_web_settings_to_env,
    get_settings,
    has_local_deviantart_tokens,
    load_web_settings,
    save_web_settings,
)
from utils.logger import logger
from utils.logger import setup_logging


if TYPE_CHECKING:
    from fastapi import FastAPI
    from scheduler.scheduler import PipelineRunner


def _build_runner() -> "PipelineRunner":
    from scheduler.scheduler import PipelineRunner

    return PipelineRunner()


class _PromptCloudRunner:
    def __init__(self) -> None:
        from database.db import init_db

        init_db()

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
        from database.db import enqueue_task

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
        from database.db import list_tasks

        return list_tasks(limit=limit, status=status)


def _build_cloud_prompt_runner() -> _PromptCloudRunner:
    return _PromptCloudRunner()


def build_dashboard() -> "FastAPI":
    from fastapi import FastAPI, HTTPException
    from database.db import get_task, list_images, list_tasks

    app = FastAPI(title="AI Art Bot", version="0.1.0")
    runner = _build_runner()

    @app.get("/images")
    def images() -> list[dict]:
        return list_images(limit=100)

    @app.post("/generate")
    def generate() -> dict:
        try:
            return runner.run_once(mode="full")
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.get("/tasks")
    def tasks() -> list[dict]:
        return list_tasks(limit=200)

    @app.get("/tasks/{task_id}")
    def task(task_id: int) -> dict:
        payload = get_task(task_id)
        if payload is None:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
        return payload

    @app.post("/tasks/{task_id}/run")
    def run_task(task_id: int) -> dict:
        try:
            return runner.process_task(task_id)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/tasks/run_next")
    def run_next_task() -> dict:
        return runner.process_next_queued_task()

    return app


def _is_streamlit_execution() -> bool:
    markers = ("STREAMLIT_SERVER_PORT", "STREAMLIT_RUNTIME_ENV")
    if any(os.getenv(marker) for marker in markers):
        return True
    return "streamlit.runtime.scriptrunner" in sys.modules


def _is_true(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _is_deployed_streamlit() -> bool:
    explicit = os.getenv("APP_STREAMLIT_DEPLOYED", "").strip()
    if explicit:
        return _is_true(explicit)
    return os.getenv("STREAMLIT_RUNTIME_ENV", "").strip().lower() in {"cloud", "community", "deployed"}


def _load_and_apply_saved_settings() -> dict[str, str]:
    saved = load_web_settings()
    if saved:
        apply_web_settings_to_env(saved)
    return saved


def _cloud_runtime_settings(saved_settings: dict[str, str]) -> dict[str, str]:
    runtime = dict(saved_settings)
    # Streamlit Cloud should default to lightweight and portable backends.
    runtime["IMAGE_BACKEND"] = "openai"
    runtime["UPSCALER_BACKEND"] = "replicate"
    return runtime


def _lookup_secret(st: Any, name: str) -> str:
    value = os.getenv(name, "").strip()
    if value:
        return value
    try:
        secret_value = st.secrets.get(name, "")
    except Exception:
        secret_value = ""
    return str(secret_value).strip()


def _is_secret_available(st: Any, name: str) -> bool:
    return bool(_lookup_secret(st, name))


def _active_setting(task_settings: dict[str, str], name: str, default: str = "") -> str:
    return str(task_settings.get(name, os.getenv(name, default))).strip()


def _required_secrets_for_mode(st: Any, pipeline_mode: str, task_settings: dict[str, str]) -> tuple[list[str], list[str]]:
    required: list[str] = []
    missing: list[str] = []

    image_backend = _active_setting(task_settings, "IMAGE_BACKEND", "openai").lower()
    upscaler_backend = _active_setting(task_settings, "UPSCALER_BACKEND", "realesrgan").lower()
    needs_generate = pipeline_mode in {"generate_only", "generate_upscale", "full"}
    needs_upscale = pipeline_mode in {"upscale_only", "generate_upscale", "upscale_upload", "full"}
    needs_upload = pipeline_mode in {"upload_only", "upscale_upload", "full"}

    if needs_generate and image_backend == "openai":
        required.append("OPENAI_API_KEY")

    if needs_upscale and upscaler_backend == "replicate":
        required.append("REPLICATE_API_TOKEN")

    if needs_upload:
        required.extend(["DEVIANTART_CLIENT_ID", "DEVIANTART_CLIENT_SECRET"])

    required = sorted(set(required))
    for name in required:
        if not _is_secret_available(st, name):
            missing.append(name)

    if needs_upload:
        has_token = (
            _is_secret_available(st, "DEVIANTART_ACCESS_TOKEN")
            or _is_secret_available(st, "DEVIANTART_REFRESH_TOKEN")
            or has_local_deviantart_tokens()
        )
        if not has_token:
            missing.append("DEVIANTART_ACCESS_TOKEN or DEVIANTART_REFRESH_TOKEN (or cached tokens)")

    return required, missing


def _render_secret_status(st: Any) -> None:
    st.subheader("Secrets Status")
    st.caption("Secrets are not editable in this UI. Set them with environment variables or Streamlit secrets.")
    rows = []
    for name in SECRET_ENV_KEYS:
        rows.append({"secret": name, "status": "set" if _is_secret_available(st, name) else "missing"})
    st.dataframe(rows, use_container_width=True, hide_index=True)


def _render_settings_editor(saved_settings: dict[str, str]) -> dict[str, str]:
    import streamlit as st

    st.subheader("Global Configuration")
    st.caption("Save once, and each queued task captures a snapshot of these values.")

    settings_payload: dict[str, str] = {}

    # Image Backend Selection
    st.markdown("##### Image Generation Backend")
    image_backend = st.selectbox(
        "IMAGE_BACKEND",
        options=["openai", "local_sd"],
        index=0 if saved_settings.get("IMAGE_BACKEND", "openai").lower() == "openai" else 1,
        key="cfg_IMAGE_BACKEND",
    )
    settings_payload["IMAGE_BACKEND"] = image_backend

    # Conditional settings for LOCAL_SD
    if image_backend == "local_sd":
        col1, col2 = st.columns(2)
        with col1:
            settings_payload["LOCAL_MODEL_ID"] = st.text_input(
                "LOCAL_MODEL_ID",
                value=saved_settings.get("LOCAL_MODEL_ID", os.getenv("LOCAL_MODEL_ID", "Lykon/dreamshaper-8")),
                key="cfg_LOCAL_MODEL_ID",
            )
            settings_payload["LOCAL_MODEL_PATH"] = st.text_input(
                "LOCAL_MODEL_PATH (optional)",
                value=saved_settings.get("LOCAL_MODEL_PATH", os.getenv("LOCAL_MODEL_PATH", "./models/dreamshaper-8")),
                key="cfg_LOCAL_MODEL_PATH",
            )
        with col2:
            settings_payload["LOCAL_IMAGE_WIDTH"] = st.number_input(
                "LOCAL_IMAGE_WIDTH",
                value=int(saved_settings.get("LOCAL_IMAGE_WIDTH", os.getenv("LOCAL_IMAGE_WIDTH", "512"))),
                key="cfg_LOCAL_IMAGE_WIDTH",
            )
            settings_payload["LOCAL_IMAGE_HEIGHT"] = st.number_input(
                "LOCAL_IMAGE_HEIGHT",
                value=int(saved_settings.get("LOCAL_IMAGE_HEIGHT", os.getenv("LOCAL_IMAGE_HEIGHT", "512"))),
                key="cfg_LOCAL_IMAGE_HEIGHT",
            )

        col1, col2 = st.columns(2)
        with col1:
            settings_payload["LOCAL_NUM_INFERENCE_STEPS"] = st.number_input(
                "LOCAL_NUM_INFERENCE_STEPS",
                value=int(saved_settings.get("LOCAL_NUM_INFERENCE_STEPS", os.getenv("LOCAL_NUM_INFERENCE_STEPS", "35"))),
                key="cfg_LOCAL_NUM_INFERENCE_STEPS",
            )
            settings_payload["LOCAL_GUIDANCE_SCALE"] = st.text_input(
                "LOCAL_GUIDANCE_SCALE",
                value=saved_settings.get("LOCAL_GUIDANCE_SCALE", os.getenv("LOCAL_GUIDANCE_SCALE", "8")),
                key="cfg_LOCAL_GUIDANCE_SCALE",
            )
            sampler_options = [
                "euler_a",
                "euler_ancestral",
                "euler_ancestral_discrete",
                "ddim",
                "dpm",
                "dpm_solver",
                "dpm_solver_multistep",
            ]
            sampler_default = saved_settings.get("LOCAL_SAMPLER", os.getenv("LOCAL_SAMPLER", "euler_a")).strip().lower()
            settings_payload["LOCAL_SAMPLER"] = st.selectbox(
                "LOCAL_SAMPLER",
                options=sampler_options,
                index=sampler_options.index(sampler_default) if sampler_default in sampler_options else 0,
                key="cfg_LOCAL_SAMPLER",
            )
            settings_payload["LOCAL_NEGATIVE_PROMPT"] = st.text_area(
                "LOCAL_NEGATIVE_PROMPT (optional)",
                value=saved_settings.get(
                    "LOCAL_NEGATIVE_PROMPT",
                    os.getenv(
                        "LOCAL_NEGATIVE_PROMPT",
                        "blurry, soft, low detail, repeated patterns, duplicate structures, distorted architecture, smudged, low resolution",
                    ),
                ),
                height=100,
                key="cfg_LOCAL_NEGATIVE_PROMPT",
                help="Used only for local_sd image generation.",
            )
        with col2:
            settings_payload["LOCAL_SEED"] = st.text_input(
                "LOCAL_SEED",
                value=saved_settings.get("LOCAL_SEED", os.getenv("LOCAL_SEED", "-1")),
                key="cfg_LOCAL_SEED",
            )
            settings_payload["LOCAL_MODEL_USE_DIRECTML"] = st.selectbox(
                "LOCAL_MODEL_USE_DIRECTML",
                options=["false", "true"],
                index=1
                if saved_settings.get("LOCAL_MODEL_USE_DIRECTML", os.getenv("LOCAL_MODEL_USE_DIRECTML", "false")).lower()
                == "true"
                else 0,
                key="cfg_LOCAL_MODEL_USE_DIRECTML",
            )
            settings_payload["LOCAL_DIRECTML_PREFER_FLOAT32"] = st.selectbox(
                "LOCAL_DIRECTML_PREFER_FLOAT32",
                options=["true", "false"],
                index=0
                if saved_settings.get(
                    "LOCAL_DIRECTML_PREFER_FLOAT32",
                    os.getenv("LOCAL_DIRECTML_PREFER_FLOAT32", "true"),
                ).lower()
                == "true"
                else 1,
                key="cfg_LOCAL_DIRECTML_PREFER_FLOAT32",
            )
            settings_payload["LOCAL_DIRECTML_FALLBACK_TO_CPU"] = st.selectbox(
                "LOCAL_DIRECTML_FALLBACK_TO_CPU",
                options=["true", "false"],
                index=0
                if saved_settings.get(
                    "LOCAL_DIRECTML_FALLBACK_TO_CPU",
                    os.getenv("LOCAL_DIRECTML_FALLBACK_TO_CPU", "true"),
                ).lower()
                == "true"
                else 1,
                key="cfg_LOCAL_DIRECTML_FALLBACK_TO_CPU",
            )
            settings_payload["LOCAL_ENABLE_ATTENTION_SLICING"] = st.selectbox(
                "LOCAL_ENABLE_ATTENTION_SLICING",
                options=["true", "false"],
                index=0
                if saved_settings.get(
                    "LOCAL_ENABLE_ATTENTION_SLICING",
                    os.getenv("LOCAL_ENABLE_ATTENTION_SLICING", "true"),
                ).lower()
                == "true"
                else 1,
                key="cfg_LOCAL_ENABLE_ATTENTION_SLICING",
            )
            settings_payload["LOCAL_ENABLE_VAE_TILING"] = st.selectbox(
                "LOCAL_ENABLE_VAE_TILING",
                options=["true", "false"],
                index=0
                if saved_settings.get(
                    "LOCAL_ENABLE_VAE_TILING",
                    os.getenv("LOCAL_ENABLE_VAE_TILING", "true"),
                ).lower()
                == "true"
                else 1,
                key="cfg_LOCAL_ENABLE_VAE_TILING",
            )
            settings_payload["LOCAL_AUTO_MEMORY_OPTIMIZATION"] = st.selectbox(
                "LOCAL_AUTO_MEMORY_OPTIMIZATION",
                options=["true", "false"],
                index=0
                if saved_settings.get(
                    "LOCAL_AUTO_MEMORY_OPTIMIZATION",
                    os.getenv("LOCAL_AUTO_MEMORY_OPTIMIZATION", "true"),
                ).lower()
                == "true"
                else 1,
                key="cfg_LOCAL_AUTO_MEMORY_OPTIMIZATION",
            )

    # Upscaler Backend Selection
    st.markdown("##### Upscaler Backend")
    upscaler_default = saved_settings.get("UPSCALER_BACKEND", os.getenv("UPSCALER_BACKEND", "realesrgan")).lower()
    upscaler_options = ["realesrgan", "replicate", "directml"]
    upscaler_backend = st.selectbox(
        "UPSCALER_BACKEND",
        options=upscaler_options,
        index=upscaler_options.index(upscaler_default) if upscaler_default in upscaler_options else 0,
        key="cfg_UPSCALER_BACKEND",
    )
    settings_payload["UPSCALER_BACKEND"] = upscaler_backend

    # Conditional settings for UPSCALER
    if upscaler_backend == "replicate":
        settings_payload["REPLICATE_UPSCALER_MODEL"] = st.text_input(
            "REPLICATE_UPSCALER_MODEL",
            value=saved_settings.get("REPLICATE_UPSCALER_MODEL", os.getenv("REPLICATE_UPSCALER_MODEL", "")),
            key="cfg_REPLICATE_UPSCALER_MODEL",
        )
        settings_payload["REPLICATE_UPSCALER_SCALE"] = st.text_input(
            "REPLICATE_UPSCALER_SCALE",
            value=saved_settings.get("REPLICATE_UPSCALER_SCALE", os.getenv("REPLICATE_UPSCALER_SCALE", "4")),
            key="cfg_REPLICATE_UPSCALER_SCALE",
        )
    elif upscaler_backend == "realesrgan":
        col1, col2 = st.columns(2)
        with col1:
            settings_payload["REALESRGAN_MODEL_NAME"] = st.text_input(
                "REALESRGAN_MODEL_NAME",
                value=saved_settings.get("REALESRGAN_MODEL_NAME", os.getenv("REALESRGAN_MODEL_NAME", "RealESRGAN_x4plus")),
                key="cfg_REALESRGAN_MODEL_NAME",
            )
            settings_payload["REALESRGAN_TILE"] = st.number_input(
                "REALESRGAN_TILE",
                value=int(saved_settings.get("REALESRGAN_TILE", os.getenv("REALESRGAN_TILE", "256"))),
                key="cfg_REALESRGAN_TILE",
            )
        with col2:
            settings_payload["REALESRGAN_TILE_PAD"] = st.number_input(
                "REALESRGAN_TILE_PAD",
                value=int(saved_settings.get("REALESRGAN_TILE_PAD", os.getenv("REALESRGAN_TILE_PAD", "10"))),
                key="cfg_REALESRGAN_TILE_PAD",
            )
            settings_payload["REALESRGAN_PRE_PAD"] = st.number_input(
                "REALESRGAN_PRE_PAD",
                value=int(saved_settings.get("REALESRGAN_PRE_PAD", os.getenv("REALESRGAN_PRE_PAD", "0"))),
                key="cfg_REALESRGAN_PRE_PAD",
            )
    elif upscaler_backend == "directml":
        col1, col2 = st.columns(2)
        with col1:
            settings_payload["ACCEL_ONNX_MODEL_PATH"] = st.text_input(
                "ACCEL_ONNX_MODEL_PATH",
                value=saved_settings.get(
                    "ACCEL_ONNX_MODEL_PATH",
                    os.getenv("ACCEL_ONNX_MODEL_PATH", "./weights/realesrgan_x4.onnx"),
                ),
                key="cfg_ACCEL_ONNX_MODEL_PATH",
            )
            settings_payload["DIRECTML_DEVICE_ID"] = str(
                st.number_input(
                    "DIRECTML_DEVICE_ID",
                    min_value=0,
                    value=int(saved_settings.get("DIRECTML_DEVICE_ID", os.getenv("DIRECTML_DEVICE_ID", "0"))),
                    step=1,
                    key="cfg_DIRECTML_DEVICE_ID",
                )
            )
            settings_payload["ACCEL_TILE"] = str(
                st.number_input(
                    "ACCEL_TILE",
                    min_value=0,
                    value=int(saved_settings.get("ACCEL_TILE", os.getenv("ACCEL_TILE", "0"))),
                    step=1,
                    key="cfg_ACCEL_TILE",
                )
            )
        with col2:
            settings_payload["ACCEL_ONNX_MODEL_URL"] = st.text_input(
                "ACCEL_ONNX_MODEL_URL",
                value=saved_settings.get("ACCEL_ONNX_MODEL_URL", os.getenv("ACCEL_ONNX_MODEL_URL", "")),
                key="cfg_ACCEL_ONNX_MODEL_URL",
            )
            settings_payload["ACCEL_TILE_PAD"] = str(
                st.number_input(
                    "ACCEL_TILE_PAD",
                    min_value=0,
                    value=int(saved_settings.get("ACCEL_TILE_PAD", os.getenv("ACCEL_TILE_PAD", "8"))),
                    step=1,
                    key="cfg_ACCEL_TILE_PAD",
                )
            )
    # DeviantArt Settings
    st.markdown("##### DeviantArt Upload")
    col1, col2 = st.columns(2)
    with col1:
        settings_payload["DEVIANTART_USERNAME"] = st.text_input(
            "DEVIANTART_USERNAME",
            value=saved_settings.get("DEVIANTART_USERNAME", os.getenv("DEVIANTART_USERNAME", "")),
            key="cfg_DEVIANTART_USERNAME",
        )
    with col2:
        settings_payload["DEVIANTART_PUBLISH"] = st.selectbox(
            "DEVIANTART_PUBLISH",
            options=["true", "false"],
            index=0 if saved_settings.get("DEVIANTART_PUBLISH", "true").lower() == "true" else 1,
            key="cfg_DEVIANTART_PUBLISH",
        )

    # Prompt Enhancement
    st.markdown("##### Prompt Enhancement")
    col1, col2 = st.columns(2)
    with col1:
        settings_payload["ENABLE_PROMPT_ENHANCER"] = st.selectbox(
            "ENABLE_PROMPT_ENHANCER",
            options=["true", "false"],
            index=0 if saved_settings.get("ENABLE_PROMPT_ENHANCER", "true").lower() == "true" else 1,
            key="cfg_ENABLE_PROMPT_ENHANCER",
        )
    with col2:
        settings_payload["PROMPT_ENHANCER_MODEL"] = st.text_input(
            "PROMPT_ENHANCER_MODEL",
            value=saved_settings.get("PROMPT_ENHANCER_MODEL", os.getenv("PROMPT_ENHANCER_MODEL", "gpt-4-mini")),
            key="cfg_PROMPT_ENHANCER_MODEL",
        )

    st.markdown("##### Prompt Library Storage")
    prompt_backend_default = saved_settings.get(
        "PROMPT_LIBRARY_BACKEND",
        os.getenv("PROMPT_LIBRARY_BACKEND", "local_json"),
    ).strip().lower()
    prompt_backend_options = ["local_json", "github_gist"]
    settings_payload["PROMPT_LIBRARY_BACKEND"] = st.selectbox(
        "PROMPT_LIBRARY_BACKEND",
        options=prompt_backend_options,
        index=prompt_backend_options.index(prompt_backend_default)
        if prompt_backend_default in prompt_backend_options
        else 0,
        key="cfg_PROMPT_LIBRARY_BACKEND",
        help="Use github_gist to sync prompt library changes between local and Streamlit Cloud.",
    )

    if settings_payload["PROMPT_LIBRARY_BACKEND"] == "github_gist":
        col1, col2 = st.columns(2)
        with col1:
            settings_payload["PROMPT_LIBRARY_GIST_ID"] = st.text_input(
                "PROMPT_LIBRARY_GIST_ID",
                value=saved_settings.get("PROMPT_LIBRARY_GIST_ID", os.getenv("PROMPT_LIBRARY_GIST_ID", "")),
                key="cfg_PROMPT_LIBRARY_GIST_ID",
            )
        with col2:
            settings_payload["PROMPT_LIBRARY_GIST_FILENAME"] = st.text_input(
                "PROMPT_LIBRARY_GIST_FILENAME",
                value=saved_settings.get(
                    "PROMPT_LIBRARY_GIST_FILENAME",
                    os.getenv("PROMPT_LIBRARY_GIST_FILENAME", "prompt_library.json"),
                ),
                key="cfg_PROMPT_LIBRARY_GIST_FILENAME",
            )

    if st.button("Save Configuration", use_container_width=True, type="primary"):
        save_web_settings(settings_payload)
        apply_web_settings_to_env(settings_payload)
        st.success("Configuration saved to ai_art_bot/data/web_settings.json")
        st.rerun()

    return settings_payload


def _get_prompt_library_path() -> str:
    from pathlib import Path

    default_path = Path(__file__).resolve().parent / "prompt_library.json"
    configured_path = os.getenv("PROMPT_LIBRARY_PATH", str(default_path)).strip()
    path = Path(configured_path).expanduser()

    # One-time migration from legacy location.
    legacy_path = Path(__file__).resolve().parents[1] / "data" / "prompt_library.json"
    if not path.exists() and legacy_path.exists():
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(legacy_path.read_text(encoding="utf-8"), encoding="utf-8")
            logger.info(f"Migrated prompt library from {legacy_path} to {path}")
        except Exception as exc:
            logger.warning(f"Failed to migrate prompt library from legacy path: {exc}")

    return str(path)


def _prompt_library_backend() -> str:
    return os.getenv("PROMPT_LIBRARY_BACKEND", "local_json").strip().lower()


def _load_prompt_library_from_gist() -> dict[str, Any]:
    gist_id = os.getenv("PROMPT_LIBRARY_GIST_ID", "").strip()
    gist_filename = os.getenv("PROMPT_LIBRARY_GIST_FILENAME", "prompt_library.json").strip() or "prompt_library.json"
    github_token = os.getenv("GITHUB_TOKEN", "").strip()

    if not gist_id or not github_token:
        logger.warning("GitHub Gist prompt library backend requires PROMPT_LIBRARY_GIST_ID and GITHUB_TOKEN")
        return {}

    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {github_token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    try:
        response = requests.get(f"https://api.github.com/gists/{gist_id}", headers=headers, timeout=20)
        response.raise_for_status()
        payload = response.json()
        files = payload.get("files", {})
        file_payload = files.get(gist_filename, {}) if isinstance(files, dict) else {}
        content = file_payload.get("content", "{}") if isinstance(file_payload, dict) else "{}"
        parsed = json.loads(content or "{}")
        if not isinstance(parsed, dict):
            logger.warning("GitHub Gist prompt library content is not a JSON object")
            return {}
        return parsed
    except Exception as exc:
        logger.warning(f"Failed to load prompt library from GitHub Gist: {exc}")
        return {}


def _save_prompt_library_to_gist(library: dict[str, Any]) -> None:
    gist_id = os.getenv("PROMPT_LIBRARY_GIST_ID", "").strip()
    gist_filename = os.getenv("PROMPT_LIBRARY_GIST_FILENAME", "prompt_library.json").strip() or "prompt_library.json"
    github_token = os.getenv("GITHUB_TOKEN", "").strip()

    if not gist_id or not github_token:
        raise RuntimeError("GitHub Gist prompt library backend requires PROMPT_LIBRARY_GIST_ID and GITHUB_TOKEN")

    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {github_token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    payload = {
        "files": {
            gist_filename: {
                "content": json.dumps(library, indent=2)
            }
        }
    }
    response = requests.patch(
        f"https://api.github.com/gists/{gist_id}",
        headers=headers,
        json=payload,
        timeout=20,
    )
    response.raise_for_status()


def _load_prompt_library() -> dict[str, Any]:
    from pathlib import Path

    backend = _prompt_library_backend()
    if backend == "github_gist":
        return _load_prompt_library_from_gist()

    path = _get_prompt_library_path()
    if Path(path).exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}
    return {}


def _save_prompt_library(library: dict[str, Any]) -> None:
    from pathlib import Path

    backend = _prompt_library_backend()
    if backend == "github_gist":
        _save_prompt_library_to_gist(library)
        return

    path = _get_prompt_library_path()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(library, f, indent=2)


def _resolve_prompt_id_by_text(library: dict[str, Any], prompt_text: str) -> str | None:
    target = prompt_text.strip()
    if not target:
        return None
    for prompt_id, prompt_data in library.items():
        if str(prompt_data.get("text", "")).strip() == target:
            return str(prompt_id)
    return None


def _set_prompt_status(
    library: dict[str, Any],
    *,
    prompt_id: str | None = None,
    prompt_text: str | None = None,
    status: str,
) -> bool:
    resolved_id = prompt_id or _resolve_prompt_id_by_text(library, prompt_text or "")
    if not resolved_id or resolved_id not in library:
        return False
    item = library.get(resolved_id)
    if not isinstance(item, dict):
        return False
    item["status"] = status
    return True


def _maybe_persist_prompt_library(
    library: dict[str, Any],
    *,
    app_running: bool,
    auto_queue_enabled: bool,
) -> None:
    if app_running and auto_queue_enabled:
        _save_prompt_library(library)


def _next_eligible_prompt_for_auto_queue(library: dict[str, Any]) -> tuple[str, str] | None:
    sorted_prompts = sorted(library.items(), key=lambda x: int(x[0]))
    for prompt_id, prompt_data in sorted_prompts:
        prompt_text = str(prompt_data.get("text", "")).strip()
        status = str(prompt_data.get("status", "")).strip().lower()
        if not prompt_text:
            continue
        if status in {"queued", "running"}:
            continue
        return str(prompt_id), prompt_text
    return None


def _render_manual_prompt_panel(
    runner: Any,
    deployed_mode: bool,
    task_settings: dict[str, str],
) -> None:
    import streamlit as st

    library = _load_prompt_library()
    task_list = runner.get_queue(limit=500)
    queued_prompts = {task.get("prompt", "").strip() for task in task_list if task.get("prompt")}
    status_priority = {"running": 3, "queued": 2, "failure": 1, "success": 1}
    prompt_runtime_status: dict[str, str] = {}
    for task in task_list:
        prompt_text = str(task.get("prompt", "")).strip()
        task_status = str(task.get("status", "")).strip().lower()
        if not prompt_text or task_status not in status_priority:
            continue
        existing = prompt_runtime_status.get(prompt_text, "")
        if status_priority.get(task_status, 0) >= status_priority.get(existing, 0):
            prompt_runtime_status[prompt_text] = task_status

    for prompt_id, prompt_data in library.items():
        prompt_text = str(prompt_data.get("text", "")).strip()
        runtime_status = prompt_runtime_status.get(prompt_text)
        if runtime_status:
            prompt_data["status"] = runtime_status

    st.subheader("Prompt Management")
    backend = _prompt_library_backend()
    if backend == "github_gist":
        st.caption("Prompt library storage: GitHub Gist")
    else:
        st.caption(f"Prompt library storage: {_get_prompt_library_path()}")
    
    tab1, tab2 = st.tabs(["Prompt Library", "Create Task from Prompt"])
    
    # TAB 1: Prompt Library
    with tab1:
        st.markdown("#### Add to Prompt Library")

        def _next_prompt_id(existing_library: dict[str, Any]) -> str:
            numeric_ids = []
            for key in existing_library.keys():
                try:
                    numeric_ids.append(int(str(key)))
                except (TypeError, ValueError):
                    continue
            return str(max(numeric_ids, default=0) + 1)

        col1, col2 = st.columns([3, 1])
        with col1:
            new_prompt = st.text_area(
                "Add new prompt",
                height=100,
                placeholder="Describe your image...",
                key="new_prompt_input",
            )
        with col2:
            st.markdown("")
            st.markdown("")
            if st.button("Add to Library", use_container_width=True, type="primary"):
                if new_prompt.strip():
                    prompt_id = _next_prompt_id(library)
                    library[prompt_id] = {
                        "text": new_prompt.strip(),
                        "added_at": str(__import__("datetime").datetime.now()),
                    }
                    _save_prompt_library(library)
                    st.success(f"Prompt added to library")
                    st.rerun()
                else:
                    st.error("Prompt cannot be empty")

        st.markdown("#### Saved Prompts")
        if library:
            # Sort by ID descending (latest first)
            sorted_prompts = sorted(library.items(), key=lambda x: int(x[0]), reverse=True)

            if "selected_prompt_id" not in st.session_state:
                st.session_state["selected_prompt_id"] = ""
            if "editing_prompt_id" not in st.session_state:
                st.session_state["editing_prompt_id"] = ""

            st.caption("Select a row to edit or delete. Full prompt text is shown directly in the table row.")

            header_cols = st.columns([1, 2, 1, 8, 2])
            header_cols[0].markdown("**ID**")
            header_cols[1].markdown("**Added**")
            header_cols[2].markdown("**In Queue**")
            header_cols[3].markdown("**Prompt**")
            header_cols[4].markdown("**Actions**")

            for prompt_id, prompt_data in sorted_prompts:
                prompt_text = str(prompt_data.get("text", "")).strip()
                added_at = str(prompt_data.get("added_at", ""))[:10]
                in_queue = "Yes" if prompt_text in queued_prompts else "No"
                is_selected = st.session_state.get("selected_prompt_id", "") == prompt_id
                is_editing = st.session_state.get("editing_prompt_id", "") == prompt_id
                edit_key = f"saved_prompt_edit_{prompt_id}"

                row_cols = st.columns([1, 2, 1, 8, 2])
                row_cols[0].write(prompt_id)
                row_cols[1].write(added_at or "Unknown")
                row_cols[2].write(in_queue)

                if is_selected and is_editing:
                    if edit_key not in st.session_state:
                        st.session_state[edit_key] = prompt_text
                    row_cols[3].text_area(
                        f"Edit prompt #{prompt_id}",
                        key=edit_key,
                        height=100,
                        label_visibility="collapsed",
                    )
                else:
                    row_cols[3].write(prompt_text)

                if row_cols[4].button(
                    "Selected" if is_selected else "Select",
                    key=f"select_prompt_{prompt_id}",
                    use_container_width=True,
                    disabled=is_selected,
                ):
                    st.session_state["selected_prompt_id"] = prompt_id
                    st.session_state["editing_prompt_id"] = ""
                    st.rerun()

                if is_selected:
                    if is_editing:
                        if row_cols[4].button("Save", key=f"save_prompt_{prompt_id}", use_container_width=True):
                            updated_text = str(st.session_state.get(edit_key, "")).strip()
                            if not updated_text:
                                st.error("Prompt cannot be empty")
                            else:
                                library[prompt_id]["text"] = updated_text
                                _save_prompt_library(library)
                                st.session_state["editing_prompt_id"] = ""
                                st.success(f"Prompt #{prompt_id} updated")
                                st.rerun()
                        if row_cols[4].button("Cancel", key=f"cancel_prompt_{prompt_id}", use_container_width=True):
                            st.session_state["editing_prompt_id"] = ""
                            st.rerun()
                    else:
                        if row_cols[4].button("Edit", key=f"edit_prompt_{prompt_id}", use_container_width=True):
                            st.session_state[edit_key] = prompt_text
                            st.session_state["editing_prompt_id"] = prompt_id
                            st.rerun()
                        if row_cols[4].button(
                            "Delete",
                            key=f"delete_prompt_{prompt_id}",
                            use_container_width=True,
                        ):
                            library.pop(prompt_id, None)
                            if st.session_state.get("selected_prompt_id") == prompt_id:
                                st.session_state["selected_prompt_id"] = ""
                                st.session_state["editing_prompt_id"] = ""
                            _save_prompt_library(library)
                            st.success(f"Prompt #{prompt_id} deleted")
                            st.rerun()

                st.divider()
        else:
            st.info("No prompts in library yet. Add one above!")

    # TAB 2: Create Task from Prompt
    with tab2:
        st.markdown("#### Create Task from Prompt")
        
        # Select prompt from library or enter manually
        prompt_source = st.radio(
            "Prompt source",
            options=["from_library", "manual"],
            format_func=lambda x: "From Library" if x == "from_library" else "Enter Manually",
            horizontal=True,
        )

        manual_prompt = ""
        if prompt_source == "from_library" and library:
            sorted_prompts = sorted(library.items(), key=lambda x: int(x[0]), reverse=True)
            prompt_options = {f"#{pid}: {data.get('text', '')[:60]}...": data.get('text', '') 
                            for pid, data in sorted_prompts}
            selected = st.selectbox("Select a prompt", options=list(prompt_options.keys()))
            manual_prompt = prompt_options[selected]
        elif prompt_source == "from_library":
            st.warning("No prompts in library. Add some prompts first!")
        else:
            manual_prompt = st.text_area(
                "Enter prompt manually",
                height=140,
                placeholder="Describe your image...",
                key="manual_prompt_input",
            )

        col1, col2 = st.columns(2)
        with col1:
            prompt_mode = st.radio(
                "Prompt handling",
                options=["as_is", "reformat"],
                format_func=lambda value: "Use as-is" if value == "as_is" else "Reformat prompt",
                horizontal=True,
            )
        with col2:
            pipeline_mode = st.selectbox(
                "Task mode",
                options=["generate_only", "generate_upscale", "full", "upscale_only", "upload_only", "upscale_upload"],
                index=2,
            )

        if "manual_auto_queue_mode" not in st.session_state:
            st.session_state["manual_auto_queue_mode"] = False

        auto_queue_mode = st.checkbox(
            "Auto-queue next eligible prompt after completion",
            key="manual_auto_queue_mode",
            help="When enabled, each successful/failed run auto-enqueues the next prompt-library item that is not queued/running.",
        )

        source_image_path = ""
        source_upscaled_path = ""
        negative_prompt = st.text_area(
            "Negative prompt (optional, local_sd only)",
            value=(task_settings.get("LOCAL_NEGATIVE_PROMPT") or ""),
            height=100,
            key="manual_task_negative_prompt",
            help="Overrides LOCAL_NEGATIVE_PROMPT for this queued task only.",
        )
        if pipeline_mode in {"upscale_only", "upscale_upload", "upload_only"}:
            source_image_path = st.text_input("Source image path (needed for upscale/upload modes)")
        if pipeline_mode in {"upload_only"}:
            source_upscaled_path = st.text_input("Source upscaled path (optional, preferred for upload)")

        if deployed_mode:
            start_immediately = False
            st.caption("Task execution controls are disabled in deployed prompt-management mode.")
        else:
            default_auto_run = _is_true(task_settings.get("AUTO_QUEUE_ON_ADD", os.getenv("AUTO_QUEUE_ON_ADD", "false")))
            start_immediately = st.checkbox(
                "Auto-run immediately after queueing",
                value=default_auto_run,
                key="manual_auto_run_after_queue",
                help="Use Save Auto Queue Preference to keep this as the default behavior.",
            )
            if st.button("Save Auto Queue Preference", key="save_auto_queue_preference"):
                merged_settings = load_web_settings()
                merged_settings["AUTO_QUEUE_ON_ADD"] = "true" if start_immediately else "false"
                save_web_settings(merged_settings)
                apply_web_settings_to_env(merged_settings)
                task_settings["AUTO_QUEUE_ON_ADD"] = merged_settings["AUTO_QUEUE_ON_ADD"]
                st.success("Auto queue preference saved")

        required_secrets, missing_secrets = _required_secrets_for_mode(st, pipeline_mode, task_settings)

        if required_secrets:
            st.caption("Required secrets for selected mode: " + ", ".join(required_secrets))

        if missing_secrets:
            st.error("Missing required secrets: " + ", ".join(missing_secrets))
        else:
            st.success("Credential validation passed for selected mode")

        can_submit = not missing_secrets

        if st.button("Add Task To Queue", type="primary", use_container_width=True, disabled=not can_submit):
            if not manual_prompt.strip():
                st.error("Prompt is required")
            else:
                task_settings_payload = dict(task_settings)
                if negative_prompt.strip():
                    task_settings_payload["LOCAL_NEGATIVE_PROMPT"] = negative_prompt.strip()
                else:
                    task_settings_payload.pop("LOCAL_NEGATIVE_PROMPT", None)

                selected_library_prompt_id: str | None = None
                if prompt_source == "from_library":
                    selected_library_prompt_id = _resolve_prompt_id_by_text(library, manual_prompt)

                task_id = runner.enqueue_manual_task(
                    prompt=manual_prompt,
                    prompt_mode=prompt_mode,
                    pipeline_mode=pipeline_mode,
                    settings=task_settings_payload,
                    source_image_path=source_image_path or None,
                    source_upscaled_path=source_upscaled_path or None,
                )

                if selected_library_prompt_id:
                    _set_prompt_status(library, prompt_id=selected_library_prompt_id, status="queued")
                    _maybe_persist_prompt_library(
                        library,
                        app_running=bool(start_immediately),
                        auto_queue_enabled=auto_queue_mode,
                    )
                st.success(f"Task #{task_id} added to queue")

                if start_immediately:
                    if selected_library_prompt_id:
                        _set_prompt_status(library, prompt_id=selected_library_prompt_id, status="running")
                        _maybe_persist_prompt_library(
                            library,
                            app_running=True,
                            auto_queue_enabled=auto_queue_mode,
                        )
                    with st.spinner("Processing task..."):
                        result = runner.process_task(task_id)
                    completion_status = "success" if result.get("status") == "success" else "failure"
                    if selected_library_prompt_id:
                        _set_prompt_status(library, prompt_id=selected_library_prompt_id, status=completion_status)
                        _maybe_persist_prompt_library(
                            library,
                            app_running=True,
                            auto_queue_enabled=auto_queue_mode,
                        )

                    if auto_queue_mode:
                        next_prompt = _next_eligible_prompt_for_auto_queue(library)
                        if next_prompt:
                            next_prompt_id, next_prompt_text = next_prompt
                            next_task_id = runner.enqueue_manual_task(
                                prompt=next_prompt_text,
                                prompt_mode=prompt_mode,
                                pipeline_mode=pipeline_mode,
                                settings=task_settings_payload,
                                source_image_path=source_image_path or None,
                                source_upscaled_path=source_upscaled_path or None,
                            )
                            _set_prompt_status(library, prompt_id=next_prompt_id, status="queued")
                            _maybe_persist_prompt_library(
                                library,
                                app_running=True,
                                auto_queue_enabled=True,
                            )
                            st.info(f"Auto-queued next prompt as task #{next_task_id}")
                    if result.get("status") == "success":
                        st.success(f"Task #{task_id} finished successfully")
                    elif result.get("status") == "no_info":
                        st.warning(result.get("message", "No info"))
                    else:
                        st.error(f"Task #{task_id} failed: {result.get('error', 'unknown error')}")
                st.rerun()


def _render_queue_status_panel(runner: "PipelineRunner", deployed_mode: bool) -> None:
    import streamlit as st

    st.subheader("Task Queue Status")
    library = _load_prompt_library()
    if "queue_panel_auto_queue" not in st.session_state:
        st.session_state["queue_panel_auto_queue"] = False

    auto_queue_mode = st.checkbox(
        "Auto-queue next eligible prompt after completion (queue panel)",
        key="queue_panel_auto_queue",
    )

    header_cols = st.columns([1, 2, 2, 2, 2])
    header_cols[0].markdown("**ID**")
    header_cols[1].markdown("**Status**")
    header_cols[2].markdown("**Mode**")
    header_cols[3].markdown("**Created**")
    header_cols[4].markdown("**Actions**")

    tasks = runner.get_queue(limit=200)
    if not tasks:
        st.info("No queued tasks yet")
        return

    if not deployed_mode:
        controls = st.columns(2)
        if controls[0].button("Start Next Queued Task"):
            next_queued_task = next((task for task in tasks if str(task.get("status", "")).strip().lower() == "queued"), None)
            next_prompt_id: str | None = None
            if next_queued_task:
                next_prompt_text = str(next_queued_task.get("prompt", "")).strip()
                next_prompt_id = _resolve_prompt_id_by_text(library, next_prompt_text)
                if next_prompt_id:
                    _set_prompt_status(library, prompt_id=next_prompt_id, status="running")
                    _maybe_persist_prompt_library(
                        library,
                        app_running=True,
                        auto_queue_enabled=auto_queue_mode,
                    )
            with st.spinner("Starting next queued task..."):
                result = runner.process_next_queued_task()

            completion_status = "success" if result.get("status") == "success" else "failure"
            if next_prompt_id:
                _set_prompt_status(library, prompt_id=next_prompt_id, status=completion_status)
                _maybe_persist_prompt_library(
                    library,
                    app_running=True,
                    auto_queue_enabled=auto_queue_mode,
                )

            if auto_queue_mode:
                next_prompt = _next_eligible_prompt_for_auto_queue(library)
                if next_prompt:
                    auto_prompt_id, auto_prompt_text = next_prompt
                    auto_task_id = runner.enqueue_manual_task(
                        prompt=auto_prompt_text,
                        prompt_mode="as_is",
                        pipeline_mode="full",
                        settings=None,
                        source_image_path=None,
                        source_upscaled_path=None,
                    )
                    _set_prompt_status(library, prompt_id=auto_prompt_id, status="queued")
                    _maybe_persist_prompt_library(
                        library,
                        app_running=True,
                        auto_queue_enabled=True,
                    )
                    st.info(f"Auto-queued next prompt as task #{auto_task_id}")
            if result.get("status") == "success":
                st.success(f"Task #{result.get('task_id')} finished")
            elif result.get("status") == "no_info":
                st.warning(result.get("message", "No info"))
            else:
                st.error(f"Task failed: {result.get('error', 'unknown error')}")
            st.rerun()
        if controls[1].button("Refresh Queue"):
            st.rerun()

    for task in tasks:
        task_id = int(task["id"])
        cols = st.columns([1, 2, 2, 2, 2])
        cols[0].write(task_id)
        cols[1].write(task.get("status", "no_info"))
        cols[2].write(task.get("pipeline_mode", "no_info"))
        cols[3].write(task.get("created_at", "no_info"))
        if not deployed_mode and task.get("status") == "queued":
            if cols[4].button(f"Run #{task_id}", key=f"run_task_{task_id}"):
                with st.spinner(f"Running task #{task_id}..."):
                    result = runner.process_task(task_id)
                if result.get("status") == "success":
                    st.success(f"Task #{task_id} finished")
                else:
                    st.error(f"Task #{task_id} failed: {result.get('error', 'unknown error')}")
                st.rerun()
        else:
            cols[4].write("-")

        if task.get("status") == "failure" and task.get("error_message"):
            st.caption(f"Task #{task_id} error: {task['error_message']}")


def _render_deployed_prompt_library_settings(saved_settings: dict[str, str]) -> dict[str, str]:
    import streamlit as st

    st.subheader("Prompt Library Settings")
    st.caption("Optional: configure GitHub Gist sync for prompt_library.json in deployed mode.")

    prompt_settings: dict[str, str] = {}
    prompt_backend_default = saved_settings.get(
        "PROMPT_LIBRARY_BACKEND",
        os.getenv("PROMPT_LIBRARY_BACKEND", "local_json"),
    ).strip().lower()
    prompt_backend_options = ["local_json", "github_gist"]
    prompt_settings["PROMPT_LIBRARY_BACKEND"] = st.selectbox(
        "PROMPT_LIBRARY_BACKEND",
        options=prompt_backend_options,
        index=prompt_backend_options.index(prompt_backend_default)
        if prompt_backend_default in prompt_backend_options
        else 0,
        key="deployed_PROMPT_LIBRARY_BACKEND",
    )

    if prompt_settings["PROMPT_LIBRARY_BACKEND"] == "github_gist":
        col1, col2 = st.columns(2)
        with col1:
            prompt_settings["PROMPT_LIBRARY_GIST_ID"] = st.text_input(
                "PROMPT_LIBRARY_GIST_ID",
                value=saved_settings.get("PROMPT_LIBRARY_GIST_ID", os.getenv("PROMPT_LIBRARY_GIST_ID", "")),
                key="deployed_PROMPT_LIBRARY_GIST_ID",
            )
        with col2:
            prompt_settings["PROMPT_LIBRARY_GIST_FILENAME"] = st.text_input(
                "PROMPT_LIBRARY_GIST_FILENAME",
                value=saved_settings.get(
                    "PROMPT_LIBRARY_GIST_FILENAME",
                    os.getenv("PROMPT_LIBRARY_GIST_FILENAME", "prompt_library.json"),
                ),
                key="deployed_PROMPT_LIBRARY_GIST_FILENAME",
            )

    if st.button("Save Prompt Library Settings", use_container_width=True, key="save_deployed_prompt_library"):
        merged_settings = dict(saved_settings)
        merged_settings.update(prompt_settings)
        save_web_settings(merged_settings)
        apply_web_settings_to_env(merged_settings)
        st.success("Prompt library settings saved")
        st.rerun()

    return prompt_settings


def run_streamlit_app() -> None:
    import streamlit as st

    setup_logging()
    saved_settings = _load_and_apply_saved_settings()
    deployed_mode = _is_deployed_streamlit()

    st.set_page_config(page_title="AI Art Bot Queue", layout="wide")
    st.title("AI Art Bot Queue")
    st.caption("Prompt library and manual prompt task creation.")

    if deployed_mode:
        runner = _build_cloud_prompt_runner()
        st.info("Deployed mode: prompt management only")
        with st.expander("Prompt Library", expanded=False):
            _render_deployed_prompt_library_settings(saved_settings)
        active_settings = _cloud_runtime_settings(saved_settings)
    else:
        try:
            runner = _build_runner()
        except ModuleNotFoundError as exc:
            # Fall back to prompt-only mode when optional local pipeline deps are unavailable.
            deployed_mode = True
            runner = _build_cloud_prompt_runner()
            logger.warning(f"Falling back to deployed prompt-only mode due to missing module: {exc}")
            st.warning(
                "Full local pipeline dependencies are missing in this environment. "
                "Switched to prompt-management-only mode."
            )
            with st.expander("Prompt Library", expanded=False):
                _render_deployed_prompt_library_settings(saved_settings)
            active_settings = _cloud_runtime_settings(saved_settings)
            _render_manual_prompt_panel(runner=runner, deployed_mode=deployed_mode, task_settings=active_settings)
            return
        with st.expander("Global Settings", expanded=False):
            active_settings = _render_settings_editor(saved_settings)
        with st.expander("Secrets Validation", expanded=True):
            _render_secret_status(st)

    _render_manual_prompt_panel(runner=runner, deployed_mode=deployed_mode, task_settings=active_settings)
    if not deployed_mode:
        _render_queue_status_panel(runner=runner, deployed_mode=deployed_mode)


def main() -> None:
    if _is_streamlit_execution():
        run_streamlit_app()
        return

    setup_logging()
    parser = argparse.ArgumentParser(description="Automated AI art generation pipeline")
    parser.add_argument(
        "command",
        choices=["run_once", "run_loop", "generate_only", "serve", "auth_deviantart", "download_model"],
        help="Pipeline command to execute",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=int(os.getenv("RUN_INTERVAL_MINUTES", "60")),
        help="Loop interval in minutes for run_loop",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Dashboard host for the serve command",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Dashboard port for the serve command",
    )
    parser.add_argument(
        "--model-id",
        default=os.getenv("LOCAL_MODEL_ID", "Lykon/dreamshaper-8"),
        help="Model id for local SD download (used with download_model)",
    )
    parser.add_argument(
        "--local-dir",
        default=os.getenv("LOCAL_MODELS_DIR", "./models"),
        help="Destination directory for downloaded local model",
    )
    args = parser.parse_args()

    if args.command == "serve":
        import uvicorn

        uvicorn.run(build_dashboard(), host=args.host, port=args.port)
        return

    if args.command == "auth_deviantart":
        from uploader.deviantart_upload import bootstrap_tokens

        payload = bootstrap_tokens()
        has_refresh = bool(payload.get("refresh_token"))
        logger.info(f"DeviantArt authorization complete. refresh_token_saved={has_refresh}")
        return

    if args.command == "download_model":
        from generator.image_generator import download_local_model

        model_path = download_local_model(model_id=args.model_id, local_dir=args.local_dir or None)
        logger.info(f"Local model ready at: {model_path}")
        return

    get_settings().validate(args.command)
    runner = _build_runner()
    if args.command == "run_once":
        runner.run_once(mode="full")
    elif args.command == "run_loop":
        runner.run_loop(interval_minutes=args.interval)
    elif args.command == "generate_only":
        runner.run_once(mode="generate_only")


if __name__ == "__main__":
    main()