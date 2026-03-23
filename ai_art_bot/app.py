import argparse
import os
import sys
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException

from database.db import get_task, list_images, list_tasks
from generator.image_generator import download_local_model
from scheduler.scheduler import PipelineRunner
from uploader.deviantart_upload import bootstrap_tokens
from utils.config import (
    ACCEL_DEFAULT_ONNX_MODEL_PATH,
    MODELS_DIR,
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


def build_dashboard() -> FastAPI:
    app = FastAPI(title="AI Art Bot", version="0.1.0")
    runner = PipelineRunner()

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
                value=saved_settings.get("LOCAL_MODEL_ID", os.getenv("LOCAL_MODEL_ID", "runwayml/stable-diffusion-v1-5")),
                key="cfg_LOCAL_MODEL_ID",
            )
            settings_payload["LOCAL_MODEL_PATH"] = st.text_input(
                "LOCAL_MODEL_PATH (optional)",
                value=saved_settings.get("LOCAL_MODEL_PATH", os.getenv("LOCAL_MODEL_PATH", str(MODELS_DIR / "stable-diffusion-v1-5"))),
                key="cfg_LOCAL_MODEL_PATH",
            )
        with col2:
            settings_payload["LOCAL_IMAGE_WIDTH"] = st.number_input(
                "LOCAL_IMAGE_WIDTH",
                value=int(saved_settings.get("LOCAL_IMAGE_WIDTH", os.getenv("LOCAL_IMAGE_WIDTH", "768"))),
                key="cfg_LOCAL_IMAGE_WIDTH",
            )
            settings_payload["LOCAL_IMAGE_HEIGHT"] = st.number_input(
                "LOCAL_IMAGE_HEIGHT",
                value=int(saved_settings.get("LOCAL_IMAGE_HEIGHT", os.getenv("LOCAL_IMAGE_HEIGHT", "1344"))),
                key="cfg_LOCAL_IMAGE_HEIGHT",
            )

        col1, col2 = st.columns(2)
        with col1:
            settings_payload["LOCAL_NUM_INFERENCE_STEPS"] = st.number_input(
                "LOCAL_NUM_INFERENCE_STEPS",
                value=int(saved_settings.get("LOCAL_NUM_INFERENCE_STEPS", os.getenv("LOCAL_NUM_INFERENCE_STEPS", "24"))),
                key="cfg_LOCAL_NUM_INFERENCE_STEPS",
            )
            settings_payload["LOCAL_GUIDANCE_SCALE"] = st.text_input(
                "LOCAL_GUIDANCE_SCALE",
                value=saved_settings.get("LOCAL_GUIDANCE_SCALE", os.getenv("LOCAL_GUIDANCE_SCALE", "7.0")),
                key="cfg_LOCAL_GUIDANCE_SCALE",
            )
        with col2:
            settings_payload["LOCAL_SEED"] = st.text_input(
                "LOCAL_SEED",
                value=saved_settings.get("LOCAL_SEED", os.getenv("LOCAL_SEED", "-1")),
                key="cfg_LOCAL_SEED",
            )
            settings_payload["LOCAL_MODEL_USE_OPENVINO"] = st.selectbox(
                "LOCAL_MODEL_USE_OPENVINO",
                options=["false", "true"],
                index=1 if saved_settings.get("LOCAL_MODEL_USE_OPENVINO", "false").lower() == "true" else 0,
                key="cfg_LOCAL_MODEL_USE_OPENVINO",
            )

    # Upscaler Backend Selection
    st.markdown("##### Upscaler Backend")
    upscaler_default = saved_settings.get("UPSCALER_BACKEND", os.getenv("UPSCALER_BACKEND", "realesrgan")).lower()
    upscaler_options = ["realesrgan", "replicate", "directml", "openvino"]
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
                    os.getenv("ACCEL_ONNX_MODEL_PATH", str(ACCEL_DEFAULT_ONNX_MODEL_PATH)),
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
    elif upscaler_backend == "openvino":
        col1, col2 = st.columns(2)
        with col1:
            settings_payload["ACCEL_ONNX_MODEL_PATH"] = st.text_input(
                "ACCEL_ONNX_MODEL_PATH",
                value=saved_settings.get(
                    "ACCEL_ONNX_MODEL_PATH",
                    os.getenv("ACCEL_ONNX_MODEL_PATH", str(ACCEL_DEFAULT_ONNX_MODEL_PATH)),
                ),
                key="cfg_ACCEL_ONNX_MODEL_PATH",
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
        settings_payload["OPENVINO_DEVICE"] = st.text_input(
            "OPENVINO_DEVICE",
            value=saved_settings.get("OPENVINO_DEVICE", os.getenv("OPENVINO_DEVICE", "GPU_FP32")),
            key="cfg_OPENVINO_DEVICE",
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

    if st.button("Save Configuration", use_container_width=True, type="primary"):
        save_web_settings(settings_payload)
        apply_web_settings_to_env(settings_payload)
        st.success("Configuration saved to ai_art_bot/data/web_settings.json")
        st.rerun()

    return settings_payload


def _get_prompt_library_path() -> str:
    from pathlib import Path
    prompt_lib_path = Path(__file__).resolve().parents[1] / "data" / "prompt_library.json"
    return str(prompt_lib_path)


def _load_prompt_library() -> dict[str, Any]:
    import json
    from pathlib import Path
    path = _get_prompt_library_path()
    if Path(path).exists():
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_prompt_library(library: dict[str, Any]) -> None:
    import json
    from pathlib import Path
    path = _get_prompt_library_path()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(library, f, indent=2)


def _render_manual_prompt_panel(
    runner: PipelineRunner,
    deployed_mode: bool,
    task_settings: dict[str, str],
) -> None:
    import streamlit as st

    library = _load_prompt_library()
    task_list = runner.get_queue(limit=500)
    queued_prompts = {task.get("prompt", "").strip() for task in task_list if task.get("prompt")}

    st.subheader("Prompt Management")
    
    tab1, tab2 = st.tabs(["Prompt Library", "Create Task from Prompt"])
    
    # TAB 1: Prompt Library
    with tab1:
        st.markdown("#### Add to Prompt Library")
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
                    prompt_id = len(library) + 1
                    library[str(prompt_id)] = {
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
            
            # Build table data
            table_data = []
            for prompt_id, prompt_data in sorted_prompts:
                in_queue = "✓ Yes" if prompt_data.get("text", "").strip() in queued_prompts else "✗ No"
                table_data.append({
                    "ID": prompt_id,
                    "Prompt": prompt_data.get("text", "")[:80] + "..." if len(prompt_data.get("text", "")) > 80 else prompt_data.get("text", ""),
                    "Added": prompt_data.get("added_at", "")[:10],
                    "In Queue": in_queue,
                })
            
            st.dataframe(table_data, use_container_width=True, hide_index=True)
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

        source_image_path = ""
        source_upscaled_path = ""
        if pipeline_mode in {"upscale_only", "upscale_upload", "upload_only"}:
            source_image_path = st.text_input("Source image path (needed for upscale/upload modes)")
        if pipeline_mode in {"upload_only"}:
            source_upscaled_path = st.text_input("Source upscaled path (optional, preferred for upload)")

        start_immediately_default = True if deployed_mode else False
        start_immediately = st.checkbox("Start immediately after queueing", value=start_immediately_default)

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
                task_id = runner.enqueue_manual_task(
                    prompt=manual_prompt,
                    prompt_mode=prompt_mode,
                    pipeline_mode=pipeline_mode,
                    settings=task_settings,
                    source_image_path=source_image_path or None,
                    source_upscaled_path=source_upscaled_path or None,
                )
                st.success(f"Task #{task_id} added to queue")

                if start_immediately:
                    with st.spinner("Processing task..."):
                        result = runner.process_task(task_id)
                    if result.get("status") == "success":
                        st.success(f"Task #{task_id} finished successfully")
                    elif result.get("status") == "no_info":
                        st.warning(result.get("message", "No info"))
                    else:
                        st.error(f"Task #{task_id} failed: {result.get('error', 'unknown error')}")
                st.rerun()


def _render_queue_status_panel(runner: PipelineRunner, deployed_mode: bool) -> None:
    import streamlit as st

    st.subheader("Task Queue Status")

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
            with st.spinner("Starting next queued task..."):
                result = runner.process_next_queued_task()
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


def run_streamlit_app() -> None:
    import streamlit as st

    setup_logging()
    saved_settings = _load_and_apply_saved_settings()
    runner = PipelineRunner()
    deployed_mode = _is_deployed_streamlit()

    st.set_page_config(page_title="AI Art Bot Queue", layout="wide")
    st.title("AI Art Bot Queue")
    st.caption("Queue manual prompts, choose pipeline mode, and track task status.")

    if deployed_mode:
        st.info("Deployed mode: manual prompt queue only")
        active_settings = saved_settings
        _render_secret_status(st)
    else:
        with st.expander("Global Settings", expanded=False):
            active_settings = _render_settings_editor(saved_settings)
        with st.expander("Secrets Validation", expanded=True):
            _render_secret_status(st)

    _render_manual_prompt_panel(runner=runner, deployed_mode=deployed_mode, task_settings=active_settings)
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
        default=os.getenv("LOCAL_MODEL_ID", "runwayml/stable-diffusion-v1-5"),
        help="Model id for local SD download (used with download_model)",
    )
    parser.add_argument(
        "--local-dir",
        default=os.getenv("LOCAL_MODELS_DIR", str(MODELS_DIR)),
        help="Destination directory for downloaded local model",
    )
    args = parser.parse_args()

    if args.command == "serve":
        uvicorn.run(build_dashboard(), host=args.host, port=args.port)
        return

    if args.command == "auth_deviantart":
        payload = bootstrap_tokens()
        has_refresh = bool(payload.get("refresh_token"))
        logger.info(f"DeviantArt authorization complete. refresh_token_saved={has_refresh}")
        return

    if args.command == "download_model":
        model_path = download_local_model(model_id=args.model_id, local_dir=args.local_dir or None)
        logger.info(f"Local model ready at: {model_path}")
        return

    get_settings().validate(args.command)
    runner = PipelineRunner()
    if args.command == "run_once":
        runner.run_once(mode="full")
    elif args.command == "run_loop":
        runner.run_loop(interval_minutes=args.interval)
    elif args.command == "generate_only":
        runner.run_once(mode="generate_only")


if __name__ == "__main__":
    main()