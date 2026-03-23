import os
from dataclasses import dataclass
from pathlib import Path
import json


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
GENERATED_DIR = DATA_DIR / "generated"
UPSCALED_DIR = DATA_DIR / "upscaled"
PROMPTS_FILE = BASE_DIR / "prompts.txt"
DB_PATH = DATA_DIR / "images.db"
DEVIANTART_TOKENS_PATH = DATA_DIR / "deviantart_tokens.json"
WEB_SETTINGS_PATH = DATA_DIR / "web_settings.json"

WEB_CONFIG_KEYS = [
    "IMAGE_BACKEND",
    "LOCAL_MODEL_ID",
    "LOCAL_MODEL_PATH",
    "LOCAL_MODEL_USE_OPENVINO",
    "LOCAL_IMAGE_WIDTH",
    "LOCAL_IMAGE_HEIGHT",
    "LOCAL_NUM_INFERENCE_STEPS",
    "LOCAL_GUIDANCE_SCALE",
    "LOCAL_SEED",
    "UPSCALER_BACKEND",
    "REPLICATE_UPSCALER_MODEL",
    "REPLICATE_UPSCALER_SCALE",
    "REALESRGAN_MODEL_NAME",
    "REALESRGAN_OUTSCALE",
    "REALESRGAN_TILE",
    "REALESRGAN_TILE_PAD",
    "REALESRGAN_PRE_PAD",
    "REALESRGAN_MAX_INPUT_SIDE",
    "REALESRGAN_MAX_INPUT_PIXELS",
    "OPENVINO_DEVICE",
    "DEVIANTART_USERNAME",
    "DEVIANTART_PUBLISH",
    "PROMPT_ENHANCER_MODEL",
    "ENABLE_PROMPT_ENHANCER",
]

SECRET_ENV_KEYS = [
    "OPENAI_API_KEY",
    "REPLICATE_API_TOKEN",
    "DEVIANTART_CLIENT_ID",
    "DEVIANTART_CLIENT_SECRET",
    "DEVIANTART_REFRESH_TOKEN",
    "DEVIANTART_ACCESS_TOKEN",
]


@dataclass(frozen=True)
class Settings:
    image_backend: str
    openai_api_key: str
    local_model_id: str
    local_model_path: str
    replicate_api_token: str
    deviantart_client_id: str
    deviantart_client_secret: str
    deviantart_refresh_token: str
    deviantart_access_token: str
    deviantart_username: str

    def validate(self, command: str) -> None:
        missing: list[str] = []
        upscaler_backend = os.getenv("UPSCALER_BACKEND", "realesrgan").strip().lower()
        has_cached_deviantart_token = has_local_deviantart_tokens()
        image_backend = (self.image_backend or "openai").strip().lower()
        if command in {"run_once", "run_loop", "generate_only"} and image_backend == "openai" and not self.openai_api_key:
            missing.append("OPENAI_API_KEY")
        if command in {"run_once", "run_loop", "generate_only"} and image_backend in {"local", "local_sd", "sd15"}:
            if not (self.local_model_path or self.local_model_id):
                missing.append("LOCAL_MODEL_PATH or LOCAL_MODEL_ID")
        if command in {"run_once", "run_loop"} and upscaler_backend == "replicate" and not self.replicate_api_token:
            missing.append("REPLICATE_API_TOKEN")
        if command in {"run_once", "run_loop"} and not (
            self.deviantart_access_token or self.deviantart_refresh_token or has_cached_deviantart_token
        ):
            missing.append("DEVIANTART_ACCESS_TOKEN or DEVIANTART_REFRESH_TOKEN")
        if command in {"run_once", "run_loop"}:
            if not self.deviantart_client_id:
                missing.append("DEVIANTART_CLIENT_ID")
            if not self.deviantart_client_secret:
                missing.append("DEVIANTART_CLIENT_SECRET")
        if missing:
            joined = ", ".join(missing)
            raise RuntimeError(f"Missing required configuration: {joined}")


def ensure_directories() -> None:
    for directory in (DATA_DIR, GENERATED_DIR, UPSCALED_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def load_web_settings() -> dict[str, str]:
    ensure_directories()
    if not WEB_SETTINGS_PATH.exists():
        return {}
    try:
        payload = json.loads(WEB_SETTINGS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    return {str(key): str(value) for key, value in payload.items() if value is not None}


def save_web_settings(settings: dict[str, str]) -> None:
    ensure_directories()
    cleaned = {str(key): str(value) for key, value in settings.items() if value is not None}
    WEB_SETTINGS_PATH.write_text(json.dumps(cleaned, indent=2), encoding="utf-8")


def apply_web_settings_to_env(settings: dict[str, str]) -> None:
    for key, value in settings.items():
        os.environ[str(key)] = str(value)


def has_local_deviantart_tokens() -> bool:
    if not DEVIANTART_TOKENS_PATH.exists():
        return False
    try:
        payload = json.loads(DEVIANTART_TOKENS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return False
    return bool(payload.get("refresh_token") or payload.get("access_token"))


def get_settings() -> Settings:
    ensure_directories()
    return Settings(
        image_backend=os.getenv("IMAGE_BACKEND", "openai"),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        local_model_id=os.getenv("LOCAL_MODEL_ID", "runwayml/stable-diffusion-v1-5"),
        local_model_path=os.getenv("LOCAL_MODEL_PATH", ""),
        replicate_api_token=os.getenv("REPLICATE_API_TOKEN", ""),
        deviantart_client_id=os.getenv("DEVIANTART_CLIENT_ID", ""),
        deviantart_client_secret=os.getenv("DEVIANTART_CLIENT_SECRET", ""),
        deviantart_refresh_token=os.getenv("DEVIANTART_REFRESH_TOKEN", ""),
        deviantart_access_token=os.getenv("DEVIANTART_ACCESS_TOKEN", ""),
        deviantart_username=os.getenv("DEVIANTART_USERNAME", "me"),
    )