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


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
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
        if command in {"run_once", "run_loop", "generate_only"} and not self.openai_api_key:
            missing.append("OPENAI_API_KEY")
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
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        replicate_api_token=os.getenv("REPLICATE_API_TOKEN", ""),
        deviantart_client_id=os.getenv("DEVIANTART_CLIENT_ID", ""),
        deviantart_client_secret=os.getenv("DEVIANTART_CLIENT_SECRET", ""),
        deviantart_refresh_token=os.getenv("DEVIANTART_REFRESH_TOKEN", ""),
        deviantart_access_token=os.getenv("DEVIANTART_ACCESS_TOKEN", ""),
        deviantart_username=os.getenv("DEVIANTART_USERNAME", "me"),
    )