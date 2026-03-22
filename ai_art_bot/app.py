import argparse
import os

import uvicorn
from fastapi import FastAPI, HTTPException

from database.db import list_images
from generator.image_generator import download_local_model
from scheduler.scheduler import PipelineRunner
from uploader.deviantart_upload import bootstrap_tokens
from utils.config import get_settings
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

    return app


def main() -> None:
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
        default=os.getenv("LOCAL_MODELS_DIR", ""),
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