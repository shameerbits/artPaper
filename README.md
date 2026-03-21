# artPaper

Simple local Python pipeline for selecting a Stable Diffusion-style prompt (from APIs/datasets plus a local generator), generating an image with OpenAI, upscaling it with local Real-ESRGAN (or Replicate), publishing it to DeviantArt, and recording metadata in SQLite.

## Project structure

```text
ai_art_bot/
	app.py
	requirements.txt
	scraper/
		prompt_provider.py
	generator/
		image_generator.py
	upscaler/
		image_upscaler.py
	uploader/
		deviantart_upload.py
	database/
		db.py
	scheduler/
		scheduler.py
	utils/
		config.py
		logger.py
	data/
		generated/
		upscaled/
	prompts.txt
```

## Setup

1. Create and activate a Python 3.11+ virtual environment.
2. Install dependencies.
3. Export the required environment variables.
4. Run the CLI.

```bash
cd ai_art_bot
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Environment variables

```bash
export OPENAI_API_KEY="..."
export REPLICATE_API_TOKEN="..."
export DEVIANTART_CLIENT_ID="..."
export DEVIANTART_CLIENT_SECRET="..."
export DEVIANTART_REFRESH_TOKEN="..."
export DEVIANTART_ACCESS_TOKEN=""
export DEVIANTART_REDIRECT_URI="http://localhost:8501/callback"
export DEVIANTART_USERNAME="me"
export DEVIANTART_PUBLISH="true"
export UPSCALER_BACKEND="realesrgan"
export REALESRGAN_TILE="256"
export REALESRGAN_TILE_PAD="10"
export REALESRGAN_PRE_PAD="0"
export REALESRGAN_MAX_INPUT_SIDE="0"
export REALESRGAN_MAX_INPUT_PIXELS="0"
export ACCEL_ONNX_MODEL_PATH=""
export ACCEL_ONNX_MODEL_URL=""
export ACCEL_DEFAULT_MODEL_PATH=""
export ACCEL_TILE="0"
export ACCEL_TILE_PAD="8"
export OPENVINO_DEVICE="GPU_FP32"
export DIRECTML_DEVICE_ID="0"
export RUN_INTERVAL_MINUTES="60"
export CIVITAI_IMAGES_API="https://civitai.com/api/v1/images"
export PROMPT_API_TIMEOUT_SECONDS="15"
export PROMPT_ENHANCER_MODEL="gpt-4.1-mini"
```

Notes:

- DeviantArt uploads usually require a user-approved OAuth refresh token. Client ID and secret alone are not enough to publish on behalf of an account.
- `DEVIANTART_ACCESS_TOKEN` is optional if you already have a fresh token and want to skip the refresh call.
- You can bootstrap and cache DeviantArt tokens locally with `python app.py auth_deviantart` after setting only `DEVIANTART_CLIENT_ID` and `DEVIANTART_CLIENT_SECRET`.
- Cached DeviantArt tokens are stored at `ai_art_bot/data/deviantart_tokens.json` and are auto-updated when DeviantArt rotates `refresh_token`.
- `DEVIANTART_PUBLISH=false` enables upload test mode (stash submit only, publish skipped).
- `UPSCALER_BACKEND` supports `realesrgan` (default, local no-token upscaling), `realesrgan_local`, `replicate`, `directml`, and `openvino`.
- `REALESRGAN_TILE` defaults to `256` and dramatically lowers peak RAM usage on CPU compared to full-frame (`0`) inference.
- `REALESRGAN_MAX_INPUT_SIDE` and `REALESRGAN_MAX_INPUT_PIXELS` are optional safety limits to pre-downscale very large inputs before upscaling.
- `REPLICATE_API_TOKEN` is only required when `UPSCALER_BACKEND=replicate`.
- `ACCEL_ONNX_MODEL_PATH` (or `ACCEL_ONNX_MODEL_URL`) is required when `UPSCALER_BACKEND` is `directml` or `openvino`.
- If `ACCEL_ONNX_MODEL_PATH` is empty, the app auto-uses `ai_art_bot/weights/realesrgan_x4.onnx` when that file exists.
- `ACCEL_DEFAULT_MODEL_PATH` can override this default local model location.
- `ACCEL_TILE` can be enabled for accelerated backends to process very large images in chunks.
- Prompt collection uses CivitAI-style API data first, then a local prompt generator, then `prompts.txt` fallback.
- Selected prompts are enhanced with OpenAI (`gpt-4.1-mini` by default) for richer detail, lighting, and composition.

Install local Real-ESRGAN dependencies when using local backend:

```bash
pip install realesrgan opencv-python-headless
```

Install accelerated backend dependencies (choose one path):

```bash
# Windows + Intel/AMD/NVIDIA via DirectML
pip install onnxruntime-directml

# OpenVINO Execution Provider
pip install onnxruntime-openvino
```

## Commands

```bash
python app.py run_once
python app.py run_loop --interval 90
python app.py generate_only
python app.py auth_deviantart
python app.py serve --host 0.0.0.0 --port 8000
```

## Dashboard

The optional FastAPI dashboard exposes:

- `GET /images` to inspect recent pipeline runs.
- `POST /generate` to trigger a full pipeline run.

## Design notes

- The code stays intentionally small and modular, with each integration isolated in its own module.
- The total code size is kept close to the requested range, but the DeviantArt OAuth flow and retry handling add a bit of necessary reliability.