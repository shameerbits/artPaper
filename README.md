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
export IMAGE_BACKEND="openai"
export LOCAL_MODEL_ID="runwayml/stable-diffusion-v1-5"
export LOCAL_MODEL_PATH="./models/stable-diffusion-v1-5"
export LOCAL_MODEL_USE_OPENVINO="false"
export LOCAL_IMAGE_WIDTH="768"
export LOCAL_IMAGE_HEIGHT="1344"
export LOCAL_NUM_INFERENCE_STEPS="24"
export LOCAL_GUIDANCE_SCALE="7.0"
export LOCAL_NEGATIVE_PROMPT="blurry, low quality, text, watermark"
export LOCAL_SEED="-1"
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
export ACCEL_ONNX_MODEL_PATH="./weights/realesrgan_x4.onnx"
export ACCEL_ONNX_MODEL_URL=""
export ACCEL_DEFAULT_MODEL_PATH="./weights/realesrgan_x4.onnx"
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
- Cached DeviantArt tokens are stored at `./data/deviantart_tokens.json` and are auto-updated when DeviantArt rotates `refresh_token`.
- `DEVIANTART_PUBLISH=false` enables upload test mode (stash submit only, publish skipped).
- `UPSCALER_BACKEND` supports `realesrgan` (default, local no-token upscaling), `realesrgan_local`, `replicate`, `directml`, and `openvino`.
- `REALESRGAN_TILE` defaults to `256` and dramatically lowers peak RAM usage on CPU compared to full-frame (`0`) inference.
- `REALESRGAN_MAX_INPUT_SIDE` and `REALESRGAN_MAX_INPUT_PIXELS` are optional safety limits to pre-downscale very large inputs before upscaling.
- `REPLICATE_API_TOKEN` is only required when `UPSCALER_BACKEND=replicate`.
- `ACCEL_ONNX_MODEL_PATH` (or `ACCEL_ONNX_MODEL_URL`) is required when `UPSCALER_BACKEND` is `directml` or `openvino`.
- If `ACCEL_ONNX_MODEL_PATH` is empty, the app auto-uses `./weights/realesrgan_x4.onnx` when that file exists.
- `ACCEL_DEFAULT_MODEL_PATH` can override this default local model location.
- `ACCEL_TILE` can be enabled for accelerated backends to process very large images in chunks.
- Prompt collection uses CivitAI-style API data first, then a local prompt generator, then `prompts.txt` fallback.
- Selected prompts are enhanced with OpenAI (`gpt-4.1-mini` by default) for richer detail, lighting, and composition.

Local image generation options:

- `IMAGE_BACKEND=openai` keeps current behavior (OpenAI image API).
- `IMAGE_BACKEND=local_sd` enables local Stable Diffusion generation.
- `LOCAL_MODEL_ID` is a Hugging Face model id (default: `runwayml/stable-diffusion-v1-5`).
- `LOCAL_MODEL_PATH` can point to a downloaded local model folder (overrides `LOCAL_MODEL_ID`).
- `LOCAL_MODEL_USE_OPENVINO=true` enables OpenVINO backend for local SD generation.
- `LOCAL_NEGATIVE_PROMPT` optionally adds a default negative prompt for local SD generation.
- For Iris Xe, start with `LOCAL_IMAGE_WIDTH=768`, `LOCAL_IMAGE_HEIGHT=1344`, and `LOCAL_NUM_INFERENCE_STEPS=20-24`.

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

Install optional local SD generation dependencies:

```bash
pip install diffusers transformers accelerate safetensors huggingface_hub
```

Install optional OpenVINO local SD dependencies:

```bash
pip install "optimum-intel[openvino]"
```

## Local SD 1.5 quick start

### 1) Download a local model

Default SD 1.5 model:

```bash
python app.py download_model --model-id runwayml/stable-diffusion-v1-5
```

You can also download popular SD 1.5 derivatives from Hugging Face (when available):

```bash
python app.py download_model --model-id Lykon/dreamshaper-8
```

If you have a local model folder already, skip download and set `LOCAL_MODEL_PATH` directly.

### 2) Switch generator to local

```bash
export IMAGE_BACKEND="local_sd"
export LOCAL_MODEL_PATH="./models/stable-diffusion-v1-5"
```

or use a model id directly:

```bash
export IMAGE_BACKEND="local_sd"
export LOCAL_MODEL_ID="runwayml/stable-diffusion-v1-5"
```

### 3) Enable OpenVINO for Intel Iris Xe (optional)

```bash
export LOCAL_MODEL_USE_OPENVINO="true"
export OPENVINO_DEVICE="GPU"
export LOCAL_IMAGE_WIDTH="768"
export LOCAL_IMAGE_HEIGHT="1344"
export LOCAL_NUM_INFERENCE_STEPS="20"
```

### 4) Generate using local model

```bash
python app.py generate_only
```

## Recommended SD 1.5 family models

- Photorealism: EpicRealism, Juggernaut Aftermath.
- Best all-arounder: DreamShaper 8.
- Anime/digital art: MeinaMix.

If a model is hosted on CivitAI as a single `.safetensors` file, convert/export it into a Diffusers-style folder before use, then point `LOCAL_MODEL_PATH` to that exported folder.

## Commands

```bash
python app.py run_once
python app.py run_loop --interval 90
python app.py generate_only
python app.py download_model --model-id runwayml/stable-diffusion-v1-5
python app.py auth_deviantart
python app.py serve --host 0.0.0.0 --port 8000
streamlit run app.py
```

## Streamlit queue app

- Configuration can be saved from the UI and is written to `./data/web_settings.json`.
- All non-secret runtime options are configurable in the UI.
- API keys and secrets are not editable in the UI; they must be set via environment variables or Streamlit secrets.
- The UI shows secret availability and validates required credentials for the selected task mode before allowing queue submission.
- Manual prompt tasks can be queued with prompt mode options:
	- `as_is`: use your prompt exactly as provided.
	- `reformat`: convert your prompt into a cleaner natural-language generation prompt.
- Queue form includes an optional negative prompt field (applies to local SD and can override global `LOCAL_NEGATIVE_PROMPT` per task).
- Pipeline task modes supported in the queue:
	- `generate_only`
	- `generate_upscale`
	- `full` (generate + upscale + upload)
	- `upscale_only`
	- `upload_only`
	- `upscale_upload`
- Queue status tracking includes `queued`, `running`, `success`, `failure`, and `no_info`.

Deployment behavior:

- Deployed Streamlit mode shows only manual prompt queue controls and task status.
- Local Streamlit mode includes full settings editor, queue list, and manual one-by-one run controls.
- Set `APP_STREAMLIT_DEPLOYED=true` to force deployed mode behavior.

## Dashboard

The optional FastAPI dashboard exposes:

- `GET /images` to inspect recent pipeline runs.
- `POST /generate` to trigger a full pipeline run.

## Design notes

- The code stays intentionally small and modular, with each integration isolated in its own module.
- The total code size is kept close to the requested range, but the DeviantArt OAuth flow and retry handling add a bit of necessary reliability.