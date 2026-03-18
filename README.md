# artPaper

Simple local Python pipeline for selecting a Stable Diffusion-style prompt (from APIs/datasets plus a local generator), generating an image with OpenAI, upscaling it with Replicate, publishing it to DeviantArt, and recording metadata in SQLite.

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
export DEVIANTART_USERNAME="me"
export RUN_INTERVAL_MINUTES="60"
export CIVITAI_IMAGES_API="https://civitai.com/api/v1/images"
export LEXICA_SEARCH_API="https://lexica.art/api/v1/search"
export PROMPT_API_TIMEOUT_SECONDS="15"
```

Notes:

- DeviantArt uploads usually require a user-approved OAuth refresh token. Client ID and secret alone are not enough to publish on behalf of an account.
- `DEVIANTART_ACCESS_TOKEN` is optional if you already have a fresh token and want to skip the refresh call.
- Prompt collection uses CivitAI-style API data first, then Lexica-style prompt search, then a local prompt generator, then `prompts.txt` fallback.

## Commands

```bash
python app.py run_once
python app.py run_loop --interval 90
python app.py generate_only
python app.py serve --host 0.0.0.0 --port 8000
```

## Dashboard

The optional FastAPI dashboard exposes:

- `GET /images` to inspect recent pipeline runs.
- `POST /generate` to trigger a full pipeline run.

## Design notes

- The code stays intentionally small and modular, with each integration isolated in its own module.
- The total code size is kept close to the requested range, but the DeviantArt OAuth flow and retry handling add a bit of necessary reliability.