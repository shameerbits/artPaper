import os
import random
import re
from pathlib import Path

from openai import OpenAI
import requests

from utils.config import PROMPTS_FILE
from utils.logger import logger


CIVITAI_IMAGES_API = os.getenv("CIVITAI_IMAGES_API", "https://civitai.com/api/v1/images")
REQUEST_TIMEOUT_SECONDS = int(os.getenv("PROMPT_API_TIMEOUT_SECONDS", "15"))
PROMPT_ENHANCER_MODEL = os.getenv("PROMPT_ENHANCER_MODEL", "gpt-4.1-mini")
PROMPT_LOG_MAX_CHARS = int(os.getenv("PROMPT_LOG_MAX_CHARS", "260"))
BLOCKED_TOKENS = ["nsfw", "nude", "nudity", "gore", "blood"]


def _build_openai_client() -> OpenAI | None:
    if not os.getenv("OPENAI_API_KEY"):
        return None
    return OpenAI()


OPENAI_CLIENT = _build_openai_client()


def _preview(text: str, max_chars: int = PROMPT_LOG_MAX_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}..."


def _clean_prompt(text: str) -> str:
    if not text:
        return ""
    if "\\" in text:
        safe_text = re.sub(r"\\(?![\\\"'abfnrtv0-7xuU])", r"\\\\", text)
        text = bytes(safe_text, "utf-8").decode("unicode_escape")
    text = re.sub(r"\s+", " ", text).strip()
    text = text.removeprefix("/imagine prompt:").strip(" :")
    return text.strip('"')


def _is_prompt_length_valid(text: str) -> bool:
    return 20 <= len(text) <= 700


def _is_blocked_prompt(text: str) -> bool:
    lowered = text.lower()
    return any(token in lowered for token in BLOCKED_TOKENS)


def _is_valid_prompt(text: str) -> bool:
    return _is_prompt_length_valid(text) and not _is_blocked_prompt(text)


def _select_safe_prompt(prompts: list[str], source: str) -> str | None:
    if not prompts:
        return None

    candidates = prompts[:]
    random.shuffle(candidates)

    for index, prompt in enumerate(candidates, start=1):
        if not _is_prompt_length_valid(prompt):
            logger.warning(f"Skipping {source} prompt #{index}: invalid length")
            continue
        if _is_blocked_prompt(prompt):
            logger.warning(f"Skipping {source} prompt #{index}: blocked token detected")
            continue
        logger.info(f"Selected safe prompt candidate #{index} from {source}")
        return prompt

    return None


def _fallback_prompts() -> list[str]:
    file_path = Path(PROMPTS_FILE)
    if not file_path.exists():
        return []
    prompts = [_clean_prompt(line) for line in file_path.read_text().splitlines() if line.strip()]
    return [prompt for prompt in prompts if _is_valid_prompt(prompt)]


def _fetch_json(url: str, params: dict | None = None) -> dict:
    headers = {"User-Agent": "artPaper-bot/1.0"}
    response = requests.get(
        url,
        params=params,
        timeout=REQUEST_TIMEOUT_SECONDS,
        headers=headers,
    )
    response.raise_for_status()
    return response.json()


def _prompts_from_civitai() -> list[str]:
    try:
        payload = _fetch_json(
            CIVITAI_IMAGES_API,
            params={"limit": 100, "sort": "Most Reactions", "period": "Week"},
        )
    except Exception as exc:
        logger.warning(f"Failed to load prompts from CivitAI API: {exc}")
        return []

    prompts: set[str] = set()
    for item in payload.get("items", []):
        meta = item.get("meta") or {}
        for field in ("prompt", "Prompt"):
            candidate = _clean_prompt(str(meta.get(field, "")))
            if _is_prompt_length_valid(candidate):
                prompts.add(candidate)
    logger.info(f"Collected {len(prompts)} prompts from CivitAI")
    return sorted(prompts)


def _enhance_prompt(prompt: str) -> str:
    if OPENAI_CLIENT is None:
        logger.warning("OPENAI_API_KEY not set; skipping GPT prompt enhancement")
        return prompt

    try:
        response = OPENAI_CLIENT.responses.create(
            model=PROMPT_ENHANCER_MODEL,
            input=(
                "Enhance this image prompt with rich details, lighting, and composition. "
                "Return only one final prompt line, no markdown or explanations:\n"
                f"{prompt}"
            ),
        )
        enhanced = _clean_prompt(response.output_text.strip())
        if _is_valid_prompt(enhanced):
            logger.info(f"Prompt enhanced with {PROMPT_ENHANCER_MODEL}")
            logger.info(f"Enhanced prompt preview: {_preview(enhanced)}")
            return enhanced
        logger.warning("GPT-enhanced prompt was invalid; using original prompt")
        return prompt
    except Exception as exc:
        logger.warning(f"GPT prompt enhancement failed: {exc}")
        return prompt


def _generate_prompts(count: int = 30) -> list[str]:
    subjects = [
        "a floating island city",
        "an astronaut botanist",
        "a biomechanical fox",
        "an ancient tree temple",
        "a futuristic street market",
        "a desert observatory",
        "an underwater library",
        "a volcanic crystal cave",
    ]
    styles = [
        "cinematic concept art",
        "high-detail matte painting",
        "editorial fashion photography",
        "retro-futurist illustration",
        "surreal fine art",
        "hyperreal digital painting",
    ]
    lighting = [
        "golden hour volumetric light",
        "dramatic rim lighting",
        "soft diffused studio light",
        "neon reflections and mist",
        "moody moonlight",
    ]
    moods = [
        "mysterious and serene",
        "epic and majestic",
        "dreamlike and atmospheric",
        "bold and energetic",
        "minimal and contemplative",
    ]
    camera = [
        "35mm lens",
        "85mm portrait lens",
        "wide-angle composition",
        "isometric composition",
    ]

    generated: set[str] = set()
    while len(generated) < count:
        prompt = ", ".join(
            [
                random.choice(subjects),
                random.choice(styles),
                random.choice(lighting),
                random.choice(moods),
                random.choice(camera),
                "ultra detailed, 8k, sharp focus",
            ]
        )
        cleaned = _clean_prompt(prompt)
        if _is_valid_prompt(cleaned):
            generated.add(cleaned)

    logger.info(f"Generated {len(generated)} local prompts")
    return sorted(generated)


def _dedupe(prompts: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for prompt in prompts:
        key = prompt.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(prompt)
    return deduped


def get_random_prompt() -> str:
    civitai_prompts = _prompts_from_civitai()
    generated_prompts = _generate_prompts(count=40)

    remote_prompts = _dedupe(civitai_prompts)
    if remote_prompts:
        selected = _select_safe_prompt(remote_prompts, source="civitai")
        if selected is None:
            logger.warning("All CivitAI prompts were blocked or invalid; trying next source")
        else:
            source = "civitai"
            logger.info(f"Selected prompt source: {source}")
            logger.info(f"Selected base prompt preview: {_preview(selected)}")
            return _enhance_prompt(selected)

    if generated_prompts:
        selected = _select_safe_prompt(generated_prompts, source="local_generator")
        if selected is None:
            logger.warning("All generated prompts were blocked or invalid; trying fallback source")
        else:
            source = "local_generator"
            logger.info(f"Selected prompt source: {source}")
            logger.info(f"Selected base prompt preview: {_preview(selected)}")
            return _enhance_prompt(selected)

    fallback = _fallback_prompts()
    if fallback:
        logger.warning("Prompt APIs unavailable; using prompts.txt fallback")
        selected = _select_safe_prompt(fallback, source="prompts_txt")
        if selected is None:
            logger.warning("All prompts.txt prompts were blocked or invalid")
        else:
            source = "prompts_txt"
            logger.info(f"Selected prompt source: {source}")
            logger.info(f"Selected base prompt preview: {_preview(selected)}")
            return _enhance_prompt(selected)

    raise RuntimeError("No prompts available from APIs, generator, or prompts.txt")


if __name__ == "__main__":
    selected_prompt = get_random_prompt()
    print(f"PROMPT: {selected_prompt}")