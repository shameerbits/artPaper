import random
import re
from pathlib import Path

from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

from utils.config import PROMPTS_FILE
from utils.logger import logger


MIDJOURNEY_URL = "https://www.midjourney.com/explore?tab=top"
PROMPT_PATTERNS = [
    r'"prompt"\s*:\s*"(.*?)"',
    r'"full_command"\s*:\s*"(.*?)"',
    r'"text"\s*:\s*"(/imagine.*?)"',
]


def _clean_prompt(text: str) -> str:
    text = bytes(text, "utf-8").decode("unicode_escape")
    text = re.sub(r"\s+", " ", text).strip()
    return text.removeprefix("/imagine prompt:").strip(" :")


def _fallback_prompts() -> list[str]:
    if not Path(PROMPTS_FILE).exists():
        return []
    prompts = [line.strip() for line in Path(PROMPTS_FILE).read_text().splitlines() if line.strip()]
    return prompts


def _extract_prompts_from_html(html: str) -> list[str]:
    prompts: set[str] = set()
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup.find_all(attrs={"alt": True}):
        alt_text = _clean_prompt(tag.get("alt", ""))
        if len(alt_text) > 20:
            prompts.add(alt_text)

    for tag in soup.find_all(["p", "span", "div", "button"]):
        text = _clean_prompt(tag.get_text(" ", strip=True))
        if 25 < len(text) < 500 and any(keyword in text.lower() for keyword in ["cinematic", "portrait", "lighting", "surreal", "prompt"]):
            prompts.add(text)

    for pattern in PROMPT_PATTERNS:
        for match in re.findall(pattern, html, flags=re.IGNORECASE | re.DOTALL):
            cleaned = _clean_prompt(match)
            if 20 < len(cleaned) < 500:
                prompts.add(cleaned)

    return sorted(prompts)


def get_random_prompt() -> str:
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 1200})
            page.goto(MIDJOURNEY_URL, wait_until="networkidle", timeout=45000)
            page.wait_for_timeout(4000)
            html = page.content()
            browser.close()
        prompts = _extract_prompts_from_html(html)
        if prompts:
            prompt = random.choice(prompts)
            logger.info("Selected MidJourney prompt from live page")
            return prompt
        raise RuntimeError("No prompts extracted from MidJourney page")
    except Exception as exc:
        logger.warning(f"MidJourney scrape failed, using prompts.txt fallback: {exc}")
        fallback = _fallback_prompts()
        if fallback:
            return random.choice(fallback)
        raise RuntimeError("No prompts available from MidJourney or prompts.txt") from exc