import random
import re
from pathlib import Path
from urllib.parse import urljoin

from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

from utils.config import PROMPTS_FILE
from utils.logger import logger


MIDJOURNEY_URL = "https://www.midjourney.com/explore?tab=top"
MIDJOURNEY_BASE = "https://www.midjourney.com"
PROMPT_PATTERNS = [
    r'"job_prompt"\s*:\s*"(.*?)"',
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
        if 20 < len(alt_text) < 600:
            prompts.add(alt_text)

    for tag in soup.find_all(["p", "span", "div", "button"]):
        text = _clean_prompt(tag.get_text(" ", strip=True))
        if 25 < len(text) < 500 and any(keyword in text.lower() for keyword in ["cinematic", "portrait", "lighting", "surreal", "prompt"]):
            prompts.add(text)

    for pattern in PROMPT_PATTERNS:
        for match in re.findall(pattern, html, flags=re.IGNORECASE | re.DOTALL):
            cleaned = _clean_prompt(match)
            if 20 < len(cleaned) < 600:
                prompts.add(cleaned)

    return sorted(prompts)


def _collect_job_urls(page) -> list[str]:
    urls: set[str] = set()
    last_count = 0
    stale_rounds = 0

    for _ in range(10):
        hrefs = page.eval_on_selector_all(
            "a[href*='/jobs/']",
            "els => els.map(el => el.getAttribute('href')).filter(Boolean)",
        )
        for href in hrefs:
            absolute = urljoin(MIDJOURNEY_BASE, href)
            if "/jobs/" in absolute:
                urls.add(absolute.split("#", 1)[0])

        if len(urls) == last_count:
            stale_rounds += 1
        else:
            stale_rounds = 0
        last_count = len(urls)

        if len(urls) >= 20 or stale_rounds >= 3:
            break

        page.mouse.wheel(0, 5000)
        page.wait_for_timeout(1800)

    return sorted(urls)


def _extract_prompt_from_job_page(page) -> str | None:
    html = page.content()
    prompts = _extract_prompts_from_html(html)
    if prompts:
        return random.choice(prompts)

    body_text = _clean_prompt(page.inner_text("body"))
    imagine_matches = re.findall(r"/imagine\s+prompt\s*:\s*([^\n]{20,700})", body_text, flags=re.IGNORECASE)
    if imagine_matches:
        return _clean_prompt(random.choice(imagine_matches))

    return None


def get_random_prompt() -> str:
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 1200})
            page.goto(MIDJOURNEY_URL, wait_until="domcontentloaded", timeout=90000)
            page.wait_for_timeout(5000)

            job_urls = _collect_job_urls(page)
            if not job_urls:
                raise RuntimeError("No MidJourney job URLs discovered from explore page")

            selected_url = random.choice(job_urls)
            logger.info(f"Selected MidJourney job URL: {selected_url}")
            page.goto(selected_url, wait_until="domcontentloaded", timeout=60000)
            page.wait_for_timeout(3000)

            prompt = _extract_prompt_from_job_page(page)
            browser.close()
        if prompt:
            logger.info("Selected MidJourney prompt from random job page")
            return prompt
        raise RuntimeError("No prompt extracted from selected MidJourney job page")
    except Exception as exc:
        logger.warning(f"MidJourney scrape failed, using prompts.txt fallback: {exc}")
        fallback = _fallback_prompts()
        if fallback:
            return random.choice(fallback)
        raise RuntimeError("No prompts available from MidJourney or prompts.txt") from exc