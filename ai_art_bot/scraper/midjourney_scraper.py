import random
import re
import os
from pathlib import Path
from urllib.parse import urljoin

from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

from utils.config import PROMPTS_FILE
from utils.logger import logger


MIDJOURNEY_URL = "https://www.midjourney.com/explore?tab=top"
MIDJOURNEY_BASE = "https://www.midjourney.com"
SECURITY_HINTS = [
    "security verification",
    "verify you are human",
    "captcha",
    "challenge",
    "just a moment",
    "cloudflare",
]
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


def _dismiss_onboarding_modal(page) -> None:
    selectors = [
        "button:has-text('Look around a bit')",
        "button:has-text('Look around')",
        "text=Look around a bit",
        "text=Look around",
    ]
    for selector in selectors:
        try:
            button = page.locator(selector).first
            if button.is_visible(timeout=2000):
                button.click(timeout=3000)
                page.wait_for_timeout(1200)
                logger.info("Dismissed MidJourney onboarding modal")
                return
        except Exception:
            continue


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


def _is_security_challenge(page) -> bool:
    url = page.url.lower()
    if any(token in url for token in ["security", "verify", "challenge", "captcha"]):
        return True
    try:
        body = page.inner_text("body").lower()
    except Exception:
        return False
    return any(hint in body for hint in SECURITY_HINTS)


def _wait_for_security_clear(page, timeout_ms: int = 120000) -> bool:
    start = page.evaluate("Date.now()")
    while True:
        if not _is_security_challenge(page):
            return True
        now = page.evaluate("Date.now()")
        if now - start > timeout_ms:
            return False
        page.wait_for_timeout(2000)


def _manual_verification_prompt(page, location: str = "explore") -> bool:
    logger.warning(f"MANUAL VERIFICATION REQUIRED at {location}")
    logger.warning("Complete the security challenge in the Chromium browser window.")
    logger.warning("Waiting up to 180 seconds. Page will continue automatically once cleared.")
    
    start = page.evaluate("Date.now()")
    max_wait = 180000
    
    for attempt in range(1, 91):
        now = page.evaluate("Date.now()")
        elapsed = now - start
        if elapsed > max_wait:
            logger.error(f"Manual verification timeout after {elapsed}ms")
            return False
        
        is_challenge = _is_security_challenge(page)
        if not is_challenge:
            logger.info(f"Verification cleared")
            page.wait_for_timeout(1500)
            return True
        
        if attempt % 10 == 0:
            logger.info(f"Waiting for verification... ({int(elapsed/1000)}s elapsed)")
        
        page.wait_for_timeout(2000)
    
    return False


def get_random_prompt() -> str:
    try:
        headless = os.getenv("MIDJOURNEY_HEADLESS", "true").lower() not in {"0", "false", "no"}
        manual_verify = os.getenv("MIDJOURNEY_MANUAL_VERIFY", "true").lower() in {"1", "true", "yes"}
        logger.info(f"MidJourney scraper starting. headless={headless}, manual_verify={manual_verify}")
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=headless)
            page = browser.new_page(viewport={"width": 1440, "height": 1200})
            logger.info(f"Browser launched, navigating to explore page")
            page.goto(MIDJOURNEY_URL, wait_until="domcontentloaded", timeout=90000)
            page.wait_for_timeout(5000)
            logger.info(f"Explore page loaded")

            if _is_security_challenge(page):
                if (not headless) and manual_verify:
                    logger.warning("Security verification detected on explore page")
                    success = _manual_verification_prompt(page, location="explore")
                    if not success:
                        raise RuntimeError("Security verification not cleared within timeout at explore page")
                    logger.info("Explore page verification cleared, continuing...")
                else:
                    raise RuntimeError("Security verification page detected; run headed mode and clear manually")

            logger.info("Attempting to dismiss onboarding modal")
            _dismiss_onboarding_modal(page)

            logger.info("Collecting job URLs by scrolling explore page...")
            job_urls = _collect_job_urls(page)
            logger.info(f"Found {len(job_urls)} job URLs")
            if not job_urls:
                raise RuntimeError("No MidJourney job URLs discovered from explore page")

            selected_url = random.choice(job_urls)
            logger.info(f"Selected MidJourney job URL: {selected_url}")
            page.goto(selected_url, wait_until="domcontentloaded", timeout=60000)
            page.wait_for_timeout(3000)
            logger.info(f"Job page loaded")

            if _is_security_challenge(page):
                if (not headless) and manual_verify:
                    logger.warning("Security verification detected on job page")
                    success = _manual_verification_prompt(page, location="job_page")
                    if not success:
                        raise RuntimeError("Job page security verification not cleared within timeout")
                    logger.info("Job page verification cleared, continuing...")
                else:
                    raise RuntimeError("Security verification page detected on job URL")

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


if __name__ == "__main__":
    selected_prompt = get_random_prompt()
    print(f"PROMPT: {selected_prompt}")