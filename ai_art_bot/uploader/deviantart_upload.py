import json
import os
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlencode, urlparse

import requests

from utils.config import DEVIANTART_TOKENS_PATH, get_settings
from utils.logger import logger


TOKEN_URL = "https://www.deviantart.com/oauth2/token"
AUTHORIZE_URL = "https://www.deviantart.com/oauth2/authorize"
SUBMIT_URL = "https://www.deviantart.com/api/v1/oauth2/stash/submit"
PUBLISH_URL = "https://www.deviantart.com/api/v1/oauth2/stash/publish"
DEVIANTART_PUBLISH = os.getenv("DEVIANTART_PUBLISH", "true").strip().lower() not in {"0", "false", "no", "off"}


def _load_cached_tokens() -> dict:
    if not DEVIANTART_TOKENS_PATH.exists():
        return {}
    try:
        return json.loads(DEVIANTART_TOKENS_PATH.read_text(encoding="utf-8"))
    except Exception:
        logger.warning(f"Failed to parse cached DeviantArt tokens at {DEVIANTART_TOKENS_PATH}")
        return {}


def _save_cached_tokens(payload: dict) -> None:
    DEVIANTART_TOKENS_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing = _load_cached_tokens()
    merged = {**existing, **payload}
    DEVIANTART_TOKENS_PATH.write_text(json.dumps(merged, indent=2), encoding="utf-8")


def _persist_token_payload(payload: dict, fallback_refresh_token: str = "") -> None:
    token_payload: dict[str, str | int] = {}
    if payload.get("access_token"):
        token_payload["access_token"] = payload["access_token"]
    refresh_token = payload.get("refresh_token") or fallback_refresh_token
    if refresh_token:
        token_payload["refresh_token"] = refresh_token
    if payload.get("expires_in"):
        token_payload["expires_in"] = int(payload["expires_in"])
        token_payload["expires_at"] = int(time.time()) + int(payload["expires_in"])
    if token_payload:
        _save_cached_tokens(token_payload)


def _exchange_auth_code_for_tokens(auth_code: str, redirect_uri: str) -> dict:
    settings = get_settings()
    response = requests.post(
        TOKEN_URL,
        data={
            "grant_type": "authorization_code",
            "code": auth_code,
            "client_id": settings.deviantart_client_id,
            "client_secret": settings.deviantart_client_secret,
            "redirect_uri": redirect_uri,
        },
        timeout=60,
    )
    if not response.ok:
        raise RuntimeError(
            "Failed to exchange DeviantArt auth code for tokens "
            f"({response.status_code}). Response: {response.text}"
        )
    payload = response.json()
    _persist_token_payload(payload)
    return payload


def _capture_auth_code(redirect_uri: str, timeout_seconds: int = 240) -> str:
    parsed = urlparse(redirect_uri)
    if parsed.hostname not in {"127.0.0.1", "localhost"}:
        raise RuntimeError("DEVIANTART_REDIRECT_URI must use localhost or 127.0.0.1 for auto auth bootstrap")

    callback_path = parsed.path or "/"
    host = parsed.hostname
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    result: dict[str, str] = {}

    class _CallbackHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            request_path = urlparse(self.path)
            if request_path.path != callback_path:
                self.send_response(404)
                self.end_headers()
                return
            params = parse_qs(request_path.query)
            if "error" in params:
                result["error"] = params.get("error_description", params["error"])[0]
            if "code" in params and params["code"]:
                result["code"] = params["code"][0]

            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(b"<h3>Authorization captured. You can close this window.</h3>")

        def log_message(self, fmt: str, *args: object) -> None:
            return

    server = HTTPServer((host, port), _CallbackHandler)
    server.timeout = timeout_seconds
    logger.info("Waiting for DeviantArt OAuth callback...")
    end_time = time.time() + timeout_seconds

    while time.time() < end_time:
        server.handle_request()
        if "error" in result:
            raise RuntimeError(f"DeviantArt authorization failed: {result['error']}")
        if "code" in result:
            return result["code"]

    raise RuntimeError("Timed out waiting for DeviantArt OAuth callback")


def bootstrap_tokens() -> dict:
    settings = get_settings()
    redirect_uri = os.getenv("DEVIANTART_REDIRECT_URI", "http://localhost:8501/callback")
    scope = os.getenv("DEVIANTART_SCOPE", "stash publish")
    query = urlencode(
        {
            "response_type": "code",
            "client_id": settings.deviantart_client_id,
            "redirect_uri": redirect_uri,
            "scope": scope,
        },
        quote_via=quote,
    )
    authorize_url = f"{AUTHORIZE_URL}?{query}"
    logger.info(f"Opening DeviantArt authorization URL: {authorize_url}")
    webbrowser.open(authorize_url)
    code = _capture_auth_code(redirect_uri=redirect_uri)
    payload = _exchange_auth_code_for_tokens(auth_code=code, redirect_uri=redirect_uri)
    logger.info(f"Saved DeviantArt tokens to {DEVIANTART_TOKENS_PATH}")
    return payload


def _get_access_token() -> str:
    settings = get_settings()
    if settings.deviantart_access_token:
        return settings.deviantart_access_token

    cached_tokens = _load_cached_tokens()
    cached_access = cached_tokens.get("access_token", "")
    expires_at = int(cached_tokens.get("expires_at", 0) or 0)
    if cached_access and expires_at > int(time.time()) + 30:
        return cached_access

    refresh_token = settings.deviantart_refresh_token or cached_tokens.get("refresh_token", "")
    if not refresh_token:
        raise RuntimeError(
            "No DeviantArt refresh token found. Run 'python app.py auth_deviantart' to authorize and cache tokens."
        )

    response = requests.post(
        TOKEN_URL,
        data={
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": settings.deviantart_client_id,
            "client_secret": settings.deviantart_client_secret,
        },
        timeout=60,
    )
    if not response.ok:
        message = (
            f"Failed to refresh DeviantArt access token ({response.status_code}). "
            f"Response: {response.text}. "
            "Verify DEVIANTART_CLIENT_ID, DEVIANTART_CLIENT_SECRET, DEVIANTART_REFRESH_TOKEN, "
            "and ensure the token was generated for this exact app."
        )
        raise RuntimeError(message)
    payload = response.json()
    _persist_token_payload(payload, fallback_refresh_token=refresh_token)
    return payload["access_token"]


def upload_image(image_path: str, prompt: str) -> dict:
    token = _get_access_token()
    headers = {"Authorization": f"Bearer {token}"}
    title = prompt[:80].strip() or "AI Art Upload"
    tags = "ai,art,automation"

    with Path(image_path).open("rb") as image_file:
        submit_response = requests.post(
            SUBMIT_URL,
            headers=headers,
            data={
                "title": title,
                "artist_comments": prompt[:1000],
                "folder": "",
                "keywords": tags,
                "is_mature": "false",
            },
            files={"file": (Path(image_path).name, image_file, "image/png")},
            timeout=180,
        )
    submit_response.raise_for_status()
    itemid = submit_response.json().get("itemid")
    if not itemid:
        raise RuntimeError(f"Unexpected DeviantArt submit response: {submit_response.text}")

    if not DEVIANTART_PUBLISH:
        logger.info("DeviantArt test mode enabled (DEVIANTART_PUBLISH=false). Uploaded to stash only; publish skipped.")
        return {
            "published": False,
            "itemid": itemid,
            "submit_response": submit_response.json(),
        }

    publish_response = requests.post(
        PUBLISH_URL,
        headers=headers,
        data={
            "itemid": itemid,
            "title": title,
            "artist_comments": prompt[:1000],
            "is_mature": "false",
            "allow_comments": "true",
            "license_options": "noai",
        },
        timeout=60,
    )
    publish_response.raise_for_status()
    payload = publish_response.json()
    logger.info("Uploaded image to DeviantArt")
    return payload