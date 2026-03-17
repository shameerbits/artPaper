from pathlib import Path

import requests

from utils.config import get_settings
from utils.logger import logger


TOKEN_URL = "https://www.deviantart.com/oauth2/token"
SUBMIT_URL = "https://www.deviantart.com/api/v1/oauth2/stash/submit"
PUBLISH_URL = "https://www.deviantart.com/api/v1/oauth2/stash/publish"


def _get_access_token() -> str:
    settings = get_settings()
    if settings.deviantart_access_token:
        return settings.deviantart_access_token
    response = requests.post(
        TOKEN_URL,
        data={
            "grant_type": "refresh_token",
            "refresh_token": settings.deviantart_refresh_token,
            "client_id": settings.deviantart_client_id,
            "client_secret": settings.deviantart_client_secret,
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["access_token"]


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