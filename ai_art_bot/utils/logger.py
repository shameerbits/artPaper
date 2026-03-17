from pathlib import Path
import sys

from loguru import logger

from utils.config import DATA_DIR, ensure_directories


def setup_logging() -> None:
    ensure_directories()
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(Path(DATA_DIR) / "app.log", level="INFO", rotation="1 MB")


__all__ = ["logger", "setup_logging"]