"""Loguru logger configuration."""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

from app.config import config


def setup_logger() -> None:
    """Configure console logging and best-effort file logging."""
    logger.remove()

    logger.add(
        sys.stdout,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{module}</cyan>.<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        level="DEBUG" if config.debug else "INFO",
        colorize=True,
        backtrace=True,
        diagnose=config.debug,
    )

    try:
        Path("logs").mkdir(parents=True, exist_ok=True)
        logger.add(
            "logs/app_{time:YYYY-MM-DD}.log",
            rotation="00:00",
            retention="7 days",
            compression="zip",
            encoding="utf-8",
            enqueue=True,
            backtrace=True,
            diagnose=True,
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {module}.{function}:{line} | {message}",
        )
    except OSError as exc:
        logger.warning(f"File logging disabled: {exc}")


setup_logger()
