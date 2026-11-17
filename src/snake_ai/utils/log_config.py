"""Logging configuration for the Snake AI package.

Logs from every logger go to the console and to a rotating file. The file
path (``Settings.log_path``) and the root log level (``Settings.log_level``)
come from the settings. The file holds at most 10 MB and keeps three
backups. ``urllib3`` is limited to ``WARNING`` because it is noisy.
"""

import logging.config

from .settings import get_settings


def configure_logging():
    """Configures console and rotating file logging from the settings.

    Reads the settings, creates the log file's directory if needed and
    applies the configuration with ``logging.config.dictConfig``. Call this
    once at startup, before any log messages are written. Loggers that
    already exist stay enabled.

    Raises:
        pydantic.ValidationError: If a ``SNAKE_AI_*`` environment variable
            holds an invalid value.
    """
    settings = get_settings()
    settings.log_path.parent.mkdir(parents=True, exist_ok=True)

    logging_dict = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s %(levelname)-8s %(name)s: %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "standard",
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "filename": settings.log_path,
                "maxBytes": 10_000_000,
                "backupCount": 3,
                "formatter": "standard",
            },
        },
        "root": {"level": settings.log_level, "handlers": ["console", "file"]},
        "loggers": {
            "urllib3": {"level": "WARNING"},  # quiet noisy dependencies
        },
    }

    logging.config.dictConfig(logging_dict)
