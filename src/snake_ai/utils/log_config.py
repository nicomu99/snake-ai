"""Logging configuration for the Snake AI package.

Logs from every logger go to the console and to a rotating file. The file is
``app.log`` in the current working directory, holds at most 10 MB, and keeps
three backups. The root logger logs at ``DEBUG``. ``urllib3`` is limited to
``WARNING`` because it is noisy.

Attributes:
    LOGGING (dict): The ``logging.config.dictConfig`` schema that
        ``configure_logging`` applies.
"""

import logging.config

LOGGING = {
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
            "filename": "app.log",
            "maxBytes": 10_000_000,
            "backupCount": 3,
            "formatter": "standard",
        },
    },
    "root": {"level": "DEBUG", "handlers": ["console", "file"]},
    "loggers": {
        "urllib3": {"level": "WARNING"},  # quiet noisy dependencies
    },
}


def configure_logging():
    """Applies the ``LOGGING`` configuration to the logging system.

    Call this once at startup, before any log messages are written. Loggers
    that already exist stay enabled.
    """
    logging.config.dictConfig(LOGGING)
