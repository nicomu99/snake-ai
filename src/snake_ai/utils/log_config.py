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
    logging.config.dictConfig(LOGGING)
