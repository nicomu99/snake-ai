"""Runtime settings, read from ``SNAKE_AI_*`` environment variables.

Every setting has a default, so no environment variables are needed. Paths
are always absolute: the defaults are anchored to the repository root, and
relative paths from the environment are resolved against the current
working directory.

Example:
    Log at ``DEBUG`` and keep models in a different directory::

        $ SNAKE_AI_LOG_LEVEL=DEBUG SNAKE_AI_MODEL_PATH=~/models \\
            uv run python -m snake_ai.local_main

Attributes:
    PROJECT_ROOT (Path): The repository root. It is derived from this file's
        location, so it is only correct when the package runs from a
        checkout (editable install).
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """Application settings.

    ``BaseSettings`` fills each field that is not passed as a keyword
    argument from the environment variable ``SNAKE_AI_<FIELD_NAME>``, for
    example ``SNAKE_AI_LOG_LEVEL``. Use ``get_settings`` instead of creating
    instances directly.

    Attributes:
        log_level: The root logger level. Must be upper case.
        log_path: The log file. Its directory is created by
            ``configure_logging``. Defaults to ``logs/app.log`` in the
            repository root.
        model_path: The directory that models are loaded from and trained
            models are saved to. Defaults to ``policy_models`` in the
            repository root.
    """

    model_config = SettingsConfigDict(env_prefix="SNAKE_AI_")

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_path: Path = PROJECT_ROOT / "logs/app.log"
    model_path: Path = PROJECT_ROOT / "policy_models"

    @field_validator("log_path", "model_path")
    @classmethod
    def _make_absolute(cls, path: Path) -> Path:
        """Expands ``~`` and resolves the path to an absolute one.

        Args:
            path: The configured path.

        Returns:
            The absolute path. Relative paths are resolved against the
            current working directory.
        """
        return path.expanduser().resolve()


@lru_cache
def get_settings() -> Settings:
    """Returns the settings, reading the environment on the first call only.

    The result is cached. Call ``get_settings.cache_clear()`` to read the
    environment again, for example in tests.

    Returns:
        The shared ``Settings`` instance.

    Raises:
        pydantic.ValidationError: If an environment variable holds an
            invalid value.
    """
    return Settings()
