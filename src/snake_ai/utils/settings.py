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

from pydantic import Field, field_validator, model_validator
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
        model_catalog: The JSON file that lists the models the server
            offers (see ``ai.PolicySelector``). Defaults to ``models.json``
            in ``model_path``, also when ``model_path`` comes from the
            environment.
        max_game_count: The maximum number of games the server keeps. When
            a new game exceeds it, the oldest game is evicted. Must be at
            least 1. Defaults to 50.
    """

    model_config = SettingsConfigDict(env_prefix="SNAKE_AI_")

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_path: Path = PROJECT_ROOT / "logs/app.log"

    model_path: Path = PROJECT_ROOT / "policy_models"
    model_catalog: Path | None = None  # None defaults to model_path / "models.json"

    max_game_count: int = Field(default=50, ge=1)

    @field_validator("log_path", "model_path", "model_catalog")
    @classmethod
    def _make_absolute(cls, path: Path | None) -> Path | None:
        """Expands ``~`` and resolves the path to an absolute one.

        ``BaseSettings`` also validates defaults, so this runs for every
        field it is registered for, even when nothing is configured.

        Args:
            path: The configured path, or None if the field is unset.

        Returns:
            The absolute path, or None if ``path`` is None. Relative paths
            are resolved against the current working directory.
        """
        if path is None:
            return None
        return path.expanduser().resolve()

    @model_validator(mode="after")
    def check_catalog_path(self):
        """Defaults ``model_catalog`` to ``models.json`` in ``model_path``.

        This runs after all fields are set, so ``model_path`` already holds
        the value from the environment, if there is one.

        Returns:
            The settings, with ``model_catalog`` set.
        """
        if self.model_catalog is None:
            self.model_catalog = self.model_path / "models.json"
        return self


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
