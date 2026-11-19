"""Loads the models listed in the model catalog and hands them out by id."""

import json
import logging
from pathlib import Path

from snake_ai.ai.model_entry import ModelEntry
from snake_ai.ai.policy import Policy

logger = logging.getLogger(__name__)


class PolicySelector:
    """Holds one loaded ``Policy`` per available model.

    The policies are only used for inference and are shared by every game
    that uses the same model, so each model is loaded only once. Use
    ``from_path`` to load the models listed in a catalog file.

    Args:
        model_catalog: The catalog entries of the loaded models, by model id.
        policies: The loaded policies, by model id. Must have the same keys
            as ``model_catalog``.

    Attributes:
        model_catalog: The catalog entries of the available models, by
            model id.
        policies: The loaded policies, by model id.
    """

    def __init__(
        self,
        model_catalog: dict[str, ModelEntry],
        policies: dict[str, Policy]
    ):
        self.model_catalog = model_catalog
        self.policies: dict[str, Policy] = policies

    def select_policy(self, model_id: str) -> Policy:
        """Returns the shared policy of a model.

        Args:
            model_id: The id of the model, as listed in the catalog.

        Returns:
            The loaded policy. The same instance is returned on every call.

        Raises:
            ValueError: If no model with this id was loaded.
        """
        if model_id not in self.model_catalog:
            raise ValueError(
                f"Model id {model_id} unknown. "
                f"Must be one of {list(self.model_catalog.keys())}"
            )
        return self.policies[model_id]

    def choose_default(self):
        """Returns the policy that new games start with.

        The default model id is hard-coded to ``"dqn-200k"`` for now.

        Returns:
            The default model's shared policy.

        Raises:
            ValueError: If the default model was not loaded, for example
                because its file is missing.
        """
        # TODO
        default_model_id = "dqn-200k"
        return self.select_policy(default_model_id)

    @classmethod
    def from_path(cls, model_catalog_path: Path, model_path: Path):
        """Reads a catalog file and loads every model listed in it.

        The catalog is a JSON object with a ``"models"`` list, where each
        item has the fields of ``ModelEntry``. Entries whose model file does
        not exist are skipped with a warning.

        Args:
            model_catalog_path: The catalog file, e.g. ``models.json``.
            model_path: The directory the entries' ``uri`` values are
                relative to.

        Returns:
            A selector with every model that could be found.

        Raises:
            FileNotFoundError: If the catalog file does not exist.
            json.JSONDecodeError: If the catalog is not valid JSON.
            KeyError: If the catalog has no ``"models"`` key.
            pydantic.ValidationError: If an entry does not match
                ``ModelEntry``.
        """
        data =json.loads(model_catalog_path.read_text())
        model_entries = [ModelEntry.model_validate(m) for m in data["models"]]
        model_catalog = {}
        policies = {}
        for m in model_entries:
            model_uri = model_path / m.uri
            if not model_uri.exists():
                logger.warning(
                    "Invalid model uri %s for model %s", model_uri, m.id)
                continue
            model_catalog[m.id] = m
            policies[m.id] = Policy.load_model(m.uri, model_path)
        return cls(model_catalog, policies)
