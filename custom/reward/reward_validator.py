"""Generic reward function validator.

Not APPS-specific — works with any reward file that follows the
REWARD_REGISTRY convention. Used by validate_env.py to catch config
errors before an expensive training run starts.

Convention:
    Reward files that support config-driven functions should export:
        REWARD_REGISTRY: dict[str, type | None]
    mapping function names to their config dataclass (or None if no config).

Usage:
    validator = RewardValidator()
    validator.validate(
        reward_path="custom/reward/APPS/APPS_reward_configed.py",
        reward_name="configed_reward_backdoor_w_hidden",
        reward_kwargs={"reward_config": {"formatter": "removeaftercode_w_hidden"}},
    )
    # Raises on failure, returns None on success.
"""

import importlib.util
import os
import sys
from typing import Any, Optional

import dacite


class RewardValidator:
    """Validates reward function existence and config correctness.

    Performs the following checks (in order, each can raise):
    1. Import the reward module (catches SyntaxError)
    2. Check the named function exists in the module
    3. If REWARD_REGISTRY exists in module, check function is registered
    4. If reward_kwargs provided and registry maps to a config class,
       construct the config via dacite (catches invalid field values)
    """

    def validate(
        self,
        reward_path: str,
        reward_name: str,
        reward_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        """Validate reward function and config. Raises on any failure.

        Args:
            reward_path: Path to the Python file containing the reward function.
            reward_name: Name of the reward function in that file.
            reward_kwargs: Optional dict of kwargs that will be passed to the function
                          (typically contains {"reward_config": {...}}).

        Raises:
            FileNotFoundError: reward_path doesn't exist
            SyntaxError: reward file has syntax errors
            ImportError: reward file can't be imported
            AttributeError: reward_name function not found in module
            ValueError: function not in REWARD_REGISTRY, or config validation failed
        """
        # 1. Check file exists
        if not os.path.exists(reward_path):
            raise FileNotFoundError(f"Reward file not found: {reward_path}")

        # 2. Import module dynamically (catches SyntaxError)
        module = self._import_module(reward_path=reward_path)

        # 3. Check function exists
        if not hasattr(module, reward_name):
            available = [
                name for name in dir(module)
                if callable(getattr(module, name, None)) and not name.startswith("_")
            ]
            raise AttributeError(
                f"Reward function '{reward_name}' not found in '{reward_path}'. "
                f"Available functions: {available}"
            )

        fn = getattr(module, reward_name)
        if not callable(fn):
            raise AttributeError(
                f"'{reward_name}' exists in '{reward_path}' but is not callable "
                f"(type: {type(fn).__name__})"
            )

        # 4. Check REWARD_REGISTRY if it exists
        registry: Optional[dict[str, type | None]] = getattr(module, "REWARD_REGISTRY", None)

        if registry is not None:
            if reward_name not in registry:
                raise ValueError(
                    f"Reward function '{reward_name}' is not in REWARD_REGISTRY. "
                    f"Registered functions: {list(registry.keys())}"
                )

            config_class: type | None = registry[reward_name]

            # 5. Validate reward_kwargs against config class if both are provided
            if reward_kwargs is not None and config_class is not None:
                self._validate_config(
                    config_class=config_class,
                    reward_kwargs=reward_kwargs,
                    reward_name=reward_name,
                )

    def _import_module(self, reward_path: str) -> Any:
        """Dynamically import a module from a file path.

        Uses the same approach as reward.py:get_custom_reward_fn() to ensure
        the module is importable and pickle-safe.
        """
        abs_path = os.path.abspath(reward_path)
        cwd = os.getcwd()

        try:
            rel_path = os.path.relpath(abs_path, cwd)
        except ValueError:
            rel_path = os.path.basename(abs_path)

        if rel_path.endswith(".py"):
            rel_path = rel_path[:-3]
        module_name = rel_path.replace(os.sep, ".")

        spec = importlib.util.spec_from_file_location(module_name, abs_path)
        if spec is None:
            raise ImportError(f"Could not create module spec for '{reward_path}'")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module

        if spec.loader is None:
            raise ImportError(f"Module spec has no loader for '{reward_path}'")

        spec.loader.exec_module(module)  # SyntaxError raised here if file has syntax errors
        return module

    def _validate_config(
        self,
        config_class: type,
        reward_kwargs: dict[str, Any],
        reward_name: str,
    ) -> None:
        """Try to construct the config dataclass from reward_kwargs.

        The reward_kwargs dict typically has the shape:
            {"reward_config": {"formatter": "removeaftercode", ...}}
        We extract the "reward_config" sub-dict and pass it to dacite.
        """
        reward_config_data: dict[str, Any] = reward_kwargs.get("reward_config", {})

        if not isinstance(reward_config_data, dict):
            raise ValueError(
                f"reward_kwargs['reward_config'] must be a dict, "
                f"got {type(reward_config_data).__name__}"
            )

        try:
            # Use a permissive cast list — try all Enum subclasses in the module
            from enum import Enum

            enum_types = [
                cls for cls in config_class.__mro__
                if isinstance(cls, type) and issubclass(cls, Enum)
            ]

            # Also look for Enum types in field annotations
            import dataclasses

            if dataclasses.is_dataclass(config_class):
                for f in dataclasses.fields(config_class):
                    if isinstance(f.type, type) and issubclass(f.type, Enum):
                        enum_types.append(f.type)

            # Import the known enum types from reward_config_types
            from custom.reward.APPS.reward_config_types import (
                FormatterType,
                PenaltySchedule,
                ScoreType,
            )

            cast_types: list[type] = [FormatterType, ScoreType, PenaltySchedule]

            instance = dacite.from_dict(
                data_class=config_class,
                data=reward_config_data,
                config=dacite.Config(cast=cast_types),
            )
        except (dacite.DaciteError, ValueError, TypeError) as e:
            raise ValueError(
                f"Invalid reward_config for '{reward_name}' "
                f"(config class: {config_class.__name__}): {e}"
            ) from e
