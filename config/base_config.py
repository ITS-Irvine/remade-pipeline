# config/base_config.py
import dataclasses
from dataclasses import dataclass, fields
from enum import Enum
from pathlib import Path
from typing import Any, List, Optional, Tuple, Type, TypeVar
import typing
from config.settings import settings
from typing import get_type_hints
from config.path_template import PathTemplate


_model_registry: List[Tuple[str, Type]] = []

T = TypeVar("T", bound="BaseModelConfig")

class BaseModelConfig:
    _cli_prefix: str = ""
    _model_label: str = ""

    def __init_subclass__(cls, prefix: str = "", label: Optional[str] = None, **kwargs):
        super().__init_subclass__(**kwargs)

        # Explicit empty label is always an error
        if label == "":
            raise ValueError(
                f"{cls.__name__}: label cannot be an empty string. "
                f"Omit it to default to the prefix."
            )

        resolved_label = label if label is not None else prefix

        # Register if prefix OR label is explicitly provided
        if prefix or label is not None:
            if not resolved_label:
                raise ValueError(
                    f"{cls.__name__}: a non-empty label or prefix is required for registration."
                )
            cls._cli_prefix  = prefix
            cls._model_label = resolved_label
            _model_registry.append((cls._model_label, cls))
            
    @classmethod
    def from_dynaconf(cls: Type[T]) -> T:
        prefix      = cls._cli_prefix
        kwargs      = {}
        sources     = {}                          # ← track sources

        # Load top-level [model] and submodel [model.<prefix>] sections
        model_ns = settings.get("model") or {}
        sub_ns   = settings.get(f"model.{prefix}") or {}

        # Merge: submodel values override top-level [model], missing keys filled from [model]
        dynaconf_ns = {**model_ns, **sub_ns}


        for f in fields(cls):
            if f.name in dynaconf_ns:
                kwargs[f.name]  = dynaconf_ns[f.name]
                sources[f.name] = "dynaconf (model)" if f.name in sub_ns else "dynaconf (global)"
            else:
                kwargs[f.name]  = f.default
                sources[f.name] = "default"

        instance          = cls(**kwargs)
        instance._sources = sources
        instance.resolve_templates()   # ← expand templates after loading
        
        # ── Automatically coerce strings to enums for any enum-typed field ────
        hints = get_type_hints(cls)
        for f in fields(cls):
            value     = getattr(instance, f.name)
            hint      = hints.get(f.name)

            # Unwrap Optional[EnumType] → EnumType
            origin = typing.get_origin(hint)
            if origin is typing.Union:
                args = [a for a in typing.get_args(hint) if a is not type(None)]
                hint = args[0] if len(args) == 1 else None

            if hint is not None and isinstance(hint, type) and issubclass(hint, Enum):
                if isinstance(value, str):
                    try:
                        setattr(instance, f.name, hint(value))
                        # preserve the original source label
                    except ValueError:
                        raise ValueError(
                            f"{cls.__name__}.{f.name}: "
                            f"{value!r} is not a valid {hint.__name__} value. "
                            f"Valid values: {[e.value for e in hint]}"
                        )

        
        
        return instance

    @classmethod
    def from_dynaconf_with_overrides(cls: Type[T], **cli_overrides) -> T:
        instance = cls.from_dynaconf()
        sources  = getattr(instance, "_sources", {})

        for key, value in cli_overrides.items():
            if value is not None:
                setattr(instance, key, value)
                sources[key] = "cli"              # ← override source to cli

        instance._sources = sources
        return instance
    

    def resolve_templates(self) -> None:
        hints   = get_type_hints(self.__class__)

        # inject base model_paths first:
        context=settings.get('model_paths').to_dict()

        # 2. Override with current section fields (self)
        # This ensures local fields take precedence if there's a name collision
        context.update({
            f.name: str(getattr(self, f.name))
            for f in fields(self)
            if getattr(self, f.name) is not None
        })

        for f in fields(self):
            hint = hints.get(f.name)

            # Unwrap Optional[PathTemplate]
            if typing.get_origin(hint) is typing.Union:
                args = [a for a in typing.get_args(hint) if a is not type(None)]
                hint = args[0] if len(args) == 1 else None

            if hint is PathTemplate:
                value = getattr(self, f.name)
                if isinstance(value, str):
                    try:
                        resolved = Path(value.format(**context))
                        setattr(self, f.name, resolved)
                        context[f.name] = str(resolved)   # update for chained templates
                        if hasattr(self, "_sources"):
                            self._sources[f.name] += " (template resolved)"
                    except KeyError as e:
                        raise KeyError(
                            f"{self.__class__.__name__}.{f.name}: "
                            f"template reference {e} not found. "
                            f"Available keys: {list(context.keys())}"
                        )
