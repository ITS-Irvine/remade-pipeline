from enum import Enum
from pathlib import Path
from typing import Callable, List, Tuple, get_type_hints

import typer

from core.common import is_list_str_instance
from config.path_template import PathTemplate


def parse_commalist_callback(
    _VALID: dict[str, Enum],
    allow_all=None
) -> Callable[[typer.Context,typer.CallbackParam,List[str]],List[Enum]]:

    def _parse_validated_commalist_callback(
        ctx:   typer.Context,
        param: typer.CallbackParam,
        value: List[str],          # raw CLI strings, before any splitting
    ) -> List[Enum]:
        """
        # FIXME: writeup before generalization
        Accepts both repeated flags and comma-delimited values — or a mix:

            --output csv --output summary        # repeated flags
            --output csv,summary                 # comma-delimited
            --output csv,summary --output plots  # mixed
        """
        if not value:
            return []

        if not is_list_str_instance(value):
            raise RuntimeError(
                f"Expected 'argument to commalist to be a List[str], "
                f"but got {type(value)} with value: {value}"
            )

        resolved: list[Enum] = []
        invalid:  list[str]  = []

        for item in value:
            for part in (p.strip().lower() for p in item.split(",")):
                if allow_all is not None and part == allow_all:
                    # if all is in the list, just include everything
                    resolved += list(_VALID.values())
                    break
                if not part:                        # skip accidental empty tokens
                    continue
                if part in _VALID:
                    resolved.append(_VALID[part])
                else:
                    invalid.append(part)

        if invalid:
            valid_str = ", ".join(_VALID)
            raise typer.BadParameter(
                f"Invalid option(s): {', '.join(repr(i) for i in invalid)}.\n"
                f"Valid options are: {valid_str}.",
                ctx=ctx,
                param=param,
            )

        # Deduplicate while preserving order (dict preserves insertion order in 3.7+)
        return list(dict.fromkeys(resolved))
    return _parse_validated_commalist_callback


#################

import inspect
import functools
import dataclasses
from dataclasses import dataclass, fields
import typer
from typing import Any, Callable, Dict, Optional, Tuple, Type


# config/inject.py
import inspect
import functools
import typer
from dataclasses import fields
from typing import Any, Callable, Dict, Optional
from config.settings import settings
from config.base_config import _model_registry


def _build_typer_option(
    default:    Any,
    cli_flag:   str,
    typer_meta: dict,
    panel:      str = ""
) -> typer.models.OptionInfo:
    flags = [cli_flag]
    if typer_meta.get("short"):
        flags.append(typer_meta["short"])

    # Detect bool fields → add --no- counterpart
    if isinstance(default, bool):
        flags.append(f"--no-{cli_flag.lstrip('--')}")

    help_text = typer_meta.get("help", "")
    if default is not dataclasses.MISSING and default is not None:
        help_text += f" [default: {default}]"

    return typer.Option(
        None,
        *flags,
        help             = help_text,
        metavar          = typer_meta.get("metavar",      None),
        show_default     = False,
        hidden           = typer_meta.get("hidden",       False),
        envvar           = typer_meta.get("envvar",       None),
        rich_help_panel  = typer_meta.get("rich_help_panel") or panel or None,  # ← field wins over injected panel
        callback        = typer_meta.get("callback",        None),   # ← added
    )
    
def inject_model_options(func: Callable) -> Callable:
    sig         = inspect.signature(func)
    annotations = dict(getattr(func, "__annotations__", {}))
    hints       = get_type_hints(func)
    model_ns      = settings.get("model") or {}   # ← top-level [run] section

    new_params = []
    for name, p in sig.parameters.items():
        if name == "model_configs":
            continue

        # ── Remap PathTemplate → str for Typer on top-level params ───────────
        hint = hints.get(name)
        if isinstance(hint, type) and issubclass(hint, PathTemplate):
            dynaconf_default = model_ns.get(name)
            if dynaconf_default is not None:
                p = p.replace(
                    annotation = Optional[str],
                    default    = typer.Option(
                                     dynaconf_default,
                                     *p.default.param_decls,
                                     help            = p.default.help,
                                     rich_help_panel = p.default.rich_help_panel,
                                 )
                )
            else:
                p = p.replace(annotation=Optional[str])
            annotations[name] = Optional[str]

        new_params.append(p)

    # ── Inject per-submodel CLI options ──────────────────────────────────────
    for label, config_cls in _model_registry:
        prefix       = config_cls._cli_prefix                          # ← pull prefix from class
        dynaconf_ns  = settings.get(f"model.{prefix}") or {} if prefix else settings.get("model") or {}
        panel        = f"{label.upper()} model options"                # ← use label for display
        config_hints = get_type_hints(config_cls)

        for f in fields(config_cls):

            # Skip fields that aren't tagged with cli_field()
            if "typer" not in f.metadata:
                continue

            param_name = f.name if not prefix else f"{prefix}_{f.name}"
            cli_flag   = f"--{f.name.replace('_', '-')}" if not prefix else f"--{prefix.replace('_', '-')}-{f.name.replace('_', '-')}"

            typer_meta = f.metadata.get("typer", {})

            # Priority: Dynaconf → dataclass default
            default  = dynaconf_ns.get(f.name, f.default)

            # Resolve the CLI-facing type — PathTemplate is just a str on the CLI
            raw_hint = config_hints.get(f.name, f.type)
            cli_type = str if (isinstance(raw_hint, type) and issubclass(raw_hint, PathTemplate)) else raw_hint

            annotations[param_name] = Optional[cli_type]
            new_params.append(
                inspect.Parameter(
                    name       = param_name,
                    kind       = inspect.Parameter.KEYWORD_ONLY,
                    default    = _build_typer_option(default, cli_flag, typer_meta, panel),
                    annotation = Optional[cli_type],
                )
            )

    @functools.wraps(func)
    def wrapper(**kwargs):

        # ── Resolve PathTemplate top-level params ─────────────────────────────
        context = {k: str(v) for k, v in kwargs.items() if v is not None}
        for name, hint in hints.items():
            if isinstance(hint, type) and issubclass(hint, PathTemplate):
                value = kwargs.get(name)
                if isinstance(value, str):
                    try:
                        resolved      = Path(value.format(**context))
                        kwargs[name]  = resolved
                        context[name] = str(resolved)   # update for chained templates
                    except KeyError as e:
                        raise KeyError(
                            f"run parameter '{name}': "
                            f"template reference {e} not found. "
                            f"Available keys: {list(context.keys())}"
                        )

        # ── Build submodel configs ────────────────────────────────────────────
        model_configs: Dict[str, Any] = {}
        for label, config_cls in _model_registry:          # ← label instead of prefix
            prefix        = config_cls._cli_prefix          # ← prefix drives param_name
            cli_overrides = {}
            for f in fields(config_cls):
                param_name = f.name if not prefix else f"{prefix}_{f.name}"  # ← but still uses prefix for CLI kwarg lookup ← but no leading _ if no prefix
                if param_name in kwargs:
                    cli_overrides[f.name] = kwargs.pop(param_name)

            model_configs[label] = config_cls.from_dynaconf_with_overrides(  # ← label as key
                **cli_overrides
            )

        return func(**kwargs, model_configs=model_configs)

    wrapper.__signature__   = sig.replace(parameters=new_params)
    wrapper.__annotations__ = annotations
    return wrapper
