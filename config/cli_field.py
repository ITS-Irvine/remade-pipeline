# config/cli_field.py
import dataclasses
from typing import Any, Callable, Optional, TypeVar, overload

_MISSING = dataclasses.MISSING
T = TypeVar("T")


@overload
def cli_field(
    default: T,
    *,
    help:         str                   = ...,
    short:        Optional[str]         = ...,
    metavar:      Optional[str]         = ...,
    show_default: bool                  = ...,
    hidden:       bool                  = ...,
    envvar:       Optional[str]         = ...,
    rich_help_panel: Optional[str]      = ...,
    callback:        Optional[Callable] = ...,   # ← added
) -> T: ...                                    # ← returns T, preserving the type


@overload
def cli_field(
    *,
    default_factory: Callable[[], T],          # ← for mutable defaults e.g. dict, list
    help:            str                = ...,
    short:           Optional[str]      = ...,
    metavar:         Optional[str]      = ...,
    show_default:    bool               = ...,
    hidden:          bool               = ...,
    envvar:          Optional[str]      = ...,
    rich_help_panel: Optional[str]      = ...,
    callback:        Optional[Callable] = ...,   # ← added
) -> T: ...


def cli_field(
    default:         Any                = _MISSING,
    *,
    default_factory: Any                = _MISSING,
    help:            str                = "",
    short:           Optional[str]      = None,
    metavar:         Optional[str]      = None,
    show_default:    bool               = True,
    hidden:          bool               = False,
    envvar:          Optional[str]      = None,
    rich_help_panel: Optional[str]      = None,
    callback:        Optional[Callable] = None,   # ← added
) -> Any:
    typer_meta = {
        "help":            help,
        "short":           short,
        "metavar":         metavar,
        "show_default":    show_default,
        "hidden":          hidden,
        "envvar":          envvar,
        "rich_help_panel": rich_help_panel,
        "callback":         callback,              # ← added
    }

    field_kwargs: dict[str, Any] = {"metadata": {"typer": typer_meta}}

    if default is not _MISSING:
        field_kwargs["default"] = default
    elif default_factory is not _MISSING:
        field_kwargs["default_factory"] = default_factory

    return dataclasses.field(**field_kwargs)
