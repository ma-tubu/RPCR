"""Model package with lazy access to legacy raw-input components."""

from typing import Any

__all__ = [
    "get_swin_classifier",
    "PromptedSwinTransformer",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from .swin import PromptedSwinTransformer, get_swin_classifier

        exports = {
            "get_swin_classifier": get_swin_classifier,
            "PromptedSwinTransformer": PromptedSwinTransformer,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
