"""Northbridge fictional legal research demo package."""

__all__ = ["run_query", "run_suite", "has_openai_key"]


def __getattr__(name: str):
    if name in __all__:
        from . import runner

        return getattr(runner, name)
    raise AttributeError(name)
