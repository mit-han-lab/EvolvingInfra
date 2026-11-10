"""Adapter registry.

Every adapter module exposes:

``list_units(options) -> (units, warnings)``
    Parent-side enumeration of picklable work units, in corpus order.
``prepare_options(options, units) -> None``
    Optional parent-side pass over the *selected* units (e.g. computing the
    corpus start epoch); mutates ``options`` in place.
``make_context(options, renderer) -> object``
    Worker-side per-process context (extra tokenizers, scaffold caches).
``convert_unit(unit, context, options, stats) -> Iterator[CanonicalTrace]``
    Worker-side conversion; mutates ``stats`` counters for source counts and
    adapter-level drops.

An adapter with ``ROWS = True`` yields finished schema-v2 rows from
``convert_unit`` and keeps its own statistics; the text builder is skipped.
"""

from __future__ import annotations

import importlib
from types import ModuleType

NAMES = (
    "trie_text",
    "trie_retokenize",
    "synthetic",
    "cc_weka",
)


def get(name: str) -> ModuleType:
    if name not in NAMES:
        raise KeyError(f"unknown adapter: {name!r}")
    return importlib.import_module(f"strict_convert.adapters.{name}")
