"""Parse the router-set per-request marks of symmetric P/D symmetric mode.

`kv_export` marks the P-leg request, `kv_import` marks the D-leg request, neither
means a Local request; a request never carries both.

Pure Python on purpose (no torch) so the CPU protocol tests can import it.
"""

from __future__ import annotations

from typing import Any, List, Mapping, Optional, Tuple

from decoupled_engine.srt.disaggregation.pull.api import (
    IMPORT_CACHE_POLICIES,
    ExportCandidate,
    ExportMark,
    HostReturn,
    ImportMark,
    validate_attempt_id,
)

# Late D stage 2: the router link's eligible mask is a u64 (bit = cand_idx).
MAX_EXPORT_CANDIDATES = 64


class MarkError(ValueError):
    """A malformed or contradictory kv_export / kv_import pair (HTTP 400)."""


def _require_int(mark: Mapping[str, Any], key: str, name: str) -> int:
    value = mark.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise MarkError(f"{name}.{key} must be an integer, got {value!r}")
    if value < 0:
        raise MarkError(f"{name}.{key} must be non-negative, got {value!r}")
    return value


def _require_str(mark: Mapping[str, Any], key: str, name: str) -> str:
    value = mark.get(key)
    if not isinstance(value, str) or not value:
        raise MarkError(f"{name}.{key} must be a non-empty string, got {value!r}")
    return value


def _optional_attempt_id(raw: Mapping[str, Any], name: str) -> Optional[int]:
    if "attempt_id" not in raw:
        return None
    attempt_id = raw["attempt_id"]
    try:
        if attempt_id is None:
            raise ValueError("attempt_id must be a nonzero uint64 integer")
        validate_attempt_id(attempt_id)
    except ValueError as exc:
        raise MarkError(f"{name}.{exc}") from exc
    return attempt_id


def _parse_candidates(raw: Mapping[str, Any]) -> List[ExportCandidate]:
    items = raw.get("candidates")
    if not isinstance(items, list) or not items:
        raise MarkError("kv_export.candidates must be a non-empty list")
    if len(items) > MAX_EXPORT_CANDIDATES:
        raise MarkError(
            f"kv_export.candidates has {len(items)} entries; at most "
            f"{MAX_EXPORT_CANDIDATES}"
        )
    out: List[ExportCandidate] = []
    for i, item in enumerate(items):
        name = f"kv_export.candidates[{i}]"
        if not isinstance(item, Mapping):
            raise MarkError(f"{name} must be an object, got {type(item).__name__}")
        cand_idx = _require_int(item, "cand_idx", name)
        if cand_idx >= MAX_EXPORT_CANDIDATES:
            raise MarkError(
                f"{name}.cand_idx must be below {MAX_EXPORT_CANDIDATES}, "
                f"got {cand_idx}"
            )
        attempt_id = _optional_attempt_id(item, name)
        if attempt_id is None:
            raise MarkError(f"{name}.attempt_id is required")
        out.append(
            ExportCandidate(
                cand_idx=cand_idx,
                reader_worker=_require_str(item, "reader_worker", name),
                reader_addr=_require_str(item, "reader_addr", name),
                attempt_id=attempt_id,
            )
        )
    if len({c.cand_idx for c in out}) != len(out):
        raise MarkError("kv_export.candidates repeat a cand_idx")
    if len({c.attempt_id for c in out}) != len(out):
        raise MarkError("kv_export.candidates repeat an attempt_id")
    return out


def parse_export_mark(raw: Mapping[str, Any]) -> ExportMark:
    """One of two forms: the scalar reader fields, or (late D stage 2)
    ``candidates`` without them and without a top-level attempt_id."""
    if not isinstance(raw, Mapping):
        raise MarkError(f"kv_export must be an object, got {type(raw).__name__}")
    transfer_id = _require_int(raw, "transfer_id", "kv_export")
    if "candidates" in raw:
        scalar = [k for k in ("reader_worker", "reader_addr", "attempt_id") if k in raw]
        if scalar:
            raise MarkError(
                f"kv_export carries candidates and {', '.join(scalar)}; "
                "a candidate export has no scalar reader fields"
            )
        return ExportMark(transfer_id=transfer_id, candidates=_parse_candidates(raw))
    return ExportMark(
        transfer_id=transfer_id,
        reader_worker=_require_str(raw, "reader_worker", "kv_export"),
        reader_addr=_require_str(raw, "reader_addr", "kv_export"),
        attempt_id=_optional_attempt_id(raw, "kv_export"),
    )


def parse_import_mark(raw: Mapping[str, Any]) -> ImportMark:
    if not isinstance(raw, Mapping):
        raise MarkError(f"kv_import must be an object, got {type(raw).__name__}")
    expected_tokens = raw.get("expected_tokens")
    if expected_tokens is not None:
        expected_tokens = _require_int(raw, "expected_tokens", "kv_import")
    cache_policy = raw.get("cache_policy", "retain")
    if cache_policy not in IMPORT_CACHE_POLICIES:
        raise MarkError(
            f"kv_import.cache_policy must be one of {IMPORT_CACHE_POLICIES}, "
            f"got {cache_policy!r}"
        )
    host_return = None
    if raw.get("host_return") is not None:
        directive = raw["host_return"]
        fields = {
            "destination_servicer_addr",
            "source_servicer_addr",
            "timeout_ms",
            "max_tokens",
        }
        if not isinstance(directive, Mapping) or set(directive) != fields:
            raise MarkError(
                "kv_import.host_return must contain exactly "
                + ", ".join(sorted(fields))
            )
        try:
            host_return = HostReturn(**directive)
        except ValueError as exc:
            raise MarkError(f"kv_import.host_return: {exc}") from exc
    try:
        return ImportMark(
            transfer_id=_require_int(raw, "transfer_id", "kv_import"),
            owner_worker=_require_str(raw, "owner_worker", "kv_import"),
            owner_addr=_require_str(raw, "owner_addr", "kv_import"),
            expected_tokens=expected_tokens,
            cache_policy=cache_policy,
            attempt_id=_optional_attempt_id(raw, "kv_import"),
            host_return=host_return,
        )
    except MarkError:
        raise
    except ValueError as exc:
        raise MarkError(f"kv_import: {exc}") from exc


def parse_kv_marks(
    kv_export: Optional[Mapping[str, Any]],
    kv_import: Optional[Mapping[str, Any]],
) -> Tuple[Optional[ExportMark], Optional[ImportMark]]:
    """Return (export_mark, import_mark); at most one is set.

    Raises MarkError when both marks are present ("a request never
    carries both") or when a mark is malformed. Callers map MarkError to 400.
    """
    if kv_export is not None and kv_import is not None:
        raise MarkError(
            "a request cannot carry both kv_export and kv_import "
            "(P-leg and D-leg are separate requests)"
        )
    export_mark = parse_export_mark(kv_export) if kv_export is not None else None
    import_mark = parse_import_mark(kv_import) if kv_import is not None else None
    return export_mark, import_mark
