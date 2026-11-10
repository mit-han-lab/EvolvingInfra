# Adapted from SGLang python/sglang/srt/utils/json_response.py (commit 9a489f8d2f), modified for evolving-infra.
"""Utilities for JSON serialization in HTTP responses."""

from typing import Any

import orjson
from fastapi.responses import Response

# Keep response serialization behavior consistent across endpoints:
# - Support non-string dictionary keys used in some metadata payloads.
# - Support numpy scalars/arrays without pre-conversion.
ORJSON_RESPONSE_OPTIONS = orjson.OPT_NON_STR_KEYS | orjson.OPT_SERIALIZE_NUMPY


def dumps_json(content: Any) -> bytes:
    """Serialize content to JSON bytes using SGLang's ORJSON options."""
    return orjson.dumps(content, option=ORJSON_RESPONSE_OPTIONS)


def first_prompt_ids_only(out: dict[str, Any], seen: set[int]) -> dict[str, Any]:
    """Keep prompt IDs on the first emitted frame of each output sequence.

    Call after pending outputs have been coalesced, with a set owned by one
    HTTP stream. Copy only the frame dictionary so producer/logging state is
    unchanged and the prompt token array is never copied here.
    """
    if out.get("prompt_token_ids") is None:
        return out
    index = out.get("index", 0)
    if index not in seen:
        seen.add(index)
        return out
    frame = dict(out)
    frame.pop("prompt_token_ids", None)
    return frame


class SGLangORJSONResponse(Response):
    """ORJSON response with SGLang-specific serialization options."""

    media_type = "application/json"

    def render(self, content: Any) -> bytes:
        return dumps_json(content)


def orjson_response(content: Any, status_code: int = 200) -> Response:
    """Create a JSON response with stable ORJSON serialization options."""
    return SGLangORJSONResponse(content=content, status_code=status_code)
