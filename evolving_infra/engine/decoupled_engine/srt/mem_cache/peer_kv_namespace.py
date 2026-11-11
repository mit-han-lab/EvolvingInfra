"""Namespace peer host pages without changing the storage/event hash protocols."""

import hashlib
import json
from typing import Optional, Sequence


def peer_page_keys(
    hashes: Sequence[str],
    cache_salt: Optional[str] = None,
    extra_key: Optional[str] = None,
) -> list[str]:
    """Use the same trusted RadixKey namespace at publication and lookup.

    Keep unsalted, unqualified keys compatible. A tagged, length-unambiguous
    encoding distinguishes None from an empty salt and salt from extra_key.
    These keys are private to P2P; storage and KV-event hashes stay unchanged.
    """
    if cache_salt is None and extra_key is None:
        return list(hashes)
    namespace = json.dumps(
        [cache_salt, extra_key], ensure_ascii=True, separators=(",", ":")
    ).encode("utf-8")
    prefix = "p2p-v1:" + hashlib.sha256(namespace).hexdigest() + ":"
    return [prefix + h for h in hashes]
