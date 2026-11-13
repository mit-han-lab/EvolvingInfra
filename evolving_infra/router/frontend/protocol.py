# SPDX-FileCopyrightText: Copyright (c) 2026 The evolving-infra Authors
# SPDX-License-Identifier: MIT
"""Versioned tokenizer envelopes shared by the service and forwarding code."""

from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

Identifier = Annotated[str, Field(min_length=1, max_length=512)]
Positive = Annotated[int, Field(gt=0)]

class NodeConfig(BaseModel):
    """Validate the fixed settings for one tokenizer node.

    Input: node/model identities, endpoints, hashing settings and limits.
    Output: an immutable NodeConfig instance with defaults applied.
    """
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    node_id: Identifier
    config_id: Identifier
    model_path: str
    tokenize_endpoint: str
    targets: dict[str, str]
    page_size: Positive = 16
    context_length: Positive = 4096
    cache_salt: str | None = None
    extra_key: str | None = None
    hash_mode: Literal["sglang_native", "dynamo"] = "sglang_native"
    is_eagle: bool = False
    # Fixed checkpoint preprocessing fields, including its image placeholder ID.
    # None retains the text-only contract. No per-request preprocessing overrides.
    image_config: dict | None = None
    public_url: str | None = None  # Worker-reachable address for failed-import image recovery.
    callback_targets: list[str] = Field(default_factory=list)
    callback_timeout_seconds: Positive = 2
    callback_outbox_max_plans: Positive = 1024
    max_body_bytes: Positive = 16777216
    max_staging_bytes: Positive = 536870912
    ttl_seconds: Positive = 300
    request_timeout_seconds: Positive = 7200
    max_response_bytes: Positive = 16777216


class Identity(BaseModel):
    """Validate the identity fields shared by POST requests.

    Input: protocol_version, request_id and config_id. Output: an Identity instance.
    """
    model_config = ConfigDict(extra="forbid", strict=True)
    protocol_version: Literal[1]
    request_id: Identifier
    config_id: Identifier


class Prepare(Identity):
    """Represent an input-preparation request without starting generation.

    Input: identity fields, input and generation dictionaries. Output: a Prepare instance.
    """
    input: dict
    generation: dict


class Release(Identity):
    """Identify a locally prepared result for release or lookup.

    Input: identity fields and ref. Output: a Release instance.
    """
    ref: Identifier


class ImageRequest(Release):
    """Fetch one original image while its prepared request is still owned."""
    image_index: Annotated[int, Field(ge=0)]


class Target(BaseModel):
    """Describe a caller-selected SGLang generation endpoint.

    Input: transport, endpoint and config_id. Output: a Target instance.
    """
    model_config = ConfigDict(extra="forbid", strict=True)
    transport: Literal["sglang_generate"]
    endpoint: str
    config_id: Identifier


class Forward(Release):
    """Represent a request to send a prepared input to a selected target.

    Input: identity fields, ref and target. Output: a Forward instance.
    """
    target: Target


class KvSession(BaseModel):
    """Input: router-issued session/version. Output: validated engine retention metadata."""

    model_config = ConfigDict(extra="forbid", strict=True)
    session_id: Identifier
    version: Annotated[int, Field(gt=0, le=2**63 - 1)]


class PrefillLeg(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    target: Target
    kv_export: dict
    pull_hint: dict | None = None
    kv_session: KvSession | None = None
    # Late D choose: the router's symmetric `late_d` field (k, default_idx).
    late_d: dict | None = None


class DecodeLeg(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    target: Target
    kv_import: dict
    kv_session: KvSession | None = None


class DecodeCandidate(BaseModel):
    """One late-D decode candidate: its own import marks and `late_d.cand_idx`."""
    model_config = ConfigDict(extra="forbid", strict=True)
    target: Target
    kv_import: dict
    late_d: dict


class Callback(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    endpoint: str
    token: Identifier = Field(repr=False)


class ForwardPD(Release):
    """An already chosen P/D plan. No placement or engine retries occur here."""

    prefill: PrefillLeg
    decode: DecodeLeg
    callback: Callback
    reservation_id: Identifier
    transfer_id: Annotated[int, Field(ge=0, le=2**64 - 1)]
    # Late D choose: one D-leg per candidate, sent together with the P-leg. The
    # prefill admits one (HTTP 2xx: the decode stream) and releases the others
    # (HTTP 409, X-Late-D-Reason: released). `decode` is the arrival pick.
    decode_candidates: list[DecodeCandidate] | None = None

    @model_validator(mode="after")
    def matching_transfer(self):
        """Validate that both KV leg markers use this plan's transfer ID.

        Args:
            None.

        Returns:
            This validated ForwardPD instance.

        Raises:
            ValueError: A prefill/export or decode/import transfer ID differs.
        """
        if any(mark.get("transfer_id") != self.transfer_id for mark in
               (self.prefill.kv_export, self.decode.kv_import)):
            raise ValueError("P/D transfer IDs must match the plan")
        if self.decode_candidates is None:
            if self.prefill.late_d is not None:
                raise ValueError("A late-D prefill needs decode candidates")
            if self.prefill.kv_session != self.decode.kv_session:
                raise ValueError("P/D KV session metadata must match")
            return self
        # Late D: the prefill forwards its session tag to the admitted decode.
        k = len(self.decode_candidates)
        if self.prefill.late_d is None or not 1 <= k <= 64:
            raise ValueError("Late D needs the prefill late_d field and 1..64 candidates")
        if self.decode.kv_session is not None:
            raise ValueError("Late-D decode legs carry no KV session tag")
        if self.prefill.late_d.get("k") != k:
            raise ValueError("late_d.k differs from the candidate count")
        for i, cand in enumerate(self.decode_candidates):
            if cand.kv_import.get("transfer_id") != self.transfer_id:
                raise ValueError("Candidate transfer IDs must match the plan")
            if cand.late_d.get("cand_idx") != i or cand.late_d.get("k") != k:
                raise ValueError("Candidate late_d fields must follow the candidate order")
        return self
