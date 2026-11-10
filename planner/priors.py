"""Predeclared prefill step-model shapes, used only when a worker's own fit is not identifiable.

Coefficients are ``[s/step, s/new token, s/causal attention pair, s/restored token]``, plus ``s/vision matmul
FLOP`` for a model that encodes images. The learning-end fit then learns one multiplier for that worker
(``fit.models.fit_prefill_scale``). A prior is a shape, not a substitute for the current run's own fit, and it
belongs to one model, GPU type and TP size; the config must name one (``prefill_shape_prior``). Build one for
another setup from a fit report with ``python -m evolving_infra.planner prior <calibration/report.json>``.
"""
import hashlib
import json
import math
import statistics

PREFILL_SHAPE_PRIORS = {
    # Coordinate-wise median of 12 measured per-worker four-term NNLS fits (minutes 0-30 of a
    # DeepSeek-V4-Flash TP4 run on H200, 12 initial P workers, symmetric workers).
    # Per-worker ranges: step 0.076-0.119 s, token 4.00-4.50e-5 s, pair 1.45-1.62e-10 s,
    # restore 2.2-6.2e-7 s; held-out total errors 0.1-2.2 %.
    "dsv4-flash-h200-tp4": {
        "coefficients": [0.08940072558102694, 4.27230666885636e-05, 1.5286943770073413e-10,
                         3.8771281940673816e-07],
        "method": "coordinate-wise median of the measured per_node_four_term_nnls prefill models",
        "scope": "Predeclared shape only for unidentifiable four-term fits; learn one current-worker multiplier",
    },
    # Coordinate-wise median of 12 measured per-worker four-term NNLS fits: two runs of GLM-5.2 (NVFP4 weights,
    # FP8 KV cache, DSA attention) at TP4 on GB300, Claude Code traces at 90 concurrent agents, 6 initial P workers
    # each, minutes 0-30. Per-worker ranges: step 0.016-0.053 s, token 2.67-3.61e-5 s, pair 1.52-1.69e-10 s,
    # restore 3.8-7.7e-7 s; held-out total errors 0.03-5.5 %. As a shape with one multiplier per worker
    # (0.98-1.01) it fits each worker's 28 learning windows with 2.3-4.1 % mean error.
    "glm52-gb300-tp4": {
        "coefficients": [0.03853496967002243, 3.094407928814123e-05, 1.580590642998097e-10,
                         5.557343163915861e-07],
        "method": "coordinate-wise median of the measured per_node_four_term_nnls prefill models",
        "scope": "Predeclared shape only for unidentifiable four-term fits; learn one current-worker multiplier",
    },
    # Coordinate-wise median of 12 measured per-worker five-term NNLS fits: two runs of Kimi K2.6 (NVFP4 weights,
    # FP8 KV cache, MLA attention, image encoder on every P worker) at TP4 on GB300, OSWorld computer-use traces
    # at 128 concurrent agents, 6 initial P workers each, minutes 0-30. The step and new-token terms trade off
    # between workers (each is 0 on some), so the shape is checked as used: with one multiplier per worker
    # (1.04-1.08) it fits each worker's 28 learning windows with 2.1-4.3 % mean error. Per-worker ranges: pair
    # 5.5-7.5e-9 s, restore 0-8.2e-5 s, vision 0-6.9e-15 s; held-out total errors 0.06-2.5 %.
    "kimi-k26-gb300-tp4": {
        "coefficients": [0.20594981169243032, 0.0, 6.377493229789121e-09, 3.446891527975689e-05,
                         4.11745639792328e-15],
        "method": "coordinate-wise median of the measured per_node_five_term_nnls prefill models",
        "scope": "Predeclared shape only for unidentifiable five-term fits; learn one current-worker multiplier",
    },
}
MEASURED_METHODS = ("per_node_four_term_nnls", "per_node_five_term_nnls")


def resolve_prior(cfg: dict) -> dict:
    """The predeclared P step-model shape: a built-in name or an inline ``{"coefficients": [...]}``.

    Used only for a worker whose own fit is not identifiable (``fit_prefill_scale`` learns one
    multiplier on it). An optional ``prefill_shape_prior_sha256`` in the config pins the exact prior.
    """
    prior = cfg.get("prefill_shape_prior")
    if prior is None:
        raise ValueError("prefill_shape_prior is required: one of "
                         f"{sorted(PREFILL_SHAPE_PRIORS)} or an inline {{\"coefficients\": [...]}}")
    if isinstance(prior, str):
        if prior not in PREFILL_SHAPE_PRIORS:
            raise ValueError(f"Unknown prefill shape prior {prior!r}; known: {sorted(PREFILL_SHAPE_PRIORS)}")
        prior = PREFILL_SHAPE_PRIORS[prior]
    coefficients = prior.get("coefficients") if isinstance(prior, dict) else None
    if (not isinstance(coefficients, list) or len(coefficients) not in (4, 5)
            or not all(isinstance(c, (int, float)) and math.isfinite(c) and c >= 0 for c in coefficients)):
        raise ValueError("prefill_shape_prior needs 4 (or 5) finite nonnegative coefficients")
    pinned = cfg.get("prefill_shape_prior_sha256")
    if pinned is not None and pinned != prior_sha256(prior):
        raise ValueError("Predeclared P model shape changed")
    return prior


def prior_sha256(prior: dict) -> str:
    return hashlib.sha256(json.dumps(prior, sort_keys=True).encode()).hexdigest()


def build_prior(report: dict) -> list[float]:
    """Coordinate-wise median of the measured per-worker P models (all four-term or all five-term) of a report."""
    rows = [m["coefficients"] for w in report["workers"].values() if w["prefill"]["origin"] == "measured"
            for m in w["prefill"]["models"] if m.get("method") in MEASURED_METHODS]
    if not rows:
        raise ValueError("No measured four- or five-term prefill models in the report")
    if len({len(r) for r in rows}) != 1:
        raise ValueError("The measured prefill models mix four- and five-term schemas")
    return [statistics.median(r[i] for r in rows) for i in range(len(rows[0]))]
