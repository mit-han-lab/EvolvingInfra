"""Predeclared prefill step-model shapes for ``evolving_infra.planner.loop``.

Coefficients are ``[s/step, s/new token, s/causal attention pair, s/restored token]``. A prior is used
only for a worker whose own four-term fit is not identifiable; the controller then learns one
multiplier for that worker (``calibration.nodes.fit_prefill_scale``). A prior is a shape, not a
substitute for the current run's own fit. Rebuild one from a fit report with
``python -m evolving_infra.planner prior <calibration/report.json>``.
"""

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
}
