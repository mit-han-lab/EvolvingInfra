"""Planner: online prefill/decode split adaptation.

- ``calibration``: step-time fits, capacity models and P/D role decisions (NumPy/SciPy);
- ``loop``: the live controller that applies decisions through the Router's role API;
- ``pd_ratio``: an explicitly invoked calibrate/recommend API for one homogeneous configuration;
- ``node_plan``: a fixed per-node P/D plan from a completed learning capture.

Run ``python -m evolving_infra.planner --help``. Importing this package loads no numerical library.
"""
