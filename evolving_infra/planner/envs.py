"""Environment variables for planner internals, in the style of vLLM's ``envs.py``.

These are not part of the method: HTTP timeouts and how many round trips a clock probe takes. Read one as
``envs.EVOLVING_INFRA_PLANNER_ROLE_API_TIMEOUT_SECS``; each read looks at the environment again. Launch parameters
of the method are flags (``planner_args``).
"""
import os


def _float(name, default):
    return lambda: float(os.environ.get(name, default))


def _int(name, default):
    return lambda: int(os.environ.get(name, default))


environment_variables = {
    # Router role registry, GET and PUT /symmetric_pd/roles.
    "EVOLVING_INFRA_PLANNER_ROLE_API_TIMEOUT_SECS": _float("EVOLVING_INFRA_PLANNER_ROLE_API_TIMEOUT_SECS", 5),
    # Router /symmetric_pd/backend/loads, read before each decision.
    "EVOLVING_INFRA_PLANNER_BACKEND_LOADS_TIMEOUT_SECS":
        _float("EVOLVING_INFRA_PLANNER_BACKEND_LOADS_TIMEOUT_SECS", 5),
    # Engine /server_info (KV capacity, request cap, TP size), read at start.
    "EVOLVING_INFRA_PLANNER_SERVER_INFO_TIMEOUT_SECS": _float("EVOLVING_INFRA_PLANNER_SERVER_INFO_TIMEOUT_SECS", 30),
    # Engine /model_info (model path for the expert counts), read by the fit.
    "EVOLVING_INFRA_PLANNER_MODEL_INFO_TIMEOUT_SECS": _float("EVOLVING_INFRA_PLANNER_MODEL_INFO_TIMEOUT_SECS", 5),
    # Engine /metrics scrape for the restored-token counters.
    "EVOLVING_INFRA_PLANNER_METRICS_TIMEOUT_SECS": _float("EVOLVING_INFRA_PLANNER_METRICS_TIMEOUT_SECS", 4),
    # Trace server /list and /read (copying a worker's trace files).
    "EVOLVING_INFRA_PLANNER_TRACE_LIST_TIMEOUT_SECS": _float("EVOLVING_INFRA_PLANNER_TRACE_LIST_TIMEOUT_SECS", 30),
    "EVOLVING_INFRA_PLANNER_TRACE_READ_TIMEOUT_SECS": _float("EVOLVING_INFRA_PLANNER_TRACE_READ_TIMEOUT_SECS", 120),
    # Trace server /time: connection timeout and round trips per clock probe (the shortest one is kept).
    "EVOLVING_INFRA_PLANNER_CLOCK_PROBE_TIMEOUT_SECS": _float("EVOLVING_INFRA_PLANNER_CLOCK_PROBE_TIMEOUT_SECS", 15),
    "EVOLVING_INFRA_PLANNER_CLOCK_PROBE_EXCHANGES": _int("EVOLVING_INFRA_PLANNER_CLOCK_PROBE_EXCHANGES", 8),
}


def __getattr__(name):
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(environment_variables)
