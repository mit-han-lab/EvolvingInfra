# SPDX-FileCopyrightText: Copyright (c) 2026 The evolving-infra Authors
# SPDX-License-Identifier: MIT
"""Where every worker and service runs, and on which ports."""
from dataclasses import dataclass

ROUTER_PORT = 30900
GATEWAY_PORT = 30800  # agent gateway: identities for coding-agent harnesses (router/gateway)
FRONTEND_PORT = 28100
INDEXER_PORT, INDEXER_QUERY_PORT = 8090, 8091
TRACE_PORT = 18500


EVENTS_STRIDE = 10  # KV-event ports per worker on a host (PUB and replay of every DP rank)


@dataclass(frozen=True)
class Worker:
    idx: int          # global, 0..n-1
    host: int         # index into the host list
    local: int        # index on its host
    ip: str
    gpus: int         # tensor parallel size
    role: str         # starting role: "prefill" or "decode"
    dp: int = 1       # DP-attention ranks; each publishes its own KV events

    @property
    def name(self) -> str:
        return f"w{self.idx}"

    @property
    def base_gpu(self) -> int:
        return self.local * self.gpus

    @property
    def port(self) -> int:
        return 30000 + 1000 * self.local

    @property
    def bootstrap_port(self) -> int:  # two ports per GPU from here
        return 8998 + 1000 * self.local

    @property
    def servicer_port(self) -> int:  # one port per GPU from here
        return 18100 + 1000 * self.local

    @property
    def events_port(self) -> int:  # KV events of DP rank r on events_port + r
        return 5557 + EVENTS_STRIDE * self.local

    @property
    def replay_port(self) -> int:  # their replay channels, after the event ports
        return self.events_port + self.dp

    @property
    def late_decode_port(self) -> int:  # the link over which a prefill worker binds its decode worker
        return 21776 + 1000 * self.local

    def ports(self) -> list:
        """Every port this worker binds on its host."""
        return ([self.port, self.late_decode_port]
                + list(range(self.events_port, self.events_port + 2 * self.dp))
                + list(range(self.bootstrap_port, self.bootstrap_port + 2 * self.gpus))
                + list(range(self.servicer_port, self.servicer_port + self.gpus)))

    @property
    def url(self) -> str:
        return f"http://{self.ip}:{self.port}"


def workers(ips: list, gpus_per_host: int, gpus_per_worker: int, prefill: int, dp_attention: bool = False) -> list:
    """Workers host by host; the first ``prefill`` start as P, the others as D. With
    ``dp_attention`` every GPU of a worker is one DP-attention rank."""
    per_host = gpus_per_host // gpus_per_worker
    if per_host < 1:
        raise ValueError(f"{gpus_per_worker} GPUs per worker but {gpus_per_host} GPUs on a host")
    dp = gpus_per_worker if dp_attention else 1
    if 2 * dp > EVENTS_STRIDE:
        raise ValueError(f"{dp} DP ranks per worker need {2 * dp} KV-event ports, the layout has {EVENTS_STRIDE}")
    out = []
    for h, ip in enumerate(ips):
        for local in range(per_host):
            idx = len(out)
            out.append(Worker(idx, h, local, ip, gpus_per_worker, "prefill" if idx < prefill else "decode", dp))
    if len(out) < 2:
        raise ValueError("P/D needs at least two workers")
    if not 1 <= prefill < len(out):
        raise ValueError(f"--prefill-workers must be between 1 and {len(out) - 1}, got {prefill}")
    return out
