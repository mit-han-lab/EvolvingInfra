// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Full HTTP proxy integration tests.
//!
//! Each submodule spins up the router via `build_router(AppContext)` and
//! drives real requests through a `common::mock_worker::MockWorker`
//! backend. For component-scope tests that don't need the router, see
//! `tests/component/`.

mod common;

mod cache_aware_input_ids;
mod chat_routing;
mod completions_image_layout;
mod decode_po2_smg;
mod decode_token_balance;
mod external_indexer_routing;
mod failover;
mod gateway_ca_port;
mod global_queue;
mod graceful_shutdown;
mod header_forwarding;
mod kv_hint_injection;
mod kv_session_port;
mod late_d_choose;
mod no_circuit_breaker;
mod pd_bootstrap_injection;
mod pd_pool_isolation;
mod pd_prefill_failure;
mod roundrobin_input_ids;
mod sticky_input_ids;
mod sticky_routing;
mod symmetric_pd_cost_function;
mod symmetric_pd_late_d;
mod symmetric_pd_routing;
mod timeout;
