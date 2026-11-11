// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! HTTP service assembly, shared application state, routes, errors and metrics.
//! Handlers delegate node selection to scheduling policies and managed request
//! state to Backend; this module does not implement model computation or KV DMA.

pub mod app;
pub mod app_context;
pub mod error;
pub mod header_utils;
pub mod metrics;
pub mod routes;
