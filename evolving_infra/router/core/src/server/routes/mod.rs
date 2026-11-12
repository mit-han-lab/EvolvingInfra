// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Route modules for generation, input preparation, placement and health.
//! Ordinary generation and prepared-input lifecycle endpoints remain distinct;
//! handlers share the application's scheduler and Backend dependencies.

pub mod cache;
pub mod chat;
pub mod completions;
pub mod decision_log;
pub mod health;
pub mod kv_session;
pub(crate) mod late_d;
pub mod metrics;
pub mod models;
pub(crate) mod pd_replay;
pub mod symmetric_pd;
pub mod tokenize;

pub mod prepared;
