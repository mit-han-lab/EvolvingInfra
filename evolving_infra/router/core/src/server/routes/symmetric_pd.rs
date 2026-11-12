// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! `POST /symmetric_pd/place` - leg placement for an external ingress
//! (`docs/api-reference.md`). Byte-compatible with the sgl-model-gateway
//! contract: the response carries `p_leg`, `d_leg`, `local`, `transfer_id`,
//! `score_terms`, `marks{kv_export,kv_import}`, `targets`, `inputs`.

use crate::scheduler::{PlaceHttpRequest, PlaceHttpResponse};
use crate::server::app_context::AppContext;
use crate::server::error::ApiError;
use axum::extract::State;
use axum::Json;
use bytes::Bytes;
use std::sync::Arc;

/// 404 `symmetric_pd_disabled` without `--policy symmetric_pd`; 400 on malformed
/// JSON (router envelope, not axum's) and with `--symmetric-pd-p-select
/// cost-function`; placement errors per the contract.
///
/// # Inputs
/// `ctx`: shared application state containing the symmetric P/D scheduler.
/// `body`: JSON placement request, or empty bytes to use request defaults.
///
/// # Outputs
/// JSON placement response with selected legs and engine metadata, or `ApiError`.
/// Depending on the request, records a bounded legacy load lease; it does not
/// execute the engine requests or use the prepared callback lifecycle.
pub async fn place(
    State(ctx): State<Arc<AppContext>>,
    body: Bytes,
) -> Result<Json<PlaceHttpResponse>, ApiError> {
    let flex = ctx
        .symmetric_pd
        .as_ref()
        .ok_or(ApiError::SymmetricPdDisabled)?;
    if flex.cost_function_enabled() {
        // The cost ledger needs the dispatch and the end of each request;
        // this endpoint only places.
        return Err(ApiError::BadRequest(
            crate::scheduler::cost_function::PLACE_REFUSAL.into(),
        ));
    }
    let req: PlaceHttpRequest = if body.is_empty() {
        PlaceHttpRequest::default()
    } else {
        serde_json::from_slice(&body)
            .map_err(|e| ApiError::BadRequest(format!("invalid /symmetric_pd/place body: {e}")))?
    };
    let resp = flex.place_http(req).await?;
    Ok(Json(resp))
}
