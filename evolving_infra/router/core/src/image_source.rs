// SPDX-FileCopyrightText: Copyright (c) 2026 The evolving-infra Authors
// SPDX-License-Identifier: Apache-2.0

//! Pixel-free late-D decode legs of image requests (`--late-d-image-source-url`).
//!
//! With `--pd-late-decode choose` every decode candidate receives the whole
//! request. A decode leg imports its KV from the prefill and never runs the
//! vision tower, yet with `image_data` in its body the engine decodes every
//! screenshot, preprocesses it on the GPU (float32 patches, ~25 MB per
//! 1080p screenshot), copies the features to /dev/shm, and every TP rank of
//! the decode clones them. With this module the router sends the decode
//! legs of a request that carries a `prepared_image_layout` (see
//! [`crate::image_layout`]) WITHOUT `image_data`, plus
//! `prepared_image_source = {"endpoint": URL, "token": <rid>}`. The engine
//! (Kimi `KimiK2_5VLImageProcessor`, metadata-only branch) then builds
//! pixel-free image items from the layout (pad values, offsets, grids).
//!
//! Only a decode that must recompute locally (a retraction, a failed
//! import) needs the pixels: it POSTs `{"token", "image_index"}` to URL
//! ([`fetch`], `POST /internal/prepared_image`) and gets the image file
//! bytes, which it checks against the layout's sha256. The router keeps a
//! request's data URIs in memory from dispatch until the request ends
//! ([`Hold`] is dropped with the client's response stream), so a fetch
//! after that answers 404.
//!
//! Prefill legs are unchanged (they run the vision tower). A request whose
//! token is already held by another in-flight request keeps the pixels on
//! its decode legs (counted as `collision`).

use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use dashmap::mapref::entry::Entry;
use dashmap::DashMap;
use serde::Deserialize;
use serde_json::{Map, Value};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

/// The body field the engine reads for the image owner.
pub const SOURCE_KEY: &str = "prepared_image_source";
/// The router route that serves the original image bytes.
pub const FETCH_PATH: &str = "/internal/prepared_image";

fn store() -> &'static DashMap<String, Arc<Vec<Value>>> {
    static STORE: OnceLock<DashMap<String, Arc<Vec<Value>>>> = OnceLock::new();
    STORE.get_or_init(DashMap::new)
}

static STRIPPED_REQUESTS: AtomicU64 = AtomicU64::new(0);
static STRIPPED_IMAGES: AtomicU64 = AtomicU64::new(0);
static COLLISIONS: AtomicU64 = AtomicU64::new(0);
static FETCH_OK: AtomicU64 = AtomicU64::new(0);
static FETCH_MISSING: AtomicU64 = AtomicU64::new(0);
static FETCH_BAD: AtomicU64 = AtomicU64::new(0);
static FETCH_BYTES: AtomicU64 = AtomicU64::new(0);

/// One request's images in the store; dropping it removes them.
#[must_use = "dropping the hold removes the request's images at once"]
pub struct Hold {
    token: String,
}

impl Hold {
    /// Reserve `token` (the request id) for one request. `None` when another
    /// in-flight request holds the same token: this request's decode legs
    /// then keep their pixels.
    pub fn reserve(token: &str) -> Option<Hold> {
        match store().entry(token.to_string()) {
            Entry::Occupied(_) => {
                COLLISIONS.fetch_add(1, Ordering::Relaxed);
                None
            }
            Entry::Vacant(slot) => {
                slot.insert(Arc::new(Vec::new()));
                Some(Hold {
                    token: token.to_string(),
                })
            }
        }
    }

    /// Store the request's `image_data` entries (data URIs, in order).
    pub fn fill(&self, images: Vec<Value>) {
        STRIPPED_REQUESTS.fetch_add(1, Ordering::Relaxed);
        STRIPPED_IMAGES.fetch_add(images.len() as u64, Ordering::Relaxed);
        store().insert(self.token.clone(), Arc::new(images));
    }

    pub fn token(&self) -> &str {
        &self.token
    }
}

impl Drop for Hold {
    fn drop(&mut self) {
        store().remove(&self.token);
    }
}

/// The `prepared_image_source` value of a decode leg.
pub fn source_value(url: &str, token: &str) -> Value {
    serde_json::json!({ "endpoint": url, "token": token })
}

/// A request body with a non-null `prepared_image_layout` and a nonempty
/// `image_data` list: its late-D decode legs can go without pixels.
pub fn has_prepared_images(value: Option<&Value>) -> bool {
    value
        .and_then(Value::as_object)
        .is_some_and(has_layout_and_images)
}

fn has_layout_and_images(obj: &Map<String, Value>) -> bool {
    let layout = obj.get("prepared_image_layout");
    layout.is_some_and(|v| !v.is_null())
        && matches!(obj.get("image_data"), Some(Value::Array(a)) if !a.is_empty())
}

/// The decode-leg body change: drop `image_data`, add `source`. Changes
/// nothing (returns `None`) unless the body carries a non-null
/// `prepared_image_layout` and a nonempty `image_data` list; otherwise
/// returns the removed list.
pub fn strip_images(obj: &mut Map<String, Value>, source: &Value) -> Option<Vec<Value>> {
    if !has_layout_and_images(obj) {
        return None;
    }
    let Some(Value::Array(images)) = obj.remove("image_data") else {
        return None;
    };
    obj.insert(SOURCE_KEY.to_string(), source.clone());
    Some(images)
}

/// The file bytes of a base64 data URI (`data:<mime>;base64,<payload>`).
pub fn decode_data_uri(uri: &str) -> Option<Vec<u8>> {
    let (head, payload) = uri.split_once(',')?;
    if !head.starts_with("data:") || !head.ends_with(";base64") {
        return None;
    }
    crate::image_layout::base64_decode(payload.as_bytes())
}

/// `POST /internal/prepared_image` body (the engine also sends the other
/// fields of `prepared_image_source`, which are ignored).
#[derive(Debug, Deserialize)]
pub struct FetchRequest {
    pub token: String,
    pub image_index: usize,
}

/// The original bytes of one image of an in-flight request: 200 with the
/// file bytes; 404 when the request ended (or never stripped its decode
/// legs) or the index is out of range; 422 when the stored entry is not a
/// base64 data URI.
pub async fn fetch(Json(req): Json<FetchRequest>) -> Response {
    let images = store().get(&req.token).map(|e| Arc::clone(e.value()));
    let entry = images.as_ref().and_then(|v| v.get(req.image_index));
    let Some(entry) = entry else {
        FETCH_MISSING.fetch_add(1, Ordering::Relaxed);
        tracing::warn!(
            token = %req.token,
            image_index = req.image_index,
            "prepared image fetch: no such in-flight image"
        );
        return (StatusCode::NOT_FOUND, "no such in-flight image").into_response();
    };
    match entry.as_str().and_then(decode_data_uri) {
        Some(bytes) => {
            FETCH_OK.fetch_add(1, Ordering::Relaxed);
            FETCH_BYTES.fetch_add(bytes.len() as u64, Ordering::Relaxed);
            tracing::info!(
                token = %req.token,
                image_index = req.image_index,
                bytes = bytes.len(),
                "prepared image fetch"
            );
            (
                StatusCode::OK,
                [(header::CONTENT_TYPE, "application/octet-stream")],
                bytes,
            )
                .into_response()
        }
        None => {
            FETCH_BAD.fetch_add(1, Ordering::Relaxed);
            (
                StatusCode::UNPROCESSABLE_ENTITY,
                "stored image is not a base64 data URI",
            )
                .into_response()
        }
    }
}

/// Prometheus text for `/metrics`.
pub fn render_metrics() -> String {
    let c = |a: &AtomicU64| a.load(Ordering::Relaxed);
    format!(
        "# HELP sgl_router_late_d_image_source_total Pixel-free late-D decode legs (--late-d-image-source-url).\n\
         # TYPE sgl_router_late_d_image_source_total counter\n\
         sgl_router_late_d_image_source_total{{outcome=\"stripped_requests\"}} {}\n\
         sgl_router_late_d_image_source_total{{outcome=\"stripped_images\"}} {}\n\
         sgl_router_late_d_image_source_total{{outcome=\"collision\"}} {}\n\
         sgl_router_late_d_image_source_total{{outcome=\"fetch_ok\"}} {}\n\
         sgl_router_late_d_image_source_total{{outcome=\"fetch_missing\"}} {}\n\
         sgl_router_late_d_image_source_total{{outcome=\"fetch_bad\"}} {}\n\
         # HELP sgl_router_late_d_image_source_fetch_bytes_total Image bytes served to decodes that recompute.\n\
         # TYPE sgl_router_late_d_image_source_fetch_bytes_total counter\n\
         sgl_router_late_d_image_source_fetch_bytes_total {}\n\
         # HELP sgl_router_late_d_image_source_held Requests whose images the router holds now.\n\
         # TYPE sgl_router_late_d_image_source_held gauge\n\
         sgl_router_late_d_image_source_held {}\n",
        c(&STRIPPED_REQUESTS),
        c(&STRIPPED_IMAGES),
        c(&COLLISIONS),
        c(&FETCH_OK),
        c(&FETCH_MISSING),
        c(&FETCH_BAD),
        c(&FETCH_BYTES),
        store().len(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::to_bytes;
    use serde_json::json;

    const PNG_URI: &str = "data:image/png;base64,iVBORw0KGgo=";

    #[test]
    fn strip_needs_a_layout_and_images() {
        let src = source_value("http://r:1/internal/prepared_image", "rid-a");
        let mut no_layout = json!({"image_data": [PNG_URI]});
        assert!(strip_images(no_layout.as_object_mut().unwrap(), &src).is_none());
        assert!(no_layout.get("image_data").is_some());
        let mut null_layout = json!({"image_data": [PNG_URI], "prepared_image_layout": null});
        assert!(strip_images(null_layout.as_object_mut().unwrap(), &src).is_none());
        let mut no_images = json!({"prepared_image_layout": {"version": 1}, "image_data": []});
        assert!(strip_images(no_images.as_object_mut().unwrap(), &src).is_none());
        assert!(no_images.get(SOURCE_KEY).is_none());
        let mut ok = json!({"prompt": [1, 2], "prepared_image_layout": {"version": 1},
                            "image_data": [PNG_URI, PNG_URI]});
        let taken = strip_images(ok.as_object_mut().unwrap(), &src).unwrap();
        assert_eq!(taken.len(), 2);
        assert!(ok.get("image_data").is_none());
        assert_eq!(ok[SOURCE_KEY], src);
        assert_eq!(ok["prompt"], json!([1, 2]));
        assert_eq!(ok["prepared_image_layout"], json!({"version": 1}));
    }

    #[test]
    fn prepared_images_need_a_layout_and_a_nonempty_list() {
        assert!(!has_prepared_images(None));
        assert!(!has_prepared_images(Some(&json!([1]))));
        assert!(!has_prepared_images(Some(
            &json!({"image_data": [PNG_URI]})
        )));
        let empty = json!({"prepared_image_layout": {"version": 1}, "image_data": []});
        assert!(!has_prepared_images(Some(&empty)));
        let ok = json!({"prepared_image_layout": {"version": 1}, "image_data": [PNG_URI]});
        assert!(has_prepared_images(Some(&ok)));
    }

    #[test]
    fn data_uri_decodes_to_the_file_bytes() {
        assert_eq!(
            decode_data_uri(PNG_URI).unwrap(),
            b"\x89PNG\r\n\x1a\n".to_vec()
        );
        assert!(decode_data_uri("data:image/png,iVBORw0KGgo=").is_none());
        assert!(decode_data_uri("iVBORw0KGgo=").is_none());
        assert!(decode_data_uri("data:image/png;base64,@@@@").is_none());
    }

    #[tokio::test]
    async fn hold_serves_until_dropped_and_refuses_a_second_reservation() {
        let hold = Hold::reserve("rid-hold-test").unwrap();
        assert!(Hold::reserve("rid-hold-test").is_none());
        hold.fill(vec![json!(PNG_URI), json!("not a data uri")]);
        let ok = fetch(Json(FetchRequest {
            token: "rid-hold-test".into(),
            image_index: 0,
        }))
        .await;
        assert_eq!(ok.status(), StatusCode::OK);
        let body = to_bytes(ok.into_body(), 1 << 20).await.unwrap();
        assert_eq!(&body[..], b"\x89PNG\r\n\x1a\n");
        let bad = fetch(Json(FetchRequest {
            token: "rid-hold-test".into(),
            image_index: 1,
        }))
        .await;
        assert_eq!(bad.status(), StatusCode::UNPROCESSABLE_ENTITY);
        let out_of_range = fetch(Json(FetchRequest {
            token: "rid-hold-test".into(),
            image_index: 2,
        }))
        .await;
        assert_eq!(out_of_range.status(), StatusCode::NOT_FOUND);
        drop(hold);
        let gone = fetch(Json(FetchRequest {
            token: "rid-hold-test".into(),
            image_index: 0,
        }))
        .await;
        assert_eq!(gone.status(), StatusCode::NOT_FOUND);
        assert!(Hold::reserve("rid-hold-test").is_some());
    }
}
