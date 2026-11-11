// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Screenshot-aware token ids for token-id `/v1/completions` requests with
//! uploaded images (`--image-layout-config`, Kimi K2.5/K2.6 only).
//!
//! A replay client sends `prompt` = token ids with ONE placeholder
//! (`image_token_id`) per image plus `image_data` = one base64 data URI per
//! placeholder. The engine expands every placeholder to the image's
//! MoonViT token count and keys its radix cache with the image's
//! `pad_value`. Without this module the router routes on the placeholder
//! ids: prompt length, prefix matches, uncached-token costs and pull hints
//! are blind after the first screenshot.
//!
//! This is a port of `image_layout.expand_image_layout` (engine
//! `engine/decoupled_engine/image_layout.py`, Kimi branch): from each image's file
//! bytes (sha256) and dimensions (PNG IHDR / JPEG SOF, no pixel decoding)
//! it derives the Kimi resize grid and the layout identity
//! `pad = 1_000_000 + int(sha256(b"sglang-kimi-image-layout-v1\0" + config
//! JSON (sorted keys) + sha256(bytes) + ":vit_h:vit_w")[-16 hex:]) % 2^30`,
//! and returns the engine's radix view of the prompt (each placeholder ->
//! `num_tokens` copies of its pad value).
//!
//! The engine uses these pad values only when the request carries
//! `prepared_image_layout` (otherwise it hashes the processed pixel tensor,
//! which a router cannot reproduce). The router therefore injects
//! `prepared_image_layout = {"version": 1, "config": .., "images": [{width,
//! height, sha256}]}`; the engine re-derives the layout from the uploaded
//! bytes and rejects a mismatch. A body that already carries a layout is
//! used as is (its config must equal ours).
//!
//! Every failure leaves the request as before (placeholder ids, body
//! untouched) and is counted in `sgl_router_image_layout_total{outcome}`.

use serde_json::Value;
use sha2::{Digest, Sha256};
use std::sync::atomic::{AtomicU64, Ordering};

const KIMI_FIELDS: [&str; 9] = [
    "layout_type",
    "image_token_id",
    "patch_size",
    "merge_kernel_size",
    "in_patch_limit",
    "patch_limit_on_one_side",
    "fixed_output_tokens",
    "image_mean",
    "image_std",
];
const MM_PAD_SHIFT_VALUE: u64 = 1_000_000;
const IDENTITY_PREFIX: &[u8] = b"sglang-kimi-image-layout-v1\0";

/// The engine's Kimi image configuration (`image_config(...)` of
/// `decoupled_engine/image_layout.py`), validated, plus its canonical JSON bytes.
#[derive(Debug, Clone)]
pub struct ImageLayoutConfig {
    /// The configuration object as given (forwarded in the layout).
    config: Value,
    /// `json.dumps(config, sort_keys=True, separators=(",", ":"))`.
    canonical: Vec<u8>,
    pub image_token_id: u32,
    patch_size: u64,
    merge_kernel_size: u64,
    in_patch_limit: u64,
    patch_limit_on_one_side: u64,
}

/// Why a request was left unexpanded (also the metric label).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Skip {
    /// `image_data` is not a list of data-URI strings.
    NotDataUris,
    /// Base64 payload invalid.
    BadBase64,
    /// Neither a PNG nor a baseline/progressive JPEG header.
    UnknownFormat,
    /// Placeholder count differs from the image count.
    CountMismatch,
    /// A body-provided `prepared_image_layout` that differs from ours.
    LayoutMismatch,
    /// The expanded prompt would not fit in u32 ids / memory bounds.
    TooLarge,
}

impl Skip {
    fn label(self) -> &'static str {
        match self {
            Skip::NotDataUris => "skip_not_data_uri",
            Skip::BadBase64 => "skip_bad_base64",
            Skip::UnknownFormat => "skip_unknown_format",
            Skip::CountMismatch => "skip_count_mismatch",
            Skip::LayoutMismatch => "skip_layout_mismatch",
            Skip::TooLarge => "skip_too_large",
        }
    }
}

/// One uploaded image's identity (the layout's `images[i]`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageInfo {
    pub width: u32,
    pub height: u32,
    pub sha256: [u8; 32],
}

/// The per-image result of the Kimi resize + identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageTokens {
    pub num_tokens: u64,
    pub pad_value: u32,
    pub vit_h: u64,
    pub vit_w: u64,
}

/// The routing view of one request.
#[derive(Debug)]
pub struct Prepared {
    /// The engine radix view: placeholders expanded to pad values.
    pub ids: Vec<u32>,
    /// `Some(layout)` when the router must inject `prepared_image_layout`;
    /// `None` when the body already carried an equal one.
    pub inject: Option<Value>,
    pub images: usize,
    pub image_tokens: u64,
}

impl ImageLayoutConfig {
    /// Validate a Kimi configuration object exactly as
    /// `image_layout.image_config(config, config["image_token_id"])`
    /// does for a dict without a processor config.
    pub fn from_value(config: Value) -> anyhow::Result<Self> {
        let obj = config
            .as_object()
            .ok_or_else(|| anyhow::anyhow!("image layout config must be a JSON object"))?;
        let mut keys: Vec<&str> = obj.keys().map(String::as_str).collect();
        keys.sort_unstable();
        let mut want = KIMI_FIELDS.to_vec();
        want.sort_unstable();
        if keys != want {
            anyhow::bail!(
                "image layout config must have exactly the Kimi fields {want:?}, got {keys:?}"
            );
        }
        if obj.get("layout_type").and_then(Value::as_str) != Some("kimi_k25") {
            anyhow::bail!("image layout config: only layout_type \"kimi_k25\" is supported");
        }
        let positive = |key: &str| -> anyhow::Result<u64> {
            match obj.get(key).and_then(Value::as_u64) {
                Some(v) if v > 0 && obj[key].is_u64() => Ok(v),
                _ => anyhow::bail!("image layout config: {key} must be a positive integer"),
            }
        };
        let image_token_id = positive("image_token_id")?;
        let patch_size = positive("patch_size")?;
        let merge_kernel_size = positive("merge_kernel_size")?;
        let in_patch_limit = positive("in_patch_limit")?;
        let patch_limit_on_one_side = positive("patch_limit_on_one_side")?;
        if !obj["fixed_output_tokens"].is_null() {
            anyhow::bail!(
                "image layout config: fixed_output_tokens must be null (variable MoonViT output)"
            );
        }
        for key in ["image_mean", "image_std"] {
            let ok = obj[key].as_array().is_some_and(|a| {
                a.len() == 3
                    && a.iter().all(|v| {
                        v.as_f64()
                            .is_some_and(|f| f.is_finite() && (key == "image_mean" || f > 0.0))
                    })
            });
            if !ok {
                anyhow::bail!("image layout config: {key} must be 3 finite numbers");
            }
        }
        let image_token_id = u32::try_from(image_token_id)
            .map_err(|_| anyhow::anyhow!("image layout config: image_token_id exceeds u32"))?;
        let canonical = python_json(&config)?.into_bytes();
        Ok(Self {
            config,
            canonical,
            image_token_id,
            patch_size,
            merge_kernel_size,
            in_patch_limit,
            patch_limit_on_one_side,
        })
    }

    /// Read `PATH` (the trace `metadata.image_config`).
    pub fn load(path: &std::path::Path) -> anyhow::Result<Self> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("--image-layout-config {}: {e}", path.display()))?;
        let value: Value = serde_json::from_str(&text)
            .map_err(|e| anyhow::anyhow!("--image-layout-config {}: {e}", path.display()))?;
        Self::from_value(value)
    }

    pub fn canonical_json(&self) -> &str {
        std::str::from_utf8(&self.canonical).unwrap_or("")
    }

    /// `kimi_resize_config` + the identity of `expand_image_layout`.
    pub fn image_tokens(&self, info: &ImageInfo) -> ImageTokens {
        let (w0, h0) = (u64::from(info.width), u64::from(info.height));
        let p = self.patch_size;
        let limit = self.patch_limit_on_one_side;
        // Python: max(1.0, width // p) * max(1.0, height // p) (exact ints < 2^53).
        let pw = ((w0 / p).max(1)) as f64;
        let ph = ((h0 / p).max(1)) as f64;
        let mut scale = 1.0_f64;
        for c in [
            (self.in_patch_limit as f64 / (pw * ph)).sqrt(),
            (limit * p) as f64 / w0 as f64,
            (limit * p) as f64 / h0 as f64,
        ] {
            if c < scale {
                scale = c;
            }
        }
        // int(width * scale): truncation toward zero of a nonnegative float.
        let w = ((w0 as f64 * scale) as u64).max(1).min(limit * p);
        let h = ((h0 as f64 * scale) as u64).max(1).min(limit * p);
        let factor = p * self.merge_kernel_size;
        let pad_w = (factor - w % factor) % factor;
        let pad_h = (factor - h % factor) % factor;
        let num_tokens = ((h + pad_h) / factor) * ((w + pad_w) / factor);
        let vit_h = (h + pad_h) / p;
        let vit_w = (w + pad_w) / p;
        let mut hasher = Sha256::new();
        hasher.update(IDENTITY_PREFIX);
        hasher.update(&self.canonical);
        hasher.update(info.sha256);
        hasher.update(format!(":{vit_h}:{vit_w}").as_bytes());
        let digest = hasher.finalize();
        // hexdigest()[-16:] = the last 8 bytes, big endian.
        let mut tail = [0u8; 8];
        tail.copy_from_slice(&digest[24..32]);
        let identity = u64::from_be_bytes(tail);
        let pad_value = (MM_PAD_SHIFT_VALUE + identity % (1u64 << 30)) as u32;
        ImageTokens {
            num_tokens,
            pad_value,
            vit_h,
            vit_w,
        }
    }

    /// The `prepared_image_layout` object for these images.
    pub fn layout_value(&self, images: &[ImageInfo]) -> Value {
        serde_json::json!({
            "version": 1,
            "config": self.config.clone(),
            "images": images.iter().map(|i| serde_json::json!({
                "width": i.width,
                "height": i.height,
                "sha256": hex(&i.sha256),
            })).collect::<Vec<_>>(),
        })
    }

    /// Expand `prompt` (one placeholder per image) to the engine radix view.
    pub fn expand(&self, prompt: &[u32], images: &[ImageInfo]) -> Result<(Vec<u32>, u64), Skip> {
        let placeholders = prompt.iter().filter(|&&t| t == self.image_token_id).count();
        if placeholders != images.len() || images.is_empty() {
            return Err(Skip::CountMismatch);
        }
        let tokens: Vec<ImageTokens> = images.iter().map(|i| self.image_tokens(i)).collect();
        let image_tokens: u64 = tokens.iter().map(|t| t.num_tokens).sum();
        let total = (prompt.len() - placeholders) as u64 + image_tokens;
        // The engine's context limit is 262,144; anything near u32 is garbage.
        if total > (1 << 26) {
            return Err(Skip::TooLarge);
        }
        let mut out = Vec::with_capacity(total as usize);
        let mut next = tokens.iter();
        for &t in prompt {
            if t == self.image_token_id {
                let it = next.next().ok_or(Skip::CountMismatch)?;
                out.extend(std::iter::repeat_n(it.pad_value, it.num_tokens as usize));
            } else {
                out.push(t);
            }
        }
        Ok((out, image_tokens))
    }

    /// The routing view of a parsed `/v1/completions` body whose token-id
    /// prompt is `prompt`. `Ok(None)`: no `image_data` (nothing to do).
    pub fn prepare(&self, body: &Value, prompt: &[u32]) -> Result<Option<Prepared>, Skip> {
        let Some(data) = body.get("image_data") else {
            return Ok(None);
        };
        if data.is_null() {
            return Ok(None);
        }
        let list = data.as_array().ok_or(Skip::NotDataUris)?;
        if let Some(given) = body.get("prepared_image_layout").filter(|v| !v.is_null()) {
            let images = self.images_from_layout(given)?;
            if images.len() != list.len() {
                return Err(Skip::LayoutMismatch);
            }
            let (ids, image_tokens) = self.expand(prompt, &images)?;
            return Ok(Some(Prepared {
                ids,
                inject: None,
                images: images.len(),
                image_tokens,
            }));
        }
        let images = image_infos(list)?;
        let (ids, image_tokens) = self.expand(prompt, &images)?;
        Ok(Some(Prepared {
            ids,
            inject: Some(self.layout_value(&images)),
            images: images.len(),
            image_tokens,
        }))
    }

    fn images_from_layout(&self, given: &Value) -> Result<Vec<ImageInfo>, Skip> {
        let obj = given.as_object().ok_or(Skip::LayoutMismatch)?;
        if obj.len() != 3
            || obj.get("version").and_then(Value::as_u64) != Some(1)
            || obj.get("config") != Some(&self.config)
        {
            return Err(Skip::LayoutMismatch);
        }
        let list = obj
            .get("images")
            .and_then(Value::as_array)
            .ok_or(Skip::LayoutMismatch)?;
        list.iter()
            .map(|i| {
                let o = i
                    .as_object()
                    .filter(|o| o.len() == 3)
                    .ok_or(Skip::LayoutMismatch)?;
                let dim = |k: &str| {
                    o.get(k)
                        .and_then(Value::as_u64)
                        .and_then(|v| u32::try_from(v).ok())
                        .filter(|&v| v > 0)
                        .ok_or(Skip::LayoutMismatch)
                };
                let sha = o
                    .get("sha256")
                    .and_then(Value::as_str)
                    .and_then(parse_sha256_hex)
                    .ok_or(Skip::LayoutMismatch)?;
                Ok(ImageInfo {
                    width: dim("width")?,
                    height: dim("height")?,
                    sha256: sha,
                })
            })
            .collect()
    }
}

/// `prepare_image_layout` for one data URI: header check, strict base64,
/// sha256 of the bytes, PNG/JPEG dimensions.
pub fn image_info_from_data_uri(uri: &str) -> Result<ImageInfo, Skip> {
    let (header, payload) = uri.split_once(',').ok_or(Skip::NotDataUris)?;
    if !header.starts_with("data:image/") || !header.ends_with(";base64") {
        return Err(Skip::NotDataUris);
    }
    // One pass: hash every decoded byte, keep the head for the dimensions
    // (a PNG IHDR is in the first 24 bytes; a JPEG SOF follows the APPn
    // segments, which stay far below this bound for screenshots).
    const HEAD: usize = 1 << 20;
    let mut hasher = Sha256::new();
    let mut head = Vec::with_capacity(4096);
    base64_decode_into(payload.as_bytes(), |b| {
        hasher.update(b);
        if head.len() < HEAD {
            head.extend_from_slice(&b[..b.len().min(HEAD - head.len())]);
        }
    })
    .ok_or(Skip::BadBase64)?;
    let (width, height) = image_dimensions(&head).ok_or(Skip::UnknownFormat)?;
    let digest = hasher.finalize();
    let mut sha256 = [0u8; 32];
    sha256.copy_from_slice(&digest);
    Ok(ImageInfo {
        width,
        height,
        sha256,
    })
}

/// [`image_info_from_data_uri`] for every entry, on up to 4 threads for
/// multi-image requests (a CUA request carries ~19 screenshots, ~6.5 MB).
fn image_infos(list: &[Value]) -> Result<Vec<ImageInfo>, Skip> {
    let uris: Vec<&str> = list
        .iter()
        .map(|v| v.as_str().ok_or(Skip::NotDataUris))
        .collect::<Result<_, _>>()?;
    let threads = uris.len().clamp(1, 4);
    if threads == 1 || uris.len() < 4 {
        return uris.iter().map(|u| image_info_from_data_uri(u)).collect();
    }
    let per = uris.len().div_ceil(threads);
    std::thread::scope(|scope| {
        let handles: Vec<_> = uris
            .chunks(per)
            .map(|part| {
                scope.spawn(move || {
                    part.iter()
                        .map(|u| image_info_from_data_uri(u))
                        .collect::<Result<Vec<_>, _>>()
                })
            })
            .collect();
        let mut out = Vec::with_capacity(uris.len());
        for h in handles {
            out.extend(h.join().map_err(|_| Skip::BadBase64)??);
        }
        Ok(out)
    })
}

/// Width and height from a PNG IHDR or a JPEG start-of-frame marker.
pub fn image_dimensions(raw: &[u8]) -> Option<(u32, u32)> {
    if raw.len() >= 24 && raw.starts_with(b"\x89PNG\r\n\x1a\n") && &raw[12..16] == b"IHDR" {
        let w = u32::from_be_bytes(raw[16..20].try_into().ok()?);
        let h = u32::from_be_bytes(raw[20..24].try_into().ok()?);
        return (w > 0 && h > 0).then_some((w, h));
    }
    if raw.len() >= 4 && raw[0] == 0xFF && raw[1] == 0xD8 {
        let mut i = 2usize;
        while i + 3 < raw.len() {
            if raw[i] != 0xFF {
                i += 1;
                continue;
            }
            let marker = raw[i + 1];
            if marker == 0xFF {
                i += 1; // fill byte
                continue;
            }
            if marker == 0xD8 || marker == 0x01 || (0xD0..=0xD7).contains(&marker) {
                i += 2;
                continue;
            }
            if marker == 0xD9 || marker == 0xDA {
                return None; // end of image / start of scan before any SOF
            }
            let len = u16::from_be_bytes([raw[i + 2], raw[i + 3]]) as usize;
            if matches!(marker, 0xC0..=0xC3 | 0xC5..=0xC7 | 0xC9..=0xCB | 0xCD..=0xCF) {
                if i + 9 > raw.len() {
                    return None;
                }
                let h = u16::from_be_bytes([raw[i + 5], raw[i + 6]]) as u32;
                let w = u16::from_be_bytes([raw[i + 7], raw[i + 8]]) as u32;
                return (w > 0 && h > 0).then_some((w, h));
            }
            if len < 2 {
                return None;
            }
            i += 2 + len;
        }
    }
    None
}

/// Decoding table of the standard base64 alphabet; 0xFF = not in it.
const B64: [u8; 256] = {
    let mut t = [0xFFu8; 256];
    let mut i = 0;
    while i < 26 {
        t[b'A' as usize + i] = i as u8;
        t[b'a' as usize + i] = (i + 26) as u8;
        i += 1;
    }
    let mut d = 0;
    while d < 10 {
        t[b'0' as usize + d] = (52 + d) as u8;
        d += 1;
    }
    t[b'+' as usize] = 62;
    t[b'/' as usize] = 63;
    t
};

/// Strict standard base64 (the alphabet only, length a multiple of 4,
/// `=` padding only at the end), as `base64.b64decode(..., validate=True)`
/// accepts for well-formed data URIs. Streams the decoded bytes to `sink`
/// in blocks (no full-size buffer); `None` on invalid input.
fn base64_decode_into(input: &[u8], mut sink: impl FnMut(&[u8])) -> Option<()> {
    if !input.len().is_multiple_of(4) {
        return None;
    }
    let pad = input.iter().rev().take_while(|&&c| c == b'=').count();
    if pad > 2 {
        return None;
    }
    let body = &input[..input.len() - pad];
    const BLOCK: usize = 16_384; // input quads per block
    let mut buf = vec![0u8; BLOCK * 3];
    let full = body.len() / 4 * 4;
    for block in body[..full].chunks(BLOCK * 4) {
        let mut bad = 0u8;
        let mut o = 0;
        for c in block.chunks_exact(4) {
            let (a, b, x, y) = (
                B64[c[0] as usize],
                B64[c[1] as usize],
                B64[c[2] as usize],
                B64[c[3] as usize],
            );
            bad |= a | b | x | y;
            let n = ((a as u32) << 18) | ((b as u32) << 12) | ((x as u32) << 6) | y as u32;
            buf[o] = (n >> 16) as u8;
            buf[o + 1] = (n >> 8) as u8;
            buf[o + 2] = n as u8;
            o += 3;
        }
        if bad & 0x80 != 0 {
            return None;
        }
        sink(&buf[..o]);
    }
    let rest = &body[full..];
    let v = |c: u8| -> Option<u32> {
        let x = B64[c as usize];
        (x != 0xFF).then_some(x as u32)
    };
    match rest.len() {
        0 => {}
        2 => {
            let n = (v(rest[0])? << 18) | (v(rest[1])? << 12);
            sink(&[(n >> 16) as u8]);
        }
        3 => {
            let n = (v(rest[0])? << 18) | (v(rest[1])? << 12) | (v(rest[2])? << 6);
            sink(&[(n >> 16) as u8, (n >> 8) as u8]);
        }
        _ => return None,
    }
    Some(())
}

/// [`base64_decode_into`] collected into one buffer.
pub fn base64_decode(input: &[u8]) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(input.len() / 4 * 3);
    base64_decode_into(input, |b| out.extend_from_slice(b))?;
    Some(out)
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn parse_sha256_hex(s: &str) -> Option<[u8; 32]> {
    if s.len() != 64
        || !s
            .bytes()
            .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
    {
        return None;
    }
    let mut out = [0u8; 32];
    for (i, o) in out.iter_mut().enumerate() {
        *o = u8::from_str_radix(&s[2 * i..2 * i + 2], 16).ok()?;
    }
    Some(out)
}

/// `json.dumps(value, sort_keys=True, separators=(",", ":"))` for the JSON
/// subset of an image configuration (ASCII strings, Python float repr).
pub fn python_json(value: &Value) -> anyhow::Result<String> {
    let mut out = String::new();
    write_python_json(value, &mut out)?;
    Ok(out)
}

fn write_python_json(value: &Value, out: &mut String) -> anyhow::Result<()> {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => {
            if n.is_u64() || n.is_i64() {
                out.push_str(&n.to_string());
            } else {
                let f = n
                    .as_f64()
                    .ok_or_else(|| anyhow::anyhow!("non-finite number in image config"))?;
                out.push_str(&python_float_repr(f)?);
            }
        }
        Value::String(s) => {
            if !s.bytes().all(|c| (0x20..0x7f).contains(&c)) {
                anyhow::bail!("image config strings must be printable ASCII");
            }
            out.push_str(&serde_json::to_string(s)?);
        }
        Value::Array(a) => {
            out.push('[');
            for (i, v) in a.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_python_json(v, out)?;
            }
            out.push(']');
        }
        Value::Object(o) => {
            let mut keys: Vec<&String> = o.keys().collect();
            keys.sort();
            out.push('{');
            for (i, k) in keys.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_python_json(&Value::String((*k).clone()), out)?;
                out.push(':');
                write_python_json(&o[k.as_str()], out)?;
            }
            out.push('}');
        }
    }
    Ok(())
}

/// Python `repr(float)`: shortest round-trip digits; fixed notation when
/// the decimal exponent is in [-4, 16), else `d.ddde+XX`.
pub fn python_float_repr(f: f64) -> anyhow::Result<String> {
    if !f.is_finite() {
        anyhow::bail!("non-finite float in image config");
    }
    Ok(crate::eft_cheap::python_float_repr(f))
}

// ---------------------------------------------------------------- metrics

static EXPANDED: AtomicU64 = AtomicU64::new(0);
static EXPANDED_GIVEN: AtomicU64 = AtomicU64::new(0);
static NO_IMAGES: AtomicU64 = AtomicU64::new(0);
static IMAGES: AtomicU64 = AtomicU64::new(0);
static IMAGE_TOKENS: AtomicU64 = AtomicU64::new(0);
static PREPARE_MICROS: AtomicU64 = AtomicU64::new(0);
static SKIPS: [AtomicU64; 6] = [
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
];
const SKIP_ORDER: [Skip; 6] = [
    Skip::NotDataUris,
    Skip::BadBase64,
    Skip::UnknownFormat,
    Skip::CountMismatch,
    Skip::LayoutMismatch,
    Skip::TooLarge,
];

/// Count one `prepare` outcome (and its wall time).
pub fn record(outcome: &Result<Option<Prepared>, Skip>, micros: u64) {
    PREPARE_MICROS.fetch_add(micros, Ordering::Relaxed);
    match outcome {
        Ok(None) => {
            NO_IMAGES.fetch_add(1, Ordering::Relaxed);
        }
        Ok(Some(p)) => {
            if p.inject.is_some() {
                EXPANDED.fetch_add(1, Ordering::Relaxed);
            } else {
                EXPANDED_GIVEN.fetch_add(1, Ordering::Relaxed);
            }
            IMAGES.fetch_add(p.images as u64, Ordering::Relaxed);
            IMAGE_TOKENS.fetch_add(p.image_tokens, Ordering::Relaxed);
        }
        Err(skip) => {
            if let Some(i) = SKIP_ORDER.iter().position(|s| s == skip) {
                SKIPS[i].fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

/// Prometheus text for `/metrics`.
pub fn render_metrics() -> String {
    let mut out = String::from(
        "# HELP sgl_router_image_layout_total /v1/completions token-id requests seen by --image-layout-config, by outcome.\n# TYPE sgl_router_image_layout_total counter\n",
    );
    let mut row = |label: &str, v: u64| {
        out.push_str(&format!(
            "sgl_router_image_layout_total{{outcome=\"{label}\"}} {v}\n"
        ));
    };
    row("expanded_injected", EXPANDED.load(Ordering::Relaxed));
    row(
        "expanded_given_layout",
        EXPANDED_GIVEN.load(Ordering::Relaxed),
    );
    row("no_images", NO_IMAGES.load(Ordering::Relaxed));
    for (i, s) in SKIP_ORDER.iter().enumerate() {
        row(s.label(), SKIPS[i].load(Ordering::Relaxed));
    }
    out.push_str(&format!(
        "# TYPE sgl_router_image_layout_images_total counter\nsgl_router_image_layout_images_total {}\n\
         # TYPE sgl_router_image_layout_image_tokens_total counter\nsgl_router_image_layout_image_tokens_total {}\n\
         # TYPE sgl_router_image_layout_prepare_seconds_total counter\nsgl_router_image_layout_prepare_seconds_total {:.6}\n",
        IMAGES.load(Ordering::Relaxed),
        IMAGE_TOKENS.load(Ordering::Relaxed),
        PREPARE_MICROS.load(Ordering::Relaxed) as f64 / 1e6,
    ));
    out
}

pub fn skip_label(skip: Skip) -> &'static str {
    skip.label()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn golden() -> Value {
        serde_json::from_str(include_str!(
            "../tests/fixtures/kimi_image_layout_golden.json"
        ))
        .unwrap()
    }

    fn cfg() -> ImageLayoutConfig {
        ImageLayoutConfig::from_value(golden()["config"].clone()).unwrap()
    }

    #[test]
    fn canonical_config_matches_python_json_dumps() {
        let g = golden();
        assert_eq!(
            cfg().canonical_json(),
            g["config_canonical"].as_str().unwrap()
        );
    }

    #[test]
    fn image_tokens_match_the_python_layout_module() {
        let g = golden();
        let c = cfg();
        for (img, want) in g["images"]
            .as_array()
            .unwrap()
            .iter()
            .zip(g["per_image"].as_array().unwrap())
        {
            let info = image_info_from_data_uri(img["data_uri"].as_str().unwrap()).unwrap();
            assert_eq!(info.width as u64, img["width"].as_u64().unwrap());
            assert_eq!(info.height as u64, img["height"].as_u64().unwrap());
            assert_eq!(hex(&info.sha256), img["sha256"].as_str().unwrap());
            let t = c.image_tokens(&info);
            assert_eq!(t.num_tokens, want["num_tokens"].as_u64().unwrap(), "{img}");
            assert_eq!(
                t.pad_value as u64,
                want["pad_value"].as_u64().unwrap(),
                "{img}"
            );
            assert_eq!(t.vit_h, want["vit_h"].as_u64().unwrap());
            assert_eq!(t.vit_w, want["vit_w"].as_u64().unwrap());
        }
    }

    #[test]
    fn prepare_expands_and_injects_the_python_layout() {
        let g = golden();
        let c = cfg();
        let prompt: Vec<u32> = serde_json::from_value(g["prompt"].clone()).unwrap();
        let uris: Vec<Value> = g["images"]
            .as_array()
            .unwrap()
            .iter()
            .map(|i| i["data_uri"].clone())
            .collect();
        let body = serde_json::json!({"model": "m", "prompt": prompt, "image_data": uris});
        let p = c.prepare(&body, &prompt).unwrap().unwrap();
        assert_eq!(p.ids.len() as u64, g["expanded_len"].as_u64().unwrap());
        let ids_json = serde_json::to_string(&p.ids).unwrap();
        // Python json.dumps(list) uses ", " separators.
        let py = ids_json.replace(',', ", ");
        assert_eq!(
            hex(&Sha256::digest(py.as_bytes())),
            g["cached_sha256"].as_str().unwrap()
        );
        let inject = p.inject.unwrap();
        assert_eq!(inject, g["layout"]);
        // A body that carries that layout is used as is (no second injection).
        let mut given = body.clone();
        given["prepared_image_layout"] = inject;
        let q = c.prepare(&given, &prompt).unwrap().unwrap();
        assert!(q.inject.is_none());
        assert_eq!(q.ids, p.ids);
    }

    #[test]
    fn prepare_skips_without_changing_anything() {
        let c = cfg();
        let prompt = vec![1, c.image_token_id, 2];
        let none = serde_json::json!({"prompt": prompt});
        assert!(c.prepare(&none, &prompt).unwrap().is_none());
        let url = serde_json::json!({"image_data": ["https://x/y.png"]});
        assert_eq!(c.prepare(&url, &prompt).unwrap_err(), Skip::NotDataUris);
        let bad = serde_json::json!({"image_data": ["data:image/png;base64,@@@@"]});
        assert_eq!(c.prepare(&bad, &prompt).unwrap_err(), Skip::BadBase64);
        let gif = serde_json::json!({"image_data": ["data:image/gif;base64,R0lGODlh"]});
        assert_eq!(c.prepare(&gif, &prompt).unwrap_err(), Skip::UnknownFormat);
        let g = golden();
        let a = g["images"][0]["data_uri"].clone();
        let b = g["images"][1]["data_uri"].clone();
        let two = serde_json::json!({ "image_data": [a.clone(), b] });
        assert_eq!(c.prepare(&two, &prompt).unwrap_err(), Skip::CountMismatch);
        let mut wrong = c.layout_value(&[image_info_from_data_uri(a.as_str().unwrap()).unwrap()]);
        wrong["config"]["patch_size"] = serde_json::json!(16);
        let given = serde_json::json!({ "image_data": [a], "prepared_image_layout": wrong });
        assert_eq!(
            c.prepare(&given, &prompt).unwrap_err(),
            Skip::LayoutMismatch
        );
    }

    #[test]
    fn config_validation_refuses_non_kimi_shapes() {
        let mut v = golden()["config"].clone();
        v["layout_type"] = serde_json::json!("dsv4");
        assert!(ImageLayoutConfig::from_value(v).is_err());
        let mut v = golden()["config"].clone();
        v["fixed_output_tokens"] = serde_json::json!(64);
        assert!(ImageLayoutConfig::from_value(v).is_err());
        let mut v = golden()["config"].clone();
        v.as_object_mut().unwrap().remove("image_std");
        assert!(ImageLayoutConfig::from_value(v).is_err());
        let mut v = golden()["config"].clone();
        v["patch_size"] = serde_json::json!(14.0);
        assert!(ImageLayoutConfig::from_value(v).is_err());
    }

    #[test]
    fn base64_is_strict_and_correct() {
        assert_eq!(base64_decode(b"").unwrap(), b"");
        assert_eq!(base64_decode(b"Zg==").unwrap(), b"f");
        assert_eq!(base64_decode(b"Zm8=").unwrap(), b"fo");
        assert_eq!(base64_decode(b"Zm9v").unwrap(), b"foo");
        assert_eq!(base64_decode(b"Zm9vYmFy").unwrap(), b"foobar");
        assert!(base64_decode(b"Zm9").is_none());
        assert!(base64_decode(b"Zm=v").is_none());
        assert!(base64_decode(b"Z===").is_none());
        assert!(base64_decode(b"Zm9v\n").is_none());
    }

    #[test]
    fn base64_round_trips_across_blocks() {
        const ALPHA: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        fn encode(raw: &[u8]) -> Vec<u8> {
            let mut out = Vec::new();
            for c in raw.chunks(3) {
                let n = (c[0] as u32) << 16
                    | (*c.get(1).unwrap_or(&0) as u32) << 8
                    | *c.get(2).unwrap_or(&0) as u32;
                for (i, shift) in [18, 12, 6, 0].into_iter().enumerate() {
                    out.push(if i <= c.len() {
                        ALPHA[((n >> shift) & 63) as usize]
                    } else {
                        b'='
                    });
                }
            }
            out
        }
        for len in [
            0usize, 1, 2, 3, 49_151, 49_152, 49_153, 200_000, 200_001, 200_002,
        ] {
            let raw: Vec<u8> = (0..len)
                .map(|i| (i.wrapping_mul(2_654_435_761) >> 7) as u8)
                .collect();
            assert_eq!(base64_decode(&encode(&raw)).unwrap(), raw, "len {len}");
        }
    }

    #[test]
    fn python_float_repr_matches_python() {
        for (f, want) in [
            (0.5, "0.5"),
            (1.0, "1.0"),
            (0.48145466, "0.48145466"),
            (1e-05, "1e-05"),
            (0.0001, "0.0001"),
            (1e16, "1e+16"),
            (123456789012345.0, "123456789012345.0"),
            (-2.5, "-2.5"),
            (0.1, "0.1"),
        ] {
            assert_eq!(python_float_repr(f).unwrap(), want, "{f}");
        }
    }

    #[test]
    fn threaded_image_infos_equal_one_by_one() {
        let g = golden();
        let mut list: Vec<Value> = g["images"]
            .as_array()
            .unwrap()
            .iter()
            .map(|i| i["data_uri"].clone())
            .collect();
        list.extend(list.clone());
        let threaded = image_infos(&list).unwrap();
        let single: Vec<ImageInfo> = list
            .iter()
            .map(|u| image_info_from_data_uri(u.as_str().unwrap()).unwrap())
            .collect();
        assert_eq!(threaded, single);
        let mut bad = list.clone();
        bad[7] = Value::String("data:image/png;base64,@@@@".into());
        assert_eq!(image_infos(&bad).unwrap_err(), Skip::BadBase64);
    }

    #[test]
    fn jpeg_dimensions_skip_app_segments() {
        let g = golden();
        let info = image_info_from_data_uri(g["images"][2]["data_uri"].as_str().unwrap()).unwrap();
        assert_eq!((info.width, info.height), (4000, 3000));
    }
}
