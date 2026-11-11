// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Golden tests: the port against the outputs of the Python reference
//! `router_e3.py` r5. Fixtures in `tests/fixtures/e3/` (provenance and
//! selection rule in `PROVENANCE.md` there):
//! - `goldens-subset.jsonl`: 600 forecast goldens of the r5 reference run
//!   (every mode and category, every replay run x mode pair);
//! - `fit-goldens-subset.jsonl`: 156 fit goldens of the same run (every fit
//!   kind and every fallback source);
//! - `synth-goldens.jsonl`: the 67 synthetic rare-path goldens;
//! - `bound-tables.json`, `bound-params.json`: the r5 bound tables.
//!
//! Every float must be within 1e-9 of the reference (the port is expected to
//! be exact; set `E3_GOLDENS_EXACT=1` to require exact equality). Integers,
//! booleans, strings, nulls and key sets must be equal. The ignored test
//! `full_nsc_goldens_match_the_reference` runs all 8,880 forecast and 445
//! fit goldens from `E3_GOLDENS_DIR` (the r5 bounds directory; not committed).

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use serde::de::DeserializeOwned;
use serde_json::{json, Value};
use crate::sha256_hex;

use super::*;

/// Absolute tolerance of every float (seconds for the times).
const TOL: f64 = 1e-9;

/// The deploy modes of `bound-params.json` (r5).
const DEPLOY_MODES: [&str; 9] = [
    "snap+prod",
    "snap+prodL",
    "push+prodL",
    "push+gpu+prod",
    "pushL+prodL",
    "pushL+gpu+prod",
    "push+split+prodL",
    "push+gpu+split+prod",
    "pushL+gpu+split+prod",
];

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/e3")
        .join(name)
}

fn read(path: &Path) -> String {
    std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn jsonl(path: &Path) -> Vec<Value> {
    read(path)
        .lines()
        .filter(|l| !l.trim().is_empty())
        .enumerate()
        .map(|(i, l)| {
            serde_json::from_str(l)
                .unwrap_or_else(|e| panic!("{} line {}: {e}", path.display(), i + 1))
        })
        .collect()
}

fn parse<T: DeserializeOwned>(v: &Value, what: &str) -> T {
    serde_json::from_value(v.clone()).unwrap_or_else(|e| panic!("{what}: {e}"))
}

/// An optional input: a missing key or null is None (reference `inp.get`).
fn opt<T: DeserializeOwned>(v: &Value, key: &str, what: &str) -> Option<T> {
    match v.get(key) {
        None | Some(Value::Null) => None,
        Some(x) => Some(parse(x, what)),
    }
}

fn to_json<T: serde::Serialize>(v: T) -> Value {
    serde_json::to_value(v).expect("serializable")
}

/// Comparison of the port's JSON with the reference's JSON.
struct Check {
    strict: bool,
    cases: u64,
    floats: u64,
    exact: u64,
    max_abs: f64,
    errors: Vec<String>,
}

impl Check {
    fn new() -> Self {
        let strict = std::env::var("E3_GOLDENS_EXACT").is_ok_and(|v| !v.is_empty() && v != "0");
        Self {
            strict,
            cases: 0,
            floats: 0,
            exact: 0,
            max_abs: 0.0,
            errors: Vec::new(),
        }
    }

    fn compare(&mut self, path: &str, actual: &Value, expected: &Value) {
        match (actual, expected) {
            (Value::Number(a), Value::Number(e)) => {
                let (a_float, e_float) = (a.is_f64(), e.is_f64());
                if !a_float && !e_float {
                    if a.as_i64() != e.as_i64() || a.as_u64() != e.as_u64() {
                        self.errors.push(format!("{path}: {a} != {e}"));
                    }
                    return;
                }
                if a_float != e_float {
                    self.errors
                        .push(format!("{path}: number type: {a} vs reference {e}"));
                    return;
                }
                let (x, y) = (
                    a.as_f64().unwrap_or(f64::NAN),
                    e.as_f64().unwrap_or(f64::NAN),
                );
                self.floats += 1;
                if x == y {
                    self.exact += 1;
                    return;
                }
                let d = (x - y).abs();
                if d > self.max_abs {
                    self.max_abs = d;
                }
                if d > TOL || d.is_nan() {
                    self.errors
                        .push(format!("{path}: {x:?} != {y:?} (|diff| {d:e})"));
                } else if self.strict {
                    self.errors
                        .push(format!("{path}: {x:?} != {y:?} (not exact, |diff| {d:e})"));
                }
            }
            (Value::Array(a), Value::Array(e)) => {
                if a.len() != e.len() {
                    self.errors
                        .push(format!("{path}: length {} != {}", a.len(), e.len()));
                    return;
                }
                for (i, (x, y)) in a.iter().zip(e).enumerate() {
                    self.compare(&format!("{path}[{i}]"), x, y);
                }
            }
            (Value::Object(a), Value::Object(e)) => {
                let (ka, ke): (BTreeSet<&String>, BTreeSet<&String>) =
                    (a.keys().collect(), e.keys().collect());
                if ka != ke {
                    self.errors.push(format!(
                        "{path}: keys: extra {:?}, missing {:?}",
                        ka.difference(&ke).collect::<Vec<_>>(),
                        ke.difference(&ka).collect::<Vec<_>>()
                    ));
                }
                for (k, y) in e {
                    if let Some(x) = a.get(k) {
                        self.compare(&format!("{path}.{k}"), x, y);
                    }
                }
            }
            _ => {
                if actual != expected {
                    self.errors.push(format!("{path}: {actual} != {expected}"));
                }
            }
        }
    }

    fn finish(&self, what: &str) {
        eprintln!(
            "{what}: {} cases, {} floats compared, {} exact, max |diff| {:e}",
            self.cases, self.floats, self.exact, self.max_abs
        );
        assert!(
            self.errors.is_empty(),
            "{what}: {} mismatches; first: {:#?}",
            self.errors.len(),
            &self.errors[..self.errors.len().min(20)]
        );
    }
}

/// One reference call (`router_e3_bounds.recompute_fit` and
/// `router_e3_synth.run_case`) on the port, as JSON.
fn run_case(kind: &str, inp: &Value, id: &str) -> Value {
    match kind {
        "forecast" => {
            let i: Input = parse(&inp["inp"], id);
            let p: Params = parse(&inp["params"], id);
            to_json(forecast(&i, &p).unwrap_or_else(|e| panic!("{id}: {e}")))
        }
        "gpu_classes" => {
            let rows: Vec<GpuSample> = parse(&inp["rows"], id);
            to_json(fit_gpu_classes(&rows, GRAPH_TOKENS, GPU_FIT_MIN))
        }
        "resolve_gpu" => {
            let chain: Vec<(String, Option<GpuClasses>)> = parse(&inp["chain"], id);
            let floor_ms: f64 = parse(&inp["floor_ms"], id);
            let (gm, src) = resolve_gpu(&chain, floor_ms);
            json!([gm, src])
        }
        "launch_intervals" => {
            let rows: Vec<LaunchSample> = parse(&inp["rows"], id);
            let gm: GpuModel = parse(&inp["gpu"], id);
            let fallback: Option<Vec<LaunchSample>> = opt(inp, "fallback_rows", id);
            let const_median: Option<f64> = opt(inp, "const", id);
            to_json(fit_launch_intervals(
                &rows,
                &gm,
                fallback.as_deref(),
                const_median,
            ))
        }
        "launch_rows" => {
            let rows: Vec<DesignRow> = parse(&inp["rows"], id);
            let (coef, resid_q) = fit_launch_rows(&rows, LAUNCH_FIT_MIN);
            json!({"coef": coef, "resid_q": resid_q})
        }
        "arrival_rates" => {
            let rows: Vec<ArrivalRow> = parse(&inp["rows"], id);
            let step: f64 = parse(&inp["step"], id);
            let window_s: f64 = parse(&inp["window_s"], id);
            to_json(arrival_rates(&rows, step, window_s, ARRIVAL_MIN))
        }
        "rx_line" => {
            let samples: Vec<(f64, f64)> = parse(&inp["samples"], id);
            let mut line = RxLine::new();
            for (x, y) in samples {
                line.add(x, y);
            }
            to_json(line.params())
        }
        "router_lag_next" => {
            let samples: Vec<LagSample> = parse(&inp["samples"], id);
            let gm: GpuModel = parse(&inp["gpu"], id);
            to_json(router_lag_next(&samples, &gm, GRAPH_TOKENS))
        }
        "calibrate" => {
            let iters: Vec<CalIter> = parse(&inp["cal_iters"], id);
            let reqs: Vec<CalReq> = parse(&inp["cal_reqs"], id);
            let chain: Vec<(i64, Option<f64>, Option<f64>)> = parse(&inp["gpu_chain"], id);
            let cal = calibrate(&iters, &reqs, &gpu_chain_from_rows(&chain), GRAPH_TOKENS);
            to_json(cal.unwrap_or_else(|e| panic!("{id}: {e}")))
        }
        "error_bound" => {
            let view: BucketView = parse(&inp["out"], id);
            let table: BoundTable = parse(&inp["table"], id);
            let level = inp["level"]
                .as_str()
                .unwrap_or_else(|| panic!("{id}: level"));
            to_json(error_bound(&view, &table, level).unwrap_or_else(|e| panic!("{id}: {e}")))
        }
        "bound_curve" => {
            let knots: Vec<f64> = parse(&inp["knots_s"], id);
            let b: Vec<f64> = parse(&inp["b_s"], id);
            let p: f64 = parse(&inp["pred_s"], id);
            to_json(bound_curve(&knots, &b, p).unwrap_or_else(|| panic!("{id}: no bound")))
        }
        "reroute_gain" => {
            let f = |k: &str| -> f64 { parse(&inp[k], id) };
            json!(reroute_gain(
                f("c_owner"),
                f("b_owner"),
                f("c_other"),
                f("b_other")
            ))
        }
        "gpu_s" => {
            let gm: GpuModel = parse(&inp["gm"], id);
            let f: Features = parse(&inp["f"], id);
            json!(gpu_s(&gm, &f, GRAPH_TOKENS))
        }
        _ => panic!("{id}: unknown golden kind {kind}"),
    }
}

/// Forecast goldens: the output and the bound of each row's mode.
fn check_forecasts(
    rows: &[Value],
    tables: &BTreeMap<String, BoundTable>,
    tables_sha: &str,
    check: &mut Check,
) {
    for g in rows {
        let id = g["golden_id"].as_str().unwrap_or("?");
        let inp: Input = parse(&g["input"], id);
        let params: Params = parse(&g["params"], id);
        let out = forecast(&inp, &params).unwrap_or_else(|e| panic!("{id}: {e}"));
        check.cases += 1;
        check.compare(&format!("{id}.output"), &to_json(&out), &g["output"]);
        let expected = &g["bound"];
        if expected.is_null() {
            continue;
        }
        let mode = g["mode"].as_str().unwrap_or_else(|| panic!("{id}: mode"));
        let Some(table) = tables.get(mode) else {
            check
                .errors
                .push(format!("{id}: no bound table for mode {mode}"));
            continue;
        };
        if g["bound_tables_sha256"].as_str() != Some(tables_sha) {
            check.errors.push(format!(
                "{id}: bound tables sha256 differs from the fixture's"
            ));
        }
        let view = BucketView::from(&out);
        let level_bound =
            |level: &str| error_bound(&view, table, level).unwrap_or_else(|e| panic!("{id}: {e}"));
        let actual = json!({
            "table_mode": mode,
            "scheme": table.scheme,
            "bucket": out.pred_s.map(|_| bucket_of(&view, table.scheme)),
            "b_0.8_s": level_bound("0.8"),
            "b_0.9_s": level_bound("0.9"),
        });
        check.compare(&format!("{id}.bound"), &actual, expected);
    }
}

fn check_cases(rows: &[Value], check: &mut Check) {
    for g in rows {
        let id = g["golden_id"].as_str().unwrap_or("?");
        let kind = g["kind"].as_str().unwrap_or_else(|| panic!("{id}: kind"));
        let actual = run_case(kind, &g["input"], id);
        check.cases += 1;
        check.compare(&format!("{id}.output"), &actual, &g["output"]);
    }
}

fn load_tables(path: &Path) -> (BTreeMap<String, BoundTable>, String) {
    let text = read(path);
    let tables: BTreeMap<String, BoundTable> =
        serde_json::from_str(&text).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    (tables, sha256_hex(text.as_bytes()))
}

#[test]
fn forecast_goldens_match_the_reference() {
    let (tables, sha) = load_tables(&fixture("bound-tables.json"));
    let rows = jsonl(&fixture("goldens-subset.jsonl"));
    assert_eq!(rows.len(), 600, "fixture rows (PROVENANCE.md)");
    let modes: BTreeSet<&str> = rows.iter().filter_map(|g| g["mode"].as_str()).collect();
    let categories: BTreeSet<&str> = rows.iter().filter_map(|g| g["category"].as_str()).collect();
    assert_eq!(
        (modes.len(), categories.len()),
        (37, 12),
        "every mode and category"
    );
    let mut check = Check::new();
    check_forecasts(&rows, &tables, &sha, &mut check);
    check.finish("forecast goldens (fixture subset)");
}

#[test]
fn fit_goldens_match_the_reference() {
    let rows = jsonl(&fixture("fit-goldens-subset.jsonl"));
    assert_eq!(rows.len(), 156, "fixture rows (PROVENANCE.md)");
    let kinds: BTreeSet<&str> = rows.iter().filter_map(|g| g["kind"].as_str()).collect();
    assert_eq!(kinds.len(), 8, "every fit kind: {kinds:?}");
    let mut check = Check::new();
    check_cases(&rows, &mut check);
    check.finish("fit goldens (fixture subset)");
}

#[test]
fn synthetic_goldens_match_the_reference() {
    let rows = jsonl(&fixture("synth-goldens.jsonl"));
    assert_eq!(rows.len(), 67);
    let mut check = Check::new();
    check_cases(&rows, &mut check);
    check.finish("synthetic goldens");
}

#[test]
fn bound_params_match_the_bound_tables() {
    let tables_text = read(&fixture("bound-tables.json"));
    let params_text = read(&fixture("bound-params.json"));
    let bp: BoundParams = serde_json::from_str(&params_text).expect("bound-params.json");
    assert_eq!(bp.schema, BOUND_PARAMS_SCHEMA);
    assert_eq!(bp.bound_tables_sha256, sha256_hex(tables_text.as_bytes()));
    assert_eq!(bp.g_min_s, 0.1);
    let modes: BTreeSet<&str> = bp.modes.keys().map(String::as_str).collect();
    assert_eq!(modes, DEPLOY_MODES.into_iter().collect::<BTreeSet<_>>());
    let tables_v: Value = serde_json::from_str(&tables_text).unwrap();
    let params_v: Value = serde_json::from_str(&params_text).unwrap();
    for (mode, m) in &bp.modes {
        assert_eq!(
            params_v["modes"][mode]["table"], tables_v[mode],
            "{mode}: table differs"
        );
        assert_eq!(m.level, "0.8", "{mode}");
        assert_eq!(m.table.scheme, BucketScheme::Idle, "{mode}");
    }
    // Every table: schema v2, a curve `all` per level, increasing knots and
    // non-decreasing bounds (the curve is continuous and monotone).
    let (tables, _) = load_tables(&fixture("bound-tables.json"));
    assert_eq!(tables.len(), 37);
    for (mode, t) in &tables {
        assert_eq!(t.schema.as_deref(), Some(BOUND_TABLE_SCHEMA), "{mode}");
        assert_eq!(
            t.levels.keys().map(String::as_str).collect::<Vec<_>>(),
            ["0.8", "0.9"],
            "{mode}"
        );
        for (level, curves) in &t.levels {
            assert!(curves.contains_key("all"), "{mode} {level}");
            for (bucket, c) in curves {
                assert_eq!(c.knots_s.len(), c.b_s.len(), "{mode} {level} {bucket}");
                assert!(
                    c.knots_s.windows(2).all(|w| w[1] > w[0]),
                    "{mode} {level} {bucket}"
                );
                assert!(
                    c.b_s.windows(2).all(|w| w[1] >= w[0]),
                    "{mode} {level} {bucket}"
                );
            }
        }
        if let Some(m) = bp.modes.get(mode) {
            assert_eq!(&m.table, t, "{mode}");
        }
    }
    // BoundParams::bound uses the mode's shipped level.
    let view = BucketView {
        pred_s: Some(1.0),
        ..BucketView::default()
    };
    let shipped = &bp.modes["snap+prod"];
    assert_eq!(
        bp.bound("snap+prod", &view),
        error_bound(&view, &shipped.table, "0.8")
    );
    assert!(bp.bound("rfng", &view).is_err());
}

#[test]
#[ignore = "needs the full r5 goldens: E3_GOLDENS_DIR=<dir with goldens.jsonl, fit-goldens.jsonl, bound-tables.json>"]
fn full_nsc_goldens_match_the_reference() {
    let dir = PathBuf::from(std::env::var_os("E3_GOLDENS_DIR").expect(
        "set E3_GOLDENS_DIR to the r5 bounds dir (goldens.jsonl, fit-goldens.jsonl, bound-tables.json)",
    ));
    let (tables, sha) = load_tables(&dir.join("bound-tables.json"));
    let rows = jsonl(&dir.join("goldens.jsonl"));
    let fits = jsonl(&dir.join("fit-goldens.jsonl"));
    assert!(!rows.is_empty() && !fits.is_empty());
    let mut forecasts = Check::new();
    check_forecasts(&rows, &tables, &sha, &mut forecasts);
    let mut fit_check = Check::new();
    check_cases(&fits, &mut fit_check);
    // r5: 8,880 forecast goldens and 445 fit goldens.
    eprintln!(
        "E3_GOLDENS_DIR {}: {} forecast rows, {} fit rows",
        dir.display(),
        rows.len(),
        fits.len()
    );
    forecasts.finish("forecast goldens (full)");
    fit_check.finish("fit goldens (full)");
}
