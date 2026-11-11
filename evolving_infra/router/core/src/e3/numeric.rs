// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Numeric routines of the E3 reference with the semantics of CPython 3.12:
//! compensated `sum()`, `max` / `min` ties, linear
//! quantiles, Gauss-Jordan elimination, ridge least squares, the
//! non-negative fit and the residual quantiles.

use std::cmp::Ordering;

use super::{RIDGE, SINGULAR_PIVOT};

/// CPython 3.12 `sum()` of floats (start value int 0).
///
/// Since Python 3.12 `sum()` adds floats with Neumaier's improvement of the
/// Kahan-Babuska compensated sum (`builtin_sum_impl` in `bltinmodule.c`):
/// the first item is added to int 0 (`0.0 + x`, which turns -0.0 into
/// +0.0); every later item updates the running sum and a separate
/// compensation; the compensation is added once at the end when it is
/// non-zero and finite. An empty sequence gives 0.
pub fn py_sum<I: IntoIterator<Item = f64>>(xs: I) -> f64 {
    let mut it = xs.into_iter();
    let mut f = match it.next() {
        None => return 0.0,
        // 0 + x in Python: +0.0 for a zero of either sign, else x.
        Some(x) => {
            if x == 0.0 {
                0.0
            } else {
                x
            }
        }
    };
    let mut c = 0.0_f64;
    for x in it {
        let t = f + x;
        if f.abs() >= x.abs() {
            c += (f - t) + x;
        } else {
            c += (x - t) + f;
        }
        f = t;
    }
    if c != 0.0 && c.is_finite() {
        f += c;
    }
    f
}

/// Python `max(a, b)`: `b` only when it is larger, so ties (and a NaN `b`)
/// return `a`.
pub fn py_max(a: f64, b: f64) -> f64 {
    if b > a {
        b
    } else {
        a
    }
}

/// Python `min(a, b)`: `b` only when it is smaller.
pub fn py_min(a: f64, b: f64) -> f64 {
    if b < a {
        b
    } else {
        a
    }
}

/// Order of Python's `sorted` on floats as a total preorder: numeric order,
/// -0.0 equal to +0.0 (the stable sort keeps their input order), NaN last.
pub fn py_cmp(a: &f64, b: &f64) -> Ordering {
    match a.partial_cmp(b) {
        Some(o) => o,
        None => a.is_nan().cmp(&b.is_nan()),
    }
}

/// Stable ascending sort with [`py_cmp`] (Python `sorted`).
pub fn sort_py(xs: &mut [f64]) {
    xs.sort_by(py_cmp);
}

/// Python `valid(x)` for an optional float: `Some(x)` when `x` is finite.
pub fn finite(x: Option<f64>) -> Option<f64> {
    x.filter(|v| v.is_finite())
}

/// Python `x or default` for a float or None: zero and None give `default`.
pub fn truthy_or(x: Option<f64>, default: f64) -> f64 {
    match x {
        Some(v) if v != 0.0 => v,
        _ => default,
    }
}

/// Linear-interpolation quantile of the finite values
/// (`estate_replay_kv.q`): sort, `k = (n - 1) p`,
/// `xs[floor k] + (xs[ceil k] - xs[floor k]) (k - floor k)`. None without a
/// finite value or with `p` outside `[0, 1]`.
pub fn quantile(xs: &[f64], p: f64) -> Option<f64> {
    if !(0.0..=1.0).contains(&p) {
        return None;
    }
    let mut v: Vec<f64> = xs.iter().copied().filter(|x| x.is_finite()).collect();
    if v.is_empty() {
        return None;
    }
    sort_py(&mut v);
    let k = (v.len() - 1) as f64 * p;
    let (a, b) = (k.floor(), k.ceil());
    let (lo, hi) = (v[a as usize], v[b as usize]);
    Some(lo + (hi - lo) * (k - a))
}

/// Gauss-Jordan elimination with partial pivoting: the pivot is
/// the first row with the largest `|a_ri|` among the rows `>= i`; None
/// (singular) when `|pivot| < 1e-300`. `a` is square and `b` has one value
/// per row; None for other shapes.
pub fn solve(a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
    let n = a.len();
    if b.len() != n || a.iter().any(|row| row.len() != n) {
        return None;
    }
    let mut m: Vec<Vec<f64>> = a
        .iter()
        .zip(b)
        .map(|(row, &bi)| {
            let mut r = row.clone();
            r.push(bi);
            r
        })
        .collect();
    for i in 0..n {
        let mut p = i;
        let mut best = m[i][i].abs();
        for (r, row) in m.iter().enumerate().skip(i + 1) {
            let v = row[i].abs();
            if v > best {
                best = v;
                p = r;
            }
        }
        m.swap(i, p);
        if m[i][i].abs() < SINGULAR_PIVOT {
            return None;
        }
        let pivot = m[i].clone();
        for (r, row) in m.iter_mut().enumerate() {
            if r == i {
                continue;
            }
            let f = row[i] / pivot[i];
            // Columns i ..= n: m[r][c] -= f * m[i][c].
            for (rc, pc) in row[i..].iter_mut().zip(&pivot[i..]) {
                *rc -= f * pc;
            }
        }
    }
    Some((0..n).map(|i| m[i][n] / m[i][i]).collect())
}

/// Ridge least squares on columns scaled to unit norm: column
/// scales `s_j = sqrt(sum_r x_rj^2)` (1 when 0), `Z = X / s`,
/// `A = Z'Z + ridge I`, `b = Z'y` (Python `sum()`, in row order), solve,
/// `coef_j = c_j / s_j`. None when `x` is empty, a row is shorter than the
/// first one, or the system is singular.
pub fn lstsq<R: AsRef<[f64]>>(x: &[R], y: &[f64], ridge: f64) -> Option<Vec<f64>> {
    let k = x.first()?.as_ref().len();
    if x.iter().any(|r| r.as_ref().len() < k) {
        return None;
    }
    let s: Vec<f64> = (0..k)
        .map(|j| {
            let norm = py_sum(x.iter().map(|r| {
                let v = r.as_ref()[j];
                v * v
            }))
            .sqrt();
            // Python `math.sqrt(...) or 1.`
            if norm != 0.0 {
                norm
            } else {
                1.0
            }
        })
        .collect();
    let z: Vec<Vec<f64>> = x
        .iter()
        .map(|r| {
            let r = r.as_ref();
            (0..k).map(|j| r[j] / s[j]).collect()
        })
        .collect();
    let a: Vec<Vec<f64>> = (0..k)
        .map(|i| {
            (0..k)
                .map(|j| {
                    let diagonal = if i == j { ridge } else { 0.0 };
                    py_sum(z.iter().map(|r| r[i] * r[j])) + diagonal
                })
                .collect()
        })
        .collect();
    let b: Vec<f64> = (0..k)
        .map(|i| py_sum(z.iter().zip(y).map(|(r, &v)| r[i] * v)))
        .collect();
    let coef = solve(&a, &b)?;
    Some(coef.iter().zip(&s).map(|(c, sj)| c / sj).collect())
}

/// Least squares with every non-intercept coefficient `>= 0`:
/// keep column 0 and every column `j >= 1` whose values are not all equal;
/// fit [`lstsq`] on the kept columns; drop the kept non-intercept columns
/// with a negative coefficient and refit, at most `k` rounds. Returns the
/// full vector (0 for dropped columns), or None on a singular system or
/// after `k` rounds.
pub fn nonneg_fit<R: AsRef<[f64]>>(xs: &[R], ys: &[f64]) -> Option<Vec<f64>> {
    let k = xs.first()?.as_ref().len();
    if k == 0 || xs.iter().any(|r| r.as_ref().len() < k) {
        return None;
    }
    let mut keep: Vec<usize> = vec![0];
    for j in 1..k {
        let first = xs[0].as_ref()[j];
        let (mut hi, mut lo) = (first, first);
        for r in &xs[1..] {
            let v = r.as_ref()[j];
            hi = py_max(hi, v);
            lo = py_min(lo, v);
        }
        if hi - lo > 0.0 {
            keep.push(j);
        }
    }
    for _ in 0..k {
        let sub: Vec<Vec<f64>> = xs
            .iter()
            .map(|r| {
                let r = r.as_ref();
                keep.iter().map(|&j| r[j]).collect()
            })
            .collect();
        let c = lstsq(&sub, ys, RIDGE)?;
        let neg: Vec<usize> = keep
            .iter()
            .zip(&c)
            .filter(|&(&j, &v)| j != 0 && v < 0.0)
            .map(|(&j, _)| j)
            .collect();
        if neg.is_empty() {
            let mut full = vec![0.0; k];
            for (&j, &v) in keep.iter().zip(&c) {
                full[j] = v;
            }
            return Some(full);
        }
        keep.retain(|j| !neg.contains(j));
    }
    None
}

/// The 0.05, 0.10, ..., 0.95 order statistics of the sorted residuals `rs`
/// : `rs[int(j / 20 x (n - 1))]` for `j = 1 ... 19` (truncation).
/// Empty for an empty input.
pub fn resid_quantiles(rs: &[f64]) -> Vec<f64> {
    if rs.is_empty() {
        return Vec::new();
    }
    let last = (rs.len() - 1) as f64;
    (1..20)
        .map(|j| {
            let qq = j as f64 / 20.0;
            rs[(qq * last) as usize]
        })
        .collect()
}

/// Page-rounded size of a chunk: `((max(0, t) + page - 1) // page) * page`
/// `page` must be at least 1 (`Params::check`).
pub fn ceil_page(t: i64, page: i64) -> i64 {
    ((t.max(0) + page - 1) / page) * page
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn py_sum_is_the_cpython_312_compensated_sum() {
        // Left to right gives 0.6000000000000001 and 0.9999999999999999.
        assert_eq!(py_sum([0.1, 0.2, 0.3]), 0.6);
        assert_eq!(py_sum([0.1; 10]), 1.0);
        assert_ne!(0.1 + 0.2 + 0.3, 0.6);
        // The compensation recovers the small term (left to right: 0.0).
        assert_eq!(py_sum([1e100, 1.0, -1e100]), 1.0);
        // Empty: 0; one item: 0 + x, so -0.0 becomes +0.0.
        assert_eq!(py_sum(std::iter::empty::<f64>()), 0.0);
        assert!(py_sum([-0.0]).is_sign_positive());
        assert!(py_sum([-0.0, -0.0]).is_sign_positive());
        assert_eq!(py_sum([2.5]), 2.5);
        // An infinite compensation is not added (it would give NaN).
        assert_eq!(py_sum([f64::MAX, f64::MAX]), f64::INFINITY);
    }

    #[test]
    fn py_max_and_py_min_keep_the_first_of_equal_values() {
        assert!(py_max(-0.0, 0.0).is_sign_negative());
        assert!(py_max(0.0, -0.0).is_sign_positive());
        assert!(py_min(0.0, -0.0).is_sign_positive());
        assert_eq!(py_max(1.0, 2.0), 2.0);
        assert_eq!(py_min(1.0, 2.0), 1.0);
        assert_eq!(py_max(1.0, f64::NAN), 1.0);
        assert_eq!(py_max(f64::NEG_INFINITY, -5.0), -5.0);
    }

    #[test]
    fn py_cmp_sorts_like_python_sorted() {
        let mut v = [3.0, -0.0, 1.0, 0.0, -2.0];
        sort_py(&mut v);
        assert_eq!(v, [-2.0, -0.0, 0.0, 1.0, 3.0]);
        // Stable: -0.0 stays before +0.0 as in the input, and the reverse.
        assert!(v[1].is_sign_negative() && v[2].is_sign_positive());
        let mut w = [0.0, -0.0];
        sort_py(&mut w);
        assert!(w[0].is_sign_positive() && w[1].is_sign_negative());
        let mut x = [f64::NAN, 1.0, f64::NAN, 0.5];
        sort_py(&mut x);
        assert_eq!(&x[..2], &[0.5, 1.0]);
        assert!(x[2].is_nan() && x[3].is_nan());
    }

    #[test]
    fn helpers_finite_and_truthy_or() {
        assert_eq!(finite(Some(1.5)), Some(1.5));
        assert_eq!(finite(Some(f64::INFINITY)), None);
        assert_eq!(finite(Some(f64::NAN)), None);
        assert_eq!(finite(None), None);
        assert_eq!(truthy_or(Some(0.0), 1.0), 1.0);
        assert_eq!(truthy_or(Some(-0.0), 1.0), 1.0);
        assert_eq!(truthy_or(None, 1.0), 1.0);
        assert_eq!(truthy_or(Some(0.25), 1.0), 0.25);
    }

    #[test]
    fn quantile_interpolates_linearly_over_finite_values() {
        assert_eq!(quantile(&[3.0, 1.0, 2.0], 0.5), Some(2.0));
        assert_eq!(quantile(&[4.0, 1.0, 3.0, 2.0], 0.5), Some(2.5));
        // k = 99 x 0.01 = 0.99: 1 + (2 - 1) x 0.99.
        let xs: Vec<f64> = (1..=100).map(f64::from).collect();
        assert_eq!(quantile(&xs, 0.01), Some(1.0 + 0.99));
        assert_eq!(quantile(&xs, 0.0), Some(1.0));
        assert_eq!(quantile(&xs, 1.0), Some(100.0));
        // Non-finite values are left out; nothing left gives None.
        assert_eq!(
            quantile(&[f64::INFINITY, 1.0, f64::NAN, 3.0], 0.5),
            Some(2.0)
        );
        assert_eq!(quantile(&[f64::NAN], 0.5), None);
        assert_eq!(quantile(&[], 0.5), None);
        assert_eq!(quantile(&[1.0], 1.5), None);
        assert_eq!(quantile(&[7.0], 0.5), Some(7.0));
    }

    #[test]
    fn solve_pivots_and_reports_singular_systems() {
        let x = solve(&[vec![2.0, 1.0], vec![1.0, 3.0]], &[3.0, 5.0]).unwrap();
        assert!(
            (x[0] - 0.8).abs() < 1e-15 && (x[1] - 1.4).abs() < 1e-15,
            "{x:?}"
        );
        // A zero on the diagonal needs the row swap.
        assert_eq!(
            solve(&[vec![0.0, 1.0], vec![1.0, 0.0]], &[2.0, 3.0]),
            Some(vec![3.0, 2.0])
        );
        assert_eq!(solve(&[vec![1.0, 2.0], vec![2.0, 4.0]], &[1.0, 2.0]), None);
        assert_eq!(solve(&[vec![1.0, 2.0]], &[1.0]), None);
        assert_eq!(solve(&[], &[]), Some(vec![]));
        // Ties in |pivot| take the first row: no swap, same result.
        let y = solve(&[vec![1.0, 1.0], vec![-1.0, 1.0]], &[2.0, 0.0]).unwrap();
        assert!(
            (y[0] - 1.0).abs() < 1e-15 && (y[1] - 1.0).abs() < 1e-15,
            "{y:?}"
        );
    }

    #[test]
    fn lstsq_recovers_a_line_and_survives_a_zero_column() {
        let x: Vec<[f64; 3]> = (1..=10).map(|i| [1.0, f64::from(i), 0.0]).collect();
        let y: Vec<f64> = (1..=10).map(|i| 2.0 + 3.0 * f64::from(i)).collect();
        let c = lstsq(&x, &y, RIDGE).unwrap();
        // The ridge term (1e-6 on unit-norm columns) biases the fit slightly;
        // the values are the reference's (router_e3.lstsq), bit for bit.
        assert!(
            (c[0] - 2.0).abs() < 1e-4 && (c[1] - 3.0).abs() < 1e-4,
            "{c:?}"
        );
        assert_eq!(c, [2.0000676660257675, 2.9999873334375575, 0.0]);
        let empty: Vec<[f64; 2]> = Vec::new();
        assert_eq!(lstsq(&empty, &[], RIDGE), None);
        assert_eq!(
            lstsq(&[vec![1.0, 2.0], vec![1.0]], &[1.0, 2.0], RIDGE),
            None
        );
        // Without the ridge a duplicated column is singular.
        let dup: Vec<[f64; 2]> = (1..=5).map(|i| [f64::from(i), f64::from(i)]).collect();
        assert_eq!(lstsq(&dup, &[1.0, 2.0, 3.0, 4.0, 5.0], 0.0), None);
    }

    #[test]
    fn nonneg_fit_drops_negative_and_constant_columns() {
        // y = 1 + 2 a - 3 b: b gets a negative coefficient, is dropped, and
        // the refit on [1, a] is returned with 0 for b; column 3 is constant.
        let rows: Vec<[f64; 4]> = (0..40)
            .map(|i| {
                let a = f64::from(i % 7);
                let b = f64::from((i * 3) % 5);
                [1.0, a, b, 4.0]
            })
            .collect();
        let ys: Vec<f64> = rows.iter().map(|r| 1.0 + 2.0 * r[1] - 3.0 * r[2]).collect();
        let c = nonneg_fit(&rows, &ys).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c[2], 0.0);
        assert_eq!(c[3], 0.0);
        assert!(c[1] >= 0.0);
        // Reference values (router_e3.nonneg_fit), bit for bit.
        assert_eq!(c, [-4.720615755468356, 1.9028245134205608, 0.0, 0.0]);
        // A clean non-negative model is recovered with every column kept.
        let ys2: Vec<f64> = rows.iter().map(|r| 1.0 + 2.0 * r[1] + 0.5 * r[2]).collect();
        let c2 = nonneg_fit(&rows, &ys2).unwrap();
        assert!(
            (c2[0] - 1.0).abs() < 1e-4 && (c2[1] - 2.0).abs() < 1e-5,
            "{c2:?}"
        );
        assert!((c2[2] - 0.5).abs() < 1e-5 && c2[3] == 0.0, "{c2:?}");
        assert_eq!(
            c2,
            [
                1.000015216914225,
                1.99999446655971,
                0.49999984585569596,
                0.0
            ]
        );
        let empty: Vec<[f64; 2]> = Vec::new();
        assert_eq!(nonneg_fit(&empty, &[]), None);
    }

    #[test]
    fn resid_quantiles_truncate_the_index() {
        let rs: Vec<f64> = (0..=20).map(f64::from).collect();
        // n - 1 = 20: index int(j / 20 x 20) = j.
        let q = resid_quantiles(&rs);
        assert_eq!(q, (1..20).map(f64::from).collect::<Vec<_>>());
        // n - 1 = 29: index int(j / 20 x 29), e.g. j = 1 -> 1, j = 19 -> 27.
        let rs30: Vec<f64> = (0..30).map(f64::from).collect();
        let q30 = resid_quantiles(&rs30);
        assert_eq!(q30.len(), 19);
        assert_eq!(q30[0], 1.0);
        assert_eq!(q30[9], 14.0);
        assert_eq!(q30[18], 27.0);
        assert!(resid_quantiles(&[]).is_empty());
        assert_eq!(resid_quantiles(&[5.0]), [5.0; 19]);
    }

    #[test]
    fn ceil_page_rounds_up_to_whole_pages() {
        assert_eq!(ceil_page(0, 64), 0);
        assert_eq!(ceil_page(1, 64), 64);
        assert_eq!(ceil_page(64, 64), 64);
        assert_eq!(ceil_page(65, 64), 128);
        assert_eq!(ceil_page(-5, 64), 0);
        assert_eq!(ceil_page(7, 1), 7);
    }
}
