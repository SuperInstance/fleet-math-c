//! Fleet math library — C-compatible math primitives for fleet operations.
//!
//! Provides Eisenstein integer operations, Laman rigidity checks, holonomy
//! verification, Manhattan distance for vector search, and Pythagorean-48
//! directional encoding. All public functions are `#[no_mangle] extern "C"`.

const SQRT_3_F32: f32 = 1.7320508_f32;

// ---------------------------------------------------------------------------
// Existing: Eisenstein lattice snapping (f32)
// ---------------------------------------------------------------------------

/// Result of snapping a point to the nearest Eisenstein integer.
#[derive(Debug, Clone)]
#[repr(C)]
pub struct SnapResult {
    pub snap_a: i32,
    pub snap_b: i32,
    pub error: f32,
    pub chamber: i32,
}

/// Snap a single (x, y) point to the nearest Eisenstein integer.
///
/// Searches the 9 Eisenstein integer candidates around the continuous
/// coordinates and picks the one with the smallest Euclidean distance.
pub fn snap(x: f32, y: f32) -> SnapResult {
    let b_approx = (2.0 * y) / SQRT_3_F32;
    let a_approx = x + y / SQRT_3_F32;

    let b0 = b_approx.floor() as i32;
    let a0 = a_approx.floor() as i32;

    let mut best_a = a0;
    let mut best_b = b0;
    let mut best_err = f32::MAX;

    for da in -1i32..=1 {
        for db in -1i32..=1 {
            let a = a0 + da;
            let b = b0 + db;
            let px = a as f32 - b as f32 * 0.5;
            let py = b as f32 * SQRT_3_F32 * 0.5;
            let dx = x - px;
            let dy = y - py;
            let err = dx * dx + dy * dy;
            if err < best_err {
                best_err = err;
                best_a = a;
                best_b = b;
            }
        }
    }

    let best_err = best_err.sqrt();

    let px = best_a as f32 - best_b as f32 * 0.5;
    let py = best_b as f32 * SQRT_3_F32 * 0.5;
    let dx = x - px;
    let dy = y - py;

    let chamber = if best_err < 1e-6 {
        0
    } else {
        let angle = dy.atan2(dx);
        let sector = ((angle + std::f32::consts::PI) / (std::f32::consts::PI / 3.0)).floor() as i32;
        (sector % 6).min(5).max(0)
    };

    SnapResult {
        snap_a: best_a,
        snap_b: best_b,
        error: best_err,
        chamber,
    }
}

/// Batch snap for interleaved (x, y) pairs.
///
/// `flat` must have even length: [x0, y0, x1, y1, ...].
pub fn batch_snap(flat: &[f32]) -> Vec<SnapResult> {
    assert!(flat.len() % 2 == 0, "flat must have even length");
    flat.chunks_exact(2)
        .map(|chunk| snap(chunk[0], chunk[1]))
        .collect()
}

// ---------------------------------------------------------------------------
// C-compatible API — Eisenstein norm
// ---------------------------------------------------------------------------

/// Eisenstein integer norm: N(a + bω) = a² − ab + b²
///
/// This is always non-negative and equals zero only for (0, 0).
#[no_mangle]
pub extern "C" fn fleet_eisenstein_norm(a: i32, b: i32) -> i64 {
    let a = a as i64;
    let b = b as i64;
    a * a - a * b + b * b
}

// ---------------------------------------------------------------------------
// C-compatible API — Laman rigidity
// ---------------------------------------------------------------------------

/// Minimum edges for a Laman-rigid graph on `vertices` vertices: 2V − 3.
///
/// Returns 0 for degenerate inputs (V < 2).
#[no_mangle]
pub extern "C" fn fleet_laman_edges(vertices: i32) -> i32 {
    if vertices < 2 {
        return 0;
    }
    2 * vertices - 3
}

/// Check whether a graph with `vertices` and `edges` satisfies the Laman
/// rigidity condition E ≥ 2V − 3.
///
/// Returns 1 (true) if rigid, 0 (false) otherwise.
#[no_mangle]
pub extern "C" fn fleet_is_rigid(vertices: i32, edges: i32) -> bool {
    if vertices < 2 {
        return edges >= 0;
    }
    edges >= 2 * vertices - 3
}

// ---------------------------------------------------------------------------
// C-compatible API — Holonomy check
// ---------------------------------------------------------------------------

/// Check if the product of integer transforms equals the identity (1).
///
/// Each element of `transforms` is treated as a multiplicative factor.
/// The product must equal exactly 1 for the check to pass.
///
/// # Safety
/// `transforms` must point to `len` valid `i64` values.
#[no_mangle]
pub extern "C" fn fleet_holonomy_check(transforms: *const i64, len: usize) -> bool {
    if transforms.is_null() || len == 0 {
        // Empty product is identity by convention
        return true;
    }
    let slice = unsafe { std::slice::from_raw_parts(transforms, len) };
    let product: i64 = slice.iter().product();
    product == 1
}

// ---------------------------------------------------------------------------
// C-compatible API — Manhattan distance
// ---------------------------------------------------------------------------

/// Compute the Manhattan (L1) distance between two integer vectors.
///
/// Sum of absolute differences: Σ|a[i] − b[i]|.
///
/// # Safety
/// `a` and `b` must each point to `len` valid `i32` values.
#[no_mangle]
pub extern "C" fn fleet_manhattan_distance(a: *const i32, b: *const i32, len: usize) -> i64 {
    if a.is_null() || b.is_null() || len == 0 {
        return 0;
    }
    let sa = unsafe { std::slice::from_raw_parts(a, len) };
    let sb = unsafe { std::slice::from_raw_parts(b, len) };
    sa.iter()
        .zip(sb.iter())
        .map(|(&x, &y)| (x as i64 - y as i64).abs())
        .sum()
}

// ---------------------------------------------------------------------------
// C-compatible API — Pythagorean-48 directional encoding
// ---------------------------------------------------------------------------

/// Number of discrete directions in Pythagorean-48 encoding.
pub const PYTHAGOREAN_48_DIRECTIONS: usize = 48;

/// Quantize a 2D angle (from x, y components) into one of 48 discrete
/// directions.
///
/// Each direction spans 7.5° (360°/48). Returns an integer in [0, 47]
/// where 0 corresponds to the positive x-axis.
///
/// Returns 0 if (x, y) is the zero vector.
#[no_mangle]
pub extern "C" fn fleet_pythagorean48_encode(x: f64, y: f64) -> i32 {
    if x == 0.0 && y == 0.0 {
        return 0;
    }
    let angle = y.atan2(x); // [-π, π]
    // Map to [0, 2π)
    let angle = if angle < 0.0 { angle + 2.0 * std::f64::consts::PI } else { angle };
    let sector = (angle / (2.0 * std::f64::consts::PI / 48.0)).floor() as i32;
    sector.min(47).max(0)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Existing snap tests --

    #[test]
    fn test_snap_origin() {
        let r = snap(0.0, 0.0);
        assert_eq!(r.snap_a, 0);
        assert_eq!(r.snap_b, 0);
        assert!(r.error < 0.001);
    }

    #[test]
    fn test_batch_snap() {
        let flat = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0];
        let results = batch_snap(&flat);
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].snap_a, 0);
        assert_eq!(results[0].snap_b, 0);
    }

    // -- Eisenstein norm tests --

    #[test]
    fn test_eisenstein_norm_zero() {
        assert_eq!(fleet_eisenstein_norm(0, 0), 0);
    }

    #[test]
    fn test_eisenstein_norm_unit() {
        // N(1, 0) = 1
        assert_eq!(fleet_eisenstein_norm(1, 0), 1);
        // N(0, 1) = 1
        assert_eq!(fleet_eisenstein_norm(0, 1), 1);
        // N(1, 1) = 1 - 1 + 1 = 1
        assert_eq!(fleet_eisenstein_norm(1, 1), 1);
    }

    #[test]
    fn test_eisenstein_norm_larger() {
        // N(2, 3) = 4 - 6 + 9 = 7
        assert_eq!(fleet_eisenstein_norm(2, 3), 7);
        // N(-2, 3) = 4 + 6 + 9 = 19
        assert_eq!(fleet_eisenstein_norm(-2, 3), 19);
    }

    #[test]
    fn test_eisenstein_norm_symmetry() {
        // N(a,b) = N(b,a) = N(-a,-b)
        assert_eq!(fleet_eisenstein_norm(3, 5), fleet_eisenstein_norm(5, 3));
        assert_eq!(fleet_eisenstein_norm(3, 5), fleet_eisenstein_norm(-3, -5));
    }

    // -- Laman edges tests --

    #[test]
    fn test_laman_edges_basic() {
        assert_eq!(fleet_laman_edges(2), 1);  // 2*2-3 = 1
        assert_eq!(fleet_laman_edges(3), 3);  // 2*3-3 = 3
        assert_eq!(fleet_laman_edges(4), 5);  // 2*4-3 = 5
        assert_eq!(fleet_laman_edges(10), 17); // 2*10-3 = 17
    }

    #[test]
    fn test_laman_edges_degenerate() {
        assert_eq!(fleet_laman_edges(0), 0);
        assert_eq!(fleet_laman_edges(1), 0);
        assert_eq!(fleet_laman_edges(-5), 0);
    }

    // -- Rigidity tests --

    #[test]
    fn test_is_rigid_true() {
        assert!(fleet_is_rigid(3, 3));  // exactly Laman
        assert!(fleet_is_rigid(4, 6));  // over-constrained
        assert!(fleet_is_rigid(2, 1));  // minimal
    }

    #[test]
    fn test_is_rigid_false() {
        assert!(!fleet_is_rigid(3, 2));  // one short
        assert!(!fleet_is_rigid(4, 4));  // need 5
        assert!(!fleet_is_rigid(10, 10)); // need 17
    }

    #[test]
    fn test_is_rigid_degenerate() {
        // V < 2: any non-negative edge count is "rigid" (vacuous)
        assert!(fleet_is_rigid(0, 0));
        assert!(fleet_is_rigid(1, 0));
        assert!(!fleet_is_rigid(0, -1)); // negative edges, not rigid
    }

    // -- Holonomy check tests --

    #[test]
    fn test_holonomy_identity() {
        let t: [i64; 1] = [1];
        assert!(fleet_holonomy_check(t.as_ptr(), 1));
    }

    #[test]
    fn test_holonomy_pair() {
        let t: [i64; 2] = [2, -2];
        // 2 * (-2) = -4, not identity
        assert!(!fleet_holonomy_check(t.as_ptr(), 2));
    }

    #[test]
    fn test_holonomy_product_one() {
        let _t: [i64; 3] = [2, 3, -6]; // not identity (product = -36)
        // For multiplicative holonomy with integers, only ±1 factors give product 1.
        let t2: [i64; 4] = [1, 1, 1, 1];
        assert!(fleet_holonomy_check(t2.as_ptr(), 4));
    }

    #[test]
    fn test_holonomy_null() {
        assert!(fleet_holonomy_check(std::ptr::null(), 0));
        assert!(fleet_holonomy_check(std::ptr::null(), 5)); // null pointer
    }

    #[test]
    fn test_holonomy_non_identity() {
        let t: [i64; 2] = [1, 2];
        assert!(!fleet_holonomy_check(t.as_ptr(), 2));
    }

    // -- Manhattan distance tests --

    #[test]
    fn test_manhattan_same() {
        let a: [i32; 3] = [1, 2, 3];
        let b: [i32; 3] = [1, 2, 3];
        assert_eq!(fleet_manhattan_distance(a.as_ptr(), b.as_ptr(), 3), 0);
    }

    #[test]
    fn test_manhattan_simple() {
        let a: [i32; 3] = [0, 0, 0];
        let b: [i32; 3] = [1, 2, 3];
        assert_eq!(fleet_manhattan_distance(a.as_ptr(), b.as_ptr(), 3), 6);
    }

    #[test]
    fn test_manhattan_negative() {
        let a: [i32; 2] = [-1, -2];
        let b: [i32; 2] = [1, 2];
        assert_eq!(fleet_manhattan_distance(a.as_ptr(), b.as_ptr(), 2), 6);
    }

    // -- Pythagorean-48 tests --

    #[test]
    fn test_pythag48_zero() {
        assert_eq!(fleet_pythagorean48_encode(0.0, 0.0), 0);
    }

    #[test]
    fn test_pythag48_positive_x() {
        // 0° → sector 0
        assert_eq!(fleet_pythagorean48_encode(1.0, 0.0), 0);
    }

    #[test]
    fn test_pythag48_positive_y() {
        // 90° → sector 12 (90 / 7.5 = 12)
        assert_eq!(fleet_pythagorean48_encode(0.0, 1.0), 12);
    }

    #[test]
    fn test_pythag48_negative_x() {
        // 180° → sector 24 (180 / 7.5 = 24)
        assert_eq!(fleet_pythagorean48_encode(-1.0, 0.0), 24);
    }

    #[test]
    fn test_pythag48_negative_y() {
        // 270° → sector 36 (270 / 7.5 = 36)
        assert_eq!(fleet_pythagorean48_encode(0.0, -1.0), 36);
    }

    #[test]
    fn test_pythag48_45deg() {
        // 45° → sector 6 (45 / 7.5 = 6)
        assert_eq!(fleet_pythagorean48_encode(1.0, 1.0), 6);
    }
}
