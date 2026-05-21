//! Minimal stub of fleet-math-c providing Eisenstein lattice snapping in f32.
//!
//! This is a pure-Rust placeholder that implements the same snap algorithm
//! used by the dodecet-encoder's `EisensteinConstraint`, but in f32 precision
//! to match the C bridge contract.

const SQRT_3_F32: f32 = 1.7320508_f32;

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
    // Convert to Eisenstein coordinates: (a, b) where z = a + b*ω
    // ω = -1/2 + i*√3/2
    // z = x + iy = a + b*(-1/2 + i*√3/2)
    // x = a - b/2,  y = b*√3/2
    // a = x + y/√3,  b = 2y/√3
    let b_approx = (2.0 * y) / SQRT_3_F32;
    let a_approx = x + y / SQRT_3_F32;

    let b0 = b_approx.floor() as i32;
    let a0 = a_approx.floor() as i32;

    let mut best_a = a0;
    let mut best_b = b0;
    let mut best_err = f32::MAX;
    let mut best_chamber: i32 = 0;

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

    // Determine chamber (0-5) based on position within the hexagonal cell
    let px = best_a as f32 - best_b as f32 * 0.5;
    let py = best_b as f32 * SQRT_3_F32 * 0.5;
    let dx = x - px;
    let dy = y - py;

    let chamber = if best_err < 1e-6 {
        0
    } else {
        // Angle-based chamber assignment (6 sectors, 0-5)
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
