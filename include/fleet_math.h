#ifndef FLEET_MATH_H
#define FLEET_MATH_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Eisenstein integer norm ─────────────────────────────────────────── */
/* N(a + bω) = a² − ab + b². Always non-negative; zero only at (0,0).   */

int64_t fleet_eisenstein_norm(int32_t a, int32_t b);

/* ── Laman rigidity ──────────────────────────────────────────────────── */
/* Minimum edges for a Laman-rigid graph: 2V − 3. Returns 0 for V < 2.  */

int32_t fleet_laman_edges(int32_t vertices);

/* Returns true if E ≥ 2V − 3 (the graph is generically rigid).         */

bool fleet_is_rigid(int32_t vertices, int32_t edges);

/* ── Holonomy check ─────────────────────────────────────────────────── */
/* Returns true if the product of `transforms[0..len)` equals 1.        */
/* Empty product (len == 0 or NULL) is identity by convention.           */

bool fleet_holonomy_check(const int64_t *transforms, size_t len);

/* ── Manhattan distance ─────────────────────────────────────────────── */
/* Sum of absolute differences: Σ|a[i] − b[i]|.                          */

int64_t fleet_manhattan_distance(const int32_t *a, const int32_t *b, size_t len);

/* ── Pythagorean-48 directional encoding ────────────────────────────── */
/* Quantize angle(x, y) into one of 48 sectors (0–47).                   */
/* 0 = positive x-axis, 12 = positive y-axis, 24 = negative x-axis.     */
/* Returns 0 for the zero vector.                                         */

int32_t fleet_pythagorean48_encode(double x, double y);

#ifdef __cplusplus
}
#endif

#endif /* FLEET_MATH_H */
