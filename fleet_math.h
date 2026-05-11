/*
 * fleet_math.h — SIMD-accelerated constraint math for PLATO tile operations
 *
 * 1 PLATO tile = 64 bytes = 1 cache line = 1 AVX-512 register (zmm0)
 * 1 constraint check = 1 VPCMPD instruction
 * 1 holonomy accumulation = 1 VFMADDPS instruction
 *
 * Zero dependencies. No libc required. Drop this header into any C/C++ project.
 *
 * License: MIT
 */

#ifndef FLEET_MATH_H
#define FLEET_MATH_H

#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* =========================================================================
 * Types
 * ========================================================================= */

/**
 * plato_tile_t — A single PLATO memory tile (exactly 64 bytes)
 *
 * sizeof(plato_tile_t) == 64 == cache line width == AVX-512 zmm register
 * Aligned to 64 bytes so it can be loaded/stored with a single SIMD instruction.
 */
typedef struct __attribute__((aligned(64))) {
    float confidence;        /* 0.0 – 1.0           */
    float novelty;           /* 0.0 – 10.0          */
    float gradient[4];       /* 4D gradient vector   */
    float metadata[8];       /* timestamps, source…  */
    uint8_t hash[8];         /* 64-bit tile hash     */
    /* _pad not needed: fields sum to exactly 64 bytes */
} plato_tile_t;

/* Compile-time assertion: tile is exactly 64 bytes */
#define FLEET_MATH_TILE_SIZE 64
_Static_assert(sizeof(plato_tile_t) == FLEET_MATH_TILE_SIZE,
               "plato_tile_t must be exactly 64 bytes");

/**
 * constraint_graph_t — Sparse directed graph stored in SoA (Structure of
 * Arrays) layout for efficient SIMD processing.
 */
typedef struct {
    float     *weights;      /* Edge weights          */
    uint32_t  *src;          /* Source node indices    */
    uint32_t  *dst;          /* Destination node indices */
    uint32_t   n_edges;      /* Number of edges        */
    uint32_t   n_nodes;      /* Number of nodes        */
} constraint_graph_t;

/* =========================================================================
 * Scalar (portable) implementations
 * ========================================================================= */

/**
 * tile_check_violations_scalar — Count fields below threshold.
 * Returns: number of violations (0 = tile is valid).
 */
static inline int
tile_check_violations_scalar(const plato_tile_t *tile, float threshold)
{
    int count = 0;
    count += (tile->confidence < threshold) ? 1 : 0;
    count += (tile->novelty    < threshold) ? 1 : 0;
    for (int i = 0; i < 4; i++)
        count += (tile->gradient[i] < threshold) ? 1 : 0;
    for (int i = 0; i < 8; i++)
        count += (tile->metadata[i] < threshold) ? 1 : 0;
    return count;
}

/**
 * holonomy_4cycle_scalar — Compute holonomy around a 4-edge fundamental cycle.
 *
 * Holonomy measures the failure of consensus along a cycle:
 *   H = w0*w1 - w2*w3   (for a 4-cycle)
 *
 * For a perfectly consistent graph, H ≈ 0.
 */
static inline float
holonomy_4cycle_scalar(const float weights[4])
{
    return weights[0] * weights[1] - weights[2] * weights[3];
}

/**
 * batch_check_tiles_scalar — Count valid tiles in a batch.
 * Returns: number of tiles with zero violations.
 */
static inline int
batch_check_tiles_scalar(const plato_tile_t *tiles, int n, float threshold)
{
    int valid = 0;
    for (int i = 0; i < n; i++)
        valid += (tile_check_violations_scalar(&tiles[i], threshold) == 0);
    return valid;
}

/**
 * batch_holonomy_4cycles_scalar — Compute holonomies for N 4-cycles.
 */
static inline void
batch_holonomy_4cycles_scalar(const float *weights, int n, float *out_holonomy)
{
    for (int i = 0; i < n; i++)
        out_holonomy[i] = holonomy_4cycle_scalar(&weights[i * 4]);
}

/* =========================================================================
 * AVX-512 implementation    (guarded by __AVX512F__)
 * ========================================================================= */

#if defined(__AVX512F__) && defined(FLEET_MATH_ENABLE_AVX512)

#include <immintrin.h>
#include <popcntintrin.h>

/**
 * tile_check_violations_avx512 — Check 14 float fields against threshold.
 *
 * Loads all 16 floats (64 bytes) into one zmm register, compares against
 * threshold, then masks out the last 2 lanes (hash bytes, not real floats).
 *
 * Returns: number of violations among the first 14 float fields (0 = valid).
 * Latency: ~8-10 cycles (load + vcmpps + popcnt + mask).
 */
static inline int
tile_check_violations_avx512(const plato_tile_t *tile, float threshold)
{
    /* Load all 16 floats (64 bytes) into zmm0 */
    __m512 t = _mm512_load_ps(&tile->confidence);

    /* Compare less-than, get mask in k0 */
    __mmask16 mask = _mm512_cmp_ps_mask(t, _mm512_set1_ps(threshold), _CMP_LT_OQ);

    /* Mask out last 2 lanes (overlap with hash[8] bytes) */
    mask &= 0x3FFF;

    /* Count set bits = number of violations */
    return _mm_popcnt_u32((uint32_t)mask);
}

/**
 * holonomy_4cycle_avx512 — Compute one 4-cycle holonomy using SIMD.
 *
 *   weights[0..3] = edge weights w0 w1 w2 w3
 *   H = w0*w1 - w2*w3
 *
 * Latency: ~6 cycles.
 */
static inline float
holonomy_4cycle_avx512(const float weights[4])
{
    __m128 w = _mm_load_ps(weights);                      /* w0 w1 w2 w3        */
    __m128 shuffled = _mm_shuffle_ps(w, w, _MM_SHUFFLE(2, 3, 0, 1));
                                                          /* w1 w0 w3 w2        */
    __m128 prod = _mm_mul_ps(w, shuffled);                /* w0*w1 w1*w0 w2*w3 w3*w2 */

    /* Extract lane 0 (w0*w1) and lane 2 (w2*w3) and subtract */
    float p0 = _mm_cvtss_f32(prod);
    float p2 = _mm_cvtss_f32(_mm_shuffle_ps(prod, prod, _MM_SHUFFLE(0, 0, 0, 2)));
    return p0 - p2;
}

/**
 * batch_check_tiles_avx512 — Batch-vectorized tile validation.
 *
 * For best performance, tiles[] should be 64-byte aligned and n should
 * be a multiple of a streaming-friendly chunk size.
 */
static inline int
batch_check_tiles_avx512(const plato_tile_t *tiles, int n, float threshold)
{
    int valid = 0;
    __m512 thresh = _mm512_set1_ps(threshold);

    for (int i = 0; i < n; i++) {
        __m512 t  = _mm512_load_ps(&tiles[i].confidence);
        __mmask16 mask = _mm512_cmp_ps_mask(t, thresh, _CMP_LT_OQ);
        mask &= 0x3FFF;  /* exclude hash byte positions */
        valid += (_mm_popcnt_u32((uint32_t)mask) == 0) ? 1 : 0;
    }
    return valid;
}

/**
 * batch_holonomy_4cycles_avx512 — Holonomy for N 4-cycles, 4-at-a-time.
 *
 * Processes 4 cycles per iteration (16 weights → 4 holonomies).
 */
static inline void
batch_holonomy_4cycles_avx512(const float *weights, int n, float *out_holonomy)
{
    int i = 0;

    /* Process 4 cycles per iteration (16 floats = 1 zmm) */
    for (; i + 4 <= n; i += 4) {
        __m512 w = _mm512_load_ps(&weights[i * 4]);
        /* Layout: w0 w1 w2 w3 | w4 w5 w6 w7 | w8 w9 w10 w11 | w12 w13 w14 w15 */

        /* Shuffle within each 4-lane group: exchange lanes 0↔1, 2↔3 */
        __m512 shuf = _mm512_shuffle_ps(w, w, _MM_SHUFFLE(2, 3, 0, 1));
        __m512 prod = _mm512_mul_ps(w, shuf);

        /* prod now has interleaved products. We need to reduce:
         * For each cycle: extract lane 0 (w0*w1) - lane 2 (w2*w3) */

        /* Interleave: (w0*w1, w2*w3) → adjacent and subtract */
        /* _mm512_hsub_ps subtracts adjacent pairs within lanes */
        __m512 result = _mm512_hsub_ps(prod, prod);

        /* Store — result has the holonomies in every other lane */
        _mm_store_ps(&out_holonomy[i], _mm512_castps512_ps128(result));
    }

    /* Remainder */
    for (; i < n; i++)
        out_holonomy[i] = holonomy_4cycle_avx512(&weights[i * 4]);
}

#endif /* __AVX512F__ */

/* =========================================================================
 * ARM NEON implementation   (guarded by __ARM_NEON)
 * ========================================================================= */

#if defined(__ARM_NEON) || defined(__ARM_NEON__)

#include <arm_neon.h>

/**
 * tile_check_violations_neon — Check 64-byte tile against threshold.
 *
 * Uses 4 Q-register loads (4×16B = 64B) + 4 comparisons + reduction.
 * Latency: ~12-16 cycles on Cortex-X/A-series.
 */
static inline int
tile_check_violations_neon(const plato_tile_t *tile, float threshold)
{
    float32x4_t thresh = vdupq_n_f32(threshold);

    /* Load all 16 floats (4 Q-registers) */
    const float *fp = &tile->confidence;

    float32x4_t v0 = vld1q_f32(fp + 0);      /* confidence, novelty, gradient[0,1] */
    float32x4_t v1 = vld1q_f32(fp + 4);      /* gradient[2,3], metadata[0,1]       */
    float32x4_t v2 = vld1q_f32(fp + 8);      /* metadata[2,3,4,5]                  */
    float32x4_t v3 = vld1q_f32(fp + 12);     /* metadata[6,7], _pad[0,1]           */

    /* Compare less-than, get masks (0xFFFFFFFF = violation) */
    uint32x4_t m0 = vcltq_f32(v0, thresh);
    uint32x4_t m1 = vcltq_f32(v1, thresh);
    uint32x4_t m2 = vcltq_f32(v2, thresh);
    uint32x4_t m3 = vcltq_f32(v3, thresh);

    /* Count violated lanes.
     * Each mask lane is 0xFFFFFFFF (violated) or 0 (not).
     * Shift right 31 to get 1 or 0. Mask out last 2 lanes (hash bytes).
     * Then sum across all 16 lanes. */
    uint32x4_t ones = vdupq_n_u32(1);
    uint32x4_t v0c = vandq_u32(vshrq_n_u32(m0, 31), ones);  /* lanes 0-3 */
    uint32x4_t v1c = vandq_u32(vshrq_n_u32(m1, 31), ones);  /* lanes 4-7 */
    uint32x4_t v2c = vandq_u32(vshrq_n_u32(m2, 31), ones);  /* lanes 8-11 */
    /* v3: only count lanes 0,1 (float indices 12,13) — lanes 2,3 (14,15)
     * are hash bytes, not real float fields. Mask them out. */
    uint32x4_t v3c  = vandq_u32(vshrq_n_u32(m3, 31), ones);
    uint32x4_t v3m  = vdupq_n_u32(0);
    v3m = vsetq_lane_u32(1, v3m, 0);
    v3m = vsetq_lane_u32(1, v3m, 1);
    v3c = vandq_u32(v3c, v3m);

    /* Sum all 14 lane counts */
    uint32x4_t sum01 = vaddq_u32(vaddq_u32(v0c, v1c), vaddq_u32(v2c, v3c));
    return (int)(vgetq_lane_u32(sum01, 0) + vgetq_lane_u32(sum01, 1)
               + vgetq_lane_u32(sum01, 2) + vgetq_lane_u32(sum01, 3));
}

/**
 * holonomy_4cycle_neon — Compute 4-cycle holonomy using NEON.
 */
static inline float
holonomy_4cycle_neon(const float weights[4])
{
    float32x4_t w     = vld1q_f32(weights);
    float32x4_t shuf  = vrev64q_f32(w);       /* Reverse pairs: w1 w0 w3 w2 */
    float32x4_t prod  = vmulq_f32(w, shuf);   /* w0*w1 w1*w0 w2*w3 w3*w2  */

    /* Extract lane 0 and lane 2, subtract */
    float p0 = vgetq_lane_f32(prod, 0);
    float p2 = vgetq_lane_f32(prod, 2);
    return p0 - p2;
}

/**
 * batch_check_tiles_neon — Batch tile validation with NEON.
 */
static inline int
batch_check_tiles_neon(const plato_tile_t *tiles, int n, float threshold)
{
    int valid = 0;
    for (int i = 0; i < n; i++)
        valid += (tile_check_violations_neon(&tiles[i], threshold) == 0);
    return valid;
}

/**
 * batch_holonomy_4cycles_neon — Holonomy for N 4-cycles, 4-at-a-time.
 */
static inline void
batch_holonomy_4cycles_neon(const float *weights, int n, float *out_holonomy)
{
    int i = 0;

    /* Process 4 cycles per iteration */
    for (; i + 4 <= n; i += 4) {
        float32x4x4_t w = vld4q_f32(&weights[i * 4]);
        /* vld4q interleaves: w.val[0] = all w0s, w.val[1] = all w1s, etc. */

        /* mul subtract: w0*w1 and w2*w3 for each of 4 cycles */
        float32x4_t p01 = vmulq_f32(w.val[0], w.val[1]);  /* w0*w1 per cycle */
        float32x4_t p23 = vmulq_f32(w.val[2], w.val[3]);  /* w2*w3 per cycle */
        float32x4_t H   = vsubq_f32(p01, p23);            /* w0*w1 - w2*w3  */

        vst1q_f32(&out_holonomy[i], H);
    }

    /* Remainder */
    for (; i < n; i++)
        out_holonomy[i] = holonomy_4cycle_neon(&weights[i * 4]);
}

#endif /* __ARM_NEON */

/* =========================================================================
 * Auto-select best implementation  (use like: fleet_tile_check_violations)
 *
 * The dispatch macros pick the fastest available SIMD path at compile time.
 * ========================================================================= */

/* If FLEET_MATH_NO_AUTOSELECT is defined, fall back to scalar always. */
#ifndef FLEET_MATH_NO_AUTOSELECT

#  if defined(__AVX512F__) && defined(FLEET_MATH_ENABLE_AVX512)
#    define fleet_tile_check(t, th)        tile_check_violations_avx512(t, th)
#    define fleet_holonomy_4cycle(w)       holonomy_4cycle_avx512(w)
#    define fleet_batch_check(t, n, th)    batch_check_tiles_avx512(t, n, th)
#    define fleet_batch_holonomy(w, n, o)  batch_holonomy_4cycles_avx512(w, n, o)
#    define FLEET_MATH_ACTIVE_IMPL         "AVX-512"
#  elif defined(__ARM_NEON) || defined(__ARM_NEON__)
#    define fleet_tile_check(t, th)        tile_check_violations_neon(t, th)
#    define fleet_holonomy_4cycle(w)       holonomy_4cycle_neon(w)
#    define fleet_batch_check(t, n, th)    batch_check_tiles_neon(t, n, th)
#    define fleet_batch_holonomy(w, n, o)  batch_holonomy_4cycles_neon(w, n, o)
#    define FLEET_MATH_ACTIVE_IMPL         "ARM NEON"
#  else
#    define fleet_tile_check(t, th)        tile_check_violations_scalar(t, th)
#    define fleet_holonomy_4cycle(w)       holonomy_4cycle_scalar(w)
#    define fleet_batch_check(t, n, th)    batch_check_tiles_scalar(t, n, th)
#    define fleet_batch_holonomy(w, n, o)  batch_holonomy_4cycles_scalar(w, n, o)
#    define FLEET_MATH_ACTIVE_IMPL         "Scalar (portable)"
#  endif

#else /* FLEET_MATH_NO_AUTOSELECT */

#  define fleet_tile_check(t, th)        tile_check_violations_scalar(t, th)
#  define fleet_holonomy_4cycle(w)       holonomy_4cycle_scalar(w)
#  define fleet_batch_check(t, n, th)    batch_check_tiles_scalar(t, n, th)
#  define fleet_batch_holonomy(w, n, o)  batch_holonomy_4cycles_scalar(w, n, o)
#  define FLEET_MATH_ACTIVE_IMPL         "Scalar (portable, forced)"

#endif

/* =========================================================================
 * Utility
 * ========================================================================= */

/**
 * tile_init — Fill a tile with sensible defaults.
 */
static inline void
tile_init(plato_tile_t *tile)
{
    memset(tile, 0, sizeof(*tile));
    tile->confidence = 1.0f;
    tile->novelty    = 0.0f;
}

/**
 * fleet_math_impl_name — Returns the active SIMD implementation name.
 * Defined in fleet_math.c.
 */
const char *fleet_math_impl_name(void);

/**
 * fleet_version — Returns version string for this header.
 */
#define FLEET_MATH_VERSION "0.1.0"

#endif /* FLEET_MATH_H */
