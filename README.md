# fleet-math-c — SIMD-Accelerated Constraint Math for PLATO Tiles

64 bytes = 1 cache line = 1 AVX-512 zmm register = 1 constraint operation.

Three C files, zero dependencies. Drop them into any project that needs fast constraint math — tile violation checks, 4-cycle holonomy, or batch operations on PLATO tile arrays.

## The Core Idea

A [PLATO tile](https://github.com/SuperInstance/plato-types) is exactly 64 bytes — one cache line on every modern CPU. That's also the width of an AVX-512 register. This means you can load an entire tile, check all its float fields against a threshold, and count violations with a single SIMD instruction chain. No loops over individual fields. No branch mispredictions.

The library provides three SIMD backends that compile conditionally:

| Backend | Guard | Hardware |
|---------|-------|----------|
| **AVX-512** | `__AVX512F__` | Intel Xeon (Cascade Lake, Ice Lake), AMD Zen 4+ |
| **ARM NEON** | `__ARM_NEON` | Apple M-series, AWS Graviton, ARM Neoverse |
| **Scalar** | (always available) | Any C99 compiler — fallback and reference |

At compile time, the preprocessor selects the best available backend. You can also force scalar with `-DFLEET_MATH_NO_AUTOSELECT`.

## Quick Start

```c
#include "fleet_math.h"

/* Create and initialize a tile */
plato_tile_t tile;
tile_init(&tile);

tile.confidence = 0.8f;
tile.novelty    = 3.2f;
tile.gradient[0] = 0.1f;
/* ... */

/* Check: how many fields fall below 0.5? */
int violations = fleet_tile_check(&tile, 0.5f);
if (violations > 0) {
    printf("Tile has %d violations (impl: %s)\n",
           violations, fleet_math_impl_name());
}

/* Compute holonomy around a 4-edge cycle */
float weights[4] = {0.8f, 0.9f, 0.7f, 0.85f};
float H = fleet_holonomy_4cycle(weights);
```

### Batch Operations

```c
/* Check 1024 tiles at once */
plato_tile_t tiles[1024];
for (int i = 0; i < 1024; i++) tile_init(&tiles[i]);

int valid = fleet_batch_check(tiles, 1024, 0.5f);
printf("%d/%d tiles passed\n", valid, 1024);

/* Compute holonomy for 256 four-cycles */
float weights[256 * 4];  /* 4 weights per cycle */
float holonomy[256];
fleet_batch_holonomy(weights, 256, holonomy);
```

## API Reference

### Types

```c
typedef struct __attribute__((aligned(64))) {
    float confidence;      /* 0.0 – 1.0             */
    float novelty;         /* 0.0 – 10.0            */
    float gradient[4];     /* 4D gradient vector     */
    float metadata[8];     /* timestamps, source…    */
    uint8_t hash[8];       /* 64-bit tile hash       */
} plato_tile_t;
_Static_assert(sizeof(plato_tile_t) == 64);
```

The tile is exactly 64 bytes — no padding needed. The layout is designed so all 14 float fields occupy the first 56 bytes, making SIMD comparison straightforward (compare 16 floats, mask out the last 2 lanes that overlap with the hash).

```c
typedef struct {
    float     *weights;
    uint32_t  *src;
    uint32_t  *dst;
    uint32_t   n_edges;
    uint32_t   n_nodes;
} constraint_graph_t;
```

Sparse directed graph in Structure-of-Arrays (SoA) layout for SIMD-friendly traversal.

### Functions

| Function | Signature | Returns |
|----------|-----------|---------|
| `tile_init` | `void tile_init(plato_tile_t *tile)` | Fills tile with defaults (confidence=1.0, rest zero) |
| `tile_check_violations_*` | `int (const plato_tile_t *tile, float threshold)` | Count of fields below threshold (0 = valid) |
| `holonomy_4cycle_*` | `float (const float weights[4])` | Holonomy H = w₀w₁ − w₂w₃ |
| `batch_check_tiles_*` | `int (const plato_tile_t *tiles, int n, float threshold)` | Count of valid tiles in batch |
| `batch_holonomy_4cycles_*` | `void (const float *weights, int n, float *out)` | Holonomy for n cycles |
| `fleet_math_impl_name` | `const char *(void)` | Active backend name string |

The `fleet_*` macros auto-dispatch to the best available backend:

```c
#define fleet_tile_check(t, th)       tile_check_violations_avx512(t, th)   // or _neon, or _scalar
#define fleet_holonomy_4cycle(w)      holonomy_4cycle_avx512(w)
#define fleet_batch_check(t, n, th)   batch_check_tiles_avx512(t, n, th)
#define fleet_batch_holonomy(w, n, o) batch_holonomy_4cycles_avx512(w, n, o)
```

You can also call backend-specific functions directly if you need control.

## SIMD Internals

### AVX-512 Path (x86)

```c
/* One tile check — 14 float fields in ~8-10 clock cycles */
__m512 t    = _mm512_load_ps(&tile->confidence);       // load entire tile
__mmask16 m = _mm512_cmp_ps_mask(t, thresh, _CMP_LT_OQ);  // compare all 16 floats
m &= 0x3FFF;                                           // mask out hash bytes
return _mm_popcnt_u32(m);                              // count violations
```

One load, one compare, one mask, one popcount. The constraint check is a single instruction chain with no branch.

The batch holonomy processes 4 cycles per iteration using `_mm512_hsub_ps` to subtract adjacent product pairs in a single instruction.

### ARM NEON Path

```c
/* Four Q-register loads (4×16B = 64B), four comparisons, horizontal reduction */
float32x4_t thresh = vdupq_n_f32(threshold);
float32x4_t q0 = vld1q_f32(&tile->confidence);    // confidence, novelty, grad[0:1]
float32x4_t q1 = vld1q_f32(&tile->gradient[2]);   // grad[2:3], metadata[0:1]
// ... compare and reduce
```

### Scalar Path

Portable C99 fallback. Same semantics, no SIMD. Useful for:
- Debug builds
- Platforms without SIMD support
- Verifying SIMD results against a reference

## Build

```bash
# Build and run tests + benchmarks
make

# Just run tests
make check

# Just run benchmarks
make bench-run

# Clean
make clean

# Show active config
make info
```

The Makefile auto-detects architecture and adjusts flags:
- **x86_64**: `-mavx512f -mavx512dq` enabled
- **ARM64/aarch64**: AVX-512 disabled, NEON used
- **Apple Silicon** (`arm64`): Same as ARM64

### Manual Compilation

```bash
# With AVX-512
gcc -O3 -mavx512f -mavx512dq -ffast-math test.c fleet_math.c -lm -o test

# Scalar only (any platform)
gcc -O3 -DFLEET_MATH_NO_AUTOSELECT test.c fleet_math.c -lm -o test
```

## Performance

Benchmarks run on ARM64 Neoverse-N1 (GCC -O3 -march=native):

| Operation | Scalar | SIMD (NEON) | Notes |
|-----------|--------|-------------|-------|
| Tile check (per tile) | 4.9 ns | 5.1 ns | GCC auto-vectorizes scalar well on modern ARM |
| Holonomy (per 4-cycle) | 2.6 ns | 2.4 ns | |
| Batch 1024 tiles (per tile) | 3.9 ns | 5.0 ns | |

> **Why is SIMD not faster on ARM?** GCC -O3 with `-march=native` auto-vectorizes the scalar code extremely well on Neoverse-N1. The manual NEON intrinsics match but don't beat the compiler. Manual SIMD shows larger gains on older ARM cores (Cortex-A53, Cortex-A72) and on x86 with AVX-512 where the single-instruction path is genuinely faster than auto-vectorized loops.

On x86 with AVX-512, expect the SIMD path to be 2-4× faster than scalar for batch operations.

## Testing

```bash
make check     # runs test suite
```

The test suite validates:
- Tile initialization and field access
- Violation detection across all 14 float fields
- Holonomy computation (known-answer tests)
- Batch operations (correctness + edge cases)
- Scalar vs SIMD result agreement
- NaN and edge-value handling

## Files

| File | Purpose |
|------|---------|
| `fleet_math.h` | Complete library — types, scalar, AVX-512, NEON implementations (404 lines) |
| `fleet_math.c` | Runtime dispatch helper (`fleet_math_impl_name`) |
| `test.c` | Test suite |
| `bench.c` | Benchmark harness (scalar vs SIMD) |
| `simd_benchmark.c` | Standalone inlined benchmark (avoids function call overhead) |
| `Makefile` | Build system with arch auto-detection |

## Design Philosophy

**One tile = one instruction chain.** The entire constraint check for a PLATO tile should be a load, a compare, a mask, and a count. No loops, no branches, no memory traffic beyond the initial load. This is only possible because the tile was designed to be exactly 64 bytes.

**Compile-time dispatch, not runtime.** The preprocessor selects the best available backend when you compile. No CPUID checks, no indirect function calls, no dispatch overhead. The binary runs the code that was compiled for it.

**Scalar is the reference.** Every SIMD function has a scalar equivalent with identical semantics. Test against both. If they disagree, the SIMD is wrong.

## Related Projects

| Repo | Language | What |
|------|----------|------|
| [flux-engine-c](https://github.com/SuperInstance/flux-engine-c) | C | Single-header constraint engine — check, fracture, sediment |
| [plato-types](https://github.com/SuperInstance/plato-types) | Python | PLATO tile lifecycle and Lamport clocks |
| [holonomy-consensus](https://github.com/SuperInstance/holonomy-consensus) | Rust | GL(9) zero-holonomy consensus for constraint systems |
| [flux-check-js](https://github.com/SuperInstance/flux-check-js) | TypeScript | Full constraint engine with fracture + sediment |
| [flux-lib-py](https://github.com/SuperInstance/flux-lib-py) | Python | Unified constraint engine library |

## License

MIT
