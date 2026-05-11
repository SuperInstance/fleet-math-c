/*
 * bench.c — Benchmark fleet-math SIMD vs scalar implementations
 *
 * Uses nanosecond-resolution monotonic clock for accurate measurements
 * across x86 and ARM platforms.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "fleet_math.h"

/* =========================================================================
 * Nanosecond timer (portable across all platforms)
 * ========================================================================= */

static inline double
time_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

/* =========================================================================
 * Benchmark runner
 * ========================================================================= */

typedef struct {
    double ns_per_op;   /* nanoseconds per single operation */
} bench_result_t;

/* Benchmark one operation. Reports ns/op averaged over n_reps calls. */
static bench_result_t
bench(const char *label,
      void (*scalar_fn)(void *ctx, int n_reps),
      void (*simd_fn)(void *ctx, int n_reps),
      void *ctx, int n_reps, int n_epochs)
{
    bench_result_t best_scalar = { .ns_per_op = 1e100 };
    bench_result_t best_simd   = { .ns_per_op = 1e100 };

    /* Warmup */
    scalar_fn(ctx, n_reps / 10);
    simd_fn(ctx, n_reps / 10);

    for (int epoch = 0; epoch < n_epochs; epoch++) {
        double t0, t1;

        /* Scalar */
        t0 = time_ns();
        scalar_fn(ctx, n_reps);
        t1 = time_ns();
        double sc_ns = (t1 - t0) / n_reps;
        if (sc_ns < best_scalar.ns_per_op)
            best_scalar.ns_per_op = sc_ns;

        /* SIMD */
        t0 = time_ns();
        simd_fn(ctx, n_reps);
        t1 = time_ns();
        double si_ns = (t1 - t0) / n_reps;
        if (si_ns < best_simd.ns_per_op)
            best_simd.ns_per_op = si_ns;
    }

    double speedup = best_simd.ns_per_op > 0
                   ? best_scalar.ns_per_op / best_simd.ns_per_op
                   : 0;

    printf("%-30s | %7.1f ns | %7.1f ns | %5.1fx\n",
           label,
           best_scalar.ns_per_op,
           best_simd.ns_per_op,
           speedup);

    return best_simd;
}

/* =========================================================================
 * Bench: per-tile check
 * ========================================================================= */

struct tile_ctx {
    plato_tile_t *tiles;
    int n;
    float threshold;
};

static void
check_scalar(void *ctx_, int reps)
{
    struct tile_ctx *c = (struct tile_ctx *)ctx_;
    volatile int sink = 0;
    for (int r = 0; r < reps; r++)
        for (int i = 0; i < c->n; i++)
            sink += tile_check_violations_scalar(&c->tiles[i], c->threshold);
    (void)sink;
}

static void
check_simd(void *ctx_, int reps)
{
    struct tile_ctx *c = (struct tile_ctx *)ctx_;
    volatile int sink = 0;
    for (int r = 0; r < reps; r++)
        for (int i = 0; i < c->n; i++)
            sink += fleet_tile_check(&c->tiles[i], c->threshold);
    (void)sink;
}

/* =========================================================================
 * Bench: per-cycle holonomy
 * ========================================================================= */

struct holo_ctx {
    float *weights;
    int n;
};

static void
holo_scalar(void *ctx_, int reps)
{
    struct holo_ctx *c = (struct holo_ctx *)ctx_;
    volatile float sink = 0;
    for (int r = 0; r < reps; r++)
        for (int i = 0; i < c->n; i++)
            sink += holonomy_4cycle_scalar(&c->weights[i * 4]);
    (void)sink;
}

static void
holo_simd(void *ctx_, int reps)
{
    struct holo_ctx *c = (struct holo_ctx *)ctx_;
    volatile float sink = 0;
    for (int r = 0; r < reps; r++)
        for (int i = 0; i < c->n; i++)
            sink += fleet_holonomy_4cycle(&c->weights[i * 4]);
    (void)sink;
}

/* =========================================================================
 * Bench: batch tile check (1024 tiles)
 * ========================================================================= */

struct batch_ctx {
    plato_tile_t *tiles;
    int n;
    float threshold;
};

static void
batch_scalar(void *ctx_, int reps)
{
    struct batch_ctx *c = (struct batch_ctx *)ctx_;
    volatile int sink = 0;
    for (int r = 0; r < reps; r++)
        sink += batch_check_tiles_scalar(c->tiles, c->n, c->threshold);
    (void)sink;
}

static void
batch_simd(void *ctx_, int reps)
{
    struct batch_ctx *c = (struct batch_ctx *)ctx_;
    volatile int sink = 0;
    for (int r = 0; r < reps; r++)
        sink += fleet_batch_check(c->tiles, c->n, c->threshold);
    (void)sink;
}

/* =========================================================================
 * Main
 * ========================================================================= */

int main(void)
{
    printf("\n=== Fleet Math SIMD Benchmarks ===\n\n");
    printf("Implementation: %s\n\n", fleet_math_impl_name());

    /* Constants: larger reps for short ops, smaller for long ops */
    enum {
        N_CHECK  = 10000,   /* tiles to check per rep */
        N_HOLO   = 10000,   /* 4-cycles per rep */
        N_BATCH  = 1024,    /* tiles in batch op */
        EPOCHS   = 5
    };

    /* Allocate tile data */
    plato_tile_t *tiles = (plato_tile_t *)aligned_alloc(64,
                              (N_CHECK > N_BATCH ? N_CHECK : N_BATCH)
                              * sizeof(plato_tile_t));

    float *weights = (float *)aligned_alloc(64, N_HOLO * 4 * sizeof(float));

    if (!tiles || !weights) {
        fprintf(stderr, "ERROR: allocation failed\n");
        free(tiles);
        free(weights);
        return 1;
    }

    /* Initialize semi-random data */
    srand(42);
    for (int i = 0; i < (N_CHECK > N_BATCH ? N_CHECK : N_BATCH); i++) {
        tile_init(&tiles[i]);
        tiles[i].confidence = (float)(rand() % 100) / 100.0f;
        tiles[i].novelty    = (float)(rand() % 1000) / 100.0f;
    }
    for (int i = 0; i < N_HOLO * 4; i++)
        weights[i] = (float)(rand() % 1000) / 100.0f;

    /* Run benchmarks */
    printf("%-30s | %-9s | %-9s | %s\n", "Operation", "Scalar", "SIMD", "Speedup");
    printf("%-30s-+-%-9s-+-%-9s-+-%s\n",
           "-------------------------------",
           "---------", "---------", "-------");

    struct tile_ctx  tc  = { tiles, N_CHECK, 0.5f };
    struct holo_ctx  hc  = { weights, N_HOLO };
    struct batch_ctx bc  = { tiles, N_BATCH, 0.5f };

    bench("Tile check (1 tile)",    check_scalar, check_simd, &tc,  N_CHECK, EPOCHS);
    bench("Holonomy (1 cycle)",     holo_scalar,  holo_simd,  &hc,  N_HOLO,  EPOCHS);
    bench("Batch 1024 tiles",       batch_scalar, batch_simd, &bc,  1,       EPOCHS);

    printf("\n");
    printf("* Tile check: per-tile cost averaged over %d ops × %d epochs\n",
           N_CHECK, EPOCHS);
    printf("* Holonomy:  per-cycle cost averaged over %d ops × %d epochs\n",
           N_HOLO, EPOCHS);
    printf("* Batch:     total cost for %d tiles\n", N_BATCH);
    printf("\n=== Benchmark complete ===\n\n");

    free(tiles);
    free(weights);
    return 0;
}
