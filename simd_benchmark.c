/*
 * simd_benchmark.c — Proper SIMD benchmark with inlining and correct loops
 *
 * This avoids the function-call overhead that was killing the original bench.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <math.h>
#include "fleet_math.h"

/* Use scalar paths directly to avoid macro dispatch issues */
#define CHECK_SCALAR tile_check_violations_scalar
#define CHECK_SIMD   tile_check_violations_neon
#define HOLO_SCALAR  holonomy_4cycle_scalar
#define HOLO_SIMD    holonomy_4cycle_neon

static inline double time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

#define ARRAY_SIZE(arr) (int)(sizeof(arr)/sizeof((arr)[0]))

int main(void)
{
    printf("\n=== Fleet Math SIMD Benchmark (Proper Inlined) ===\n\n");
    printf("Implementation: %s\n", fleet_math_impl_name());

    /* Tile data */
    const int N_TILES = 1024;
    plato_tile_t *tiles = (plato_tile_t *)aligned_alloc(64, N_TILES * sizeof(plato_tile_t));
    if (!tiles) { fprintf(stderr, "alloc fail\n"); return 1; }

    /* Init tiles with varied data */
    srand(42);
    for (int i = 0; i < N_TILES; i++) {
        tile_init(&tiles[i]);
        tiles[i].confidence = (float)(rand() % 100) / 100.0f;
        tiles[i].novelty    = (float)(rand() % 1000) / 100.0f;
        for (int j = 0; j < 4; j++) tiles[i].gradient[j]  = (float)(rand() % 100) / 100.0f;
        for (int j = 0; j < 8; j++) tiles[i].metadata[j]  = (float)(rand() % 100) / 100.0f;
    }

    /* Weight data for holonomy */
    const int N_CYCLES = 1024;
    float *weights = (float *)aligned_alloc(64, N_CYCLES * 4 * sizeof(float));
    if (!weights) { fprintf(stderr, "alloc fail\n"); free(tiles); return 1; }
    for (int i = 0; i < N_CYCLES * 4; i++)
        weights[i] = (float)(rand() % 1000) / 100.0f;

    float threshold = 0.5f;
    const int REPS = 10000;
    const int EPOCHS = 5;

    /* ── Tile check: scalar ── */
    double best_scalar_tile = 1e100;
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        double t0 = time_ns();
        volatile int sink = 0;
        for (int rep = 0; rep < REPS; rep++) {
            for (int i = 0; i < N_TILES; i++)
                sink += CHECK_SCALAR(&tiles[i], threshold);
        }
        double t1 = time_ns();
        double ns_per_op = (t1 - t0) / (REPS * N_TILES);
        if (ns_per_op < best_scalar_tile) best_scalar_tile = ns_per_op;
        (void)sink;
    }

    /* ── Tile check: SIMD ── */
    double best_simd_tile = 1e100;
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        double t0 = time_ns();
        volatile int sink = 0;
        for (int rep = 0; rep < REPS; rep++) {
            for (int i = 0; i < N_TILES; i++)
                sink += CHECK_SIMD(&tiles[i], threshold);
        }
        double t1 = time_ns();
        double ns_per_op = (t1 - t0) / (REPS * N_TILES);
        if (ns_per_op < best_simd_tile) best_simd_tile = ns_per_op;
        (void)sink;
    }

    /* ── Holonomy: scalar ── */
    double best_scalar_holo = 1e100;
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        double t0 = time_ns();
        volatile float sink = 0;
        for (int rep = 0; rep < REPS; rep++) {
            for (int i = 0; i < N_CYCLES; i++)
                sink += HOLO_SCALAR(&weights[i * 4]);
        }
        double t1 = time_ns();
        double ns_per_op = (t1 - t0) / (REPS * N_CYCLES);
        if (ns_per_op < best_scalar_holo) best_scalar_holo = ns_per_op;
        (void)sink;
    }

    /* ── Holonomy: SIMD ── */
    double best_simd_holo = 1e100;
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        double t0 = time_ns();
        volatile float sink = 0;
        for (int rep = 0; rep < REPS; rep++) {
            for (int i = 0; i < N_CYCLES; i++)
                sink += HOLO_SIMD(&weights[i * 4]);
        }
        double t1 = time_ns();
        double ns_per_op = (t1 - t0) / (REPS * N_CYCLES);
        if (ns_per_op < best_simd_holo) best_simd_holo = ns_per_op;
        (void)sink;
    }

    /* ── Batch tile check (all 1024 at once): scalar ── */
    double best_scalar_batch = 1e100;
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        double t0 = time_ns();
        volatile int sink = 0;
        for (int rep = 0; rep < REPS; rep++) {
            sink += batch_check_tiles_scalar(tiles, N_TILES, threshold);
        }
        double t1 = time_ns();
        double ns_per_op = (t1 - t0) / (REPS * N_TILES);
        if (ns_per_op < best_scalar_batch) best_scalar_batch = ns_per_op;
        (void)sink;
    }

    /* ── Batch tile check (all 1024 at once): SIMD ── */
    double best_simd_batch = 1e100;
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        double t0 = time_ns();
        volatile int sink = 0;
        for (int rep = 0; rep < REPS; rep++) {
            sink += batch_check_tiles_neon(tiles, N_TILES, threshold);
        }
        double t1 = time_ns();
        double ns_per_op = (t1 - t0) / (REPS * N_TILES);
        if (ns_per_op < best_simd_batch) best_simd_batch = ns_per_op;
        (void)sink;
    }

    /* ── Batch holonomy (all 1024 at once): scalar ── */
    float *out_holo = (float *)aligned_alloc(64, N_CYCLES * sizeof(float));
    double best_scalar_holo_batch = 1e100;
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        double t0 = time_ns();
        for (int rep = 0; rep < REPS; rep++) {
            batch_holonomy_4cycles_scalar(weights, N_CYCLES, out_holo);
        }
        double t1 = time_ns();
        double ns_per_op = (t1 - t0) / (REPS * N_CYCLES);
        if (ns_per_op < best_scalar_holo_batch) best_scalar_holo_batch = ns_per_op;
    }

    /* ── Batch holonomy (all 1024 at once): SIMD ── */
    double best_simd_holo_batch = 1e100;
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        double t0 = time_ns();
        for (int rep = 0; rep < REPS; rep++) {
            batch_holonomy_4cycles_neon(weights, N_CYCLES, out_holo);
        }
        double t1 = time_ns();
        double ns_per_op = (t1 - t0) / (REPS * N_CYCLES);
        if (ns_per_op < best_simd_holo_batch) best_simd_holo_batch = ns_per_op;
    }

    printf("\n%30s | %9s | %9s | %s\n", "Operation", "Scalar", "SIMD", "Speedup");
    printf("%30s-+-%9s-+-%9s-+-%s\n", "-----------------------------", "---------", "---------", "-------");
    printf("%30s | %7.1f ns | %7.1f ns | %5.1fx\n", "Tile check (per tile)", best_scalar_tile, best_simd_tile, best_scalar_tile / best_simd_tile);
    printf("%30s | %7.1f ns | %7.1f ns | %5.1fx\n", "Holonomy (per cycle)", best_scalar_holo, best_simd_holo, best_scalar_holo / best_simd_holo);
    printf("%30s | %7.1f ns | %7.1f ns | %5.1fx\n", "Batch 1024 tiles (per tile)", best_scalar_batch, best_simd_batch, best_scalar_batch / best_simd_batch);
    printf("%30s | %7.1f ns | %7.1f ns | %5.1fx\n", "Batch 1024 holonomy (per cycle)", best_scalar_holo_batch, best_simd_holo_batch, best_scalar_holo_batch / best_simd_holo_batch);

    /* Correctness check */
    printf("\n--- Correctness (first 5 holonomy values) ---\n");
    batch_holonomy_4cycles_neon(weights, 5, out_holo);
    for (int i = 0; i < 5; i++) {
        float h_scalar;
        holonomy_4cycle_scalar(&weights[i*4]);
        /* recompute scalar */
        float w0=weights[i*4], w1=weights[i*4+1], w2=weights[i*4+2], w3=weights[i*4+3];
        float expected = w0*w1 - w2*w3;
        printf("  cycle[%d]: w=(%.2f,%.2f,%.2f,%.2f) scalar=%.4f simd=%.4f %s\n",
               i, w0, w1, w2, w3, expected, out_holo[i],
               (fabsf(expected - out_holo[i]) < 1e-6f) ? "OK" : "MISMATCH");
    }

    free(tiles);
    free(weights);
    free(out_holo);
    return 0;
}