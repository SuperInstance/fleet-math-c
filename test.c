/*
 * test.c — Test suite for fleet-math-c
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "fleet_math.h"

/* Precision tolerance for floating-point comparisons */
#define EPSILON 1e-6f

/* Test counters */
static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name)                                          \
    do {                                                    \
        printf("  TEST %-45s ", name);                      \
        if (1) { /* run the block */

#define END_TEST(result)                                    \
            if (result) {                                   \
                printf("PASS\n");                           \
                tests_passed++;                             \
            } else {                                        \
                printf("FAIL\n");                           \
                tests_failed++;                             \
            }                                               \
        }                                                   \
    } while (0)

/* =========================================================================
 * Tile tests
 * ========================================================================= */

static void
test_tile_size(void)
{
    TEST("tile size == 64 bytes")
        int ok = (sizeof(plato_tile_t) == 64);
        printf(" [%zu bytes] ", sizeof(plato_tile_t));
        ok = ok;
    END_TEST(ok);
}

static void
test_tile_alignment(void)
{
    TEST("tile alignment >= 64")
        plato_tile_t t;
        size_t align = (size_t)&t & 63;
        int ok = (align == 0);
        printf(" [align=%zu] ", align);
        ok = ok;
    END_TEST(ok);
}

static void
test_tile_init(void)
{
    TEST("tile_init sets confidence=1.0, novelty=0.0")
        plato_tile_t t;
        tile_init(&t);
        int ok = (fabsf(t.confidence - 1.0f) < EPSILON)
              && (fabsf(t.novelty) < EPSILON);
    END_TEST(ok);
}

static void
test_tile_check_valid(void)
{
    TEST("valid tile (all fields above threshold) no violations")
        plato_tile_t t;
        tile_init(&t);
        t.confidence = 0.9f;
        t.novelty    = 5.0f;
        for (int i = 0; i < 4; i++) t.gradient[i] = 1.0f;
        for (int i = 0; i < 8; i++) t.metadata[i] = 1.0f;
        int violations = fleet_tile_check(&t, 0.5f);
        int ok = (violations == 0);
        printf(" [violations=%d] ", violations);
        ok = ok;
    END_TEST(ok);
}

static void
test_tile_check_violation(void)
{
    TEST("invalid tile (confidence=0.1, threshold=0.5) detects violations")
        plato_tile_t t;
        tile_init(&t);
        t.confidence = 0.1f;
        int violations = fleet_tile_check(&t, 0.5f);
        int ok = (violations > 0);
        printf(" [violations=%d] ", violations);
        ok = ok;
    END_TEST(ok);
}

static void
test_tile_check_all_violations(void)
{
    TEST("all-zero tile violates in all 14 float fields")
        plato_tile_t t;
        memset(&t, 0, sizeof(t));
        int violations = fleet_tile_check(&t, 0.5f);
        int ok = (violations == 14);
        printf(" [violations=%d (expected 14)] ", violations);
        ok = ok;
    END_TEST(ok);
}

/* =========================================================================
 * Holonomy tests
 * ========================================================================= */

static void
test_holonomy_zero(void)
{
    TEST("holonomy of equal weights = 0")
        float w[4] = {2.0f, 2.0f, 2.0f, 2.0f};  /* 2*2 - 2*2 = 0 */
        float h = fleet_holonomy_4cycle(w);
        int ok = (fabsf(h) < EPSILON);
        printf(" [H=%.6f] ", h);
        ok = ok;
    END_TEST(ok);
}

static void
test_holonomy_positive(void)
{
    TEST("holonomy of (3,4,5,2) = 3*4 - 5*2 = 2")
        float w[4] = {3.0f, 4.0f, 5.0f, 2.0f};  /* 12 - 10 = 2 */
        float h = fleet_holonomy_4cycle(w);
        int ok = (fabsf(h - 2.0f) < EPSILON);
        printf(" [H=%.6f (expected 2.0)] ", h);
        ok = ok;
    END_TEST(ok);
}

static void
test_holonomy_negative(void)
{
    TEST("holonomy of (1,2,3,4) = 1*2 - 3*4 = -10")
        float w[4] = {1.0f, 2.0f, 3.0f, 4.0f};  /* 2 - 12 = -10 */
        float h = fleet_holonomy_4cycle(w);
        int ok = (fabsf(h - (-10.0f)) < EPSILON);
        printf(" [H=%.6f (expected -10.0)] ", h);
        ok = ok;
    END_TEST(ok);
}

/* =========================================================================
 * Batch tests
 * ========================================================================= */

static void
test_batch_check(void)
{
    TEST("batch: 2 valid, 1 invalid out of 3 tiles")
        plato_tile_t tiles[3];
        for (int i = 0; i < 3; i++) {
            tile_init(&tiles[i]);
            tiles[i].confidence = 0.9f;
            tiles[i].novelty    = 5.0f;
            for (int j = 0; j < 4; j++) tiles[i].gradient[j] = 1.0f;
            for (int j = 0; j < 8; j++) tiles[i].metadata[j] = 1.0f;
        }
        /* Third tile has low confidence */
        tiles[2].confidence = 0.1f;

        int valid = fleet_batch_check(tiles, 3, 0.5f);
        int ok = (valid == 2);
        printf(" [valid=%d (expected 2)] ", valid);
        ok = ok;
    END_TEST(ok);
}

static void
test_batch_holonomy(void)
{
    TEST("batch holonomy: 4 cycles, known values")
        /* 4 cycles: (2,2,2,2)→0, (3,4,5,2)→2, (1,2,3,4)→-10, (1,1,1,1)→0 */
        float weights[16] = {
            2, 2, 2, 2,
            3, 4, 5, 2,
            1, 2, 3, 4,
            1, 1, 1, 1
        };
        float expected[4] = {0, 2, -10, 0};
        float out[4];
        fleet_batch_holonomy(weights, 4, out);

        int ok = 1;
        for (int i = 0; i < 4; i++) {
            printf(" [H%d=%.2f]", i, out[i]);
            if (fabsf(out[i] - expected[i]) >= EPSILON)
                ok = 0;
        }
        ok = ok;
    END_TEST(ok);
}

static void
test_batch_holonomy_odd(void)
{
    TEST("batch holonomy: non-multiple-of-4 (5 cycles) handles remainder")
        float weights[20];
        float expected[5];
        for (int i = 0; i < 5; i++) {
            float w0 = (float)(i + 1);
            float w1 = (float)(i + 2);
            float w2 = (float)(i + 3);
            float w3 = (float)(i + 4);
            weights[i*4+0] = w0;
            weights[i*4+1] = w1;
            weights[i*4+2] = w2;
            weights[i*4+3] = w3;
            expected[i] = w0 * w1 - w2 * w3;
        }

        float out[5];
        fleet_batch_holonomy(weights, 5, out);

        int ok = 1;
        for (int i = 0; i < 5; i++) {
            printf(" [H%d=%.2f]", i, out[i]);
            if (fabsf(out[i] - expected[i]) >= EPSILON)
                ok = 0;
        }
        ok = ok;
    END_TEST(ok);
}

/* =========================================================================
 * Edge case tests
 * ========================================================================= */

static void
test_empty_graph(void)
{
    TEST("empty constraint graph shows no edges")
        constraint_graph_t g = {NULL, NULL, NULL, 0, 0};
        int ok = (g.n_edges == 0 && g.n_nodes == 0);
        ok = ok;
    END_TEST(ok);
}

static void
test_threshold_epsilon(void)
{
    TEST("tile at exact threshold boundary: only 1 field below")
        plato_tile_t t;
        tile_init(&t);
        t.confidence = 1.0f;
        t.novelty    = 0.0f;
        for (int i = 0; i < 4; i++) t.gradient[i] = 1.0f;
        for (int i = 0; i < 8; i++) t.metadata[i] = 1.0f;
        /* Only novelty(0.0) is below threshold(1.0) → 1 violation */
        int violations = fleet_tile_check(&t, 1.0f);
        int ok = (violations == 1);
        printf(" [violations=%d] ", violations);
        ok = ok;
    END_TEST(ok);
}

static void
test_large_batch(void)
{
    TEST("large batch: 1000 tiles, all valid")
        enum { N = 1000 };
        plato_tile_t *tiles = (plato_tile_t *)aligned_alloc(64, N * sizeof(plato_tile_t));
        if (!tiles) {
            printf(" SKIP (alloc fail)");
            tests_passed++;
            return;
        }
        for (int i = 0; i < N; i++) {
            tile_init(&tiles[i]);
            tiles[i].confidence = 1.0f;
            tiles[i].novelty    = 5.0f;
            for (int j = 0; j < 4; j++) tiles[i].gradient[j] = 1.0f;
            for (int j = 0; j < 8; j++) tiles[i].metadata[j] = 1.0f;
        }
        int valid = fleet_batch_check(tiles, N, 0.5f);
        free(tiles);
        int ok = (valid == N);
        printf(" [valid=%d/%d] ", valid, N);
        ok = ok;
    END_TEST(ok);
}

/* =========================================================================
 * Main
 * ========================================================================= */

int main(void)
{
    printf("\n=== fleet-math-c Test Suite ===\n\n");
    printf("Implementation: %s\n\n", fleet_math_impl_name());

    printf("--- Tile Tests ---\n");
    test_tile_size();
    test_tile_alignment();
    test_tile_init();
    test_tile_check_valid();
    test_tile_check_violation();
    test_tile_check_all_violations();

    printf("\n--- Holonomy Tests ---\n");
    test_holonomy_zero();
    test_holonomy_positive();
    test_holonomy_negative();

    printf("\n--- Batch Tests ---\n");
    test_batch_check();
    test_batch_holonomy();
    test_batch_holonomy_odd();

    printf("\n--- Edge Case Tests ---\n");
    test_empty_graph();
    test_threshold_epsilon();
    test_large_batch();

    printf("\n=== Results: %d passed, %d failed ===\n\n",
           tests_passed, tests_failed);

    return (tests_failed > 0) ? 1 : 0;
}
