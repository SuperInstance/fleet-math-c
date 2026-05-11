/*
 * fleet_math.c — Non-inline implementation (if any)
 *
 * Most functions are static inline in fleet_math.h.
 * This file currently provides only runtime dispatch helpers.
 */

#include "fleet_math.h"

/* Returns a human-readable string for the active SIMD implementation */
const char *fleet_math_impl_name(void)
{
    return FLEET_MATH_ACTIVE_IMPL;
}
