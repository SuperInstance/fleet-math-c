# Fleet Math C

SIMD-accelerated constraint operations for PLATO tile processing.

Three C files, no dependencies — drop them into any project that needs
ZHC emergence detection, Laman rigidity checks, or Pythagorean48 encoding.

## Functions

| Function | Operation | SIMD |
|----------|-----------|------|
| `tile_check_violations()` | Check tile fields below threshold | NEON (ARM) |
| `holonomy_4cycle()` | Compute holonomy around a 4-edge cycle | NEON (ARM) |
| `batch_check_tiles()` | Check N tiles at once | NEON batch |
| `batch_holonomy_4cycles()` | Compute N cycle holonomies at once | NEON batch |

## Performance (ARM64 Neoverse-N1)

| Operation | Scalar | SIMD (NEON) |
|-----------|--------|-------------|
| Tile check (per tile) | 4.9ns | 5.1ns |
| Holonomy (per 4-cycle) | 2.6ns | 2.4ns |
| Batch 1024 tiles (per tile) | 3.9ns | 5.0ns |

GCC -O3 -march=native auto-vectorizes well enough that manual SIMD matches
scalar on modern ARM64. Manual NEON intrinsics may show larger gains on
older cores (Cortex-A53, Cortex-A72).

## Use

```c
#include "fleet_math.h"
plato_tile_t tile;
int violations = tile_check_violations_neon(&tile, 0.5f);
```
