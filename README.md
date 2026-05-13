# Fleet Math C

SIMD-accelerated constraint operations. 3 C files. ZHC emergence, Laman rigidity, Pythagorean48 encoding.

## Performance

Tested on ARM64 Neoverse-N1 (Oracle Cloud) with GCC -O3 -march=native:

| Operation | Scalar | SIMD (NEON) | Speedup |
|-----------|--------|-------------|---------|
| Tile check (per tile) | 4.9 ns | 5.1 ns | 1.0x |
| Holonomy (per 4-cycle) | 2.6 ns | 2.4 ns | 1.1x |
| Batch 1024 tiles (per tile) | 3.9 ns | 5.1 ns | 0.8x |
| Batch 1024 holonomy (per cycle) | auto-vec | 0.6 ns | — |

On modern ARM64 with aggressive compiler optimization, GCC's auto-vectorization at `-O3 -march=native` leaves little room for manual SIMD improvement. The scalar and NEON implementations are within measurement noise of each other.

Manual SIMD may show larger gains on older ARM cores (Cortex-A53, Cortex-A72) or when cross-compiling without `-march=native`.

No dependencies. Three files.

## License

Apache 2.0 — Cocapn fleet infrastructure.
