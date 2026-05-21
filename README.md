# fleet-math-c

C-compatible math library for fleet operations. Eisenstein integers, Laman rigidity, holonomy checks, Manhattan distance, and Pythagorean-48 encoding — all exposed as `extern "C"` functions callable from any language.

Built in Rust. Zero dependencies. `rustc 1.70+`.

## What's in here

| Function | What it does |
|---|---|
| `fleet_eisenstein_norm(a, b)` | Eisenstein integer norm: a² − ab + b² |
| `fleet_laman_edges(vertices)` | Minimum edges for Laman rigidity: 2V − 3 |
| `fleet_is_rigid(vertices, edges)` | Check if E ≥ 2V − 3 |
| `fleet_holonomy_check(transforms, len)` | Verify product of transforms equals identity |
| `fleet_manhattan_distance(a, b, len)` | L1 distance between two integer vectors |
| `fleet_pythagorean48_encode(x, y)` | Quantize angle to 48-direction encoding (0–47) |

The crate also includes an Eisenstein lattice snapper (`snap`, `batch_snap`) that searches the 9 nearest Eisenstein integers and returns the best fit.

## Building

```sh
cargo build --release
```

That gives you `libfleet_math_c.a` (static) and `libfleet_math_c.so` (dynamic) in `target/release/`.

## Using from C

Include the header:

```c
#include "include/fleet_math.h"

int main(void) {
    // Eisenstein norm of (2, 3)
    int64_t n = fleet_eisenstein_norm(2, 3);  // 7

    // Is a 4-vertex, 6-edge graph rigid?
    bool rigid = fleet_is_rigid(4, 6);  // true (need ≥ 5)

    // Manhattan distance between vectors
    int32_t a[] = {0, 0, 0};
    int32_t b[] = {1, 2, 3};
    int64_t d = fleet_manhattan_distance(a, b, 3);  // 6

    // Direction encoding
    int32_t dir = fleet_pythagorean48_encode(0.0, 1.0);  // 12 (90°)

    return 0;
}
```

Compile and link:

```sh
gcc -o myapp myapp.c -L target/release -lfleet_math_c -lm -lpthread -ldl
```

## Running tests

```sh
cargo test
```

22 tests covering edge cases, symmetry, degenerate inputs, and the full API.

## The math

**Eisenstein integers** live on the hexagonal lattice Z[ω] where ω = e^(2πi/3). The norm N(a + bω) = a² − ab + b² measures squared distance from the origin. It's always ≥ 0 and equals 1 for the six Eisenstein units: (1,0), (0,1), (1,1), (−1,0), (0,−1), (−1,−1).

**Laman rigidity** characterizes generically rigid bar-and-joint frameworks in 2D. A graph with V vertices needs at least 2V − 3 edges to be rigid. Triangle (3V, 3E) is the minimal rigid graph.

**Holonomy** checks that a sequence of transforms composes back to the identity — useful for verifying closed-loop consistency in fleet coordinate transforms.

**Manhattan distance** (L1 norm) is the sum of absolute differences across dimensions. Cheap to compute, useful for vector similarity search and grid navigation.

**Pythagorean-48** divides the full circle into 48 sectors of 7.5° each. Direction 0 is the positive x-axis, direction 12 is straight up, direction 24 is the negative x-axis. Useful for coarse directional encoding in navigation.

## License

MIT
