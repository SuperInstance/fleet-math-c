CC       = gcc
CFLAGS   = -O3 -ffast-math -march=native
LDFLAGS  = -lm
WARN     = -Wall -Wextra -Wpedantic -Wstrict-prototypes
ARCH     = $(shell uname -m)

# Enable AVX-512 only when explicitly requested (CI runners lack it)
ifeq ($(ENABLE_AVX512), 1)
    CFLAGS += -mavx512f -mavx512dq -DFLEET_MATH_ENABLE_AVX512
endif

# On ARM, drop x86 flags
ifeq ($(ARCH), aarch64)
    CFLAGS = -O3 -ffast-math -march=native -DFLEET_MATH_ENABLE_AVX512=0
endif

# On Apple Silicon, use native arch
ifeq ($(ARCH), arm64)
    CFLAGS = -O3 -ffast-math -march=native -DFLEET_MATH_ENABLE_AVX512=0
endif

OBJS = fleet_math.o

.PHONY: all test bench clean

all: test bench

fleet_math.o: fleet_math.c fleet_math.h
	$(CC) $(CFLAGS) $(WARN) -c -o $@ $<

test: test.c fleet_math.h fleet_math.o
	$(CC) $(CFLAGS) $(WARN) -o $@ test.c fleet_math.o $(LDFLAGS)

bench: bench.c fleet_math.h fleet_math.o
	$(CC) $(CFLAGS) $(WARN) -o $@ bench.c fleet_math.o $(LDFLAGS)

check: test
	./test

bench-run: bench
	./bench

# Build as standalone unit (single translation unit, no .c needed)
standalone:
	$(CC) $(CFLAGS) -DFLEET_MATH_STANDALONE -o test_standalone test.c $(LDFLAGS)
	$(CC) $(CFLAGS) -DFLEET_MATH_STANDALONE -o bench_standalone bench.c $(LDFLAGS)
	@echo "Standalone: test_standalone and bench_standalone ready"
	./test_standalone

clean:
	rm -f test bench test_standalone bench_standalone *.o

# Print active implementation
info:
	@echo "Architecture:  $(ARCH)"
	@echo "Compiler:      $(CC)"
	@echo "CFLAGS:        $(CFLAGS)"
	@./test 2>/dev/null | head -2 || echo "Run 'make check' first"
