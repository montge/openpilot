# libFuzzer Makefile for openpilot safety code
#
# Usage:
#   make all        - Build all fuzz targets
#   make run        - Run quick fuzzing session (10 seconds each)
#   make corpus     - Run with corpus building (longer)
#   make clean      - Remove build artifacts
#
# Requirements:
#   - Clang with libFuzzer support (clang-14 or later)
#   - ASAN/UBSAN runtime

CC := clang
CFLAGS := -g -O1 -fno-omit-frame-pointer
FUZZ_FLAGS := -fsanitize=fuzzer,address,undefined
INCLUDES := -I../../opendbc_repo

# Safety code location
SAFETY_DIR := ../../opendbc_repo/opendbc/safety

TARGETS := fuzz_safety_tx_hook fuzz_can_rx

.PHONY: all run corpus clean

all: $(TARGETS)

fuzz_safety_tx_hook: fuzz_safety_tx_hook.c stubs.h
	$(CC) $(CFLAGS) $(FUZZ_FLAGS) $(INCLUDES) -o $@ $<

fuzz_can_rx: fuzz_can_rx.c stubs.h
	$(CC) $(CFLAGS) $(FUZZ_FLAGS) $(INCLUDES) -o $@ $<

run: all
	@echo "Running quick fuzz sessions (10s each)..."
	@mkdir -p corpus_tx corpus_rx
	@for target in $(TARGETS); do \
		echo "\n=== Fuzzing $$target ==="; \
		timeout 10s ./$$target corpus_$${target#fuzz_} -max_total_time=10 2>&1 || true; \
	done
	@echo "\n=== Fuzzing complete ==="

corpus: all
	@echo "Building corpus (60s each)..."
	@mkdir -p corpus_tx corpus_rx
	@for target in $(TARGETS); do \
		echo "\n=== Building corpus for $$target ==="; \
		./$$target corpus_$${target#fuzz_} -max_total_time=60 2>&1 || true; \
	done

clean:
	rm -f $(TARGETS)
	rm -rf corpus_* crash-* *.log
