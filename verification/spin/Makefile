# Makefile for SPIN Protocol Verification
#
# Usage:
#   make verify      - Run full verification (safety + liveness)
#   make safety      - Run safety property verification only
#   make simulate    - Run random simulation
#   make trail       - Display counterexample trail (after finding error)
#   make clean       - Remove generated files
#
# Prerequisites:
#   - SPIN model checker: apt-get install spin
#   - GCC compiler for pan.c verifier

MODEL = msgq_protocol.pml
PAN = pan
SPIN = spin

# Verification options
# -a: generate verifier for all properties
# -m: max search depth (states)
# -w: hash table size (2^w entries)
# -DSAFETY: safety properties only (no liveness)
# -DNOREDUCE: disable partial order reduction (for debugging)
# -DCOLLAPSE: state compression
# -DVECTORSZ=4096: increase vector size for larger models

# Default max depth for bounded verification
MAX_DEPTH ?= 100000
HASH_SIZE ?= 24

.PHONY: all verify safety liveness simulate trail clean help

all: verify

# Generate the verifier from Promela model
$(PAN).c: $(MODEL)
	$(SPIN) -a $(MODEL)

# Compile verifier with optimizations
$(PAN): $(PAN).c
	gcc -O2 -DVECTORSZ=4096 -o $(PAN) $(PAN).c

# Compile verifier for safety checks only (faster)
$(PAN)_safety: $(PAN).c
	gcc -O2 -DSAFETY -DVECTORSZ=4096 -o $(PAN)_safety $(PAN).c

# Full verification: safety + liveness properties
verify: $(PAN)
	@echo "=========================================="
	@echo "Running full verification (safety + LTL)"
	@echo "Max depth: $(MAX_DEPTH) states"
	@echo "=========================================="
	./$(PAN) -a -m$(MAX_DEPTH) -w$(HASH_SIZE)
	@echo ""
	@echo "Verification complete. Check output above for errors."

# Safety properties only (no liveness/progress)
safety: $(PAN)_safety
	@echo "=========================================="
	@echo "Running safety verification"
	@echo "=========================================="
	./$(PAN)_safety -m$(MAX_DEPTH) -w$(HASH_SIZE)

# Verify specific LTL property
# Usage: make ltl PROP=publisher_progress
ltl: $(PAN).c
	@if [ -z "$(PROP)" ]; then \
		echo "Usage: make ltl PROP=<property_name>"; \
		echo "Available properties:"; \
		grep "^ltl " $(MODEL) | sed 's/ltl \([a-z_]*\).*/  - \1/'; \
		exit 1; \
	fi
	gcc -O2 -DVECTORSZ=4096 -o $(PAN)_ltl $(PAN).c
	./$(PAN)_ltl -a -N $(PROP) -m$(MAX_DEPTH) -w$(HASH_SIZE)

# Random simulation (useful for debugging)
simulate:
	@echo "=========================================="
	@echo "Running random simulation"
	@echo "=========================================="
	$(SPIN) -p -l -g $(MODEL)

# Guided simulation with seed
# Usage: make simulate-seed SEED=12345
simulate-seed:
	@if [ -z "$(SEED)" ]; then \
		echo "Usage: make simulate-seed SEED=<number>"; \
		exit 1; \
	fi
	$(SPIN) -p -l -g -n$(SEED) $(MODEL)

# Interactive simulation
interactive:
	$(SPIN) -i $(MODEL)

# Display counterexample trail (after verification finds an error)
trail: $(MODEL).trail
	@echo "=========================================="
	@echo "Displaying counterexample trail"
	@echo "=========================================="
	$(SPIN) -t -p -l -g $(MODEL)

# Pretty-print the trail with message sequence chart
trail-msc: $(MODEL).trail
	$(SPIN) -t -M $(MODEL)

# Check syntax only
syntax:
	$(SPIN) -a $(MODEL) 2>&1 | head -20

# Show statistics about the model
stats: $(PAN)
	./$(PAN) -d

# Exhaustive verification with no reductions (for debugging)
exhaustive: $(PAN).c
	gcc -O2 -DNOREDUCE -DVECTORSZ=4096 -o $(PAN)_exh $(PAN).c
	./$(PAN)_exh -a -m$(MAX_DEPTH)

# Memory-efficient verification (state compression)
compressed: $(PAN).c
	gcc -O2 -DCOLLAPSE -DVECTORSZ=4096 -o $(PAN)_cmp $(PAN).c
	./$(PAN)_cmp -a -m$(MAX_DEPTH) -w$(HASH_SIZE)

# Bitstate hashing (approximate, for very large state spaces)
bitstate: $(PAN).c
	gcc -O2 -DBITSTATE -DVECTORSZ=4096 -o $(PAN)_bit $(PAN).c
	./$(PAN)_bit -a -m$(MAX_DEPTH)

# Clean up generated files
clean:
	rm -f $(PAN) $(PAN)_* pan.* *.trail _spin_nvr.tmp
	rm -f $(MODEL).trail

# Help
help:
	@echo "SPIN Protocol Verification Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  verify      - Run full verification (safety + liveness)"
	@echo "  safety      - Run safety properties only (faster)"
	@echo "  ltl         - Verify specific LTL property (PROP=name)"
	@echo "  simulate    - Run random simulation"
	@echo "  interactive - Interactive step-by-step simulation"
	@echo "  trail       - Display counterexample after finding error"
	@echo "  stats       - Show model statistics"
	@echo "  clean       - Remove generated files"
	@echo ""
	@echo "Options (environment variables):"
	@echo "  MAX_DEPTH   - Maximum search depth (default: 100000)"
	@echo "  HASH_SIZE   - Hash table size as 2^n (default: 24)"
	@echo ""
	@echo "Examples:"
	@echo "  make verify MAX_DEPTH=50000"
	@echo "  make ltl PROP=publisher_progress"
	@echo "  make simulate-seed SEED=42"
