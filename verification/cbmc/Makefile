# CBMC Verification Makefile
#
# Usage:
#   make all          - Run all verifications
#   make controls     - Verify controls_allowed property
#   make relay        - Verify relay_malfunction property
#   make torque       - Verify torque bounds property
#   make longitudinal - Verify longitudinal accel property
#   make clean        - Remove generated files

# Paths - include from opendbc_repo to support "opendbc/safety/..." includes
OPENDBC_REPO := ../../opendbc_repo
INCLUDE_DIRS := -I$(OPENDBC_REPO) -I.

# CBMC options
CBMC := cbmc
CBMC_FLAGS := \
	--bounds-check \
	--pointer-check \
	--signed-overflow-check \
	--unsigned-overflow-check \
	--conversion-check \
	--unwind 10 \
	--unwinding-assertions

# Harness files
HARNESSES := \
	harness_controls_allowed.c \
	harness_relay_malfunction.c \
	harness_torque_bounds.c \
	harness_longitudinal.c

.PHONY: all controls relay torque longitudinal clean check-cbmc

all: check-cbmc controls relay torque longitudinal
	@echo "All CBMC verifications passed!"

check-cbmc:
	@which $(CBMC) > /dev/null || (echo "Error: CBMC not found. Install with: sudo apt-get install cbmc" && exit 1)

controls: check-cbmc harness_controls_allowed.c
	@echo "=== Verifying controls_allowed property ==="
	$(CBMC) $(CBMC_FLAGS) $(INCLUDE_DIRS) harness_controls_allowed.c
	@echo "PASSED: controls_allowed verification"

relay: check-cbmc harness_relay_malfunction.c
	@echo "=== Verifying relay_malfunction property ==="
	$(CBMC) $(CBMC_FLAGS) $(INCLUDE_DIRS) harness_relay_malfunction.c
	@echo "PASSED: relay_malfunction verification"

torque: check-cbmc harness_torque_bounds.c
	@echo "=== Verifying torque bounds property ==="
	$(CBMC) $(CBMC_FLAGS) $(INCLUDE_DIRS) harness_torque_bounds.c
	@echo "PASSED: torque bounds verification"

longitudinal: check-cbmc harness_longitudinal.c
	@echo "=== Verifying longitudinal accel property ==="
	$(CBMC) $(CBMC_FLAGS) $(INCLUDE_DIRS) harness_longitudinal.c
	@echo "PASSED: longitudinal accel verification"

# Quick check - skip expensive checks for fast iteration
quick: check-cbmc
	@echo "=== Quick verification (no overflow checks) ==="
	$(CBMC) --unwind 5 $(INCLUDE_DIRS) harness_controls_allowed.c
	$(CBMC) --unwind 5 $(INCLUDE_DIRS) harness_relay_malfunction.c
	@echo "Quick check passed"

clean:
	rm -f *.goto *.out

# Help target
help:
	@echo "CBMC Safety Verification Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  all          - Run all verifications"
	@echo "  controls     - Verify controls_allowed blocks torque"
	@echo "  relay        - Verify relay_malfunction blocks TX"
	@echo "  torque       - Verify torque bounds enforcement"
	@echo "  longitudinal - Verify longitudinal accel bounds"
	@echo "  quick        - Fast verification for development"
	@echo "  clean        - Remove generated files"
	@echo ""
	@echo "Requirements:"
	@echo "  Install CBMC: sudo apt-get install cbmc"
