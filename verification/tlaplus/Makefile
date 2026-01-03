# Makefile for TLA+ verification of openpilot selfdrived state machine
#
# Usage:
#   make check        - Run TLC model checker (default)
#   make check-fast   - Run with safety spec only (faster, no liveness)
#   make download     - Download TLA+ tools
#   make clean        - Remove generated files
#   make help         - Show this help
#
# Requirements:
#   - Java 11+ (java command must be in PATH)
#   - Internet connection (for downloading TLA+ tools)

# Configuration
TLA_VERSION := 1.8.0
TLA_JAR := tla2tools.jar
TLA_URL := https://github.com/tlaplus/tlaplus/releases/download/v$(TLA_VERSION)/$(TLA_JAR)

# Spec files
SPEC := SelfDrived.tla
CFG := SelfDrived.cfg
SAFETY_CFG := SelfDrived-safety.cfg

# Java options
JAVA_OPTS := -XX:+UseParallelGC -Xmx4g

# TLC options
# -workers auto: Use all available CPU cores
# -deadlock: Check for deadlocks
# -cleanup: Remove state files after successful run
TLC_OPTS := -workers auto -deadlock -cleanup

.PHONY: all check check-fast download clean help

# Default target
all: check

# Download TLA+ tools if not present
$(TLA_JAR):
	@echo "Downloading TLA+ tools v$(TLA_VERSION)..."
	curl -fsSL -o $(TLA_JAR) $(TLA_URL)
	@echo "Downloaded $(TLA_JAR)"

download: $(TLA_JAR)

# Run full model check (safety + liveness with fairness)
check: $(TLA_JAR)
	@echo "Running TLC model checker..."
	@echo "Spec: $(SPEC)"
	@echo "Config: $(CFG)"
	@echo ""
	java $(JAVA_OPTS) -jar $(TLA_JAR) $(TLC_OPTS) -config $(CFG) $(SPEC)

# Run safety-only check (faster, no temporal properties)
check-fast: $(TLA_JAR) $(SAFETY_CFG)
	@echo "Running TLC safety check (no liveness)..."
	java $(JAVA_OPTS) -jar $(TLA_JAR) $(TLC_OPTS) -config $(SAFETY_CFG) $(SPEC)

# Generate safety-only config
$(SAFETY_CFG): $(CFG)
	@echo "Generating safety-only config..."
	@echo "\* Safety-only configuration (no temporal properties)" > $(SAFETY_CFG)
	@echo "" >> $(SAFETY_CFG)
	@echo "CONSTANTS" >> $(SAFETY_CFG)
	@echo "  SOFT_DISABLE_TIME = 5" >> $(SAFETY_CFG)
	@echo "" >> $(SAFETY_CFG)
	@echo "SPECIFICATION SafetySpec" >> $(SAFETY_CFG)
	@echo "" >> $(SAFETY_CFG)
	@echo "INVARIANTS" >> $(SAFETY_CFG)
	@echo "  TypeInvariant" >> $(SAFETY_CFG)
	@echo "  StateValid" >> $(SAFETY_CFG)
	@echo "  TimerBounded" >> $(SAFETY_CFG)

# Run with increased timer value (larger state space)
check-full: $(TLA_JAR)
	@echo "Running TLC with larger timer value (slower)..."
	@echo "Creating temporary config with SOFT_DISABLE_TIME = 10..."
	@sed 's/SOFT_DISABLE_TIME = 5/SOFT_DISABLE_TIME = 10/' $(CFG) > SelfDrived-full.cfg
	java $(JAVA_OPTS) -jar $(TLA_JAR) $(TLC_OPTS) -config SelfDrived-full.cfg $(SPEC)
	@rm -f SelfDrived-full.cfg

# Print state space statistics
stats: $(TLA_JAR)
	@echo "Generating state space statistics..."
	java $(JAVA_OPTS) -jar $(TLA_JAR) -config $(CFG) -dump dot,colorize states.dot $(SPEC) || true
	@if [ -f states.dot ]; then \
		echo "State graph written to states.dot"; \
		echo "Visualize with: dot -Tpng states.dot -o states.png"; \
	fi

# Clean generated files
clean:
	rm -f $(TLA_JAR)
	rm -f $(SAFETY_CFG)
	rm -f SelfDrived-full.cfg
	rm -f states.dot states.png
	rm -rf states/

# Show help
help:
	@echo "TLA+ Verification Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  check       - Run full TLC model check (default)"
	@echo "  check-fast  - Run safety check only (faster)"
	@echo "  check-full  - Run with larger timer (10, more thorough)"
	@echo "  download    - Download TLA+ tools"
	@echo "  stats       - Generate state space graph"
	@echo "  clean       - Remove generated files"
	@echo "  help        - Show this help"
	@echo ""
	@echo "Requirements:"
	@echo "  - Java 11+ runtime"
	@echo ""
	@echo "Example:"
	@echo "  make check"
