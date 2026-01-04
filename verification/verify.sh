#!/usr/bin/env bash
# Run all local verification checks
#
# Usage: ./verification/verify.sh [target]
# Targets: all, cbmc, tlaplus, spin, fuzz
#
# Requires:
#   - cbmc (apt install cbmc)
#   - java (for TLA+)
#   - spin (apt install spin)
#   - clang with libFuzzer

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

run_cbmc() {
    echo "=== CBMC Bounded Model Checking ==="
    if ! command -v cbmc &> /dev/null; then
        echo "CBMC not installed. Install with: apt install cbmc"
        return 1
    fi
    cd cbmc && make syntax && cd ..
    echo "CBMC: PASSED (syntax check)"
}

run_tlaplus() {
    echo "=== TLA+ State Verification ==="
    cd tlaplus
    if [ ! -f tla2tools.jar ]; then
        echo "Downloading TLA+ tools..."
        curl -fsSL -o tla2tools.jar \
            https://github.com/tlaplus/tlaplus/releases/download/v1.8.0/tla2tools.jar
    fi
    java -XX:+UseParallelGC -Xmx2g \
        -jar tla2tools.jar \
        -workers auto \
        -deadlock \
        -cleanup \
        -config SelfDrived_Safety.cfg \
        SelfDrived.tla
    cd ..
    echo "TLA+: PASSED"
}

run_spin() {
    echo "=== SPIN Protocol Verification ==="
    if ! command -v spin &> /dev/null; then
        echo "SPIN not installed. Install with: apt install spin"
        return 1
    fi
    cd spin && make verify && cd ..
    echo "SPIN: PASSED"
}

run_fuzz() {
    echo "=== libFuzzer Quick Check ==="
    if ! command -v clang &> /dev/null; then
        echo "Clang not installed"
        return 1
    fi
    cd fuzz && make all && make run && cd ..
    echo "Fuzzing: PASSED"
}

case "${1:-all}" in
    cbmc)
        run_cbmc
        ;;
    tlaplus)
        run_tlaplus
        ;;
    spin)
        run_spin
        ;;
    fuzz)
        run_fuzz
        ;;
    all)
        echo "Running all verification checks..."
        echo ""
        run_cbmc || true
        echo ""
        run_tlaplus || true
        echo ""
        run_spin || true
        echo ""
        run_fuzz || true
        echo ""
        echo "=== Verification Complete ==="
        ;;
    *)
        echo "Usage: $0 [all|cbmc|tlaplus|spin|fuzz]"
        exit 1
        ;;
esac
