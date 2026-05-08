#!/usr/bin/env bash
# smoke_test_drivers.sh — verify every fit_example_*.py driver constructs
# its model + analysis without error before submitting to Cannon.
#
# Uses PYAUTOFIT_TEST_MODE=1 which makes Nautilus skip sampling and return
# a random prior draw — so each "fit" exits in seconds. We only check that:
#   1. argparse accepts the --part choice
#   2. The dataset loads
#   3. The model + analysis + search construct
#   4. PyAutoFit can do a single likelihood evaluation
#
# Failures here are guaranteed to fail on Cannon — running this before
# `sbatch` saves a 30-minute round-trip per typo.
#
# Usage (from repo root):
#   bash Modules/10_Cluster_Computing/scripts/smoke_test_drivers.sh
#   bash Modules/10_Cluster_Computing/scripts/smoke_test_drivers.sh --pipeline
#       (the --pipeline flag tests only the 4-stage cosmography pipeline:
#        DSPL → MGE → physical → cosmography. Default tests every driver.)

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
TMP_OUT="$(mktemp -d)"
trap "rm -rf '$TMP_OUT'" EXIT

GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[0;33m'; NC='\033[0m'
PASS_COUNT=0
FAIL_COUNT=0
FAILED_DRIVERS=()

run_smoke() {
    # $1 = driver basename (without fit_example_ prefix and .py suffix)
    # $2 = --part value
    # $3 = dataset subdir relative to Examples/<basename>/
    local driver="$1" part="$2" dataset_subdir="$3"
    local script="$REPO_ROOT/Modules/10_Cluster_Computing/scripts/fit_example_${driver}.py"
    local dataset_root="$REPO_ROOT/Examples/${driver}/${dataset_subdir}"
    local output_root="$TMP_OUT/${driver}_${part}"
    mkdir -p "$output_root"

    if [ ! -f "$script" ]; then
        echo -e "  ${RED}✗${NC} ${driver} --part=${part}  (driver script missing)"
        FAIL_COUNT=$((FAIL_COUNT+1))
        FAILED_DRIVERS+=("$driver:$part")
        return
    fi
    if [ ! -d "$dataset_root" ]; then
        echo -e "  ${YELLOW}!${NC} ${driver} --part=${part}  (dataset dir missing: $dataset_root)"
        return
    fi

    local log="$output_root/smoke.log"
    # Pick a portable timeout: GNU timeout on Linux, gtimeout on macOS (coreutils),
    # or no-op if neither is available (PYAUTOFIT_TEST_MODE returns in seconds anyway).
    local timeout_cmd=""
    if command -v timeout >/dev/null 2>&1; then
        timeout_cmd="timeout 180"
    elif command -v gtimeout >/dev/null 2>&1; then
        timeout_cmd="gtimeout 180"
    fi
    (
        cd "$REPO_ROOT" && \
        PYAUTOFIT_TEST_MODE=1 \
        $timeout_cmd python "$script" \
            --part="$part" \
            --repo-root="$REPO_ROOT" \
            --dataset-root="$dataset_root" \
            --output-root="$output_root" \
            --n-live=10 \
            > "$log" 2>&1
    )
    local rc=$?
    if [ "$rc" = "0" ]; then
        echo -e "  ${GREEN}✓${NC} ${driver} --part=${part}"
        PASS_COUNT=$((PASS_COUNT+1))
    else
        echo -e "  ${RED}✗${NC} ${driver} --part=${part}  (exit $rc; tail of log:)"
        tail -10 "$log" | sed 's/^/      /'
        FAIL_COUNT=$((FAIL_COUNT+1))
        FAILED_DRIVERS+=("$driver:$part")
    fi
}

PIPELINE_ONLY=0
[[ "${1:-}" == "--pipeline" ]] && PIPELINE_ONLY=1

echo "=== Smoke testing drivers (PYAUTOFIT_TEST_MODE=1, n_live=10, 3-min timeout) ==="
echo

if [ "$PIPELINE_ONLY" = "1" ]; then
    echo "Pipeline mode: DSPL → MGE → Physical → Cosmography"
    echo
    echo "--- Stage 1: DSPL ---"
    run_smoke double_source_plane direct      mocks
    run_smoke double_source_plane beta_chain  mocks

    echo
    echo "--- Stage 2: MGE ---"
    run_smoke mge_to_physical light       mocks
    run_smoke mge_to_physical stars_dark  mocks

    echo
    echo "--- Stage 3: Physical (no driver — Module 11 audit only) ---"
    echo "  (skipped — Stage 3 is post-fit notebook analysis)"

    echo
    echo "--- Stage 4: Cosmography ---"
    run_smoke quad_time_delay direct                 mocks
    run_smoke quad_time_delay direct_h0_free_tight   mocks
    run_smoke quad_time_delay joint_fit_h0_free      mocks_with_host
else
    echo "--- DSPL ---"
    run_smoke double_source_plane direct      mocks
    run_smoke double_source_plane beta_chain  mocks

    echo
    echo "--- MGE ---"
    run_smoke mge_to_physical light       mocks
    run_smoke mge_to_physical stars_dark  mocks

    echo
    echo "--- Cosmography (TDCOSMO) ---"
    run_smoke quad_time_delay direct                 mocks
    run_smoke quad_time_delay direct_h0_free_tight   mocks
    run_smoke quad_time_delay positions_only         mocks
    run_smoke quad_time_delay joint_fit              mocks_with_host
    run_smoke quad_time_delay joint_fit_h0_free      mocks_with_host

    echo
    echo "--- Compound (cross-link) ---"
    run_smoke compound_lens compound_direct_fit  mocks
fi

echo
echo "================================================================"
echo " Smoke summary:  PASS=${PASS_COUNT}   FAIL=${FAIL_COUNT}"
if [ "$FAIL_COUNT" -gt 0 ]; then
    echo " Failed drivers:"
    for d in "${FAILED_DRIVERS[@]}"; do echo "   - $d"; done
fi
echo "================================================================"

exit $FAIL_COUNT
