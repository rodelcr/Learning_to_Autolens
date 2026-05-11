#!/usr/bin/env bash
# smoke_test_drivers.sh — verify every fit_example_*.py driver imports
# cleanly and exposes the expected --part choices.
#
# Why this is light: PYAUTOFIT_TEST_MODE was removed from autofit 2026.4+,
# so we can't short-circuit Nautilus to do a dry-run construct. The
# alternatives are (a) run the actual fit on the cluster, or (b) verify
# at minimum that argparse + driver imports work. We do (b) here to
# catch typos and missing modules; the actual cluster-runnability is
# proven by the recent successful runs (Track D joint fit, dspl_beta_chain,
# truth_fc_m3_v4, etc.).
#
# Each test is a `python <driver> --help` invocation. If the driver
# imports cleanly and argparse parses, --help exits 0 in <5 s. If the
# driver has a syntax error / import failure / argparse typo, it fails.
#
# Usage (from repo root):
#   bash Modules/10_Cluster_Computing/scripts/smoke_test_drivers.sh
#   bash Modules/10_Cluster_Computing/scripts/smoke_test_drivers.sh --pipeline
#       (the --pipeline flag tests only the 4-stage cosmography pipeline:
#        DSPL → MGE → physical → cosmography. Default tests every driver.)

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"

GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[0;33m'; NC='\033[0m'
PASS_COUNT=0
FAIL_COUNT=0
FAILED_DRIVERS=()

check_driver() {
    # $1 = driver basename
    # $2,$3,... = --part choices that should be exposed
    local driver="$1"; shift
    local script="$REPO_ROOT/Modules/10_Cluster_Computing/scripts/fit_example_${driver}.py"

    if [ ! -f "$script" ]; then
        echo -e "  ${RED}✗${NC} ${driver}  (driver script missing)"
        FAIL_COUNT=$((FAIL_COUNT+1))
        FAILED_DRIVERS+=("$driver")
        return
    fi

    # Step 1: argparse --help works (driver imports cleanly)
    local help_out
    help_out=$(cd "$REPO_ROOT" && python "$script" --help 2>&1)
    local help_rc=$?
    if [ "$help_rc" != "0" ]; then
        echo -e "  ${RED}✗${NC} ${driver}  (--help failed, exit $help_rc)"
        echo "$help_out" | tail -5 | sed 's/^/      /'
        FAIL_COUNT=$((FAIL_COUNT+1))
        FAILED_DRIVERS+=("$driver")
        return
    fi

    # Step 2: every expected --part choice appears in the help text
    local missing=()
    for part in "$@"; do
        if ! echo "$help_out" | grep -q "\\b${part}\\b"; then
            missing+=("$part")
        fi
    done
    if [ "${#missing[@]}" -gt 0 ]; then
        echo -e "  ${RED}✗${NC} ${driver}  (missing --part choices: ${missing[*]})"
        FAIL_COUNT=$((FAIL_COUNT+1))
        FAILED_DRIVERS+=("$driver")
        return
    fi

    echo -e "  ${GREEN}✓${NC} ${driver}  (--help OK; --part choices present: $*)"
    PASS_COUNT=$((PASS_COUNT+1))
}

PIPELINE_ONLY=0
[[ "${1:-}" == "--pipeline" ]] && PIPELINE_ONLY=1

echo "=== Smoke testing driver argparse + imports (--help) ==="
echo

if [ "$PIPELINE_ONLY" = "1" ]; then
    echo "Pipeline mode: DSPL → MGE → Physical → Cosmography"
    echo

    echo "--- Stage 1: DSPL ---"
    check_driver double_source_plane direct beta_fixedcosmo beta_freecosmo_v3 beta_chain

    echo
    echo "--- Stage 2: MGE ---"
    check_driver mge_to_physical light stars_only stars_dark stars_dark_v2 all

    echo
    echo "--- Stage 3: Physical (no driver — Module 11 audit only) ---"
    echo "  (skipped — Stage 3 is post-fit notebook analysis)"

    echo
    echo "--- Stage 4: Cosmography ---"
    check_driver quad_time_delay direct direct_h0_free_tight positions_only joint_fit joint_fit_h0_free
else
    echo "--- All drivers ---"
    check_driver double_source_plane    direct beta_fixedcosmo beta_freecosmo_v3 beta_chain
    check_driver mge_to_physical        light stars_only stars_dark stars_dark_v2 all
    check_driver quad_time_delay        direct direct_h0_free_tight positions_only joint_fit joint_fit_h0_free
    check_driver galaxy_galaxy_single_arc direct truth_anchored all
    check_driver compound_lens          direct direct_epl slam_effective slam_staged direct_with_positions_lh
    check_driver compound_lens_zoo_climb R5_truth_freecosmo
    check_driver agel_real_target       direct_clean
    check_driver cluster_scale          direct truth_anchored
    check_driver group_scale            bgg_shear_only bgg_plus_satellites truth_anchored
    check_driver group_scale_slam       # no --part choices to validate; argparse-only check
    check_driver subhalo_sensitivity    smooth with_subhalo both
    check_driver interferometer_basic   interferometer_basic
    check_driver disky_spiral_lens      single_sersic bulge_disk all
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
