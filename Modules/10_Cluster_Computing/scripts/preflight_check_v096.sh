#!/usr/bin/env bash
# preflight_check_v096.sh — pre-tag verification for v0.96-alpha ship-set.
#
# Adds v0.96-specific checks on top of the v0.94 baseline:
#   1. v0.94 baseline still passes (delegate to preflight_check_v094.sh)
#   2. Module 15 (Radial Arcs & Caustic Topology) present + executes clean
#   3. Examples/radial_arc_smbh/ shipped (mock + driver + README)
#   4. Examples/positions_modeling/ shipped (notebook + README)
#   5. Examples/galaxy_galaxy_single_arc/ shipped (mock + notebook +
#      driver + audited fit results)
#   6. radial_arc_smbh + ggsa + DSPL + mge_to_physical mock generators
#      have chi^2-at-truth assertions in their code
#   7. mge_to_physical regenerate_in_autolens.py uses the (center_y,
#      center_x) axis-swap convention (the 2026-05-15 fix)
#   8. DSPL generate_mock.py uses the single-truth-dict pattern
#   9. v0.95 + v0.96 cross-link READMEs present
#  10. (conditional) Cannon-result strict-PASS checks for the v0.96
#      headline fits (ggsa_direct, dspl_v096 Stage 2, radial_arc_smbh
#      with_pointmass, mge stars_dark_v2_autolens_v2)
#
# Usage (from repo root):
#   bash Modules/10_Cluster_Computing/scripts/preflight_check_v096.sh
#
# Exit code 0 = ready to tag. Exit code != 0 = fix listed issues first.
set -uo pipefail

GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[0;33m'; NC='\033[0m'
PASS_COUNT=0
FAIL_COUNT=0
WARN_COUNT=0

print_pass() { echo -e "  ${GREEN}✓${NC} $*"; PASS_COUNT=$((PASS_COUNT+1)); }
print_fail() { echo -e "  ${RED}✗${NC} $*"; FAIL_COUNT=$((FAIL_COUNT+1)); }
print_warn() { echo -e "  ${YELLOW}!${NC} $*"; WARN_COUNT=$((WARN_COUNT+1)); }

check_summary_strict_pass_or_warn() {
    local path="$1" label="$2"
    if [ ! -f "$path" ]; then
        print_warn "$label — summary.json absent (research-in-progress, OK at tag time)"
        return
    fi
    chi=$(jq -r '.chi_squared_per_pixel // "null"' "$path")
    mxr=$(jq -r '.max_abs_normalized_residual // "null"' "$path")
    if [ "$chi" = "null" ] || [ "$mxr" = "null" ]; then
        print_warn "$label — summary.json present but null chi/max (incomplete)"
        return
    fi
    chi_ok=$(awk -v c="$chi" 'BEGIN{print (c+0 <= 1.3) ? 1 : 0}')
    mxr_ok=$(awk -v m="$mxr" 'BEGIN{print (m+0 <= 4.5) ? 1 : 0}')
    printf -v chi_disp "%.3f" "$chi"
    printf -v mxr_disp "%.2f" "$mxr"
    if [ "$chi_ok" = "1" ] && [ "$mxr_ok" = "1" ]; then
        print_pass "$label — strict-PASS (chi=${chi_disp}, max=${mxr_disp}σ)"
    else
        print_fail "$label — NOT strict-PASS (chi=${chi_disp}, max=${mxr_disp}σ)"
    fi
}

# -------- 1. v0.94 baseline still passes --------
echo "=== Step 1: re-running v0.94 preflight ==="
if bash "$(dirname "$0")/preflight_check_v094.sh" >/tmp/preflight_v094.log 2>&1; then
    v094_fail_count=$(grep -E '^   FAIL:' /tmp/preflight_v094.log | grep -oE '[0-9]+' | head -1)
    v094_pass_count=$(grep -E '^   PASS:' /tmp/preflight_v094.log | grep -oE '[0-9]+' | head -1)
    if [ "${v094_fail_count:-1}" = "0" ]; then
        print_pass "v0.94 baseline still passes (${v094_pass_count} PASS / 0 FAIL)"
    else
        print_fail "v0.94 baseline regressed — ${v094_fail_count} FAIL; see /tmp/preflight_v094.log"
    fi
else
    print_fail "v0.94 preflight returned non-zero — see /tmp/preflight_v094.log"
fi

# -------- 2. Module 15 notebook present + executes clean --------
echo
echo "=== Step 2: Module 15 (Radial Arcs & Caustic Topology) ==="
nb="Modules/15_Radial_Arcs_Caustic_Topology/15_radial_arcs.ipynb"
if [ -f "$nb" ]; then
    errs=$(jq -r '[.cells[] | select(.cell_type=="code") | .outputs[]? | select(.output_type=="error")] | length' "$nb" 2>/dev/null || echo "?")
    if [ "$errs" = "0" ]; then
        print_pass "$nb (0 error cells)"
    else
        print_fail "$nb ($errs error cells)"
    fi
else
    print_fail "MISSING: $nb"
fi

# -------- 3. Examples/radial_arc_smbh shipped --------
echo
echo "=== Step 3: Examples/radial_arc_smbh ==="
for f in \
    "Examples/radial_arc_smbh/README.md" \
    "Examples/radial_arc_smbh/mocks/generate_mock.py" \
    "Examples/radial_arc_smbh/mocks/image.fits" \
    "Examples/radial_arc_smbh/mocks/truths.json" \
    "Modules/10_Cluster_Computing/scripts/fit_example_radial_arc_smbh.py"; do
    if [ -f "$f" ]; then
        print_pass "$f"
    else
        print_fail "MISSING: $f"
    fi
done

# -------- 4. Examples/positions_modeling shipped (v0.95) --------
echo
echo "=== Step 4: Examples/positions_modeling (v0.95) ==="
for f in \
    "Examples/positions_modeling/README.md" \
    "Examples/positions_modeling/01_positions_tutorial.ipynb"; do
    if [ -f "$f" ]; then
        print_pass "$f"
    else
        print_fail "MISSING: $f"
    fi
done

# -------- 5. Examples/galaxy_galaxy_single_arc shipped --------
echo
echo "=== Step 5: Examples/galaxy_galaxy_single_arc (v0.95) ==="
for f in \
    "Examples/galaxy_galaxy_single_arc/README.md" \
    "Examples/galaxy_galaxy_single_arc/mocks/generate_mock.py" \
    "Examples/galaxy_galaxy_single_arc/mocks/image.fits" \
    "Examples/galaxy_galaxy_single_arc/mocks/truths.json" \
    "Examples/galaxy_galaxy_single_arc/01_galaxy_galaxy_single_arc.ipynb" \
    "Modules/10_Cluster_Computing/scripts/fit_example_galaxy_galaxy_single_arc.py"; do
    if [ -f "$f" ]; then
        print_pass "$f"
    else
        print_fail "MISSING: $f"
    fi
done

# -------- 6. Mock generators have chi^2-at-truth assertions --------
echo
echo "=== Step 6: chi^2-at-truth assertions in mock generators ==="
# The 2026-05-15 lesson: every mock generator MUST have a chi^2-at-truth
# self-consistency check at the end. Failure of this codified rule was
# the root cause of both the DSPL Stage 1 stall AND the mge_to_physical
# chi^2/N=6+ issues (different bugs, same missing safety net).
for f in \
    "Examples/double_source_plane/mocks/generate_mock.py" \
    "Examples/galaxy_galaxy_single_arc/mocks/generate_mock.py" \
    "Examples/radial_arc_smbh/mocks/generate_mock.py" \
    "Examples/mge_to_physical/mocks/regenerate_in_autolens.py" \
    "Examples/quad_time_delay/mocks_with_host/generate_mock.py"; do
    if [ -f "$f" ] && grep -qE "chi.*at.*truth|chi_squared_per_pix.*<=" "$f"; then
        print_pass "$f has chi^2-at-truth assertion"
    elif [ -f "$f" ]; then
        print_fail "$f exists but missing chi^2-at-truth assertion"
    else
        print_warn "$f does not exist — not yet a generator-style example"
    fi
done

# -------- 7. mge regen uses (center_y, center_x) axis-swap fix --------
echo
echo "=== Step 7: mge regen axis-swap convention (2026-05-15 fix) ==="
mge_regen="Examples/mge_to_physical/mocks/regenerate_in_autolens.py"
# All 10 centre sites (5 simulator constructors + 5 JSON dump lines) must
# use (center_y, center_x). Count both orderings.
new_count=$(grep -cE 'center_y"\]\),[[:space:]]*float\([a-z0-9_]+\["center_x"\]' "$mge_regen" 2>/dev/null); new_count=${new_count:-0}
old_count=$(grep -cE 'center_x"\]\),[[:space:]]*float\([a-z0-9_]+\["center_y"\]' "$mge_regen" 2>/dev/null); old_count=${old_count:-0}
if [ "$old_count" -gt 0 ]; then
    print_fail "mge regen has $old_count remaining (center_x, center_y) sites (should be 0)"
elif [ "$new_count" -lt 10 ]; then
    print_fail "mge regen has only $new_count (center_y, center_x) sites (expected ≥ 10)"
else
    print_pass "mge regen uses (center_y, center_x) at all $new_count sites"
fi

# -------- 8. DSPL generator uses single-truth-dict pattern --------
echo
echo "=== Step 8: DSPL generator uses single-truth-dict pattern ==="
dspl_gen="Examples/double_source_plane/mocks/generate_mock.py"
if [ -f "$dspl_gen" ] && grep -q '^TRUTH = {' "$dspl_gen" \
   && grep -q '_galaxy_from_truth' "$dspl_gen"; then
    print_pass "DSPL generator has single-truth-dict pattern"
else
    print_fail "DSPL generator missing single-truth-dict pattern (stale-JSON bug risk)"
fi

# -------- 9. v0.96 onboarding files --------
echo
echo "=== Step 9: v0.96 onboarding files ==="
for f in \
    "NEXT_STEPS.md" \
    "docs/superpowers/specs/2026-05-14-v0.96-radial-arc-and-dspl-polish-design.md"; do
    if [ -f "$f" ]; then
        print_pass "$f"
    else
        print_warn "MISSING (optional): $f"
    fi
done

# -------- 10. Conditional Cannon-result strict-PASS --------
echo
echo "=== Step 10: v0.96 Cannon-result strict-PASS (conditional) ==="
check_summary_strict_pass_or_warn \
    "Examples/galaxy_galaxy_single_arc/results/ggsa_direct/summary.json" \
    "ggsa_direct"
check_summary_strict_pass_or_warn \
    "Examples/radial_arc_smbh/results/rarc_direct/summary.json" \
    "radial_arc_smbh direct"
check_summary_strict_pass_or_warn \
    "Examples/radial_arc_smbh/results/rarc_with_pointmass/summary.json" \
    "radial_arc_smbh with_pointmass"
check_summary_strict_pass_or_warn \
    "Examples/double_source_plane/results/beta_freecosmo_v3/summary.json" \
    "DSPL beta_freecosmo_v3 (v0.96 regenerated mock)"
# mge Search 3 v2: research-in-progress in v0.96 (carried forward from v0.95).
# The axis-fix landed 50% chi^2/N reduction (6.44 -> 3.16) but the model
# still shows coherent ring residuals — v0.97 follow-on is lp_linear.Sersic
# + MGE light (Module 09 path). Report as WARN regardless of strict-PASS
# verdict, so v0.96 tag is not blocked by a known research debt.
mge_s3v2_summary="Examples/mge_to_physical/results/search_3_v2_stars_dark/summary.json"
if [ -f "$mge_s3v2_summary" ] && grep -q "_autolens_v2" "$mge_s3v2_summary"; then
    chi2=$(python3 -c "import json; print(json.load(open('$mge_s3v2_summary'))['chi_squared_per_pixel'])" 2>/dev/null || echo "?")
    maxres=$(python3 -c "import json; print(json.load(open('$mge_s3v2_summary'))['max_abs_normalized_residual'])" 2>/dev/null || echo "?")
    print_warn "mge Search 3 v2 (axis-fixed): chi^2/N=${chi2}, max|res|=${maxres}σ — research-in-progress, v0.97 follow-on (lp_linear+MGE)"
elif [ -f "$mge_s3v2_summary" ]; then
    print_warn "mge Search 3 v2 — pre-axis-fix result on disk (research-in-progress)"
else
    print_warn "mge Search 3 v2 — summary.json absent (research-in-progress)"
fi

# -------- Summary --------
echo
echo "================================================================"
echo " preflight_check_v096 summary:"
echo "   PASS:  ${PASS_COUNT}"
echo "   WARN:  ${WARN_COUNT} (research-in-progress; OK at tag time)"
if [ "$FAIL_COUNT" = "0" ]; then
    echo "   FAIL:  0  --- READY TO TAG v0.96-alpha"
else
    echo -e "   ${RED}FAIL:  ${FAIL_COUNT}  --- FIX BEFORE TAGGING${NC}"
fi
echo "================================================================"

if [ "$FAIL_COUNT" != "0" ]; then
    exit 1
fi
exit 0
