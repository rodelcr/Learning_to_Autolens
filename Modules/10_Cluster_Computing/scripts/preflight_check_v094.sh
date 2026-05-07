#!/usr/bin/env bash
# preflight_check_v094.sh — pre-tag verification for v0.94-alpha ship-set.
#
# Adds v0.94-specific checks on top of the v0.93 baseline:
#   1. v0.93 baseline still passes (delegate to preflight_check_v093.sh)
#   2. Module 11 (Physical Mass Models) + Solutions/SOLVED present and execute clean
#   3. 6 cross-referencing READMEs no longer say "Module 11 planned"
#   4. CLAUDE.md curriculum table marks Module 11 as ✓ ship
#   5. New v0.94 onboarding files exist:
#        - RELEASE_NOTES_v0.94.md
#        - Modules/10_Cluster_Computing/scripts/diagnose_nautilus_resume.py
#        - "Checkpoint hygiene" section in CLUSTER_WORKFLOW_NOTES.md
#   6. fit_example_double_source_plane.py exposes --part=beta_chain
#   7. fit_example_compound_lens_zoo_climb.py uses TruncatedGaussianPrior
#      on Om0/w0 in build_R5_truth_freecosmo_model
#   8. (conditional) DSPL beta_freecosmo_v3 summary.json strict-PASS — only
#      enforced if the Cannon result exists; otherwise WARN (research-in-progress)
#   9. (conditional) compound_lens_zoo mock_3 R5_truth_freecosmo summary.json
#      strict-PASS — same conditional rule
#  10. mge_to_physical README has the 2026-05-07 chi²-at-truth update
#
# Usage (from repo root):
#   bash Modules/10_Cluster_Computing/scripts/preflight_check_v094.sh
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
    # $1 = path to summary.json, $2 = label
    # If file missing: WARN (research-in-progress).
    # If present but not strict-PASS: FAIL.
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
    mxr_ok=$(awk -v m="$mxr" 'BEGIN{print (m+0 <= 4.1) ? 1 : 0}')
    printf -v chi_disp "%.3f" "$chi"
    printf -v mxr_disp "%.2f" "$mxr"
    if [ "$chi_ok" = "1" ] && [ "$mxr_ok" = "1" ]; then
        print_pass "$label — strict-PASS (chi=${chi_disp}, max=${mxr_disp}σ)"
    else
        print_fail "$label — NOT strict-PASS (chi=${chi_disp}, max=${mxr_disp}σ)"
    fi
}

# -------- 1. v0.93 baseline still passes --------
echo "=== Step 1: re-running v0.93 preflight ==="
if bash "$(dirname "$0")/preflight_check_v093.sh" >/tmp/preflight_v093.log 2>&1; then
    v093_fail_count=$(grep -E '^   FAIL:' /tmp/preflight_v093.log | grep -oE '[0-9]+' | head -1)
    v093_pass_count=$(grep -E '^   PASS:' /tmp/preflight_v093.log | grep -oE '[0-9]+' | head -1)
    if [ "${v093_fail_count:-1}" = "0" ]; then
        print_pass "v0.93 baseline still passes (${v093_pass_count} PASS / 0 FAIL)"
    else
        print_fail "v0.93 baseline regressed — ${v093_fail_count} FAIL; see /tmp/preflight_v093.log"
    fi
else
    print_fail "v0.93 preflight returned non-zero — see /tmp/preflight_v093.log"
fi

# -------- 2. Module 11 + Module 12 notebooks present + execute clean --------
echo
echo "=== Step 2: Modules 11 + 12 (Physical Mass Models, TDCOSMO + MSD) ==="
for nb in \
    "Modules/11_Physical_Mass_Models/11_physical_mass_models.ipynb" \
    "Solutions/11_physical_mass_models_SOLVED.ipynb" \
    "Modules/12_Time_Delay_Cosmography_MSD/12_time_delay_cosmography_msd.ipynb" \
    "Solutions/12_time_delay_cosmography_msd_SOLVED.ipynb"; do
    if [ ! -f "$nb" ]; then
        print_fail "MISSING: $nb"
        continue
    fi
    errs=$(jq -r '[.cells[] | select(.cell_type=="code") | .outputs[]? | select(.output_type=="error")] | length' "$nb" 2>/dev/null || echo "?")
    if [ "$errs" = "0" ]; then
        print_pass "$nb (0 error cells)"
    else
        print_fail "$nb ($errs error cells)"
    fi
done

# -------- 3. Cross-referencing READMEs no longer say "Module 11 planned" --------
echo
echo "=== Step 3: Module 11 cross-reference updates ==="
for f in \
    "README.md" \
    "START_HERE.md" \
    "CLAUDE.md" \
    "Examples/mge_to_physical/README.md" \
    "Examples/cluster_scale/README.md" \
    "Examples/group_scale/README.md"; do
    if [ ! -f "$f" ]; then
        print_fail "MISSING: $f"
        continue
    fi
    if grep -qiE "module 11.*planned|planned.*module 11" "$f"; then
        print_fail "$f still references 'Module 11 planned'"
    else
        print_pass "$f cross-link updated"
    fi
done

# -------- 4. Onboarding files --------
echo
echo "=== Step 4: v0.94 onboarding files ==="
for f in \
    "RELEASE_NOTES_v0.94.md" \
    "Modules/10_Cluster_Computing/scripts/diagnose_nautilus_resume.py"; do
    if [ -f "$f" ]; then
        print_pass "$f"
    else
        print_fail "MISSING: $f"
    fi
done

# Checkpoint hygiene section in workflow notes
if grep -q "Checkpoint hygiene" Modules/10_Cluster_Computing/CLUSTER_WORKFLOW_NOTES.md 2>/dev/null; then
    print_pass "CLUSTER_WORKFLOW_NOTES.md has 'Checkpoint hygiene' section"
else
    print_fail "CLUSTER_WORKFLOW_NOTES.md missing 'Checkpoint hygiene' section"
fi

# -------- 5. DSPL driver --part=beta_chain --------
echo
echo "=== Step 5: DSPL driver staged-chain ==="
dspl_drv="Modules/10_Cluster_Computing/scripts/fit_example_double_source_plane.py"
if [ -f "$dspl_drv" ] && grep -q "beta_chain" "$dspl_drv" \
   && grep -q "build_beta_fixedcosmo_fit" "$dspl_drv" \
   && grep -q "build_beta_freecosmo_v3_fit" "$dspl_drv"; then
    print_pass "fit_example_double_source_plane.py has staged-chain machinery"
else
    print_fail "DSPL driver missing staged-chain machinery"
fi

# -------- 6. compound_zoo freecosmo TruncatedGaussian on cosmology --------
echo
echo "=== Step 6: compound_zoo freecosmo prior bounds ==="
zoo_drv="Modules/10_Cluster_Computing/scripts/fit_example_compound_lens_zoo_climb.py"
# Grep directly for the cosmology TruncatedGaussianPrior assignments — both
# Om0 and w0 lines must be present.
if [ -f "$zoo_drv" ] \
   && grep -q "cosmology\.Om0 = af\.TruncatedGaussianPrior" "$zoo_drv" \
   && grep -q "cosmology\.w0\s*=\s*af\.TruncatedGaussianPrior" "$zoo_drv"; then
    print_pass "build_R5_truth_freecosmo_model uses TruncatedGaussianPrior on Om0 + w0"
else
    print_fail "build_R5_truth_freecosmo_model missing TruncatedGaussian on cosmology"
fi

# -------- 7. DSPL Cannon result (conditional) --------
echo
echo "=== Step 7 (conditional): DSPL beta_freecosmo_v3 Cannon result ==="
check_summary_strict_pass_or_warn \
    "Examples/double_source_plane/results/beta_freecosmo_v3/summary.json" \
    "DSPL beta_freecosmo_v3"

# -------- 8. compound_zoo mock_3 freecosmo Cannon result (conditional) --------
echo
echo "=== Step 8 (conditional): compound_zoo mock_3 truth_freecosmo Cannon ==="
check_summary_strict_pass_or_warn \
    "Examples/compound_lens_zoo/results/mock_3_R5_truth_freecosmo/summary.json" \
    "compound_zoo mock_3 truth_freecosmo"

# -------- 9. mge_to_physical README has v0.94 update --------
echo
echo "=== Step 9: mge_to_physical README v0.94 chi²-at-truth update ==="
if grep -q "2026-05-07.*chi²-at-truth\|chi²-at-truth diagnostic.*2026-05-07" \
       Examples/mge_to_physical/README.md 2>/dev/null \
   || grep -q "v0.94 chi²-at-truth diagnostic" \
       Examples/mge_to_physical/README.md 2>/dev/null; then
    print_pass "mge_to_physical README has v0.94 chi²-at-truth update"
else
    print_fail "mge_to_physical README missing v0.94 update"
fi

# -------- Summary --------
echo
echo "================================================================"
echo " preflight_check_v094 summary:"
echo "   PASS:  ${PASS_COUNT}"
echo "   WARN:  ${WARN_COUNT} (research-in-progress; OK at tag time)"
if [ "$FAIL_COUNT" = "0" ]; then
    echo "   FAIL:  0  --- READY TO TAG v0.94-alpha"
else
    echo -e "   ${RED}FAIL:  ${FAIL_COUNT}  --- FIX BEFORE TAGGING${NC}"
fi
echo "================================================================"

if [ "$FAIL_COUNT" != "0" ]; then
    exit 1
fi
exit 0
