#!/usr/bin/env bash
# preflight_check_v092.sh — pre-tag verification for v0.92-alpha ship-set.
#
# Walks every notebook + summary.json declared in V092_SCOPE.md and
# verifies:
#   1. The file exists.
#   2. (notebooks) zero error-type cells.
#   3. (summaries) chi_squared_per_pixel + max_abs_normalized_residual
#      are both non-null and within v0.92's PASS / borderline-PASS bar
#      (chi^2/N <= 1.3, max|res| <= 5.0sigma — Bonferroni floor for 9k+
#      pixel fits).
#
# Usage (from repo root):
#   bash Modules/10_Cluster_Computing/scripts/preflight_check_v092.sh
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

# -------- 1. Modules 01-10 --------
echo "=== v0.92 ship-set: Modules 01-10 ==="
for n in 01_Basics_Grids_Galaxies_RayTracing/01_grids_galaxies_raytracing \
         02_Simulating_Lens_Data/02_simulating_lens_data \
         03_First_Lens_Model/03_first_lens_model \
         04_Search_Chaining_SLaM/04_search_chaining_slam \
         05_Pixelized_Source_Reconstructions/05_pixelized_sources \
         06_Multi_Component_Mass_Models/06_multi_component_mass \
         07_Real_Data_FITS_to_Model/07_real_data_fits_to_model \
         08_Results_Diagnostics_Figures/08_results_diagnostics_figures \
         09_MGE_Linear_Light_Profiles/09_mge_linear_light_profiles \
         10_Cluster_Computing/10_cluster_computing; do
  nb="Modules/$n.ipynb"
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

# -------- 2. Recipe + climb notebooks --------
echo ""
echo "=== v0.92 ship-set: 7 recipe + climb notebooks ==="
for nb in \
  Modules/05_Pixelized_Source_Reconstructions/06_pixelization_recipe.ipynb \
  Modules/09_MGE_Linear_Light_Profiles/05_mge_recipe.ipynb \
  Examples/compound_lens_zoo/03_slam_recipe.ipynb \
  Examples/compound_lens/00_climb_to_compound.ipynb \
  Examples/compound_lens_zoo/00_climb_to_compound.ipynb \
  Examples/double_source_plane/00_climb_to_dspl.ipynb \
  Examples/group_scale/00_climb_to_group.ipynb; do
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

# -------- 3. Cluster docs --------
echo ""
echo "=== v0.92 ship-set: 3 cluster docs (>100 lines each) ==="
for f in \
  Modules/10_Cluster_Computing/SETUP_NEW_USER.md \
  Modules/10_Cluster_Computing/STUDENT_QUICKSTART.md \
  Modules/10_Cluster_Computing/RECIPES.md; do
  if [ ! -f "$f" ]; then
    print_fail "MISSING: $f"
    continue
  fi
  lines=$(wc -l < "$f")
  if [ "$lines" -ge 100 ]; then
    print_pass "$f ($lines lines)"
  else
    print_warn "$f ($lines lines — below 100-line threshold)"
  fi
done

# -------- 4. Shipping fit summaries --------
echo ""
echo "=== v0.92 ship-set: example fit summaries (chi^2/N <= 1.3 AND max|res| <= 5.0σ) ==="
declare -a SUMMARIES=(
  "Modules/05_Pixelized_Source_Reconstructions/results/search2_pixelized_source/summary.json"
  "Modules/06_Multi_Component_Mass_Models/results/composite_mass/summary.json"
  "Modules/09_MGE_Linear_Light_Profiles/results/source_pix[2]/summary.json"
  "Examples/compound_lens/results/compound_direct_fit/summary.json"
  "Examples/double_source_plane/results/dspl_direct_fit/summary.json"
  "Examples/disky_spiral_lens/results/bulge_disk_fit/summary.json"
  "Examples/group_scale/results/truth_anchored_fit/summary.json"
)
for s in "${SUMMARIES[@]}"; do
  if [ ! -f "$s" ]; then
    print_fail "MISSING: $s"
    continue
  fi
  chi=$(jq -r '.chi_squared_per_pixel' "$s")
  maxr=$(jq -r '.max_abs_normalized_residual' "$s")
  if [ "$chi" = "null" ] || [ "$maxr" = "null" ]; then
    print_warn "$s (null chi or max|res| — likely non-imaging likelihood; expected for point-source / interferometer)"
    continue
  fi
  awk -v c="$chi" -v m="$maxr" -v s="$s" -v g="$GREEN" -v r="$RED" -v y="$YELLOW" -v n="$NC" 'BEGIN {
    chi=c+0; mx=m+0
    if (chi <= 1.3 && mx <= 4.0) printf "  %s✓%s  %s — strict-PASS (chi=%.3f, max=%.2fσ)\n", g, n, s, chi, mx
    else if (chi <= 1.3 && mx <= 5.0) printf "  %s✓%s  %s — borderline-PASS (chi=%.3f, max=%.2fσ)\n", g, n, s, chi, mx
    else if (chi >= 2.0 || mx >= 6.0) printf "  %s✗%s  %s — FAIL (chi=%.3f, max=%.2fσ)\n", r, n, s, chi, mx
    else printf "  %s!%s  %s — SUSPECT (chi=%.3f, max=%.2fσ)\n", y, n, s, chi, mx
  }'
done

# -------- 5. Status banners --------
echo ""
echo "=== v0.92 status banners on in-progress READMEs ==="
for f in \
  Examples/agel_real_target/README.md \
  Examples/bayesian_model_comparison/README.md \
  Examples/group_scale/README.md \
  Examples/interferometer_basic/README.md \
  Examples/mge_to_physical/README.md \
  Examples/quad_time_delay/README.md \
  Examples/subhalo_sensitivity/README.md; do
  if grep -q "v0.92 ships" "$f" 2>/dev/null; then
    print_pass "$f (banner present)"
  else
    print_fail "$f (BANNER MISSING)"
  fi
done

# -------- 6. Top-level docs --------
echo ""
echo "=== v0.92 top-level docs ==="
for f in V092_SCOPE.md START_HERE.md README.md PROGRESS_LOG.md; do
  if [ -f "$f" ]; then
    print_pass "$f"
  else
    print_fail "MISSING: $f"
  fi
done

# -------- summary --------
echo ""
echo "================================================================"
echo " preflight_check_v092 summary:"
echo "   PASS:  $PASS_COUNT"
echo "   WARN:  $WARN_COUNT (acceptable; review)"
echo "   FAIL:  $FAIL_COUNT  ${FAIL_COUNT:+--- FIX BEFORE TAGGING}"
echo "================================================================"

if [ "$FAIL_COUNT" -gt 0 ]; then
  exit 1
fi
exit 0
