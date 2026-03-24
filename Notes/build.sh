#!/bin/bash
# ============================================================
# build.sh — Build LaTeX notes for Learning to Autolens
# ============================================================
# Compiles each module's .tex file into a standalone PDF.
# Usage: cd Notes && bash build.sh
# Or build a single module: bash build.sh 01
# ============================================================

set -e

OUTDIR="../Output"
mkdir -p "$OUTDIR"

# Module directories and their main .tex files
declare -A MODULES=(
  ["01"]="01_Basics/01_basics_theory.tex"
  ["02"]="02_Simulating/02_simulating_theory.tex"
  ["03"]="03_First_Model/03_first_model_theory.tex"
  ["04"]="04_SLaM/04_slam_theory.tex"
)

build_module() {
  local num="$1"
  local texfile="${MODULES[$num]}"
  local dir=$(dirname "$texfile")
  local base=$(basename "$texfile" .tex)

  echo "Building Module $num: $texfile"
  cd "$dir"
  pdflatex -interaction=nonstopmode "$base.tex" > /dev/null 2>&1
  pdflatex -interaction=nonstopmode "$base.tex" > /dev/null 2>&1  # Second pass for refs
  mv "${base}.pdf" "../../$OUTDIR/"
  # Clean aux files
  rm -f "${base}.aux" "${base}.log" "${base}.out" "${base}.toc"
  cd ..
  echo "  → $OUTDIR/${base}.pdf"
}

if [ -n "$1" ]; then
  # Build single module
  build_module "$1"
else
  # Build all modules
  for num in $(echo "${!MODULES[@]}" | tr ' ' '\n' | sort); do
    build_module "$num"
  done
  echo ""
  echo "All modules built. PDFs in $OUTDIR/"
fi
