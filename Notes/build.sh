#!/bin/sh
# ============================================================
# build.sh — Build LaTeX notes for Learning to Autolens
# ============================================================
# Compiles each module's .tex file into a standalone PDF.
# Usage: cd Notes && sh build.sh
# Or build a single module: sh build.sh 01
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTDIR="$SCRIPT_DIR/../Output"
mkdir -p "$OUTDIR"

build_module() {
  dir="$1"
  texfile="$2"
  base=$(basename "$texfile" .tex)

  echo "Building: $dir/$texfile"
  cd "$SCRIPT_DIR/$dir"
  pdflatex -interaction=nonstopmode "$base.tex" > /dev/null 2>&1
  pdflatex -interaction=nonstopmode "$base.tex" > /dev/null 2>&1
  mv "${base}.pdf" "$OUTDIR/"
  rm -f "${base}.aux" "${base}.log" "${base}.out" "${base}.toc"
  echo "  -> Output/${base}.pdf"
}

if [ -n "$1" ]; then
  case "$1" in
    01) build_module "01_Basics"       "01_basics_theory.tex" ;;
    02) build_module "02_Simulating"   "02_simulating_theory.tex" ;;
    03) build_module "03_First_Model"  "03_first_model_theory.tex" ;;
    04) build_module "04_SLaM"         "04_slam_theory.tex" ;;
    05) build_module "05_Pixelized"    "05_pixelized_theory.tex" ;;
    *)  echo "Unknown module: $1" ; exit 1 ;;
  esac
else
  build_module "01_Basics"       "01_basics_theory.tex"
  build_module "02_Simulating"   "02_simulating_theory.tex"
  build_module "03_First_Model"  "03_first_model_theory.tex"
  build_module "04_SLaM"         "04_slam_theory.tex"
  # build_module "05_Pixelized"  "05_pixelized_theory.tex"  # TODO
  echo ""
  echo "All modules built. PDFs in $OUTDIR/"
fi
