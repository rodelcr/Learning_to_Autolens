#!/bin/bash
#SBATCH --job-name=autolens
#SBATCH --partition=siag_gpu
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

# =============================================================================
# Generic Cannon submit for any Learning_to_Autolens module.
#
# User-specific settings (account, mail-user, partition overrides, conda env
# name, lab-storage path) come from Modules/10_Cluster_Computing/cannon.env,
# which the submit_to_cannon.sh wrapper forwards as sbatch CLI overrides and
# as environment variables.
#
# Direct invocation (without the wrapper) still works but will miss the
# per-user overrides — set SBATCH_ACCOUNT / SBATCH_MAIL_USER yourself or
# pass --account / --mail-user on the sbatch command line:
#
#   SBATCH_ACCOUNT=mylab_lab sbatch --export=ALL,MODULE=04 submit_cannon.slurm
#
# Pick the module to run via the MODULE env var at submit time:
#
#   sbatch --export=ALL,MODULE=04 submit_cannon.slurm
#   sbatch --export=ALL,MODULE=05 submit_cannon.slurm
#   sbatch --export=ALL,MODULE=09 submit_cannon.slurm --mem=64G --time=48:00:00
#
# Nautilus checkpoints survive requeues — resubmitting the same command
# automatically resumes from the last `checkpoint.hdf5`.
# =============================================================================

set -euo pipefail

MODULE="${MODULE:-04}"

# ---- Safety: refuse to submit a test-mode run ------------------------------
# PYAUTOFIT_TEST_MODE=1 (documented in autolens_workspace_latest/CLAUDE.md as
# an integration-testing flag) makes every PyAutoFit search skip sampling and
# return a random prior draw. If it leaks into the submit environment, we'd
# silently burn cluster hours on a fit that writes junk artifacts.
if [[ -n "${PYAUTOFIT_TEST_MODE:-}" ]]; then
    echo "ERROR: PYAUTOFIT_TEST_MODE=${PYAUTOFIT_TEST_MODE} is set."
    echo "       PyAutoFit would skip sampling — every fit would be a random"
    echo "       prior draw. Refusing to submit. Unset it and resubmit:"
    echo "           unset PYAUTOFIT_TEST_MODE"
    echo "           sbatch --export=ALL,MODULE=${MODULE} submit_cannon.slurm"
    exit 1
fi

# ---- Paths -----------------------------------------------------------------
# On Cannon, $HOME = /n/home02/<user>/ (small quota — code only).
# The repo lives under lab storage on holystore01, and output goes there
# too. The canonical path depends on which lab the user belongs to; set
# REPO_ROOT in cannon.env or via the submit_to_cannon.sh wrapper.
# Example: /n/holystore01/LABS/<your_lab>_lab/Lab/<user>/learning_to_autolens
REPO_ROOT="${REPO_ROOT:-/n/holystore01/LABS/hernquist_lab/Lab/${USER}/learning_to_autolens}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/output}"
CONDA_ENV="${CONDA_ENV:-autolens312}"
# Miniforge installer path. FASRC-provided Miniforge lives here as of
# 2026-04; override via cannon.env if you bootstrap your own copy.
CONDA_ACTIVATE_SCRIPT="${CONDA_ACTIVATE_SCRIPT:-/n/sw/Miniforge3-25.3.1-0/etc/profile.d/conda.sh}"

# Dataset root differs by module:
#   04, 05 → autolens_workspace_original
#   09     → autolens_workspace_latest
case "${MODULE}" in
    04|05) DATASET_ROOT="${REPO_ROOT}/autolens_workspace_original/dataset/imaging" ;;
    09)    DATASET_ROOT="${REPO_ROOT}/autolens_workspace_latest/dataset/imaging"  ;;
    *)     echo "unknown MODULE=${MODULE}"; exit 2 ;;
esac

FIT_SCRIPT="${REPO_ROOT}/Modules/10_Cluster_Computing/scripts/fit_module${MODULE}.py"
if [[ ! -f "${FIT_SCRIPT}" ]]; then
    echo "fit script not found: ${FIT_SCRIPT}"; exit 2
fi

# ---- Environment ------------------------------------------------------------
# FASRC conda lives under Miniforge, not a module. Source the activate script
# rather than `module load Anaconda3` (which is out of date / gone on Cannon 2.0).
source "${CONDA_ACTIVATE_SCRIPT}"
conda activate "${CONDA_ENV}"

# Unbuffered Python output so logs appear live instead of in bursts when the
# job ends — essential for monitoring multi-hour Nautilus runs via `tail -f`.
export PYTHONUNBUFFERED=1

# Pin BLAS libraries to a single thread each. Nautilus parallelizes the
# likelihood evaluation via multiprocessing (number_of_cores=16); if OpenBLAS
# or MKL also try to parallelize, the two layers contend and the workers
# deadlock on a shared fork-lock. Symptom: all workers go to `S` state with
# 0% CPU and zero log progress for hours. Mod 05 (search2_pixelized_source)
# hit this in job 6571565 — 11 h of wall clock, zero samples written.
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p logs "${OUTPUT_ROOT}"

# ---- Diagnostics ------------------------------------------------------------
# Print enough provenance that a future debug of "why did my run produce
# the wrong numbers?" can be answered by reading this log alone. Specifically:
#   - SHA256 of the fit script actually executed (catches stale-rsync bugs)
#   - Git commit of the repo on Cannon (catches branch / unpushed drift)
#   - autolens / autofit versions (catches env drift)
echo "====================================================================="
echo "Module:        ${MODULE}"
echo "Job ID:        ${SLURM_JOB_ID}"
echo "Job name:      ${SLURM_JOB_NAME}"
echo "Node:          $(hostname)"
echo "Start time:    $(date -Is)"
echo "CPUs per task: ${SLURM_CPUS_PER_TASK}"
echo "Mem:           ${SLURM_MEM_PER_NODE:-?} MB"
echo "Repo root:     ${REPO_ROOT}"
echo "Dataset root:  ${DATASET_ROOT}"
echo "Output root:   ${OUTPUT_ROOT}"
echo "Fit script:    ${FIT_SCRIPT}"
echo "--- provenance -------------------------------------------------------"
FIT_SHA=$(sha256sum "${FIT_SCRIPT}" 2>/dev/null | awk '{print $1}')
echo "fit_module${MODULE}.py SHA256: ${FIT_SHA}"
echo "fit_module${MODULE}.py mtime:  $(stat -c '%y' "${FIT_SCRIPT}" 2>/dev/null || stat -f '%Sm' "${FIT_SCRIPT}")"
if (cd "${REPO_ROOT}" && git rev-parse --git-dir > /dev/null 2>&1); then
    echo "git HEAD:     $(cd "${REPO_ROOT}" && git rev-parse HEAD 2>/dev/null)"
    echo "git branch:   $(cd "${REPO_ROOT}" && git rev-parse --abbrev-ref HEAD 2>/dev/null)"
    # Any local edits after the last commit? This is the smoking gun for
    # "the laptop pushed uncommitted changes and the log can't reproduce".
    if ! (cd "${REPO_ROOT}" && git diff-index --quiet HEAD -- 2>/dev/null); then
        echo "git state:    DIRTY (uncommitted changes on Cannon — not reproducible from HEAD)"
        (cd "${REPO_ROOT}" && git status --short 2>/dev/null | head -20 | sed 's/^/              /')
    else
        echo "git state:    clean"
    fi
else
    echo "git:          (not a git repo)"
fi
echo "---"
python -c "import autolens as al, autofit as af; \
           print('PyAutoLens', al.__version__, 'PyAutoFit', af.__version__)"
echo "====================================================================="

# ---- Run --------------------------------------------------------------------
cd "${REPO_ROOT}"

srun python "${FIT_SCRIPT}" \
    --repo-root    "${REPO_ROOT}" \
    --dataset-root "${DATASET_ROOT}" \
    --output-root  "${OUTPUT_ROOT}"

echo "End time: $(date -Is)"

# ---- Post-process: export lightweight artifacts -----------------------------
# Produces Modules/${MODULE}_.../results/<search>/{fit_subplot.pdf, corner.pdf,
#   info.txt, summary.json, samples.csv}
# These are small and git-trackable, so a user can view finished results
# without needing the full Nautilus output tree.
echo "Exporting lightweight result artifacts..."
python "${REPO_ROOT}/Modules/10_Cluster_Computing/scripts/export_results.py" \
    --output-root "${OUTPUT_ROOT}/module_${MODULE}" \
    --module      "${MODULE}" \
    --repo-root   "${REPO_ROOT}" || echo "(export_results.py failed — see above)"

echo "All done: $(date -Is)"
