#!/bin/bash
#SBATCH --job-name=autolens
#SBATCH --partition=shared
#SBATCH --account=siag_lab
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=rodrigo.cordova_rosado@cfa.harvard.edu

# =============================================================================
# Generic Cannon submit for any Learning_to_Autolens module.
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

# ---- Paths (EDIT for your Cannon account) -----------------------------------
# On Cannon, $HOME = /n/home02/<user>/ (small quota — code only).
# Compute is billed to --account=siag_lab (fast scheduling), but heavy output
# is written to the Hernquist lab storage on holystore02. If the path below
# doesn't exist on your first run, check `ls /n/holystore02/LABS/hernquist_lab/`
# to confirm the exact layout and override OUTPUT_ROOT at submit time.
REPO_ROOT="${REPO_ROOT:-${HOME}/learning_to_autolens}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/n/holystore02/LABS/hernquist_lab/Users/${USER}/learning_to_autolens/output}"
CONDA_ENV="${CONDA_ENV:-autolens}"

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
source /n/sw/Miniforge3-25.3.1-0/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"

# Unbuffered Python output so logs appear live instead of in bursts when the
# job ends — essential for monitoring multi-hour Nautilus runs via `tail -f`.
export PYTHONUNBUFFERED=1

mkdir -p logs "${OUTPUT_ROOT}"

# ---- Diagnostics ------------------------------------------------------------
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
