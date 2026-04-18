#!/bin/bash
#SBATCH --job-name=autolens_mod04
#SBATCH --partition=shared
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=logs/mod04_%j.out
#SBATCH --error=logs/mod04_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=rodrigo.cordova_rosado@cfa.harvard.edu

# =============================================================================
# Cannon submission script for Module 04 (search chaining + SLaM).
#
# Nautilus checkpoints survive job requeues: if the previous job hit the time
# limit, resubmitting this script picks up from the last `checkpoint.hdf5`
# written by Nautilus. No special flag needed.
#
# Edit the path variables below to match your Cannon layout.
# =============================================================================

set -euo pipefail

# ---- Paths (EDIT THESE) -----------------------------------------------------
REPO_ROOT="${HOME}/learning_to_autolens"
DATASET_ROOT="${REPO_ROOT}/autolens_workspace_original/dataset/imaging"
OUTPUT_ROOT="${SCRATCH:-$HOME/scratch}/learning_to_autolens/output"
CONDA_ENV="autolens"

# ---- Environment ------------------------------------------------------------
# FASRC Cannon uses `module load`; Anaconda is available at /n/sw/helmod-rocky8/
module purge
module load python/3.11.8-fasrc01
module load Anaconda3/2024.06-fasrc01 || module load python
source activate "${CONDA_ENV}"

mkdir -p logs "${OUTPUT_ROOT}"

# ---- Diagnostics ------------------------------------------------------------
echo "Job ID:        ${SLURM_JOB_ID}"
echo "Node:          $(hostname)"
echo "Start time:    $(date -Is)"
echo "CPUs per task: ${SLURM_CPUS_PER_TASK}"
echo "Repo root:     ${REPO_ROOT}"
echo "Dataset root:  ${DATASET_ROOT}"
echo "Output root:   ${OUTPUT_ROOT}"
python -c "import autolens as al, autofit as af; \
           print('PyAutoLens', al.__version__, 'PyAutoFit', af.__version__)"

# ---- Run --------------------------------------------------------------------
cd "${REPO_ROOT}"

srun python "${REPO_ROOT}/Modules/10_Cluster_Computing/scripts/fit_module04.py" \
    --part all \
    --repo-root    "${REPO_ROOT}" \
    --dataset-root "${DATASET_ROOT}" \
    --output-root  "${OUTPUT_ROOT}" \
    --n-live-s1    100 \
    --n-live-s2    150 \
    --slam-n-live  100

echo "End time: $(date -Is)"
