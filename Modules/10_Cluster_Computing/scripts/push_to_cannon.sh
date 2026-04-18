#!/usr/bin/env bash
# push_to_cannon.sh — sync the local repo (code + datasets + preserved
# Nautilus checkpoints) to Cannon.
#
# Usage:
#   ./push_to_cannon.sh                     # dry run
#   ./push_to_cannon.sh --go                # actual transfer
#
# Override the Cannon target via env vars:
#   CANNON_USER=rcordovarosado CANNON_HOST=login.rc.fas.harvard.edu \
#     ./push_to_cannon.sh --go

set -euo pipefail

CANNON_USER="${CANNON_USER:-rcordovarosado}"
CANNON_HOST="${CANNON_HOST:-login.rc.fas.harvard.edu}"
CANNON_DEST="${CANNON_DEST:-learning_to_autolens}"

LOCAL_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"

DRY_FLAG="--dry-run"
if [[ "${1:-}" == "--go" ]]; then
    DRY_FLAG=""
fi

# -----------------------------------------------------------------------------
# Included on purpose:
#   Modules/                                            all module code + notebooks
#   Solutions/                                          (small; cheap to mirror)
#   slam_v2026.py                                       Mod 04 imports this
#   autolens_workspace_original/dataset/imaging/        Mods 04/05 datasets (few MB)
#   autolens_workspace_latest/dataset/imaging/          Mod 09 dataset
#   Modules/**/output/**/checkpoint.hdf5                lets Nautilus resume
#
# Excluded (saves bandwidth + keeps the cluster copy clean):
#   .git/ .claude/ __pycache__/ *.pyc .DS_Store .ipynb_checkpoints/
#   *.egg-info/ .venv/
#   Output/                                             cluster writes to $SCRATCH instead
#   autolens_workspace_latest/scripts, notebooks, config  (only need dataset/)
# -----------------------------------------------------------------------------

rsync -avh --progress ${DRY_FLAG} \
    --exclude='.git/' \
    --exclude='.claude/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='.DS_Store' \
    --exclude='.ipynb_checkpoints/' \
    --exclude='*.egg-info/' \
    --exclude='.venv/' \
    --exclude='Output/' \
    --include='autolens_workspace_latest/' \
    --include='autolens_workspace_latest/dataset/***' \
    --exclude='autolens_workspace_latest/*' \
    "${LOCAL_ROOT}/" \
    "${CANNON_USER}@${CANNON_HOST}:~/${CANNON_DEST}/"

if [[ -n "${DRY_FLAG}" ]]; then
    echo
    echo "This was a DRY RUN. Re-run with --go to actually transfer."
else
    echo
    echo "Next: submit a job."
    echo "  ssh ${CANNON_USER}@${CANNON_HOST}"
    echo "  cd ${CANNON_DEST}"
    echo "  sbatch --export=ALL,MODULE=04 --job-name=mod04 Modules/10_Cluster_Computing/scripts/submit_cannon.slurm"
fi
