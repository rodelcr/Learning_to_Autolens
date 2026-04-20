#!/usr/bin/env bash
# push_to_cannon.sh — sync the local repo (code + datasets + preserved
# Nautilus checkpoints) to Cannon.
#
# Default destination is the lab-storage path used by submit_cannon.slurm
# and pull_from_cannon.sh, so push / submit / pull all operate on the
# same tree. Prior versions pushed to $HOME/learning_to_autolens which
# is a DIFFERENT filesystem from the one the slurm job reads — caused
# the 2026-04-20 stale-hash incident.
#
# Uses the SSH alias `cannon` by default (expected in ~/.ssh/config with
# ControlMaster auto + ControlPath, so Duo 2FA happens once per session
# instead of per command). Override with CANNON_SSH if your alias is
# different, or point it at user@host for a non-multiplexed setup.
#
# Usage:
#   ./push_to_cannon.sh                     # dry run
#   ./push_to_cannon.sh --go                # actual transfer
#
# Override the Cannon target via env vars:
#   CANNON_SSH=cannon                       # ssh alias (default)
#   CANNON_USER=rcordova                    # used only for path construction
#   CANNON_REPO_ROOT=/n/home02/rcordova/learning_to_autolens
#     ./push_to_cannon.sh --go

set -euo pipefail

# Source user-specific overrides if a cannon.env exists alongside this
# script (see ../cannon.env.example). Gitignored — each user edits their
# own copy with their Cannon username, lab path, slurm account, etc.
_CANNON_ENV="$(cd "$(dirname "$0")" && pwd)/../cannon.env"
if [[ -f "${_CANNON_ENV}" ]]; then
    # shellcheck source=/dev/null
    source "${_CANNON_ENV}"
fi

CANNON_SSH="${CANNON_SSH:-cannon}"       # SSH alias (see ~/.ssh/config)
CANNON_USER="${CANNON_USER:-rcordova}"   # only for building the remote path
# Absolute path on Cannon — matches submit_cannon.slurm's REPO_ROOT default.
# Using an absolute path (not ~/...) avoids ambiguity between home and lab
# storage, which live on different filesystems on Cannon.
CANNON_REPO_ROOT="${CANNON_REPO_ROOT:-/n/holystore01/LABS/hernquist_lab/Lab/${CANNON_USER}/learning_to_autolens}"
# Back-compat: CANNON_DEST still works if someone sets it, but we now
# resolve the destination from CANNON_REPO_ROOT.
CANNON_DEST="${CANNON_DEST:-${CANNON_REPO_ROOT}}"

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
    "${CANNON_SSH}:${CANNON_DEST}/"

if [[ -n "${DRY_FLAG}" ]]; then
    echo
    echo "This was a DRY RUN. Re-run with --go to actually transfer."
else
    echo
    echo "Pushed to: ${CANNON_SSH}:${CANNON_DEST}"
    echo
    echo "Next: submit a job."
    echo "  ssh ${CANNON_SSH}"
    echo "  cd ${CANNON_DEST}"
    echo "  sbatch --export=ALL,MODULE=04 --job-name=mod04 Modules/10_Cluster_Computing/scripts/submit_cannon.slurm"
fi
