#!/usr/bin/env bash
# push_to_cannon.sh — sync the local repo (code + datasets + checkpoints) to
# Cannon, skipping caches, editor cruft, and the huge vendored workspaces we
# don't need on the cluster.
#
# Usage:
#   ./push_to_cannon.sh                     # dry run
#   ./push_to_cannon.sh --go                # actual transfer
#
# Edit CANNON_USER, CANNON_HOST, and CANNON_DEST for your account.

set -euo pipefail

CANNON_USER="${CANNON_USER:-rcordovarosado}"
CANNON_HOST="${CANNON_HOST:-login.rc.fas.harvard.edu}"
CANNON_DEST="${CANNON_DEST:-learning_to_autolens}"

LOCAL_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"

DRY_FLAG="--dry-run"
if [[ "${1:-}" == "--go" ]]; then
    DRY_FLAG=""
fi

# Include everything that's in the repo tree, EXCEPT:
#   - pycache/editor/os metadata
#   - .git                               (keep cluster copy decoupled from local git)
#   - .claude/                           (session artifacts)
#   - autolens_workspace_latest/         (big vendored workspace; copy separately if needed)
#   - .ipynb_checkpoints/
#
# We INTENTIONALLY include:
#   - autolens_workspace_original/dataset/  (small FITS; needed for fit)
#   - slam_v2026.py                         (Module 04 imports this)
#   - Modules/04_.../output/                (preserved checkpoint.hdf5 — enables resume)
#   - Modules/10_Cluster_Computing/         (scripts themselves)

rsync -avh --progress ${DRY_FLAG} \
    --exclude='.git/' \
    --exclude='.claude/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='.DS_Store' \
    --exclude='.ipynb_checkpoints/' \
    --exclude='*.egg-info/' \
    --exclude='.venv/' \
    --exclude='autolens_workspace_latest/' \
    --exclude='Output/' \
    "${LOCAL_ROOT}/" \
    "${CANNON_USER}@${CANNON_HOST}:~/${CANNON_DEST}/"

if [[ -n "${DRY_FLAG}" ]]; then
    echo
    echo "This was a DRY RUN. Re-run with --go to actually transfer."
fi
