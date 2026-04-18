#!/usr/bin/env bash
# pull_from_cannon.sh — pull Nautilus outputs (samples, posteriors, images)
# from Cannon back to the local repo so we can inspect results in the notebook.
#
# Usage:
#   ./pull_from_cannon.sh              # dry run
#   ./pull_from_cannon.sh --go         # actual transfer

set -euo pipefail

CANNON_USER="${CANNON_USER:-rcordovarosado}"
CANNON_HOST="${CANNON_HOST:-login.rc.fas.harvard.edu}"
CANNON_SCRATCH="${CANNON_SCRATCH:-/n/holyscratch01/users/${CANNON_USER}/learning_to_autolens/output}"

LOCAL_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
LOCAL_DEST="${LOCAL_ROOT}/Modules/10_Cluster_Computing/cannon_output"

DRY_FLAG="--dry-run"
if [[ "${1:-}" == "--go" ]]; then
    DRY_FLAG=""
fi

mkdir -p "${LOCAL_DEST}"

rsync -avh --progress ${DRY_FLAG} \
    "${CANNON_USER}@${CANNON_HOST}:${CANNON_SCRATCH}/" \
    "${LOCAL_DEST}/"

if [[ -n "${DRY_FLAG}" ]]; then
    echo
    echo "This was a DRY RUN. Re-run with --go to actually transfer."
fi
