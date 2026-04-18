#!/usr/bin/env bash
# seed_cannon_data.sh — one-time sync of the two things `git pull` can't carry:
#
#   1. autolens_workspace_latest/dataset/ (gitignored — needed for Module 09).
#   2. Preserved Nautilus checkpoint.hdf5 files under Modules/*/output/ and
#      Solutions/output/ (gitignored — lets cluster jobs resume instead of
#      starting cold).
#
# The checkpoints are rsync'd into their correct per-module output/ paths so
# Nautilus finds them automatically on re-submit; no post-rsync reshuffle.
#
# Prereq: an active ssh ControlMaster session to Cannon so rsync doesn't
# re-prompt for 2FA. Your ~/.ssh/config already defines the `cannon` alias
# with ControlMaster auto; just open `ssh cannon` in another terminal first
# and leave it running.
#
# Usage:
#   ./seed_cannon_data.sh          # dry run
#   ./seed_cannon_data.sh --go     # actual transfer

set -euo pipefail

DRY_FLAG="--dry-run"
if [[ "${1:-}" == "--go" ]]; then
    DRY_FLAG=""
fi

REMOTE_HOST="${REMOTE_HOST:-cannon}"
REMOTE_ROOT="${REMOTE_ROOT:-/n/holystore01/LABS/hernquist_lab/Lab/rcordova/learning_to_autolens}"

LOCAL_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "${LOCAL_ROOT}"

CHECKPOINT_SOURCES=(
    "Modules/04_Search_Chaining_SLaM/output"
    "Modules/05_Pixelized_Source_Reconstructions/output"
    "Modules/09_MGE_Linear_Light_Profiles/output"
    "Solutions/output"
)

# macOS ships rsync 2.6.9, which lacks --mkpath. Create remote parents by hand.
echo "Creating remote parent dirs..."
REMOTE_MKDIRS=("${REMOTE_ROOT}/autolens_workspace_latest/dataset")
for src in "${CHECKPOINT_SOURCES[@]}"; do
    REMOTE_MKDIRS+=("${REMOTE_ROOT}/${src}")
done
ssh "${REMOTE_HOST}" mkdir -p "${REMOTE_MKDIRS[@]}"

echo
echo "=== 1. autolens_workspace_latest/dataset/ (for Module 09) ==="
rsync -avh ${DRY_FLAG} \
    autolens_workspace_latest/dataset/ \
    "${REMOTE_HOST}:${REMOTE_ROOT}/autolens_workspace_latest/dataset/"

echo
echo "=== 2. Preserved Nautilus checkpoints ==="
# For each source tree, sync only checkpoint.hdf5 files plus the directory
# structure that holds them, into the identical path on the remote.
#   --include='*/'           keep directories so traversal reaches the files
#   --include='checkpoint.hdf5'   keep the target file
#   --exclude='*'            drop everything else (samples, logs, etc.)
for src in "${CHECKPOINT_SOURCES[@]}"; do
    echo "--- ${src} ---"
    rsync -avh ${DRY_FLAG} \
        --include='*/' \
        --include='checkpoint.hdf5' \
        --exclude='*' \
        --prune-empty-dirs \
        "${src}/" \
        "${REMOTE_HOST}:${REMOTE_ROOT}/${src}/"
done

echo
if [[ -n "${DRY_FLAG}" ]]; then
    echo "DRY RUN complete. Re-run with --go to actually transfer."
else
    echo "Done. Cannon now has:"
    echo "  - autolens_workspace_latest/dataset/ (Mod 09 ready)"
    echo "  - Preserved checkpoint.hdf5 files under their original paths"
    echo "    (Nautilus will auto-resume on re-submit)"
    echo
    echo "Next: ssh ${REMOTE_HOST} and sbatch --export=ALL,MODULE=04 \\"
    echo "        Modules/10_Cluster_Computing/scripts/submit_cannon.slurm"
fi
