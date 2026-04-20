#!/usr/bin/env bash
# pull_from_cannon.sh — two-step pull from Cannon:
#
#   1. Lightweight, git-trackable artifacts (Modules/XX_*/results/*)
#      produced by export_results.py at the end of the Slurm job —
#      fit_subplot.png, corner.pdf, summary.json, model_results.txt,
#      samples.csv. These are the files your module notebook's results-
#      viewer cell reads.
#
#   2. Raw Nautilus output tree (optional — skipped by default).
#      Only needed if you want to poke at checkpoint.hdf5 or re-export
#      with different artifact choices locally. Large (hundreds of MB
#      per search) so enable with --include-raw.
#
# Usage:
#   ./pull_from_cannon.sh                   # dry run, artifacts only
#   ./pull_from_cannon.sh --go              # pull artifacts
#   ./pull_from_cannon.sh --go --include-raw  # artifacts + raw output
#
# Uses the SSH alias `cannon` by default (expected in ~/.ssh/config with
# ControlMaster auto + ControlPath for one-time Duo auth per session).
# Override via env vars:
#   CANNON_SSH=cannon                        # ssh alias (default)
#   CANNON_USER=rcordova                     # only for path construction
#   CANNON_REPO_ROOT=/n/.../learning_to_autolens
#   CANNON_SCRATCH=<same>/output
#     ./pull_from_cannon.sh --go

set -euo pipefail

# Source user-specific overrides if a cannon.env exists alongside this
# script (see ../cannon.env.example).
_CANNON_ENV="$(cd "$(dirname "$0")" && pwd)/../cannon.env"
if [[ -f "${_CANNON_ENV}" ]]; then
    # shellcheck source=/dev/null
    source "${_CANNON_ENV}"
fi

CANNON_SSH="${CANNON_SSH:-cannon}"       # SSH alias (see ~/.ssh/config)
CANNON_USER="${CANNON_USER:-rcordova}"   # only for building the remote path
# Default matches submit_cannon.slurm's REPO_ROOT / OUTPUT_ROOT defaults.
# The results/ artifacts live under the repo; raw Nautilus output lives
# alongside under output/ (both on holystore, not holyscratch).
CANNON_REPO_ROOT="${CANNON_REPO_ROOT:-/n/holystore01/LABS/hernquist_lab/Lab/${CANNON_USER}/learning_to_autolens}"
CANNON_SCRATCH="${CANNON_SCRATCH:-${CANNON_REPO_ROOT}/output}"

LOCAL_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
LOCAL_RAW_DEST="${LOCAL_ROOT}/Modules/10_Cluster_Computing/cannon_output"

DRY_FLAG="--dry-run"
INCLUDE_RAW="0"
for arg in "$@"; do
    case "$arg" in
        --go)          DRY_FLAG="" ;;
        --include-raw) INCLUDE_RAW="1" ;;
        *) echo "Unknown arg: $arg"; exit 2 ;;
    esac
done

# ---- Step 1: Lightweight results/ artifacts --------------------------------
# rsync's include/exclude semantics: we want Modules/*/results/** and
# nothing else under Modules/. The trailing '***' matches everything
# under results/ recursively. The final '*' exclude drops unmatched paths.
echo "===================================================================="
echo "  Step 1: pulling Modules/*/results/ artifacts (git-trackable)"
echo "===================================================================="
rsync -avh --progress ${DRY_FLAG} \
    --include='Modules/' \
    --include='Modules/*/' \
    --include='Modules/*/results/' \
    --include='Modules/*/results/***' \
    --exclude='*' \
    "${CANNON_SSH}:${CANNON_REPO_ROOT}/" \
    "${LOCAL_ROOT}/"

# ---- Step 2: Raw Nautilus output (optional) --------------------------------
if [[ "${INCLUDE_RAW}" == "1" ]]; then
    echo
    echo "===================================================================="
    echo "  Step 2: pulling raw Nautilus output tree (large — several hundred MB)"
    echo "===================================================================="
    mkdir -p "${LOCAL_RAW_DEST}"
    rsync -avh --progress ${DRY_FLAG} \
        "${CANNON_SSH}:${CANNON_SCRATCH}/" \
        "${LOCAL_RAW_DEST}/"
fi

if [[ -n "${DRY_FLAG}" ]]; then
    echo
    echo "This was a DRY RUN. Re-run with --go to actually transfer."
    echo "Add --include-raw to also pull the full Nautilus output tree."
else
    echo
    echo "Pulled results → ${LOCAL_ROOT}/Modules/*/results/"
    if [[ "${INCLUDE_RAW}" == "1" ]]; then
        echo "Pulled raw Nautilus output → ${LOCAL_RAW_DEST}/"
    fi
    echo
    echo "Next: open the module notebook and run the results-viewer cell."
fi
