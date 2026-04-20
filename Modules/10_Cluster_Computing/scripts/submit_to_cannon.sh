#!/usr/bin/env bash
# submit_to_cannon.sh — one-command cluster submission.
#
# Does what "push + ssh + sbatch" does, but verifies every step so a
# stale-hash re-run can't silently happen:
#
#   1. Warn about uncommitted / unpushed git changes (optional but loud).
#   2. Run push_to_cannon.sh --go to rsync the current tree to Cannon.
#   3. Compute SHA256 of the local fit_module${MODULE}.py and the remote
#      copy. Abort if they differ (rsync failure, permission issue, etc.).
#   4. ssh cannon and sbatch with --export=ALL,MODULE=${MODULE}.
#   5. Print the returned job ID + squeue / tail commands.
#
# Usage:
#   ./submit_to_cannon.sh 04
#   ./submit_to_cannon.sh 04 --mem 64G --time 48:00:00     # extra sbatch args
#   SKIP_PUSH=1 ./submit_to_cannon.sh 04                    # already pushed
#
# Env overrides (same defaults as push_to_cannon.sh):
#   CANNON_USER      (default: rcordova)
#   CANNON_HOST      (default: login.rc.fas.harvard.edu)
#   CANNON_DEST      (default: learning_to_autolens — repo dir on Cannon,
#                     under the SUBMIT_ROOT you're rsyncing to)

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <MODULE> [extra sbatch args]"
    echo "       e.g.  $0 04"
    echo "             $0 09 --mem 64G --time 48:00:00"
    exit 2
fi

MODULE="$1"; shift
EXTRA_SBATCH=("$@")

CANNON_USER="${CANNON_USER:-rcordova}"
CANNON_HOST="${CANNON_HOST:-login.rc.fas.harvard.edu}"
CANNON_DEST="${CANNON_DEST:-learning_to_autolens}"

LOCAL_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
LOCAL_FIT="${LOCAL_ROOT}/Modules/10_Cluster_Computing/scripts/fit_module${MODULE}.py"
LOCAL_SLURM="${LOCAL_ROOT}/Modules/10_Cluster_Computing/scripts/submit_cannon.slurm"
REMOTE_REPO="/n/holystore01/LABS/hernquist_lab/Lab/${CANNON_USER}/learning_to_autolens"
REMOTE_FIT="${REMOTE_REPO}/Modules/10_Cluster_Computing/scripts/fit_module${MODULE}.py"

# --- 0. Sanity --------------------------------------------------------------
if [[ ! -f "${LOCAL_FIT}" ]]; then
    echo "error: ${LOCAL_FIT} does not exist."
    echo "       Did you forget to run:"
    echo "         cp ${LOCAL_ROOT}/Modules/10_Cluster_Computing/scripts/fit_template.py \\"
    echo "            ${LOCAL_FIT}"
    exit 2
fi

# --- 1. Git state -----------------------------------------------------------
cd "${LOCAL_ROOT}"
if git rev-parse --git-dir > /dev/null 2>&1; then
    # Working directory
    if ! git diff-index --quiet HEAD -- Modules/10_Cluster_Computing/scripts/ 2>/dev/null; then
        echo "WARNING: uncommitted changes under Modules/10_Cluster_Computing/scripts/:"
        git -c color.ui=always status --short Modules/10_Cluster_Computing/scripts/
        echo
        echo "  The rsync will still push them, but the HEAD commit won't match the"
        echo "  file that ran on Cannon. Commit first for a reproducible run:"
        echo "    git add Modules/10_Cluster_Computing/scripts/ && git commit -m '...'"
        echo
        read -r -p "  Continue with uncommitted changes? [y/N] " reply
        [[ "${reply,,}" == "y" ]] || exit 1
    fi
    HEAD_COMMIT=$(git rev-parse --short HEAD)
    echo "[submit] Local HEAD: ${HEAD_COMMIT}"
fi

# --- 2. Push ----------------------------------------------------------------
if [[ -z "${SKIP_PUSH:-}" ]]; then
    echo "[submit] Running push_to_cannon.sh --go ..."
    bash "${LOCAL_ROOT}/Modules/10_Cluster_Computing/scripts/push_to_cannon.sh" --go \
        > /tmp/push_to_cannon.log 2>&1 \
        || { echo "error: push_to_cannon.sh failed — see /tmp/push_to_cannon.log"; exit 1; }
    echo "[submit] Push complete (log: /tmp/push_to_cannon.log)"
else
    echo "[submit] SKIP_PUSH=1 — skipping push_to_cannon.sh"
fi

# --- 3. Verify fit_module script hashes match -------------------------------
LOCAL_HASH=$(shasum -a 256 "${LOCAL_FIT}" | awk '{print $1}')
echo "[submit] Local  fit_module${MODULE}.py SHA256: ${LOCAL_HASH}"
REMOTE_HASH=$(ssh -o BatchMode=no "${CANNON_USER}@${CANNON_HOST}" \
    "sha256sum ${REMOTE_FIT} 2>/dev/null | awk '{print \$1}'" \
    || { echo "error: could not ssh to ${CANNON_HOST}"; exit 1; })
echo "[submit] Remote fit_module${MODULE}.py SHA256: ${REMOTE_HASH}"

if [[ "${LOCAL_HASH}" != "${REMOTE_HASH}" ]]; then
    echo
    echo "ERROR: local and remote fit_module${MODULE}.py DIFFER. Aborting before sbatch."
    echo "       Rerun push_to_cannon.sh --go and check rsync for errors, then retry."
    exit 1
fi
echo "[submit] Hashes match → fit script is in sync."

# --- 4. sbatch --------------------------------------------------------------
SBATCH_CMD="cd ${CANNON_DEST:-learning_to_autolens} && \
    sbatch --export=ALL,MODULE=${MODULE} --job-name=mod${MODULE} \
           ${EXTRA_SBATCH[@]:-} \
           Modules/10_Cluster_Computing/scripts/submit_cannon.slurm"
echo "[submit] Remote command: ${SBATCH_CMD}"
SBATCH_OUT=$(ssh -o BatchMode=no "${CANNON_USER}@${CANNON_HOST}" "${SBATCH_CMD}")
echo "[submit] ${SBATCH_OUT}"
JOB_ID=$(echo "${SBATCH_OUT}" | grep -oE '[0-9]+' | head -1)

# --- 5. Summary -------------------------------------------------------------
cat <<EOF

============================================================
  Job submitted: ${JOB_ID}
  Module:        ${MODULE}
  Local HEAD:    ${HEAD_COMMIT:-(not a git repo)}
  Script hash:   ${LOCAL_HASH:0:12}...
============================================================

Next:
  Monitor:    ssh ${CANNON_USER}@${CANNON_HOST} "squeue -j ${JOB_ID}"
  Tail log:   ssh ${CANNON_USER}@${CANNON_HOST} "tail -f ${REMOTE_REPO}/logs/mod${MODULE}_${JOB_ID}.out"
  Cancel:     ssh ${CANNON_USER}@${CANNON_HOST} "scancel ${JOB_ID}"
  Pull back:  bash Modules/10_Cluster_Computing/scripts/pull_from_cannon.sh --go

EOF
