---
name: cluster-cannon
description: Move heavy compute from a laptop to Harvard FASRC Cannon (or any SLURM cluster). Use when the user wants to submit a batch job, sync a repo to the cluster, set up a conda env on Cannon, decide between $HOME / $SCRATCH / lab storage, or pull results back. Triggers on words like "cannon", "cluster", "slurm", "sbatch", "fasrc", "login.rc.fas.harvard.edu", or when a local fit/training run is too expensive to run on the laptop.
---

# Cluster workflow (Cannon / SLURM)

This skill captures how to move work from a laptop to **Harvard FASRC Cannon** (or any SLURM cluster with minor substitutions). It's repo-agnostic: apply it whether the workload is lens modeling, ML training, Monte Carlo, simulations, etc.

The golden rule: **develop locally, sync to cluster, submit via `sbatch`, write outputs to `$SCRATCH` or lab storage, pull a lightweight artifact bundle back.** Do not run multi-hour jobs on a laptop and do not write heavy output under `$HOME`.

---

## 1. Before you start — does this actually belong on the cluster?

Put it on the cluster if **any** of these are true:
- Single run >~30 min wall time, OR
- You want to run multiple in parallel, OR
- RAM footprint approaches your laptop's total, OR
- The workload will repeat (sweep, re-fit, resumed training).

If the run is <10 min and single-shot, just run it locally — the SSH/queue/sync overhead is not worth it.

When you decide to move to the cluster, **stop the local run cleanly** (send SIGTERM, not SIGKILL) so any checkpoint files (`checkpoint.hdf5`, PyTorch `.pt`, etc.) are preserved. Most samplers (Nautilus, dynesty, emcee with HDF5 backend) can resume from these on the cluster.

---

## 2. Storage layout on Cannon — the thing newcomers get wrong

Cannon has **several different filesystems** with very different properties. Using the wrong one is the #1 mistake.

| Location | Quota | Speed | Backup | Use for |
|----------|-------|-------|--------|---------|
| `$HOME` (`/n/home02/<user>/` or `/n/home01/...` etc.) | ~100 GB | slow | yes | code, configs, small artifacts. **Never put datasets or job output here.** |
| Lab `holylfs` (`/n/holylfs*/LABS/<pi_lab>/Users/<user>/`) | lab-dependent, often TB-scale | fast | usually yes | datasets, checkpoints, model weights, long-lived job output |
| Lab `holystore` (`/n/holystore*/LABS/<pi_lab>/`) | lab-dependent | medium | usually yes | bulk long-term storage |
| `$SCRATCH` / `/n/netscratch/...` | TB-scale | fast | **no, 90-day purge** | ephemeral intermediates; only if your lab doesn't give you holylfs |
| `/tmp` on compute node | ~100 GB, node-local | fastest | no, gone at job end | scratch within a single job (unpack archives, etc.) |
| Web-viewable dir (some labs) | per lab | n/a | n/a | `/n/holylfs*/LABS/<pi_lab>/Everyone/*/www/<user>/` → `https://faun.rc.fas.harvard.edu/<user>/` |

**Cannon 2.0 reality check:** `$SCRATCH` as a user env var is not consistently set on Cannon 2.0. The primary fast storage is your **lab's holylfs allotment** — discover it with `ls /n/holylfs*/LABS/` and pick the one your PI owns. Don't assume `$SCRATCH` is defined; reference lab paths explicitly.

**What this means in practice:**
- `REPO_ROOT=$HOME/<project>` — clone code here (small, git-tracked).
- `OUTPUT_ROOT=/n/holylfs*/LABS/<pi_lab>/Users/<user>/<project>/output` — write job outputs here, referenced by `--output-root` in your script.
- Publishable results — copy the small, final artifacts into the repo (git-tracked) or a web-viewable dir; leave the raw sampler state on lab storage.

Find your paths with: `echo $HOME`, `ls /n/holylfs*/LABS/`, `groups $USER`.

---

## 3. Conda environments — create fresh per project

Cannon 2.0 provides conda via a **shared Miniforge install**, not via `module load Anaconda3`. Don't try to load `Anaconda3/*` — it may still exist as a module but is out of favor and can conflict.

```bash
# On Cannon, after ssh — one-time env setup:
source /n/sw/Miniforge3-25.3.1-0/etc/profile.d/conda.sh
conda create -n <project_name> python=3.11 -y
conda activate <project_name>
pip install <deps>                           # or pip install -e .
```

In Slurm submit scripts, source Miniforge and `conda activate`:

```bash
source /n/sw/Miniforge3-25.3.1-0/etc/profile.d/conda.sh
conda activate <project_name>
```

**Gotchas:**
- **Never install into `base`** — it's shared and upgrades break everything.
- The Miniforge path above may version-bump over time. Check: `ls /n/sw/ | grep -i miniforge`.
- If you see mysterious errors after a cluster reboot or upgrade, re-check the Miniforge path and any `module avail` changes.
- You can pass a full env path instead of a name if the env lives outside the default location: `conda activate /n/home02/<user>/.conda/envs/<env>`.
- Separate env per project. They're cheap. Pin Python at env creation and key library versions in a committed `requirements.txt`/`environment.yml`.

To verify the env before submitting a job: `which python && python -c "import <key_pkg>; print(<key_pkg>.__version__)"`.

---

## 4. Login node vs compute node — do not run work on login

`login.rc.fas.harvard.edu` is a **shared login node**. Running multi-minute CPU work there is rude and will be killed by admins (or trigger a warning email). Use it only for:
- `git`, `rsync`, `ssh`, editing files
- Submitting jobs (`sbatch`)
- Quick inspection (`head`, `less`, `du -sh`)

For interactive compute, request an allocation:
```bash
# Interactive shell on a compute node (30 min, 4 CPUs, 8 GB):
salloc -p test --time=00:30:00 -n 1 -c 4 --mem=8G
# then you land on a compute node, run things normally, `exit` to release
```

For heavy batch work, use `sbatch`.

---

## 5. SLURM account — always specify one

**CRITICAL:** every `sbatch` / `salloc` must carry `#SBATCH --account=<group>`. Omitting it charges fairshare to the wrong group — which can tank priority for a PI you didn't intend to bill, or silently fail to start.

Discover your accounts: `sacctmgr show associations where user=$USER format=Account`.

Typical CfA-affiliated accounts:
- `hernquist_lab` — Hernquist group (CfA).
- `hernquist_lab` — Hernquist-lab subgroup; often has faster GPU scheduling via dedicated partitions.
- Others — ask your PI or check `sshare -u $USER`.

Rotate between accounts if you have multiple — `sshare -u $USER` shows fairshare per account and lets you pick the least-burned one.

## 6. Partition choice

Cannon partitions differ in time limits, availability, and preemption. Account and partition are coupled — some partitions require a specific account.

| Partition | Kind | Time limit | Notes |
|-----------|------|-----------|-------|
| `test` | CPU | 15 min | Quick sanity checks; starts fast. |
| `shared` | CPU | 3 days | General CPU default. Works with any account. |
| `serial_requeue` | CPU | 3 days | **Preemptible** — only use if your job checkpoints and resumes. |
| `bigmem` | CPU | 3 days | For jobs needing >256 GB RAM. |
| `gpu` | GPU | varies | Slow scheduling but guaranteed (no preemption). |
| `gpu_requeue` | GPU | varies | **Preemptible**. Faster queue at the cost of mid-run kills. |
| `gpu_test` | GPU | 1 hour | Quick GPU smoke tests, minimal fairshare cost. |
| `hernquist` | GPU | varies | Lab-specific; requires `--account=hernquist_lab`. Usually fastest for Hernquist-lab members. |
| Other `<lab>_gpu` / `<lab>_compute` | varies | varies | Your PI may own dedicated nodes — check with your group. |

Default: `shared` for CPU work, `gpu_test` for GPU smoke tests, your lab's dedicated partition for GPU production, `gpu_requeue` if you're comfortable with preemption.

Check live partition state: `sinfo -p shared`, `sinfo -p hernquist`.

---

## 7. Slurm submit script — a template

A reusable `submit_cannon.slurm` for a **CPU job**:

```bash
#!/bin/bash
#SBATCH --job-name=myjob
#SBATCH --partition=shared
#SBATCH --account=hernquist_lab             # CRITICAL — always specify, see §5
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your.email@harvard.edu

set -euo pipefail

# Account (who pays for compute) and storage (which lab's disk writes go to)
# can be different. Pick --account for speed/fairshare, pick OUTPUT_ROOT for
# whichever lab share has the space.
REPO_ROOT="${REPO_ROOT:-${HOME}/myproject}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/n/holystore02/LABS/<pi_lab>/Users/${USER}/myproject/output}"
CONDA_ENV="${CONDA_ENV:-myproject}"

source /n/sw/Miniforge3-25.3.1-0/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"

# Essential for live log tailing — without this, Python buffers stdout and
# logs appear empty until the job exits or flushes.
export PYTHONUNBUFFERED=1

mkdir -p logs "${OUTPUT_ROOT}"
cd "${REPO_ROOT}"

# Diagnostics — cheap to always print, invaluable for debugging mismatches.
echo "=== Job $SLURM_JOB_ID on $(hostname) at $(date -Is) ==="
echo "Python: $(which python) ($(python --version 2>&1))"
echo "Account: ${SLURM_JOB_ACCOUNT}    Partition: ${SLURM_JOB_PARTITION}"
echo "CPUs:    ${SLURM_CPUS_PER_TASK}  Mem: ${SLURM_MEM_PER_NODE:-?} MB"
echo "Output:  ${OUTPUT_ROOT}"
echo "==================================================================="

srun python run.py \
    --repo-root   "${REPO_ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    --n-cpus      "${SLURM_CPUS_PER_TASK}"

echo "=== Done at $(date -Is) ==="
```

For **GPU work**, additionally add:
```bash
#SBATCH --partition=hernquist            # or gpu, gpu_requeue, gpu_test
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
# ... after activate:
module load cuda/12.4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

**Principles embedded in this template:**
- `set -euo pipefail` — fail fast; don't silently continue past errors.
- `--account` is non-optional (see §5). Use `SLURM_JOB_ACCOUNT` in diagnostics so the log tells you who got billed.
- Parameterize paths via env vars (`REPO_ROOT`, `OUTPUT_ROOT`, `CONDA_ENV`) so the same script works across users/machines.
- Log to `logs/%x_%j.out` where `%x` is job name, `%j` is job ID. Name logs so you can find them later.
- Read `$SLURM_CPUS_PER_TASK` in the worker so it parallelizes to whatever Slurm gave you, not a hard-coded number.
- `srun python ...` rather than bare `python ...` — srun gives proper process accounting and signal forwarding.
- `PYTHONUNBUFFERED=1` for live `tail -f` on the log file.
- Source Miniforge + `conda activate`; do **not** `module load Anaconda3` on Cannon 2.0 (§3).

**Resource sizing:** request headroom but not 2×. If your job uses 28 GB, request 32 GB; if it uses 10 CPUs, request 12. Over-requesting hurts your queue priority. Check after a job finishes: `seff <jobid>` shows actual vs. requested.

---

## 8. Rsync pattern — the two-direction dance

### Push (laptop → Cannon)

Write a small wrapper script. Dry-run by default, `--go` for real.

```bash
#!/usr/bin/env bash
set -euo pipefail

CANNON_USER="${CANNON_USER:-your_fasrc_username}"
CANNON_HOST="${CANNON_HOST:-login.rc.fas.harvard.edu}"
CANNON_DEST="${CANNON_DEST:-myproject}"
LOCAL_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

DRY_FLAG="--dry-run"
[[ "${1:-}" == "--go" ]] && DRY_FLAG=""

rsync -avh --progress ${DRY_FLAG} \
    --exclude='.git/' \
    --exclude='.venv/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='.DS_Store' \
    --exclude='.ipynb_checkpoints/' \
    --exclude='*.egg-info/' \
    --exclude='Output/' \
    "${LOCAL_ROOT}/" \
    "${CANNON_USER}@${CANNON_HOST}:~/${CANNON_DEST}/"
```

**Include/exclude tips:**
- `.git/` is huge and useless on the cluster unless you're committing *from* there. Exclude unless needed.
- Excluding a subdir but including one file under it requires `--include` ordered **before** `--exclude`: `--include='workspace/' --include='workspace/dataset/***' --exclude='workspace/*'`.
- Preserve checkpoints! Don't exclude `*.hdf5`, `*.pkl`, `checkpoint*` — they're what lets jobs resume.
- Heavy datasets: if >1 GB, consider staging them once via `scp` or directly to lab storage rather than round-tripping with rsync.

### Pull (Cannon → laptop) — lightweight artifacts only

Don't pull raw sampler output (can be hundreds of MB per search). Have your job script emit a small `results/` directory with PDFs, JSON summaries, and CSV samples, then pull *that*.

```bash
rsync -avh --progress \
    "${CANNON_USER}@${CANNON_HOST}:~/${CANNON_DEST}/results/" \
    "${LOCAL_ROOT}/cluster_results/"
```

---

## 9. Checkpoint + resume — why `serial_requeue` is your friend

Any sampler/trainer that writes `checkpoint.hdf5` (or equivalent) and can reload from it should use `--requeue` and run on preemptible partitions. Slurm will checkpoint-ready jobs restart cleanly on preemption, so you often get faster queue times without losing progress.

```
#SBATCH --partition=serial_requeue
#SBATCH --requeue
#SBATCH --time=3-00:00:00
```

Verify resume works once locally before trusting it on a preempting partition.

---

## 10. Monitoring a running job

```bash
squeue --me                       # your jobs, running + pending
squeue -u $USER -t RUNNING        # just running
sstat -j <jobid> --format=JobID,AveCPU,MaxRSS   # live resource use
tail -f logs/myjob_<jobid>.out   # stream stdout
scancel <jobid>                   # cancel one
scancel -u $USER                  # cancel ALL your jobs (careful)
seff <jobid>                      # efficiency report (after completion)
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS,ExitCode
```

Set `--mail-type=BEGIN,END,FAIL --mail-user=you@harvard.edu` so you get emailed at state transitions — skip the manual refresh loop.

---

## 11. First-time setup checklist for a new user

If the user doesn't have Cannon access yet, point them here:
1. **FASRC account:** https://portal.rc.fas.harvard.edu/request/account/new — lab sponsor required.
2. **2FA:** FASRC requires OpenAuth; the account page walks through it.
3. **SSH key:** `ssh-keygen -t ed25519 -C "you@harvard.edu"`, then `ssh-copy-id user@login.rc.fas.harvard.edu`. After this, `ssh` still prompts for 2FA but not a password.
4. **Optional but strongly recommended:** configure `~/.ssh/config` on the laptop with `ControlMaster auto`, `ControlPath ~/.ssh/cm-%r@%h:%p`, `ControlPersist 10m` — avoids re-2FA on every rsync.
5. **Docs:** https://docs.rc.fas.harvard.edu/ — FASRC's own reference is good.

---

## 12. Common failure modes and quick diagnoses

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Job pending forever | requested too much time/mem, or partition full | Check `sinfo -p <part>`, reduce request, try `serial_requeue` / `gpu_requeue` |
| Fairshare priority tanking | billed the wrong account (no `--account`, or heavy recent use) | Always set `#SBATCH --account=<group>`; check `sshare -u $USER` |
| `conda: command not found` in job | Miniforge not sourced | `source /n/sw/Miniforge3-25.3.1-0/etc/profile.d/conda.sh` before `conda activate` |
| `ModuleNotFoundError` but works interactively | wrong env activated in job | Echo `which python` early; verify env name matches login node's |
| Logs empty during run | Python stdout buffered | Set `export PYTHONUNBUFFERED=1` (or run `python -u`) |
| `Disk quota exceeded` on `$HOME` | writing output under `$HOME` | Move `OUTPUT_ROOT` to lab storage (`/n/holystore*/LABS/<pi>/Users/$USER/...`) |
| Output disappears after ~3 months | `$SCRATCH` / netscratch purge | Copy results to holystore or `$HOME` promptly |
| Job gets preempted repeatedly | on `*_requeue` without resume logic | Switch to non-requeue partition, or make the sampler checkpoint |
| `srun: error: Unable to allocate resources` | cluster congestion or wrong partition | Wait / try another partition / `sinfo` to inspect |
| OOM killed | `--mem` too low, or delayed OOM from long input | Rerun with higher `--mem`; `seff` shows actual use; smoke-test long inputs |
| "Slurm job not submitted" silently | `#SBATCH` typo below first real line | `#SBATCH` directives must be **before** any non-comment/non-`#SBATCH` line |
| GPU is smaller than requested | asked for "any A100" but got a 40 GB one | Use `--gres=gpu:nvidia_a100-sxm4-80gb:1` for 80 GB specifically |
| `device_map="auto"` training crash | cross-device tensor mismatch in loss | Works for inference only; use DeepSpeed ZeRO-3 for multi-GPU training |
| SQLite `database is locked` / corruption | concurrent access on NFS/Lustre | Precompute to JSON/Parquet; don't run parallel jobs against one SQLite file |
| `logs/` write error, silent refusal to start | output dir doesn't exist | `mkdir -p logs` before `sbatch` (or inside the script, early) |
| HF `ConnectionError` from compute node | compute nodes may lack internet | Download model on login node first; set `HF_HUB_OFFLINE=1` in job |
| 2FA re-prompt on every rsync | no SSH multiplex | Set `ControlMaster auto` + `ControlPersist 10m` in `~/.ssh/config` (§11 #4) |

---

## 13. Results-viewer pattern — make finished work shareable

After the job finishes, have a post-processor emit a small `results/` directory with:
- `summary.json` — key numbers (max log likelihood, evidence, final loss, metric).
- A handful of PDFs/PNGs — corner plot, residual plot, training curve.
- Optionally a trimmed `samples.csv`.

This bundle is small enough to **commit to git**. New collaborators who clone the repo can see the finished result without needing cluster access. Raw sampler state stays on `$SCRATCH`; only the distilled artifacts cross the network back to the laptop and into the repo.

This pattern is what `Modules/10_Cluster_Computing/scripts/export_results.py` does in the Learning_to_Autolens repo — use it as a concrete reference.

---

## 14. When helping a user adapt a repo to Cannon

1. **Ask what's heavy.** Which step is multi-hour or multi-GB? That's what goes on the cluster — the rest stays on the laptop.
2. **Extract the heavy step into a standalone script.** Notebooks don't run on compute nodes. If the logic lives in a notebook, factor it into a `.py` with CLI args (`--repo-root`, `--output-root`, `--dataset-root`, any tunables).
3. **Make it parameterize from env:** read `$SLURM_CPUS_PER_TASK`, lab-storage paths, so the same script runs in different Slurm configurations without editing.
4. **Add a `submit_cannon.slurm`** like §7 — parameterize by `MODULE` or `CONFIG` if you want one script for multiple workloads.
5. **Add `push_to_cannon.sh` / `pull_from_cannon.sh`** wrappers with dry-run defaults.
6. **Add a post-processor** that emits lightweight artifacts (§13).
7. **Write a short README section** pointing users at the scripts. Include: how to request FASRC access, where outputs land, how to resume a preempted job.
8. **Commit Nautilus/sampler checkpoints** if they're small (<100 MB) — lets fresh clones on the cluster auto-resume in-progress runs.

Keep the scripts in one directory (e.g. `Cluster_Computing/scripts/` or `cluster/`) so they're easy to find.

---

## 15. Multiple repos — keep them isolated on the cluster

Most users end up with several projects on Cannon simultaneously (ML training, lens modeling, a side project). They must not share state.

**Per-repo isolation rules:**
- **One conda env per repo.** `conda create -n <repo_name> ...` — never reuse an env across repos. Dependencies will fight.
- **One `$HOME/<repo>` clone per repo.** Don't symlink shared source dirs between repos — it breaks `git` and makes "which version is running" unanswerable.
- **One `OUTPUT_ROOT/<repo>/` tree per repo.** Your Slurm script's `OUTPUT_ROOT` must name the repo. Otherwise two jobs clobber each other's `output/` dirs.
- **One lab-storage subdir per repo** for long-lived artifacts: `/n/holystore*/LABS/<pi>/Users/<user>/<repo>/`.
- **Account ≠ storage.** You can bill compute to one account (`--account=hernquist_lab` for fast scheduling) while writing output under a different lab's disk (`/n/holystore02/LABS/hernquist_lab/...` because that's where the space is). Pick each independently.

Put the repo name in the Slurm `--job-name` (`#SBATCH --job-name=<repo>-<task>`) so `squeue --me` and email subject lines are unambiguous.

---

## 16. Syncing config across machines via git — the pattern

The reliable way to keep shared config (skills, scripts, templates, dotfiles) in sync between a laptop and the cluster is **git, not rsync**. Rsync is one-way; git is bidirectional with a conflict-resolution story.

**The pattern:**
1. Decide what's shareable per-repo: `.claude/skills/` (custom Claude skills), `cluster/` scripts, shared templates.
2. Track those paths in the repo's `.gitignore` with an allowlist — e.g.:
   ```
   .claude/*
   !.claude/skills/
   ```
   This keeps `settings.local.json` and other per-machine state ignored while letting `skills/` travel.
3. Commit on whichever machine you edited, push, and pull on the other. The commit history becomes the changelog.
4. Prefer the **project-scoped skill** (`<repo>/.claude/skills/<name>/`) over the user-scoped skill (`~/.claude/skills/<name>/`) when the skill should ride with the repo — it'll sync automatically via git.
5. Copy a skill up to `~/.claude/skills/` only when you want it available across *all* repos on one machine.

**Before pulling skill/config changes the other machine pushed, check:**
- Are there unpushed commits on your side that touch the same files? → expect a merge or rebase, not a fast-forward.
- Are there uncommitted working-tree changes? → stash or commit first.
- Does the pushed commit sit on a base you have? If the remote did force-pushes or history rewrites, the pull will be non-trivial.

A clean sync is `git pull --ff-only`. If that fails, stop and look — don't auto-merge.

---

## 17. Handling cross-machine instructions (cluster-side Claude ↔ laptop-side Claude)

When you receive an instruction that was drafted on the other machine (typical form: "I pushed X, please pull and verify"), before executing anything stateful:

1. **Verify the target repo matches the current working directory.** The sending machine referenced its own path (`~/path/to/<repo>/`). Confirm that repo name matches the current session's cwd. If not, don't `cd` and execute — surface the mismatch.
2. **Fetch first, don't pull.** `git fetch origin` gives you the remote refs without touching your working tree. Inspect `git log HEAD..origin/main` and `git diff HEAD origin/main` before merging.
3. **Check local state isn't ahead or dirty.** `git status` and `git log origin/main..HEAD`. If local has unpushed commits or uncommitted changes, a pull will merge/rebase — flag this before proceeding.
4. **Only then pull.** Prefer `git pull --ff-only`; fall back to `git pull --rebase` or an explicit merge only after you've understood the divergence.
5. **If skills were overwritten:** the original sender warned to check for lost local refinements. Do a `git log -p -- .claude/skills/<name>/` to see what the incoming commits changed, compared against what the local side had.

This is the same "measure twice, cut once" principle the top-level instructions describe, applied to cross-machine coordination. The cost of pausing to confirm is low; the cost of silently merging away two days of local edits is high.

---

## 18. HuggingFace / network-dependent tools on compute nodes

Compute nodes on Cannon **may not have outbound internet**. Anything that hits a remote URL from inside a job — HuggingFace Hub, `pip install`, `git clone`, `wget` — is likely to fail mid-run with a `ConnectionError` or silently hang.

**Workflow:**
1. On the **login node** (which has internet), pre-download the model/data once:
   ```bash
   python -c "from transformers import AutoModelForCausalLM; \
              AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.3-70B-Instruct')"
   ```
2. In the Slurm submit script, force offline mode so the job doesn't attempt a refresh:
   ```bash
   export HF_HOME="${HOME}/.cache/huggingface"     # or lab-storage path for big caches
   export HF_HUB_OFFLINE=1
   export TRANSFORMERS_OFFLINE=1
   ```
3. For `pip install`: don't. All dependencies belong in the env that was set up on the login node.

The `HF_HOME` cache can grow to tens of GB for large models — if `$HOME` is tight, relocate it under lab storage.

---

## 19. SQLite / shared-filesystem pitfall

SQLite file-locking is unreliable on networked filesystems (NFS / Lustre). Running multiple parallel jobs that touch the same `.sqlite3` / `.db` file almost always corrupts it.

This bites: ChromaDB, MLflow, DVC, any tool with an embedded SQLite backend.

**Fixes, in order of preference:**
1. **Precompute into a flat file.** Extract what the parallel jobs need into a JSON/Parquet/NPZ on the login node, then shard over that read-only file.
2. **Copy to node-local `/tmp` per job.** Each job starts by `cp db.sqlite3 /tmp/` and works there; merge at the end. Only safe if jobs are independent.
3. **Swap backend.** PostgreSQL on a shared DB server, or just flat files, for anything multi-writer.

Never: "it'll probably be fine if I run 8 reader jobs". It won't.

---

## 20. Auto-commit pattern for unattended jobs

For long jobs that produce small, trackable artifacts (JSON summaries, PDFs, CSVs), commit them from inside the Slurm script so results land in git without manual intervention:

```bash
# At the end of the submit script, after the work finishes:
cd "${REPO_ROOT}"
if [ -n "$(git status --porcelain results/)" ]; then
    git add results/
    git commit -m "Job $SLURM_JOB_ID results ($(date +%Y-%m-%d))" \
        --author="Cluster Job <noreply@fas.harvard.edu>" || true
    git push || echo "WARNING: git push failed (compute node may lack network)"
fi
```

- `|| true` on commit handles the "nothing to commit" case.
- `--author` distinguishes automated commits from manual ones in `git log`.
- `|| echo` on push: compute nodes may have no outbound network, in which case the commit is made locally and you can push from the login node later.
- Only commit **lightweight** artifacts. Don't commit raw sampler state, model weights, or anything >10 MB — they belong on lab storage.
- Don't auto-commit to `main` if others are actively working there. Use a per-job branch if collisions are likely.
