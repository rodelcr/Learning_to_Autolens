# Student Quickstart — Cannon Workflow Cheat Sheet

**Audience:** a student who has finished `SETUP_NEW_USER.md` and is now running fits day-to-day. This is the cheat sheet for daily operations, no AI required.

If you haven't done one-time setup yet, see [`SETUP_NEW_USER.md`](SETUP_NEW_USER.md) first (~30-60 min).

---

## The 4-step daily loop

```
LAPTOP                CANNON                LAPTOP
edit code  ─push─►   sbatch  ─wait─►       pull ─►  audit notebook
```

```bash
# 1. PUSH your local edits to Cannon
bash Modules/10_Cluster_Computing/scripts/push_to_cannon.sh --go

# 2. SUBMIT a job (see RECIPES.md for the full menu)
ssh cannon \
  "cd \$CANNON_REPO_ROOT && \
   sbatch --time=24:00:00 \
     --export=ALL,EXAMPLE=group_scale,FIT_EXTRA_ARGS=--part=truth_anchored \
     Modules/10_Cluster_Computing/scripts/submit_cannon.slurm"

# 3. CHECK queue status (anytime; will show RUNNING / PENDING / nothing)
ssh cannon "squeue -u \$USER -o '%i %j %T %M' --noheader"

# 4. PULL results once the job lands
bash Modules/10_Cluster_Computing/scripts/pull_from_cannon.sh --go
```

That's it. The rest of this document explains each piece.

---

## Environment on Cannon — what you activate, what you avoid

**Use this env, every time:**
```bash
conda activate autolens312
```

This is the **only working environment** on Cannon for this project as of 2026-05. It has:
- Python 3.12
- autolens ≥ 2026.4.13.6 (the modern API; older versions are missing classes our scripts use)
- nautilus-sampler (the default sampler — replaces dynesty in modern autolens)
- All transitive deps (autoarray / autoconf / autofit / autogalaxy)

**DO NOT use:**
- The old `autolens` env (Python 3.11, autolens 2026.2.x — too old, missing `RectangularAdaptDensity` / `RectangularAdaptImage` / `reg.Adapt`).
- A bare `pip install` from inside an env — there's a stray `~/.local/bin/pip` (Python 3.10) that shadows the env's pip in PATH. It silently installs into `~/.local/lib/python3.10/site-packages/`, which the env can't see. **Always use `python -m pip install ...` instead.**

**If you see** `AttributeError: module 'autolens' has no attribute 'X'`: you're probably on the old env. Run `which python` and confirm it's `/.../envs/autolens312/bin/python`.

---

## What "submitting a job" actually does

The slurm script `submit_cannon.slurm` is the single entry point. Its job is:

1. Source `cannon.env` (your account/storage/partition settings — see SETUP_NEW_USER §2).
2. Source `~/.bashrc` so conda is on the PATH; then `conda activate $CONDA_ENV`.
3. Pin `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, etc. so Nautilus's Python multiprocessing doesn't fight BLAS thread-pool deadlocks.
4. Pick a fit-script + dataset-root based on `MODULE=...` or `EXAMPLE=...` env var (see [`RECIPES.md`](RECIPES.md) for the full table).
5. Run the fit script on `$SLURM_CPUS_PER_TASK` cores.
6. Post-process: copy lightweight artifacts (`fit_subplot.png`, `summary.json`, `model_results.txt`, etc.) into `Modules/*/results/` or `Examples/*/results/` so a `pull_from_cannon.sh` round-trip lets you view them on the laptop.

**Both `MODULE` and `EXAMPLE` work** — `MODULE=04` runs the curriculum's Module 04 fit; `EXAMPLE=group_scale` runs the example fit. They're routed to different driver scripts.

---

## Mental model: where files live

| What | Laptop | Cannon |
|---|---|---|
| Source code (this repo) | `~/Documents/AGEL/Learning_to_Autolens/` | `$CANNON_REPO_ROOT` (set in `cannon.env`) |
| Cannon-side fit output | — | `$CANNON_REPO_ROOT/output/...` (~hundreds of MB per fit) |
| Lightweight artifacts | `Modules/*/results/`, `Examples/*/results/` | same path, mirrored |
| Slurm logs | — | `$CANNON_REPO_ROOT/logs/<jobname>_<jobid>.out` |

`push_to_cannon.sh` rsyncs **laptop → cannon** (excluding `output/`).
`pull_from_cannon.sh` rsyncs **cannon → laptop** (only `results/` artifacts and slurm logs, never the heavy `output/`).

This means: **you commit edits + `Modules/*/results/` + `Examples/*/results/`** to git. The full `output/` tree never leaves Cannon.

---

## Reading the queue + logs

```bash
# What's running for me?
ssh cannon "squeue -u \$USER -o '%i %j %T %M' --noheader"
# columns: JobID JobName State(RUNNING/PENDING) Elapsed(HH:MM:SS)

# What partition am I on?
ssh cannon "scontrol show job <JOBID> | grep -E 'Partition|Reason|StdOut'"

# Watch a job's progress (Nautilus prints status every 50 likelihoods)
ssh cannon "tail -f \$CANNON_REPO_ROOT/logs/<jobname>_<jobid>.out"
```

**Good progress signals:** `f_live` dropping (1.0 → 0.1 → 0.01 → 0); `N_eff` rising; `log_Z` increasing then stabilising.

**Stall signals:** `f_live=1.0` for hours (the chain hasn't found high-likelihood basin); `log_Z` flat for hours after Stage transitions; very low `bound` count for many minutes.

**Convergence:** Nautilus considers the fit done when `f_live < 0.01` (i.e. less than 1% of evidence is in live points). Then it produces `samples_summary.json` and the slurm post-process runs.

---

## When a job fails

| Symptom | Meaning | Action |
|---|---|---|
| `TIMEOUT` after `--time` budget | Fit didn't converge in time | Increase budget (`--time=48:00:00`) and resubmit — Nautilus auto-resumes from `output/.../search_internal/` if same `path_prefix`+`unique_tag` |
| `OUT_OF_MEMORY` | Too many cores × too much memory each | Lower `SLURM_CPUS_PER_TASK` (set per-job: `--export=ALL,SLURM_CPUS_PER_TASK=16`) or raise `--mem=128G` |
| `FAILED` (exit code != 0) | Python error in the fit script | `tail logs/<jobname>_<jobid>.err` for the traceback; common causes: missing dataset, prior-construction bug, autolens version mismatch |
| `CANCELLED+` | You killed it with `scancel <jobid>` | (intentional) |

---

## Hands-on: your first end-to-end run

If you've never done the round-trip before, this 5-minute walk-through gives you confidence everything works.

```bash
# 1. Pick a small known-good example (Module 04 with the test SLaM mock)
cd ~/Documents/AGEL/Learning_to_Autolens

# 2. Push your repo to Cannon
bash Modules/10_Cluster_Computing/scripts/push_to_cannon.sh --go
# (~10s — rsyncs only changed files)

# 3. (First time only) seed the test datasets onto Cannon
bash Modules/10_Cluster_Computing/scripts/seed_cannon_data.sh --go

# 4. Submit Module 04 fit
ssh cannon \
  "cd \$CANNON_REPO_ROOT && \
   sbatch --export=ALL,MODULE=04 \
     Modules/10_Cluster_Computing/scripts/submit_cannon.slurm"
# Returns "Submitted batch job <id>"

# 5. Watch it
ssh cannon "tail -f \$CANNON_REPO_ROOT/logs/autolens_<id>.out"
# Wait until you see "f_live=0" or chi^2 stabilises. ~1-2h on first run.

# 6. Pull artifacts back
bash Modules/10_Cluster_Computing/scripts/pull_from_cannon.sh --go

# 7. Look at the result
open Modules/04_Search_Chaining_SLaM/results/slam/mass_total[1]/fit_subplot.png
# Expect: salt-and-pepper Normalized Residual Map, peak < 5σ, chi²/N near 1.
```

If steps 1-6 work, your full pipeline is operational. Move on to RECIPES.md to see the full menu.

---

## When in doubt

1. **Status hasn't updated in 5 minutes.** Probably normal — Nautilus prints every ~50 likelihoods which takes seconds-to-minutes per likelihood depending on model size. Check `tail -100` on the log.
2. **Job died and you don't know why.** `cat logs/<jobname>_<jobid>.err` first; tracebacks land there.
3. **Your push or pull asks for password every time.** SSH ControlMaster isn't set up correctly. See SETUP_NEW_USER §1.
4. **The fit converged but the residuals look bad.** Open the `fit_subplot.png` and apply the bar in `Modules/11_Physical_Mass_Models/` §6 — a converged fit with coherent ring residuals is a *failed* fit, not a converged one.
5. **You hit an autolens API error.** Check `pip index versions autolens` against your env's Python — autolens releases monthly and minor-version drift is the most common cause.

---

## See also

- [`SETUP_NEW_USER.md`](SETUP_NEW_USER.md) — one-time setup walk-through (SSH alias, conda env, cannon.env)
- [`RECIPES.md`](RECIPES.md) — all `EXAMPLE=` × `FIT_EXTRA_ARGS=` recipes, with time budgets
- [`CLUSTER_WORKFLOW_NOTES.md`](CLUSTER_WORKFLOW_NOTES.md) — design notes on the rsync/sbatch architecture (read if you want to extend it)
- `cannon.env.example` — annotated config template (generic, all labs)
- `cannon.env.hernquist` — pre-filled config for Hernquist-lab members (only `CANNON_USER` to fill in)
