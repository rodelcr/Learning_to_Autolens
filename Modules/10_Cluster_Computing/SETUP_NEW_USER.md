# New user setup — running Learning_to_Autolens on Cannon

This guide onboards a new user to the full round-trip workflow:
laptop → push → Cannon (sbatch) → pull → view results. Expected
setup time: 30–60 minutes if you already have a Cannon account;
add a day or two if you have to request one.

## What you'll have at the end

- An SSH `cannon` alias that only prompts for Duo 2FA once per
  terminal session (not once per command).
- A personal `cannon.env` file with your username, lab storage
  path, and slurm account — so the repo's cluster scripts work
  for you without any code edits.
- A `autolens312` conda env on Cannon (Python 3.12 + autolens
  ≥ 2026.4.13) sitting at a known path.
- Confidence that `bash submit_to_cannon.sh 04` submits a job
  that writes to your lab storage and you can pull results from.

## Prerequisites

- A Cannon account (apply at <https://www.rc.fas.harvard.edu/>).
  Harvard-affiliated researchers qualify for free shared-partition
  access; tell the RCs which PI/lab sponsors you so they can add
  you to the right group.
- Duo 2FA set up for the account.
- Local clone of this repo on your laptop:
  ```bash
  git clone <repo-url> Learning_to_Autolens
  cd Learning_to_Autolens
  ```

---

## Step 1 — Set up an SSH alias (one-time)

Add this block to `~/.ssh/config` on your laptop:

```
Host cannon
    User <your_cannon_username>
    HostName login.rc.fas.harvard.edu
    ControlMaster auto
    ControlPath ~/.ssh/connections/%r@%h:%p
    ControlPersist 10m
```

Create the connections directory:

```bash
mkdir -p ~/.ssh/connections
chmod 700 ~/.ssh/connections
```

Test it:

```bash
ssh cannon hostname          # first call: Duo prompt
ssh cannon date              # next call within 10 min: silent
```

**Why it matters:** every `push` / `ssh sha256sum` / `sbatch` in the
workflow becomes a separate ssh connection. Without connection
multiplexing you get three Duo prompts per submit; with it, one.

See `CANNON_HANDOFF.md` for alternatives (SSH keys + Kerberos, longer
`ControlPersist`).

---

## Step 2 — Bootstrap a conda env on Cannon

Login once and install Miniforge + the `autolens312` env. You only
do this once per Cannon account.

```bash
ssh cannon
salloc --partition=test --time=1:00:00 --cpus-per-task=4 --mem=8G \
       --account=<your_slurm_account>

# Bootstrap Miniforge if you don't want to use FASRC's shared copy
# (skip this block and use /n/sw/Miniforge3-25.3.1-0/... instead if
# you do want the shared copy; then your CONDA_ACTIVATE_SCRIPT in
# cannon.env stays the default).
curl -L -o Miniforge3.sh \
  "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3.sh -b -p "$HOME/miniforge3"
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda init bash && exec bash

# Create the autolens312 env
conda create -n autolens312 python=3.12 -y
conda activate autolens312
python -m pip install --upgrade pip            # NEVER bare `pip` — shadow on PATH
python -m pip install -r learning_to_autolens/requirements.txt
python -c "import autolens as al; print(al.__version__)"  # expect 2026.4.13.x
exit
```

**Pitfalls:**
- If you see `AttributeError: module 'autolens' has no attribute
  'RectangularAdaptDensity'`, you're on the old `autolens` env
  (autolens 2026.2). SLaM pipelines need `autolens312`.
- If `pip install` reports success but `import autolens` fails, a
  stray `~/.local/bin/pip` is shadowing the conda env's pip. Use
  `python -m pip ...` explicitly.

---

## Step 3 — Decide where the repo lives on Cannon

Three places, each with different tradeoffs:

| Path | Persistence | Quota | Use for |
|---|---|---|---|
| `$HOME` (`/n/home02/<user>/`) | permanent, backed up | 100 GB | small configs |
| Lab storage (`/n/holystore01/LABS/<pi>_lab/Lab/<user>/`) | permanent, not backed up | lab-dependent | **recommended** for this project |
| Scratch (`/n/holyscratch01/users/<user>/`) | **purged after 90 days** | 50 TB | large intermediate files |

For `Learning_to_Autolens`, use lab storage. Nautilus outputs can hit
~GB per search × ~8 searches, and scratch's 90-day purge will delete
finished runs mid-project.

If you don't have lab storage yet, ask your PI to grant access to
`/n/holystore01/LABS/<pi>_lab/Lab/<you>/` — it's the normal FASRC
"add a lab member" ticket.

---

## Step 4 — Create your `cannon.env`

On the laptop, copy the template and edit with your values:

```bash
cp Modules/10_Cluster_Computing/cannon.env.example \
   Modules/10_Cluster_Computing/cannon.env
$EDITOR Modules/10_Cluster_Computing/cannon.env
```

Fields to set:

| Variable | What it is | How to find |
|---|---|---|
| `CANNON_USER` | Your Cannon login | whoami on Cannon |
| `CANNON_SSH` | Usually `cannon` (the ~/.ssh/config alias) | whatever Host name you picked in Step 1 |
| `CANNON_REPO_ROOT` | Absolute path where the repo lives on Cannon | `ssh cannon "echo /n/holystore01/LABS/<pi>_lab/Lab/\$USER/learning_to_autolens"` |
| `SLURM_ACCOUNT` | Fairshare account | `ssh cannon "sshare -U"` lists your accounts |
| `SLURM_PARTITION` | Usually `shared` (7-day) or `test` (8-hr) | see FASRC partition docs |
| `SLURM_MAIL_USER` | Your email | leave blank to disable notifications |
| `CONDA_ENV` | Usually `autolens312` | whatever you named it in Step 2 |
| `CONDA_ACTIVATE_SCRIPT` | Path to your Miniforge `conda.sh` | `/n/sw/Miniforge3-25.3.1-0/etc/profile.d/conda.sh` (FASRC shared) or `$HOME/miniforge3/etc/profile.d/conda.sh` (your own) |

`cannon.env` is **gitignored** — your settings don't leak into the
public repo.

---

## Step 5 — First push (creates the remote repo tree)

From the laptop repo root:

```bash
bash Modules/10_Cluster_Computing/scripts/push_to_cannon.sh          # dry run — shows what would transfer
bash Modules/10_Cluster_Computing/scripts/push_to_cannon.sh --go     # actual transfer
```

This rsyncs the repo (plus the `autolens_workspace_original/dataset/`
and `autolens_workspace_latest/dataset/` trees) to your
`CANNON_REPO_ROOT`. Expect the first transfer to take 2–5 minutes
depending on your network.

Verify on Cannon:

```bash
ssh cannon "ls $CANNON_REPO_ROOT/Modules/"
# should list 01_Basics_Grids_Galaxies_RayTracing ... 10_Cluster_Computing
```

---

## Step 6 — First job (the easy one)

Submit Module 04 — the two-search chain + SLaM pipeline:

```bash
bash Modules/10_Cluster_Computing/scripts/submit_to_cannon.sh 04
```

What happens:

1. Warns if you have uncommitted changes under
   `Modules/10_Cluster_Computing/scripts/` (skip with `y` if you're
   iterating).
2. Runs `push_to_cannon.sh --go`.
3. Computes SHA256 of the local and remote `fit_module04.py`, aborts
   if they differ (catches stale-rsync bugs).
4. ssh's to Cannon and sbatches the job with your cannon.env
   overrides applied as CLI args.
5. Prints the job ID, monitoring commands, and the pull command.

Expected runtime for Module 04 on 16 cores: ~1.5–3 h.

---

## Step 7 — Monitor the job

```bash
# In another terminal
ssh cannon "squeue -u \$USER"
ssh cannon "tail -f $CANNON_REPO_ROOT/logs/mod04_<JOB_ID>.out"
```

The `logs/` directory is created under `$CANNON_REPO_ROOT` by sbatch
automatically. Early log output you should see:

```
fit_module04.py SHA256: ...
fit_module04.py mtime:  ...
git HEAD:     <commit-hash>
git branch:   main
git state:    clean
PyAutoLens 2026.4.13.x PyAutoFit 2026.4.13.x
```

If `git state: DIRTY`, the repo on Cannon has uncommitted changes
(likely rsync'd from an uncommitted laptop working tree). That's
fine for one-off debugging but not reproducible; commit on the
laptop before the next submit.

---

## Step 8 — Pull results back

When the job email arrives (or `squeue` no longer shows it):

```bash
bash Modules/10_Cluster_Computing/scripts/pull_from_cannon.sh --go
```

Artifacts land at `Modules/04_Search_Chaining_SLaM/results/<stage>/`.
Each directory has:

- `fit_subplot.png` — the diagnostic panel (open first)
- `corner.pdf` — posterior corner plot
- `summary.json` — top-level quality metrics
- `samples.csv`, `samples_summary.json`, `info.txt`, `model_results.txt`

Open the notebook and run the **"Viewing pre-computed results from
the Cannon cluster"** cell to see the pulled result rendered inline.

---

## Step 9 — Verify the fit actually converged

Before you trust the numbers, apply the quality bar. This is
condensed from the `autolens-fit-diagnostics` skill:

```
PASS if all:
  summary.chi_squared_per_pixel        ≤ 1.3
  summary.max_abs_normalized_residual  ≤ 4σ
  fit_subplot.png residual panel       no coherent ring / arc / cross
  fit_subplot.png source plane panel   recognizable extended source (for pixelized)
  SLaM log_evidence                     monotonically non-decreasing across stages
```

See `CANNON_HANDOFF.md` §"Fit-quality thresholds" and the
`autolens-fit-diagnostics` skill's `references/residual-patterns.md`
for named failure modes (ring, quadrupole, mesh collapse, etc.).

---

## Step 10 — Done

You now have a working round-trip. To adapt the template to your
own science target:

- Copy `fit_template.py` → `fit_moduleNN.py`; fill in `build(...)`.
- Add a `MODULE=NN` case to `submit_cannon.slurm`'s dispatch block.
- Add a `Viewing pre-computed results` cell to your module notebook.
- Submit via `bash submit_to_cannon.sh NN`.

See `CLUSTER_WORKFLOW_NOTES.md` for lessons learned and future
improvements — skim before you start adding a lot of modules.

---

## Troubleshooting the most common first-time failures

### "Permission denied (keyboard-interactive)" or repeated Duo prompts

Your `~/.ssh/config` `cannon` alias isn't configured for connection
multiplexing, OR the control socket has expired. Run `ssh -O exit
cannon` to clean up any stuck socket and retry.

### `ERROR: local and remote fit_module04.py DIFFER`

`push_to_cannon.sh` didn't land where `submit_to_cannon.sh`
expects. Check `CANNON_REPO_ROOT` in your `cannon.env` matches what
the slurm script reads. On Cannon:

```bash
ssh cannon "ls -la $CANNON_REPO_ROOT/Modules/10_Cluster_Computing/scripts/fit_module04.py"
ssh cannon "sha256sum $CANNON_REPO_ROOT/Modules/10_Cluster_Computing/scripts/fit_module04.py"
shasum -a 256 Modules/10_Cluster_Computing/scripts/fit_module04.py
```

### Job "finishes quickly" with identical results as before

Stale-hash silent resume — `fit_module.py` on Cannon is out of sync
with the laptop, so Nautilus re-used old `.completed` markers.
The hashes-match check in `submit_to_cannon.sh` now prevents this,
but if you use raw `sbatch` instead of the wrapper, you're on your
own. See `CLUSTER_WORKFLOW_NOTES.md` for the full diagnosis
protocol.

### `ModuleNotFoundError: autolens`

Your `CONDA_ENV` doesn't exist on Cannon, or `CONDA_ACTIVATE_SCRIPT`
points to a Miniforge that doesn't have it. Verify:

```bash
ssh cannon "ls $CONDA_ACTIVATE_SCRIPT && source $CONDA_ACTIVATE_SCRIPT && conda env list"
```

The `autolens312` env should be listed.

### `PYAUTOFIT_TEST_MODE is set`

Unset it on the laptop:

```bash
unset PYAUTOFIT_TEST_MODE
# Restart any open Jupyter kernels too
```

This env var makes PyAutoFit skip sampling — every fit becomes a
random prior draw. Guards throughout this repo refuse to run while
it's set. See `CLUSTER_WORKFLOW_NOTES.md` §4 for the history.

---

## Further reading

- **`CANNON_HANDOFF.md`** — for a Claude Code session running
  *on* Cannon (SSH'd in). Covers queue commands, stale-hash
  diagnosis, manual `export_results.py` invocation, fit-quality
  thresholds.
- **`CLUSTER_WORKFLOW_NOTES.md`** — retrospective of the pain
  points that shaped this architecture + prioritized improvement
  roadmap. Read before adding new infrastructure.
- **Module 10 notebook** (`10_cluster_computing.ipynb`) — the
  pedagogical version of this workflow. Section 8 "Converting
  Your Own Notebook" is the tutorial-style equivalent of this
  setup guide.
