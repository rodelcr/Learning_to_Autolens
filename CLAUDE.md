# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

**Learning to Autolens** is a step-by-step PyAutoLens tutorial suite (Modules 01–15
shipped), paired with an `Examples/` practice gym of ~14 lens architectures
(compound multi-plane, double-source-plane, group-scale, quad time delays,
radial-arc + SMBH detection, real AGEL HST data, …) and a dedicated
**Cluster-Computing module (Mod 10)** for moving heavy fits to Harvard FASRC
Cannon. Strict ship discipline: only fits audited STRICT-PASS by the
`/autolens-fit-diagnostics` skill enter the ship-set. Current release tag is
`v0.96-alpha`; v0.97 / v0.98 are mid-flight.

User-facing entry points are `README.md` and `START_HERE.md` — don't duplicate
their install steps here. This file is the **operator** doc: how to find your
way around, how the Cannon round-trip works, and how the audit + handoff
cadence is structured.

## Where things live (non-obvious parts)

- `Modules/NN_*/` — numbered tutorial notebooks + their committed Cannon
  results (`results/<search>/` — fit_subplot.png, corner.pdf, samples.csv,
  summary.json, model_results.txt — the lightweight artifact bundle).
- `Examples/<lens_architecture>/` — the practice gym; each example has a
  driver `Modules/10_Cluster_Computing/scripts/fit_example_<name>.py` and a
  notebook `Examples/<name>/01_<name>.ipynb`. Many drivers expose `--part=`
  sub-modes (e.g. `radial_arc_smbh --part={direct,with_BH,with_kinematics}`,
  `quad_time_delay --part={direct_fit,joint_fit_h0_kin}`).
- `Solutions/` — `NN_<topic>_SOLVED.ipynb` for module exercises.
- `Notes/` — per-module LaTeX theory companions; `Notes/build.sh` compiles all.
- `Mathematica/` — `.wl` symbolic-verification scripts (optional).
- `autolens_workspace_latest/` — PyAutoLens workspace v2026.2 (live datasets);
  `autolens_workspace_original/` — v2025.11 reference copy. Treat both as
  read-only third-party trees.
- `Modules/10_Cluster_Computing/scripts/` — every cluster wrapper lives here:
  `push_to_cannon.sh`, `pull_from_cannon.sh`, `submit_to_cannon.sh`,
  `submit_cannon.slurm`, `seed_cannon_data.sh`, `export_results.py`,
  `fit_module{04,05,06,09}.py`, `fit_example_*.py`, `preflight_check_v0*.sh`.
- `Modules/10_Cluster_Computing/cannon.env` — **gitignored**, per-user Cannon
  settings (SSH alias, lab path, slurm account, mem/time defaults, conda env).
  `cannon.env.example` is the template.
- `docs/superpowers/specs/` + `docs/superpowers/plans/` — design specs and
  step-by-step implementation plans for the **paper-reproduction program**
  (Li+2023, Ballard+2023, Li+2026, Nightingale+2023). Plans use checkbox
  syntax for tracking; the `superpowers:subagent-driven-development` or
  `superpowers:executing-plans` skills consume them.
- `private/` — **gitignored** local-only sandbox for paper reproductions
  (`private/2307_09271_li2023_cosmography_population/`,
  `private/2309_04535_ballard2023_tspl_jackpot/` ←9.8 GB HST data,
  `private/2602_20889_li2026_dspl_imf_nfw/`, `private/STRATEGY.md`,
  `private/PROGRESS_YYYY_MM_DD.md`).
- `Modules/10_Cluster_Computing/HANDOFF_YYYY_MM_DD.md` — running narrative.
  **Always read the most recent one at session start**:
  `ls -t Modules/10_Cluster_Computing/HANDOFF_*.md | head -1`.
- Root-level: `RELEASE_NOTES_v0.NN.md`, `PROGRESS_LOG.md`, `NEXT_STEPS.md`,
  `LEARNING_LOG.md` — version-tagged narrative state.

## Cannon round-trip — how we pass information back and forth

The cluster connection is **already set up** with SSH multiplexing — Duo 2FA
happens once per laptop session, then every subsequent `ssh cannon …` /
`rsync … cannon:…` reuses the master socket. This is what lets Claude drive
the cluster from this session.

### The SSH alias

In `~/.ssh/config`:
```
Host cannon
   User rcordova
   HostName login.rc.fas.harvard.edu
   ControlMaster auto
   ControlPath ~/.ssh/connections/%r@%h:%p
```

**Always use `ssh cannon …`, never `ssh login.rc.fas.harvard.edu …` directly.**
The latter doesn't share the control socket and will trigger a fresh Duo
prompt every call.

Check the socket is alive before issuing cluster commands:
```bash
ssh -o BatchMode=yes -o ConnectTimeout=5 cannon "echo ok" 2>&1 | head -1
```
If that returns `ok`, you can fire off `squeue` / `sacct` / `tail` directly.
If it returns `Permission denied (keyboard-interactive)`, the master died —
ask the user to re-`ssh cannon` from a terminal once to re-establish 2FA;
Claude can't drive Duo from inside the harness.

### cannon.env — per-user overrides

All four cluster scripts (`submit_to_cannon.sh`, `push_to_cannon.sh`,
`pull_from_cannon.sh`, `seed_cannon_data.sh`) source
`Modules/10_Cluster_Computing/cannon.env` if it exists. Current values
(2026-05-18) on this laptop:

```bash
CANNON_SSH=cannon
CANNON_USER=rcordova
CANNON_REPO_ROOT=/n/holystore01/LABS/hernquist_lab/Lab/$CANNON_USER/learning_to_autolens
SLURM_ACCOUNT=siag_lab               # primary analysis (see policy below)
SLURM_PARTITION=siag_gpu
SLURM_MEM=64G, SLURM_TIME=24:00:00, SLURM_CPUS_PER_TASK=32
CONDA_ENV=autolens312
```

### Account / partition policy

- **Primary analysis** (paper-reproduction program, v0.97/v0.98 demos,
  research-grade fits): `--account=siag_lab` on a `siag` / `siag_gpu` /
  `siag_combo` partition. This is the default — `cannon.env` sets it,
  `submit_to_cannon.sh` and `submit_cannon.slurm` inherit it.
- **Student-facing material** (tutorial Cannon results that ship as
  committed artifacts in `Modules/NN_*/results/`, demos rerun for the
  shared ship-set): `--account=hernquist_lab --partition=hernquist`. Keeps
  the tutorial track from burning the science fairshare.

`submit_cannon.slurm` hard-codes `#SBATCH --partition=hernquist` for backward
compatibility with the v0.96-era tutorial runs, but `cannon.env`'s
`SLURM_PARTITION=siag_gpu` overrides it for primary work via the
`submit_to_cannon.sh` wrapper. **Always confirm the account in `squeue` output
before assuming** — the two pools coexist and a stale wrapper invocation can
silently route a paper-repro fit to hernquist.

To repoint a pending job that's already been queued under the wrong account:
```bash
ssh cannon "scontrol update JobId=<id> Account=siag_lab Partition=siag"
```
This works on `PENDING` jobs without losing queue position (Slurm just
re-evaluates priority). Use it after a default-account mistake instead of
cancel + resubmit, since cancellation re-prices the job from scratch.

### The round-trip — push, submit, monitor, pull

```bash
# 1. Push code + checkpoints → Cannon (rsync; preserves checkpoint.hdf5)
bash Modules/10_Cluster_Computing/scripts/push_to_cannon.sh --go

# 2. (Once per dataset version) seed datasets
bash Modules/10_Cluster_Computing/scripts/seed_cannon_data.sh --go

# 3. Submit a module fit (push + SHA256 verify + sbatch in one shot)
bash Modules/10_Cluster_Computing/scripts/submit_to_cannon.sh 04
bash Modules/10_Cluster_Computing/scripts/submit_to_cannon.sh 09 --mem 64G --time 48:00:00

# 3b. Submit an Example driver — sbatch directly on Cannon
ssh cannon "cd \$CANNON_REPO_ROOT && sbatch --export=ALL,SCRIPT=fit_example_radial_arc_smbh.py,PART=with_kinematics \
            Modules/10_Cluster_Computing/scripts/submit_cannon.slurm"

# 4. Monitor — drives the live socket from the laptop, no re-2FA
ssh cannon "squeue --me --format='%.10i %.20j %.10P %.10T %.12M %.10L %R'"
ssh cannon "sacct -X --starttime=$(date +%Y-%m-%d) --format=JobID,JobName%25,State,Elapsed,ExitCode,Partition"
ssh cannon "tail -f \$CANNON_REPO_ROOT/logs/<jobname>_<jobid>.out"
ssh cannon "seff <jobid>"

# 5. Pull lightweight artifacts back (corner, fit_subplot, summary.json, csv)
bash Modules/10_Cluster_Computing/scripts/pull_from_cannon.sh --go
# Add --include-raw only if you need checkpoint.hdf5 or the full Nautilus tree.
```

### Stale-hash safeguard

`submit_to_cannon.sh` SHA256-compares the local `fit_module${MODULE}.py` to
the remote copy after `push_to_cannon.sh`. If they differ, it aborts before
`sbatch` — this is the guard against the 2026-04-20 incident where a
re-submitted job appeared to "succeed" in minutes because Nautilus resumed
from a stale `.completed` marker. Don't bypass this check.

### Cluster-side reference

When work happens **inside** a Cannon SSH session (not from the laptop), use
`Modules/10_Cluster_Computing/CANNON_HANDOFF.md` — it's the self-contained
runbook for repo path, conda env, sbatch, monitoring, and the stale-hash
recovery procedure. Written so a Claude instance started directly on Cannon
can operate without the laptop's skills / memory.

## Audit workflow — never ship a fit on numbers alone

After a Cannon job lands and `pull_from_cannon.sh` brings artifacts back:

1. **Always open `fit_subplot.png`.** Summary JSON numbers alone have passed
   fits that a visual residual would fail. (Codified in memory as
   `feedback_open_fit_png.md`.)
2. Invoke `/autolens-fit-diagnostics` on the result directory. It produces a
   PASS / SUSPECT / FAIL verdict and names the failure mode
   (mass-model-too-simple, regularization-over-smoothing, PSF mismatch,
   astrometric offset, mesh collapse, SLaM stage regression, the Pattern
   A–F catalogue, etc.).
3. For Examples being prepared for a release tag, run the version's preflight:
   `bash Modules/10_Cluster_Computing/scripts/preflight_check_v096.sh` (or
   the current version). The preflight enforces **chi²-at-truth AND
   driver-truth two-check methodology** for every mock — codified after the
   DSPL stale-JSON + mge axis-swap bugs of 2026-05-15.
4. Bayes-factor improvement is **not** parameter recovery — check rails +
   caustic + truth before passing a fit on ΔlogZ alone
   (`feedback_bayes_factor_vs_truth.md`).
5. When a freely-fit search stalls or regresses, build a tight-prior
   truth-anchored variant **before** adding model complexity — distinguishes
   search-space failures from model-space failures
   (`feedback_truth_anchored_validation.md`).

## Handoff cadence

The repo runs on rolling handoff docs, not just git log.

- **`Modules/10_Cluster_Computing/HANDOFF_YYYY_MM_DD.md`** — written at the
  end of substantial work arcs (often immediately before `/compact`). Reads
  like a session log: headline since prior handoff, active Cannon jobs at
  handover, what's ready to execute next, open questions/blockers, first
  actions for next session. Update **proactively** alongside `PROGRESS_LOG.md`
  and the project memory files.
- **`PROGRESS_LOG.md`** — timestamped, append-only narrative for the whole repo.
- **`RELEASE_NOTES_v0.NN.md`** — per-version ship/defer breakdown.
- **`private/STRATEGY.md`** + **`private/PROGRESS_YYYY_MM_DD.md`** — same
  cadence for the paper-reproduction program (gitignored).

## Module curriculum (status at a glance)

| #  | Module | Status |
|----|--------|--------|
| 01 | Basics: Grids, Galaxies, Ray-Tracing | ✓ |
| 02 | Simulating Lens Data | ✓ |
| 03 | Your First Lens Model | ✓ |
| 04 | Search Chaining & SLaM | ✓ (Cannon results committed) |
| 05 | Pixelized Source Reconstructions | ✓ (Cannon results committed) |
| 06 | Multi-Component Mass Models | ✓ |
| 07 | Real Data: FITS to Model | ✓ |
| 08 | Results, Diagnostics & Figures | ✓ |
| 09 | MGE & Linear Light Profiles | ✓ (production standard; Cannon-only fits) |
| 10 | Cluster Computing | ✓ (this is the runbook) |
| 11 | Physical Mass Models | ✓ shipped — 6-panel physical-bar audit, Pattern A–F catalogue, f_DM(<θ_E), γ′ vs Auger+10 |
| 12 | Time-Delay Cosmography & MSD | ✓ shipped — Fermat potential, MSD on SIE quad, TDCOSMO chain |
| 13 | TDCOSMO with Kinematics | in progress (Jeans solver shared in `_jeans_sigma_v.py`; `AnalysisKinematics` stubbed) |
| 14 | Compound (Multi-Plane) Lensing | in progress |
| 15 | Radial Arcs & Caustic Topology | ✓ shipped (v0.96) — λ_t/λ_r, γ′ from radial-arc position, γ′–M_BH degeneracy |

`Examples/` (ships independently of the modules): `agel_real_target`,
`bayesian_model_comparison`, `cluster_scale`, `compound_lens`,
`compound_lens_zoo`, `cosmography_joint_posterior`, `disky_spiral_lens`,
`double_source_plane`, `galaxy_galaxy_single_arc`, `group_scale`,
`interferometer_basic`, `mge_to_physical`, `positions_modeling`,
`quad_time_delay`, `radial_arc_smbh`, `subhalo_sensitivity`.

## Key dependencies

Authoritative pins live in `requirements.txt`. Hard requirements:
**Python 3.12** (autolens 2026.4+ requires it for `RectangularAdaptDensity` /
`RectangularAdaptImage` / `reg.Adapt`), **autolens ≥ 2026.4.13**, **matplotlib
< 3.9** (PyAutoLens plotting incompat with 3.9+), **nautilus-sampler** (default
sampler, replaces dynesty).

Install: `python -m pip install -r requirements.txt` — **never bare `pip`**.
A stray `~/.local/bin/pip` (Py 3.10) on `$PATH` will silently install into the
wrong interpreter; symptom is `pip install` succeeds but `import autolens`
fails. Verify with `python check_install.py`.

## Cannon environment (Cannon 2.0, 2026-04-18 onward)

- Conda env is **`autolens312`** (Python 3.12 + autolens 2026.4.13.6+). The
  slurm submit script defaults `CONDA_ENV=autolens312`. The legacy `autolens`
  env (Py 3.11 + autolens 2026.2) is too old for SLaM-using scripts — do not
  target it.
- Activate via Miniforge, **not** `module load Anaconda3`:
  `source /n/sw/Miniforge3-25.3.1-0/etc/profile.d/conda.sh && conda activate autolens312`.
- Repo lives at
  `/n/holystore01/LABS/hernquist_lab/Lab/${USER}/learning_to_autolens`. Never
  put Nautilus output in `$HOME` (10 GB quota, slow I/O).
- `siag_gpu` has 64 CPUs / 515 GB RAM per node; `SLURM_CPUS_PER_TASK=32` is
  the Nautilus sweet spot.
- When you see `AttributeError: module 'autolens' has no attribute 'X'`, check
  `pip index versions autolens` against the env's Python before assuming the
  API was renamed — autolens releases monthly and minor-version drift is the
  most common cause.

## Important reminders

- **Self-contained modules.** Each module must run independently — don't
  introduce cross-module imports beyond `Modules/10_Cluster_Computing/scripts/`.
- **Theory first, code second.** Every code cell preceded by a markdown cell
  explaining the physics.
- **Update `PROGRESS_LOG.md` + the latest handoff doc** alongside code changes
  — handoff docs are the compact-survival mechanism.
- **`autolens_workspace_*/` directories are reference copies** — modify only
  the educational layer in `Modules/` and `Examples/`.
- **`private/` is gitignored on purpose** — it holds 10+ GB of paper-reproduction
  HST data and the scratch work feeding the v0.98 paper-repro program. The
  public-facing equivalents are `docs/superpowers/specs/` and
  `docs/superpowers/plans/`.
