# Cannon handoff — for a Claude instance running on the cluster

Self-contained reference for operating the `Learning_to_Autolens` cluster
side of the round-trip. Written so a Claude session started on Cannon
(SSH'd in, or launched from a login-node terminal) can diagnose, rerun,
and report back **without** needing the user's laptop `~/.claude/skills/`
or `~/.claude/projects/.../memory/` directories.

## Repo location on Cannon

```
/n/holystore01/LABS/hernquist_lab/Lab/${USER}/learning_to_autolens
```

That path is the canonical one for both `REPO_ROOT` and `OUTPUT_ROOT` used
by `submit_cannon.slurm`. Nautilus writes its output under
`${REPO_ROOT}/output/module_${MODULE}/...`. The lightweight exported
artifacts (the ones `pull_from_cannon.sh` ships to the laptop) live at
`${REPO_ROOT}/Modules/XX_*/results/<search>/`.

`$HOME` is **too small** (10 GB quota, slow I/O) — never put Nautilus
outputs there. The login node also has heavy filters against large
tarballs — don't `rsync` into `$HOME`.

## Conda environment

The only supported env is **`autolens312`**:
- Python 3.12
- autolens ≥ 2026.4.13

Activate via Miniforge (FASRC's `module load Anaconda3` is out of date
on Cannon 2.0):

```bash
source /n/sw/Miniforge3-25.3.1-0/etc/profile.d/conda.sh
conda activate autolens312
python -c "import autolens as al; print(al.__version__)"   # expect 2026.4.13.x
```

If the env is missing (new Cannon account):

```bash
conda create -n autolens312 python=3.12 -y
conda activate autolens312
python -m pip install --upgrade pip      # NOT bare `pip` — shadow on PATH
python -m pip install -r "${REPO_ROOT}/requirements.txt"
```

The legacy `autolens` env (Python 3.11 + autolens 2026.2) is unusable:
it pre-dates `RectangularAdaptDensity`, `RectangularAdaptImage`, and
`reg.Adapt`, which every SLaM-using script depends on.

## Submitting a job

```bash
cd "${REPO_ROOT}"
sbatch --export=ALL,MODULE=04 --job-name=mod04 \
       Modules/10_Cluster_Computing/scripts/submit_cannon.slurm
```

For heavier jobs (Module 09 MGE SLaM especially) override time/mem:

```bash
sbatch --export=ALL,MODULE=09 --job-name=mod09 \
       --mem=64G --time=48:00:00 \
       Modules/10_Cluster_Computing/scripts/submit_cannon.slurm
```

## Monitoring

```bash
squeue -u $USER                       # all my jobs
squeue -j ${JOB_ID}                   # one job
sacct -j ${JOB_ID} --format=JobID,State,Elapsed,MaxRSS,ExitCode
seff ${JOB_ID}                        # efficiency summary (after completion)
scontrol show job ${JOB_ID}           # full detail

tail -f ${REPO_ROOT}/logs/modXX_${JOB_ID}.out    # live stdout
tail -f ${REPO_ROOT}/logs/modXX_${JOB_ID}.err    # stderr

scancel ${JOB_ID}                     # kill a single job
scancel -u $USER                      # kill all my jobs (nuclear — confirm with user)
```

## The stale-hash failure mode (seen 2026-04-20)

**Symptom:** a re-submitted job completes in minutes rather than hours.
The post-export `Modules/XX/results/` numbers are **identical** to the
previous run, with `summary.json` `chi_squared_per_pixel` unchanged to
four decimals. Only the `fit_subplot.png` mtime differs.

**Root cause:** `fit_module${MODULE}.py` on Cannon is out of sync with
the laptop version. Nautilus's `.completed` markers at the existing
model hash are still in place, so every search skips sampling; only
`export_results.py` ran.

**Diagnosis commands:**

```bash
# Compare laptop and Cannon fit-script hashes (run on laptop, not here)
shasum -a 256 Modules/10_Cluster_Computing/scripts/fit_module04.py
ssh cannon "sha256sum ${REPO_ROOT}/Modules/10_Cluster_Computing/scripts/fit_module04.py"

# On Cannon: see which hash Nautilus wrote this run's output to
ls -d ${REPO_ROOT}/output/module_04/slam/simple/source_lp\[1\]/*/
stat ${REPO_ROOT}/output/module_04/slam/simple/source_lp\[1\]/*/.completed
# If .completed is old (matches a prior run's date) and mtime wasn't bumped,
# Nautilus resumed → no fresh sampling happened.

# Compare info.txt N= counts against what the current fit script's model expects
head -20 ${REPO_ROOT}/Modules/04_Search_Chaining_SLaM/results/source_lp\[1\]/info.txt
# e.g. `Isothermal (N=3)` = frozen centre, old behavior
#      `Isothermal (N=5)` = free centre with new GaussianPrior
```

**Recovery:** force a fresh model hash. Two options:

1. **Push the updated script and re-sbatch.** From the laptop:
   ```bash
   bash Modules/10_Cluster_Computing/scripts/submit_to_cannon.sh 04
   ```
   That wrapper pushes, verifies SHA256 matches on both sides, and
   only then submits — preventing exactly this failure mode going
   forward.

2. **If the laptop script is already in sync but you want a forced
   rerun anyway** (e.g., to retest with a changed env):
   ```bash
   # On Cannon — delete the specific stage's output tree; the rest auto-resumes
   rm -rf ${REPO_ROOT}/output/module_04/slam/simple/source_lp\[1\]
   # Re-submit; SOURCE LP runs fresh and the subsequent stages inherit its result
   sbatch --export=ALL,MODULE=04 ...
   ```
   ⚠  Only delete scoped stage directories. Never `rm -rf output/`
   without user confirmation — that discards all finished runs.

## PYAUTOFIT_TEST_MODE

If `PYAUTOFIT_TEST_MODE` is in the env, **PyAutoFit skips sampling and
returns a random prior draw**. Every SLaM stage would be meaningless.
Multi-layer guards are in place:

- `check_install.py` — marks it as a FAIL
- `submit_cannon.slurm` — refuses to submit
- `slam_v2026.py` — raises at import (covers Mods 04, 09)
- Every notebook's imports cell — raises on load

Check before anything else if a run looks suspicious:

```bash
env | grep PYAUTOFIT_TEST_MODE     # should print nothing
```

If set, `unset PYAUTOFIT_TEST_MODE`, restart any active shells and
kernels, and rerun.

## Exporting results manually (if the slurm job's post-step failed)

```bash
cd ${REPO_ROOT}
python Modules/10_Cluster_Computing/scripts/export_results.py \
    --output-root ${REPO_ROOT}/output/module_04 \
    --module 04 \
    --repo-root ${REPO_ROOT}
```

For a single stage:

```bash
python Modules/10_Cluster_Computing/scripts/export_results.py \
    --search-dir ${REPO_ROOT}/output/module_04/slam/simple/mass_total\[1\]/<hash> \
    --dest ${REPO_ROOT}/Modules/04_Search_Chaining_SLaM/results/mass_total\[1\]
```

The `--force` flag re-renders the corner plot even if one is already
present.

## Fit-quality thresholds (condensed from the laptop skill)

Apply these to every `results/<stage>/summary.json` **and** to the
visual residual panels in the corresponding `fit_subplot.png`. The
numbers can look fine while the image tells you the fit is broken —
always open both.

| Field | PASS | SUSPECT | FAIL |
|---|---|---|---|
| `chi_squared_per_pixel` | ≤ 1.3 | 1.3 – 2.0 | ≥ 2.0 |
| `max_abs_normalized_residual` (σ) | ≤ 4.0 | 4.0 – 6.0 | ≥ 6.0 |

SLaM stage-specific expected ranges:

| Stage | `chi²/pix` | `max|res|` | Notes |
|---|---|---|---|
| SIS-only "baseline" | 15 | 20σ | **Expected bad** — pedagogical. |
| SOURCE LP | 1.0 – 1.3 | 3.5 – 5.0σ | Parametric source; faint arc residual normal. |
| SOURCE PIX[1/2] | 0.85 – 1.4 | 3.5 – 7σ | Pixelized; no coherent arc allowed. |
| LIGHT LP | 0.85 – 1.0 | ≤ 4σ | Noise-like residual; lens light alone. |
| MASS TOTAL | 0.85 – 1.0 | ≤ 4σ | Publishable; any arc residual → FAIL. |

**Monotonicity rule:** `log_evidence` must weakly increase across SLaM
stages (allow ~2–5 nat tolerance if regularization changed). A decrease
is a regression and always FAIL.

Visual residual patterns that FAIL regardless of the scalar numbers:

- **Ring / arc in residual map** → mass model too simple or astrometric offset
- **Quadrupole cross** → missing external shear
- **Point blobs at image positions** → PSF mismatch
- **Single bright pixel in Source Plane panel** → mesh collapse
- **Diffuse featureless blob in Source Plane panel** → over-regularized

## Rule of thumb for "is this run legitimate"

Copy-paste check any time a job completes:

```bash
MODULE=04
cd ${REPO_ROOT}
for stage in Modules/${MODULE}_*/results/*/; do
    sname=$(basename "$stage")
    s=$(cat "${stage}summary.json")
    chi=$(echo "$s" | python -c "import sys,json; d=json.load(sys.stdin); print(d.get('chi_squared_per_pixel'))")
    mr=$(echo "$s" | python -c "import sys,json; d=json.load(sys.stdin); print(d.get('max_abs_normalized_residual'))")
    lz=$(echo "$s" | python -c "import sys,json; d=json.load(sys.stdin); print(d.get('log_evidence'))")
    printf "  %-28s  chi2/pix=%-8s  max|res|=%-8s  logZ=%s\n" "$sname" "$chi" "$mr" "$lz"
done
```

Then open the relevant `fit_subplot.png`(s) in an image viewer. Stale
numbers repeated exactly = stale-hash rerun; escalate per §stale-hash.

## What to report back to the user

After any cluster operation, tell the user:

1. **Job ID** and the `squeue`/`tail` commands for it (from
   `submit_cannon.slurm`'s opening banner, or `sbatch` output)
2. **The provenance echo** from the slurm log — specifically the
   `fit_module${MODULE}.py SHA256`, `git HEAD`, `git state` lines
3. **Per-stage PASS/SUSPECT/FAIL** after the run finishes, derived
   from `summary.json` + the `fit_subplot.png` visual check
4. **Any regressions** (log_evidence going down between stages)
5. **Any missing fit.fits** warnings (seen in the slurm log as
   "residual metrics null")

The user should never have to open `summary.json` or `fit_subplot.png`
before you've already applied the thresholds and named the pattern.

## One-page command cheat sheet

```bash
# Env
source /n/sw/Miniforge3-25.3.1-0/etc/profile.d/conda.sh
conda activate autolens312

# Paths
REPO_ROOT=/n/holystore01/LABS/hernquist_lab/Lab/${USER}/learning_to_autolens
cd ${REPO_ROOT}

# Submit
sbatch --export=ALL,MODULE=04 Modules/10_Cluster_Computing/scripts/submit_cannon.slurm

# Monitor
squeue -u $USER
tail -f ${REPO_ROOT}/logs/mod04_${JOB_ID}.out

# Export (normally done at the end of the slurm script)
python Modules/10_Cluster_Computing/scripts/export_results.py \
    --output-root ${REPO_ROOT}/output/module_04 \
    --module 04 --repo-root ${REPO_ROOT}

# Force a fresh run of one stage (after confirming with user)
rm -rf ${REPO_ROOT}/output/module_04/slam/simple/source_lp\[1\]
```
