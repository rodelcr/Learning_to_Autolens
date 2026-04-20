# Progress Log — Learning to Autolens

Timestamped record of major milestones and work completed.

---

## 2026-03-24 — Project Initialization

- Created repository structure: `Modules/` (8 module directories), `Mathematica/`, `Figures/`, `Output/`
- Copied structural essentials from `autolens_workspace` (125 MB): scripts, notebooks, SLaM pipeline, config, datasets — excluding output (492 MB)
- Created repo infrastructure: `CLAUDE.md`, `PROGRESS_LOG.md`, `.gitignore`, `README.md`
- Initialized git repository

### Module curriculum planned:
1. Basics: Grids, Galaxies, Ray-Tracing
2. Simulating Lens Data
3. Your First Lens Model (Non-Linear Search)
4. Search Chaining & the SLaM Pipeline
5. Pixelized Source Reconstructions
6. Multi-Component Mass Models
7. Real Data: FITS to Model
8. Results, Diagnostics & Publication Figures

### Reference texts (same as Learning to Lens):
- Congdon & Keeton (2018) — Primary lensing reference
- Narayan & Bartelmann (1997) — Classic lensing lectures
- Saha et al. (2024) — Modern strong lensing review
- Meneghetti (2021) — Lensing with Python examples
- Schneider et al. (1992) — Classic monograph
- Nightingale, Dye & Massey (2018) — AutoLens methodology paper

---

## 2026-03-24 — Modules 01–04 Complete (Notebooks + LaTeX + Mathematica)

### Module 01: Basics — Grids, Galaxies, Ray-Tracing
- **Notebook** (`Modules/01_.../01_grids_galaxies_raytracing.ipynb`): 8 sections — Grid2D, Sérsic profiles, SIS/SIE/NFW mass, Galaxy composition, Tracer ray-tracing, critical curves/caustics, multi-plane lensing, 5 exercises
- **Mathematica** (`Mathematica/01_lens_equation_and_profiles.wl`): Symbolic verification of lens equation, SIS properties, Sérsic b_n, ellipticity parameterization, critical curve conditions, enclosed mass
- **LaTeX** (`Notes/01_Basics/01_basics_theory.tex`): 6 sections of theory (lens equation, convergence/shear/magnification, Sérsic, mass profiles, multi-plane, cosmology). Compiles to 5-page PDF.

### Module 02: Simulating Lens Data
- **Notebook** (`Modules/02_.../02_simulating_lens_data.ipynb`): 7 sections — forward model, PSF construction (space/ground/AO), SimulatorImaging, HST/Euclid/ground instrument configs, over-sampling, FITS I/O, 4 exercises
- **Mathematica** (`Mathematica/02_psf_convolution_and_noise.wl`): PSF FWHM↔σ, convolution theorem, noise model derivation, Nyquist sampling
- **LaTeX** (`Notes/02_Simulating/02_simulating_theory.tex`): Forward model, PSF theory, noise statistics, instrument table, over-sampling

### Module 03: Your First Lens Model
- **Notebook** (`Modules/03_.../03_first_lens_model.ipynb`): 9 sections — Bayesian inference theory, data loading/masking, af.Model/af.Collection, prior customization, Nautilus nested sampling, running fits, posteriors/corner plots, degeneracies, 5 exercises
- **Mathematica** (`Mathematica/03_bayesian_inference_and_chi2.wl`): Gaussian likelihood, Bayes' theorem, nested sampling evidence, mass-sheet degeneracy verification
- **LaTeX** (`Notes/03_First_Model/03_first_model_theory.tex`): Bayes' theorem, likelihood, priors, nested sampling, degeneracies (MSD, source-size, ε-γ), Fisher information

### Module 04: Search Chaining & SLaM Pipeline
- **Notebook** (`Modules/04_.../04_search_chaining_slam.ipynb`): 10 sections — curse of dimensionality, prior passing mechanics, two-search chain example, SLaM architecture, SOURCE LP/PIX/LIGHT LP/MASS TOTAL stages, full pipeline walkthrough, 4 exercises
- **LaTeX** (`Notes/04_SLaM/04_slam_theory.tex`): Dimensionality curse, prior passing, SLaM architecture table, pixelized source inversion (linear system, regularization, mesh types), evidence for pixelized sources, adapt images

### LaTeX infrastructure
- Shared `Notes/preamble.tex` with physics macros (θ_E, Σ_cr, etc.), tcolorbox environments (keyresult, pythonbox, exercisebox), code listing style
- `Notes/build.sh` script for compiling individual or all module PDFs
- All 4 module PDFs compiled successfully to `Output/`

---

## 2026-03-24 — Module 05: Pixelized Source Reconstructions

- **Notebook** (`Modules/05_.../05_pixelized_sources.ipynb`): 8 sections — why pixelized sources, linear inversion theory, mesh types (Rectangular/Delaunay/Voronoi), regularization comparison (under/well/over-regularized), Hilbert adaptive mesh, Bayesian evidence for complexity, hands-on fitting workflow, 4 exercises
- **LaTeX** (`Notes/05_Pixelized/05_pixelized_theory.tex`): Linear forward model, regularized MAP inversion (Suyu+06), regularization matrix (constant + adaptive), Bayesian evidence derivation, mesh types, connection to mass modeling. PDF compiled.
- TODO: Mathematica .wl script for Module 05 (matrix inversion verification)

---

## 2026-03-24 — Modules 07 & 08: Tutorial Suite COMPLETE!

### Module 07: Real Data — From FITS to Model
- **Notebook** (`Modules/07_.../07_real_data_fits_to_model.ipynb`): 9 sections — real vs simulated data comparison, loading FITS with astropy, cutouts & pixel scales, empirical PSF handling, noise map conversions (weight/variance/ivar → σ), masking strategy, complete preparation workflow template, AGEL target template, 4 exercises
- **LaTeX** (`Notes/07_RealData/07_real_data_theory.tex`): FITS structure, pixel scale & WCS, PSF requirements & mismatch effects, noise map construction, mask sizing formula
- **Mathematica** (`Mathematica/07_data_preparation_formulas.wl`): Pixel scale conversions, noise map conversions (Poisson verification), mask size from θ_E for different instruments

### Module 08: Results, Diagnostics & Publication Figures
- **Notebook** (`Modules/08_.../08_results_diagnostics_figures.ipynb`): 9 sections — result object anatomy, corner plots, residual analysis (histogram + χ²_red), Einstein mass & velocity dispersion, source-plane reconstruction, publication-quality 3-panel figures, Bayesian model comparison (Bayes factor), JSON export, 4 exercises. Includes congratulations/next-steps section.
- **LaTeX** (`Notes/08_Results/08_results_theory.tex`): χ² distribution & expected scatter, parameter uncertainties (percentile reporting), Bayes factor & Jeffreys' scale, Einstein mass formula, SIE velocity dispersion, magnification
- **Mathematica** (`Mathematica/08_model_comparison_and_diagnostics.wl`): χ² distribution properties, Bayes factor odds ratios, Einstein mass for θ_E = 0.5"–3.0", velocity dispersion from θ_E

### TUTORIAL SUITE COMPLETE (Modules 01–08)
- **8 modules**, each with: Jupyter notebook + LaTeX theory companion + Mathematica symbolic verification
- **8 PDFs** compiled in `Output/`
- All pushed to https://github.com/rodelcr/Learning_to_Autolens (private)

---

## 2026-04-07 — Module 05 Bug Fixes

- Fixed pixelized source fit (Section 7): added `PositionsLH` for demagnified solutions, `use_jax=False` for JAX tracing issues, `SafeAnalysisImaging` wrapper for `LinAlgError`, restructured to two-search chain (parametric → pixelized)
- Fixed `mesh=al.mesh.Delaunay` (class) → `al.mesh.Delaunay()` (instance) to avoid `missing 'self'` error
- Applied same fixes to Solutions notebook

---

## 2026-04-08 — Module 09: Multi-Gaussian Expansion & Linear Light Profiles

- Upgraded PyAutoLens to 2026.2.26.4 (from 2025.11.18.1) with all auto* packages
- Cloned latest autolens_workspace into `autolens_workspace_latest/` (sandboxed)
- **Notebook** (`Modules/09_.../09_mge_linear_light_profiles.ipynb`): 10 sections — linear light profiles, linear inversion theory, Basis functions, manual MGE fit (60 Gaussians, 0 nonlinear params), MGE modeling with non-linear search, running the fit, `mge_model_from` utility, MGE SLaM pipeline (3-stage), Sérsic vs MGE comparison, 4 exercises
- **LaTeX** (`Notes/09_MGE/09_mge_theory.tex`): Linear algebra of light profile fitting, NNLS positive-only constraint, MGE theory (Emsellem+ 1994, Cappellari 2002), basis expansion & sigma grid, parameter space reduction, MGE in SLaM
- **Mathematica** (`Mathematica/09_mge_and_basis_functions.wl`): Gaussian normalization & half-light radius, PSF convolution property, MGE approximation of Sérsic n=4, linear system condition number, 3D deprojection (Abel transform)
- **Solutions** (`Solutions/09_mge_linear_light_profiles_SOLVED.ipynb`): 4 exercise solutions (Gaussian count, manual vs utility, MGE+pixelized, decomposition analysis)

---

## 2026-04-17 — v2026.2.26.4 Full-Suite Port

Goal: rewrite the tutorial suite to run end-to-end against PyAutoLens **2026.2.26.4** (the version we pinned alongside Module 09), eliminating v2025-era API calls that were left behind in Modules 04 and 05.

### New: `slam_v2026.py` shim
- Bundles the inline SLaM functions from `autolens_workspace_latest/scripts/guides/modeling/slam_start_here.py` as a single importable module
- Exposes `source_lp`, `source_pix`, `light_lp`, `mass_total` via `SimpleNamespace` so Module 04 can keep calling `.run()` / `.run_1()` / `.run_2()` as before
- Defaults: `mesh_init=RectangularAdaptDensity(28,28)` → `mesh=RectangularAdaptImage(28,28)`, `regularization=al.reg.Adapt`, `use_jax=False`

### Module 04 + Solution 04 — inline SLaM port
- Swapped imports from `autolens_workspace_original/slam` (v2025) to the new `slam_v2026` shim
- Removed all `al.AnalysisImaging(...)` construction at call sites — the shim now takes `dataset=dataset` and builds its own analysis
- Deleted v2025-only kwargs from SOURCE PIX runs: `adapt_image_maker=al.AdaptImageMaker(...)`, `mesh_init=al.mesh.Delaunay`, `mesh=al.mesh.Delaunay`, `image_mesh=al.image_mesh.Hilbert`, `regularization=al.reg.AdaptSplit`, `settings_inversion=al.SettingsInversion(...)` — v2026 uses the shim's defaults
- Dropped `SLAM_AVAILABLE` guard cells

### Module 05 + Solution 05 — Pixelization v2026 port
- `al.mesh.Delaunay(pixels=N)` → `al.mesh.RectangularAdaptDensity(shape=(N,N))` for the initial adaptive mesh
- `al.reg.AdaptiveBrightnessSplit` / `AdaptSplit` → `al.reg.Adapt`
- `al.SettingsInversion` → `al.Settings`
- `al.AdaptImageMaker` → `al.AdaptImages` + `al.galaxy_name_image_dict_via_result_from`

### Verification
- All 9 Modules + 9 Solutions re-executed under `PYAUTOFIT_TEST_MODE=1` against PyAutoLens 2026.2.26.4 — exit code 0 on every notebook
- Notebooks re-saved with v2026 outputs embedded

### Repo hygiene
- `.gitignore`: exclude `autolens_workspace_latest/` (300 MB pip-install reference copy), local `check_install.py`, and `Modules/**/*_debug.py` scratch scripts

### Regression caught: test-mode outputs overwrote real fits
- Commit `8fe3ddd` re-ran every notebook under `PYAUTOFIT_TEST_MODE=1` — this skips Nautilus sampling and returns a single prior draw, so the embedded figures in Solutions 03–09 (and the matching Modules) became nonsense: residuals at +25σ, sources outside the caustic, χ² ≈ 300
- Also discovered that autofit writes a `.completed` cache marker in `output/output/.../` — subsequent real-mode runs short-circuit to the cached 1-sample result. **Fix: `rm -rf <module_output_dir>` before each real-mode rerun.**

**Real-mode rerun progress (2026-04-17):**

| Notebook | Status | Max LL | Notes |
|---|---|---|---|
| Modules/03 | committed `6b1c56a` | +6516 | θ_E=1.60", χ²_pp≈33 |
| Solutions/03 | committed `6b1c56a` | +6516 | θ_E=1.60", source inside caustic |
| Modules/06 | committed `6b1c56a` | +5978 | |
| Solutions/06 | committed `6b1c56a` | +5949 | |
| Modules/07 | committed `6b1c56a` | n/a | no heavy fits |
| Solutions/07 | committed `6b1c56a` | n/a | no heavy fits |
| Modules/04 | running (5-stage SLaM) | — | search 2/5 active, ~5 h CPU in |
| Solutions/04 | **crashed** (LinAlgError) | — | `numpy.linalg.cholesky` failed on Nautilus neural-bound covariance after ~7 h on search_1 (n_live=75 → numerically singular). Notebook not overwritten; still holds test-mode placeholder outputs. TODO: bump n_live → 100 or add retry logic. |
| Modules/05 | running (2-search) | — | launched 2026-04-17 23:05 |
| Solutions/05 | committed | +6515 | search_1 parametric, 28600 samples, 28 min |
| Modules/09 | running (MGE SLaM) | — | launched 2026-04-17 23:05 |
| Solutions/09 | running (MGE SLaM) | — | launched 2026-04-17 23:05 |

- **Machine-sleep gotcha**: laptop auto-sleep stalled all runs for ~2 h around 19:00–21:00. Fix: `caffeinate -dims -w <kernel_pid>` tied to each Python kernel PID — exits automatically when the kernel dies. All six current runs have caffeinate guards active.
- 16-core / 48 GB machine, load avg ~5 with 6 nbconvert kernels; headroom OK.

---

## 2026-04-18 — Module 10 (Cluster Computing) + retreat from local runs

### The tipping point
After ~11 h of parallel local runs (Mod 05, Mod 09, Sol 09), the MacBook hit:
- Load average **390** on a 12-core machine (healthy ~12)
- **0.24% idle**; 62% user + 37% sys
- **47/48 GB RAM in use**, active swap thrashing (17M swapouts cumulative)
- Each Python kernel pegging 400–500% CPU (~5 cores)

Decision: kill all three, preserve Nautilus checkpoints, move the remaining compute to the Cannon cluster.

### Module 10: Cluster Computing
New tutorial module (`Modules/10_Cluster_Computing/`) — the cluster bridge for every prior module. Three-part pattern plus a post-processor.

**Scripts shipped** (`scripts/`):
- `fit_module04.py` — Mod 04 as CLI-driven standalone (two-search chain + 5-stage SLaM)
- `fit_module05.py` — Mod 05 (parametric + pixelized) with `SafeAnalysisImaging` inline
- `fit_module09.py` — Mod 09 full MGE SLaM pipeline (SOURCE LP → SOURCE PIX × 2 → LIGHT LP → MASS TOTAL/PowerLaw)
- `submit_cannon.slurm` — generic SBATCH, dispatches on `MODULE` env var (`sbatch --export=ALL,MODULE=05 ...`)
- `export_results.py` — walks Nautilus output; writes `fit_subplot.pdf`, `corner.pdf`, `info.txt`, `summary.json`, `samples.csv` per search into `Modules/XX/results/`
- `push_to_cannon.sh` / `pull_from_cannon.sh` — rsync wrappers, dry-run by default; push includes `autolens_workspace_latest/dataset/` for Mod 09

All fit scripts:
- Bump `n_live` over notebook defaults (75 → 100+) to avoid the `LinAlgError: Matrix is not positive definite` Nautilus crash we hit in Sol 04
- Write `flush=True` on every print so Slurm stdout is live-tailable
- Read `number_of_cores` from `$SLURM_CPUS_PER_TASK`
- Nautilus `checkpoint.hdf5` auto-resumes → re-submitting after a timeout picks up where it left off

### The "results viewer" pattern
Raw Nautilus output is 100–500 MB per search — not git-trackable. But new users shouldn't need a cluster account to see finished results. `export_results.py` solves this: after every cluster run, it writes ~5 MB of PDFs/JSON per module into `Modules/XX/results/`. **These are committed to git.** Any module notebook can display them with a small loader cell that reads `results/<search>/{fit_subplot.pdf, corner.pdf, summary.json}` — no PyAutoFit machinery required.

### Preserved local checkpoints (all pushed to Cannon via rsync for resume)

| Path | Size | State at kill |
|------|------|---------------|
| `Modules/04_.../output/.../search_2_sie_nolenslight/checkpoint.hdf5` | 96 MB | 203 bounds, stuck |
| `Modules/05_.../output/.../search2_pixelized_source/checkpoint.hdf5` | 1.9 MB | early |
| `Modules/09_.../output/.../search1_mge_lens_sersic_source/checkpoint.hdf5` | 33 MB | ~30% |
| `Solutions/output/.../search_1_sis_nolenslight/checkpoint.hdf5` | 58 MB | LinAlgError crash |
| `Solutions/output/.../search1_mge_lens_sersic_source/checkpoint.hdf5` | 30 MB | ~30% |

### Cluster submission plan
```bash
./Modules/10_Cluster_Computing/scripts/push_to_cannon.sh --go
ssh cannon "cd learning_to_autolens && \
    sbatch --export=ALL,MODULE=04 --job-name=mod04 \
           Modules/10_Cluster_Computing/scripts/submit_cannon.slurm"
# (and MODULE=05, MODULE=09 in parallel)
./Modules/10_Cluster_Computing/scripts/pull_from_cannon.sh --go
git add Modules/0{4,5,9}_*/results/
```

Expected wall times on `shared` partition / 16 cores / 32 GB:
- Mod 04: ~5.5 h (resume benefit large — search_2 was 15 h stuck locally)
- Mod 05: ~3 h
- Mod 09: ~8–12 h (5 stages including PowerLaw MASS TOTAL)

### Status after today
- All prior modules committed and real-mode validated through Solutions 01–08 (Sol 04 placeholder still from test mode — will be replaced with cluster result).
- Mod 09 / Sol 09 have test-mode placeholder outputs; cluster run will replace them.
- Local repo is quiet. Laptop back to normal usage (66% idle, 13 GB RAM).

---

## 2026-04-18 — Cannon: env upgrade to autolens 2026.4 (Python 3.12)

### Context
First Cannon-side session in this repo. Cloned to
`/n/holystore01/LABS/hernquist_lab/Lab/rcordova/learning_to_autolens`.

Initial Cannon submissions failed with `AttributeError: module 'autolens' has
no attribute 'RectangularAdaptDensity'` (and friends). I **mis-diagnosed**
this as the scripts referencing fictional classes and spent ~10 commits
substituting them for v2026.2 equivalents. **All of those were wrong** and
have been reverted (see commits `4528c0e..9562693`).

The actual root cause: the Cannon env had `autolens==2026.2.26.4`, but the
scripts and `slam_v2026.py` helper were correctly written against
`autolens==2026.4.13.6` (released April 2026, **requires Python ≥3.12**).
The cluster env was Python 3.11, which capped pip at 2026.2.26.4 — one
minor version too old to have the `RectangularAdapt*` / `reg.Adapt`
classes.

### What actually changed (kept on `main`)
1. **New conda env `autolens312`** built on Cannon with Python 3.12 + the
   latest autolens 2026.4.13.6. Build pattern:

   ```bash
   source /n/sw/Miniforge3-25.3.1-0/etc/profile.d/conda.sh
   conda create -n autolens312 python=3.12 -y
   conda activate autolens312
   python -m pip install --upgrade pip          # IMPORTANT: `python -m pip`,
   python -m pip install autolens jupyterlab \   # not bare `pip` — see below
       astropy "matplotlib<3.9" corner numba
   ```

   **Tricky part:** `~/.local/bin/pip` (a stray Python 3.10 user-pip) shadows
   the env's pip in PATH. Bare `pip install autolens` silently installs into
   `~/.local/lib/python3.10/site-packages/` instead of the active conda env.
   Always use `python -m pip` to bind to the active interpreter.

2. **`submit_cannon.slurm` default env** flipped from `autolens` → `autolens312`.

3. **`.gitignore`** now ignores `logs/` (slurm job stdout/stderr).

### Cluster jobs to resubmit
After this entry, resubmit Modules 04, 05, 09 against the new env. The
existing checkpoint dirs (`Modules/0{4,5,9}_*/output/`) are already on
Cannon and Nautilus will resume from them automatically.

```bash
sbatch --export=ALL,MODULE=04 Modules/10_Cluster_Computing/scripts/submit_cannon.slurm
sbatch --export=ALL,MODULE=05 Modules/10_Cluster_Computing/scripts/submit_cannon.slurm
sbatch --mem=64G --time=48:00:00 --export=ALL,MODULE=09 Modules/10_Cluster_Computing/scripts/submit_cannon.slurm
```

### Resubmission log (autolens312 env)

| JobID | Module | Submitted | Resources | Outcome |
|-------|--------|-----------|-----------|---------|
| 6571564 | 04 | 2026-04-18 22:45 | 32 G / 24 h | **FAILED** at ~1 min — stale checkpoint (see below) |
| 6571565 | 05 | 2026-04-18 22:45 | 32 G / 24 h | Running |
| 6571566 | 09 | 2026-04-18 22:45 | 64 G / 48 h | Running |
| 6572024 | 04 | 2026-04-18 23:14 | 32 G / 24 h | **FAILED** at 80 min — missing `positions_likelihood_list` in slam_v2026 |
| 6584132 | 04 | 2026-04-19 00:37 | 32 G / 24 h | **COMPLETED** 01:26 (48 min wall) |
| 6584987 | 09 | 2026-04-19 00:42 | 64 G / 48 h | **COMPLETED** 01:53 (70 min wall) |

Mod 09 (6571566) confirmed Stage 1 θ_E = 1.601″ before the Stage-3
crash — matches the expected value. Stage 1 ran in 50.4 min, Stage 2
(source_pix run 1) in 39.0 min. Resubmission 6584987 will skip both
stages and resume at Stage 3 with the patched `fit_module09.py`.

### Mod 04 stage timings (6584132, final run)

| Stage | Wall time | Notes |
|-------|-----------|-------|
| CHAIN Search 1 | 0.3 min | SIS — already checkpointed, quick finalize |
| CHAIN Search 2 | ~0 min | SIE+shear — checkpointed, finalize only |
| SLaM SOURCE LP | ~0 min | checkpointed, finalize only; θ_E = 1.550″ |
| SLaM SOURCE PIX run_1 | ~0 min | checkpointed, finalize only |
| SLaM SOURCE PIX run_2 | 5.4 min | fresh run (this was the crash site) |
| SLaM LIGHT LP | 15.0 min | fresh |
| SLaM MASS TOTAL | 27.1 min | fresh; final Isothermal mass model |
| **Total** | **0.80 h (48 min)** | mostly fresh runs from run_2 onward |

stderr had only benign `autofit.SearchWarning`s.

### Mod 09 stage timings (6584987, final run)

| Stage | Wall time | Notes |
|-------|-----------|-------|
| Stage 1 SOURCE LP | 0.1 min | checkpointed, finalize only; θ_E = 1.601″ |
| Stage 2 SOURCE PIX 1 | ~0 min | checkpointed, finalize only |
| Stage 3 SOURCE PIX 2 | 8.4 min | fresh (this was the crash site) |
| Stage 4 LIGHT LP | 17.4 min | fresh |
| Stage 5 MASS TOTAL (PowerLaw) | 43.5 min | fresh; final power-law mass model |
| **Total** | **1.16 h (70 min)** | from Stage 3 onward fresh |

Combined with the prior 6571566 attempt (Stages 1+2, 89 min), total
Mod 09 compute was ~2h39min across two jobs. stderr had only benign
`autofit.SearchWarning`s.

**Mod 04 second attempt (6572024) — what happened.** Cleared Search 1,
Search 2, SLaM SOURCE LP, and SLaM SOURCE PIX run_1 successfully, then
crashed entering SLaM SOURCE PIX run_2 with:

```
autogalaxy.exc.AnalysisException:
    You have begun a model-fit which reconstructs the source using a
    pixelization. However, you have not input a `positions_likelihood_list`
    object. It is likely your model-fit will infer an inaccurate solution.
```

In autolens 2026.4.13.6, pixelized fits now **raise** this exception
instead of just warning (as they did in 2026.2.x). The `slam_v2026.py`
helper only passed `positions_likelihood_list` to `_source_pix_run_1`,
not `_source_pix_run_2` or `_light_lp_run`. `fit_module09.py` was also
missing it at Stage 3 and Stage 4.

**Fix (commit `869a91d`).** Added
`<prev_pix_result>.positions_likelihood_from(factor=3.0, minimum_threshold=0.2)`
to all four pixelized analyses. Resubmitted Mod 04 as 6584132 — will
resume from the source_pix[1] checkpoint and pick up at run_2.
Mod 09 (6571566, still running Stage 1) will hit the same exception
when it reaches Stage 3; when it does, resubmit against the patched
`fit_module09.py` and Nautilus auto-resumes from the Stage 2 checkpoint.

**Mod 04 first attempt (6571564) — what happened.** Search 1 resumed from
`output/.../search_1_sis_nolenslight/.../checkpoint.hdf5` (53 MB, saved by
the prior old-env run at 22:18). With 263 k existing likelihood calls
already in the state, Nautilus entered bound-construction mode and
hit `LinAlgError: Matrix is not positive definite` in
`np.linalg.cholesky` inside `nautilus/bounds/basic.py:308`. The
checkpoint was saved under autolens 2026.2.26.4 / nautilus-sampler at
that version; autolens 2026.4.13.6 pulls in a newer nautilus whose
internal state layout is incompatible — when it thawed the old live
points into the new bound logic, the resulting covariance was singular.

**Fix.** Archived the stuck file as `checkpoint.hdf5.old-env-bak`
(not deleted — recoverable) and resubmitted as 6572024. Fresh bound
construction from scratch should converge normally.

**Generalization.** When upgrading autolens across a minor version with
existing Nautilus checkpoints on disk, **assume checkpoints are not
portable**. Sweep `Modules/*/output/**/checkpoint.hdf5` and move them
aside before the first job under the new env. Otherwise LinAlgError or
silent numerical corruption are the most likely failure modes.

Prior failed attempts on the old `autolens` env (all cancelled/failed,
jobs 6466300/6466689/6466690 and 6546267/6546268/6546269) are visible
via `sacct -u rcordova --starttime=2026-04-18`. Those are the signal
that led to building `autolens312`; leave them in the accounting trail.

### Lessons learned (for future sessions)

- **Diagnose env first, code second.** A missing class is far more often a
  version/interpreter mismatch than a genuine API fiction. Before touching
  scripts, check:
  - `pip index versions autolens` against the env's Python version
  - the class's existence in the upstream repo at the exact installed
    version tag
  - whether an `salloc` test-node reproduces the import failure
- **`python -m pip`, always.** On Cannon, `~/.local/bin/pip` points at an
  old user Python and hijacks `pip install` inside conda envs without any
  warning. The symptom is "pip install succeeds, import fails" — exactly
  what happened the first time I tried to install into `autolens312`.
- **Paper trail for reverts.** 8-commit `git revert` chains are
  acceptable when the reverted work was wrong and the intermediate code
  was pushed; one unified "undo" commit would lose the diff detail that
  explains *why* each substitution was wrong. All reverts this session
  are `4528c0e..9562693` on `main`.
- **PAT hygiene.** This session pushed ~10 times with a PAT embedded in
  a `GIT_ASKPASS` temp script containing the token in plaintext. The
  token now lives in this transcript and the command-history of each
  invocation. Rotate at a convenient moment and move to either an SSH
  deploy key or `git credential-store` with mode-600 credentials.

### TODO once all three jobs finish
1. **Audit each run.** Confirm `logs/autolens_<jobid>.err` is empty/warnings
   only, and that the pipeline reached its final stage (`grep "done in"
   logs/autolens_<jobid>.out`). Sanity-check best-fit θ_E ≈ 1.6″ for
   simple/simple__no_lens_light.
2. **Commit results.** Pull via `scripts/pull_from_cannon.sh` from the
   laptop, then `git add Modules/0{4,5,9}_*/results/`. The `output/` dirs
   stay git-ignored — only the curated `results/` artifacts are tracked.
3. **If a stage still crashes:** check the traceback against the installed
   2026.4.13.6 API. Avoid the previous mistake of assuming a class is
   "fictional" before checking *which version* defines it (`pip index
   versions <pkg>`) and *which Python* the env is on. The autolens release
   cadence is monthly — keep an eye on which minor version the upstream
   workspace and SLaM helpers target.
4. **Lessons logged for future sessions** (CLAUDE.md note):
   - When `AttributeError: module 'X' has no attribute 'Y'` appears in
     PyAutoLens, **first check `pip index versions X` and the env's Python
     version**, not GitHub code search. The class probably exists in a
     newer release that requires a newer Python.
   - On Cannon, **always `python -m pip`**, never bare `pip`, when working
     in a conda env. The user-level `~/.local/bin/pip` will silently
     hijack installs otherwise.

---

## 2026-04-19 — Post-downtime cleanup: `export_results.py` rewrite + Mod 04/09 artifacts committed

Cannon came back up; job state on resume:
- **6584132 (Mod 04):** COMPLETED 04-19 01:25 (49 min wall)
- **6584987 (Mod 09):** COMPLETED 04-19 01:52 (71 min wall)
- **6571565 (Mod 05):** REQUEUED, restarted 08:35, still running 2h15m in (fresh Nautilus — the downtime invalidated the prior checkpoint state)

### `export_results.py` fix — discovery + v2026.4 API
Both completed jobs ran `export_results.py` as their post-process step, but the original script was broken under autolens 2026.4.13.6:

1. **Discovery broken.** `rglob("files/search_internal")` only matched *stale mid-run* directories (one partial search_pix[1] on Mod 09, zero on Mod 04). Nautilus cleans `search_internal/` on successful finalization, so completed searches were invisible. → Switched to `rglob("files/samples_summary.json")` which is the durable "completed" marker.
2. **Corner-plot API gone.** `autolens.plot.MatPlot1D` / `NestPlotter` were removed in 2026.4 — `MatPlot1D in dir(aplt) == False`. → Switched to `autofit.plot.corner_cornerpy(samples, path, filename, format)` which is the canonical new API.
3. **Fit-subplot rebuild unnecessary.** Original called `aplt.FitImagingPlotter(fit=result.max_log_likelihood_fit).subplot_fit()`, but `af.SearchOutput(...).result` returns `None` in 2026.4 (the result-pickle contract changed). → Copy the pre-rendered `<hash>/image/fit.png` that autolens writes during every search. Simpler and more faithful to what the fit actually looked like.
4. **Info file location.** `model.info` lives at `<hash>/model.info`, not under `files/`. Also added `model.results` (best-fit table with 1σ/3σ uncertainties) to the export bundle.

### Artifacts generated
Cleared stale `Modules/09_.../results/source_pix[1]/` and re-ran export for both modules.

| Module | Stages | Total size |
|--------|--------|-----------|
| 04 | search_1_sis, search_2_sie, source_lp, source_pix[1/2], mass_total | 23 MB |
| 09 | source_lp, source_pix[1/2], light, mass_total | 30 MB |

Each stage has `{fit_subplot.png, corner.pdf, info.txt, model_results.txt, samples.csv, samples_summary.json, summary.json}`.

Sanity check on Mod 04 best-fit θ_E from `model_results.txt`: `einstein_radius = 1.5969 (1.5964, 1.5975)` at 1σ, matches expected 1.60″ for `simple__no_lens_light` dataset. Mod 09 `source_lp` Stage 1 θ_E = 1.601″ previously confirmed.

### Status
- Mod 04 + Mod 09 `results/` committed (this change).
- Mod 05 (6571565) running; will re-run `export_results.py` on its output once `COMPLETED` and commit separately.
- `output/` remains git-ignored — only curated `results/` tracked, per the "results viewer" pattern.

### Mod 05 (6571565) — deadlocked, cancelled + resubmitted (6697218)

While exporting Mod 04/09 I spot-checked 6571565 and found it catastrophically stalled:

- **Wall clock:** 11 h 20 min since 08:35 this morning
- **Nautilus stdout:** Only the "Starting new Nautilus non-linear search (no previous samples found)" line + the column header for the status table. **Zero iteration rows in 11 hours.**
- **Log file mtime:** 08:37 — hasn't been written to since startup
- **Output tree:** `search2_pixelized_source/<hash>/files/search_internal/` exists but no `checkpoint.hdf5`, no `samples.csv`. Search1 (parametric) had already completed cleanly.
- **`sstat` AveCPU = 16:26** over 680 min wall = **2.4% CPU utilization** across the 16-core allocation
- **Process inspection via `srun --jobid=6571565 --overlap ps -u rcordova`:** all 16+ python workers in `S` (sleeping) state at 0% CPU

**Root cause: OpenBLAS × multiprocessing fork-lock deadlock.** The Nautilus log at startup had already printed its canonical warning:

```
OPENBLAS_NUM_THREADS
MKL_NUM_THREADS
OMP_NUM_THREADS
VECLIB_MAXIMUM_THREADS
NUMEXPR_NUM_THREADS
(not set to 1)

This can lead to performance issues, because both the non-linear search and
libraries that may be used in your `log_likelihood_function` evaluation may
attempt to parallelize over all cores available.
```

The old `submit_cannon.slurm` ignored that warning. For Mod 04 and Mod 09 it didn't bite (their pixelized searches are shorter; the deadlock is probabilistic per bound-construction and scales with likelihood call count). For Mod 05's `search2_pixelized_source` it locked up on the very first bound construction and never recovered.

### Fix (commit this change)
`submit_cannon.slurm` now exports the five BLAS/threading env vars before `srun`:

```bash
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

### Resubmission
- `scancel 6571565` (final `TIME=11:21:48`, state `CG`)
- `rm -rf output/module_05/search2_pixelized_source/` — no salvageable state (no checkpoint)
- Kept `output/module_05/search1_parametric_source/<good_hash>/` since search1 had completed samples
- `sbatch --export=ALL,MODULE=05 Modules/10_Cluster_Computing/scripts/submit_cannon.slurm` → **6697218** (running on holy7c04101 as of 19:58)

### Generalization
Hot-running Nautilus jobs without `*_NUM_THREADS=1` are a time bomb. The 04-18 Cannon migration checklist should include a one-liner check:

```bash
grep -q "OPENBLAS_NUM_THREADS=1" Modules/10_Cluster_Computing/scripts/submit_cannon.slurm
```

Added as a memory entry for future sessions (`feedback_blas_threading.md`).
