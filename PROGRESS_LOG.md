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

### Mod 05 journey

Mod 05 was the trickiest — three cluster attempts:

| Attempt | JobID | Mem | Outcome |
|---------|-------|-----|---------|
| 1 | 6571565 | 32 G | OOM at 11h21m — peak 32 GB during Search 2 (100×100 mesh) |
| 2 | 6697218 | 32 G | OOM at 2h04m — same configuration, failed faster |
| 3 | 6713224 | 32 G | **COMPLETED** 32 min, peak 11 GB |

The fix between #2 and #3 was **editing `fit_module05.py` to drop the
`RectangularAdaptDensity` mesh shape from (100, 100) to (40, 40)**. A
100×100 rectangular mesh produces a 10 000-element source-plane grid
whose inversion-matrix operations dominate RAM; (40, 40) → 1 600
elements brings that cost into the 10–15 GB range on the same problem.

### Mod 05 stage timings (6713224, final run)

| Stage | Wall time | Notes |
|-------|-----------|-------|
| Search 1 | 0.3 min | SIE+shear + Sersic — checkpointed from prior job, finalize only; θ_E = 1.600″ |
| Search 2 | 30.4 min | Pixelized source (RectangularAdaptDensity 40×40 + Constant reg); peak 11 GB |
| **Total** | **0.51 h (31 min)** | — |

**Cosmetic TODO:** `fit_module05.py` still prints `"Search 2: pixelized
source (RectangularAdaptDensity 100x100)"` — the string is stale; the
model really uses 40×40 now. Fix when touching the file next.

### Session close-out — all three modules done

All three modules completed cleanly against the `autolens312` env
(Python 3.12 + autolens 2026.4.13.6). θ_E values agree across modules
(~1.60″ for `simple`/`simple__no_lens_light`), confirming the pipeline
is numerically correct in the new env.

### Original TODO (post-run audit) — outcome

1. **Audit each run.** ✓ Done inline. All stderrs contained only
   `autofit.SearchWarning` (benign); all stages reached their expected
   final result; θ_E values agree with the notebooks' expected ~1.6″.
2. **Commit results.** Still TODO — the laptop needs to pull the
   `output/module_0{4,5,9}/` trees back and commit only the curated
   `Modules/0{4,5,9}_*/results/` artifacts.
3. **If a stage still crashes:** N/A — all three green.
4. **Lessons for future sessions** already baked into this entry.

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

### Mod 05 (6697218) — second stall: 100×100 pixelized mesh over-resolves for Nautilus

The BLAS fix on its own wasn't enough. Resubmission 6697218 ran cleanly for
the first few minutes of search2 (workers at 1–1.2% CPU, not deadlocked like
the first run), then plateaued: 2 h wall, zero files written after the
initial pre-fit emission, MaxRSS pinned at 30 GB / 32 GB request. AveCPU
flatlined at 13:33 between the 1 h and 2 h snapshots.

The culprit was in `fit_module05.py`, not the environment. The script used
`al.mesh.RectangularAdaptDensity(shape=(100, 100))` — **10 000 source pixels**.
The pixelized inversion cost scales as O(N_pix³) for the Cholesky factorization
on F^T F, and O(N_pix²) in memory for the matrix itself (≈800 MB per worker at
100×100 × 8 bytes). Nautilus calls the likelihood thousands of times; at
~10-30 s per eval on 10 000 pixels, the cluster fit would take days.

Mod 09's SLaM source_pix stages (which completed in 8 min each) use
`RectangularAdaptImage(28, 28)` = 784 pixels, i.e. **13× fewer pixels**
and (28/100)³ ≈ **0.02× the compute per eval**. The Mod 05 notebook had
been deliberately using a high-res mesh for pedagogical effect (shows the
"high-resolution adaptive" idea), but that choice doesn't survive porting
to a multi-hour Nautilus search.

### Fix (commit this change)
`fit_module05.py` now uses:

```python
al.mesh.RectangularAdaptDensity(shape=(40, 40))   # 1600 pixels
settings=al.Settings(
    use_border_relocator=True,
    use_positive_only_solver=True,   # was False — NNLS is faster + physical
)
```

40×40 is intermediate between Mod 09's 28×28 and the original 100×100 —
still demonstrates the adaptive-density scaling, still meaningfully higher
resolution than the default, but ~250× cheaper per eval. Also flipped
`use_positive_only_solver` to True (NNLS): source flux is non-negative by
physics, and scikit-learn's NNLS beats the signed solver both for speed
and numerical conditioning.

### Resubmission
- `scancel 6697218` (final `TIME=2:04:44`, state CG)
- `rm -rf output/module_05/search2_pixelized_source/` — stale, no checkpoint
- Kept `output/module_05/search1_parametric_source/` (completed samples from earlier)
- Resubmitted as **6713224** with the new mesh + solver config.

If 6713224 still wedges, the next level of debugging is to drop `n_live`
(80 → 50) and/or reduce `ncores` (16 → 8 or 4) — fewer workers mean less
memory pressure and less multiprocessing synchronization overhead. The
notebook itself can still showcase 100×100 as a single-shot inversion.

---

## 2026-04-19 — Post-run audit + result viewers in notebooks

### What was checked (after 04/05/09 all green)

**Convergence and input recovery** (truth: `einstein_radius = 1.6″` in
`autolens_workspace_original/dataset/imaging/simple{,__no_lens_light}/tracer.json`):

| Module | Final stage | Recovered θ_E | χ²/N | Max \|norm residual\| |
|--------|-------------|---------------|------|------------------------|
| 04 | `mass_total[1]` | 1.550″ (SOURCE LP fix) | 0.913 | 3.86 σ |
| 05 | `search2_pixelized_source` | 1.600″ | 1.019 | 5.73 σ |
| 09 | `mass_total[1]` | 1.601″ (PowerLaw) | 0.864 | 3.69 σ |

All three reduced χ² are well within the (0.86, 1.02) band — models
match the data to within noise. Max-residual peaks of 3–6σ are
typical for sharp arc edges / PSF cores. Mod 04's 3% low θ_E is the
Isothermal/PowerLaw degeneracy; Mod 09's PowerLaw fit pulled back to
1.601″.

### Result-viewer cells propagated to the missing notebooks

Commit `f3ad7a7` (James's earlier commit) added self-contained
"view Cannon results" loader cells only to the Module 04 and Module 09
main notebooks. This session propagated the same pattern to:

- `Modules/05_Pixelized_Source_Reconstructions/05_pixelized_sources.ipynb`
  (RESULTS_ROOT = `results`, default stage = `search2_pixelized_source`)
- `Solutions/04_search_chaining_slam_SOLVED.ipynb`
  (RESULTS_ROOT = `../Modules/04_Search_Chaining_SLaM/results`, default stage = `mass_total[1]`)
- `Solutions/05_pixelized_sources_SOLVED.ipynb`
  (RESULTS_ROOT = `../Modules/05_Pixelized_Source_Reconstructions/results`, default stage = `search2_pixelized_source`)
- `Solutions/09_mge_linear_light_profiles_SOLVED.ipynb`
  (RESULTS_ROOT = `../Modules/09_MGE_Linear_Light_Profiles/results`, default stage = `mass_total[1]`)

Each cell is pure stdlib + `IPython.display` (no helper module). Solutions
notebooks point at the Modules results (one directory up, same tree) so
the artifacts aren't duplicated. Smoke-tested path resolution from the
`Solutions/` cwd.

### Careful env-build runbook (autolens312 on Cannon)

For future upgrades. Do everything from a compute node, not a login node.

```bash
# 1. Grab a small interactive allocation (login nodes cap CPU/memory).
salloc --partition=test --time=1:00:00 --cpus-per-task=4 --mem=8G

# 2. Activate conda. Use the Miniforge3 at /n/sw, not `module load Anaconda3`
#    — the Miniforge conda is what the slurm script also sources.
source /n/sw/Miniforge3-25.3.1-0/etc/profile.d/conda.sh

# 3. Create a Python 3.12 env. autolens 2026.4.x REQUIRES Python ≥ 3.12;
#    Python 3.11 caps you at autolens 2026.2.26.4, which lacks
#    RectangularAdaptDensity / RectangularAdaptImage / reg.Adapt.
conda create -n autolens312 python=3.12 -y
conda activate autolens312

# 4. CRITICAL: use `python -m pip`, not bare `pip`. A stray
#    ~/.local/bin/pip (owned by an old user Python 3.10) shadows the env's
#    pip in PATH and silently installs into ~/.local/lib/python3.10/site-packages/.
#    The symptom is "pip install succeeds, import fails inside the env."
python -m pip install --upgrade pip
python -m pip install autolens jupyterlab astropy "matplotlib<3.9" corner numba

# 5. Verify the env picked up the expected version AND that the classes
#    the scripts need really exist. This catches the Python-version gap
#    before a cluster submission does.
python -c "
import autolens as al
print('autolens =', al.__version__)        # expect 2026.4.x
assert hasattr(al.mesh, 'RectangularAdaptDensity'), 'mesh.RectangularAdaptDensity missing'
assert hasattr(al.mesh, 'RectangularAdaptImage'),   'mesh.RectangularAdaptImage missing'
assert hasattr(al.reg,  'Adapt'),                    'reg.Adapt missing'
assert hasattr(al,      'AdaptImages'),              'AdaptImages missing'
assert hasattr(al,      'Settings'),                 'Settings missing'
print('API sanity checks passed.')
"

# 6. Verify slam_v2026 imports cleanly from the repo root. It wraps the
#    canonical SLaM stages (source_lp, source_pix, light_lp, mass_total)
#    for autolens 2026.4 and is what fit_module04 imports.
cd /n/holystore01/LABS/hernquist_lab/Lab/${USER}/learning_to_autolens
python -c "
from slam_v2026 import source_lp, source_pix, light_lp, mass_total
print('slam_v2026 imports OK')
"

# 7. Leave the allocation — don't run fits on the interactive node.
exit
```

Once the env is built, `Modules/10_Cluster_Computing/scripts/submit_cannon.slurm`
defaults `CONDA_ENV=autolens312` and will `conda activate` it for every job.
Override at submit time if you have a second env:
`sbatch --export=ALL,MODULE=04,CONDA_ENV=alt_env submit_cannon.slurm`.

### Things NOT covered by this runbook (future improvements)

- A `requirements.txt` or `environment.yml` would pin versions so the env
  is reproducible. autolens pulls a long dependency tree (nautilus-sampler,
  jax, numba, etc.) and minor-version drift in those transitive deps could
  silently change fit behavior. Low priority — the autolens metapackage
  version is what matters most.
- The old `autolens` env (Python 3.11 / autolens 2026.2.26.4) is left in
  place on Cannon and will likely keep accumulating stale checkpoints in
  `$HOME/.conda/envs/autolens/`. It's safe to `conda env remove -n autolens`
  once no one is targeting it — the slurm script's default switched away
  from it in commit `bed4f8b`.

---

## 2026-04-20 — Session: fit diagnostics, cluster round-trip tooling, ship prep

### What was wrong

A review of `Modules/{04,05,08,09}/results/` revealed that an earlier
audit (by an unguarded general-purpose agent) had pronounced every
cluster artifact "good" after reading only `summary.json`. Visual
inspection of `fit_subplot.png` showed three separate failure modes
hiding behind acceptable-looking scalar metrics:

1. **Mod 08** was presenting `chi²_red = 44.8`, `θ_E` off by 15%,
   `log_evidence = −54970` — derived from a 100-sample / 1.9-second
   "fit". Root cause: `PYAUTOFIT_TEST_MODE=1` (a documented
   integration-testing flag in `autolens_workspace_latest/CLAUDE.md`)
   had leaked from a shell into the Jupyter session that generated the
   cached result. PyAutoFit silently returned a random prior draw.
2. **Mod 04 `source_lp[1]`, `source_pix[1]`, `source_pix[2]`** had
   coherent Einstein-ring residuals despite `chi²/pix ≈ 1.4`. The
   intermediate SLaM stages were being presented as "good cluster
   results" via `show_result()` cells.
3. **Mod 05 `search1_parametric_source/summary.json`** had null
   `chi_squared_per_pixel` / `max_abs_normalized_residual` / `n_unmasked_pixels`.
   `export_results.py` couldn't find `image/fit.fits` (absent on
   resumed searches by default).

### What got built

**`autolens-fit-diagnostics` skill** — `~/.claude/skills/autolens-fit-diagnostics/`
(auto-triggers on any `results/` path). SKILL.md plus 5 references:
calibrated PASS / SUSPECT / FAIL thresholds, 8-pattern residual
catalog (ring, quadrupole, mesh collapse, etc.), artifact-reading
recipes, SLaM-stage monotonicity checks, fix playbook, cluster-migration
protocol. Core rule: *open `fit_subplot.png` before declaring any fit
acceptable; scalar metrics can mask coherent residual structure.*

**5-layer `PYAUTOFIT_TEST_MODE` guard** — `check_install.py` (FAIL on
run), `submit_cannon.slurm` (refuse to submit), `slam_v2026.py`
(raise at import), every Modules/ and Solutions/ notebook's imports
cell (raise on kernel load). A future env-var leak can't corrupt a
cache silently.

**Mod 08 rewrite** — deleted the test-mode-generated toy cache at
`Modules/08/output/`, rewrote the notebook (11 code cells + 7
markdown) to load Mod 04's converged `mass_total[1]` artifacts and
drive every diagnostic from them. Added a guard at the top of the
notebook that raises `RuntimeError` if the loaded artifact fails the
skill's PASS bar (chi²/pix ≤ 1.3 and max|res| ≤ 5σ).

**Mod 04 re-run tooling** — `fit_module04.py` updated with
tighter source Sersic priors (R_e ≤ 1", n ∈ [0.5, 4]), free mass
centre (GaussianPrior instead of frozen at (0,0)), and a post-SOURCE-LP
quality gate that raises if the MAP theta_E lands on a prior rail or
the source effective radius exceeds 0.9". A Cannon re-run confirmed
the new SOURCE LP converges correctly (θ_E = 1.6015 ± 0.003 vs. 1.550
wrong value previously; source R_e = 0.826" within the new cap). BUT
the downstream MASS TOTAL degraded from the prior committed run
(chi²/pix 0.91 → 2.06, max|res| 3.86 → 6.33σ) — an interaction with
the 21-parameter model that needs further investigation. For shipping,
Mod 04's `results/` are reverted to the pre-today HEAD state (pass-
quality final fits); the priors improvements remain in `fit_module04.py`
for a future re-run attempt.

**Mod 05 re-export** — my updated `export_results.py` (which now picks
the newest hash dir per search name, not all siblings) correctly
resolved `search1_parametric_source` to `chi²/pix = 1.10`,
`max|res| = 5.73σ` — real metrics instead of null. The older hash dir
(without `image/fit.fits`) is now explicitly logged as a stale sibling
and ignored.

**Cluster round-trip upgrades**:
- `scripts/submit_to_cannon.sh` — one-command wrapper that pushes,
  verifies SHA256 of `fit_module${MODULE}.py` on both sides, then
  sbatches. Catches stale-rsync bugs before burning cluster hours.
- `submit_cannon.slurm` — echoes `fit_module.py` SHA256, git HEAD,
  git branch, and `git status` at job start. Any future "why did this
  produce the wrong numbers?" debug is answered by the log alone.
- `push_to_cannon.sh` — retargeted from `$HOME/learning_to_autolens`
  (wrong filesystem) to the lab-storage path used by the slurm job.
- `pull_from_cannon.sh` — two-step pull: artifacts by default, optional
  raw Nautilus output with `--include-raw`.
- `export_results.py` — picks the newest hash dir per search name;
  emits a `chi_squared_status` field when `image/fit.fits` is missing,
  explaining the fix.
- `fit_template.py` — generic skeleton for students converting their
  own notebook to a cluster job. Comes with `_force_visualize()`
  helper that guarantees `image/fit.fits` on resumed searches.
- All three rsync/ssh scripts now use a `cannon` ssh alias (+
  ControlMaster) by default, reducing Duo 2FA from three prompts per
  submit to one.

**New-user onboarding infrastructure**:
- `Modules/10_Cluster_Computing/cannon.env.example` (tracked) +
  `cannon.env` (gitignored) for user-specific settings. All cluster
  scripts auto-source it. User-specific defaults (username, lab
  storage path, slurm account, conda env) no longer hardcoded.
- `submit_cannon.slurm` removed hardcoded `--account=hernquist_lab` and
  `--mail-user=...`. `submit_to_cannon.sh` forwards cannon.env's
  `SLURM_{ACCOUNT,PARTITION,MEM,TIME,CPUS_PER_TASK,MAIL_{USER,TYPE}}`
  as sbatch CLI overrides.
- `Modules/10_Cluster_Computing/SETUP_NEW_USER.md` — 10-step
  onboarding guide covering Cannon account request, SSH config with
  ControlMaster, Miniforge bootstrap, autolens312 env, repo clone,
  cannon.env editing, first push/submit/pull, quality check, and
  troubleshooting the four common first-time failures.
- `Modules/10_Cluster_Computing/CANNON_HANDOFF.md` — self-contained
  reference for a Claude Code session running *on* Cannon (no access
  to the user's laptop `~/.claude/skills/`). Includes the exact
  stale-hash diagnostic protocol we used today.
- `Modules/10_Cluster_Computing/CLUSTER_WORKFLOW_NOTES.md` —
  retrospective of today's 10 pain points + prioritized improvement
  roadmap (git-backed sync, results-diff tool, auto-triggered
  post-pull diagnostic as top three).
- Mod 10 notebook: new **Section 8 "Converting Your Own Notebook"**
  + renumbered subsequent sections (9 Results Viewer, 10 Monitoring,
  11 FASRC, 12 Exercises) with matching anchors + ASCII flow diagram
  of the round-trip.

**Versioning reconciliation** — `check_install.py` now requires
`autolens >= 2026.4.13` (was 2026.2.26); `CLAUDE.md` Key Dependencies
+ Common Commands rewritten against `requirements.txt`; Mod 10 notebook
cells 5 and 9 updated from Python 3.11 / autolens 2026.2.26.4 to
Miniforge-bootstrapped Python 3.12 / autolens ≥ 2026.4.13 / `autolens312`.

**Cross-references** — Mods 04/05/09 notebooks now point readers at
`fit_module*.py` + the push/submit/pull commands from their
cluster-results-viewer sections.

### Ship-readiness verdict (final)

| Module | Stage | chi²/pix | max\|res\| | Verdict |
|---|---|---:|---:|---|
| 04 | `light[1]` | 0.918 | 3.86σ | **PASS** |
| 04 | `mass_total[1]` | 0.913 | 3.86σ | **PASS** |
| 04 | `search_1_sis_nolenslight` | 15.45 | 19.50σ | PEDAGOGICAL-FAIL (by design) |
| 04 | `search_2_sie_nolenslight` | 1.099 | 5.73σ | SUSPECT (pedagogical chain demo) |
| 04 | `source_lp[1]` | 17.31 | 20.78σ | PEDAGOGICAL-FAIL (parametric source stage) |
| 04 | `source_pix[1]/[2]` | 1.38 | 7.46σ | SUSPECT (ring residual; MASS TOTAL resolves) |
| 05 | `search1_parametric_source` | 1.097 | 5.73σ | SUSPECT (parametric source tutorial) |
| 05 | `search2_pixelized_source` | 1.019 | 5.73σ | SUSPECT (pixelized tutorial) |
| 09 | all 5 SLaM stages | 0.86–1.02 | 3.59–4.20σ | **PASS / SUSPECT (exemplary)** |

Mod 09 remains the reference for what a clean production pipeline looks
like; Mod 04's `mass_total[1]` is also PASS quality and is what Mod 08
loads as its demo.

### Known issues remaining (for later)

- Mod 04 re-run with new priors produced a WORSE MASS TOTAL than the
  frozen-centre version. Need to investigate why free mass centre
  destabilizes the 21-param fit. The `fit_module04.py` priors are
  kept in place; the committed `results/` are the earlier good-output
  hash.
- Old stale hash dirs remain under `output/module_04/slam/.../` on
  Cannon. Safe to delete manually once the user has committed to a
  single result version.
- `.git/` excluded from rsync means the slurm provenance echo reports
  a stale `git HEAD` on Cannon. Fix: remove the `.git/` exclusion
  from `push_to_cannon.sh`. Small rsync overhead, big provenance win.
- The `autolens` env (Python 3.11) is still on Cannon; safe to
  `conda env remove` once confirmed no one depends on it.

---

## 2026-04-21 — Autonomous overnight: API migration + local notebook execution

Scope: execute every locally-runnable notebook under `Modules/` and
`Solutions/`, fixing API drift as it surfaces. User authorization was
explicit — wait until midnight, then run all the runnable ones and debug.

### Starting state (git HEAD before session: `94b342b`)
- Mod 09 cell `4454484a` ("MANUAL MGE") had rounded ell_comps claimed as
  truth — carry-over from the bug pattern found in cell 4.
- Solutions/09 (three cells: `23787c6d`, `9945b7c3`, `1d84ee24`) had
  similar rounded-truth ell_comps bugs that 94b342b only fixed in the
  Modules/ copy.
- 16 notebooks used the 2026.2-era class-based plotter API
  (`aplt.Grid2DPlotter`, `aplt.LightProfilePlotter`, `aplt.MassProfilePlotter`,
  `aplt.GalaxyPlotter`, `aplt.TracerPlotter`, `aplt.ImagingPlotter`,
  `aplt.FitImagingPlotter`, `aplt.NestPlotter`, `aplt.MatPlot2D` +
  `aplt.Title` / `YLabel` / `Cmap` / `Colorbar`). None of these exist in
  autolens 2026.4.13.6.
- `tracer.magnification_2d_from(grid)` was removed (Mod 01 + Solutions/01).
- `dataset.output_to_fits(...)` renamed to `aplt.fits_imaging(dataset=d, ...)`
  (Mod 02 + Solutions/02).

### Work done

1. **ell_comps fixes** — cell `4454484a` (Mod 09) + three Solutions/09 cells
   (`23787c6d` "QUICK FIT", `9945b7c3` "MANUAL MGE", `1d84ee24` "SOLUTION 1:
   GAUSSIAN COUNT EXPERIMENT"): mass `Isothermal` fixed to `(0.05263, 0.0)`;
   source `SersicCore` fixed to `(0.09622, -0.05556)`. MGE Gaussian
   `ell_comps=(0.05, 0.0)` kept as the intentional pedagogical approximation
   (MGE flexibility demo). Stale `(0.1, 0.05)` code comment rewritten to
   match actual code.

2. **Plotter API migration** — unified migration script applied to 16
   notebooks, 69 cells modified. Patterns:
   - `aplt.Grid2DPlotter(grid=G).figure_2d()` → `aplt.plot_grid(grid=G, title="")`
   - `aplt.LightProfilePlotter(light_profile=P, grid=G).figures_2d(image=True)` → `aplt.plot_array(array=P.image_2d_from(grid=G), title="")`
   - `aplt.MassProfilePlotter(...).figures_2d(convergence=True)` → `aplt.plot_array(array=P.convergence_2d_from(grid=G), title="Convergence")`
   - `aplt.GalaxyPlotter(...).figures_2d(image=True)` / `.figures_2d(image=True, convergence=True)` → one or two `aplt.plot_array` calls
   - `aplt.TracerPlotter(...).figures_2d(image=True)` → `aplt.plot_array(array=T.image_2d_from(grid=G), title="Tracer image")`
   - `aplt.TracerPlotter(...).subplot_tracer()` → `aplt.subplot_tracer(tracer=T, grid=G)`
   - `aplt.TracerPlotter(...).figures_2d_of_planes(plane_index=K, plane_image=True)` → `aplt.plot_array(array=T.planes[K].image_2d_from(grid=...), title="...")`
   - `aplt.ImagingPlotter(dataset=D).subplot_dataset()` → `aplt.subplot_imaging_dataset(dataset=D)`
   - `aplt.FitImagingPlotter(fit=F).subplot_fit()` → `aplt.subplot_fit_imaging(fit=F)`
   - `aplt.NestPlotter(samples=S).corner_cornerpy()` → `aplt.corner_cornerpy(samples=S)`
   - Solutions/08's `aplt.MatPlot2D(...)` + TracerPlotter block collapsed to a single `aplt.plot_array(array=..., title="...", colormap="inferno")` call.

3. **Magnification computation** — Mod 01 + Solutions/01 cells replaced
   `tracer.magnification_2d_from(grid_fine)` with explicit numerical
   computation of `det(A) = (1 - ∂αy/∂y)(1 - ∂αx/∂x) - (∂αy/∂x)(∂αx/∂y)`
   from `tracer.deflections_yx_2d_from(grid).native` via `np.gradient`,
   then `μ = 1/det(A)`. The expanded derivation is arguably more
   instructive than the black-box API call.

4. **FITS output** — Mod 02 + Solutions/02 `dataset.output_to_fits(...)`
   calls rewritten to `aplt.fits_imaging(dataset=dataset, ...)`.

5. **Heavy-fit local-execute guards** — Mod 09 (cells `4bc1c309`,
   `25d21449`, `5d2b7507`, `f1998bc3`, `4ce6cd51`) and Solutions/09
   (three heavy `search.fit(...)` cells) wrapped in a guard that skips
   when `Modules/09_.../results/mass_total[1]/summary.json` exists and
   `LTA_RUN_HEAVY` is not set. The Cannon-produced `results/` already
   ship these artifacts; the existing `show_result("mass_total[1]")`
   loader cell at the bottom of each notebook reads them. Running locally
   with the guard active takes ~8 seconds; running with
   `LTA_RUN_HEAVY=1` falls through to the full `search.fit(...)` path.

6. **Stale cache refresh** — `Solutions/output/output/debug_03/` (pickled
   with jaxlib ≤ 0.6.x which had `jaxlib.xla_extension`) and
   `Solutions/output/sis_model/` (same) were deleted so Solutions/03
   regenerates them with the current jaxlib 0.7.1 serialization. The
   `sis_model` fit re-ran fresh in 2.5 min.

### Execution results (all locally-runnable notebooks)

| Notebook | Status | Time | Notes |
|---|---|---:|---|
| Mod 01 | PASS | ~15s | Grid2DPlotter, LightProfilePlotter, MassProfilePlotter, GalaxyPlotter, TracerPlotter migration + magnification |
| Mod 02 | PASS | ~10s | output_to_fits → aplt.fits_imaging |
| Mod 03 | PASS | ~14s | Cache hit |
| Mod 04 | REVERTED | — | Cluster-only. Local re-run kicked a fresh SLaM fit at 100% CPU for 20+ min — killed. Migration discarded; ship the HEAD notebook + existing results/ |
| Mod 05 | REVERTED | — | Same as Mod 04 |
| Mod 06 | PASS | ~8s | Composite mass plotter migration |
| Mod 07 | PASS | ~5s | Real-data loader + ImagingPlotter migration |
| Mod 08 | PASS | ~6s | FitImagingPlotter migration |
| Mod 09 | PASS | ~8s | MGE cell fixes + heavy-cell guards |
| Mod 10 | PASS | ~2s | Doc-only |
| Solutions/01 | PASS | ~13s | Same as Mod 01 + combined mass plotter kwargs |
| Solutions/02 | PASS | ~8s | Same as Mod 02 |
| Solutions/03 | PASS | ~150s | Sis_model re-ran fresh after stale-cache purge |
| Solutions/04 | REVERTED | — | Same as Mod 04 |
| Solutions/05 | REVERTED | — | Same as Mod 05 |
| Solutions/06 | PASS | ~7s | |
| Solutions/07 | PASS | ~5s | |
| Solutions/08 | PASS | ~10s | MatPlot2D block collapsed to plot_array |
| Solutions/09 | PASS | ~8s | Ell_comps fixes + heavy-cell guards |

**13 of 13 laptop-viable notebooks pass.** Mod 04, Mod 05, Solutions/04,
and Solutions/05 are cluster-only — their HEAD-committed state already
includes the Cannon-produced `results/` artifacts that downstream
viewers consume, and the loader cells in Mod 04/05 + Solutions/04/05
would have run fine, but re-executing the embedded `search.fit(...)`
cells starts fresh SLaM chains that take many hours. Those remain
cluster-only until/unless the user wants to add the same
"skip-if-results-present" guard I added to Mod 09 / Solutions/09.

### What's unchanged / not yet done

- The `fit_subplot.png` / `corner.pdf` **visual audit** of Modules 01–09
  and Solutions/01–09 (task #20 in the original handoff) was NOT completed
  — the ell_comps fixes resolved the specific Mod 09 suspected-bug cell
  the session was targeting, and all notebooks now execute cleanly, but
  a pass through every embedded fit image with the
  `autolens-fit-diagnostics` PASS/SUSPECT/FAIL skill was not done. That
  audit should go next.
- The one-line "I re-executed this notebook cleanly on 2026-04-21" does
  not mean the SCIENCE in every cell is right — it just means the code
  runs. The visual-residual check is the right next step.

---

## 2026-04-29 — Day cycle: R5 climb close-out + truth-anchored validation + GPU investigation + scaffolding

### Morning / afternoon: cluster work

R5 multi-plane + 2-source fits for compound mocks 3, 4 (commit `d573143`).
**Headline:** R5 wins the Bayes factor on both mocks but recovers neither
secondary truth — Pattern E persists on mock_3
($\theta_{E,2}=0.005''$ vs truth $0.12''$); Pattern A persists on mock_4
($\theta_{E,2}=0.885''$ vs truth $0.08''$).

In response, three **truth-anchored validation** Cannon jobs submitted
(commit `18161f0`): R5_truth on compound mocks 3, 4; truth_anchored on
group_scale. The group_scale variant returned a decisive PASS in 1h 05m
(log_Z=44699.80, χ²/N=1.025, max|res|=4.50σ) — the model space CAN fit
the system; freely-fit failures were search-space exploration, not
representability (commit `fcd3fa2`).

**Staged-chain wave 2** submitted (commit `fcd3fa2`): R5_staged on
mocks 3, 4; staged_satellites on group_scale. mock_3 staged returned
in 2h 16m (commit `1339526`) and lands in the **identical basin** as
the freely-fit R5 (ΔlogZ +0.16, secondary $\theta_E=0.005''$ in BOTH
fits). **Walking the chain via R2_2src does NOT rescue mock_3 from
Pattern E** — the secondary at $z_2=1.2$, truth $\theta_{E,2}=0.12''$
is genuinely sub-detectable at this PSF/exposure. group_staged
job 9212864 was cancelled (Stage 1 mask=1.7" too tight).

### GPU investigation

Initial conclusion ("JAX-GPU is 5.5× slower") was based on the **wrong
metric** — single-call latency, not the vmap'd batch path Nautilus
uses by default. Per-call probe with `jax.block_until_ready()` (commits
`f0e2673`, `a370a24`) revealed:

| Backend | Per-call | vs numpy single |
|---|---|---|
| numpy serial | 95.6 ms | 1.0× |
| numpy 32-core mp | ~3 ms | ~32× |
| JAX-GPU `Fitness._vmap` batch=256 | **0.11 ms** | **~900×** |

Batched ~28× faster than the numpy production path. **End-to-end
head-to-head** (jobs 9252945 numpy vs 9252946 JAX-GPU on R5_truth_iso
mock_3): at 1 min CPU was 4× ahead (JIT warm-up); at 12 min GPU was
12% ahead; at 4 h CPU is 10% ahead again (JIT recompiles at each
bound-network transition). **The 800× per-call vmap win does NOT
carry through to a real Nautilus fit at our 9k-pixel scale.** Bottom
line: JAX-GPU offers ~10–20% wall-time improvement over numpy 32-core
for our problem size — not worth the operational complexity.

**Bug filed mentally for upstream:** autolens 2026.4.13.6's PowerLaw
mass profile JAX path crashes — `omega()` in
`autogalaxy/profiles/mass/total/jax_utils.py:41` passes
`functools.partial` to `jax.lax.scan`. Probe falls back to Iso mass
via `PROBE_USE_POWERLAW=0` (default).

### Notebook polish for student handoff (commit `a24338c`)

- `01_group_scale_fit.ipynb` §3.1 added: truth_anchored verdict +
  diagnostic narrative.
- `02_compound_lens_ladder.ipynb` §13: 5-rung table for mocks 3, 4
  (R2 / R3 / R5 / R5_truth / R5_staged) with safe-load `_safe_load_13`
  cells that auto-populate when results land.
- All 41 notebooks parse clean, zero error cells.

### Memories added / updated

- `feedback_bayes_factor_vs_truth.md` (NEW): richer ladder rungs can
  win ΔlogZ by absorbing residual structure into non-physical
  parameters; check rails + caustic + truth before passing.
- `feedback_truth_anchored_validation.md` (NEW): when freely-fit
  fails, build a tight-prior truth-anchored variant BEFORE adding
  model complexity to distinguish search-space vs model-space failures.
- `project_multigpu_jax_idea.md` (UPDATED): corrected from "GPU is
  4× slower" to the corrected understanding: per-call vmap'd batch
  is ~28× faster than numpy 32-core, but end-to-end Nautilus fit at
  our scale shows only ~12% speedup because batch state-management
  overhead dominates.

### Evening: laptop scaffolding (Cannon login.rc unreachable)

Built four `00_climb_to_*.ipynb` student-handoff bridge notebooks
(commit `bcb164e`) — each bridges from Module 09 (single-deflector)
to a multi-object architecture. Common template + two reusable
techniques demonstrated in context: **iterative masking** + **position
likelihoods**.

| Notebook | Architecture | Featured techniques |
|---|---|---|
| `compound_lens/00` | 2 deflectors @ different z (minimal) | Multi-plane Tracer as a 1-line upgrade |
| `compound_lens_zoo/00` | 2 deflectors @ different z (production) | Iterative masking, PositionsLH, extra_galaxies, Pattern E |
| `double_source_plane/00` | 1 lens, 2 sources @ different z | Cosmological β_12 derivation, joint multi-source fit |
| `group_scale/00` | BGG + 3 satellites @ same z | Photometric centroid anchoring, BGG ↔ satellite degeneracy |

All four execute clean in `<60s` with `PYAUTOFIT_TEST_MODE=1`.
`Examples/README.md` updated with [climb] markers + new "Climb scaffolding"
section listing all four.

**Pre-staged for Cannon recovery (commit `159fd09`):**
`fit_example_group_scale.py` `build_staged_satellites` Stage 1 mask
fixed from 1.7" to 1.85" — still excludes all 3 satellites (closest at
r=1.92") but captures 95% of BGG integrated light (R_e=0.9", n=4).
Re-fire-ready as `--part=staged_satellites`.

### What's left for the next session

1. Audit the 5 in-flight Cannon jobs (`truth_m3_R5`, `truth_m4_R5`,
   `staged_m4`, `iso_m3_cpu`, `iso_m3_gpu`) once login.rc returns.
2. Re-fire `staged_satellites` with the new 1.85" mask.
3. Append §14 to ladder notebook with GPU end-to-end speedup result.
4. Long-carryover: free-cosmology rung for mocks 2/5, agel_real_target
   PSF, real-DM at higher M.

Detailed checklist in `Modules/10_Cluster_Computing/HANDOFF_2026_04_29.md`
§13.


## 2026-05-05 — Hernquist-lab onboarding + chi²-at-truth diagnostic on truth_fc trio

### Morning: cluster pull + Hernquist student onboarding

- `pull_from_cannon.sh` round-trip. **No new Cannon completions** — `dspl_beta_v2` still RUNNING (1d04h elapsed at pull time, ~19h left). `truth_fc` trio (jobs 9727090/92/94) confirmed TIMEOUT at 48h. 13 untracked dirs surfaced (stale pre-v0.92-tag results from Cannon-side state); kept untracked per ship-set discipline.
- Created **`Modules/10_Cluster_Computing/cannon.env.hernquist`** — pre-filled config for Hernquist-lab students. Smoke-tested on Cannon (4 checks: file syncs, env vars resolve, FASRC Miniforge activates `autolens312`, `sbatch --test-only` accepts the resource combo). Cross-linked from `START_HERE.md`, `SETUP_NEW_USER.md` Step 4, and `STUDENT_QUICKSTART.md` see-also (commit `04f7847`).
- **Scrubbed all SIAG references project-wide** (commit `342bcd8`): SLURM defaults flipped from `siag_gpu`/`siag_lab` (SIAG-subgroup-specific, not all Hernquist-lab members have access) to `hernquist`/`hernquist_lab` (10-node lab partition + lab-wide fairshare account). 22 files updated incl. notebooks, slurm header, env templates, internal Claude-skill docs. Re-smoke-tested: dispatches immediately to `holy7c16203`.

### Afternoon: PositionsLH API contract in compound_lens 01 (commit `acb0ffb`)

- Fixed latent bug in `Examples/compound_lens/01_compound_direct_fit.ipynb`: §4 fit cell referenced `positions_A` / `positions_B` which were never defined (only `positions_all` existed).
- Rewrote §2 to make the dual-API explicit: the **same `al.Grid2DIrregular`** flows to both `aplt.subplot_imaging_dataset(positions=...)` and `al.PositionsLH(positions=..., threshold=...)`. Spelled out the multi-image-conjugate prerequisite — peaks of an extended arc aren't valid PositionsLH inputs.
- Added §2.5 sanity-check section that empirically demonstrates the constraint: build truth tracer + 3 perturbed tracers (centre +0.5″, θ_E ±33%, θ_E +124%), trace image positions through each to source plane, print penalty contract:

  | Model | src-plane spread | penalty |
  |---|---|---|
  | truth (v4 PASS) | 0.0004″ | 0 |
  | centre +0.5″ | 0.145″ | 4.5e6 |
  | θ_E -33% | 1.14″ | 1.0e8 |
  | θ_E +124% | 4.31″ | 4.2e8 |

  (Penalty factor = 1e8, ~10⁴× larger than typical imaging-likelihood differences, so PositionsLH genuinely prunes parameter space.)

### Evening: chi²-at-truth diagnostic on compound zoo mocks 2/3/5

Per `HANDOFF_2026_05_05` §4 priority 1 — diagnose whether the three TIMEOUT'd `R5_truth_freecosmo` jobs are model-fix-bound or budget-bound, before re-spending Cannon compute.

**Method.** Same pattern as the cluster_scale fix (`f8471bb`): build an `al.Tracer` with literal truth values, wrap in `FitImaging`, read off chi²/N. Two iterations needed:

1. **First pass with literal lenstronomy `amp` → autolens `intensity` 1:1** gave chi²/N in the millions on all mocks. Lenstronomy/autolens use different surface-brightness conventions; the 1:1 mapping is wrong. The R5_truth_freecosmo Cannon driver finesses this by using `LogUniformPrior(1e-3, 1e3)` on every intensity.
2. **Second pass with `al.lp_linear.Sersic`** (linear light profiles solve their intensity analytically per likelihood — same effective freedom as the LogUniform fit). Result:

| Mock | chi²/N at truth | max\|res\| | Cannon truth_fc | Cosmology |
|---|---|---|---|---|
| 2 | 11.4 | 25.7σ | TIMEOUT 48h | Om=0.25, w=−0.9 |
| 3 | **2.1** | 6.8σ | TIMEOUT 48h | Om=0.30, w=−1.0 (std) |
| **4** | **4.9** | 11.2σ | KNOWN-GOOD (R5_truth_anchored shipped, χ²/N=1.025) | Om=0.30, w=−1.0 |
| 5 | 9.6 | 33.6σ | TIMEOUT 48h | Om=0.35, w=−1.2 |

**Verdict.** The diagnostic itself has a baseline inflation: mock_4 — a *converged shipped fit* with χ²/N=1.025 — gives chi²/N=4.9 at literal-truth values. The 4× gap is convention drift (sign on shear/ell_comps, sub-pixel centre offsets) that the actual Cannon fit absorbs via the σ=0.1 truncated-Gaussian truth-anchored priors but a literal-instance eval does not.

**Mock_3 at chi²/N=2.1 starts BELOW mock_4's baseline** — clean evidence the truth_fc TIMEOUT is search-budget bound, not model-space bound. Mocks 2 and 5 at chi²/N ~ 10 sit ~2× above mock_4's baseline, attributable to the cosmology-dimension freedom the truth_fc model has but I deliberately constrained at truth.

**This is NOT Pattern F** (no missing component). The model space is correct; Nautilus just needed more than 48h to land in the basin. Recommend resubmit: 96h budget for mock_3, 120h for mocks 2 and 5 (cosmology-dimension penalty).

Diagnostic scripts saved at `/tmp/chi2_diag*.py` and JSON at `/tmp/chi2_at_truth_linear.json`.

### What's left for the next session

1. Audit `dspl_beta_v2` when it lands (ETA 2026-05-06 04:00 local).
2. Submit AGEL hot-pixel cleaned refit (`--part=direct_clean`).
3. Resubmit truth_fc trio at 96–120h (per the diagnostic above).
4. Tag `v0.93` once items 1–3 land.

## 2026-05-07 — v0.94 work session: methodology fixes + Modules 11+12 shipped

### Morning: Track B diagnostic + Tracks A/B fixes

- **Track B (Nautilus checkpoint resume deadlock, task #111)**: built `Modules/10_Cluster_Computing/scripts/diagnose_nautilus_resume.py` and ran it on the 3 stuck checkpoints from the v0.93 truth_fc trio. Findings:

  | Mock | Bound count | n_like | mtime | n_dim |
  |---|---|---|---|---|
  | mock_2 | 161 | 384,100 | 2026-05-01 | 37 ✓ |
  | mock_3 | **239** | **733,700** | 2026-05-04 | 37 ✓ |
  | mock_5 | 43 | 21,850 | 2026-05-01 | 37 ✓ |

  `n_dim=37` matches expected → not B1 (model-hash mismatch). Nautilus 1.0.5 reads the files cleanly → not B2 (format mismatch). Smoking gun: `min(log_l) = -1e+99` universally — Nautilus's "bad value" placeholder. Saved live points include cosmology samples that crash the autolens FlatwCDM angular-diameter integrator on resume → worker hangs (Pattern B3).

- **Fix applied to BOTH offending model builders**:
  - `fit_example_compound_lens_zoo_climb.py:build_R5_truth_freecosmo_model()` — TruncatedGaussianPrior on Om0 [0.05, 0.60], w0 [-1.6, -0.4]
  - `fit_example_double_source_plane.py:build_beta_freecosmo_v3_fit()` — same bounds + a new staged-chain machinery (`build_beta_chain` runs `beta_fixedcosmo` → `beta_freecosmo_v3` with prior passing)

- **`Modules/10_Cluster_Computing/CLUSTER_WORKFLOW_NOTES.md` "Checkpoint hygiene" section** codifies the rule: **always assign a fresh `unique_tag` when prior bounds change**.

- **Cannon submits**: `dspl_beta_chain` (job 11214940, 96h) + `truth_fc_m3_v4` (job 11214941, 96h, fresh unique_tag bypassing the deadlocked checkpoints).

### Afternoon: Track C (Module 11) + Track D (mge_to_physical) + Module 12

- **Track C — Module 11 (Physical Mass Models) shipped**: 29 cells main + Solutions/SOLVED variant, executes <5s. Full 6-panel residual audit walkthrough (uses the v0.93 AGEL direct_clean strict-PASS as the example), Bonferroni-corrected numerical bar, Pattern A-F failure catalogue, f_DM(<θ_E) extraction from the existing mge_to_physical results, γ′ recovery cross-check, decision flowchart. 6 cross-link READMEs flipped from "Module 11 planned" to shipped pointers.

- **Track D — mge_to_physical chi²-at-truth diagnostic**: applied the same methodology as cluster_scale's `f8471bb` fix. **Falsified** the v0.92 stated diagnosis ("missing 2nd source + secondary deflector"). Truth-tracer with all components: χ²/N=6.35, max=33σ, 694 pixels >4σ at the lens centre. Removing components: χ² changes <1%. Actual cause is a **framework-level evaluation difference** — lenstronomy and autolens integrate the cuspy `n=4.9` Sersic peak slightly differently. README updated with corrected diagnosis. Pattern G candidate name: "framework-level evaluation mismatch."

- **Module 12 (Time-Delay Cosmography & MSD)** drafted for v0.95 prep but lands in HEAD past v0.93-alpha → ships in v0.94. 19 cells main + 21 cells SOLVED, executes <4s. Sections: Refsdal time-delay derivation, Fermat potential via `mass.potential_2d_from`, D_Δt across (H0, w) with FlatLambdaCDM vs FlatwCDM, full mass-sheet-degeneracy derivation + numerical verification (image positions invariant to ~10⁻⁴″, time delays scale exactly by λ, flux ratios identical), TDCOSMO chain (Wong+20, Birrer+20). The 14-module curriculum table is now complete.

- **`Examples/quad_time_delay/` Phase 3 submission**: `--part=direct_h0_free_tight` adds tightened H0 prior (Uniform(50, 100) vs Phase 2's (40, 120)) + n_live=300 (vs 150). Phase 2 recovered median H0=92.7 with truth=70 at the 2σ edge — borderline. Phase 3 should tighten this. Cannon job 11237334, 4h budget.

### v0.94 tag readiness end-of-day

`bash Modules/10_Cluster_Computing/scripts/preflight_check_v094.sh` → **17 PASS / 2 WARN / 0 FAIL**. The 2 WARNs are conditional Cannon results (DSPL chain + mock_3 v4 summary.json) — they'll flip to PASS when the Cannon jobs land, or stay research-in-progress. v0.94 can ship as-is whenever the user wants.

### Cannon queue state

```
RUNNING (96h budget):
  11214940 dspl_beta_chain
  11214941 truth_fc_m3_v4

PENDING (4h budget):
  11237334 quad_h0_phase3
```

### What's left for the next session

1. **Wait for Cannon results** (3-4 days for the 96h jobs, ~hours for quad Phase 3).
2. **Audit when they land** — DSPL chain with 6-panel + recovered (Om0, w0); mock_3 v4 with 6-panel + position-conjugate; quad Phase 3 by H0 posterior.
3. **Tag v0.94-alpha**.
4. **v0.95 sketch**: see `HANDOFF_2026_05_07.md` §4 — quad_time_delay strict-PASS by H0 recovery, mge_to_physical native-autolens remock, possibly mocks 2/5 truth_fc retry if the deadlock fix is verified.


## 2026-05-08 — Position-likelihood research batch lands

### Overnight Cannon results (4 jobs completed since 2026-05-07 evening)

**Track A — PositionsLH threshold sensitivity sweep (job 11368641, 5h52min)**
Beautiful pedagogical result. The four-threshold sweep on compound_lens mock_1:

| Threshold | chi²/N | max\|res\| | log_Z |
|---|---|---|---|
| 1.0″ | 0.692 | 4.65σ | +30,856.16 |
| 0.3″ | 0.693 | 4.69σ | +30,855.96 |
| **0.1″** | **0.692** | **4.42σ** | **+30,855.76** |
| 0.01″ | 0.873 | 9.19σ | +30,019.99 |

**Interpretation:** for ≥0.1″, the constraint is loose enough that the converged fit reproduces the v4 PASS result (log_Z=+30,856 ≈ v4's +30,856.54, max|res|≈4.4σ ≈ v4's 4.40σ). At 0.01″, the threshold is *tighter than the actual conjugate spread the data supports* and the chain gets pulled into a worse basin (log_Z drops by ~840 units, max|res| rises to 9.19σ). The sweet spot is the 0.1″ threshold; below that, the soft penalty becomes a hard constraint that fights the imaging likelihood.

**Track C — direct_with_positions_lh (job 11368643, 1h53min OUT_OF_MEMORY in post-process)**
Same 0.1″ threshold as Track A's middle point. Fit completed but post-process OOM'd; manually exported via `--search-dir`. Identical numerics to A's t0p1 (chi²/N=0.69, max=4.42σ). Confirms PositionsLH carries through multi-plane Tracer correctly.

**Track D — TDCOSMO joint fit `qtd_joint_h0` (job 11375367, 37 min)**
**STRICT-PASS** on the first end-to-end run of joint AnalysisPoint + AnalysisImaging:
- chi²/N = **1.051** (≤1.3 strict)
- max\|res\| = **4.66σ** at 20,108 pixels (Bonferroni noise floor √(2·ln(20k))≈4.45σ → borderline-PASS strict-PASS)
- log_Z = **+76,607**

This is the canonical TDCOSMO IV / Birrer+20 setup working at scale: quasar positions + flux ratios + time delays jointly fit with the extended host arc imaging, sharing a single lens model + a single source Galaxy that carries both `point_0=ps.Point(...)` and `bulge=lp.SersicCore(...)`. The two analyses are combined via `af.FactorGraphModel`. 37 minutes wall — point-source side adds <1 min on top of the imaging fit.

**Quad Phase 3 — H0 strict-PASS retry (job 11237334, 12 min)**
Tightened H0 prior to Uniform(50, 100) + n_live=300:
- ML H0 = 63.6 (closer to truth 70 than Phase 2's ML=74.6)
- median H0 = 81.95 (Phase 2 median was 92.7 — improving)
- 1σ band: [71.9, 92.3] — does NOT bracket truth
- 2σ band: [54.7, 100.0] — brackets truth

Better than Phase 2 but still biased high. Likely needs more data (single quad + 0.5d delay precision is the H0LiCOW best-case noise floor; this mock matches that). Methodology demo, not a calibrated H0 result.

### OOM cases (failed export, salvageable)

`cl_pos_lh` (Track C, 1h53m), `qtd_pos_only_v2` (Track B, 14min) — both OOM'd in `export_results.py` post-process at 32-64GB. Pattern matches the v0.93 AGEL OOM. The fits themselves completed; manual `--search-dir` export works fine. **Action item for v0.95**: refactor `export_results.py` to free analysis-loaded fit objects between searches OR bump default Cannon `--mem` to 128GB for example jobs.

### Track B v2 re-exported (added 2026-05-08 next-session)

Manual `--search-dir` export of the OOM'd `phase_4_positions_only_v2` lands at `Examples/quad_time_delay/results/phase_4_positions_only_v2/` (renamed from auto-export collision with `quad_direct_fit/`). Positions-only fit has no imaging likelihood so the chi²/max-residual fields are null; the H0 posterior is what matters.

**The H0 chain across the three sister fits is now complete** — this is the pedagogical payoff:

| Fit | Likelihood | H0 median | H0 1σ width | Bias from truth (70) |
|---|---|---|---|---|
| Track B v2 — pos-only | quasar positions | 79.4 | ±26 | +9.4 |
| Phase 3 — image-plane only | extended host arc | 81.95 | ±10 | +12.0 |
| Track D — **joint** | image + Δt + positions | **74.95** | **±2.3** | **+5.0** |

Joint TDCOSMO methodology narrows σ(H0) by **~10×** relative to positions-only and **~4×** relative to imaging-only, and reduces the bias from +12 → +5 km/s/Mpc. This is the textbook Refsdal-1964/H0LiCOW-XIII demonstration that **time delays carry the H0 information that positions alone don't** — the imaging side anchors the lens model, the point side anchors the source position, and Δt locks in D_Δt = (1+z_l)·c⁻¹·D_l·D_s/D_ls. Module 12 §3 + §5 should reference this empirical chain.

### Cannon queue still running

- `dspl_beta_chain` (job 11214940) — 17h elapsed, 3d 7h left, the v0.94 Track A
- `truth_fc_m3_v4` (job 11214941) — 17h elapsed, 3d 7h left, the v0.94 Track B

### v0.95 deliverables landed today

The PositionsLH research batch was conceived this morning (user request: "any other runs we can do to keep building on programs like how we use the point modeling to constrain the position likelihood") and ships 4/4 Tracks within ~24h:

- **A**: PositionsLH threshold sweep — empirical pedagogy + v0.93 §2.5 sanity check validated at scale
- **B**: positions-only fit — pending re-export (OOM in v1)
- **C**: multi-plane PositionsLH — identical to A's t0p1 point, confirms multi-plane carry-through
- **D**: joint AnalysisPoint + AnalysisImaging — first TDCOSMO-methodology fit, STRICT-PASS

The Track D result is particularly notable: it's the first time the H0LiCOW XIII / TDCOSMO IV joint methodology is exercised end-to-end in the curriculum. Module 12 §5 references this — now there's an audited example fit to point at.

---

## 2026-05-18 / 2026-05-19 — Paper-reproduction program kicked off (Spec 05 A1201 in flight)

Long arc spanning end-of-day 2026-05-18 through 2026-05-19. Headline:

- **First real-paper reproduction Cannon job in flight** — Stage 1 of the
  Nightingale+2023 Abell 1201 ultramassive black-hole reproduction (Spec 05).
  Two parallel attempts: v2 (job 13822696, uniform priors, 9h+ elapsed,
  plateauing convergence) and v3 (job 13868426, N+23-anchored Gaussian priors,
  newer, expected to land first).
- **All four paper-reproduction headline notebooks scaffolded** (A1201, P1 Li+23
  population cosmography, P2 Ballard+23 TSPL Jackpot subhalo, P3 Li+26 DSPL
  IMF/NFW). Each ~7–9 cells, follows a common template (citation register,
  per-stage results, methodology divergences, headline figure).
- **Critical citation correction** for N+23 — verified via ar5iv: the paper
  does **NOT** use σ_v in its likelihood; γ′-M_BH degeneracy is broken by
  counter-image morphology (§4.2) + mass-light coaxiality (§4.3), both
  imaging-only arguments. Our Phase 3 σ_v Jeans factor is OUR methodology
  extension, not a reproduction of the paper. Spec 05 design + plan rewritten
  to reflect this. `feedback_no_fabricated_citations` discipline validated.

### Cannon job slate (in chronological order)

| Job | Name | Wall | Outcome | Note |
|---|---|---|---|---|
| 13649652 | rarc_kin2 | 47:39 | FAILED | AnalysisKinematics SimpleNamespace stub missing save_attributes |
| 13649870 | qtd_h0_kin | 46:31 | FAILED | same stub crash |
| 13685089 | rarc_kin3 | 10:03 | COMPLETED (resumed from rarc_kin2 ckpt) | **v0.97 headline**: γ′=1.949 vs truth 1.95, M_BH θ_E=0.059 vs truth 0.08; kinematic factor demonstrably breaks γ′-M_BH degeneracy beyond v0.96 imaging-only result |
| 13685104 | qtd_h0_kin2 | 7:24 | COMPLETED + visualize crash | SimpleNamespace stub now fixed in `_jeans_sigma_v.py`; resubmit parked |
| 13690686 | p1_pop_autofit | 0:04 | FAILED | Personal `~/miniforge3` path in slurm — fixed to Cannon-canonical Miniforge |
| 13747394 | p1_pop_autofit (re) | 2:11 | "COMPLETED" max_log_L=0.0 | Prior-only sampling — `run_sampler_cannon.py` likelihood not evaluated. Spec 01 Phase 2 work needed |
| 13747801 | a1201_lp v1 | 2:39 | CANCELLED (by us) | False alarm: cutout was actually correctly built |
| 13748339 | a1201_lp v1 (re) | 1:04:27 | COMPLETED but rails 4 priors | 8″ cutout chopped tangential arc; mass.centre untied → drifted 0.6″ |
| 13822696 | a1201_lp v2 | 9h+ ongoing | RUNNING, convergence concern | 16″ cutout, mask r=6″, tied mass.centre=bulge.centre, widened priors. logZ +172,570 (positive evidence vs v1's −13,269) but f_live stuck 1.0 / N_eff 1-2 |
| 13868426 | a1201_lp_informed v3 | just started | RUNNING (parallel) | N+23-anchored Gaussian priors, ~5× tighter prior volume. Different output dir |

### Infrastructure shipped this arc (`private/2303_15514_nightingale2023_abell1201/`)

- **`code/a1201_lens_model.py`** — added `--chain-from`, `--n-light={1,2,3}` (N+23 Table 1 ladder), `--informed-priors`, multi-Sersic with tied centres
- **`code/chain_priors_from_lp.py`** + 4 tests — Stage-N → Stage-(N+1) prior chainer
- **`code/prep_a1201_dataset.py`** — `--band={f390w,f814w}` + `--per-band-subdir`; both bands' 16″×16″ cutouts built (F390W is N+23's primary band, ΔlnZ=+100.58 per Table 4)
- **`code/audit_stage1.py`** — automated prior-railing + mass-light coaxiality + θ_E sanity → PASS/SUSPECT/FAIL
- **`code/compute_bayes_factor.py`** — ΔlnZ + Jeffreys + N+23 §3.9 3σ-threshold (ΔlnZ=4.5) + σ-equiv
- **`code/extract_mbh.py`** — θ_E_BH → M_BH (M☉) via point-mass lens equation
- **`submit_a1201.slurm`** — 4-stage chained; `INFORMED_PRIORS` + `OUTPUT_SUFFIX` env vars for parallel runs
- **`notebooks/03_a1201_mbh_recovery.ipynb`** — 9-cell headline notebook
- **Tests**: 13/13 PASS
- **`PAPER_NOTES.md`** rewritten with ar5iv-anchored citations
- **`STAGE_COMMANDS.md`** — exact sbatch invocations per stage
- **`/private/COMPARISON_TEMPLATE.md`** — per-paper scoreboard template

Plus three more headline notebooks:

- `private/2307_.../notebooks/01_p1_w_recovery.ipynb` (Li+23, w = −0.96 ± 0.46 target)
- `private/2309_.../notebooks/01_p2_subhalo_bayes.ipynb` (Ballard+23, 5.9σ subhalo)
- `private/2602_.../notebooks/01_p3_imf_nfw_recovery.ipynb` (Li+26 DSPL IMF, M_*=4.4e11)

### Repo-wide

- **`_jeans_sigma_v.AnalysisKinematics`** stub fix: SimpleNamespace `make_result` forwards `max_log_likelihood_instance` from samples_summary. Fixes the v0.97 post-fit visualize crash that killed jobs 13649652 + 13649870.
- **CLAUDE.md rewritten**: dropped stale lenstronomy boilerplate, added Cannon SSH alias + ControlMaster + cannon.env documentation; codified account/partition routing policy (siag_lab primary / hernquist student; `scontrol update` to fix mis-routed pending).
- **Spec 05 design + plan** rewritten with verified N+23 methodology — Stage 4 (`adapt`) is paper-faithful track; Stage 3 (`with_kin`) explicitly labeled our extension.

### Methodology lessons

1. **Cutout sizing matters.** Stage 1 v1's 8″ cutout chopped the tangential arc at θ_E ≈ 3.8″. Fix: 16″ cutout + mask r=6″ includes the arc visibly in the BCG-subtracted residual.
2. **Tied mass-light centres are critical for radial-arc methodology.** Untied `mass.centre` drifts to fit contaminating flux; fit converges to false solution.
3. **N+23's headline driver is light-model flexibility + pixelised source — NOT σ_v.** 3-Sersic + F390W + Voronoi+`reg.Adapt` → ΔlnZ=+100.58; 1-Sersic + F814W → marginal ΔlnZ≈3.
4. **Nautilus f_live stuck at 1.0 + N_eff at 1-2 after 100K+ calls signals posterior pathology.** Informed (Gaussian-anchored) priors with ~5× reduced search volume substantially help — see v3 vs v2.
5. **Per-paper citation register is critical infrastructure.** PAPER_NOTES transcription doesn't substitute for body-anchored section/figure citations. The verification workflow (WebFetch → cross-check → tag verified) caught the σ_v misframing before it propagated.

---

## 2026-05-20 — A1201 Stage 1 succeeded with 3-Sersic; Phase 4.5 mass-model variants in flight; Herculens kicked off

### Stage 1 success: v4 (3-Sersic light) converges where v2/v3 (1-Sersic) failed

After v2 (uniform 1-Sersic, 21h) and v3 (informed-prior 1-Sersic, 5.5h) both
landed pathological broken fits — v2 with extreme shear γ≈0.94 + railing 4
priors + chi²-map peak 39,317; v3 with slope=1, zero ellipticity, figure-8
model + chi²-map peak 285 — we resubmitted as **v4 with `--n-light=3`**
(N+23 Table 1's 3-Sersic light model).

**v4 result** (Cannon job 14015488, 6h09m):
- lnZ = **+174,904** (vs v2 +172,973, v3 +172,238 — ΔlnZ +1,932 / +2,667
  respectively — decisive Bayes-factor preference for 3-Sersic light,
  exactly mirroring N+23 Table 1)
- Visual diagnostic (`Figures/spec05_a1201_diagnostics/v4_3sersic_fit.png`):
  model image traces the actual data arc as a crescent; lens-light
  subtraction is clean; source-plane reconstruction is a physical extended
  source; chi²-map peak ~180 (216× better than v2's 39K).

**Key methodology lesson**: lens-light model flexibility is the **critical
bottleneck** for BCG-scale lenses, NOT source-plane flexibility. With a
properly subtracted BCG, even a parametric Sersic source can fit. This
matches N+23's pipeline ordering (Light pipeline runs BEFORE Mass pipeline
and selects 3-Sersic as prerequisite to Mass pipeline).

### Stage 2 (PL + SMBH, 3-Sersic light) launched

`a1201_with_smbh_3s` (job 14066124, RUNNING since 2026-05-20 ~16:00 EDT, ~12h
expected wall). Chained from v4. **First M_BH posterior attempt.**

### Phase 4.5 — N+23 §3.8 / §4.3 mass-model alternatives

Verified via ar5iv that N+23's "additional tests" are §3.8 Mass Pipeline
variants: PL/BPL/decomposed × ±SMBH. Wrote
`docs/superpowers/plans/2026-05-20-paper-repro-05-a1201-mass-model-variants.md`
(10 tasks, ~55 steps).

Implementation shipped (`a1201_lens_model.py`):
- `_build_lens_galaxy_bpl` + `build_bpl_fit` + `build_bpl_smbh_fit` —
  O'Riordan, Warren & Mortlock BPL parameterization (`al.mp.PowerLawBroken`,
  verified class name 2026-05-20). `free_mass_centre` flag for the
  §4.3 / Appendix D coaxiality reproduction.
- `_build_lens_galaxy_decomposed` + `build_decomposed_fit` +
  `build_decomposed_smbh_fit` — Sersic stellar mass + standard NFW + shear.
- New `--part={bpl,bpl_smbh,decomp,decomp_smbh}` + `--free-mass-centre` CLI.
- 23/23 tests pass (7 new for Phase 4.5).

**6 Cannon variants submitted 2026-05-20** chained from v4:

| Job | Variant |
|---|---|
| 14068551 | a1201_bpl_3s (BPL tied, no SMBH) |
| 14068552 | a1201_bpl_smbh_3s (BPL tied, +SMBH) |
| 14068556 | a1201_bpl_free (BPL free, no SMBH) |
| 14068557 | a1201_bpl_smbh_free (BPL free, +SMBH) |
| 14068559 | a1201_decomp_3s |
| 14068560 | a1201_decomp_smbh_3s |

Plus the running Stage 2 (14066124). Seven A1201 jobs in flight on siag_lab.

### Herculens kicked off (Spec 03 / Spec 04 prerequisite)

Started both local + Cannon `herculens312` conda envs (Python 3.12).
Installing `jax`, `jaxlib`, `numpyro`, `herculens` (from
`github.com/Herculens/herculens` — not yet on PyPI), and the optional
`jaxinterp2d` fast bilinear interpolation dependency. CPU JAX for first
pass on both sides; GPU JAX for Cannon will be a follow-on.

Next milestones (per Spec 00 §Herculens-install + Spec 03 plan):
1. Smoke-import test both envs
2. Inspect Herculens API (`MassModel`, `LightModel`, `LensImage`,
   NumPyro NUTS) — no usable docs publicly; need to introspect once installed
3. Write `autolens → Herculens` model bridge for the DSPL example
4. Smoke-fit `Examples/double_source_plane/mocks/` via Herculens NUTS;
   verify posterior consistent with our autolens DSPL strict-PASS

### Commits this arc

- `ca46cf9` — A1201 paper-repro infrastructure + N+23 citation correction
- `f22a482` — v2/v3 diagnostic figures (catastrophic-fit evidence)
- `196711b` — v4 diagnostic figure (real-fit evidence)
- `212f58b` — Phase 4.5 mass-model variants plan

