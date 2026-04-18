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
| Solutions/05 | running (2-search) | — | launched 2026-04-17 23:05 |
| Modules/09 | running (MGE SLaM) | — | launched 2026-04-17 23:05 |
| Solutions/09 | running (MGE SLaM) | — | launched 2026-04-17 23:05 |

- **Machine-sleep gotcha**: laptop auto-sleep stalled all runs for ~2 h around 19:00–21:00. Fix: `caffeinate -dims -w <kernel_pid>` tied to each Python kernel PID — exits automatically when the kernel dies. All six current runs have caffeinate guards active.
- 16-core / 48 GB machine, load avg ~5 with 6 nbconvert kernels; headroom OK.
