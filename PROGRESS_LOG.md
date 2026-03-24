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

### Remaining modules (TODO):
- Module 06: Multi-Component Mass Models
- Module 07: Real Data: FITS to Model
- Module 08: Results, Diagnostics & Figures
