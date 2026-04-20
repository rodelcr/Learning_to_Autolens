# Learning to Autolens

**Strong Gravitational Lens Modeling with PyAutoLens — A Step-by-Step Tutorial Suite**

A 10-module tutorial series teaching computational strong lens modeling from first principles through publication-ready results, with a dedicated cluster-computing module (Mod 10) showing how to run the heavy searches on Harvard FASRC Cannon. Each module pairs detailed **Jupyter notebooks** (with thorough physics commentary) with **LaTeX theory notes** and optional **Wolfram Mathematica** scripts for symbolic verification.

This project is a companion to [Learning to Lens](https://github.com/rodelcr/Learning_to_Lens) (GR & lensing theory in Mathematica), but is fully self-contained — all necessary theory is developed inline with references to the primary literature.

> **Alpha build (v0.92)** — April 2026. Modules 01–10 complete; cluster runs for Mods 04/05/09 pending. Run `python check_install.py` before starting. Feedback welcome at rodrigo.cordova_rosado@cfa.harvard.edu.

---

## Quick Start

### 1. Create a conda environment

```bash
conda create -n autolens python=3.12 -y
conda activate autolens
```

Python **3.12 is required** — autolens 2026.4+ (which Modules 04/05/09 depend on for `RectangularAdaptDensity`, `RectangularAdaptImage`, `reg.Adapt`, and `AdaptImageMaker`) is not released for Python 3.11 or earlier.

### 2. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` pins the top-level packages (autolens ≥ 2026.4.13, jupyterlab, matplotlib < 3.9, corner, astropy, numba). We cap `matplotlib < 3.9` because PyAutoLens has a known plotting incompatibility with 3.9+.

> **Use `python -m pip`, not bare `pip`.** A stray user-level `pip` on `$PATH` can shadow the env's pip and silently install into the wrong interpreter — this happens on HPC accounts and occasionally on laptops with multiple Python installs. The symptom is "`pip install` succeeds, `import autolens` fails." See `PROGRESS_LOG.md` for the Cannon-specific runbook if installing on Harvard FASRC.

### 3. Verify the install

```bash
cd Learning_to_Autolens_alpha_v0.91
python check_install.py
```

This checks all package versions, runs API smoke tests, and confirms the datasets are in place. **Fix any FAIL items before proceeding.**

### 4. Launch Jupyter

```bash
jupyter lab
```

Then open any notebook from `Modules/` — start with **Module 01** if you're new, or **Module 09** if you want the production MGE/SLaM workflow.

### 4. (Optional) Wolfram Mathematica

Symbolic derivation scripts in `Mathematica/` require Mathematica 13.0+ or Wolfram Engine. These are supplementary — all key results are derived in the notebooks and LaTeX notes.

---

## Prerequisites

### Knowledge
- Gravitational lensing fundamentals: lens equation, Einstein radius, convergence, shear, magnification
- Basic Python (NumPy, Matplotlib)
- Familiarity with Bayesian inference is helpful but covered in Module 03

### Software

| Software | Version | Purpose | Required? |
|----------|---------|---------|-----------|
| Python | 3.12 | Runtime (autolens 2026.4+ requires ≥ 3.12) | **Yes** |
| PyAutoLens | >= 2026.4.13 | Lens modeling | **Yes** |
| Jupyter Lab | any | Notebook environment | **Yes** |
| Wolfram Mathematica | 13.0+ | Symbolic derivations | Optional |

---

## Module Curriculum

### Part I: Foundations

| # | Module | What You'll Learn | Runtime |
|---|--------|-------------------|---------|
| 01 | [Basics: Grids, Galaxies, Ray-Tracing](Modules/01_Basics_Grids_Galaxies_RayTracing/) | Grid2D, light/mass profiles, Galaxy, Tracer, critical curves & caustics | < 1 min |
| 02 | [Simulating Lens Data](Modules/02_Simulating_Lens_Data/) | PSFs, noise, instrument models (HST, Euclid, Keck AO), FITS I/O | < 1 min |
| 03 | [Your First Lens Model](Modules/03_First_Lens_Model/) | Priors, Nautilus nested sampling, posteriors, corner plots, degeneracies | ~15 min |

### Part II: Advanced Modeling

| # | Module | What You'll Learn | Runtime |
|---|--------|-------------------|---------|
| 04 | [Search Chaining & SLaM](Modules/04_Search_Chaining_SLaM/) | Prior passing, the 4-stage SLaM pipeline, search chaining | ~30 min |
| 05 | [Pixelized Source Reconstructions](Modules/05_Pixelized_Source_Reconstructions/) | Delaunay meshes, regularization, Bayesian evidence, adaptive pixelization | ~45 min |
| 06 | [Multi-Component Mass Models](Modules/06_Multi_Component_Mass_Models/) | Stellar (Sersic) + dark matter (NFW), mass-to-light ratio, bulge-halo conspiracy | ~15 min |

### Part III: Production Techniques

| # | Module | What You'll Learn | Runtime |
|---|--------|-------------------|---------|
| 09 | [MGE & Linear Light Profiles](Modules/09_MGE_Linear_Light_Profiles/) | Multi-Gaussian Expansion, linear inversions, `mge_model_from`, full 5-stage SLaM pipeline — the current production standard | ~90 min |

### Part IV: Real Science

| # | Module | What You'll Learn | Runtime |
|---|--------|-------------------|---------|
| 07 | [Real Data: FITS to Model](Modules/07_Real_Data_FITS_to_Model/) | Data preparation, masking, PSF handling, noise maps, AGEL survey examples | < 5 min |
| 08 | [Results, Diagnostics & Figures](Modules/08_Results_Diagnostics_Figures/) | Corner plots, residual analysis, Einstein mass, publication figures, Bayes factors | < 5 min |

### Part V: Production at Scale

| # | Module | What You'll Learn | Runtime |
|---|--------|-------------------|---------|
| 10 | [Cluster Computing on Cannon](Modules/10_Cluster_Computing/) | Converting notebooks to Slurm jobs, Nautilus checkpoint resume, rsync patterns, `export_results.py` for git-trackable artifacts, FASRC-specific filesystem and partition guidance | reading ~30 min; end-to-end cluster run ~5–12 h |

> **Runtimes** are approximate for a laptop with 8+ cores. Modules with non-linear searches (03–06, 09) take longer on first run; results are cached for subsequent runs. **Modules 04, 05, 09** have ready-made cluster drop-ins (`Modules/10_Cluster_Computing/scripts/fit_module{04,05,09}.py`) if your laptop can't keep up.

### Viewing pre-computed results without running anything

Every module that requires a non-trivial fit publishes its finished results under `Modules/XX_*/results/` as small PDFs and JSON files (fit subplot, corner plot, `info.txt`, `summary.json`). These are committed to the repo and can be viewed directly — no cluster account, no multi-hour wait. See Module 10 §8 (Results Viewer) for the pattern and loader snippet.

---

## Repository Structure

```
Learning_to_Autolens/
├── Modules/                              # Tutorial notebooks (start here!)
│   ├── 01_Basics_Grids_Galaxies_RayTracing/
│   ├── 02_Simulating_Lens_Data/
│   ├── 03_First_Lens_Model/
│   ├── 04_Search_Chaining_SLaM/
│   ├── 05_Pixelized_Source_Reconstructions/
│   ├── 06_Multi_Component_Mass_Models/
│   ├── 07_Real_Data_FITS_to_Model/
│   ├── 08_Results_Diagnostics_Figures/
│   ├── 09_MGE_Linear_Light_Profiles/
│   └── 10_Cluster_Computing/             # Slurm jobs + rsync + export_results
│       └── scripts/                      # fit_module{04,05,09}.py, submit_cannon.slurm, ...
├── Solutions/                            # Solved exercise notebooks
├── Notes/                                # LaTeX theory companions (one per module)
│   ├── preamble.tex                      # Shared macros & environments
│   └── XX_Topic/XX_topic_theory.tex
├── Mathematica/                          # Symbolic verification scripts (.wl)
├── Figures/                              # Exported figures
├── Output/                               # Compiled PDFs (git-ignored)
├── autolens_workspace_latest/            # PyAutoLens workspace (v2026.2, datasets)
├── autolens_workspace_original/          # PyAutoLens workspace (v2025.11, reference)
├── CLAUDE.md                             # Project context for Claude Code
├── PROGRESS_LOG.md                       # Timestamped work log
└── README.md                             # This file
```

---

## Important Notes

### JAX compatibility
Some modules (05, 09) set `use_jax=False` on `AnalysisImaging` because pixelized source inversions are incompatible with JAX tracing. This is handled automatically in the notebooks — no user action needed.

### Cached results
Non-linear search results are saved to `output/` inside each module directory. If you re-run a notebook, completed searches will load from cache instantly. To force a fresh run, delete the corresponding `output/` directory.

### Exercises
Each module ends with 4 exercises. Solutions are in `Solutions/XX_module_name_SOLVED.ipynb`.

### LaTeX theory notes
Compile individual notes with:
```bash
cd Notes/XX_Topic && pdflatex XX_topic_theory.tex
```
Or compile all:
```bash
cd Notes && bash build.sh
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: No module named 'autolens'` | Activate the conda environment: `conda activate autolens` |
| `ModuleNotFoundError: No module named 'autoarray.inversion.mesh'` | Version mismatch — run `pip install --upgrade autolens autoarray autofit autogalaxy autoconf` |
| `TracerArrayConversionError` (JAX error) | Ensure the notebook uses `use_jax=False` on `AnalysisImaging` |
| `AnalysisException: positions_likelihood_list` | Pixelized source fits require positions — see Module 05, Section 7 |
| `LinAlgError: Singular matrix` during search | Normal for some parameter samples — the `SafeAnalysisImaging` wrapper handles this |
| Search takes very long | Reduce `n_live` (e.g., 50 instead of 100) for faster but less precise results |
| `FileNotFoundError` on dataset | Check that `autolens_workspace_latest/` exists with datasets |

---

## Reference Texts

| Abbreviation | Full Citation |
|-------------|---------------|
| **C&K** | Congdon & Keeton (2018) — *Principles of Gravitational Lensing* |
| **N&B** | Narayan & Bartelmann (1997) — *Lectures on Gravitational Lensing* ([arXiv](https://arxiv.org/abs/astro-ph/9606001)) |
| **Saha+24** | Saha et al. (2024) — *Essentials of Strong Gravitational Lensing* |
| **Meneghetti** | Meneghetti (2021) — *Introduction to Gravitational Lensing* |
| **S92** | Schneider, Ehlers & Falco (1992) — *Gravitational Lenses* |
| **Nightingale+18** | Nightingale, Dye & Massey (2018) — *AutoLens* ([arXiv](https://arxiv.org/abs/1708.07377)) |
| **Cappellari02** | Cappellari (2002) — *Efficient multi-Gaussian expansion of galaxies* |
| **Emsellem+94** | Emsellem, Monnet & Bacon (1994) — *The multi-Gaussian expansion method* |

---

## Author

**Rodrigo Córdova Rosado**
Harvard-Smithsonian Center for Astrophysics
[rodrigo.cordova_rosado@cfa.harvard.edu](mailto:rodrigo.cordova_rosado@cfa.harvard.edu)

Built with assistance from [Claude Code](https://claude.ai/claude-code).

---

## License

Tutorial content (notebooks, LaTeX notes, Mathematica scripts, figures) is licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). The included PyAutoLens workspaces follow the [PyAutoLens MIT License](https://github.com/Jammy2211/autolens_workspace).
