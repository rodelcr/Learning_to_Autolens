# Learning to Autolens

**Strong Gravitational Lens Modeling with PyAutoLens — A Step-by-Step Tutorial Suite**

An 8-module tutorial series teaching computational strong lens modeling from first principles through publication-ready results. Each module pairs detailed **Jupyter notebooks** (with thorough physics commentary) with optional **Wolfram Mathematica** scripts for symbolic verification of key equations.

This project is a companion to [Learning to Lens](https://github.com/rodelcr/Learning_to_Lens) (GR & lensing theory in Mathematica), but is fully self-contained — all necessary theory is developed inline with references to the primary literature.

---

## Prerequisites

### Knowledge
- Gravitational lensing fundamentals: lens equation, Einstein radius, convergence, shear, magnification
- Basic Python (NumPy, Matplotlib, Astropy)
- Familiarity with MCMC / Bayesian inference is helpful but covered in Module 03

### Software

| Software | Version | Purpose | Required? |
|----------|---------|---------|-----------|
| **Python** | 3.9+ | Runtime | **Yes** |
| **PyAutoLens** | >= 2024.1 | Lens modeling | **Yes** |
| **Jupyter Lab** | any | Notebook environment | **Yes** |
| **Wolfram Mathematica** | 13.0+ | Symbolic derivations | Optional |

### Installation

```bash
# Clone the repository
git clone git@github.com:rodelcr/Learning_to_Autolens.git
cd Learning_to_Autolens

# Create a conda environment (recommended)
conda create -n autolens python=3.11
conda activate autolens

# Install PyAutoLens and dependencies
pip install autolens jupyterlab astropy matplotlib

# Launch Jupyter
jupyter lab
```

---

## Module Curriculum

### Part I: Foundations

| Module | Title | What You'll Learn |
|--------|-------|-------------------|
| **01** | [Basics: Grids, Galaxies, Ray-Tracing](Modules/01_Basics_Grids_Galaxies_RayTracing/) | How PyAutoLens represents the lens equation computationally: Grid2D objects, light profiles (Sersic), mass profiles (SIE, NFW), Galaxy objects, and the Tracer that performs multi-plane ray-tracing. |
| **02** | [Simulating Lens Data](Modules/02_Simulating_Lens_Data/) | Generate mock lensed images with realistic PSFs, noise, and instrument models (HST, Euclid, Keck AO). Understand what goes into the data before you try to model it. |
| **03** | [Your First Lens Model](Modules/03_First_Lens_Model/) | Bayesian lens modeling with Dynesty: setting up priors, running a non-linear search, interpreting posteriors, and understanding degeneracies. |

### Part II: Advanced Modeling

| Module | Title | What You'll Learn |
|--------|-------|-------------------|
| **04** | [Search Chaining & SLaM](Modules/04_Search_Chaining_SLaM/) | The Source, Light, and Mass (SLaM) pipeline: chaining searches for robust, scalable lens modeling. |
| **05** | [Pixelized Source Reconstructions](Modules/05_Pixelized_Source_Reconstructions/) | Non-parametric source modeling with Voronoi meshes and regularization — reconstructing source galaxies without assuming a light profile. |
| **06** | [Multi-Component Mass Models](Modules/06_Multi_Component_Mass_Models/) | Decomposing lens mass into stellar (Sersic) + dark matter (NFW) components, and what this tells us about galaxy structure. |

### Part III: Real Science

| Module | Title | What You'll Learn |
|--------|-------|-------------------|
| **07** | [Real Data: FITS to Model](Modules/07_Real_Data_FITS_to_Model/) | Taking real FITS data (HST, ground-based) through preparation, masking, PSF handling, and fitting — with examples from the AGEL survey. |
| **08** | [Results, Diagnostics & Figures](Modules/08_Results_Diagnostics_Figures/) | Extracting science from model results: corner plots, residual analysis, Einstein radius measurements, and publication-quality figures. |

---

## Repository Structure

```
Learning_to_Autolens/
├── Modules/                              # Tutorial modules (start here!)
│   └── XX_Module_Name/
│       ├── XX_module_name.ipynb          # Main tutorial notebook
│       └── *.py                          # Supporting scripts
├── Mathematica/                          # Symbolic derivations (.wl scripts)
├── Figures/                              # Exported publication figures
├── autolens_workspace_original/          # Unmodified PyAutoLens workspace (reference)
│   ├── scripts/howtolens/               # Original HowToLens tutorials
│   ├── slam/                            # SLaM pipeline source code
│   ├── config/                          # PyAutoLens YAML config
│   └── dataset/                         # Example datasets
├── CLAUDE.md                            # Project context for Claude Code
├── PROGRESS_LOG.md                      # Work log
└── README.md                            # This file
```

---

## Reference Texts

| Abbreviation | Full Citation |
|-------------|---------------|
| **C&K** | Congdon, A.B. & Keeton, C.R. (2018) — *Principles of Gravitational Lensing* |
| **N&B** | Narayan, R. & Bartelmann, M. (1997) — *Lectures on Gravitational Lensing* ([arXiv](https://arxiv.org/abs/astro-ph/9606001)) |
| **Saha+24** | Saha, P. et al. (2024) — *Essentials of Strong Gravitational Lensing* |
| **Meneghetti** | Meneghetti, M. (2021) — *Introduction to Gravitational Lensing* |
| **S92** | Schneider, P., Ehlers, J. & Falco, E.E. (1992) — *Gravitational Lenses* |
| **Nightingale+18** | Nightingale, J.W., Dye, S. & Massey, R.J. (2018) — *AutoLens* ([arXiv](https://arxiv.org/abs/1708.07377)) |

---

## Author

**Rodrigo Córdova Rosado**
Harvard-Smithsonian Center for Astrophysics
[rodrigo.cordova_rosado@cfa.harvard.edu](mailto:rodrigo.cordova_rosado@cfa.harvard.edu)

Built with assistance from [Claude Code](https://claude.ai/claude-code).

---

## License

Tutorial content (notebooks, scripts, figures) is licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). The included `autolens_workspace_original/` follows the [PyAutoLens MIT License](https://github.com/Jammy2211/autolens_workspace).
