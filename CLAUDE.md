# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Learning to Autolens** is a step-by-step educational tutorial suite teaching strong gravitational lens modeling with **PyAutoLens**. It is designed for researchers and advanced students who understand lensing theory and want to learn computational lens modeling — from grids and ray-tracing through full SLaM pipeline fitting of real data.

This project is a companion to (but independent of) [Learning to Lens](https://github.com/rodelcr/Learning_to_Lens), which covers the GR and lensing theory in Mathematica. Both projects reference the same source texts and can be used together or independently.

**Project lead:** Rodrigo Córdova Rosado (rodrigo.cordova_rosado@cfa.harvard.edu, Harvard CfA)

## Repository Structure

```
Learning_to_Autolens/
├── Modules/                              # Numbered tutorial modules (Jupyter notebooks)
│   ├── 01_Basics_Grids_Galaxies_RayTracing/
│   ├── 02_Simulating_Lens_Data/
│   ├── 03_First_Lens_Model/
│   ├── 04_Search_Chaining_SLaM/
│   ├── 05_Pixelized_Source_Reconstructions/
│   ├── 06_Multi_Component_Mass_Models/
│   ├── 07_Real_Data_FITS_to_Model/
│   └── 08_Results_Diagnostics_Figures/
├── Mathematica/                          # Symbolic derivations (.wl scripts, .nb notebooks)
├── Figures/                              # Exported figures (PDF/PNG)
├── Output/                               # Generated fitting output (git-ignored)
├── autolens_workspace_original/          # PyAutoLens workspace (v2025.11 reference copy)
│   ├── scripts/                          # Original PyAutoLens tutorial scripts
│   ├── notebooks/                        # Original PyAutoLens tutorial notebooks
│   ├── slam/                             # SLaM pipeline modules
│   ├── config/                           # PyAutoLens YAML configuration
│   └── dataset/                          # Example lens datasets
├── autolens_workspace_latest/            # PyAutoLens workspace (v2026.2, for Module 09+)
│   ├── scripts/                          # Latest tutorial scripts (MGE, SLaM, etc.)
│   └── dataset/                          # Latest example datasets
├── CLAUDE.md                             # This file
├── PROGRESS_LOG.md                       # Timestamped work log
└── README.md                             # Setup instructions and module curriculum
```

## Module Curriculum

| # | Module | Key Topics | Theory References |
|---|--------|------------|-------------------|
| 01 | Basics: Grids, Galaxies, Ray-Tracing | Grid2D, LightProfile, MassProfile, Galaxy, Tracer | C&K Ch.3-4, N&B Sec.2-3 |
| 02 | Simulating Lens Data | Simulator, PSF, noise, instrument models | C&K Ch.8 |
| 03 | Your First Lens Model | NonLinearSearch, Dynesty, priors, results | C&K Ch.8, Meneghetti Ch.8 |
| 04 | Search Chaining & SLaM Pipeline | Chaining, source_lp, source_pix, mass_total | — |
| 05 | Pixelized Source Reconstructions | Voronoi mesh, regularization, inversions | Suyu+ 2006, Vegetti+ 2009 |
| 06 | Multi-Component Mass Models | Stellar + dark matter, NFW + Sersic, scaling relations | C&K Ch.4, Meneghetti Ch.5 |
| 07 | Real Data: FITS to Model | Data preparation, masking, PSF handling, AGEL targets | — |
| 08 | Results, Diagnostics & Figures | Corner plots, residuals, publication figures | — |
| 09 | MGE & Linear Light Profiles | lp_linear, lp_basis.Basis, mge_model_from, MGE SLaM | Cappellari 2002, Emsellem+ 1994 |

**Key:** C&K = Congdon & Keeton (2018), N&B = Narayan & Bartelmann (1997)

## Key Conventions

### Notebook style
- **Header block** at the top of every notebook: title, purpose, prerequisites, references
- **Markdown cells** before every code cell explaining the physics and the "why"
- **Inline comments** in code cells for non-obvious operations
- **Textbook references** where applicable (e.g., "cf. Congdon & Keeton eq. 4.12")
- **Self-contained**: each module defines all variables; no reliance on global state from other modules

### File naming
- `snake_case` for all files
- Numbered prefixes for ordered modules (`01_`, `02_`, ...)
- Mathematica: `.wl` for scripts (headless), `.nb` for interactive notebooks

### Code commenting
- Every Python file: header block with purpose, inputs, outputs, source reference
- Every notebook: markdown cells explaining each section's physics and motivation
- Reference equation numbers from source texts where applicable

## Key Dependencies

```
autolens >= 2024.1
autofit
dynesty
numpy
matplotlib
astropy
scipy
```

## Common Commands

```bash
# Install PyAutoLens
pip install autolens

# Run a tutorial notebook
jupyter lab Modules/01_Basics_Grids_Galaxies_RayTracing/01_grids_galaxies_raytracing.ipynb

# Run a Mathematica script
wolframscript -file Mathematica/lens_equation_symbolic.wl

# Run Jupyter Lab
jupyter lab
```

## Reference Texts

These tutorials draw from the following texts (referenced by abbreviation):
- **C&K**: Congdon & Keeton (2018) — *Principles of Gravitational Lensing*
- **N&B**: Narayan & Bartelmann (1997) — *Lectures on Gravitational Lensing* (arXiv:astro-ph/9606001)
- **Saha+24**: Saha et al. (2024) — *Essentials of Strong Gravitational Lensing*
- **Meneghetti**: Meneghetti (2021) — *Introduction to Gravitational Lensing*
- **S92**: Schneider, Ehlers & Falco (1992) — *Gravitational Lenses*
- **Nightingale+**: Nightingale, Dye & Massey (2018) — *AutoLens* (arXiv:1708.07377)

## Important Reminders

- **Progress log:** Update `PROGRESS_LOG.md` after completing significant work.
- **Self-contained:** Each module must be runnable independently — do not require the user to have completed prior modules (though they build conceptually).
- **Theory first, code second:** Every code block should be preceded by a markdown cell explaining the physics.
- **Reference the original:** The `autolens_workspace_original/` directory preserves the unmodified PyAutoLens workspace for reference. Our modules in `Modules/` are the educational layer built on top.
