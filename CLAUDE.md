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
| 10 | Cluster Computing | push/pull/submit on Cannon, Nautilus checkpointing, slurm provenance | — |
| 11 | Physical Mass Models | ✓ ship — 6-panel physical-bar audit, Pattern A–F failure catalogue, f_DM(<θ_E) extraction, γ′ recovery vs Auger+10, decision flowchart | C&K Ch.4-6, Auger+10, Sonnenfeld+13 |
| 12 | Time-Delay Cosmography & MSD | ✓ ship — Fermat potential, $D_{\Delta t} \propto 1/H_0$ vs $w$ numerical comparison, mass-sheet degeneracy ($\kappa \to \lambda\kappa + (1-\lambda)$) derived analytically and verified numerically on an SIE quad (image positions, flux ratios, time delays), TDCOSMO chain, hand-off to Module 13 (kinematics) and Module 14 (multi-plane) | Refsdal 1964, S92 Ch.5+11, Wong+20, Birrer+20, Treu & Marshall 2016 |
| 13 | TDCOSMO with Kinematics | Anisotropic Jeans, σ_v aperture projection, internal vs external MST, λ_int | B&T 2008 §4, Mamon & Łokas 2005, Birrer+20 (TDCOSMO IV), Schneider & Sluse 2013 |
| 14 | Compound (Multi-Plane) Lensing | Recursive multi-plane lens equation, distance ratios β_jk, multi-plane Fermat cross-terms | S92 Ch.9, Blandford & Narayan 1986, Schneider 2019, McCully+ 2014, Keeton 2001 |
| 15 | Radial Arcs & Caustic Topology | ✓ ship (v0.96) — Jacobian eigenvalues λ_t / λ_r, radial vs tangential critical curves, magnification asymptotics (1/d vs 1/√d), γ′ constraint from radial-arc position, γ′–M_BH degeneracy, hand-off to `Examples/radial_arc_smbh` for the AGEL Einstein-spiral methodology | C&K Ch.6, S92 §5, Sonnenfeld+13, Auger+10, Shajib+ (1st Einstein spiral), Ferrami+24 (DESJ0206) |

**Key:** C&K = Congdon & Keeton (2018), N&B = Narayan & Bartelmann (1997), S92 = Schneider Ehlers Falco (1992), B&T = Binney & Tremaine (2008)

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

The authoritative pins live in `requirements.txt`. Summary:

```
python           3.12            # autolens 2026.4+ requires ≥ 3.12
autolens         >= 2026.4.13    # pulls in autoarray/autoconf/autofit/autogalaxy transitively
nautilus-sampler (transitive)    # default sampler — replaces dynesty in modern autolens
matplotlib       >= 3.7, < 3.9   # PyAutoLens has a plotting incompat with 3.9+
jupyterlab       >= 4.5
astropy          >= 7.0
corner           >= 2.2
numba            >= 0.65
```

Install with `python -m pip install -r requirements.txt` (never bare `pip`, to
avoid a PATH-shadowed user-level pip installing into the wrong interpreter).
Verify with `python check_install.py`.

## Common Commands

```bash
# --- Install (one-time) ---------------------------------------------------
conda create -n autolens python=3.12 -y
conda activate autolens
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name=autolens --display-name="Python (autolens)"
python check_install.py

# --- Launch / run --------------------------------------------------------
jupyter lab                                # then Kernel → Python (autolens)
jupyter lab Modules/01_Basics_Grids_Galaxies_RayTracing/01_grids_galaxies_raytracing.ipynb

# --- Cannon round-trip (Modules 04/05/09) --------------------------------
bash Modules/10_Cluster_Computing/scripts/push_to_cannon.sh --go
bash Modules/10_Cluster_Computing/scripts/seed_cannon_data.sh --go   # first push only
# On Cannon:
#   sbatch --export=ALL,MODULE=04 Modules/10_Cluster_Computing/scripts/submit_cannon.slurm
bash Modules/10_Cluster_Computing/scripts/pull_from_cannon.sh --go

# --- Mathematica (optional) ----------------------------------------------
wolframscript -file Mathematica/lens_equation_symbolic.wl
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

## Cannon environment (2026-04-18)

- Production env on Cannon is **`autolens312`** (Python 3.12 + autolens 2026.4.13.6+). The slurm submit script defaults `CONDA_ENV=autolens312`.
- The legacy `autolens` env (Python 3.11 + autolens 2026.2.26.4) is left in place but is one minor version too old to have the `RectangularAdaptDensity` / `RectangularAdaptImage` / `reg.Adapt` classes our scripts use. Do not target it.
- **When installing into a Cannon conda env, always use `python -m pip`, not bare `pip`.** A stray `~/.local/bin/pip` (Python 3.10) shadows the env's pip in PATH and will silently install into `~/.local/lib/python3.10/site-packages/` instead.
- **When you see `AttributeError: module 'autolens' has no attribute 'X'`,** check `pip index versions autolens` against the current Python before assuming the API was renamed. Autolens releases monthly and minor-version drift is the most common cause.
