# Learning to Autolens

**Strong Gravitational Lens Modeling with PyAutoLens — A Step-by-Step Tutorial Suite**

A 10-module tutorial series teaching computational strong lens modeling from first principles through publication-ready results, with a dedicated cluster-computing module (Mod 10) showing how to run the heavy searches on Harvard FASRC Cannon. Each module pairs detailed **Jupyter notebooks** (with thorough physics commentary) with **LaTeX theory notes** and optional **Wolfram Mathematica** scripts for symbolic verification.

This project is a companion to [Learning to Lens](https://github.com/rodelcr/Learning_to_Lens) (GR & lensing theory in Mathematica), but is fully self-contained — all necessary theory is developed inline with references to the primary literature.

> **Alpha build (v0.92)** — April 2026. Modules 01–10 complete; Cannon cluster results for Mods 04, 05, and 09 are included under each module's `results/` directory. Feedback welcome at rodrigo.cordova_rosado@cfa.harvard.edu.

---

## Local Install

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

### 3. Register the env as a Jupyter kernel

```bash
python -m ipykernel install --user --name=autolens --display-name="Python (autolens)"
```

Without this step, a `jupyter lab` session launched from a different env (e.g. `base`) will not list the `autolens` kernel and you'll end up running notebooks under the wrong interpreter.

### 4. Verify the install

```bash
python -c "import autolens as al, autofit as af; print('autolens', al.__version__, '| autofit', af.__version__)"
```

Expect `autolens 2026.4.13.x` (or later in the 2026.4 line) on both packages.

### 5. Launch Jupyter

```bash
jupyter lab
```

Open any notebook from `Modules/`, then **Kernel → Change Kernel → Python (autolens)**. Start with **Module 01** if you're new, or **Module 09** if you want the production MGE/SLaM workflow.

### 6. (Optional) Wolfram Mathematica

Symbolic derivation scripts in `Mathematica/` require Mathematica 13.0+ or Wolfram Engine. These are supplementary — all key results are derived in the notebooks and LaTeX notes.

---

## Viewing pre-computed Cannon results (no fitting required)

Modules 04, 05, and 09 each publish their finished Cannon fits under `Modules/XX_*/results/` as small, git-tracked artifacts (fit subplot PNG, corner PDF, `samples.csv`, `summary.json`, `info.txt`, `model_results.txt`). You can inspect them end-to-end without a cluster account or a multi-hour local run.

In Jupyter Lab (with the `Python (autolens)` kernel selected):

1. Open the module notebook — e.g. `Modules/09_MGE_Linear_Light_Profiles/09_mge_linear_light_profiles.ipynb`.
2. Scroll to the section titled **"Viewing pre-computed results from the Cannon cluster"**.
3. Run that cell plus the `show_result(...)` cells underneath. They read the committed artifacts directly and render the fit subplot + corner plot inline.

The same viewer cells live in each `Solutions/XX_*_SOLVED.ipynb`, pointing at the shared `Modules/XX_*/results/` tree.

---

## Running fits yourself on Harvard FASRC Cannon

If you want to *reproduce* or *extend* the fits (rather than view the bundled results), Module 10 is the end-to-end runbook. The short version:

```bash
# on your laptop, from the repo root, after activating the autolens env
bash Modules/10_Cluster_Computing/scripts/push_to_cannon.sh --go         # mirror repo to Cannon
bash Modules/10_Cluster_Computing/scripts/seed_cannon_data.sh --go       # copy datasets + checkpoint.hdf5

# on Cannon
cd /n/holystore01/LABS/hernquist_lab/Lab/$USER/learning_to_autolens
sbatch --export=ALL,MODULE=04 Modules/10_Cluster_Computing/scripts/submit_cannon.slurm

# back on your laptop, after the job finishes
bash Modules/10_Cluster_Computing/scripts/pull_from_cannon.sh --go       # pull lightweight artifacts
```

The submit script bills compute to `--account=siag_lab` (fast scheduling) and writes output to Hernquist lab storage. Nautilus checkpoints in `checkpoint.hdf5` auto-resume on re-submit, so a requeued job picks up where it left off. See **Module 10** for the full layout, failure modes, and `export_results.py` artifact pattern.

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

### Part VI: Physics, Not Just Pipelines

| # | Module | What You'll Learn | Runtime |
|---|--------|-------------------|---------|
| 11 | [Physical Mass Models](Modules/11_Physical_Mass_Models/) | Auditing a converged fit against the *physical* bar (decomposability, light-consistency, reasonable scales, stability), six mandatory diagnostics, reading `model_results.txt` as a reviewer, upgrade path from Isothermal → composite → PowerLaw, when to refuse to publish | < 5 min (read-only on committed results) |

> **Runtimes** are approximate for a laptop with 8+ cores. Modules with non-linear searches (03–06, 09) take longer on first run; results are cached for subsequent runs. **Modules 04, 05, 06, 09** have ready-made cluster drop-ins (`Modules/10_Cluster_Computing/scripts/fit_module{04,05,06,09}.py`) if your laptop can't keep up.

See the **"Viewing pre-computed Cannon results"** section above for how to inspect finished fits without running anything.

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
