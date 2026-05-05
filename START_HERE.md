# Start Here — Learning to Autolens v0.92-alpha

**You're new. This page tells you exactly what to read and run, in what order. ~30 min to get oriented; ~3 hours to be productive on a real fit.**

---

## What this repo is, in 3 sentences

A 14-module tutorial series teaching computational strong gravitational lens modeling with [PyAutoLens](https://github.com/Jammy2211/PyAutoLens), from grids and ray-tracing through full SLaM-pipeline fits of real HST data. Each module is a Jupyter notebook with detailed physics commentary; each example in `Examples/` is a different lens architecture (compound multi-plane, double source-plane, group-scale BGG+satellites, real AGEL HST data, etc.). The point is to teach the **methodology**, not just the API — every fit is audited, every failure mode catalogued.

## What v0.92-alpha contains

**Audited and ready for students:**

- ✓ **Modules 01-10**: the full curriculum, from grids to cluster computing
- ✓ **3 recipe notebooks**: pixelization (Mod 05), MGE light (Mod 09), SLaM staging (Examples/compound_lens_zoo)
- ✓ **4 climb bridge notebooks**: from single-deflector to compound, group, and double-source-plane architectures
- ✓ **3 cluster docs**: SETUP_NEW_USER, STUDENT_QUICKSTART, RECIPES — full Cannon workflow without needing AI assistance
- ✓ **5 audited example fits**: compound_lens, compound_lens_zoo R0-R5 ladder, double_source_plane, disky_spiral_lens, group_scale truth_anchored

**Marked as "research-in-progress"** (visible but not expected to converge for you): cluster_scale, mge_to_physical, agel_real_target post-hot-pixel-cleanup, several other examples. See [`V092_SCOPE.md`](V092_SCOPE.md) for the full ship/defer breakdown.

---

## The 30-minute orientation tour

Read these in order. Don't run anything yet — just read.

1. **This file (5 min)** — orients you to the repo philosophy.
2. **[`README.md`](README.md) (10 min)** — install instructions, prerequisites, the module-by-module table.
3. **[`Modules/01_Basics_Grids_Galaxies_RayTracing/01_grids_galaxies_raytracing.ipynb`](Modules/01_Basics_Grids_Galaxies_RayTracing/01_grids_galaxies_raytracing.ipynb)** — open in Jupyter and skim the markdown cells. Don't run yet. This is what every other module builds on.
4. **[`Examples/README.md`](Examples/README.md) (5 min)** — the practice-gym roadmap with 11 architectures.
5. **[`Modules/10_Cluster_Computing/STUDENT_QUICKSTART.md`](Modules/10_Cluster_Computing/STUDENT_QUICKSTART.md) (5 min)** — daily-loop cluster cheat sheet (you'll need it for any production-scale fit).

---

## The 3-hour productive path

After orientation, work through this:

### Hour 1 — laptop install + first ray-traced lens

```bash
# One-time setup
conda create -n autolens python=3.12 -y
conda activate autolens
python -m pip install -r requirements.txt
python -m ipykernel install --user --name=autolens --display-name="Python (autolens)"
python check_install.py
```

Then run **Modules/01-03** end-to-end:
- 01: ray-trace your first lens (a few minutes)
- 02: simulate your own lensing data
- 03: fit the simulated data with Nautilus (~2 min on a laptop)

After Module 03 you've successfully run the full forward + inverse pipeline.

### Hour 2 — pick an architecture from `Examples/`

Each example has a **`00_climb_to_*.ipynb`** bridge notebook (skip-guarded, runs in <60 s) that walks you from "single deflector" (what you just learned in Mod 03) to the architecture in question. Pick one based on what interests you:

| Want to learn… | Open this |
|---|---|
| Multi-plane ray-tracing (2 lens galaxies @ different z) | `Examples/compound_lens/00_climb_to_compound.ipynb` |
| Why a single Sersic source can't capture complex morphology | `Examples/disky_spiral_lens/01_disky_spiral_fit.ipynb` |
| Cosmological β-ratio from 1 lens + 2 sources | `Examples/double_source_plane/00_climb_to_dspl.ipynb` |
| BGG + satellites (group-scale physics) | `Examples/group_scale/00_climb_to_group.ipynb` |
| The big picture of the 5-mock zoo | `Examples/compound_lens_zoo/00_climb_to_compound.ipynb` |

Each climb notebook ends with a hand-off pointer to the production notebook in the same directory.

### Hour 3 — pick a technique recipe

The three "recipe" notebooks teach the cross-cutting techniques you'll use on real data:

| Recipe | When you need it |
|---|---|
| `Modules/05_Pixelized_Source_Reconstructions/06_pixelization_recipe.ipynb` | Source has complex morphology that a Sersic can't capture |
| `Modules/09_MGE_Linear_Light_Profiles/05_mge_recipe.ipynb` | Lens galaxy isn't elliptical — bulge+disk, isophotal twist |
| `Examples/compound_lens_zoo/03_slam_recipe.ipynb` | Your direct fit has too many parameters and won't converge — stage it |

Each is a 5-6 step recipe card you can adapt to a new dataset.

---

## Working with real AGEL HST data

If you're an AGEL collaborator and want to fit a *new* AGEL target end-to-end:
**[`Examples/agel_real_target/AGEL_QUICKSTART.md`](Examples/agel_real_target/AGEL_QUICKSTART.md)** is the AGEL-specific recipe — assumes you know PyAutoLens, walks you through the AGEL data peculiarities (cutout extraction, empirical PSF from drizzled stars, hot-pixel masking, Keck spectroscopic redshifts, slurm routing for a new target). ~2 hours to first Cannon submit.

## When you're ready to run a real Cannon fit

Once you've done the orientation + first fit + climb + recipe pass, you're ready for production. The path:

1. **One-time**: [`Modules/10_Cluster_Computing/SETUP_NEW_USER.md`](Modules/10_Cluster_Computing/SETUP_NEW_USER.md) — Cannon account, SSH alias, conda env on the cluster.
2. **Daily loop**: [`Modules/10_Cluster_Computing/STUDENT_QUICKSTART.md`](Modules/10_Cluster_Computing/STUDENT_QUICKSTART.md) — push/submit/check/pull cheat sheet.
3. **Recipes**: [`Modules/10_Cluster_Computing/RECIPES.md`](Modules/10_Cluster_Computing/RECIPES.md) — copy-pasteable `sbatch` invocation for every fit in this repo.

The slurm entry point is **[`Modules/10_Cluster_Computing/scripts/submit_cannon.slurm`](Modules/10_Cluster_Computing/scripts/submit_cannon.slurm)** — generic, parametrized via `cannon.env`. **Hernquist-lab members** can `cp Modules/10_Cluster_Computing/cannon.env.hernquist Modules/10_Cluster_Computing/cannon.env` (only `CANNON_USER` to fill in); other lab members use [`cannon.env.example`](Modules/10_Cluster_Computing/cannon.env.example) as a template.

Wall-time budgets per recipe are listed in RECIPES.md (most fits are 1-12 h on Cannon's `hernquist` partition with 32 cores).

---

## Auditing fits

Every committed result in this repo has been audited via the **`/autolens-fit-diagnostics`** standard (a numerical + visual quality bar). The verdicts you'll see in `summary.json` files:

| Bar | strict-PASS | borderline-PASS | SUSPECT | FAIL |
|---|---|---|---|---|
| `chi_squared_per_pixel` | ≤1.3 | ≤1.3 | 1.3-2.0 | ≥2.0 |
| `max_abs_normalized_residual` | ≤4σ | 4-5σ at 9k+ pixels (Bonferroni-expected outliers) | 5-6σ | ≥6σ |
| Visual residual map | white noise | white noise + isolated outliers | clipped coherent structure | ring/cross/arc structure |

For a 9000-pixel mask, max\|res\| ~4.3σ is the **Bonferroni-corrected expected max under pure white noise**. So a 4-5σ peak with no coherent structure is the *noise floor*, not a fit failure. v0.92 ships fits in this band as borderline-PASS.

Open `fit_subplot.png` for any committed result; the residual map panel is the truth-teller.

---

## What's research-frontier (NOT in v0.92, may show up in v0.93+)

These are visible in the repo, marked with banners on their READMEs, but **not expected to converge for you**:

- `Examples/cluster_scale/` — direct fit FAILED (BCG-vs-FJ-amplitude degeneracy collapse); truth-anchored variant currently in flight on Cannon.
- `Examples/mge_to_physical/` — v2 fits show improvement but max\|res\| ~9.7σ remains.
- `Examples/agel_real_target/` — direct fit completed but with one 32σ hot-pixel residual; needs data-prep cleanup.
- `Examples/quad_time_delay/`, `subhalo_sensitivity/`, `interferometer_basic/`, `bayesian_model_comparison/` — scaffolded only, results pending.
- `Examples/compound_lens_zoo/` §13-§15 of the ladder notebook — truth + staged + free-cosmology rungs are research-in-progress.

The plan for v0.93 is to land the in-flight cluster + DSPL β-cosmography + compound zoo truth-anchored results, and the AGEL hot-pixel cleanup.

---

## When you're stuck

1. **Notebook errors / install issues**: `python check_install.py` first; then `Modules/10_Cluster_Computing/STUDENT_QUICKSTART.md` "When in doubt" section.
2. **Cluster errors**: `Modules/10_Cluster_Computing/STUDENT_QUICKSTART.md` "When a job fails" table.
3. **Fit converged but residuals look bad**: open `fit_subplot.png`, apply the bar above. If max\|res\| > 6σ or there's coherent structure, the fit is FAILED — don't trust the parameters.
4. **API errors after autolens version bump**: see `CLAUDE.md` § "Cannon environment" for the conda-env / pip pitfalls.
5. **Anything else**: open an issue or email rodrigo.cordova_rosado@cfa.harvard.edu.

---

*Learning to Autolens v0.92-alpha — Rodrigo Córdova Rosado, Harvard CfA*
