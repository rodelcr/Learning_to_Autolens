# Example: Compound-Lens Zoo

## Status

◐ **In progress** — five mocks (mock_2 through mock_6) adapted from lenstronomy, unified driver + zoo notebook scaffolded, Cannon results pending.

## What this example is for

The Examples shipped so far (`compound_lens`, `mge_to_physical`) each fit **one** mock with carefully-chosen priors and a tuned methodology. That tells you the methodology *can* work, but not whether it *generalises*. The zoo answers the generalisation question:

> Hold the model and the priors **fixed**. Throw five different lensing geometries (different z_l, z_s, γ′, ε, cosmologies) at the same fit. How does the recovered θ_E, γ′, e₁, e₂ correlate with the truths *across* the zoo?

A scientific methodology that recovers the truth on one curated mock is unconvincing. A methodology that recovers truths on five mocks with varying geometries — using one prior set and one fit recipe — is robust enough to point at real data.

## The five mocks

All five are **lenstronomy-generated compound systems** from `tutorials_DB_2025_09/mocks/images/mock_{2,3,4,5,6}_image.fits`. They share the architecture (EPL + ExternalShear + EPL secondary + 2 Sersic sources + 1 Sersic lens light), but vary the geometry and cosmology. The Pattern E story from `compound_lens` (secondary Einstein radius θ_E ≤ 0.12″, small enough to be absorbed into shear at first pass) holds for all of them — see Exercise 1 for explicit re-introduction.

| Mock | z_l | z_s | θ_E (primary) | γ′ | Cosmology | What's interesting |
|---|---|---|---|---|---|---|
| **2** | 0.3 | 1.7 | 1.87″ | 1.95 | Ωₘ=0.25, w=−0.9 | shallow γ′ + non-standard cosmology |
| **3** | 0.5 | 2.0 | 2.00″ | 2.50 | Standard | steep γ′ (γ′=2.5 is at the rail of EPL physical range) |
| **4** | 0.4 | 1.3 | 1.60″ | 1.90 | Standard | small lens, low z_s (compact arc) |
| **5** | 0.3 | 1.7 | 1.71″ | 1.95 | Ωₘ=0.35, w=−1.2 | matches mock_2 geometry but with different cosmology — direct test of how parameter inferences shift when you mis-specify cosmology |
| **6** | 0.24 | 2.0 | 1.88″ | 2.25 | Standard | low z_l, secondary mass = 0 (true single-deflector — Pattern E in pure form) |

Each mock is 110×110 px @ 0.05″/px, exp_time=500 s, bg_RMS=0.004. PSF is shared (`lenstronomy_mock_psf.fits`, 7×7 stamp).

Truths are parsed into `truths_mock_<N>.json` for verification. The image FITS, params text file, and PSF are copied verbatim from the lenstronomy source.

## Methodology

**One model, one prior set, fit five times.** The unified model:

- `lens.mass = al.mp.PowerLaw` — γ′ is FREE (`TruncatedGaussianPrior(2.0, 0.3, [1.5, 2.7])`); centre tied to `lens.bulge`.
- `lens.bulge = al.lp.Sersic` — wide priors on R_e, n, intensity, ellipticity.
- `lens.shear = al.mp.ExternalShear` — `Gaussian(0, 0.1)` on each component (absorbs the secondary deflector + LOS structure).
- `source.bulge = al.lp.SersicCore` — wide source priors.
- Cosmology: **fixed at FlatLambdaCDM(70, 0.30) for all five mocks** — including those with non-standard truths. Mocks 2 and 5 thus probe how cosmology mis-specification feeds into the recovered γ′ and θ_E.
- Sampler: `af.Nautilus(n_live=200)`. ~25 free parameters per fit.

The driver loops over mocks 2–6 sequentially. Each fit uses the *same* prior set; only the loaded image/noise FITS differ.

## Notebook

`01_zoo_overview.ipynb` — loads truths + Cannon results for all five mocks, builds a single comparison table (recovered vs truth for each mock), and overlays the residual maps. The audit cell summarises which mocks pass on the physical bar (chi²/N≤1.3, max|res|≤4σ) and where the Cannon-fit values land relative to truth in σ-units.

## Running on Cannon

```bash
sbatch --export=ALL,EXAMPLE=compound_lens_zoo,FIT_EXTRA_ARGS=--mock=all \
       Modules/10_Cluster_Computing/scripts/submit_cannon.slurm --time=12:00:00
```

`--mock=N` (where N ∈ {2,3,4,5,6}) for one mock. `--mock=all` for the full zoo (~10 h). Each individual mock takes ~1–2 h on 32 cores.

## Exercises

1. **Compound deflector check (per-mock).** Add the secondary EPL back as an `extra_galaxies` perturber with photometric centre (from truth) and free Einstein radius. Compare the recovered θ_E_secondary against the truth values (0.08″ to 0.12″, except mock_6 which has true 0.0″). Mock_6 is the cleanest test of Pattern E — does the secondary collapse to 0 as expected?
2. **Cosmology mis-specification (mocks 2 and 5).** These were generated with non-standard cosmology. Fit them BOTH at the truth cosmology AND at the standard one, and compare the recovered γ′ posteriors. The shift is the cosmology-mass-profile degeneracy showing up.
3. **γ′ recovery vs truth.** Plot recovered γ′ against truth γ′ for all five mocks. Is the spread consistent with the formal posterior errors? A systematic offset would indicate a bias in the prior or methodology.
4. **Single-source vs two-source.** Each truth has TWO Sersic sources. The canonical fit uses ONE SersicCore. Re-run mock_3 (steep γ′) with two sources and compare log_Z. Does the single-source approximation cost evidence?
5. **Pixelised source (mock with two sources).** For one mock, swap the parametric source for a pixelised source (`al.mesh.RectangularAdaptImage` + `al.reg.Adapt`, mirroring `compound_lens` 03 PIX path). Does the pixelisation absorb the second-source flux that the parametric model misses?

## References

- The original lenstronomy mocks are documented at `lenstronomy_AGEL_modules/tutorials_DB_2025_09/README_parameters.md`.
- Lenstronomy → autolens parameter conventions: `Examples/mge_to_physical/mocks/PROVENANCE.md` (the canonical reference for this repo).
- `Examples/compound_lens/` — the curated single-mock companion that established the priors used in this zoo.
- `LEARNING_LOG.md` — Pattern E (forced-compound suboptimum), centre-tuple convention swap.
