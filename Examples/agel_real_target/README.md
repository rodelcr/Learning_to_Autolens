# Example: AGEL013322-125201A — Real HST Lens

> **v0.92 ships:** the data-prep recipe (`extract_cutout.py` + `build_empirical_psf.py`) and the empirical PSF that replaces the placeholder Gaussian. The direct-fit Cannon result (job 9676009, χ²/N=0.30) shows one 32σ residual at a hot pixel (cosmic-ray survivor) — this is real-data character not a fit failure.
> 
> **Research in progress (NOT in v0.92):** post-hot-pixel-mask refit; full SLaM-with-MGE fit; published parameters. The point of this example is *what changes when you move from clean mocks to real HST data*, not a science result on the AGEL013322 target.


## Status

◐ **In progress (scaffold)** — real HST cutout + noise map + **empirical PSF** (median-stacked from 14 bright isolated stars in the same drizzled frame) in place; driver + notebook scaffolded; Cannon results pending; mask + lens-light treatment caveats documented in §Caveats.

**2026-05-01 update.** The placeholder Gaussian PSF (σ = 0.036″) was replaced with an empirical PSF built by `data/build_empirical_psf.py` (FWHM ≈ 0.113″ — broader than the diffraction limit because of drizzle resampling + real instrumental wings). The Gaussian backup remains at `data/psf_gaussian_placeholder.fits` for comparison.

## What this example is for

All other Examples in this collection use simulated mocks with known truth parameters. This example takes **a real AGEL lens** with **real HST imaging** and runs the full modelling pipeline to show what changes between a clean mock and a messy real lens.

The target is **AGEL013322-125201A** (also catalogued as DCLS0133-1252) — a galaxy-scale strong lens at z_L ≈ 0.30 with a background source at z_S ≈ 1.6, both established spectroscopically by the AGEL DR2 effort. The HST imaging is a 674 s exposure with ACS/WFC F606W (HST proposal 17307, file `jf544b_drc.fits`).

## Data

Cutout extracted by `data/extract_cutout.py` from the full HST/ACS frame (5848 × 5855 px). The lens sits at full-frame pixel coordinate (3622, 882), found via segment-catalog lookup at the catalog RA/Dec. We extract a 200 × 200 px (10″ × 10″) stamp.

| File | Description |
|---|---|
| `data/image.fits` | 200×200 px @ 0.05″/px, units electrons/s |
| `data/noise_map.fits` | `1 / sqrt(WHT)` from the drizzled inverse-variance map; masked pixels (WHT=0) get σ = 10⁶ |
| `data/psf.fits` | **Empirical PSF** — 51×51 px median-stack of 14 bright isolated stars from the same drizzled frame, sigma-clipped, sub-pixel re-centred, normalised. FWHM ≈ 0.113″. Built by `data/build_empirical_psf.py`. |
| `data/psf_gaussian_placeholder.fits` | Backup of the original Gaussian placeholder (σ = 0.036″) for comparison. |
| `data/metadata.json` | provenance + redshifts + recipe documentation |
| `data/extract_cutout.py` | the extraction script (re-runnable from the original full-frame source) |

The original full-frame FITS lives outside the repository at `~/Documents/AGEL/AGEL013322-125201A_HST_ACS_606/hst_17307_4b_acs_wfc_f606w_jf544b_drc.fits` — the extraction script's `SRC` constant points there.

## Methodology

For the scaffold, a single direct fit:

- `lens.bulge = al.lp.Sersic` — lens light, fitted simultaneously with mass.
- `lens.mass = al.mp.PowerLaw` — γ′ free (Auger+10 found γ′ = 2.08 ± 0.16 for SLACS ellipticals; we let the data tell us).
- `lens.shear = al.mp.ExternalShear`.
- `source.bulge = al.lp.SersicCore` — single source component (the AGEL source's morphology is unknown a priori).
- Cosmology: FlatLambdaCDM(70, 0.30).
- Sampler: `af.Nautilus(n_live=200)`.

This is the *minimum viable* model — a publication-grade analysis would add (a) MGE for the lens light, (b) pixelised source for asymmetric arc structure, (c) fixed-centre external perturber if the AGEL DR2 catalog flags neighbouring galaxies, (d) the SLaM staged pipeline. See §Exercises.

## Running on Cannon

```bash
sbatch --export=ALL,EXAMPLE=agel_real_target,FIT_EXTRA_ARGS=--part=direct \
       Modules/10_Cluster_Computing/scripts/submit_cannon.slurm --time=04:00:00
```

Wall time: ~1–2 h on 32 cores for the direct fit. The faint lens light + sub-arcsec arcs make likelihood evaluations more expensive than the toy mocks.

## Caveats — what makes real data harder than mocks

Each item below is a *real* difference between this example and the synthetic ones. Awareness of these is the point of the capstone.

### 1. PSF — empirical (current) vs. publication-grade

**As of 2026-05-01**, `data/psf.fits` is an **empirical PSF** built from 14 bright isolated stars in the same drizzled frame (FWHM ≈ 0.113″). This captures real instrumental wings + drizzle resampling effects that a pure-Gaussian misses. The build is reproducible via `data/build_empirical_psf.py`.

**Remaining limitations** for publication-grade work:
- **Position dependence** — our stack averages PSFs from 14 different positions across the WFC field. The lens position has a slightly different PSF than the spatially-averaged stack. Solution: TinyTim-model the PSF at the exact lens (x, y, MJD).
- **Drizzle-induced PSF broadening** — drizzle resamples the native WFC PSF onto a finer grid, broadening it. Both the empirical stack and the actual lens-position PSF share this effect, so it cancels in our analysis IF the lens-position PSF is well-approximated by the spatially-averaged stack.
- **Time variation** — HST orbital thermal cycles cause slight focus drifts. The empirical stack is averaged over the visit, the lens is observed at one phase. Negligible for ACS/WFC at this exposure depth.

The original Gaussian placeholder (`data/psf_gaussian_placeholder.fits`) remains for comparison; rerunning the fit with both PSFs is a useful exercise to bracket the PSF systematic.

`autolens_workspace_latest/scripts/imaging/data_preparation/` has additional recipes for TinyTim integration.

### 2. Noise map — drizzle correlations

The `noise_map.fits` is `1 / sqrt(WHT)` per pixel, treating each pixel as independent. **HST drizzle introduces pixel-to-pixel correlations** — adjacent pixels share signal because drizzle resamples the original WFC pixels onto a finer output grid. The correct treatment is to use the AstroDrizzle output covariance, or to inflate the formal σ by the drizzle correlation length (typically a factor of ~1.3–1.6).

For our scaffold the simple per-pixel σ is acceptable — the lens dominates over the noise. For substructure work (`subhalo_sensitivity/`) it matters.

### 3. Mask — manual editing

The 2.7″ circular mask used by the driver is a placeholder. Real AGEL targets often have:
- Foreground galaxies inside the mask radius that need explicit exclusion.
- Diffraction spikes from nearby bright stars.
- Edge-of-detector vignetting.

A `data/mask_manual.fits` with hand-edited exclusions would replace the circular mask in production. See `autolens_workspace_latest/scripts/imaging/data_preparation/examples/manual_mask.py`.

### 4. Lens-light subtraction is harder than for mocks

The AGEL013322 lens light is faint (peak ~2 e/s in F606W) and the arcs are nearly comparable to the lens-light wings. A single `Sersic` lens light will likely leave residuals that *look like* arcs and confuse the source fit. **MGE lens light is strongly recommended** — see `mge_to_physical/` for the methodology.

### 5. Redshifts have finite uncertainty

The metadata records z_L = 0.30 and z_S = 1.6 from the AGEL DR2 spectroscopy. The 1σ uncertainties are **few-percent** on z_L and **~10%** on z_S. For cosmological inference (cf. Module 12) these need to be marginalised over, not held fixed. The scaffold fixes them; an exercise relaxes that.

## Exercises

1. **MGE lens light.** Replace the single Sersic lens light with an MGE basis (mirror `mge_to_physical/` Search 1). Re-fit. Did the arc residuals decrease? Bayes factor?
2. **Empirical PSF.** Build an empirical PSF by stacking 5 bright unsaturated stars from the segment catalog. Re-fit. Did anything shift?
3. **Pixelised source.** Replace the SersicCore source with a pixelised mesh (mirror `compound_lens/` 03 PIX). Real arcs often have asymmetric morphology that parametric profiles cannot capture.
4. **Redshift marginalisation.** Add `z_L` and `z_S` as nuisance parameters with Gaussian priors matching the spectroscopic uncertainty. Re-fit. How much does the recovered θ_E shift?
5. **Compare to the AGEL DR2 catalog.** Look up the published θ_E for AGEL013322 and compare to your recovered posterior. If they disagree at >2σ, why?
6. **Compare to the literature lensing model.** If a published lensing analysis exists for this target, compare γ′, ε, source size.
7. **SLaM staged pipeline.** Mirror `compound_lens/` 02 (or `mge_to_physical/`). Does the staged approach beat the direct fit on log_Z?

## References

- **AGEL DR2 documentation** — `~/Documents/AGEL/20250910-keerthi-Keck-AGELDR2-main/`.
- **HST proposal 17307** — the proposal text describes the AGEL HST imaging strategy.
- **`Examples/mge_to_physical/`** — the MGE-light methodology for Exercise 1.
- **`Examples/compound_lens/`** notebook 03 — the PIX source methodology for Exercise 3.
- **`Modules/07_Real_Data_FITS_to_Model/`** — the real-data preparation curriculum entry.
- **`autolens_workspace_latest/scripts/imaging/data_preparation/`** — official PyAutoLens data-prep recipes.

## What this example is NOT

Not a science publication on AGEL013322. The PSF, noise treatment, and lens-light handling all have placeholders that would need to be replaced before claiming a publication-grade result. This is a **template** showing the pipeline structure on real data — the scientific result is the methodology, not the inferred parameters.