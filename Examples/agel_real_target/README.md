# Example (stub): Real AGEL Target on HST Data

## Status

◯ **Planned** — no notebook yet. This is the capstone.

## Problem

All prior Modules and Examples use simulated mocks with known truth parameters. This example takes a **real AGEL lens** from the survey, with **real HST imaging** and **real redshift uncertainties**, and runs the full modelling pipeline.

The goal: show a learner what *changes* between a clean mock and a messy real lens. Spoilers:

- Noise maps are not the image × 3% — they're correlated by drizzle, have systematic floors, and need explicit modelling.
- The PSF is derived from field stars, has position dependence you'll ignore, and has sub-pixel structure that matters at the arc.
- The mask needs manual editing to exclude unrelated foreground/background galaxies inside the ROI.
- The redshifts (particularly z_S) have finite uncertainty and should be marginalised over.
- Photometric lens-light subtraction leaves residuals that you *will* confuse with lensed arcs until you learn the signature.

## Data source

- AGEL DR2 catalog — Rodrigo has direct access via `20250910-keerthi-Keck-AGELDR2-main/`.
- Target selection: pick a well-characterised single-lens with confirmed spectroscopic redshifts. Good candidates:
  - AGEL013322 (has Keck spec)
  - DESJ02026 (disky, already has modelling tree under `202509_DESJ02026/`)
  - DESI-329 (well-studied)
  - CSWA164 (modelled in `202509_CSWA164_modeling/`)

## Method hint

- Start with **Module 07** (FITS → Model) as the reference for real-data handling.
- The pipeline should be the full **Mod 04 SLaM** workflow — *unchanged* from the toy mock in principle, but with each step carefully diagnosed because the data is no longer idealised.
- Post-fit audit with Module 11's physical bar is **mandatory**. Real data is where the *physical* bar matters most — a numerically good fit on real data with unphysical caustics is the most common failure mode in the literature.

## Exercises

1. End-to-end: FITS → fit → physical-bar audit → written verdict. All cells runnable.
2. Compare the recovered θ_E against the AGEL DR2 catalog value. How do your uncertainties compare?
3. Cross-check with σ_v from the Keck spectra (Exercise 4 in Module 11).
4. If the lens has a published literature model, compare posteriors.

## References

- AGEL survey paper (Jacobs+19, Huang+21).
- AGEL DR2 ongoing publications.
- `Modules/07_Real_Data_FITS_to_Model/`.
- SLACS modelling papers (Bolton+06, Auger+10) as the methodological template.

## To build this out

This is the most involved Example. Build order:
1. Pick the target and commit the FITS cutouts (image, noise, PSF, mask) under `mocks/` — or point at a read-only shared path.
2. Copy the `Examples/compound_lens/` template.
3. Custom mask (either interactive or committed boolean array).
4. Full SLaM pipeline with Mod 11 audit.
5. This notebook will be referenced back to from the other Examples as "the real-data reality check" — it's the capstone in spirit.
