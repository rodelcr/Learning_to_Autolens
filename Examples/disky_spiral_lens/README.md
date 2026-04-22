# Example (stub): Disky / Spiral Lens

## Status

◯ **Planned** — no notebook yet.

## Problem

Most strong-lens pipelines assume an elliptical galaxy (Sérsic with ellipticity parameters, or MGE). But **spiral and disky galaxies** have non-elliptical morphology — bars, isophotal twists, m=4 boxiness — that a single elliptical profile can't capture. This example shows **how MGE-linear-light (Module 09)** lets you sidestep the elliptical assumption entirely, and what the image-plane residuals look like when you force a Sérsic on a disk.

## Data source

- Simulate: generate a disk-like light profile with `al.lp.ExponentialCore` + isophotal twist, add a Sersic+Isothermal mass.
- Real-world: AGEL has at least one spiral lens on record (DESJ0206; see `20250910-keerthi-Keck-AGELDR2-main/` or the AGEL DR2 catalog).
- `Spiral_2_spectra_fitting/` and `202509_DESJ02026/` in the sibling AGEL workspace have real DESJ0206 data we could reuse.

## Method hint

- **Track A (anti-example):** fit with Sérsic lens light + Isothermal mass. Expect residuals showing the disk structure that Sérsic can't capture — coherent m=4 cross residual, PA misalignment.
- **Track B (MGE light):** swap Sérsic lens light for `al.lp.Basis` with ~30 Gaussians linearised in intensity, as in Module 09. Mass stays Isothermal. The light residual should drop dramatically; the *mass* model — driven by the arc — should change by ≲1%.
- The pedagogical point: MGE light is decoupled from the mass fit. You can fit a faithful light model *without* changing your mass model's ellipticity.

## Exercises

1. Compare the mass PA recovered with Sérsic light vs MGE light. Does the mass think the galaxy is rotated differently depending on which light model it's paired with? (This is the same mass/light-PA consistency check from Module 11 §1.)
2. Does the MGE light fit have the same Einstein radius as the Sérsic fit, to within the Section-1 5% stability bar?
3. Apply Module 11's six-diagnostic audit to both. Track A should fail on panel 5 (Lens Light Subtracted has disk structure). Track B should pass.

## References

- Cappellari (2002), MNRAS 333, 400 — MGE methodology.
- Barnabè et al. (2012) — dynamical + lensing constraints on disky lenses.
- `autolens_workspace_latest/scripts/howtolens/chapter_2_lens_modeling/tutorial_4_light_model.py`.
- `Modules/09_MGE_Linear_Light_Profiles/` — the in-repo canonical MGE-light reference.

## To build this out

Same template as `Examples/compound_lens/`.
