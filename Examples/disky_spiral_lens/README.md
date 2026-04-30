# Example: Disky / Spiral Lens

## Status

✓ **Shipped.** `01_disky_spiral_fit.ipynb` walks through both the single-Sérsic and the bulge+disk multi-component fits with committed Cannon results in `results/single_sersic_fit/` and `results/bulge_disk_fit/`. Audited under `/autolens-fit-diagnostics`.

## Problem

Real lens galaxies often have **two light components at different position angles** — a high-Sersic-index bulge and a lower-n disk that's more flattened and rotated relative to the bulge. A single-Sersic light profile is mathematically a single ellipse in surface brightness; it cannot represent two orientations simultaneously. This example shows what that looks like in practice and demonstrates Bayes-factor model comparison between a single-Sersic vs bulge+disk light model.

## Data source

Simulated mock under `mocks/`: lens at $z=0.45$ with bulge (PA≈0°, *n*=4, *R_e*=0.45″) + disk (PA≈35°, *n*=1, *R_e*=1.0″), Isothermal+shear mass aligned with the bulge PA, compact Sersic source at $z=1.6$. Generation script in `mocks/generate_mock.py`.

## Method (as implemented in `01_disky_spiral_fit.ipynb`)

- **Variant 1 (single-Sersic light):** standard Module 03 model. Lens has one Sersic light component. Expect **catastrophic FAIL** at the lens centre — a 4-lobed red/blue residual cross at >40σ, the azimuthal *difference* between the two truth components (bulge PA ≈ 0° and disk PA ≈ 35°).
- **Variant 2 (bulge + disk):** add a second `Sersic` light component with independent `ell_comps`. The two components rotate independently and the residual collapses to noise — clean PASS at max\|res\| ≈ 4.2σ.
- The Bayes factor between the two is decisive (ΔlogZ several hundred), demonstrating that bulge+disk is required for this morphology.

## Exercises

1. **PA-difference threshold scan**: regenerate the mock (`mocks/generate_mock.py`) with smaller PA differences — say 10° instead of 35°. At what PA difference does the Bayes factor drop below $e^{10}$? Below that threshold a single Sersic is a sufficient approximation.
2. **Mass PA stability**: does the recovered mass PA differ between Variant 1 and Variant 2? (This is the mass/light-PA consistency check from Module 11 §1.)
3. **Compare to MGE-light** (Module 09): swap the bulge+disk Sersic pair for an MGE basis (`al.lp.Basis` with ~30 Gaussians) and verify the residual is at least as clean. MGE is the *non-parametric* generalisation of bulge+disk that handles arbitrarily complex morphologies.

## References

- Cappellari (2002), MNRAS 333, 400 — MGE methodology.
- Barnabè et al. (2012) — dynamical + lensing constraints on disky lenses.
- `autolens_workspace_latest/scripts/howtolens/chapter_2_lens_modeling/tutorial_4_light_model.py`.
- `Modules/09_MGE_Linear_Light_Profiles/` — the in-repo canonical MGE-light reference.

## Possible extensions

- **MGE variant** as a third notebook (`02_disky_mge.ipynb`) — would generalise this lesson from "two-component bulge+disk handles two orientations" to "MGE handles arbitrary morphology."
- **Real-data variant** using DESJ0206 (a known AGEL spiral lens). Data in the sibling AGEL workspace at `Spiral_2_spectra_fitting/` and `202509_DESJ02026/`.
