# Mock provenance — `lenstronomy_mock_1`

## Origin

This mock dataset was generated with **lenstronomy 1.10.4** by the AGEL group as part of the `lenstronomy_AGEL_modules/tutorials_DB_2025_09/` tutorial collection. The original lives at:

```
~/Documents/AGEL/lenstronomy_AGEL_modules/tutorials_DB_2025_09/mocks/images/mock_1_image.fits
~/Documents/AGEL/lenstronomy_AGEL_modules/tutorials_DB_2025_09/mocks/PSF/mock_psf.fits
~/Documents/AGEL/lenstronomy_AGEL_modules/tutorials_DB_2025_09/mock_true_params/mock_1_params.txt
```

The image FITS and PSF FITS are copied verbatim into this `mocks/` dir. The truth-params text file is also copied verbatim and additionally parsed by `build_mock_artifacts.py` into a tidy `truths.json` consumed by the notebook for verification.

## Why this mock

`mock_1` was chosen for the MGE-to-physical example because:

1. **Standard cosmology** (`FlatLambdaCDM(H0=70, Om0=0.30)`) — no need to disentangle non-standard cosmography from the mass-decomposition story.
2. **Cleanly-separated deflectors**: primary at z=0.5 (`θ_E = 1.65″`) and secondary at z=0.8 (`θ_E = 0.11″`). The secondary is small enough that its contribution can be treated as part of the external shear at first pass, the same way we handle the secondary in `compound_lens` v4. An exercise will explicitly add the secondary back as a fixed-centre `extra_galaxies` perturber.
3. **Cuspy lens light** (Sersic with `R_e = 1.9″`, `n = 4.9`) — exactly the regime where MGE buys you accuracy that single-Sersic does not. A pure de Vaucouleurs has subtle deviations from a single Sersic at large radii; an MGE captures them.
4. **Two source planes for free**: the truth source is two Sersic components (one near each deflector). For the canonical fit we use ONE source `SersicCore` (the dominant component) and treat the second as an exercise — this keeps the focus on the *lens* mass decomposition rather than source modelling.
5. **AGEL-native data style**: 110×110 px @ 0.05″/px (HST WFC3 IR scale), background RMS 0.004 × 500 s exp_time — matches typical AGEL pipeline output.

## Noise map

The lenstronomy mock files do NOT include a noise FITS. We synthesise one analytically using the truth-file `background_rms = 0.004` and `exp_time = 500 s`:

```python
sigma_pix = sqrt( max(image_e_per_s, 0) / exp_time + background_rms**2 )
```

This Gaussian-equivalent recipe matches autolens's own `simulator.py` convention. For real data you would substitute the noise map output by the reduction pipeline (e.g. `astropy.nddata` from drizzled flat-fields).

## Truths

`truths.json` decodes the lenstronomy `kwargs_lens / kwargs_source / kwargs_lens_light` dicts into a JSON-friendly structure. Key truth values (for verification at the end of the audit):

| Quantity | True value |
|---|---|
| `H0`, `Ωₘ` | 70.0, 0.30 |
| `z_lens_primary` | 0.5 |
| `z_lens_secondary` | 0.8 |
| `z_source` | 1.7 |
| Primary EPL `θ_E` | 1.65″ |
| Primary EPL `γ'` | 2.15 |
| Primary EPL `e1, e2` | (−0.13, −0.07) |
| Primary EPL `centre` | (0.01, −0.08)″ |
| External `γ_ext`, `ψ_ext` | 0.016, −0.20 rad |
| Lens light `R_e`, `n` | 1.9″, 4.9 |
| Source 1 `R_e`, `n` | 0.19″, 2.3 |
| Source 2 `R_e`, `n` | 0.15″, 1.5 |

## Lenstronomy → autolens parameter conventions

The `truths.json` keeps the lenstronomy convention. Notes for translation:

- **EPL** in lenstronomy → `al.mp.PowerLaw` in autolens. `theta_E` and `gamma` (= `slope`) carry over directly. **Ellipticity components `e1, e2` use the same definition** (axis-ratio + PA flavour, not `1−q²/1+q²`).
- **SHEAR_GAMMA_PSI** (`γ_ext`, `ψ_ext`) → `al.mp.ExternalShear` (`gamma_1`, `gamma_2`). Conversion: `gamma_1 = γ_ext * cos(2 ψ_ext)`, `gamma_2 = γ_ext * sin(2 ψ_ext)`. For mock_1: `γ_ext=0.016, ψ_ext=−0.2 rad → gamma_1 ≈ 0.0148, gamma_2 ≈ −0.00624`.
- **SERSIC_ELLIPSE** → `al.lp.Sersic`. `R_sersic` → `effective_radius`, `n_sersic` → `sersic_index`, `e1, e2` → `ell_comps.ell_comps_0, ell_comps_1`. The `amp` parameter in lenstronomy is the surface brightness at `R_sersic`; in autolens it's `intensity`. They scale identically once the units are matched (both per pixel).
- **`center_x, center_y`** → `centre.centre_1, centre.centre_0` (note the index swap — autolens uses (y, x) tuples whereas lenstronomy uses (x, y)).
