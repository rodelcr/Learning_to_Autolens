"""Generate a disky / two-component lens mock for Examples/disky_spiral_lens/.

Pedagogical goal: show a lens galaxy whose LIGHT has a morphology no
single Sersic can capture — a high-n bulge + low-n disk at a different
position angle. The mass is Isothermal aligned with the bulge PA, so
the mass and light PA's disagree (a common, physically real situation).

The fit notebook demonstrates:
- Single-Sersic lens light leaves visible residuals at the lens.
- An MGE (Multi-Gaussian Expansion) or two-component Sersic+Sersic
  lens light subtracts cleanly.

Writes mock_image.fits, mock_noise.fits, mock_psf.fits, mock_truth.json.
Run locally. Usage: python generate_mock.py
"""
from __future__ import annotations

import json
from pathlib import Path

import autolens as al
import numpy as np

OUT = Path(__file__).parent

pixel_scales = 0.05
shape        = (120, 120)

z_lens   = 0.45
z_source = 1.6

# -- Lens (z=0.45) — bulge + disk at DIFFERENT PAs -----------------------
# A bulge at PA ~ 0° (ell_comps aligned to x-axis) + a disk at PA ~ 35°
# (ell_comps rotated). The mass shares the bulge PA — so mass and TOTAL
# light PA disagree.
lens_bulge = al.lp.Sersic(
    centre=(0.0, 0.0),
    ell_comps=(0.10, 0.0),          # PA = 0°, q ≈ 0.82
    intensity=1.5,
    effective_radius=0.45,
    sersic_index=4.0,               # de Vaucouleurs-like bulge
)
lens_disk = al.lp.Sersic(
    centre=(0.0, 0.0),
    ell_comps=(0.18, 0.20),         # PA ≈ 35°, q ≈ 0.55 (flatter disk)
    intensity=0.8,
    effective_radius=1.0,
    sersic_index=1.0,               # exponential disk
)
lens_mass = al.mp.Isothermal(
    centre=(0.0, 0.0),
    ell_comps=(0.08, 0.0),          # aligned with bulge, slightly rounder
    einstein_radius=1.3,
)
# Modest external shear to reflect tidal environment
lens_shear = al.mp.ExternalShear(gamma_1=0.02, gamma_2=0.03)

lens = al.Galaxy(
    redshift=z_lens,
    bulge=lens_bulge,
    disk=lens_disk,
    mass=lens_mass,
    shear=lens_shear,
)

# -- Source (z=1.6) — compact Sersic off-axis ----------------------------
source = al.Galaxy(
    redshift=z_source,
    bulge=al.lp.SersicCore(
        centre=(0.08, 0.12),
        ell_comps=(-0.10, 0.25),
        intensity=3.0,
        effective_radius=0.08,
        sersic_index=1.3,
    ),
)

tracer = al.Tracer(galaxies=[lens, source])

# -- Simulate -----------------------------------------------------------
grid = al.Grid2D.uniform(shape_native=shape, pixel_scales=pixel_scales)
psf = al.Convolver.from_gaussian(
    shape_native=(7, 7), pixel_scales=pixel_scales, sigma=0.08, normalize=True,
)
simulator = al.SimulatorImaging(
    exposure_time=2000.0,
    psf=psf,
    background_sky_level=0.05,
    add_poisson_noise_to_data=True,
    noise_seed=123,
)
dataset = simulator.via_tracer_from(tracer=tracer, grid=grid)

from astropy.io import fits as _fits
_fits.PrimaryHDU(data=np.asarray(dataset.data.native)).writeto(
    OUT / "mock_image.fits", overwrite=True)
_fits.PrimaryHDU(data=np.asarray(dataset.noise_map.native)).writeto(
    OUT / "mock_noise.fits", overwrite=True)
# psf.kernel is slim (49,); `.native` gives the 2D (7, 7) autolens
# reads back via al.Imaging.from_fits.
_fits.PrimaryHDU(data=np.asarray(psf.kernel.native)).writeto(
    OUT / "mock_psf.fits", overwrite=True)

truth = {
    "pixel_scales": pixel_scales,
    "shape": list(shape),
    "redshifts": {"lens": z_lens, "source": z_source},
    "lens": {
        "bulge": {"centre": [0, 0], "ell_comps": [0.10, 0.0],
                  "intensity": 1.5, "effective_radius": 0.45, "sersic_index": 4.0,
                  "PA_degrees_approx": 0},
        "disk":  {"centre": [0, 0], "ell_comps": [0.18, 0.20],
                  "intensity": 0.8, "effective_radius": 1.0, "sersic_index": 1.0,
                  "PA_degrees_approx": 35},
        "mass":  {"centre": [0, 0], "ell_comps": [0.08, 0.0], "einstein_radius": 1.3,
                  "PA_aligned_with": "bulge"},
        "shear": {"gamma_1": 0.02, "gamma_2": 0.03},
    },
    "source": {
        "bulge": {"centre": [0.08, 0.12], "ell_comps": [-0.10, 0.25],
                  "intensity": 3.0, "effective_radius": 0.08, "sersic_index": 1.3},
    },
    "pedagogical_note": (
        "Lens has a bulge at PA≈0° and a disk at PA≈35°. A single-Sersic "
        "lens-light fit will leave coherent residuals at the lens centre "
        "because it cannot rotate two components independently. A two-"
        "component (Sersic+Sersic) or MGE lens-light fit will subtract "
        "cleanly. The mass model is Isothermal aligned with the bulge — "
        "the mass/total-light PA misalignment is a realistic feature that "
        "must not be 'fixed' by tying mass PA to the total light."
    ),
    "simulator": {"exposure_time": 2000, "psf_sigma": 0.08,
                  "background_sky_level": 0.05, "noise_seed": 123},
    "mask_suggestion": {"type": "circular", "radius_arcsec": 2.8},
}
(OUT / "mock_truth.json").write_text(json.dumps(truth, indent=2))

print(f"Wrote disky_spiral mock to {OUT}")
print(f"  image sum:   {float(dataset.data.sum()):.2f}")
print(f"  peak S/N:    {float(dataset.signal_to_noise_map.max()):.1f}")
