"""Generate a group-scale lens mock for Examples/group_scale/.

Pedagogical goal: show a lens *group* — one brightest-group-galaxy (BGG)
plus a few satellites, all at the same redshift, deflecting a single
background source. The total deflection is the SUM over all galaxies
(unlike compound_lens, where galaxies sit at different redshifts and the
ray-tracing is recursive plane-to-plane).

Teaches:
- `al.Tracer` handles multiple galaxies at one redshift just fine —
  you pass a list with all lens galaxies + the source, all lenses share
  the same redshift plane.
- The fit is an entry point to the `extra_galaxies` pattern from
  `autolens_workspace_latest/scripts/imaging/features/extra_galaxies/`:
  for survey-scale work you typically fix satellite centres to their
  photometric positions and fit only their masses.
- How much does each satellite actually contribute? Bayes-factor
  comparing "BGG + shear" vs "BGG + 3 satellites" tells you whether
  the satellites are resolvable or absorbable-into-shear.

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
shape        = (160, 160)   # wider field to include satellites

z_lens   = 0.40
z_source = 1.8

# -- BGG (brightest group galaxy) at centre -----------------------------
bgg = al.Galaxy(
    redshift=z_lens,
    bulge=al.lp.Sersic(
        centre=(0.0, 0.0),
        ell_comps=(0.08, -0.02),
        intensity=1.5,
        effective_radius=0.9,
        sersic_index=4.0,        # de Vaucouleurs-like
    ),
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0),
        ell_comps=(0.08, -0.02),
        einstein_radius=1.5,     # dominant deflection
    ),
)

# -- Three satellites at fixed positions, contributing modest mass ------
# Each SIS (spherical — fewer params), smaller θ_E, smaller Sersic light.
# Positions chosen to be visible on the cutout and to perturb the ring.
satellite_1 = al.Galaxy(
    redshift=z_lens,
    bulge=al.lp.Sersic(
        centre=(1.8, 0.7),
        ell_comps=(0.05, 0.0),
        intensity=0.4,
        effective_radius=0.25,
        sersic_index=2.0,
    ),
    mass=al.mp.IsothermalSph(
        centre=(1.8, 0.7),
        einstein_radius=0.35,
    ),
)

satellite_2 = al.Galaxy(
    redshift=z_lens,
    bulge=al.lp.Sersic(
        centre=(-1.5, -1.2),
        ell_comps=(-0.03, 0.02),
        intensity=0.5,
        effective_radius=0.3,
        sersic_index=1.8,
    ),
    mass=al.mp.IsothermalSph(
        centre=(-1.5, -1.2),
        einstein_radius=0.45,
    ),
)

satellite_3 = al.Galaxy(
    redshift=z_lens,
    bulge=al.lp.Sersic(
        centre=(0.5, -2.0),
        ell_comps=(0.0, -0.05),
        intensity=0.3,
        effective_radius=0.2,
        sersic_index=2.5,
    ),
    mass=al.mp.IsothermalSph(
        centre=(0.5, -2.0),
        einstein_radius=0.25,
    ),
)

# -- Source (single, off-axis) -----------------------------------------
source = al.Galaxy(
    redshift=z_source,
    bulge=al.lp.SersicCore(
        centre=(0.12, 0.08),
        ell_comps=(-0.08, 0.15),
        intensity=3.5,
        effective_radius=0.08,
        sersic_index=1.4,
    ),
)

tracer = al.Tracer(galaxies=[bgg, satellite_1, satellite_2, satellite_3, source])

# -- Simulate ----------------------------------------------------------
grid = al.Grid2D.uniform(shape_native=shape, pixel_scales=pixel_scales)
psf = al.Convolver.from_gaussian(
    shape_native=(7, 7), pixel_scales=pixel_scales, sigma=0.08, normalize=True,
)
simulator = al.SimulatorImaging(
    exposure_time=2000.0,
    psf=psf,
    background_sky_level=0.05,
    add_poisson_noise_to_data=True,
    noise_seed=7,
)
dataset = simulator.via_tracer_from(tracer=tracer, grid=grid)

from astropy.io import fits as _fits
_fits.PrimaryHDU(data=np.asarray(dataset.data.native)).writeto(
    OUT / "mock_image.fits", overwrite=True)
_fits.PrimaryHDU(data=np.asarray(dataset.noise_map.native)).writeto(
    OUT / "mock_noise.fits", overwrite=True)
_fits.PrimaryHDU(data=np.asarray(psf.kernel.native)).writeto(
    OUT / "mock_psf.fits", overwrite=True)

truth = {
    "pixel_scales": pixel_scales,
    "shape": list(shape),
    "redshifts": {"lens": z_lens, "source": z_source},
    "bgg": {
        "bulge": {"centre": [0, 0], "ell_comps": [0.08, -0.02],
                  "intensity": 1.5, "effective_radius": 0.9, "sersic_index": 4.0},
        "mass":  {"centre": [0, 0], "ell_comps": [0.08, -0.02], "einstein_radius": 1.5},
    },
    "satellites": [
        {"centre": [ 1.8,  0.7], "einstein_radius": 0.35, "intensity": 0.4,
         "effective_radius": 0.25, "sersic_index": 2.0, "mass": "SIS"},
        {"centre": [-1.5, -1.2], "einstein_radius": 0.45, "intensity": 0.5,
         "effective_radius": 0.30, "sersic_index": 1.8, "mass": "SIS"},
        {"centre": [ 0.5, -2.0], "einstein_radius": 0.25, "intensity": 0.3,
         "effective_radius": 0.20, "sersic_index": 2.5, "mass": "SIS"},
    ],
    "source": {
        "bulge": {"centre": [0.12, 0.08], "ell_comps": [-0.08, 0.15],
                  "intensity": 3.5, "effective_radius": 0.08, "sersic_index": 1.4},
    },
    "pedagogical_note": (
        "Four galaxies at z=0.4 deflecting a source at z=1.8. The BGG "
        "(θ_E=1.5″) dominates; three satellites at offset positions "
        "(θ_E=0.25-0.45″) perturb the image positions and shape of the "
        "ring. All lenses share ONE redshift plane — Tracer sums "
        "their deflection fields. The fit can (a) absorb the satellite "
        "mass into shear (simpler), or (b) model each satellite's mass "
        "explicitly with centres fixed at photometric positions (the "
        "survey-scale pattern). Compare log_Z."
    ),
    "simulator": {"exposure_time": 2000, "psf_sigma": 0.08,
                  "background_sky_level": 0.05, "noise_seed": 7},
    "mask_suggestion": {"type": "circular", "radius_arcsec": 3.5},
}
(OUT / "mock_truth.json").write_text(json.dumps(truth, indent=2))

print(f"Wrote group_scale mock to {OUT}")
print(f"  image sum:   {float(dataset.data.sum()):.2f}")
print(f"  peak S/N:    {float(dataset.signal_to_noise_map.max()):.1f}")
