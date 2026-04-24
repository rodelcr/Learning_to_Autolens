"""Generate a double-source-plane mock for Examples/double_source_plane/.

Runs locally (no Cannon needed). Writes:
    mock_image.fits, mock_noise.fits, mock_psf.fits, mock_truth.json

Geometry: one lens at z=0.5 deflecting TWO sources at z=1.0 and z=2.5.
The source-plane angular scales differ by the cosmological distance
ratio β = D_ds1/D_s1 × D_s2/D_ds2 ≈ 0.59 for FlatLambdaCDM(70, 0.3),
which is the cosmography-sensitive quantity a DSPL fit can constrain.

Sources are placed so each forms a distinct arc system on the sky —
a compact inner Einstein ring for the lower-redshift source, a larger
outer ring for the higher-redshift one.

Usage:
    python generate_mock.py
"""
from __future__ import annotations

import json
from pathlib import Path

import autolens as al
import numpy as np

OUT = Path(__file__).parent

# -- Geometry ---------------------------------------------------------
pixel_scales = 0.05  # HST-like
shape        = (120, 120)

z_lens = 0.5
z_src1 = 1.0
z_src2 = 2.5

# -- Lens (one galaxy at z=0.5) --------------------------------------
lens = al.Galaxy(
    redshift=z_lens,
    bulge=al.lp.Sersic(
        centre=(0.0, 0.0),
        ell_comps=(0.05, -0.05),
        intensity=1.2,
        effective_radius=0.8,
        sersic_index=3.5,
    ),
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0),
        ell_comps=(0.08, -0.03),
        einstein_radius=1.4,
    ),
    shear=al.mp.ExternalShear(gamma_1=0.03, gamma_2=-0.02),
)

# -- Source 1 (z=1.0, smaller Einstein radius due to lower redshift) --
# Offset north of the optical axis → arc forms on the south side.
source_1 = al.Galaxy(
    redshift=z_src1,
    bulge=al.lp.SersicCore(
        centre=(0.15, 0.0),           # +y offset
        ell_comps=(-0.15, 0.05),
        intensity=3.0,                 # bright enough to be visible above lens light
        effective_radius=0.07,
        sersic_index=1.3,
    ),
)

# -- Source 2 (z=2.5, larger Einstein radius → outer arc system) ------
# Offset in a DIFFERENT direction so the arcs don't overlap.
source_2 = al.Galaxy(
    redshift=z_src2,
    bulge=al.lp.SersicCore(
        centre=(-0.1, 0.22),           # offset in a different direction
        ell_comps=(0.1, -0.08),
        intensity=3.5,
        effective_radius=0.06,
        sersic_index=1.5,
    ),
)

tracer = al.Tracer(galaxies=[lens, source_1, source_2])

# -- Simulate ---------------------------------------------------------
grid = al.Grid2D.uniform(shape_native=shape, pixel_scales=pixel_scales)

psf = al.Convolver.from_gaussian(
    shape_native=(7, 7),
    pixel_scales=pixel_scales,
    sigma=0.08,  # ~HST-like Gaussian
    normalize=True,
)

simulator = al.SimulatorImaging(
    exposure_time=2000.0,
    psf=psf,
    background_sky_level=0.05,
    add_poisson_noise_to_data=True,
    noise_seed=42,
)

dataset = simulator.via_tracer_from(tracer=tracer, grid=grid)

from astropy.io import fits as _fits
_fits.PrimaryHDU(data=np.asarray(dataset.data.native)).writeto(
    OUT / "mock_image.fits", overwrite=True)
_fits.PrimaryHDU(data=np.asarray(dataset.noise_map.native)).writeto(
    OUT / "mock_noise.fits", overwrite=True)
# Convolver's underlying kernel array — `psf.kernel` is slim (49,);
# use its `.native` for the 2D (7, 7) representation autolens expects
# when reading back via `al.Imaging.from_fits`.
_fits.PrimaryHDU(data=np.asarray(psf.kernel.native)).writeto(
    OUT / "mock_psf.fits", overwrite=True)

truth = {
    "pixel_scales": pixel_scales,
    "shape": list(shape),
    "redshifts": {"lens": z_lens, "source_1": z_src1, "source_2": z_src2},
    "lens": {
        "bulge": {"centre": [0.0, 0.0], "ell_comps": [0.05, -0.05],
                  "intensity": 1.2, "effective_radius": 0.8, "sersic_index": 3.5},
        "mass": {"centre": [0.0, 0.0], "ell_comps": [0.08, -0.03],
                 "einstein_radius": 1.4},
        "shear": {"gamma_1": 0.03, "gamma_2": -0.02},
    },
    "source_1": {
        "bulge": {"centre": [-0.1, 0.05], "ell_comps": [-0.2, 0.1],
                  "intensity": 0.6, "effective_radius": 0.08, "sersic_index": 1.3},
    },
    "source_2": {
        "bulge": {"centre": [0.18, -0.12], "ell_comps": [0.15, -0.1],
                  "intensity": 0.5, "effective_radius": 0.07, "sersic_index": 1.5},
    },
    "cosmology": "FlatLambdaCDM(H0=70, Om0=0.3)  [autolens default]",
    "simulator": {"exposure_time": 2000, "psf_sigma": 0.08,
                  "background_sky_level": 0.05, "noise_seed": 42},
    "mask_suggestion": {"type": "circular", "radius_arcsec": 2.8},
}
(OUT / "mock_truth.json").write_text(json.dumps(truth, indent=2))

print(f"Wrote mock FITS + truth.json to {OUT}")
print(f"  image shape: {dataset.data.shape_native}")
print(f"  image sum:   {float(dataset.data.sum()):.2f}")
print(f"  peak S/N:    {float(dataset.signal_to_noise_map.max()):.1f}")
