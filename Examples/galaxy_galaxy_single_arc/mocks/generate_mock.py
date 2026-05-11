"""generate_mock.py — SLACS-style single-arc galaxy-galaxy lens mock.

Why this exists
---------------
Every existing Example in this repo starts with a compound system, a DSPL,
a multi-plane lens, or a real-data target. A new lens modeler going from
Module 03's toy mocks straight to e.g. `Examples/agel_real_target/` skips
the canonical pedagogical starting point: **one deflector, one source, one
ring/arc**. That's the Bolton+08 / Auger+10 SLACS setup, by far the most
cited geometry in the strong-lensing literature.

This mock recreates the standard SLACS galaxy-galaxy lens:

  - Lens galaxy at z_L = 0.3 (typical SLACS lens redshift)
      * mass:  Isothermal (SIE), θ_E = 1.2"
      * shear: ExternalShear, γ_ext ≈ 0.03
      * light: Sersic, n=4, R_e = 1.0"  (de Vaucouleurs early-type)
  - Source at z_s = 1.0
      * Sersic, n=1, R_e = 0.15"  (typical Sb-disc background galaxy)

The lens light + source light + PSF + Gaussian noise are combined into a
60×60 0.05"/px HST WFC3-IR-like cutout. Self-consistency check at the end
asserts chi²/N ≤ 1.5 at truth — same recipe as
`Examples/mge_to_physical/mocks/regenerate_in_autolens.py`.

Output artifacts (in this directory):
  - image.fits, noise_map.fits, psf.fits  (consumed by al.Imaging.from_fits)
  - truths.json                           (autolens-convention truth dict)
  - tracer.pkl (optional)                 (the true Tracer for overplots)

Run from repo root:
    python Examples/galaxy_galaxy_single_arc/mocks/generate_mock.py
"""

from __future__ import annotations

import json
from pathlib import Path

import autogalaxy as ag
import autolens as al
import autolens.plot as aplt
import numpy as np

OUT = Path(__file__).parent

# --- Geometry & cosmology --------------------------------------------------
shape = (60, 60)
pixel_scale = 0.05
exp_time = 1000.0
bg_rms = 0.01
z_l = 0.3
z_s = 1.0
H0, Om0 = 70.0, 0.30
cosmology = ag.cosmo.FlatLambdaCDM(H0=H0, Om0=Om0)

print(f"[mock] geometry: {shape[0]}×{shape[1]} @ {pixel_scale}″/px, "
      f"exp={exp_time}s, bg_rms={bg_rms}")
print(f"[mock] redshifts: z_L={z_l}, z_s={z_s}")
print(f"[mock] cosmology: FlatLambdaCDM(H0={H0}, Om0={Om0})")

# --- Truth: lens + source --------------------------------------------------
lens = al.Galaxy(
    redshift=z_l,
    bulge=al.lp.Sersic(
        centre=(0.0, 0.0),
        ell_comps=(0.10, 0.05),
        intensity=1.0,
        effective_radius=1.0,
        sersic_index=4.0,
    ),
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0),
        ell_comps=(0.10, 0.05),
        einstein_radius=1.2,
    ),
    shear=al.mp.ExternalShear(gamma_1=0.025, gamma_2=-0.015),
)
source = al.Galaxy(
    redshift=z_s,
    bulge=al.lp.Sersic(
        centre=(0.12, -0.08),
        ell_comps=(0.05, 0.02),
        intensity=0.35,
        effective_radius=0.15,
        sersic_index=1.0,
    ),
)
print(f"[mock] lens: Isothermal θ_E=1.2, ell=(0.10, 0.05) + Sersic n=4 "
      f"R_e=1.0; shear (γ₁,γ₂)=(+0.025, -0.015)")
print(f"[mock] source: Sersic n=1, R_e=0.15, offset=(0.12, -0.08)")

# --- Build tracer + simulate -----------------------------------------------
tracer = al.Tracer(galaxies=[lens, source], cosmology=cosmology)
grid = al.Grid2D.uniform(shape_native=shape, pixel_scales=pixel_scale,
                         over_sample_size=4)

# Gaussian PSF, FWHM = 0.12″ (HST WFC3-IR ish).
fwhm = 0.12
sigma_psf = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
psf_shape = (11, 11)
yy, xx = np.indices(psf_shape) - (psf_shape[0] - 1) / 2.0
psf_arr = np.exp(-(yy**2 + xx**2) * pixel_scale**2 / (2.0 * sigma_psf**2))
psf_arr /= psf_arr.sum()
psf = al.Convolver(
    image=al.Array2D.no_mask(values=np.zeros(shape), pixel_scales=pixel_scale),
    kernel=al.Array2D.no_mask(values=psf_arr, pixel_scales=pixel_scale),
)
# Construct PSF via the high-level API (Convolver) only for the simulator —
# autolens's Convolver.from_fits is how the fit-time path reads it back.
psf_path = OUT / "psf.fits"
al.Array2D.no_mask(values=psf_arr, pixel_scales=pixel_scale).output_to_fits(
    file_path=psf_path, overwrite=True,
) if hasattr(al.Array2D, "output_to_fits") else None

simulator = al.SimulatorImaging(
    exposure_time=exp_time,
    psf=psf,
    background_sky_level=0.0,
    add_poisson_noise_to_data=True,
    noise_seed=1,
    noise_if_add_noise_false=bg_rms,
)
print("[mock] simulating...")
dataset = simulator.via_tracer_from(tracer=tracer, grid=grid)
print(f"[mock] simulated: data shape {dataset.data.shape_native}")

# --- Write artifacts -------------------------------------------------------
img_path = OUT / "image.fits"
noise_path = OUT / "noise_map.fits"
psf_path = OUT / "psf.fits"
aplt.fits_imaging(
    dataset=dataset,
    data_path=img_path,
    psf_path=psf_path,
    noise_map_path=noise_path,
    overwrite=True,
)
print(f"[mock] wrote {img_path.relative_to(OUT.parents[2])}")
print(f"[mock] wrote {noise_path.relative_to(OUT.parents[2])}")
print(f"[mock] wrote {psf_path.relative_to(OUT.parents[2])}")

truths = {
    "framework": "PyAutoLens 2026.4.13.6 (native simulator)",
    "scenario": "SLACS-style galaxy-galaxy single arc",
    "generated_date": "2026-05-11",
    "imaging": {
        "shape_native": list(shape),
        "pixel_scales_arcsec": pixel_scale,
        "exp_time_s": exp_time,
        "background_rms": bg_rms,
        "psf_fwhm_arcsec": fwhm,
    },
    "redshifts": {"lens": z_l, "source": z_s},
    "cosmology": {"H0": H0, "Om": Om0, "w": -1.0},
    "lens": {
        "bulge": {
            "type": "al.lp.Sersic",
            "centre": [0.0, 0.0],
            "ell_comps": [0.10, 0.05],
            "intensity": 1.0,
            "effective_radius": 1.0,
            "sersic_index": 4.0,
        },
        "mass": {
            "type": "al.mp.Isothermal",
            "centre": [0.0, 0.0],
            "ell_comps": [0.10, 0.05],
            "einstein_radius": 1.2,
        },
        "shear": {
            "type": "al.mp.ExternalShear",
            "gamma_1": 0.025,
            "gamma_2": -0.015,
        },
    },
    "source": {
        "bulge": {
            "type": "al.lp.Sersic",
            "centre": [0.12, -0.08],
            "ell_comps": [0.05, 0.02],
            "intensity": 0.35,
            "effective_radius": 0.15,
            "sersic_index": 1.0,
        },
    },
}
truths_path = OUT / "truths.json"
truths_path.write_text(json.dumps(truths, indent=2))
print(f"[mock] wrote {truths_path.relative_to(OUT.parents[2])}")

# --- Self-consistency check (chi²-at-truth) -------------------------------
print("[mock] running chi²-at-truth self-consistency check...")
mask = al.Mask2D.circular(
    shape_native=shape, pixel_scales=pixel_scale, radius=2.5,
)
masked = dataset.apply_mask(mask=mask)
fit = al.FitImaging(dataset=masked, tracer=tracer)
chi2_total = float(fit.chi_squared)
n_unmasked = int(np.sum(~mask))
chi2_per_pix = chi2_total / max(n_unmasked, 1)
max_norm_res = float(np.nanmax(np.abs(fit.normalized_residual_map)))
print(f"[mock] chi²-at-truth: chi²/N = {chi2_per_pix:.3f} (target ≤ 1.5)")
print(f"[mock] chi²-at-truth: max|norm res| = {max_norm_res:.2f}σ "
      f"(target ≤ 5.0σ)")
assert chi2_per_pix <= 1.5, (
    f"chi²/N = {chi2_per_pix:.3f} > 1.5 — mock is NOT self-consistent."
)
assert max_norm_res <= 5.0, (
    f"max|res| = {max_norm_res:.2f}σ > 5.0σ — mock is NOT self-consistent."
)
print("[mock] chi²-at-truth: PASS — mock is self-consistent in autolens.")
