"""Generate the DSPL mock natively in autolens.

Runs locally (no Cannon needed). Writes:
    mock_image.fits, mock_noise.fits, mock_psf.fits, mock_truth.json

Geometry: one lens at z=0.5 deflecting TWO sources at z=1.0 and z=2.5.
The source-plane angular scales differ by the cosmological distance
ratio beta = D_ds1/D_s1 * D_s2/D_ds2 ~ 0.59 for FlatLambdaCDM(70, 0.3),
which is the cosmography-sensitive quantity a DSPL fit can constrain.

Sources are placed so each forms a distinct arc system on the sky.

Self-consistency check at the end: chi^2-at-truth <= 1.5.

v0.96 rewrite (2026-05-15): the previous version of this generator had
a STALE TRUTH JSON — the simulator used one set of source params and
the JSON dump used different (older) values. The Stage 1 staged-chain
fit's truth-anchored priors were therefore anchored on wrong positions,
producing the 96h TIMEOUT documented in CLUSTER_WORKFLOW_NOTES.
This rewrite uses ONE python dict as the source of truth — same dict
seeds the al.Galaxy constructors AND gets dumped to JSON. Also adds
the chi^2-at-truth self-consistency assertion at the end (same pattern
as radial_arc_smbh, ggsa, and the regenerated mge mock).

Usage:
    python Examples/double_source_plane/mocks/generate_mock.py
"""

from __future__ import annotations

import json
from pathlib import Path

import autogalaxy as ag
import autolens as al
import autolens.plot as aplt
import numpy as np

OUT = Path(__file__).parent

# Single source-of-truth dict (used for BOTH simulation AND JSON dump)
TRUTH = {
    "framework": "PyAutoLens 2026.4.13.6 (native simulator)",
    "generated_date": "2026-05-15",
    "imaging": {
        "pixel_scales_arcsec": 0.05,
        "shape_native": [120, 120],
        "exp_time_s": 2000.0,
        "background_sky_level": 0.05,
        "noise_seed": 42,
        "psf_sigma_arcsec": 0.08,
    },
    "redshifts": {"lens": 0.5, "source_1": 1.0, "source_2": 2.5},
    "cosmology": {"H0": 70.0, "Om": 0.30, "w": -1.0},
    "lens": {
        "bulge": {
            "type": "al.lp.Sersic",
            "centre": [0.0, 0.0],
            "ell_comps": [0.05, -0.05],
            "intensity": 1.2,
            "effective_radius": 0.8,
            "sersic_index": 3.5,
        },
        "mass": {
            "type": "al.mp.Isothermal",
            "centre": [0.0, 0.0],
            "ell_comps": [0.08, -0.03],
            "einstein_radius": 1.4,
        },
        "shear": {
            "type": "al.mp.ExternalShear",
            "gamma_1": 0.03,
            "gamma_2": -0.02,
        },
    },
    "source_1": {
        "bulge": {
            "type": "al.lp.SersicCore",
            "centre": [0.15, 0.0],
            "ell_comps": [-0.15, 0.05],
            "intensity": 3.0,
            "effective_radius": 0.07,
            "sersic_index": 1.3,
        },
    },
    "source_2": {
        "bulge": {
            "type": "al.lp.SersicCore",
            "centre": [-0.1, 0.22],
            "ell_comps": [0.1, -0.08],
            "intensity": 3.5,
            "effective_radius": 0.06,
            "sersic_index": 1.5,
        },
    },
    "mask_suggestion": {"type": "circular", "radius_arcsec": 2.8},
}

print(f"[mock] geometry: {TRUTH['imaging']['shape_native']} @ "
      f"{TRUTH['imaging']['pixel_scales_arcsec']}\"/px")
print(f"[mock] redshifts: lens={TRUTH['redshifts']['lens']}, "
      f"src_1={TRUTH['redshifts']['source_1']}, "
      f"src_2={TRUTH['redshifts']['source_2']}")
print(f"[mock] cosmology: FlatLambdaCDM(H0={TRUTH['cosmology']['H0']}, "
      f"Om0={TRUTH['cosmology']['Om']})")


def _galaxy_from_truth(t, z):
    """Build an al.Galaxy from a slice of TRUTH (uses keys present)."""
    kwargs = {"redshift": z}
    if "bulge" in t:
        b = t["bulge"]
        cls = getattr(al.lp, b["type"].split(".")[-1])
        kwargs["bulge"] = cls(
            centre=tuple(b["centre"]),
            ell_comps=tuple(b["ell_comps"]),
            intensity=b["intensity"],
            effective_radius=b["effective_radius"],
            sersic_index=b["sersic_index"],
        )
    if "mass" in t:
        m = t["mass"]
        kwargs["mass"] = al.mp.Isothermal(
            centre=tuple(m["centre"]),
            ell_comps=tuple(m["ell_comps"]),
            einstein_radius=m["einstein_radius"],
        )
    if "shear" in t:
        s = t["shear"]
        kwargs["shear"] = al.mp.ExternalShear(
            gamma_1=s["gamma_1"], gamma_2=s["gamma_2"],
        )
    return al.Galaxy(**kwargs)


cosmology = ag.cosmo.FlatLambdaCDM(
    H0=TRUTH["cosmology"]["H0"], Om0=TRUTH["cosmology"]["Om"],
)
lens = _galaxy_from_truth(TRUTH["lens"], z=TRUTH["redshifts"]["lens"])
source_1 = _galaxy_from_truth(TRUTH["source_1"],
                              z=TRUTH["redshifts"]["source_1"])
source_2 = _galaxy_from_truth(TRUTH["source_2"],
                              z=TRUTH["redshifts"]["source_2"])
tracer = al.Tracer(galaxies=[lens, source_1, source_2], cosmology=cosmology)
print("[mock] tracer built — multi-plane (z_l=0.5, z_s1=1.0, z_s2=2.5)")

# Simulator
shape = tuple(TRUTH["imaging"]["shape_native"])
pixel_scales = TRUTH["imaging"]["pixel_scales_arcsec"]
grid = al.Grid2D.uniform(shape_native=shape, pixel_scales=pixel_scales,
                         over_sample_size=4)
psf = al.Convolver.from_gaussian(
    shape_native=(7, 7),
    pixel_scales=pixel_scales,
    sigma=TRUTH["imaging"]["psf_sigma_arcsec"],
    normalize=True,
)
simulator = al.SimulatorImaging(
    exposure_time=TRUTH["imaging"]["exp_time_s"],
    psf=psf,
    background_sky_level=TRUTH["imaging"]["background_sky_level"],
    add_poisson_noise_to_data=True,
    noise_seed=TRUTH["imaging"]["noise_seed"],
)
print("[mock] simulating...")
dataset = simulator.via_tracer_from(tracer=tracer, grid=grid)
print(f"[mock] simulated: data shape {dataset.data.shape_native}")

# Write artifacts
img_path = OUT / "mock_image.fits"
noise_path = OUT / "mock_noise.fits"
psf_path = OUT / "mock_psf.fits"
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

truths_path = OUT / "mock_truth.json"
truths_path.write_text(json.dumps(TRUTH, indent=2))
print(f"[mock] wrote {truths_path.relative_to(OUT.parents[2])}")
print(f"[mock] image sum: {float(dataset.data.sum()):.2f}")
print(f"[mock] peak S/N : {float(dataset.signal_to_noise_map.max()):.1f}")

# Self-consistency check
print("[mock] running chi^2-at-truth self-consistency check...")
mask = al.Mask2D.circular(
    shape_native=shape, pixel_scales=pixel_scales,
    radius=TRUTH["mask_suggestion"]["radius_arcsec"],
)
masked = dataset.apply_mask(mask=mask)
fit = al.FitImaging(dataset=masked, tracer=tracer)
n_unmasked = int(np.sum(~mask))
chi2_per_pix = float(fit.chi_squared) / max(n_unmasked, 1)
max_norm_res = float(np.nanmax(np.abs(fit.normalized_residual_map)))
print(f"[mock] chi^2-at-truth: chi^2/N = {chi2_per_pix:.3f} (target <= 1.5)")
print(f"[mock] chi^2-at-truth: max|norm res| = {max_norm_res:.2f} sigma "
      f"(target <= 5.0 sigma)")
assert chi2_per_pix <= 1.5, (
    f"chi^2/N = {chi2_per_pix:.3f} > 1.5 — mock is NOT self-consistent."
)
assert max_norm_res <= 5.0, (
    f"max|res| = {max_norm_res:.2f} sigma > 5.0 sigma — mock is NOT self-consistent."
)
print("[mock] chi^2-at-truth: PASS — mock is self-consistent in autolens.")
