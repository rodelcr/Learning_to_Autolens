"""
generate_mock.py — Synthetic cluster-scale lens with multi-source-plane geometry.

Produces three FITS files (image, noise, psf) plus a `mock_truth.json` with
all parameters needed to fit the system in `01_cluster_scale_fit.ipynb`.

Architecture (loosely modeled on the SLACS / Bergamini+23 cluster-lensing
methodology, scaled down for tractable demonstration):
  - 1 BCG (brightest cluster galaxy) at (0, 0), z=0.4
    - SersicSph light, IsothermalSph mass, theta_E ~ 4.0"
  - 10 cluster member galaxies at radii 2-8" from BCG, all z=0.4
    - SersicSph light + IsothermalSph mass, theta_E scaled via Faber-Jackson
      from per-member luminosity
  - 2 sources at z=1.5 and z=2.8
    - SersicCore light profiles
    - The two sources at different redshifts produce two Einstein rings,
      enabling beta_12-cosmography in the fit

Mocks an HST-like cutout: 0.05" pixel scale, 200 x 200 image (10" across).

Usage:
    cd Examples/cluster_scale/mocks
    python generate_mock.py

References:
  - autolens_workspace_latest/scripts/group/simulator.py — group-scale template
  - autolens_workspace_latest/scripts/imaging/features/scaling_relation/modeling.py
    — Faber-Jackson scaling-relation API
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

# Headless matplotlib for any plotting in non-interactive runs.
import matplotlib
matplotlib.use("Agg")

import autolens as al
import autolens.plot as aplt


HERE = Path(__file__).parent.resolve()


def faber_jackson_theta_E(luminosity: float, luminosity_star: float = 1.0,
                          theta_E_star: float = 0.5) -> float:
    """Faber-Jackson scaling: sigma_v ~ L^(1/4) and theta_E ~ sigma_v^2,
    so theta_E ~ L^(1/2)."""
    return theta_E_star * (luminosity / luminosity_star) ** 0.5


def main():
    # -------- Geometry + simulator -----------------------------------------
    pixel_scales = 0.05
    shape_native = (200, 200)
    grid = al.Grid2D.uniform(shape_native=shape_native, pixel_scales=pixel_scales)

    # Adaptive over-sampling at every galaxy centre for accurate Sersic light.
    bcg_centre = (0.0, 0.0)
    members_centres = [
        (+2.5, +1.5), (-1.8, +2.4), (+0.5, -3.0), (-3.0, -1.0), (+3.5, -2.0),
        (-2.5, +3.5), (+4.5, +0.5), (-0.5, -4.5), (+1.0, +4.0), (-4.0, +0.0),
    ]
    members_luminosities = [
        0.7, 0.5, 0.4, 0.6, 0.3, 0.55, 0.45, 0.35, 0.5, 0.4
    ]
    over_sample = al.util.over_sample.over_sample_size_via_radial_bins_from(
        grid=grid,
        sub_size_list=[32, 8, 2],
        radial_list=[0.3, 0.6],
        centre_list=[bcg_centre] + members_centres,
    )
    grid = grid.apply_over_sampling(over_sample_size=over_sample)

    psf = al.Convolver.from_gaussian(
        shape_native=(11, 11), sigma=0.05, pixel_scales=pixel_scales
    )

    simulator = al.SimulatorImaging(
        exposure_time=2000.0,
        psf=psf,
        background_sky_level=0.05,
        add_poisson_noise_to_data=True,
    )

    # -------- BCG ---------------------------------------------------------
    z_l = 0.4
    z_s1 = 1.5
    z_s2 = 2.8
    bcg_theta_E = 4.0
    bcg = al.Galaxy(
        redshift=z_l,
        bulge=al.lp.SersicSph(
            centre=bcg_centre, intensity=1.5, effective_radius=2.0, sersic_index=4.0,
        ),
        mass=al.mp.IsothermalSph(centre=bcg_centre, einstein_radius=bcg_theta_E),
    )

    # -------- Cluster members with FJ-scaled theta_E -----------------------
    members = []
    members_truth = []
    for c, L in zip(members_centres, members_luminosities):
        theta_E = faber_jackson_theta_E(L, luminosity_star=1.0, theta_E_star=0.6)
        intensity = 0.4 * L  # purely cosmetic for now
        R_eff = 0.4 * (L ** 0.5)
        g = al.Galaxy(
            redshift=z_l,
            bulge=al.lp.SersicSph(
                centre=c, intensity=intensity, effective_radius=R_eff, sersic_index=3.0,
            ),
            mass=al.mp.IsothermalSph(centre=c, einstein_radius=theta_E),
        )
        members.append(g)
        members_truth.append({
            "centre": c,
            "luminosity": L,
            "einstein_radius": theta_E,
            "intensity": intensity,
            "effective_radius": R_eff,
            "sersic_index": 3.0,
            "mass": "IsothermalSph",
        })

    # -------- Two sources at different z -----------------------------------
    source_1 = al.Galaxy(
        redshift=z_s1,
        bulge=al.lp.SersicCore(
            centre=(+0.5, +0.2),
            ell_comps=al.convert.ell_comps_from(axis_ratio=0.7, angle=45.0),
            intensity=4.0,
            effective_radius=0.10,
            sersic_index=1.5,
        ),
    )
    source_2 = al.Galaxy(
        redshift=z_s2,
        bulge=al.lp.SersicCore(
            centre=(-0.3, +0.4),
            ell_comps=al.convert.ell_comps_from(axis_ratio=0.85, angle=120.0),
            intensity=3.0,
            effective_radius=0.08,
            sersic_index=1.3,
        ),
    )

    # -------- Tracer + simulate --------------------------------------------
    tracer = al.Tracer(galaxies=[bcg] + members + [source_1, source_2])
    print(f"Tracer planes: {sorted(set(g.redshift for g in tracer.galaxies))}")

    dataset = simulator.via_tracer_from(tracer=tracer, grid=grid)

    # -------- Write artifacts ---------------------------------------------
    image_path = HERE / "mock_image.fits"
    noise_path = HERE / "mock_noise.fits"
    psf_path = HERE / "mock_psf.fits"
    aplt.fits_imaging(
        dataset=dataset,
        data_path=image_path,
        noise_map_path=noise_path,
        psf_path=psf_path,
        overwrite=True,
    )

    truth = {
        "pixel_scales": pixel_scales,
        "shape": list(shape_native),
        "redshifts": {"lens": z_l, "source_1": z_s1, "source_2": z_s2},
        "bcg": {
            "centre": list(bcg_centre),
            "einstein_radius": bcg_theta_E,
            "intensity": 1.5,
            "effective_radius": 2.0,
            "sersic_index": 4.0,
            "mass": "IsothermalSph",
        },
        "members": members_truth,
        "fj_relation": {
            "theta_E_star": 0.6,
            "luminosity_star": 1.0,
            "exponent": 0.5,
            "note": "theta_E_i = theta_E_star * (L_i / L_star)^0.5",
        },
        "source_1": {
            "centre": [0.5, 0.2],
            "axis_ratio": 0.7, "angle_deg": 45.0,
            "intensity": 4.0, "effective_radius": 0.10, "sersic_index": 1.5,
        },
        "source_2": {
            "centre": [-0.3, 0.4],
            "axis_ratio": 0.85, "angle_deg": 120.0,
            "intensity": 3.0, "effective_radius": 0.08, "sersic_index": 1.3,
        },
        "simulator": {
            "exposure_time_s": 2000.0, "psf_sigma_arcsec": 0.05,
            "background_sky": 0.05,
        },
    }
    truth_path = HERE / "mock_truth.json"
    truth_path.write_text(json.dumps(truth, indent=2))
    print(f"Wrote {image_path.name}, {noise_path.name}, {psf_path.name}, {truth_path.name}")

    # ---- Sanity check: chi^2 at the simulator's tracer must be near 1 -----
    # Catches generator-vs-truth inconsistency BEFORE shipping the mock.
    # 2026-05-04: cluster_truth_v2 Cannon TIMEOUT was traced to a fit
    # driver that omitted member light — assertion below would NOT have
    # caught that (the assertion runs on the simulator's own tracer, which
    # has the light); but it WOULD catch a coordinate or PSF bug.
    mask = al.Mask2D.circular(
        shape_native=dataset.shape_native, pixel_scales=pixel_scales,
        radius=4.5,
    )
    ds_masked = dataset.apply_mask(mask=mask)
    fit = al.FitImaging(dataset=ds_masked, tracer=tracer)
    chi2_per_pixel = fit.chi_squared / ds_masked.mask.pixels_in_mask
    max_resid = float(np.max(np.abs(fit.normalized_residual_map.native)))
    print(f"Generator self-consistency check:")
    print(f"  chi^2/pixel = {chi2_per_pixel:.4f}  (expect ~1.0)")
    print(f"  max|resid|  = {max_resid:.2f} sigma  (expect <5.0)")
    if chi2_per_pixel > 1.5 or max_resid > 6.0:
        raise RuntimeError(
            f"Generator self-consistency FAILED: chi^2/N={chi2_per_pixel:.2f} "
            f"max|res|={max_resid:.1f}σ — the simulator's tracer doesn't match "
            f"its own output. Investigate before shipping this mock."
        )
    print(f"  ✓ Self-consistency OK.")


if __name__ == "__main__":
    main()
