"""
generate_mock.py — Joint AnalysisPoint + AnalysisImaging mock for the
TDCOSMO methodology (Birrer+20, Wong+20). Companion to
`Examples/quad_time_delay/mocks/generate_mock.py`.

Geometry (identical lens + quasar to the point-only mock; new addition is the
extended host galaxy):

  Lens (z_L = 0.5):
    PowerLaw          centre=(0,0), ell_comps=(0.10, 0.15),
                      einstein_radius=1.2", slope=2.0
    ExternalShear     gamma_1=0.04, gamma_2=0.02

  Source (z_S = 2.0):
    ps.Point          centre=(0.05, 0.07)        — quasar (point-source)
    lp.SersicCore     centre=(0.05, 0.07),
                      effective_radius=0.15", sersic_index=2.0,
                      intensity=0.5             — quasar host galaxy

  Cosmology:           FlatLambdaCDM(H0=70, Om0=0.30)
  Seed:                42 (identical to mocks/generate_mock.py)

Outputs (in `Examples/quad_time_delay/mocks_with_host/`):
  point_dataset.json   — quasar positions + fluxes + time delays (same as mocks/)
  image.fits           — 200x200 HST F814W-like host-galaxy imaging
  noise_map.fits       — Poisson + background sigma map
  psf.fits             — 0.10" FWHM Gaussian PSF
  tracer.json          — true tracer for reference / overplotting
  truths.json          — input parameters for verification

The joint fit consumes BOTH datasets:
  - AnalysisPoint(point_dataset)   for the quasar positions + delays + fluxes
  - AnalysisImaging(imaging)       for the extended-arc host galaxy

This is exactly the TDCOSMO IV (Birrer+20) / H0LiCOW XIII (Wong+20)
methodology, modulo stellar kinematics (covered in Module 13). See
Examples/quad_time_delay/README.md §"Joint fit" for the architecture and
Module 12 §5 for the cosmographic motivation.

Self-consistency assertion (mirrors cluster_scale/mocks/generate_mock.py):
  - chi^2/N at truth on the imaging FitImaging must be <= 1.5
  - max|normalized_residual| at truth must be <= 6 sigma

Run with the autolens conda env active:

    cd Examples/quad_time_delay/mocks_with_host
    python generate_mock.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")

import autolens as al
import autolens.plot as aplt


HERE = Path(__file__).parent.resolve()
RNG = np.random.default_rng(seed=42)


def main() -> None:
    cosmology = al.cosmo.FlatLambdaCDM(H0=70.0, Om0=0.30)

    z_lens, z_source = 0.5, 2.0

    # --- Mass + shear (lens plane) ----------------------------------------
    mass = al.mp.PowerLaw(
        centre=(0.0, 0.0),
        ell_comps=(0.10, 0.15),
        einstein_radius=1.2,
        slope=2.0,
    )
    shear = al.mp.ExternalShear(gamma_1=0.04, gamma_2=0.02)
    lens_galaxy = al.Galaxy(redshift=z_lens, mass=mass, shear=shear)

    # --- Source: quasar (point) + host (extended Sersic) ------------------
    source_centre = (0.05, 0.07)
    source_flux = 1.0  # intrinsic quasar flux (arbitrary units)

    # The host is a galaxy at the same z_S = 2.0 as the quasar (the quasar
    # lives inside it). It shares the centre but has its own light profile.
    host_bulge = al.lp.SersicCore(
        centre=source_centre,
        ell_comps=(0.0, 0.0),       # round host (kept simple for the mock)
        effective_radius=0.15,
        sersic_index=2.0,
        intensity=0.5,
    )

    # The Galaxy carries BOTH a `bulge` (for the imaging arc) and a `point_0`
    # (for the quasar positions/delays). The joint fit shares this single
    # Galaxy across both AnalysisPoint and AnalysisImaging.
    source_galaxy = al.Galaxy(
        redshift=z_source,
        bulge=host_bulge,
        point_0=al.ps.Point(centre=source_centre),
    )

    tracer = al.Tracer(
        galaxies=[lens_galaxy, source_galaxy],
        cosmology=cosmology,
    )

    # =====================================================================
    # PART 1: Point dataset (positions + fluxes + time delays)
    # =====================================================================
    # Use the SAME 200x200 / 0.05" grid + PointSolver settings as the
    # point-only mock, so the recovered image positions are bit-identical.
    grid_solver = al.Grid2D.uniform(shape_native=(200, 200), pixel_scales=0.05)
    solver = al.PointSolver.for_grid(
        grid=grid_solver,
        pixel_scale_precision=0.001,
        magnification_threshold=0.1,
    )
    positions_truth = solver.solve(
        tracer=tracer, source_plane_coordinate=source_centre
    )
    n_images = len(positions_truth)
    print(f"Solver found {n_images} image positions:")
    for i, p in enumerate(np.asarray(positions_truth)):
        print(f"  image {i}: ({p[0]:+.4f}, {p[1]:+.4f}) arcsec")

    if n_images != 4:
        raise SystemExit(
            f"Expected 4-image quad but solver returned {n_images}. "
            "Adjust source_centre or einstein_radius."
        )

    # Time delays at the truth tracer (cosmology = FlatLambdaCDM(70, 0.30))
    time_delays_truth = tracer.time_delays_from(grid=positions_truth)
    print(f"\nTime delays (truth, days, image_0 reference):")
    for i, td in enumerate(np.asarray(time_delays_truth)):
        print(f"  image {i}: {td:+8.3f}")

    # Magnifications -> fluxes (intrinsic source_flux=1)
    lens_calc = al.LensCalc.from_tracer(tracer=tracer)
    magnifications = lens_calc.magnification_2d_via_hessian_from(grid=positions_truth)
    fluxes_truth = source_flux * np.abs(np.asarray(magnifications))
    print(f"\nMagnifications + fluxes (truth):")
    for i, (mu, f) in enumerate(zip(magnifications, fluxes_truth)):
        print(f"  image {i}: mu = {mu:+8.3f}, flux = {f:8.4f}")

    # Same noise model as the point-only mock.
    position_noise = 0.005
    flux_noise_frac = 0.05
    time_delay_noise = 0.5

    positions_obs = np.asarray(positions_truth) + RNG.normal(
        scale=position_noise, size=(n_images, 2)
    )
    fluxes_obs = fluxes_truth * (
        1.0 + RNG.normal(scale=flux_noise_frac, size=n_images)
    )
    delays_obs = np.asarray(time_delays_truth) + RNG.normal(
        scale=time_delay_noise, size=n_images
    )

    fluxes_noise_map = np.abs(fluxes_truth) * flux_noise_frac
    time_delays_noise_map = np.full(n_images, time_delay_noise)

    point_dataset = al.PointDataset(
        name="point_0",
        positions=al.Grid2DIrregular(values=[tuple(p) for p in positions_obs]),
        positions_noise_map=al.ArrayIrregular(values=[position_noise] * n_images),
        fluxes=al.ArrayIrregular(values=fluxes_obs.tolist()),
        fluxes_noise_map=al.ArrayIrregular(values=fluxes_noise_map.tolist()),
        time_delays=al.ArrayIrregular(values=delays_obs.tolist()),
        time_delays_noise_map=al.ArrayIrregular(values=time_delays_noise_map.tolist()),
    )

    al.output_to_json(
        obj=point_dataset, file_path=HERE / "point_dataset.json"
    )
    al.output_to_json(obj=tracer, file_path=HERE / "tracer.json")

    # =====================================================================
    # PART 2: Imaging dataset (host-galaxy arc)
    # =====================================================================
    # 200x200, 0.05" / pixel = 10" field of view; FWHM 0.10" Gaussian PSF.
    pixel_scales = 0.05
    shape_native = (200, 200)

    # Adaptive over-sampling at the source-plane host centre traced back to
    # the image plane (captures the brightest arcs accurately).
    grid_image = al.Grid2D.uniform(
        shape_native=shape_native, pixel_scales=pixel_scales
    )
    over_sample = al.util.over_sample.over_sample_size_via_radial_bins_from(
        grid=grid_image,
        sub_size_list=[32, 8, 2],
        radial_list=[0.3, 0.6],
        # Apply heavier sub-sampling near each truth image position +
        # near the lens centre (where the convergence diverges).
        centre_list=[(0.0, 0.0)] + [tuple(p) for p in np.asarray(positions_truth)],
    )
    grid_image = grid_image.apply_over_sampling(over_sample_size=over_sample)

    # PSF: FWHM 0.10" -> sigma = 0.10 / 2.355 = 0.0425"
    psf_fwhm = 0.10
    psf_sigma = psf_fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    psf = al.Convolver.from_gaussian(
        shape_native=(11, 11), sigma=psf_sigma, pixel_scales=pixel_scales,
    )

    simulator = al.SimulatorImaging(
        exposure_time=2000.0,
        psf=psf,
        background_sky_level=0.05,
        add_poisson_noise_to_data=True,
    )

    # IMPORTANT for self-consistency: the simulator must use the SAME tracer
    # we'll ship as ground truth. The `point_0` attribute on the source is
    # ignored by SimulatorImaging (it only renders LightProfiles), so the
    # extended arc is the only thing imprinted on `image.fits`. The point-
    # source spots are NOT injected into the imaging — that's the right
    # split: positions are handled by AnalysisPoint, the arc by
    # AnalysisImaging (consistent with how Wong+20 / Birrer+20 model the
    # quasar + host system; in real data the quasar PSF spots ARE in the
    # imaging and require deblending, which we leave for a follow-up — see
    # README §"What this DOESN'T cover").
    imaging = simulator.via_tracer_from(tracer=tracer, grid=grid_image)

    # Write image / noise / psf to FITS via the standard helper.
    image_path = HERE / "image.fits"
    noise_path = HERE / "noise_map.fits"
    psf_path = HERE / "psf.fits"
    aplt.fits_imaging(
        dataset=imaging,
        data_path=image_path,
        noise_map_path=noise_path,
        psf_path=psf_path,
        overwrite=True,
    )

    # =====================================================================
    # Self-consistency: chi^2/N at truth on the imaging fit must be ~1.
    # =====================================================================
    # Same pattern as Examples/cluster_scale/mocks/generate_mock.py. Catches
    # generator / fit-driver tracer mismatch BEFORE shipping.
    mask = al.Mask2D.circular(
        shape_native=imaging.shape_native,
        pixel_scales=pixel_scales,
        radius=4.0,
    )
    imaging_masked = imaging.apply_mask(mask=mask)
    fit = al.FitImaging(dataset=imaging_masked, tracer=tracer)
    chi2_per_pixel = fit.chi_squared / imaging_masked.mask.pixels_in_mask
    max_resid = float(np.max(np.abs(fit.normalized_residual_map.native)))
    print(f"\nGenerator self-consistency check (imaging):")
    print(f"  chi^2/pixel = {chi2_per_pixel:.4f}  (expect ~1.0)")
    print(f"  max|resid|  = {max_resid:.2f} sigma  (expect <6.0)")
    if chi2_per_pixel > 1.5 or max_resid > 6.0:
        raise RuntimeError(
            f"Generator self-consistency FAILED: chi^2/N={chi2_per_pixel:.2f} "
            f"max|res|={max_resid:.1f}σ — the simulator's tracer doesn't match "
            f"its own output. Investigate before shipping this mock."
        )
    print(f"  OK: imaging self-consistency passed.")

    # =====================================================================
    # Truth metadata
    # =====================================================================
    truths = {
        "cosmology": {"name": "FlatLambdaCDM", "H0": 70.0, "Om0": 0.30},
        "redshifts": {"z_lens": z_lens, "z_source": z_source},
        "mass": {
            "kind": "PowerLaw",
            "centre": [0.0, 0.0],
            "ell_comps": [0.10, 0.15],
            "einstein_radius": 1.2,
            "slope": 2.0,
        },
        "shear": {"gamma_1": 0.04, "gamma_2": 0.02},
        "source_point": {
            "kind": "ps.Point",
            "centre": list(source_centre),
            "flux_intrinsic": source_flux,
        },
        "source_host": {
            "kind": "lp.SersicCore",
            "centre": list(source_centre),
            "ell_comps": [0.0, 0.0],
            "effective_radius": 0.15,
            "sersic_index": 2.0,
            "intensity": 0.5,
        },
        "imaging": {
            "shape": list(shape_native),
            "pixel_scales": pixel_scales,
            "psf_fwhm_arcsec": psf_fwhm,
            "psf_sigma_arcsec": psf_sigma,
            "exposure_time_s": 2000.0,
            "background_sky_level": 0.05,
        },
        "noise": {
            "position": position_noise,
            "flux_frac": flux_noise_frac,
            "time_delay": time_delay_noise,
        },
        "truth_positions": [list(map(float, p)) for p in positions_truth],
        "truth_time_delays": list(map(float, np.asarray(time_delays_truth))),
        "truth_magnifications": list(map(float, np.asarray(magnifications))),
        "truth_fluxes": list(map(float, fluxes_truth)),
        "self_consistency": {
            "chi2_per_pixel": float(chi2_per_pixel),
            "max_normalized_residual": max_resid,
        },
    }
    truths_path = HERE / "truths.json"
    truths_path.write_text(json.dumps(truths, indent=2))

    print(f"\nWrote:")
    print(f"  {HERE / 'point_dataset.json'}")
    print(f"  {HERE / 'image.fits'}")
    print(f"  {HERE / 'noise_map.fits'}")
    print(f"  {HERE / 'psf.fits'}")
    print(f"  {HERE / 'tracer.json'}")
    print(f"  {HERE / 'truths.json'}")


if __name__ == "__main__":
    main()
