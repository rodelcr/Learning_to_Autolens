"""
Mock generator for the quad_time_delay example.

Generates a quadruply-imaged point-source quasar at z_S = 2.0 lensed by
a PowerLaw + ExternalShear galaxy at z_L = 0.5. Output:

    point_dataset.json   — al.PointDataset with positions, time delays, fluxes
    tracer.json          — true tracer (for reference / overplotting)
    truths.json          — input parameters (mass, source, cosmology) for
                           verification after fitting

Run with the autolens conda env active:

    python generate_mock.py

Truth values (TDCOSMO-typical):
    H0           = 70.0 km/s/Mpc      (FlatLambdaCDM)
    Om0          = 0.30
    z_lens       = 0.5
    z_source     = 2.0
    PowerLaw:
        centre           = (0.0, 0.0)
        einstein_radius  = 1.2"
        slope            = 2.0          (isothermal special case)
        ell_components   = (0.10, 0.15) (axis ratio ~ 0.7, PA ~ 28 deg)
    ExternalShear:
        gamma_1, gamma_2 = (0.04, 0.02)
    Source:
        centre = (0.05, 0.07)            inside tangential caustic -> 4 images
        flux   = 1.0 (arbitrary units; magnification scales these)

Observational uncertainties (HST-class):
    position_noise   = 0.005"   (~0.1 HST pixel)
    flux_noise_frac  = 0.05     (5% photometry)
    time_delay_noise = 0.5 days (TDCOSMO best-case, e.g. RXJ1131 H0LiCOW)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import autofit as af
import autolens as al

OUT_DIR = Path(__file__).parent
RNG = np.random.default_rng(seed=42)


def main() -> None:
    cosmology = al.cosmo.FlatLambdaCDM(H0=70.0, Om0=0.30)

    z_lens, z_source = 0.5, 2.0

    mass = al.mp.PowerLaw(
        centre=(0.0, 0.0),
        ell_comps=(0.10, 0.15),
        einstein_radius=1.2,
        slope=2.0,
    )
    shear = al.mp.ExternalShear(gamma_1=0.04, gamma_2=0.02)

    lens_galaxy = al.Galaxy(redshift=z_lens, mass=mass, shear=shear)

    source_centre = (0.05, 0.07)
    source_flux = 1.0
    source_galaxy = al.Galaxy(
        redshift=z_source,
        point_0=al.ps.Point(centre=source_centre),
    )

    tracer = al.Tracer(
        galaxies=[lens_galaxy, source_galaxy],
        cosmology=cosmology,
    )

    grid = al.Grid2D.uniform(shape_native=(200, 200), pixel_scales=0.05)
    solver = al.PointSolver.for_grid(
        grid=grid,
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

    time_delays_truth = tracer.time_delays_from(grid=positions_truth)
    print(f"\nTime delays (truth, days, image_0 reference):")
    for i, td in enumerate(np.asarray(time_delays_truth)):
        print(f"  image {i}: {td:+8.3f}")

    deflections = tracer.deflections_yx_2d_from(grid=positions_truth)
    src_plane_pts = np.asarray(positions_truth) - np.asarray(deflections)
    print(f"\nSource-plane mapping (sanity check; should all match {source_centre}):")
    for i, p in enumerate(src_plane_pts):
        print(f"  image {i} -> source ({p[0]:+.4f}, {p[1]:+.4f})")

    lens_calc = al.LensCalc.from_tracer(tracer=tracer)
    magnifications = lens_calc.magnification_2d_via_hessian_from(grid=positions_truth)
    fluxes_truth = source_flux * np.abs(np.asarray(magnifications))
    print(f"\nMagnifications + fluxes (truth):")
    for i, (mu, f) in enumerate(zip(magnifications, fluxes_truth)):
        print(f"  image {i}: mu = {mu:+8.3f}, flux = {f:8.4f}")

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

    dataset = al.PointDataset(
        name="point_0",
        positions=al.Grid2DIrregular(values=[tuple(p) for p in positions_obs]),
        positions_noise_map=al.ArrayIrregular(values=[position_noise] * n_images),
        fluxes=al.ArrayIrregular(values=fluxes_obs.tolist()),
        fluxes_noise_map=al.ArrayIrregular(values=fluxes_noise_map.tolist()),
        time_delays=al.ArrayIrregular(values=delays_obs.tolist()),
        time_delays_noise_map=al.ArrayIrregular(values=time_delays_noise_map.tolist()),
    )

    al.output_to_json(obj=dataset, file_path=OUT_DIR / "point_dataset.json")
    al.output_to_json(obj=tracer, file_path=OUT_DIR / "tracer.json")

    truths = {
        "cosmology": {"name": "FlatLambdaCDM", "H0": 70.0, "Om0": 0.30},
        "redshifts": {"z_lens": z_lens, "z_source": z_source},
        "mass": {
            "kind": "PowerLaw",
            "centre": [0.0, 0.0],
            "ell_comps": [0.05, 0.10],
            "einstein_radius": 1.2,
            "slope": 2.0,
        },
        "shear": {"gamma_1": 0.04, "gamma_2": 0.02},
        "source": {
            "kind": "ps.Point",
            "centre": list(source_centre),
            "flux_intrinsic": source_flux,
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
    }
    (OUT_DIR / "truths.json").write_text(json.dumps(truths, indent=2))
    print(f"\nWrote:\n  {OUT_DIR / 'point_dataset.json'}\n"
          f"  {OUT_DIR / 'tracer.json'}\n"
          f"  {OUT_DIR / 'truths.json'}")


if __name__ == "__main__":
    main()
