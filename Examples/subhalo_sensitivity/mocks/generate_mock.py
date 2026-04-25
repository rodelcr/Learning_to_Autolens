"""
Mock generator for the subhalo_sensitivity example.

Generates a simulated strong lens where a SMALL dark-matter subhalo
overlaps the source emission and perturbs the image plane in a way that
should be detectable via Bayesian model comparison.

Lens architecture:
    z_lens = 0.5
    Main mass: Isothermal (theta_E=1.6", ell_comps=(0.05, 0.05))
    External shear: (gamma_1, gamma_2) = (0.05, 0.05)
    Subhalo: NFW (truncated, sub-galactic) at fixed offset from lens centre,
             mass ~10^9 Msun (just above PyAutoLens detection threshold)
    z_source = 1.0
    Source: SersicCore (R_e=0.1", n=1.5)

The subhalo is placed at offset (+0.5, +0.5)" from the main lens — well
inside the Einstein ring where it can perturb the lensed source arcs.

Output (in mocks/):
    image.fits, noise_map.fits, psf.fits — standard imaging triple
    tracer.json — true tracer
    truths.json — input parameters

Run with the autolens conda env:
    python generate_mock.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import autolens as al
import autolens.plot as aplt

OUT = Path(__file__).parent


def main():
    grid = al.Grid2D.uniform(shape_native=(110, 110), pixel_scales=0.05)
    psf = al.Convolver.from_gaussian(
        shape_native=(11, 11), sigma=0.05, pixel_scales=grid.pixel_scales,
    )

    simulator = al.SimulatorImaging(
        exposure_time=300.0,
        psf=psf,
        background_sky_level=0.1,
        add_poisson_noise_to_data=True,
        noise_seed=42,
    )

    z_lens, z_source = 0.5, 1.0
    main_mass = al.mp.Isothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.6,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
    )
    shear = al.mp.ExternalShear(gamma_1=0.05, gamma_2=0.05)

    # Subhalo: small truncated-NFW inside the Einstein ring.
    # NFWTruncatedMCRDuffySph picks the Duffy+08 mass-concentration relation
    # and lets us pick the mass directly. mass_at_200 = 1e9 Msun is right
    # at the PyAutoLens detection threshold (Vegetti+ 2010 type).
    subhalo_mass = al.mp.NFWTruncatedMCRDuffySph(
        centre=(0.5, 0.5),
        mass_at_200=1.0e9,
    )

    lens_galaxy = al.Galaxy(
        redshift=z_lens, mass=main_mass, shear=shear, subhalo=subhalo_mass,
    )

    source_galaxy = al.Galaxy(
        redshift=z_source,
        bulge=al.lp.SersicCore(
            centre=(0.0, 0.05),
            ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
            intensity=0.5,
            effective_radius=0.1,
            sersic_index=1.5,
        ),
    )

    tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])
    dataset = simulator.via_tracer_from(tracer=tracer, grid=grid)

    aplt.fits_imaging(
        dataset=dataset,
        data_path=OUT / "image.fits",
        noise_map_path=OUT / "noise_map.fits",
        psf_path=OUT / "psf.fits",
        overwrite=True,
    )
    al.output_to_json(obj=tracer, file_path=OUT / "tracer.json")

    truths = {
        "redshifts": {"lens": z_lens, "source": z_source},
        "main_lens": {
            "mass": "Isothermal",
            "centre": [0.0, 0.0],
            "einstein_radius": 1.6,
            "ell_comps_from_axis_ratio": [0.9, 45.0],
        },
        "shear": {"gamma_1": 0.05, "gamma_2": 0.05},
        "subhalo": {
            "mass_class": "NFWTruncatedMCRDuffySph",
            "centre": [0.5, 0.5],
            "mass_at_200": 1.0e9,
            "comment": ("Just above the canonical PyAutoLens detection "
                        "threshold (~10^8.5 Msun for HST-class data)"),
        },
        "source": {
            "kind": "SersicCore",
            "centre": [0.0, 0.05],
            "ell_comps_from_axis_ratio": [0.8, 60.0],
            "intensity": 0.5,
            "effective_radius": 0.1,
            "sersic_index": 1.5,
        },
        "imaging": {
            "shape_native": [110, 110],
            "pixel_scales_arcsec": 0.05,
            "exposure_time_s": 300.0,
            "psf_sigma_arcsec": 0.05,
            "background_sky_level": 0.1,
        },
    }
    (OUT / "truths.json").write_text(json.dumps(truths, indent=2))

    print(f"Wrote: image.fits, noise_map.fits, psf.fits, tracer.json, "
          f"truths.json in {OUT}")
    print(f"Subhalo: NFW truncated, M_200=1e9 Msun at offset (0.5, 0.5)\"")


if __name__ == "__main__":
    main()
