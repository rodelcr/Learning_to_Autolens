"""
Mock generator for the interferometer_basic example.

Generates an interferometer dataset (visibilities) of a galaxy-scale strong
lens using the SMA uv-coverage:

    Lens at z=0.5: Isothermal mass + ExternalShear
    Source at z=1.0: SersicCore light

Output (in mocks/):
    data.fits, noise_map.fits — visibilities (real + imag) and noise
    tracer.json — true tracer for verification
    truths.json — input parameters (Isothermal, shear, source) for audit

Reference: autolens_workspace_latest/scripts/interferometer/simulator.py.
The SMA is a low-resolution array — modest visibility count (~6000), so
fits are quick. Use ALMA uv_wavelengths for high-resolution work.

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
    grid = al.Grid2D.uniform(shape_native=(256, 256), pixel_scales=0.05)

    uv_wavelengths = al.ndarray_via_fits_from(
        file_path=OUT / "sma_uv_wavelengths.fits", hdu=0,
    )

    simulator = al.SimulatorInterferometer(
        uv_wavelengths=uv_wavelengths,
        exposure_time=300.0,
        noise_sigma=1000.0,
        transformer_class=al.TransformerDFT,
    )

    z_lens, z_source = 0.5, 1.0
    lens_galaxy = al.Galaxy(
        redshift=z_lens,
        mass=al.mp.Isothermal(
            centre=(0.0, 0.0),
            einstein_radius=1.6,
            ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
        ),
        shear=al.mp.ExternalShear(gamma_1=0.05, gamma_2=0.05),
    )
    source_galaxy = al.Galaxy(
        redshift=z_source,
        bulge=al.lp.SersicCore(
            centre=(0.0, 0.0),
            ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
            intensity=0.3,
            effective_radius=1.0,
            sersic_index=2.5,
        ),
    )
    tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])
    dataset = simulator.via_tracer_from(tracer=tracer, grid=grid)

    aplt.fits_interferometer(
        dataset=dataset,
        data_path=OUT / "data.fits",
        noise_map_path=OUT / "noise_map.fits",
        uv_wavelengths_path=OUT / "uv_wavelengths.fits",
        overwrite=True,
    )
    al.output_to_json(obj=tracer, file_path=OUT / "tracer.json")

    truths = {
        "redshifts": {"lens": z_lens, "source": z_source},
        "lens": {
            "mass": "Isothermal",
            "centre": [0.0, 0.0],
            "einstein_radius": 1.6,
            "ell_comps_from_axis_ratio": [0.9, 45.0],
            "shear": {"gamma_1": 0.05, "gamma_2": 0.05},
        },
        "source": {
            "kind": "SersicCore",
            "centre": [0.0, 0.0],
            "ell_comps_from_axis_ratio": [0.8, 60.0],
            "intensity": 0.3,
            "effective_radius": 1.0,
            "sersic_index": 2.5,
        },
        "uv_array": "SMA (Submillimeter Array, low-res)",
        "exposure_time_s": 300.0,
        "noise_sigma": 1000.0,
    }
    (OUT / "truths.json").write_text(json.dumps(truths, indent=2))

    print(f"Wrote: data.fits, noise_map.fits, uv_wavelengths.fits, "
          f"tracer.json, truths.json in {OUT}")
    n_vis = uv_wavelengths.shape[0]
    print(f"Dataset has {n_vis} visibilities (SMA uv-coverage)")


if __name__ == "__main__":
    main()
