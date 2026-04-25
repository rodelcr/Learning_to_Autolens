"""
fit_example_interferometer_basic.py — Cannon driver for Examples/interferometer_basic/.

Single-search fit on the SMA-resolution interferometer mock:

    Lens: Isothermal + ExternalShear
    Source: SersicCore
    Likelihood: visibility-plane chi^2 via TransformerDFT

The SMA dataset is small (190 visibilities) so this is a fast fit (~5-15 min
on 32 cores). For ALMA-class data, swap to TransformerNUFFT and use_jax=True.

Usage (Cannon):
    python fit_example_interferometer_basic.py --part direct \\
        --dataset-root /path/to/Examples/interferometer_basic/mocks \\
        --output-root  /path/to/output

Nautilus auto-resumes from any existing checkpoint.hdf5.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path


def _force_visualize(analysis, result, tag: str = ""):
    try:
        analysis.visualize(
            paths=result.paths,
            instance=result.max_log_likelihood_instance,
            during_analysis=False,
        )
    except Exception as e:
        print(f"[INTF]   warning: post-fit visualize {tag} failed: {e}",
              flush=True)


def load_dataset(dataset_root: Path, real_space_radius: float = 3.0):
    import autolens as al

    real_space_mask = al.Mask2D.circular(
        shape_native=(256, 256), pixel_scales=0.05, radius=real_space_radius,
    )
    return al.Interferometer.from_fits(
        data_path=dataset_root / "data.fits",
        noise_map_path=dataset_root / "noise_map.fits",
        uv_wavelengths_path=dataset_root / "uv_wavelengths.fits",
        real_space_mask=real_space_mask,
        transformer_class=al.TransformerDFT,
    )


def build_direct(dataset, output_root: Path, n_live: int = 150):
    """Direct Isothermal + shear + SersicCore source fit."""
    import autofit as af
    import autolens as al

    print("\n[INTF/direct] Isothermal + ExternalShear + SersicCore visibility fit.",
          flush=True)

    mass = af.Model(al.mp.Isothermal)
    mass.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.1)
    mass.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.1)
    mass.einstein_radius = af.UniformPrior(lower_limit=0.5, upper_limit=3.0)
    mass.ell_comps.ell_comps_0 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)
    mass.ell_comps.ell_comps_1 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)
    shear = af.Model(al.mp.ExternalShear)
    shear.gamma_1 = af.GaussianPrior(mean=0.0, sigma=0.1)
    shear.gamma_2 = af.GaussianPrior(mean=0.0, sigma=0.1)
    lens = af.Model(al.Galaxy, redshift=0.5, mass=mass, shear=shear)

    src_b = af.Model(al.lp.SersicCore)
    src_b.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.3)
    src_b.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.3)
    src_b.ell_comps.ell_comps_0 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)
    src_b.ell_comps.ell_comps_1 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)
    src_b.intensity = af.LogUniformPrior(lower_limit=1e-3, upper_limit=10.0)
    src_b.effective_radius = af.UniformPrior(lower_limit=0.1, upper_limit=3.0)
    src_b.sersic_index = af.UniformPrior(lower_limit=0.5, upper_limit=4.0)
    source = af.Model(al.Galaxy, redshift=1.0, bulge=src_b)

    model = af.Collection(galaxies=af.Collection(lens=lens, source=source))
    print(f"Model priors: {model.prior_count}", flush=True)

    analysis = al.AnalysisInterferometer(dataset=dataset, use_jax=False)

    search = af.Nautilus(
        path_prefix=output_root,
        name="interferometer_basic",
        unique_tag="direct_iso_shear_sersic",
        n_live=n_live,
        n_batch=50,
        iterations_per_update=5000,
        number_of_cores=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
    )

    t0 = time.time()
    result = search.fit(model=model, analysis=analysis)
    print(f"[INTF/direct] done in {(time.time()-t0)/60:.1f} min", flush=True)
    _force_visualize(analysis, result, tag="direct")
    print(result.info, flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--part", choices=("direct",), default="direct")
    p.add_argument("--output-root", type=Path,
                   default=Path("./output").resolve())
    p.add_argument("--dataset-root", type=Path, required=True,
                   help="Path containing data.fits + noise_map.fits + uv_wavelengths.fits")
    p.add_argument("--repo-root", type=Path, required=False,
                   help="(slurm-driver compat)")
    p.add_argument("--n-live", type=int, default=150)
    args = p.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)

    import autolens as al
    print(f"PyAutoLens: {al.__version__}", flush=True)
    print(f"Output root:  {args.output_root}", flush=True)
    print(f"Dataset root: {args.dataset_root}", flush=True)
    print(f"Part:         {args.part}", flush=True)
    print(f"SLURM_JOB_ID: {os.environ.get('SLURM_JOB_ID', '(none)')}", flush=True)

    t_start = time.time()
    dataset = load_dataset(args.dataset_root)
    print(f"Loaded interferometer dataset: {dataset.data.shape[0]} visibilities",
          flush=True)

    build_direct(dataset, args.output_root, n_live=args.n_live)

    print(f"\nTotal wall time: {(time.time()-t_start)/3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
