"""
fit_example_agel_real_target.py — Cannon driver for Examples/agel_real_target/.

Direct fit on the real AGEL013322-125201A HST/ACS F606W cutout:

    --part direct       Sersic lens light + PowerLaw mass + ExternalShear +
                        SersicCore source. n_live=200. ~1-2 h on 32 cores.

This is a SCAFFOLD on real data. PSF is a Gaussian placeholder; noise
treatment ignores drizzle correlations; mask is a simple 2.7" circle.
See Examples/agel_real_target/README.md §Caveats for the publication-
grade upgrades.

Usage:
    python fit_example_agel_real_target.py --part direct \\
        --dataset-root /path/to/Examples/agel_real_target/data \\
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
        print(f"[AGEL]   warning: post-fit visualize {tag} failed: {e}",
              flush=True)


def load_dataset(dataset_root: Path, mask_radius: float = 2.7):
    import autolens as al

    dataset = al.Imaging.from_fits(
        data_path=dataset_root / "image.fits",
        noise_map_path=dataset_root / "noise_map.fits",
        psf_path=dataset_root / "psf.fits",
        pixel_scales=0.05,
    )
    mask = al.Mask2D.circular(
        shape_native=dataset.shape_native,
        pixel_scales=dataset.pixel_scales,
        radius=mask_radius,
    )
    return dataset.apply_mask(mask=mask)


def build_direct(dataset, output_root: Path, n_live: int = 200,
                 z_lens: float = 0.30, z_source: float = 1.6):
    """Sersic+PowerLaw+ExternalShear lens, SersicCore source, single search."""
    import autofit as af
    import autolens as al

    print(f"\n[AGEL/direct] z_lens={z_lens}, z_source={z_source}", flush=True)

    bulge = af.Model(al.lp.Sersic)
    mass = af.Model(al.mp.PowerLaw)

    # Lens light + mass: independent centres (real lenses sometimes have
    # mass/light offsets at sub-pixel scale).
    bulge.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.3)
    bulge.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.3)
    mass.centre.centre_0  = af.GaussianPrior(mean=0.0, sigma=0.3)
    mass.centre.centre_1  = af.GaussianPrior(mean=0.0, sigma=0.3)

    bulge.intensity        = af.LogUniformPrior(lower_limit=1e-3, upper_limit=10.0)
    bulge.effective_radius = af.UniformPrior(lower_limit=0.05, upper_limit=2.0)
    bulge.sersic_index     = af.UniformPrior(lower_limit=0.5, upper_limit=8.0)
    bulge.ell_comps.ell_comps_0 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)
    bulge.ell_comps.ell_comps_1 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)

    mass.einstein_radius = af.UniformPrior(lower_limit=0.3, upper_limit=2.0)
    mass.slope           = af.TruncatedGaussianPrior(
        mean=2.0, sigma=0.3, lower_limit=1.5, upper_limit=2.7)
    mass.ell_comps.ell_comps_0 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)
    mass.ell_comps.ell_comps_1 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)

    shear = af.Model(al.mp.ExternalShear)
    shear.gamma_1 = af.GaussianPrior(mean=0.0, sigma=0.1)
    shear.gamma_2 = af.GaussianPrior(mean=0.0, sigma=0.1)

    lens = af.Model(al.Galaxy, redshift=z_lens,
                    bulge=bulge, mass=mass, shear=shear)

    src_b = af.Model(al.lp.SersicCore)
    src_b.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.5)
    src_b.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.5)
    src_b.ell_comps.ell_comps_0 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)
    src_b.ell_comps.ell_comps_1 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)
    src_b.intensity        = af.LogUniformPrior(lower_limit=1e-3, upper_limit=10.0)
    src_b.effective_radius = af.UniformPrior(lower_limit=0.0, upper_limit=1.0)
    src_b.sersic_index     = af.UniformPrior(lower_limit=0.5, upper_limit=4.0)

    source = af.Model(al.Galaxy, redshift=z_source, bulge=src_b)

    model = af.Collection(galaxies=af.Collection(lens=lens, source=source))
    print(f"Model priors: {model.prior_count}", flush=True)

    analysis = al.AnalysisImaging(dataset=dataset, use_jax=False)
    search = af.Nautilus(
        path_prefix=output_root,
        name="agel013322_direct",
        unique_tag="agel013322_sersic_powerlaw_shear_sersiccore",
        n_live=n_live, n_batch=50, iterations_per_update=10000,
        number_of_cores=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
    )

    t0 = time.time()
    result = search.fit(model=model, analysis=analysis)
    print(f"[AGEL/direct] done in {(time.time()-t0)/60:.1f} min", flush=True)
    _force_visualize(analysis, result, tag="direct")
    print(result.info, flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--part", choices=("direct",), default="direct")
    p.add_argument("--output-root", type=Path,
                   default=Path("./output").resolve())
    p.add_argument("--dataset-root", type=Path, required=True,
                   help="Path containing image.fits, noise_map.fits, psf.fits")
    p.add_argument("--repo-root", type=Path, required=False,
                   help="(slurm-driver compat)")
    p.add_argument("--mask-radius", type=float, default=2.7)
    p.add_argument("--n-live", type=int, default=200)
    p.add_argument("--z-lens", type=float, default=0.30)
    p.add_argument("--z-source", type=float, default=1.6)
    args = p.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)

    import autolens as al
    print(f"PyAutoLens: {al.__version__}", flush=True)
    print(f"Output root:  {args.output_root}", flush=True)
    print(f"Dataset root: {args.dataset_root}", flush=True)
    print(f"Part:         {args.part}", flush=True)
    print(f"Redshifts:    z_l={args.z_lens}, z_s={args.z_source}", flush=True)

    t_start = time.time()
    dataset = load_dataset(args.dataset_root, mask_radius=args.mask_radius)
    print(f"Loaded: {dataset.shape_native}, "
          f"pixels_in_mask={dataset.mask.pixels_in_mask}", flush=True)

    build_direct(
        dataset, args.output_root, n_live=args.n_live,
        z_lens=args.z_lens, z_source=args.z_source,
    )

    print(f"\nTotal wall time: {(time.time()-t_start)/3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
