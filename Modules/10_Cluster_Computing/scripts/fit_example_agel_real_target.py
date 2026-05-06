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


def load_dataset(dataset_root: Path, mask_radius: float = 2.7,
                 hot_pixel_threshold: float = 0.0):
    """Load FITS + apply circular mask. If hot_pixel_threshold > 0, also OR-in
    a hot-pixel mask (|S/N| > threshold pixels excluded). The notebook §1.5
    found 44 cosmic-ray survivor pixels at threshold=8.0 in this AGEL cutout
    that were responsible for a 32σ residual peak in the v0.92 direct fit.
    """
    import autolens as al
    import numpy as np

    dataset = al.Imaging.from_fits(
        data_path=dataset_root / "image.fits",
        noise_map_path=dataset_root / "noise_map.fits",
        psf_path=dataset_root / "psf.fits",
        pixel_scales=0.05,
    )
    base_mask = al.Mask2D.circular(
        shape_native=dataset.shape_native,
        pixel_scales=dataset.pixel_scales,
        radius=mask_radius,
    )
    if hot_pixel_threshold > 0.0:
        sn = np.abs(np.array(dataset.data.native)) / np.array(dataset.noise_map.native)
        outliers = sn > hot_pixel_threshold
        n_outliers = int(outliers.sum())
        combined = np.array(base_mask) | outliers
        mask = al.Mask2D(mask=combined, pixel_scales=dataset.pixel_scales)
        print(f"[AGEL] hot-pixel mask: threshold={hot_pixel_threshold}σ, "
              f"{n_outliers} outliers excluded.", flush=True)
    else:
        mask = base_mask
    return dataset.apply_mask(mask=mask)


def build_direct(dataset, output_root: Path, n_live: int = 200,
                 z_lens: float = 0.30, z_source: float = 1.6,
                 name: str = "agel013322_direct",
                 unique_tag: str = "agel013322_sersic_powerlaw_shear_sersiccore"):
    """Sersic+PowerLaw+ExternalShear lens, SersicCore source, single search.
    Same model regardless of whether the dataset has the hot-pixel mask
    applied — the mask lives on the dataset, not on the model."""
    import autofit as af
    import autolens as al

    print(f"\n[AGEL/{name}] z_lens={z_lens}, z_source={z_source}", flush=True)

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
    # `name` differentiates {direct, direct_clean} so they don't collide on
    # output paths and the diff between them (=just the hot-pixel mask) is
    # readable from the artifact tree.
    search = af.Nautilus(
        path_prefix=output_root,
        name=name,
        unique_tag=unique_tag,
        n_live=n_live, n_batch=50, iterations_per_update=10000,
        number_of_cores=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
    )

    t0 = time.time()
    result = search.fit(model=model, analysis=analysis)
    print(f"[AGEL/{name}] done in {(time.time()-t0)/60:.1f} min", flush=True)
    _force_visualize(analysis, result, tag=name)
    print(result.info, flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--part",
        choices=("direct", "direct_clean"),
        default="direct",
        help="direct: 2.7\" circular mask only. "
             "direct_clean: same model, but |S/N|>8 hot pixels also masked "
             "(catches the 32σ residual that kept v0.92 direct fit borderline)."
    )
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
    hot_threshold = 8.0 if args.part == "direct_clean" else 0.0
    dataset = load_dataset(
        args.dataset_root, mask_radius=args.mask_radius,
        hot_pixel_threshold=hot_threshold,
    )
    print(f"Loaded: {dataset.shape_native}, "
          f"pixels_in_mask={dataset.mask.pixels_in_mask}", flush=True)

    if args.part == "direct":
        build_direct(
            dataset, args.output_root, n_live=args.n_live,
            z_lens=args.z_lens, z_source=args.z_source,
        )
    elif args.part == "direct_clean":
        build_direct(
            dataset, args.output_root, n_live=args.n_live,
            z_lens=args.z_lens, z_source=args.z_source,
            name="agel013322_direct_clean",
            unique_tag="agel013322_sersic_powerlaw_shear_sersiccore_hotpix8",
        )
    else:
        raise ValueError(f"unknown --part {args.part}")

    print(f"\nTotal wall time: {(time.time()-t_start)/3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
