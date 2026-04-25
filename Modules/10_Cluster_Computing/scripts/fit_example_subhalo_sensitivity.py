"""
fit_example_subhalo_sensitivity.py — Cannon driver for Examples/subhalo_sensitivity/.

Two-fit Bayesian model comparison for DM substructure detection:

    --part smooth         Isothermal + ExternalShear + SersicCore source.
                          12 free params, n_live=150. ~30-60 min.

    --part with_subhalo   Same + NFWTruncatedMCRDuffySph perturber with free
                          (centre, mass_at_200). 15 free params, n_live=200.
                          ~1-2 h.

    --part both           Sequential smooth -> with_subhalo.

Compute Δlog_Z = log_Z(with_subhalo) - log_Z(smooth) post-fit. Trotta scale:
> 5 strong, > 10 decisive (see Examples/bayesian_model_comparison/).

This is a SCAFFOLD — the publication-grade methodology adds a SLaM source
pipeline + grid-search of subhalo positions. See Examples/subhalo_sensitivity/
README.md for the planned extensions.

Usage:
    python fit_example_subhalo_sensitivity.py --part both \\
        --dataset-root /path/to/Examples/subhalo_sensitivity/mocks \\
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
        print(f"[SUB]   warning: post-fit visualize {tag} failed: {e}",
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


def _build_lens_smooth():
    import autofit as af
    import autolens as al

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

    return af.Model(al.Galaxy, redshift=0.5, mass=mass, shear=shear)


def _build_source():
    import autofit as af
    import autolens as al

    bulge = af.Model(al.lp.SersicCore)
    bulge.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.3)
    bulge.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.3)
    bulge.ell_comps.ell_comps_0 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)
    bulge.ell_comps.ell_comps_1 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)
    bulge.intensity = af.LogUniformPrior(lower_limit=1e-3, upper_limit=10.0)
    bulge.effective_radius = af.UniformPrior(lower_limit=0.01, upper_limit=1.0)
    bulge.sersic_index = af.UniformPrior(lower_limit=0.5, upper_limit=4.0)
    return af.Model(al.Galaxy, redshift=1.0, bulge=bulge)


def build_smooth(dataset, output_root: Path, n_live: int = 150):
    import autofit as af
    import autolens as al

    print("\n[SUB/smooth] Isothermal+shear+SersicCore (no subhalo).", flush=True)
    lens = _build_lens_smooth()
    source = _build_source()
    model = af.Collection(galaxies=af.Collection(lens=lens, source=source))
    print(f"smooth model priors: {model.prior_count}", flush=True)

    analysis = al.AnalysisImaging(dataset=dataset, use_jax=False)
    search = af.Nautilus(
        path_prefix=output_root,
        name="subhalo_smooth", unique_tag="iso_shear_sersic_no_perturber",
        n_live=n_live, n_batch=50, iterations_per_update=10000,
        number_of_cores=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
    )

    t0 = time.time()
    result = search.fit(model=model, analysis=analysis)
    print(f"[SUB/smooth] done in {(time.time()-t0)/60:.1f} min", flush=True)
    _force_visualize(analysis, result, tag="smooth")
    print(result.info, flush=True)
    return result


def build_with_subhalo(dataset, output_root: Path, result_smooth=None,
                       n_live: int = 200):
    import autofit as af
    import autolens as al

    print("\n[SUB/with_subhalo] +NFWTruncatedMCRDuffySph perturber.", flush=True)

    # Reuse smooth-fit posteriors for main lens components if available
    if result_smooth is not None:
        lens = result_smooth.model.galaxies.lens
        source = result_smooth.model.galaxies.source
        # Add subhalo to existing lens (need new af.Model wrapping the
        # original — easier to rebuild with priors-from-result)
        mass = result_smooth.model.galaxies.lens.mass
        shear = result_smooth.model.galaxies.lens.shear
    else:
        mass = _build_lens_smooth().mass
        shear = _build_lens_smooth().shear
        source = _build_source()

    subhalo = af.Model(al.mp.NFWTruncatedMCRDuffySph)
    subhalo.centre.centre_0 = af.UniformPrior(lower_limit=-2.0, upper_limit=2.0)
    subhalo.centre.centre_1 = af.UniformPrior(lower_limit=-2.0, upper_limit=2.0)
    subhalo.mass_at_200 = af.LogUniformPrior(lower_limit=1e7, upper_limit=1e11)

    lens = af.Model(al.Galaxy, redshift=0.5,
                    mass=mass, shear=shear, subhalo=subhalo)
    if result_smooth is None:
        source = _build_source()

    model = af.Collection(galaxies=af.Collection(lens=lens, source=source))
    print(f"with_subhalo model priors: {model.prior_count}", flush=True)

    analysis = al.AnalysisImaging(dataset=dataset, use_jax=False)
    search = af.Nautilus(
        path_prefix=output_root,
        name="subhalo_with_perturber", unique_tag="iso_shear_sersic_plus_nfw",
        n_live=n_live, n_batch=50, iterations_per_update=10000,
        number_of_cores=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
    )

    t0 = time.time()
    result = search.fit(model=model, analysis=analysis)
    print(f"[SUB/with_subhalo] done in {(time.time()-t0)/60:.1f} min", flush=True)
    _force_visualize(analysis, result, tag="with_subhalo")
    print(result.info, flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--part",
                   choices=("smooth", "with_subhalo", "both"), default="both")
    p.add_argument("--output-root", type=Path,
                   default=Path("./output").resolve())
    p.add_argument("--dataset-root", type=Path, required=True,
                   help="Path containing image.fits + noise_map.fits + psf.fits")
    p.add_argument("--repo-root", type=Path, required=False,
                   help="(slurm-driver compat)")
    p.add_argument("--n-live-smooth", type=int, default=150)
    p.add_argument("--n-live-with-subhalo", type=int, default=200)
    args = p.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)

    import autolens as al
    print(f"PyAutoLens: {al.__version__}", flush=True)
    print(f"Output root:  {args.output_root}", flush=True)
    print(f"Dataset root: {args.dataset_root}", flush=True)
    print(f"Part:         {args.part}", flush=True)

    t_start = time.time()
    dataset = load_dataset(args.dataset_root)
    print(f"Loaded: {dataset.shape_native}, "
          f"pixels_in_mask={dataset.mask.pixels_in_mask}", flush=True)

    result_smooth = None
    if args.part in ("smooth", "both"):
        result_smooth = build_smooth(
            dataset, args.output_root, n_live=args.n_live_smooth)

    if args.part in ("with_subhalo", "both"):
        build_with_subhalo(
            dataset, args.output_root, result_smooth=result_smooth,
            n_live=args.n_live_with_subhalo)

    print(f"\nTotal wall time: {(time.time()-t_start)/3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
