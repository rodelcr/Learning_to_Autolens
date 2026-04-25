"""
fit_example_compound_lens_zoo.py — Cannon driver for Examples/compound_lens_zoo/.

Unified fit across the five lenstronomy compound-lens mocks (mock_2 through
mock_6). One model, one prior set, fit per mock. The prior set is the
v4-compound-lens recipe carried over from Examples/compound_lens/:

    PowerLaw(slope free) + ExternalShear(absorbs secondary deflector) +
    Sersic lens light + SersicCore source + cosmology fixed at standard.

Cosmology is held at FlatLambdaCDM(70, 0.30) for ALL five mocks, including
mocks 2 and 5 which were generated with non-standard cosmologies. This is
intentional — Exercise 2 in the README compares the recovered parameters
against the truth-cosmology fits to expose the cosmology-mass-profile
degeneracy.

Usage:
    python fit_example_compound_lens_zoo.py --mock 2 \\
        --dataset-root /path/to/Examples/compound_lens_zoo/mocks \\
        --output-root  /path/to/output

    --mock all   — sequential 2 -> 3 -> 4 -> 5 -> 6 (~10 h)
    --mock <N>   — just one mock (~1-2 h)

Nautilus auto-resumes from existing checkpoint.hdf5.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
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
        print(f"[ZOO]   warning: post-fit visualize {tag} failed: {e}",
              flush=True)


def load_dataset(dataset_root: Path, mock_index: int, mask_radius: float = 2.7):
    import autolens as al

    dataset = al.Imaging.from_fits(
        data_path      = dataset_root / f"lenstronomy_mock_{mock_index}_image.fits",
        noise_map_path = dataset_root / f"lenstronomy_mock_{mock_index}_noise.fits",
        psf_path       = dataset_root / f"lenstronomy_mock_psf.fits",
        pixel_scales   = 0.05,
    )
    mask = al.Mask2D.circular(
        shape_native = dataset.shape_native,
        pixel_scales = dataset.pixel_scales,
        radius       = mask_radius,
    )
    return dataset.apply_mask(mask=mask)


def build_unified_model():
    """The shared model used for every mock."""
    import autofit as af
    import autolens as al

    bulge = af.Model(al.lp.Sersic)
    mass  = af.Model(al.mp.PowerLaw)
    shear = af.Model(al.mp.ExternalShear)

    # Centres independent — let the data decide if light and mass agree.
    mass.centre.centre_0  = af.GaussianPrior(mean=0.0, sigma=0.3)
    mass.centre.centre_1  = af.GaussianPrior(mean=0.0, sigma=0.3)
    bulge.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.5)
    bulge.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.5)

    # Mass: PowerLaw with slope free, wide θ_E.
    mass.einstein_radius = af.UniformPrior(lower_limit=0.5, upper_limit=3.5)
    mass.slope           = af.TruncatedGaussianPrior(
        mean=2.0, sigma=0.3, lower_limit=1.5, upper_limit=2.7)
    mass.ell_comps.ell_comps_0 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)
    mass.ell_comps.ell_comps_1 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)

    # Lens light: wide Sersic.
    bulge.intensity        = af.LogUniformPrior(lower_limit=1e-6, upper_limit=1e6)
    bulge.effective_radius = af.UniformPrior(lower_limit=0.1, upper_limit=5.0)
    bulge.sersic_index     = af.UniformPrior(lower_limit=0.5, upper_limit=8.0)
    bulge.ell_comps.ell_comps_0 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)
    bulge.ell_comps.ell_comps_1 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)

    shear.gamma_1 = af.GaussianPrior(mean=0.0, sigma=0.1)
    shear.gamma_2 = af.GaussianPrior(mean=0.0, sigma=0.1)

    lens = af.Model(al.Galaxy, redshift=0.5,  # placeholder; overridden per mock
                    bulge=bulge, mass=mass, shear=shear)

    src_b = af.Model(al.lp.SersicCore)
    src_b.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.5)
    src_b.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.5)
    src_b.ell_comps.ell_comps_0 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)
    src_b.ell_comps.ell_comps_1 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)
    src_b.intensity        = af.LogUniformPrior(lower_limit=1e-3, upper_limit=1e3)
    src_b.effective_radius = af.UniformPrior(lower_limit=0.0, upper_limit=2.0)
    src_b.sersic_index     = af.UniformPrior(lower_limit=0.5, upper_limit=4.0)

    source = af.Model(al.Galaxy, redshift=1.7,  # placeholder; overridden
                      bulge=src_b)

    return af.Collection(galaxies=af.Collection(lens=lens, source=source))


def build_fit(dataset, output_root: Path, mock_index: int, truths: dict,
              n_live: int = 200):
    """Run one Nautilus fit on one mock with the unified model."""
    import autofit as af
    import autolens as al

    print(f"\n[ZOO/mock_{mock_index}] starting fit "
          f"(z_l={truths['redshifts']['lens_primary']}, "
          f"z_s={truths['redshifts']['source']}, "
          f"truth θ_E={truths['kwargs_lens'][0]['theta_E']:.3f}, "
          f"γ'={truths['kwargs_lens'][0]['gamma']:.2f})",
          flush=True)

    model = build_unified_model()

    # Inject the per-mock redshifts (override placeholders).
    model.galaxies.lens.redshift   = truths["redshifts"]["lens_primary"]
    model.galaxies.source.redshift = truths["redshifts"]["source"]

    analysis = al.AnalysisImaging(dataset=dataset, use_jax=False)

    search = af.Nautilus(
        path_prefix=output_root,
        name=f"mock_{mock_index}",
        unique_tag="powerlaw_shear_sersic_unified",
        n_live=n_live,
        n_batch=50,
        iterations_per_update=15000,
        number_of_cores=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
    )

    t0 = time.time()
    result = search.fit(model=model, analysis=analysis)
    print(f"[ZOO/mock_{mock_index}] done in {(time.time()-t0)/60:.1f} min",
          flush=True)
    _force_visualize(analysis, result, tag=f"mock_{mock_index}")
    print(result.info, flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mock", type=str, default="all",
                   help="2, 3, 4, 5, 6, or 'all' (default: all)")
    p.add_argument("--output-root", type=Path,
                   default=Path("./output").resolve())
    p.add_argument("--dataset-root", type=Path, required=True,
                   help="Path containing lenstronomy_mock_*.fits and truths_*.json")
    p.add_argument("--repo-root", type=Path, required=False,
                   help="Path to Learning_to_Autolens (slurm-driver compat)")
    p.add_argument("--n-live", type=int, default=200)
    args = p.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)

    if args.mock == "all":
        mocks_to_fit = [2, 3, 4, 5, 6]
    else:
        mocks_to_fit = [int(args.mock)]

    import autolens as al
    print(f"PyAutoLens: {al.__version__}", flush=True)
    print(f"Mocks to fit: {mocks_to_fit}", flush=True)
    print(f"Output root:  {args.output_root}", flush=True)
    print(f"Dataset root: {args.dataset_root}", flush=True)

    t_start = time.time()
    for n in mocks_to_fit:
        truths = json.loads(
            (args.dataset_root / f"truths_mock_{n}.json").read_text()
        )
        dataset = load_dataset(args.dataset_root, mock_index=n)
        print(f"\nLoaded mock_{n}: shape={dataset.shape_native}, "
              f"pixels_in_mask={dataset.mask.pixels_in_mask}",
              flush=True)
        build_fit(dataset, args.output_root, mock_index=n, truths=truths,
                  n_live=args.n_live)

    print(f"\nTotal wall time: {(time.time()-t_start)/3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
