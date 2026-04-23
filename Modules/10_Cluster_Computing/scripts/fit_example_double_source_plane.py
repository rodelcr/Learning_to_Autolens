"""fit_example_double_source_plane.py — Standalone Cannon driver.

Fits the one-lens / two-source DSPL mock at
Examples/double_source_plane/mocks/. Native multi-plane via
`al.Tracer` with 3 redshift planes (z_L=0.5, z_S1=1.0, z_S2=2.5).

Part:
    direct     Single Nautilus fit of 1 lens + 2 sources (~26 free params)

Usage (Cannon):
    sbatch --export=ALL,EXAMPLE=double_source_plane,FIT_EXTRA_ARGS=--part=direct \
        submit_cannon.slurm
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


def _force_visualize(analysis, result, tag: str = ""):
    try:
        analysis.visualize(paths=result.paths,
                           instance=result.max_log_likelihood_instance,
                           during_analysis=False)
    except Exception as e:
        print(f"[DSPL] warning: post-fit visualize {tag} failed: {e}",
              flush=True)


def load_dataset(dataset_root: Path, mask_radius: float = 2.8):
    import autolens as al
    dataset = al.Imaging.from_fits(
        data_path      = dataset_root / "mock_image.fits",
        noise_map_path = dataset_root / "mock_noise.fits",
        psf_path       = dataset_root / "mock_psf.fits",
        pixel_scales   = 0.05,
    )
    mask = al.Mask2D.circular(
        shape_native = dataset.shape_native,
        pixel_scales = dataset.pixel_scales,
        radius       = mask_radius,
    )
    return dataset.apply_mask(mask=mask)


def build_direct_fit(dataset, output_root: Path, n_live: int = 200):
    """One-lens / two-source free fit. Loose-but-informative priors.

    The key thing DSPL teaches: the RATIO of Einstein radii for the two
    sources at (z_S1, z_S2) is a cosmological distance ratio β, so both
    sources must share the same lens_0.mass posterior — multi-plane
    ray-tracing handles that automatically as long as we pass all three
    galaxies to al.Tracer.
    """
    import autofit as af
    import autolens as al

    # ---- Lens (z=0.5) — Sersic bulge + Isothermal mass + shear ----
    bulge = af.Model(al.lp.Sersic)
    mass  = af.Model(al.mp.Isothermal)
    shear = af.Model(al.mp.ExternalShear)

    mass.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.1)
    mass.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.1)
    bulge.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.1)
    bulge.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.1)

    bulge.ell_comps.ell_comps_0 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)
    bulge.ell_comps.ell_comps_1 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)
    bulge.intensity        = af.LogUniformPrior(lower_limit=1e-3, upper_limit=1e2)
    bulge.effective_radius = af.UniformPrior(lower_limit=0.1, upper_limit=3.0)
    bulge.sersic_index     = af.UniformPrior(lower_limit=0.8, upper_limit=5.0)

    mass.ell_comps.ell_comps_0 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)
    mass.ell_comps.ell_comps_1 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)
    mass.einstein_radius = af.UniformPrior(lower_limit=0.5, upper_limit=3.0)

    shear.gamma_1 = af.GaussianPrior(mean=0.0, sigma=0.1)
    shear.gamma_2 = af.GaussianPrior(mean=0.0, sigma=0.1)

    lens = af.Model(al.Galaxy, redshift=0.5,
                    bulge=bulge, mass=mass, shear=shear)

    # ---- Source 1 (z=1.0) + Source 2 (z=2.5) ----
    def _source_model(z):
        b = af.Model(al.lp.SersicCore)
        b.centre.centre_0  = af.GaussianPrior(mean=0.0, sigma=0.3)
        b.centre.centre_1  = af.GaussianPrior(mean=0.0, sigma=0.3)
        b.ell_comps.ell_comps_0 = af.TruncatedGaussianPrior(
            mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)
        b.ell_comps.ell_comps_1 = af.TruncatedGaussianPrior(
            mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)
        b.intensity        = af.LogUniformPrior(lower_limit=1e-3, upper_limit=1e2)
        b.effective_radius = af.UniformPrior(lower_limit=0.02, upper_limit=0.5)
        b.sersic_index     = af.UniformPrior(lower_limit=0.8, upper_limit=4.0)
        return af.Model(al.Galaxy, redshift=z, bulge=b)

    source_1 = _source_model(1.0)
    source_2 = _source_model(2.5)

    model = af.Collection(galaxies=af.Collection(
        lens=lens, source_1=source_1, source_2=source_2))
    print(f"[DSPL/direct] total free parameters: {model.total_free_parameters}",
          flush=True)

    analysis = al.AnalysisImaging(dataset=dataset, use_jax=False)

    search = af.Nautilus(
        path_prefix           = output_root / "double_source_plane",
        name                  = "dspl_direct_fit",
        unique_tag            = "mock_1",
        n_live                = n_live,
        n_batch               = 50,
        iterations_per_update = 30000,
        number_of_cores       = int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
    )

    print("[DSPL/direct] Nautilus starting...", flush=True)
    t0 = time.time()
    result = search.fit(model=model, analysis=analysis)
    print(f"[DSPL/direct] done in {(time.time()-t0)/60:.1f} min", flush=True)
    _force_visualize(analysis, result, tag="direct")
    print(result.info, flush=True)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--part",         choices=("direct",), default="direct")
    p.add_argument("--repo-root",    type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--output-root",  type=Path, required=True)
    p.add_argument("--n-live",       type=int, default=200)
    args = p.parse_args()

    sys.path.insert(0, str(args.repo_root))
    dataset = load_dataset(args.dataset_root, mask_radius=2.8)

    if args.part == "direct":
        build_direct_fit(dataset, args.output_root, n_live=args.n_live)


if __name__ == "__main__":
    main()
