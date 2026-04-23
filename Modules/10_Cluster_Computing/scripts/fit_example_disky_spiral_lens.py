"""fit_example_disky_spiral_lens.py — Cannon driver for Examples/disky_spiral_lens/.

The mock has a lens with a bulge (PA≈0°) + a disk (PA≈35°) — two light
components at different orientations that a single Sersic cannot capture.
This driver runs two parts to make the pedagogical comparison concrete:

Part 1: single_sersic — Sersic + Isothermal mass + shear + SersicCore source.
         Will leave coherent residuals at the lens centre.
Part 2: bulge_disk    — Sersic bulge + Sersic disk (different PA) + Isothermal
         + shear + SersicCore source. Should subtract the lens light cleanly.

Comparing log_Z between the two is the quantitative version of "look at the
lens-light-subtracted panel": if Part 2 log_Z >> Part 1, the extra 7 disk
parameters are earning their freedom.

Usage (Cannon):
    sbatch --export=ALL,EXAMPLE=disky_spiral_lens,FIT_EXTRA_ARGS=--part=all \
        submit_cannon.slurm

Each part takes ~1-2 h on 32 cores. `all` runs both sequentially.
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
        print(f"[DISKY] warning: post-fit visualize {tag} failed: {e}",
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


def _shared_mass_shear():
    """Mass + shear priors shared between both fits."""
    import autofit as af
    import autolens as al
    mass  = af.Model(al.mp.Isothermal)
    shear = af.Model(al.mp.ExternalShear)
    mass.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.1)
    mass.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.1)
    mass.ell_comps.ell_comps_0 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)
    mass.ell_comps.ell_comps_1 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)
    mass.einstein_radius = af.UniformPrior(lower_limit=0.5, upper_limit=3.0)
    shear.gamma_1 = af.GaussianPrior(mean=0.0, sigma=0.1)
    shear.gamma_2 = af.GaussianPrior(mean=0.0, sigma=0.1)
    return mass, shear


def _shared_source():
    import autofit as af
    import autolens as al
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
    return af.Model(al.Galaxy, redshift=1.6, bulge=b)


def build_single_sersic(dataset, output_root: Path, n_live: int = 200):
    """Single-Sersic lens light — expected to leave coherent residuals."""
    import autofit as af
    import autolens as al

    bulge = af.Model(al.lp.Sersic)
    bulge.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.1)
    bulge.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.1)
    bulge.ell_comps.ell_comps_0 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)
    bulge.ell_comps.ell_comps_1 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)
    bulge.intensity        = af.LogUniformPrior(lower_limit=1e-3, upper_limit=1e2)
    bulge.effective_radius = af.UniformPrior(lower_limit=0.1, upper_limit=3.0)
    bulge.sersic_index     = af.UniformPrior(lower_limit=0.8, upper_limit=5.0)

    mass, shear = _shared_mass_shear()
    lens = af.Model(al.Galaxy, redshift=0.45,
                    bulge=bulge, mass=mass, shear=shear)
    source = _shared_source()
    model = af.Collection(galaxies=af.Collection(lens=lens, source=source))
    print(f"[DISKY/single] total free parameters: {model.total_free_parameters}",
          flush=True)

    analysis = al.AnalysisImaging(dataset=dataset, use_jax=False)
    search = af.Nautilus(
        path_prefix           = output_root / "disky_spiral_lens",
        name                  = "single_sersic_fit",
        unique_tag            = "mock_1",
        n_live                = n_live,
        n_batch               = 50,
        iterations_per_update = 30000,
        number_of_cores       = int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
    )
    print("[DISKY/single] Nautilus starting...", flush=True)
    t0 = time.time()
    result = search.fit(model=model, analysis=analysis)
    print(f"[DISKY/single] done in {(time.time()-t0)/60:.1f} min", flush=True)
    _force_visualize(analysis, result, tag="single")
    print(result.info, flush=True)
    return result


def build_bulge_disk(dataset, output_root: Path, n_live: int = 200):
    """Two-component lens light: bulge (high-n) + disk (low-n, different PA)."""
    import autofit as af
    import autolens as al

    bulge = af.Model(al.lp.Sersic)
    disk  = af.Model(al.lp.Sersic)

    # Both centred at the galaxy centre (tied)
    bulge.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.1)
    bulge.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.1)
    disk.centre.centre_0  = bulge.centre.centre_0
    disk.centre.centre_1  = bulge.centre.centre_1

    # Bulge: compact, high n, roundish
    bulge.ell_comps.ell_comps_0 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)
    bulge.ell_comps.ell_comps_1 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)
    bulge.intensity        = af.LogUniformPrior(lower_limit=1e-3, upper_limit=1e2)
    bulge.effective_radius = af.UniformPrior(lower_limit=0.1, upper_limit=1.5)
    bulge.sersic_index     = af.UniformPrior(lower_limit=2.0, upper_limit=6.0)

    # Disk: larger, n ≈ 1, INDEPENDENT ell_comps (different PA)
    disk.ell_comps.ell_comps_0 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)
    disk.ell_comps.ell_comps_1 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)
    disk.intensity        = af.LogUniformPrior(lower_limit=1e-3, upper_limit=1e2)
    disk.effective_radius = af.UniformPrior(lower_limit=0.3, upper_limit=3.0)
    disk.sersic_index     = af.UniformPrior(lower_limit=0.5, upper_limit=2.0)

    mass, shear = _shared_mass_shear()
    lens = af.Model(al.Galaxy, redshift=0.45,
                    bulge=bulge, disk=disk, mass=mass, shear=shear)
    source = _shared_source()
    model = af.Collection(galaxies=af.Collection(lens=lens, source=source))
    print(f"[DISKY/bulge_disk] total free parameters: {model.total_free_parameters}",
          flush=True)

    analysis = al.AnalysisImaging(dataset=dataset, use_jax=False)
    search = af.Nautilus(
        path_prefix           = output_root / "disky_spiral_lens",
        name                  = "bulge_disk_fit",
        unique_tag            = "mock_1",
        n_live                = n_live,
        n_batch               = 50,
        iterations_per_update = 30000,
        number_of_cores       = int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
    )
    print("[DISKY/bulge_disk] Nautilus starting...", flush=True)
    t0 = time.time()
    result = search.fit(model=model, analysis=analysis)
    print(f"[DISKY/bulge_disk] done in {(time.time()-t0)/60:.1f} min", flush=True)
    _force_visualize(analysis, result, tag="bulge_disk")
    print(result.info, flush=True)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--part",         choices=("single_sersic", "bulge_disk", "all"), default="all")
    p.add_argument("--repo-root",    type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--output-root",  type=Path, required=True)
    p.add_argument("--n-live",       type=int, default=200)
    args = p.parse_args()

    sys.path.insert(0, str(args.repo_root))
    dataset = load_dataset(args.dataset_root, mask_radius=2.8)

    if args.part in ("single_sersic", "all"):
        build_single_sersic(dataset, args.output_root, n_live=args.n_live)
    if args.part in ("bulge_disk", "all"):
        build_bulge_disk(dataset, args.output_root, n_live=args.n_live)


if __name__ == "__main__":
    main()
