"""
fit_example_quad_time_delay.py — Standalone Python for Examples/quad_time_delay/.

Fits the simulated quad-imaged quasar (z_S=2.0) lensed by a single PowerLaw
+ ExternalShear galaxy (z_L=0.5) using `al.AnalysisPoint`. Two parts:

    --part direct          Lens parameters fit; cosmology fixed at H0=70.
                           ~9 free params, n_live=100. Recovers all
                           mass+source params from positions+delays alone.

    --part direct_h0_free  Same model + free `cosmology.H0` (uniform 40-120).
                           ~10 free params, n_live=150. Recovers H0 to a
                           few percent. The cosmographic punchline.

    --part all             Both, sequentially.

Usage (Cannon):
    python fit_example_quad_time_delay.py --part all \\
        --dataset-root /path/to/Examples/quad_time_delay/mocks \\
        --output-root  /path/to/output

Point-source fits are MUCH cheaper than imaging — expect 10-30 min/part on
32 cores instead of hours. JAX is supported (use_jax=True default), but the
parameter dimensionality is so low that GPU speedup is modest.

Nautilus auto-resumes from any existing checkpoint.hdf5.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


def _force_visualize(analysis, result, tag: str = ""):
    """Re-emit the standard visualisations after a resumed search."""
    try:
        analysis.visualize(
            paths=result.paths,
            instance=result.max_log_likelihood_instance,
            during_analysis=False,
        )
    except Exception as e:
        print(f"[QTD]   warning: post-fit visualize {tag} failed: {e}",
              flush=True)


def load_dataset(dataset_root: Path):
    """Load the al.PointDataset from JSON."""
    import autolens as al

    return al.from_json(file_path=dataset_root / "point_dataset.json")


def build_solver(use_jax: bool = False):
    """Construct the standard PointSolver used for both fits."""
    import autolens as al

    grid = al.Grid2D.uniform(shape_native=(150, 150), pixel_scales=0.04)
    if use_jax:
        try:
            import jax.numpy as jnp
            return al.PointSolver.for_grid(
                grid=grid, pixel_scale_precision=0.001,
                magnification_threshold=0.1, xp=jnp,
            )
        except Exception as e:
            print(f"[QTD]   JAX unavailable ({e}); falling back to NumPy solver",
                  flush=True)
    return al.PointSolver.for_grid(
        grid=grid, pixel_scale_precision=0.001, magnification_threshold=0.1,
    )


def build_direct(dataset, output_root: Path, n_live: int = 100,
                 use_jax: bool = False):
    """Phase 1 — fit lens parameters with cosmology fixed at H0=70."""
    import autofit as af
    import autolens as al

    print("\n[QTD/direct] Phase 1: lens fit, cosmology fixed (H0=70).",
          flush=True)

    cosmology = al.cosmo.FlatLambdaCDM(H0=70.0, Om0=0.30)

    mass = af.Model(al.mp.PowerLaw)
    mass.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.1)
    mass.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.1)
    mass.einstein_radius = af.UniformPrior(lower_limit=0.5, upper_limit=2.5)
    mass.slope           = af.TruncatedGaussianPrior(
        mean=2.0, sigma=0.2, lower_limit=1.5, upper_limit=2.5)
    mass.ell_comps.ell_comps_0 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)
    mass.ell_comps.ell_comps_1 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)

    shear = af.Model(al.mp.ExternalShear)
    shear.gamma_1 = af.GaussianPrior(mean=0.0, sigma=0.1)
    shear.gamma_2 = af.GaussianPrior(mean=0.0, sigma=0.1)

    lens = af.Model(al.Galaxy, redshift=0.5, mass=mass, shear=shear)

    point_0 = af.Model(al.ps.Point)
    point_0.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.3)
    point_0.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.3)
    source = af.Model(al.Galaxy, redshift=2.0, point_0=point_0)

    model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

    solver = build_solver(use_jax=use_jax)
    analysis = al.AnalysisPoint(
        dataset=dataset, solver=solver, cosmology=cosmology, use_jax=use_jax,
    )

    search = af.Nautilus(
        path_prefix=output_root,
        name="quad_direct_fit",
        unique_tag="phase_1_cosmology_fixed",
        n_live=n_live,
        n_batch=50,
        iterations_per_update=5000,
        number_of_cores=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
    )

    t0 = time.time()
    result = search.fit(model=model, analysis=analysis)
    print(f"[QTD/direct] done in {(time.time()-t0)/60:.1f} min", flush=True)
    _force_visualize(analysis, result, tag="direct")
    print(result.info, flush=True)
    return result


def build_direct_h0_free(dataset, output_root: Path, n_live: int = 150,
                         use_jax: bool = False):
    """Phase 2 — same model + free H0. Recovers cosmographic distance."""
    import autofit as af
    import autolens as al

    print("\n[QTD/direct_h0_free] Phase 2: H0 free, prior Uniform(40, 120).",
          flush=True)

    mass = af.Model(al.mp.PowerLaw)
    mass.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.1)
    mass.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.1)
    mass.einstein_radius = af.UniformPrior(lower_limit=0.5, upper_limit=2.5)
    mass.slope           = af.TruncatedGaussianPrior(
        mean=2.0, sigma=0.2, lower_limit=1.5, upper_limit=2.5)
    mass.ell_comps.ell_comps_0 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)
    mass.ell_comps.ell_comps_1 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)

    shear = af.Model(al.mp.ExternalShear)
    shear.gamma_1 = af.GaussianPrior(mean=0.0, sigma=0.1)
    shear.gamma_2 = af.GaussianPrior(mean=0.0, sigma=0.1)

    lens = af.Model(al.Galaxy, redshift=0.5, mass=mass, shear=shear)

    point_0 = af.Model(al.ps.Point)
    point_0.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.3)
    point_0.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.3)
    source = af.Model(al.Galaxy, redshift=2.0, point_0=point_0)

    cosmology = af.Model(al.cosmo.FlatLambdaCDM)
    cosmology.H0 = af.UniformPrior(lower_limit=40.0, upper_limit=120.0)
    cosmology.Om0 = 0.30

    model = af.Collection(
        galaxies=af.Collection(lens=lens, source=source),
        cosmology=cosmology,
    )

    solver = build_solver(use_jax=use_jax)
    analysis = al.AnalysisPoint(
        dataset=dataset, solver=solver, use_jax=use_jax,
    )

    search = af.Nautilus(
        path_prefix=output_root,
        name="quad_direct_fit",
        unique_tag="phase_2_h0_free",
        n_live=n_live,
        n_batch=50,
        iterations_per_update=5000,
        number_of_cores=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
    )

    t0 = time.time()
    result = search.fit(model=model, analysis=analysis)
    print(f"[QTD/direct_h0_free] done in {(time.time()-t0)/60:.1f} min",
          flush=True)
    _force_visualize(analysis, result, tag="direct_h0_free")
    print(result.info, flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--part",
                   choices=("direct", "direct_h0_free", "all"),
                   default="direct")
    p.add_argument("--output-root", type=Path,
                   default=Path("./output").resolve())
    p.add_argument("--dataset-root", type=Path, required=True,
                   help="Path containing point_dataset.json")
    p.add_argument("--repo-root", type=Path, required=False,
                   help="Path to Learning_to_Autolens (unused here, "
                        "kept for slurm-driver compatibility)")
    p.add_argument("--n-live", type=int, default=100,
                   help="n_live for the lens-only fit (Phase 1)")
    p.add_argument("--n-live-h0", type=int, default=150,
                   help="n_live for the H0-free fit (Phase 2)")
    p.add_argument("--use-jax", action="store_true",
                   help="Try the JAX PointSolver (falls back to NumPy if "
                        "JAX import fails)")
    args = p.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)

    import autolens as al
    print(f"PyAutoLens: {al.__version__}", flush=True)
    print(f"Output root:  {args.output_root}", flush=True)
    print(f"Dataset root: {args.dataset_root}", flush=True)
    print(f"Part:         {args.part}", flush=True)
    print(f"SLURM_JOB_ID: {os.environ.get('SLURM_JOB_ID', '(none)')}", flush=True)
    print(f"SLURM_CPUS_PER_TASK: "
          f"{os.environ.get('SLURM_CPUS_PER_TASK', '(none)')}",
          flush=True)

    t_start = time.time()
    dataset = load_dataset(args.dataset_root)
    print(f"Loaded PointDataset: {len(dataset.positions)} images",
          flush=True)

    if args.part in ("direct", "all"):
        build_direct(dataset, args.output_root,
                     n_live=args.n_live, use_jax=args.use_jax)

    if args.part in ("direct_h0_free", "all"):
        build_direct_h0_free(dataset, args.output_root,
                             n_live=args.n_live_h0, use_jax=args.use_jax)

    print(f"\nTotal wall time: {(time.time()-t_start)/3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
