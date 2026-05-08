"""
fit_example_quad_time_delay.py — Standalone Python for Examples/quad_time_delay/.

Fits the simulated quad-imaged quasar (z_S=2.0) lensed by a single PowerLaw
+ ExternalShear galaxy (z_L=0.5).

POINT-ONLY parts (consume `Examples/quad_time_delay/mocks/`):

    --part direct          Lens parameters fit; cosmology fixed at H0=70.
                           ~9 free params, n_live=100. Recovers all
                           mass+source params from positions+delays alone.

    --part direct_h0_free  Same model + free `cosmology.H0` (uniform 40-120).
                           ~10 free params, n_live=150. Recovers H0 to a
                           few percent. The cosmographic punchline.

    --part direct_h0_free_tight   Tighter prior + n_live=300 follow-up.
    --part positions_only         Strip time-delays; H0 should be unconstrained.
    --part all                    Phase 1 + Phase 2 sequentially.

JOINT parts (consume `Examples/quad_time_delay/mocks_with_host/`, which
provides BOTH a PointDataset and an Imaging FITS triplet):

    --part joint_fit              AnalysisPoint + AnalysisImaging on the
                                  shared lens model + shared source Galaxy
                                  (point_0 + bulge). Cosmology fixed.
                                  ~13 free params, n_live=200.

    --part joint_fit_h0_free      Same as above + free `cosmology.H0`.
                                  ~14 free params, n_live=250. The TDCOSMO
                                  IV / H0LiCOW XIII methodology — quasar
                                  positions+delays jointly with extended
                                  host arc constrain H0.

The joint analyses combine the two likelihoods via PyAutoFit's
`af.FactorGraphModel`: each analysis is wrapped in an `af.AnalysisFactor`
that pairs it with the same global model (a single Collection with a
shared lens + shared source Galaxy that carries BOTH the SersicCore bulge
and the ps.Point). The joint log-likelihood is the sum of both.

Usage (Cannon):
    python fit_example_quad_time_delay.py --part all \\
        --dataset-root /path/to/Examples/quad_time_delay/mocks \\
        --output-root  /path/to/output

    python fit_example_quad_time_delay.py --part joint_fit_h0_free \\
        --dataset-root /path/to/Examples/quad_time_delay/mocks_with_host \\
        --output-root  /path/to/output

Point-source fits are MUCH cheaper than imaging — expect 10-30 min/part on
32 cores. The joint parts include imaging convolution at every step and
are ~10-30x more expensive (target wall ~6-12h on 32 cores for the
joint_fit_h0_free).

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


def build_direct_h0_free_tight(dataset, output_root: Path, n_live: int = 300,
                               use_jax: bool = False):
    """Phase 3 — same as Phase 2 but with a tightened H0 prior and bumped
    n_live, targeting strict-PASS H0 recovery.

    Phase 2 (Uniform(40, 120) on H0, n_live=150) recovered median H0=92.7
    with truth=70 at the 2σ edge — borderline. Tightening the prior to
    Uniform(50, 100) (still 30 km/s/Mpc on each side of truth, far beyond
    any survey-tension consideration) and bumping n_live to 300 should
    pull the posterior down toward truth and tighten the 1σ band. Lens
    parameters were already within 1σ of truth in Phase 2, so this run is
    essentially a cosmography-only refinement.

    Wall: ~30-60 min on 32 cores (point-source fits are cheap).
    """
    import autofit as af
    import autolens as al

    print("\n[QTD/direct_h0_free_tight] Phase 3: H0 free, prior "
          "Uniform(50, 100), n_live=300.", flush=True)

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
    cosmology.H0 = af.UniformPrior(lower_limit=50.0, upper_limit=100.0)
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
        unique_tag="phase_3_h0_free_tight",
        n_live=n_live,
        n_batch=50,
        iterations_per_update=5000,
        number_of_cores=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
    )

    t0 = time.time()
    result = search.fit(model=model, analysis=analysis)
    print(f"[QTD/direct_h0_free_tight] done in {(time.time()-t0)/60:.1f} min",
          flush=True)
    _force_visualize(analysis, result, tag="direct_h0_free_tight")
    print(result.info, flush=True)
    return result


def build_positions_only(dataset, output_root: Path, n_live: int = 200,
                         use_jax: bool = False):
    """Phase 4 of the v0.95 PositionsLH research batch (Track B from
    plan).

    Strips time delays from the PointDataset and refits with positions +
    fluxes only. Cosmology FREE (Uniform(40, 120) on H0).

    The expected outcome is the *negative* result: H0 should be
    essentially **unconstrained** by positions + flux ratios alone — the
    posterior should match the prior. This validates Module 12's claim
    that *time delays carry the cosmography signal*, not multiple imaging
    per se. Compare against Phase 3's ~few-percent H0 recovery.

    Wall: ~30 min on 32 cores.
    """
    import autofit as af
    import autolens as al
    import time

    print("\n[QTD/positions_only] stripping time_delays from dataset…",
          flush=True)
    # Rebuild the PointDataset without time_delays.
    dataset_positions_only = al.PointDataset(
        name=dataset.name,
        positions=dataset.positions,
        positions_noise_map=dataset.positions_noise_map,
        fluxes=dataset.fluxes,
        fluxes_noise_map=dataset.fluxes_noise_map,
        # NOTE: time_delays + time_delays_noise_map deliberately omitted.
    )
    print(f"[QTD/positions_only] {len(dataset_positions_only.positions)} "
          f"image positions retained, time_delays dropped.", flush=True)

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
        dataset=dataset_positions_only, solver=solver, use_jax=use_jax,
    )

    search = af.Nautilus(
        path_prefix=output_root,
        name="quad_direct_fit",
        unique_tag="phase_4_positions_only",
        n_live=n_live,
        n_batch=50,
        iterations_per_update=5000,
        number_of_cores=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
    )

    t0 = time.time()
    result = search.fit(model=model, analysis=analysis)
    print(f"[QTD/positions_only] done in {(time.time()-t0)/60:.1f} min",
          flush=True)
    _force_visualize(analysis, result, tag="positions_only")
    print(result.info, flush=True)
    return result


def load_imaging(dataset_root: Path, mask_radius: float = 4.0):
    """Load the host-galaxy imaging FITS triplet + apply a circular mask.

    Used by the `joint_fit*` parts. Expects `image.fits`, `noise_map.fits`,
    `psf.fits` to live in `dataset_root` (alongside `point_dataset.json`),
    which is the layout produced by
    `Examples/quad_time_delay/mocks_with_host/generate_mock.py`.
    """
    import autolens as al

    pixel_scales = 0.05  # matches the mock_with_host generator
    imaging = al.Imaging.from_fits(
        data_path=dataset_root / "image.fits",
        noise_map_path=dataset_root / "noise_map.fits",
        psf_path=dataset_root / "psf.fits",
        pixel_scales=pixel_scales,
    )
    mask = al.Mask2D.circular(
        shape_native=imaging.shape_native,
        pixel_scales=imaging.pixel_scales,
        radius=mask_radius,
    )
    return imaging.apply_mask(mask=mask)


def _build_joint_lens_source(use_truth_centre_prior: bool = False):
    """Construct the shared (lens, source) `af.Model` Collection for the
    joint AnalysisPoint + AnalysisImaging fits.

    The source Galaxy holds BOTH:
      - `bulge`   : `lp.SersicCore`  → drives the imaging arc likelihood
      - `point_0` : `ps.Point`       → drives the point-source positions /
                                       delays / fluxes likelihood
    PyAutoLens routes them automatically: AnalysisImaging only renders
    LightProfiles, AnalysisPoint only consumes the Point profile.

    The bulge centre is shared with `point_0.centre` via the af.Model
    paramerization — see how the priors are tied below.
    """
    import autofit as af
    import autolens as al

    # --- Mass + shear (lens plane) ----------------------------------------
    mass = af.Model(al.mp.PowerLaw)
    mass.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.1)
    mass.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.1)
    mass.einstein_radius = af.UniformPrior(lower_limit=0.5, upper_limit=2.5)
    mass.slope = af.TruncatedGaussianPrior(
        mean=2.0, sigma=0.2, lower_limit=1.5, upper_limit=2.5
    )
    mass.ell_comps.ell_comps_0 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0
    )
    mass.ell_comps.ell_comps_1 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0
    )

    shear = af.Model(al.mp.ExternalShear)
    shear.gamma_1 = af.GaussianPrior(mean=0.0, sigma=0.1)
    shear.gamma_2 = af.GaussianPrior(mean=0.0, sigma=0.1)

    lens = af.Model(al.Galaxy, redshift=0.5, mass=mass, shear=shear)

    # --- Source: shared centre between point_0 and bulge ------------------
    # Construct bulge first; we then *force* point_0.centre = bulge.centre
    # so the joint fit doesn't drift the quasar centre and the host centre
    # apart (the quasar lives inside its host).
    bulge = af.Model(al.lp.SersicCore)
    bulge.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.3)
    bulge.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.3)
    bulge.ell_comps.ell_comps_0 = af.GaussianPrior(mean=0.0, sigma=0.3)
    bulge.ell_comps.ell_comps_1 = af.GaussianPrior(mean=0.0, sigma=0.3)
    bulge.intensity = af.LogUniformPrior(lower_limit=1e-3, upper_limit=10.0)
    bulge.effective_radius = af.UniformPrior(lower_limit=0.02, upper_limit=0.6)
    bulge.sersic_index = af.UniformPrior(lower_limit=0.5, upper_limit=4.0)
    # radius_break, gamma, alpha left at SersicCore defaults (fixed).

    point_0 = af.Model(al.ps.Point)
    # Tie the quasar centre to the host centre (canonical TDCOSMO setup —
    # the AGN sits at the photometric centre of its host).
    point_0.centre = bulge.centre

    source = af.Model(
        al.Galaxy, redshift=2.0, bulge=bulge, point_0=point_0
    )

    return lens, source


def build_joint_fit(dataset_point, dataset_imaging, output_root: Path,
                    n_live: int = 200, use_jax: bool = False,
                    h0_free: bool = False):
    """Joint AnalysisPoint + AnalysisImaging fit. Phase: H0 fixed (h0_free=False)
    or H0 free (h0_free=True). The joint likelihood is the SUM of both
    analyses' likelihoods, computed via `af.FactorGraphModel`.
    """
    import autofit as af
    import autolens as al

    tag = "joint_fit_h0_free" if h0_free else "joint_fit"
    print(f"\n[QTD/{tag}] Joint AnalysisPoint + AnalysisImaging "
          f"({'H0 free' if h0_free else 'H0=70 fixed'}).", flush=True)

    lens, source = _build_joint_lens_source()

    if h0_free:
        cosmology = af.Model(al.cosmo.FlatLambdaCDM)
        cosmology.H0 = af.UniformPrior(lower_limit=40.0, upper_limit=120.0)
        cosmology.Om0 = 0.30
        model = af.Collection(
            galaxies=af.Collection(lens=lens, source=source),
            cosmology=cosmology,
        )
        analysis_cosmology = None  # cosmology comes from the model
    else:
        cosmology_fixed = al.cosmo.FlatLambdaCDM(H0=70.0, Om0=0.30)
        model = af.Collection(galaxies=af.Collection(lens=lens, source=source))
        analysis_cosmology = cosmology_fixed

    # --- Analyses ---------------------------------------------------------
    solver = build_solver(use_jax=use_jax)
    analysis_point = al.AnalysisPoint(
        dataset=dataset_point, solver=solver,
        cosmology=analysis_cosmology,
        use_jax=use_jax,
    )
    analysis_imaging = al.AnalysisImaging(
        dataset=dataset_imaging,
        cosmology=analysis_cosmology,
        use_jax=use_jax,
    )

    # --- Combine via FactorGraphModel ------------------------------------
    # Each analysis gets its own AnalysisFactor wrapping the SAME global
    # model. FactorGraphModel sums their log-likelihoods. This is the
    # canonical pattern from
    #   autolens_workspace_latest/scripts/multi/features/imaging_and_interferometer/modeling.py
    # adapted from imaging+interferometer to imaging+point-source.
    af_point   = af.AnalysisFactor(prior_model=model, analysis=analysis_point,
                                   name="point")
    af_imaging = af.AnalysisFactor(prior_model=model, analysis=analysis_imaging,
                                   name="imaging")
    factor_graph = af.FactorGraphModel(af_point, af_imaging)

    print(f"[QTD/{tag}] Global model summary:", flush=True)
    print(factor_graph.global_prior_model.info, flush=True)

    search = af.Nautilus(
        path_prefix=output_root,
        name="quad_joint_fit",
        unique_tag=("phase_joint_h0_free" if h0_free else "phase_joint_fixed"),
        n_live=n_live,
        n_batch=50,
        iterations_per_update=5000,
        number_of_cores=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
    )

    t0 = time.time()
    result_list = search.fit(
        model=factor_graph.global_prior_model, analysis=factor_graph
    )
    print(f"[QTD/{tag}] done in {(time.time()-t0)/60:.1f} min", flush=True)

    # FactorGraphModel returns a list-like result (one per AnalysisFactor).
    # Visualise both.
    try:
        for i, r in enumerate(result_list):
            _force_visualize(
                [analysis_point, analysis_imaging][i], r,
                tag=f"{tag}_factor{i}",
            )
    except Exception as e:
        print(f"[QTD/{tag}] warning: post-fit visualize loop failed: {e}",
              flush=True)

    try:
        print(result_list[0].info, flush=True)
    except Exception:
        pass

    return result_list


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--part",
                   choices=("direct", "direct_h0_free",
                            "direct_h0_free_tight",
                            "positions_only",
                            "joint_fit", "joint_fit_h0_free",
                            "all"),
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
    p.add_argument("--n-live-joint", type=int, default=200,
                   help="n_live for joint_fit (cosmology fixed)")
    p.add_argument("--n-live-joint-h0", type=int, default=250,
                   help="n_live for joint_fit_h0_free")
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

    is_joint = args.part in ("joint_fit", "joint_fit_h0_free")

    if is_joint:
        # Joint parts need BOTH point + imaging from mocks_with_host/.
        dataset_point = load_dataset(args.dataset_root)
        print(f"Loaded PointDataset: {len(dataset_point.positions)} images",
              flush=True)
        dataset_imaging = load_imaging(args.dataset_root)
        print(f"Loaded Imaging: shape={dataset_imaging.shape_native}, "
              f"pixels in mask={dataset_imaging.mask.pixels_in_mask}",
              flush=True)
    else:
        dataset = load_dataset(args.dataset_root)
        print(f"Loaded PointDataset: {len(dataset.positions)} images",
              flush=True)

    if args.part in ("direct", "all"):
        build_direct(dataset, args.output_root,
                     n_live=args.n_live, use_jax=args.use_jax)

    if args.part in ("direct_h0_free", "all"):
        build_direct_h0_free(dataset, args.output_root,
                             n_live=args.n_live_h0, use_jax=args.use_jax)

    if args.part == "direct_h0_free_tight":
        build_direct_h0_free_tight(dataset, args.output_root,
                                   n_live=300, use_jax=args.use_jax)

    if args.part == "positions_only":
        build_positions_only(dataset, args.output_root,
                             n_live=200, use_jax=args.use_jax)

    if args.part == "joint_fit":
        build_joint_fit(dataset_point, dataset_imaging, args.output_root,
                        n_live=args.n_live_joint, use_jax=args.use_jax,
                        h0_free=False)

    if args.part == "joint_fit_h0_free":
        build_joint_fit(dataset_point, dataset_imaging, args.output_root,
                        n_live=args.n_live_joint_h0, use_jax=args.use_jax,
                        h0_free=True)

    print(f"\nTotal wall time: {(time.time()-t_start)/3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
