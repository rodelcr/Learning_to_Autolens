"""
fit_example_mge_to_physical.py — Cannon driver for Examples/mge_to_physical/.

Three-search chain on the lenstronomy mock_1 dataset (compound EPL+EPL with
a cuspy n=4.9 lens light, see mocks/PROVENANCE.md):

    --part light        Search 1 — fit lens light only with MGE basis
                        (2 x 30 Gaussians, no mass, no source). ~30 min.

    --part stars_only   Search 2 — take MGE shape, swap to lmp.Sersic, fit
                        single mass-to-light ratio + ExternalShear + source
                        (no dark matter). Priors from Search 1 via
                        `take_attributes`. ~1-2 h.

    --part stars_dark   Search 3 — same lmp.Sersic stars (priors from
                        Search 2) PLUS al.mp.NFW dark matter centred on
                        the bulge. Joint stars+DM fit. ~2-3 h.

    --part all          Sequential 1 -> 2 -> 3.

Mirrors autolens_workspace_latest/scripts/imaging/features/advanced/
mass_stellar_dark/chaining.py with adaptations for our compound mock:

    * Lens light is MGE rather than single Sersic (the truth is a cuspy
      n=4.9 Sersic; a single-Sersic light fit underperforms — see
      Exercise 3 in the README).
    * The secondary deflector at z=0.8 (theta_E=0.11") is absorbed into
      ExternalShear at first pass (Pattern E from project_fit_failure_patterns.md).
    * Source is a single SersicCore in the canonical fit; the truth is two
      Sersics — Exercise 2 adds the second back.
    * Mass-follows-light coupling: lmp.Sersic uses one mass-to-light
      ratio parameter shared by light and mass.

Usage (Cannon):
    python fit_example_mge_to_physical.py --part all \\
        --repo-root    /path/to/Learning_to_Autolens \\
        --dataset-root /path/to/Examples/mge_to_physical/mocks \\
        --output-root  /path/to/output

Nautilus auto-resumes from any existing checkpoint.hdf5.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


def _force_visualize(analysis, result, tag: str = ""):
    """Re-emit standard visualisations after a resumed search."""
    try:
        analysis.visualize(
            paths=result.paths,
            instance=result.max_log_likelihood_instance,
            during_analysis=False,
        )
    except Exception as e:
        print(f"[MGE]   warning: post-fit visualize {tag} failed: {e}",
              flush=True)


def load_dataset(dataset_root: Path, mask_radius: float = 2.7):
    """Load the lenstronomy_mock_1 imaging with a 2.7'' circular mask."""
    import autolens as al

    dataset = al.Imaging.from_fits(
        data_path      = dataset_root / "lenstronomy_mock_1_image.fits",
        noise_map_path = dataset_root / "lenstronomy_mock_1_noise.fits",
        psf_path       = dataset_root / "lenstronomy_mock_1_psf.fits",
        pixel_scales   = 0.05,
    )
    mask = al.Mask2D.circular(
        shape_native = dataset.shape_native,
        pixel_scales = dataset.pixel_scales,
        radius       = mask_radius,
    )

    # Build over_sample on the UNMASKED grid (this is what 2026.4 expects;
    # building it from `dataset.grid` after masking gives a slim-shape
    # mismatch because `dataset.grid` post-mask already returns the
    # masked grid in slim form, but the over_sample setter wants the
    # unmasked native shape).
    over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
        grid=dataset.grid, sub_size_list=[4, 2, 1],
        radial_list=[0.3, 0.6], centre_list=[(0.0, 0.0)],
    )
    return dataset.apply_over_sampling(
        over_sample_size_lp=over_sample_size,
    ).apply_mask(mask=mask)


def build_search_1_light(dataset, output_root: Path, mask_radius: float,
                         n_live: int = 100):
    """Search 1: MGE lens light only. No mass, no source."""
    import autofit as af
    import autolens as al

    print("\n[MGE/Search 1] Lens light only — MGE basis (2 x 30 Gaussians).",
          flush=True)

    bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius,
        total_gaussians=30,
        gaussian_per_basis=2,
        centre_prior_is_uniform=True,
    )
    lens = af.Model(al.Galaxy, redshift=0.5, bulge=bulge)
    model = af.Collection(galaxies=af.Collection(lens=lens))

    # Mask out the arc region so the MGE light fit isn't pulled by source flux.
    # We use a simple central-region-only mask via positions_likelihood-free
    # AnalysisImaging — the standard 2.7" mask with the arcs IN it is fine
    # because Nautilus + linear inversion will favor the smooth lens-light MGE
    # over fitting the arcs (MGE Gaussians can't reproduce the arc topology).
    analysis = al.AnalysisImaging(dataset=dataset, use_jax=False)

    search = af.Nautilus(
        path_prefix=output_root,
        name="search_1_light",
        unique_tag="mge_lens_light",
        n_live=n_live,
        n_batch=50,
        iterations_per_update=5000,
        number_of_cores=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
    )

    t0 = time.time()
    result = search.fit(model=model, analysis=analysis)
    print(f"[MGE/Search 1] done in {(time.time()-t0)/60:.1f} min", flush=True)
    _force_visualize(analysis, result, tag="search_1")
    print(result.info, flush=True)
    return result


def build_search_2_stars_only(dataset, output_root: Path, result_1=None,
                              n_live: int = 150):
    """Search 2: stars-only mass (MGE-light-as-mass via lmp.Sersic) + shear + source."""
    import autofit as af
    import autolens as al

    print("\n[MGE/Search 2] Stars-only mass (lmp.Sersic from MGE light) + shear + source.",
          flush=True)

    # Single Sersic light-AND-mass profile — the canonical autolens recipe.
    # See chaining.py:197-200 in mass_stellar_dark. The MGE light fit's
    # centre + ellipticity priors get passed via `take_attributes`.
    bulge = af.Model(al.lmp.Sersic)
    if result_1 is not None:
        bulge.take_attributes(source=result_1.model)

    shear = af.Model(al.mp.ExternalShear)
    shear.gamma_1 = af.GaussianPrior(mean=0.0, sigma=0.05)
    shear.gamma_2 = af.GaussianPrior(mean=0.0, sigma=0.05)

    lens = af.Model(al.Galaxy, redshift=0.5, bulge=bulge, shear=shear)

    source_bulge = af.Model(al.lp.SersicCore)
    source_bulge.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.5)
    source_bulge.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.5)
    source_bulge.ell_comps.ell_comps_0 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)
    source_bulge.ell_comps.ell_comps_1 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)
    source_bulge.intensity = af.LogUniformPrior(lower_limit=1e-3, upper_limit=1e3)
    source_bulge.effective_radius = af.UniformPrior(lower_limit=0.0, upper_limit=2.0)
    source_bulge.sersic_index = af.UniformPrior(lower_limit=0.5, upper_limit=4.0)

    source = af.Model(al.Galaxy, redshift=1.7, bulge=source_bulge)

    model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

    analysis = al.AnalysisImaging(dataset=dataset, use_jax=False)

    search = af.Nautilus(
        path_prefix=output_root,
        name="search_2_stars_only",
        unique_tag="lmp_sersic_no_dark",
        n_live=n_live,
        n_batch=50,
        iterations_per_update=10000,
        number_of_cores=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
    )

    t0 = time.time()
    result = search.fit(model=model, analysis=analysis)
    print(f"[MGE/Search 2] done in {(time.time()-t0)/60:.1f} min", flush=True)
    _force_visualize(analysis, result, tag="search_2")
    print(result.info, flush=True)
    return result


def build_search_3_stars_dark(dataset, output_root: Path, result_2=None,
                              n_live: int = 200):
    """Search 3: stars (lmp.Sersic) + NFW dark matter, joint fit."""
    import autofit as af
    import autolens as al

    print("\n[MGE/Search 3] Stars + NFW dark matter (joint).", flush=True)

    bulge = af.Model(al.lmp.Sersic)
    dark = af.Model(al.mp.NFW)
    bulge.centre = dark.centre  # align stars and DM at the same centre

    if result_2 is not None:
        bulge.take_attributes(source=result_2.model)
        # Don't pass dark-matter priors from Search 2 (no DM in Search 2);
        # initialise the NFW with broad astrophysical priors.

    dark.kappa_s = af.LogUniformPrior(lower_limit=1e-4, upper_limit=10.0)
    dark.scale_radius = af.UniformPrior(lower_limit=0.5, upper_limit=30.0)
    dark.ell_comps.ell_comps_0 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)
    dark.ell_comps.ell_comps_1 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)

    shear = af.Model(al.mp.ExternalShear)
    if result_2 is not None:
        shear = result_2.model.galaxies.lens.shear
    else:
        shear.gamma_1 = af.GaussianPrior(mean=0.0, sigma=0.05)
        shear.gamma_2 = af.GaussianPrior(mean=0.0, sigma=0.05)

    lens = af.Model(al.Galaxy, redshift=0.5,
                    bulge=bulge, dark=dark, shear=shear)

    if result_2 is not None:
        source = result_2.model.galaxies.source
    else:
        source_bulge = af.Model(al.lp.SersicCore)
        source_bulge.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.5)
        source_bulge.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.5)
        source_bulge.ell_comps.ell_comps_0 = af.TruncatedGaussianPrior(
            mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)
        source_bulge.ell_comps.ell_comps_1 = af.TruncatedGaussianPrior(
            mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)
        source_bulge.intensity = af.LogUniformPrior(lower_limit=1e-3, upper_limit=1e3)
        source_bulge.effective_radius = af.UniformPrior(lower_limit=0.0, upper_limit=2.0)
        source_bulge.sersic_index = af.UniformPrior(lower_limit=0.5, upper_limit=4.0)
        source = af.Model(al.Galaxy, redshift=1.7, bulge=source_bulge)

    model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

    analysis = al.AnalysisImaging(dataset=dataset, use_jax=False)

    search = af.Nautilus(
        path_prefix=output_root,
        name="search_3_stars_dark",
        unique_tag="lmp_sersic_plus_nfw",
        n_live=n_live,
        n_batch=50,
        iterations_per_update=10000,
        number_of_cores=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
    )

    t0 = time.time()
    result = search.fit(model=model, analysis=analysis)
    print(f"[MGE/Search 3] done in {(time.time()-t0)/60:.1f} min", flush=True)
    _force_visualize(analysis, result, tag="search_3")
    print(result.info, flush=True)
    return result


def _build_secondary_deflector():
    """Fixed-centre secondary EPL at z=0.8, theta_E free (truth=0.11").

    Truth params from Examples/mge_to_physical/mocks/truths.json:
      kwargs_lens[2] = EPL at center_x=-0.05, center_y=0.02 (lenstronomy
                        x,y convention) -> autolens centre=(0.02, -0.05)
      theta_E_truth=0.11, gamma_truth=2.0, e1_truth=0.2, e2_truth=-0.12
    Centre + ellipticity + slope held FIXED at truth (no priors); only
    theta_E is free with Uniform(0, 0.4) — Pattern E says it might
    collapse to 0 if the data don't need it, otherwise it'll find the
    truth around 0.11.
    """
    import autofit as af
    import autolens as al

    secondary = af.Model(al.mp.PowerLaw)
    secondary.centre.centre_0 = 0.02
    secondary.centre.centre_1 = -0.05
    secondary.einstein_radius = af.UniformPrior(lower_limit=0.0, upper_limit=0.4)
    secondary.slope = 2.0
    secondary.ell_comps.ell_comps_0 = 0.2
    secondary.ell_comps.ell_comps_1 = -0.12
    return af.Model(al.Galaxy, redshift=0.8, mass=secondary)


def _build_two_source_galaxy():
    """Source at z=1.7 with TWO Sersic components.

    Truth has two Sersic sources (truths.kwargs_source):
      0: center=(-0.05, 0.02), R_e=0.19, n=2.3
      1: center=( 0.30, 0.22), R_e=0.15, n=1.5
    autolens (y, x) convention: (0.02, -0.05) and (0.22, 0.30)
    Both fit with wide-but-truth-anchored priors.
    """
    import autofit as af
    import autolens as al

    def _sersic_at(prior_centre):
        b = af.Model(al.lp.SersicCore)
        b.centre.centre_0 = af.GaussianPrior(mean=prior_centre[0], sigma=0.2)
        b.centre.centre_1 = af.GaussianPrior(mean=prior_centre[1], sigma=0.2)
        b.ell_comps.ell_comps_0 = af.TruncatedGaussianPrior(
            mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)
        b.ell_comps.ell_comps_1 = af.TruncatedGaussianPrior(
            mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)
        b.intensity = af.LogUniformPrior(lower_limit=1e-3, upper_limit=1e3)
        b.effective_radius = af.UniformPrior(lower_limit=0.05, upper_limit=0.5)
        b.sersic_index = af.UniformPrior(lower_limit=0.5, upper_limit=4.0)
        return b

    src_bulge = _sersic_at((0.02, -0.05))   # truth Source 0
    src_disk  = _sersic_at((0.22,  0.30))   # truth Source 1
    return af.Model(al.Galaxy, redshift=1.7, bulge=src_bulge, disk=src_disk)


def build_search_2_v2_stars_only(dataset, output_root: Path, result_1=None,
                                 n_live: int = 200):
    """Search 2 v2 — stars-only + secondary deflector + TWO sources.

    Same as Search 2 but:
      - Adds fixed-centre EPL secondary at z=0.8 with theta_E free.
      - Source has TWO Sersic components (matches truth structure).
    Goal: bring the canonical chi²/N from ~3 down to ~1 by removing the
    model misspecification documented in mge_to_physical/README.md
    §Caveats.
    """
    import autofit as af
    import autolens as al

    print("\n[MGE/Search 2 v2] +secondary deflector at z=0.8 +two sources.",
          flush=True)

    bulge = af.Model(al.lmp.Sersic)
    if result_1 is not None:
        bulge.take_attributes(source=result_1.model)

    shear = af.Model(al.mp.ExternalShear)
    shear.gamma_1 = af.GaussianPrior(mean=0.0, sigma=0.05)
    shear.gamma_2 = af.GaussianPrior(mean=0.0, sigma=0.05)

    lens = af.Model(al.Galaxy, redshift=0.5, bulge=bulge, shear=shear)
    lens_2 = _build_secondary_deflector()
    source = _build_two_source_galaxy()

    model = af.Collection(galaxies=af.Collection(
        lens=lens, lens_2=lens_2, source=source,
    ))
    print(f"Search 2 v2 priors: {model.prior_count}", flush=True)

    analysis = al.AnalysisImaging(dataset=dataset, use_jax=False)

    search = af.Nautilus(
        path_prefix=output_root,
        name="search_2_v2_stars_only",
        unique_tag="lmp_sersic_2src_secondary_no_dark",
        n_live=n_live,
        n_batch=50,
        iterations_per_update=10000,
        number_of_cores=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
    )

    t0 = time.time()
    result = search.fit(model=model, analysis=analysis)
    print(f"[MGE/Search 2 v2] done in {(time.time()-t0)/60:.1f} min",
          flush=True)
    _force_visualize(analysis, result, tag="search_2_v2")
    print(result.info, flush=True)
    return result


def build_search_3_v2_stars_dark(dataset, output_root: Path, result_2=None,
                                 n_live: int = 250):
    """Search 3 v2 — stars + NFW dark + secondary deflector + TWO sources."""
    import autofit as af
    import autolens as al

    print("\n[MGE/Search 3 v2] Stars+NFW + secondary deflector + two sources.",
          flush=True)

    bulge = af.Model(al.lmp.Sersic)
    dark = af.Model(al.mp.NFW)
    bulge.centre = dark.centre

    if result_2 is not None:
        bulge.take_attributes(source=result_2.model)

    dark.kappa_s = af.LogUniformPrior(lower_limit=1e-4, upper_limit=10.0)
    dark.scale_radius = af.UniformPrior(lower_limit=0.5, upper_limit=30.0)
    dark.ell_comps.ell_comps_0 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)
    dark.ell_comps.ell_comps_1 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)

    if result_2 is not None:
        shear = result_2.model.galaxies.lens.shear
        source = result_2.model.galaxies.source
        lens_2 = result_2.model.galaxies.lens_2
    else:
        shear = af.Model(al.mp.ExternalShear)
        shear.gamma_1 = af.GaussianPrior(mean=0.0, sigma=0.05)
        shear.gamma_2 = af.GaussianPrior(mean=0.0, sigma=0.05)
        source = _build_two_source_galaxy()
        lens_2 = _build_secondary_deflector()

    lens = af.Model(al.Galaxy, redshift=0.5,
                    bulge=bulge, dark=dark, shear=shear)

    model = af.Collection(galaxies=af.Collection(
        lens=lens, lens_2=lens_2, source=source,
    ))
    print(f"Search 3 v2 priors: {model.prior_count}", flush=True)

    analysis = al.AnalysisImaging(dataset=dataset, use_jax=False)

    search = af.Nautilus(
        path_prefix=output_root,
        name="search_3_v2_stars_dark",
        unique_tag="lmp_sersic_2src_secondary_plus_nfw",
        n_live=n_live,
        n_batch=50,
        iterations_per_update=10000,
        number_of_cores=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
    )

    t0 = time.time()
    result = search.fit(model=model, analysis=analysis)
    print(f"[MGE/Search 3 v2] done in {(time.time()-t0)/60:.1f} min",
          flush=True)
    _force_visualize(analysis, result, tag="search_3_v2")
    print(result.info, flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--part",
                   choices=("light", "stars_only", "stars_dark",
                            "stars_only_v2", "stars_dark_v2",
                            "all", "all_v2"),
                   default="all")
    p.add_argument("--output-root", type=Path,
                   default=Path("./output").resolve())
    p.add_argument("--dataset-root", type=Path, required=True,
                   help="Path containing lenstronomy_mock_1_*.fits")
    p.add_argument("--repo-root", type=Path, required=False,
                   help="Path to Learning_to_Autolens "
                        "(unused; kept for slurm-driver compatibility)")
    p.add_argument("--mask-radius", type=float, default=2.7)
    p.add_argument("--n-live-light", type=int, default=100)
    p.add_argument("--n-live-stars", type=int, default=150)
    p.add_argument("--n-live-stars-dark", type=int, default=200)
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
    dataset = load_dataset(args.dataset_root, mask_radius=args.mask_radius)
    print(f"Loaded dataset: {dataset.shape_native}, "
          f"{dataset.mask.pixels_in_mask} masked pixels",
          flush=True)

    result_1 = result_2 = result_2_v2 = None
    if args.part in ("light", "all", "all_v2"):
        result_1 = build_search_1_light(
            dataset, args.output_root, mask_radius=args.mask_radius,
            n_live=args.n_live_light,
        )

    if args.part in ("stars_only", "all"):
        result_2 = build_search_2_stars_only(
            dataset, args.output_root, result_1=result_1,
            n_live=args.n_live_stars,
        )

    if args.part in ("stars_dark", "all"):
        build_search_3_stars_dark(
            dataset, args.output_root, result_2=result_2,
            n_live=args.n_live_stars_dark,
        )

    if args.part in ("stars_only_v2", "all_v2"):
        result_2_v2 = build_search_2_v2_stars_only(
            dataset, args.output_root, result_1=result_1,
            n_live=200,
        )

    if args.part in ("stars_dark_v2", "all_v2"):
        build_search_3_v2_stars_dark(
            dataset, args.output_root, result_2=result_2_v2,
            n_live=250,
        )

    print(f"\nTotal wall time: {(time.time()-t_start)/3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
