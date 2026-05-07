"""fit_example_double_source_plane.py — Standalone Cannon driver.

Fits the one-lens / two-source DSPL mock at
Examples/double_source_plane/mocks/. Native multi-plane via
`al.Tracer` with 3 redshift planes (z_L=0.5, z_S1=1.0, z_S2=2.5).

Parts:
    direct                Single free fit of 1 lens + 2 sources (~26 free params)
    beta_fixedcosmo       Tight truth-anchored, cosmology FIXED at FlatLambdaCDM(70, 0.30)
    beta_freecosmo_v3     Tight truth-anchored, cosmology FREE with TruncatedGaussian
                          on Om0/w0 (replaces the v0.93 beta_freecosmo which stalled
                          at f_live=1.0 due to Om0 ≤ 0 / extreme w0 sampling)
    beta_chain            Stage 1 = beta_fixedcosmo, Stage 2 = beta_freecosmo_v3 with
                          Stage 1 lens/source posteriors as priors. Recommended.

Usage (Cannon):
    sbatch --export=ALL,EXAMPLE=double_source_plane,FIT_EXTRA_ARGS=--part=beta_chain \
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

    # Priors seeded near the mock truth values (see mocks/mock_truth.json).
    # v1 attempt with fully uninformative priors (UniformPrior 0.5-3 on
    # einstein_radius, LogUniformPrior 1e-3 to 1e2 on all intensities) stuck
    # at f_live=1.0 for 2h without compression — a 28-D problem with 5
    # orders of magnitude of intensity prior is too wide for Nautilus at
    # n_live=200 to explore in useful time.

    # ---- Lens (z=0.5) — Sersic bulge + Isothermal mass + shear ----
    bulge = af.Model(al.lp.Sersic)
    mass  = af.Model(al.mp.Isothermal)
    shear = af.Model(al.mp.ExternalShear)

    mass.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.1)
    mass.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.1)
    bulge.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.1)
    bulge.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.1)

    bulge.ell_comps.ell_comps_0 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.2, lower_limit=-1.0, upper_limit=1.0)
    bulge.ell_comps.ell_comps_1 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.2, lower_limit=-1.0, upper_limit=1.0)
    # Truth: intensity=1.2, R_e=0.8, n=3.5
    bulge.intensity        = af.LogUniformPrior(lower_limit=0.1, upper_limit=10.0)
    bulge.effective_radius = af.TruncatedGaussianPrior(
        mean=0.8, sigma=0.3, lower_limit=0.1, upper_limit=3.0)
    bulge.sersic_index     = af.TruncatedGaussianPrior(
        mean=3.5, sigma=1.0, lower_limit=0.8, upper_limit=5.0)

    mass.ell_comps.ell_comps_0 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.2, lower_limit=-1.0, upper_limit=1.0)
    mass.ell_comps.ell_comps_1 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.2, lower_limit=-1.0, upper_limit=1.0)
    # Truth einstein_radius = 1.4, seed tightly
    mass.einstein_radius = af.TruncatedGaussianPrior(
        mean=1.4, sigma=0.2, lower_limit=0.5, upper_limit=3.0)

    shear.gamma_1 = af.GaussianPrior(mean=0.0, sigma=0.05)
    shear.gamma_2 = af.GaussianPrior(mean=0.0, sigma=0.05)

    lens = af.Model(al.Galaxy, redshift=0.5,
                    bulge=bulge, mass=mass, shear=shear)

    # ---- Source 1 (z=1.0) + Source 2 (z=2.5) ----
    # Truths from mocks/mock_truth.json:
    #   source_1 centre=(0.15, 0.0), intensity=3.0, R_e=0.07, n=1.3
    #   source_2 centre=(-0.1, 0.22), intensity=3.5, R_e=0.06, n=1.5
    # Seed each source's priors at its truth position — this is the key
    # difference vs v1. Two sources at wide, uninformed centres blow up
    # the burn-in cost for Nautilus.
    def _source_model(z, centre_truth, intensity_truth, re_truth, n_truth):
        b = af.Model(al.lp.SersicCore)
        b.centre.centre_0  = af.GaussianPrior(mean=centre_truth[0], sigma=0.15)
        b.centre.centre_1  = af.GaussianPrior(mean=centre_truth[1], sigma=0.15)
        b.ell_comps.ell_comps_0 = af.TruncatedGaussianPrior(
            mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)
        b.ell_comps.ell_comps_1 = af.TruncatedGaussianPrior(
            mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)
        b.intensity        = af.LogUniformPrior(lower_limit=intensity_truth*0.1,
                                                 upper_limit=intensity_truth*10)
        b.effective_radius = af.TruncatedGaussianPrior(
            mean=re_truth, sigma=0.03, lower_limit=0.02, upper_limit=0.3)
        b.sersic_index     = af.TruncatedGaussianPrior(
            mean=n_truth, sigma=0.5, lower_limit=0.8, upper_limit=4.0)
        return af.Model(al.Galaxy, redshift=z, bulge=b)

    source_1 = _source_model(1.0, centre_truth=(0.15, 0.0),
                              intensity_truth=3.0, re_truth=0.07, n_truth=1.3)
    source_2 = _source_model(2.5, centre_truth=(-0.1, 0.22),
                              intensity_truth=3.5, re_truth=0.06, n_truth=1.5)

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


def _truth_anchored_lens_sources(dataset_root: Path):
    """Build truth-anchored lens + 2 source af.Models from mock_truth.json.

    Shared helper for the three β-cosmography variants (fixedcosmo,
    freecosmo_v3, chain Stage 2). Returns (lens_model, source_1_model,
    source_2_model, redshifts_dict).
    """
    import autofit as af
    import autolens as al
    import json

    truth = json.loads((dataset_root / "mock_truth.json").read_text())
    z_l = truth["redshifts"]["lens"]
    z_s1 = truth["redshifts"]["source_1"]
    z_s2 = truth["redshifts"]["source_2"]
    lens_truth = truth["lens"]
    src1_truth = truth["source_1"]
    src2_truth = truth["source_2"]

    bulge = af.Model(al.lp.Sersic)
    bulge.centre.centre_0 = af.GaussianPrior(mean=lens_truth["bulge"]["centre"][0], sigma=0.05)
    bulge.centre.centre_1 = af.GaussianPrior(mean=lens_truth["bulge"]["centre"][1], sigma=0.05)
    bulge.intensity        = af.LogUniformPrior(lower_limit=1e-3, upper_limit=10.0)
    bulge.effective_radius = af.GaussianPrior(mean=lens_truth["bulge"]["effective_radius"], sigma=0.1)
    bulge.sersic_index     = af.GaussianPrior(mean=lens_truth["bulge"]["sersic_index"], sigma=0.3)
    bulge.ell_comps.ell_comps_0 = af.GaussianPrior(mean=lens_truth["bulge"]["ell_comps"][0], sigma=0.05)
    bulge.ell_comps.ell_comps_1 = af.GaussianPrior(mean=lens_truth["bulge"]["ell_comps"][1], sigma=0.05)

    mass = af.Model(al.mp.Isothermal)
    mass.centre.centre_0 = af.GaussianPrior(mean=lens_truth["mass"]["centre"][0], sigma=0.05)
    mass.centre.centre_1 = af.GaussianPrior(mean=lens_truth["mass"]["centre"][1], sigma=0.05)
    mass.einstein_radius = af.GaussianPrior(mean=lens_truth["mass"]["einstein_radius"], sigma=0.05)
    mass.ell_comps.ell_comps_0 = af.GaussianPrior(mean=lens_truth["mass"]["ell_comps"][0], sigma=0.05)
    mass.ell_comps.ell_comps_1 = af.GaussianPrior(mean=lens_truth["mass"]["ell_comps"][1], sigma=0.05)

    shear = af.Model(al.mp.ExternalShear)
    shear.gamma_1 = af.GaussianPrior(mean=lens_truth["shear"]["gamma_1"], sigma=0.02)
    shear.gamma_2 = af.GaussianPrior(mean=lens_truth["shear"]["gamma_2"], sigma=0.02)

    lens = af.Model(al.Galaxy, redshift=z_l, bulge=bulge, mass=mass, shear=shear)

    def _src(z, t):
        s = af.Model(al.lp.SersicCore)
        s.centre.centre_0 = af.GaussianPrior(mean=t["bulge"]["centre"][0], sigma=0.05)
        s.centre.centre_1 = af.GaussianPrior(mean=t["bulge"]["centre"][1], sigma=0.05)
        s.intensity        = af.LogUniformPrior(lower_limit=1e-3, upper_limit=10.0)
        s.effective_radius = af.GaussianPrior(mean=t["bulge"]["effective_radius"], sigma=0.02)
        s.sersic_index     = af.GaussianPrior(mean=t["bulge"]["sersic_index"], sigma=0.3)
        s.ell_comps.ell_comps_0 = af.GaussianPrior(mean=t["bulge"]["ell_comps"][0], sigma=0.1)
        s.ell_comps.ell_comps_1 = af.GaussianPrior(mean=t["bulge"]["ell_comps"][1], sigma=0.1)
        return af.Model(al.Galaxy, redshift=z, bulge=s)

    source_1 = _src(z_s1, src1_truth)
    source_2 = _src(z_s2, src2_truth)
    return lens, source_1, source_2, {"z_l": z_l, "z_s1": z_s1, "z_s2": z_s2}


def _bounded_freecosmo_model():
    """FlatwCDM cosmology af.Model with TruncatedGaussian priors on
    Om0/w0. The truncated bounds prevent the autolens FlatwCDM
    angular-diameter integrator from being asked about Om0 ≤ 0 or extreme
    phantom-DE w0 < -1.5 — exactly the family of inputs that crashed the
    integrator and produced the v0.93 Pattern A stall (task #110) and
    deadlocked the truth_fc resume (task #111).
    """
    import autofit as af
    import importlib.util
    here = Path(__file__).parent
    spec = importlib.util.spec_from_file_location(
        "climb_drv", here / "fit_example_compound_lens_zoo_climb.py")
    climb_drv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(climb_drv)
    FlatwCDMWrap = climb_drv.make_FlatwCDMWrap_class()
    cosmology = af.Model(FlatwCDMWrap)
    cosmology.H0 = 70.0
    cosmology.Om0 = af.TruncatedGaussianPrior(
        mean=0.30, sigma=0.10, lower_limit=0.05, upper_limit=0.60)
    cosmology.w0  = af.TruncatedGaussianPrior(
        mean=-1.0, sigma=0.20, lower_limit=-1.6, upper_limit=-0.4)
    cosmology.Tcmb0 = 2.7255
    cosmology.Neff  = 3.046
    cosmology.m_nu  = 0.0
    cosmology.Ob0   = 0.04897
    return cosmology


def build_beta_fixedcosmo_fit(dataset, output_root: Path, dataset_root: Path,
                              n_live: int = 200):
    """Stage 1 of the staged β-cosmography chain.

    Truth-anchored lens + sources, cosmology FIXED at FlatLambdaCDM(70, 0.30).
    Goal: nail down the lens / source posteriors before introducing the
    Om0/w0 nuisance dimensions in Stage 2. Wall: ~12-24h on 32 cores.

    Returns the af.NonLinearSearch result so Stage 2 can pass posteriors.
    """
    import autofit as af
    import autolens as al
    import time

    lens, source_1, source_2, _ = _truth_anchored_lens_sources(dataset_root)
    model = af.Collection(galaxies=af.Collection(
        lens=lens, source_1=source_1, source_2=source_2))
    print(f"[DSPL/beta_fixedcosmo] {model.total_free_parameters} free params "
          f"(no cosmology — fixed FlatLambdaCDM(70, 0.30))", flush=True)

    analysis = al.AnalysisImaging(dataset=dataset, use_jax=False)
    search = af.Nautilus(
        path_prefix=output_root / "double_source_plane",
        name="beta_fixedcosmo",
        unique_tag="mock_1_v0_94",
        n_live=n_live, n_batch=50, iterations_per_update=30000,
        number_of_cores=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
    )
    print("[DSPL/beta_fixedcosmo] Nautilus starting...", flush=True)
    t0 = time.time()
    result = search.fit(model=model, analysis=analysis)
    print(f"[DSPL/beta_fixedcosmo] done in {(time.time()-t0)/60:.1f} min, "
          f"log_Z={result.samples.log_evidence:.2f}", flush=True)
    _force_visualize(analysis, result, tag="beta_fixedcosmo")
    return result


def build_beta_freecosmo_v3_fit(dataset, output_root: Path, dataset_root: Path,
                                n_live: int = 200,
                                stage1_result=None):
    """β-cosmography fit on the DSPL system — v3 with TruncatedGaussian
    cosmology priors AND optional Stage 1 prior passing.

    Why "v3": v1 of the original `build_beta_freecosmo_fit` used uncapped
    GaussianPrior(0.30, 0.10) on Om0 and GaussianPrior(-1, 0.20) on w0. That
    sampled Om0 ≤ 0 and extreme phantom-DE w0 < -1.5, both of which crashed
    the autolens FlatwCDM integrator and produced -inf log-likelihood. The
    chain stalled at f_live=1.0 (task #110, Pattern A). v3 truncates both
    priors to physical regions.

    If `stage1_result` is supplied, the lens + source priors are taken from
    Stage 1 (build_beta_fixedcosmo_fit) posteriors via prior-passing — this
    is the Stage 2 of the recommended `--part=beta_chain` flow.

    Reads truth values from {dataset_root}/mock_truth.json.
    """
    import autofit as af
    import autolens as al
    import time

    if stage1_result is not None:
        # Stage 2: pass posteriors from Stage 1 as priors
        lens = stage1_result.model.galaxies.lens
        source_1 = stage1_result.model.galaxies.source_1
        source_2 = stage1_result.model.galaxies.source_2
        print(f"[DSPL/beta_freecosmo_v3] using Stage 1 posteriors as priors",
              flush=True)
    else:
        lens, source_1, source_2, _ = _truth_anchored_lens_sources(dataset_root)
        print(f"[DSPL/beta_freecosmo_v3] using truth-anchored priors (no Stage 1)",
              flush=True)

    cosmology = _bounded_freecosmo_model()

    model = af.Collection(
        galaxies=af.Collection(lens=lens, source_1=source_1, source_2=source_2),
        cosmology=cosmology,
    )
    print(f"[DSPL/beta_freecosmo_v3] {model.total_free_parameters} free params "
          f"(lens={lens.prior_count}, src1={source_1.prior_count}, "
          f"src2={source_2.prior_count}, cosmo={cosmology.prior_count})", flush=True)

    analysis = al.AnalysisImaging(dataset=dataset, use_jax=False)
    # Fresh unique_tag (v0_94) so this run does NOT pick up the deadlocked
    # 2026-05-01-era checkpoints under the original "mock_1" tag (task #111).
    tag = "mock_1_v0_94_chain" if stage1_result is not None else "mock_1_v0_94_standalone"
    search = af.Nautilus(
        path_prefix=output_root / "double_source_plane",
        name="beta_freecosmo_v3",
        unique_tag=tag,
        n_live=n_live, n_batch=50, iterations_per_update=30000,
        number_of_cores=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
    )
    print(f"[DSPL/beta_freecosmo_v3] Nautilus starting (unique_tag={tag})...",
          flush=True)
    t0 = time.time()
    result = search.fit(model=model, analysis=analysis)
    print(f"[DSPL/beta_freecosmo_v3] done in {(time.time()-t0)/60:.1f} min, "
          f"log_Z={result.samples.log_evidence:.2f}", flush=True)
    _force_visualize(analysis, result, tag="beta_freecosmo_v3")
    print(result.info, flush=True)
    return result


def build_beta_chain(dataset, output_root: Path, dataset_root: Path,
                     n_live: int = 200):
    """Recommended β-cosmography flow — Stage 1 fixedcosmo, Stage 2
    freecosmo_v3 with prior passing.

    Stage 1 (~12-24h): nail down the lens / source posteriors at fixed
    FlatLambdaCDM(70, 0.30). Truth-anchored Gaussian priors keep the
    chain near the basin from the start.

    Stage 2 (~24-48h): swap in TruncatedGaussian priors on Om0/w0 and
    inherit lens/source priors from Stage 1 posteriors. The chain now
    has only 2 free dimensions (cosmography) on top of a tightly
    constrained nuisance manifold.

    Total: ~36-72h. Rationale for splitting: the v0.93 single-stage
    `beta_freecosmo` stalled at f_live=1.0 because the cosmology dim
    was wide AND the lens prior box was Gaussian (not yet narrow at
    the Stage 1 posterior), so cosmology absorbed lens-model misfit.
    Splitting forces the cosmology dimension to face only what the
    lens/source can't already absorb.
    """
    print("[DSPL/beta_chain] Stage 1 — fixedcosmo", flush=True)
    stage1 = build_beta_fixedcosmo_fit(
        dataset, output_root, dataset_root, n_live=n_live)
    print("[DSPL/beta_chain] Stage 2 — freecosmo_v3 with Stage 1 priors",
          flush=True)
    stage2 = build_beta_freecosmo_v3_fit(
        dataset, output_root, dataset_root, n_live=n_live,
        stage1_result=stage1)
    return stage1, stage2


def build_beta_freecosmo_fit(dataset, output_root: Path, dataset_root: Path,
                             n_live: int = 200):
    """Backward-compat shim — redirects to the v3 (TruncatedGaussian) fit.

    Pre-v0.94 callers (sbatch scripts that pass `--part=beta_freecosmo`)
    transparently get the safe v3 priors.
    """
    print("[DSPL/beta_freecosmo] redirecting to v3 (TruncatedGaussian "
          "cosmology priors). v0.94 task #110 fix.", flush=True)
    return build_beta_freecosmo_v3_fit(
        dataset, output_root, dataset_root, n_live=n_live)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--part",
        choices=("direct", "beta_fixedcosmo", "beta_freecosmo_v3",
                 "beta_chain", "beta_freecosmo"),
        default="direct",
        help="direct: 1 lens + 2 sources free fit (~26 free params). "
             "beta_fixedcosmo: Stage 1 of chain — truth-anchored, "
             "cosmology fixed. "
             "beta_freecosmo_v3: free Om0/w0 with TruncatedGaussian "
             "priors (no Stage 1). "
             "beta_chain: Stage 1 -> Stage 2 with prior passing — "
             "RECOMMENDED. "
             "beta_freecosmo: backward-compat shim, redirects to v3.",
    )
    p.add_argument("--repo-root",    type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--output-root",  type=Path, required=True)
    p.add_argument("--n-live",       type=int, default=200)
    args = p.parse_args()

    sys.path.insert(0, str(args.repo_root))
    dataset = load_dataset(args.dataset_root, mask_radius=2.8)

    if args.part == "direct":
        build_direct_fit(dataset, args.output_root, n_live=args.n_live)
    elif args.part == "beta_fixedcosmo":
        build_beta_fixedcosmo_fit(dataset, args.output_root, args.dataset_root,
                                  n_live=args.n_live)
    elif args.part == "beta_freecosmo_v3":
        build_beta_freecosmo_v3_fit(dataset, args.output_root, args.dataset_root,
                                    n_live=args.n_live)
    elif args.part == "beta_chain":
        build_beta_chain(dataset, args.output_root, args.dataset_root,
                         n_live=args.n_live)
    elif args.part == "beta_freecosmo":
        # Backward-compat shim
        build_beta_freecosmo_fit(dataset, args.output_root, args.dataset_root,
                                 n_live=args.n_live)


if __name__ == "__main__":
    main()
