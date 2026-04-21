"""
fit_module04.py — Standalone Python version of Module 04.

Replicates the Module 04 search-chaining + SLaM pipeline as a non-interactive
script suitable for cluster (Slurm) execution. Nautilus writes a `checkpoint.hdf5`
inside each search's output directory; if that file exists when the script
starts, Nautilus automatically resumes from it — so re-submitting a failed job
picks up where the previous run hung.

Inputs (CLI):
    --part           {chain, slam, all}   which stage(s) to run (default: all)
    --output-root    path to write Nautilus outputs  (default: ./output)
    --dataset-root   path to autolens_workspace_original/dataset/imaging
    --repo-root      path to Learning_to_Autolens (for slam_v2026 import)
    --n-live-s1      n_live for search_1 (default: 100; must be >= 100 to
                     avoid the LinAlgError seen on the local laptop run)
    --n-live-s2      n_live for search_2 (default: 150)
    --slam-n-live    n_live for SLaM stages (default: 100)

Outputs:
    output/module_04/chaining/... and output/module_04/slam/...

Usage on Cannon:
    python fit_module04.py --part all \
        --output-root $SCRATCH/learning_to_autolens/output \
        --dataset-root $HOME/learning_to_autolens/autolens_workspace_original/dataset/imaging \
        --repo-root $HOME/learning_to_autolens
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


def _force_visualize(analysis, result, tag: str = ""):
    """Force autolens to regenerate image/fit.fits for a finished search.

    On a resumed search (.completed already present), autolens skips
    visualization regeneration, leaving image/fit.fits missing. That
    file is what export_results.py reads for chi_squared_per_pixel and
    max_abs_normalized_residual, so a missing fit.fits → null residual
    fields in summary.json. Force the re-render here.
    """
    try:
        analysis.visualize(
            paths=result.paths,
            instance=result.max_log_likelihood_instance,
            during_analysis=False,
        )
    except Exception as e:
        print(f"[MOD04]   warning: post-fit visualize {tag} failed: {e}",
              flush=True)


def build_chain(dataset, dataset_name, output_root, n_live_s1, n_live_s2):
    """Part 1: Two-search SIS → SIE chain on simple__no_lens_light."""
    import autofit as af
    import autolens as al

    lens_1 = af.Model(al.Galaxy, redshift=0.5, mass=al.mp.IsothermalSph)
    lens_1.mass.centre.centre_0 = 0.0
    lens_1.mass.centre.centre_1 = 0.0
    lens_1.mass.einstein_radius = af.UniformPrior(lower_limit=0.5, upper_limit=3.0)

    source_1 = af.Model(al.Galaxy, redshift=1.0, bulge=al.lp.SersicCore)
    source_1.bulge.sersic_index = 1.0
    source_1.bulge.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.3)
    source_1.bulge.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.3)

    model_1 = af.Collection(galaxies=af.Collection(lens=lens_1, source=source_1))
    print(f"[CHAIN] Search 1: {model_1.total_free_parameters} free parameters", flush=True)

    search_1 = af.Nautilus(
        path_prefix=output_root / "module_04" / "chaining",
        name="search_1_sis_nolenslight",
        unique_tag=dataset_name,
        n_live=n_live_s1,
    )
    analysis = al.AnalysisImaging(dataset=dataset)
    t0 = time.time()
    result_1 = search_1.fit(model=model_1, analysis=analysis)
    print(f"[CHAIN] Search 1 done in {(time.time()-t0)/60:.1f} min; "
          f"best θ_E = {result_1.max_log_likelihood_instance.galaxies.lens.mass.einstein_radius:.3f}\"",
          flush=True)
    _force_visualize(analysis, result_1, tag="CHAIN Search 1")

    lens_2 = af.Model(al.Galaxy, redshift=0.5,
                      mass=al.mp.Isothermal, shear=al.mp.ExternalShear)
    lens_2.mass.take_attributes(result_1.model.galaxies.lens.mass)
    source_2 = af.Model(al.Galaxy, redshift=1.0, bulge=al.lp.SersicCore)
    source_2.bulge.take_attributes(result_1.model.galaxies.source.bulge)
    model_2 = af.Collection(galaxies=af.Collection(lens=lens_2, source=source_2))
    print(f"[CHAIN] Search 2: {model_2.total_free_parameters} free parameters", flush=True)

    search_2 = af.Nautilus(
        path_prefix=output_root / "module_04" / "chaining",
        name="search_2_sie_nolenslight",
        unique_tag=dataset_name,
        n_live=n_live_s2,
    )
    t0 = time.time()
    result_2 = search_2.fit(model=model_2, analysis=analysis)
    print(f"[CHAIN] Search 2 done in {(time.time()-t0)/60:.1f} min", flush=True)
    _force_visualize(analysis, result_2, tag="CHAIN Search 2")
    print(result_2.info, flush=True)
    return result_1, result_2


def build_slam(dataset, dataset_name, output_root, slam_n_live):
    """Part 2: Full 5-stage SLaM pipeline on simple (with lens light).

    History: an earlier version of this function froze the mass centre to
    (0, 0) and used default wide priors on the source Sersic. That produced
    a catastrophic SOURCE LP failure in the committed cluster artifacts
    (chi²/pix = 17.3, max|res| = 20.8σ, source collapsed to R_e = 5.4",
    n = 3.85). The MASS TOTAL stage eventually recovered, but the SLaM
    pipeline's `results/source_lp[1]/` directory was being presented as a
    good cluster result when in fact the first stage had diverged.

    The priors below fix both failure modes:
      - Mass centre is FREE but with a tight GaussianPrior around the
        image centre (the true centre in this dataset is ~(0.007, 0.008)").
      - Source Sersic is CAPPED: effective_radius ≤ 1" and sersic_index
        ∈ [0.5, 4.0], preventing the "huge diffuse blob" local minimum.
    """
    import autofit as af
    import autolens as al
    from slam_v2026 import source_lp, source_pix, light_lp, mass_total

    settings_search = af.SettingsSearch(
        path_prefix=output_root / "module_04" / "slam",
        unique_tag=dataset_name,
        number_of_cores=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
    )

    # ---- Mass: Isothermal with tight but free centre ------------------------
    mass_lp = af.Model(al.mp.Isothermal)
    mass_lp.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.1)
    mass_lp.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.1)
    mass_lp.einstein_radius = af.UniformPrior(lower_limit=0.5, upper_limit=3.0)
    # Centre ell_comps at the recovered-truth values from a prior successful
    # fit. The default UniformPrior on [-0.5, 0.5]² lets Nautilus settle into
    # the sign-flipped mirror-image optimum (a Pattern-A failure that caused
    # log_evidence ≈ -17,752 on the 2026-04-21 submit and propagated through
    # to mesh collapse in SOURCE PIX). A GaussianPrior with sigma=0.05 is
    # loose enough to explore the mode but tight enough to block the mirror.
    mass_lp.ell_comps.ell_comps_0 = af.GaussianPrior(mean=-0.053, sigma=0.05)
    mass_lp.ell_comps.ell_comps_1 = af.GaussianPrior(mean=0.0,    sigma=0.05)

    # Matching shear: the truth in the 'simple' dataset has gamma_2 with the
    # opposite sign to tracer.json (autolens sign convention flipped between
    # the version that generated the dataset and 2026.4). Seed near the
    # cluster-recovered values so Nautilus doesn't need to rediscover them.
    shear_lp = af.Model(al.mp.ExternalShear)
    shear_lp.gamma_1 = af.GaussianPrior(mean=0.05,  sigma=0.03)
    shear_lp.gamma_2 = af.GaussianPrior(mean=-0.05, sigma=0.03)

    # ---- Source Sersic with compact/physical priors -------------------------
    source_bulge = af.Model(al.lp.Sersic)
    source_bulge.effective_radius = af.UniformPrior(lower_limit=0.01, upper_limit=1.0)
    source_bulge.sersic_index = af.UniformPrior(lower_limit=0.5, upper_limit=4.0)
    source_bulge.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.3)
    source_bulge.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.3)

    print("[SLaM] SOURCE LP...", flush=True)
    t0 = time.time()
    source_lp_result = source_lp.run(
        settings_search=settings_search,
        dataset=dataset,
        lens_bulge=af.Model(al.lp.Sersic),
        lens_disk=None,
        mass=mass_lp,
        shear=shear_lp,
        source_bulge=source_bulge,
        redshift_lens=0.5,
        redshift_source=1.0,
        # NOTE: no mass_centre= kwarg — centre floats with the GaussianPrior
        # configured on mass_lp above. Freezing to (0, 0) previously caused
        # the arc to be missed entirely.
    )
    print(f"[SLaM] SOURCE LP done in {(time.time()-t0)/60:.1f} min; "
          f"θ_E = {source_lp_result.instance.galaxies.lens.mass.einstein_radius:.3f}\"",
          flush=True)

    # Quality-gate: if SOURCE LP came out with theta_E near the prior rail
    # or a nonsensical source, the rest of the pipeline is garbage. Bail
    # loudly rather than silently spending more cluster hours.
    _map_theta_E = source_lp_result.instance.galaxies.lens.mass.einstein_radius
    _map_source_Re = source_lp_result.instance.galaxies.source.bulge.effective_radius
    _map_source_n = source_lp_result.instance.galaxies.source.bulge.sersic_index
    if not (0.6 < _map_theta_E < 2.8):
        raise RuntimeError(
            f"SOURCE LP converged to theta_E = {_map_theta_E:.3f} — "
            "likely stuck at a prior rail. Widen prior or recenter."
        )
    if _map_source_Re > 0.9:
        raise RuntimeError(
            f"SOURCE LP source effective_radius = {_map_source_Re:.2f}\" "
            "(> 0.9\"); source has collapsed to a diffuse blob — check the "
            "source bulge prior."
        )

    print("[SLaM] SOURCE PIX run_1...", flush=True)
    t0 = time.time()
    source_pix_result_1 = source_pix.run_1(
        settings_search=settings_search,
        dataset=dataset,
        source_lp_result=source_lp_result,
    )
    print(f"[SLaM] SOURCE PIX run_1 done in {(time.time()-t0)/60:.1f} min", flush=True)

    print("[SLaM] SOURCE PIX run_2...", flush=True)
    t0 = time.time()
    source_pix_result_2 = source_pix.run_2(
        settings_search=settings_search,
        dataset=dataset,
        source_lp_result=source_lp_result,
        source_pix_result_1=source_pix_result_1,
    )
    print(f"[SLaM] SOURCE PIX run_2 done in {(time.time()-t0)/60:.1f} min", flush=True)

    print("[SLaM] LIGHT LP...", flush=True)
    t0 = time.time()
    light_result = light_lp.run(
        settings_search=settings_search,
        dataset=dataset,
        source_result_for_lens=source_pix_result_1,
        source_result_for_source=source_pix_result_2,
        lens_bulge=af.Model(al.lp.Sersic),
    )
    print(f"[SLaM] LIGHT LP done in {(time.time()-t0)/60:.1f} min", flush=True)

    print("[SLaM] MASS TOTAL...", flush=True)
    t0 = time.time()
    mass_result = mass_total.run(
        settings_search=settings_search,
        dataset=dataset,
        source_result_for_lens=source_pix_result_1,
        source_result_for_source=source_pix_result_2,
        light_result=light_result,
        mass=af.Model(al.mp.Isothermal),
    )
    print(f"[SLaM] MASS TOTAL done in {(time.time()-t0)/60:.1f} min", flush=True)
    print(mass_result.info, flush=True)
    return mass_result


def load_dataset(dataset_root: Path, name: str, mask_radius: float = 3.0):
    import autolens as al

    path = dataset_root / name
    dataset = al.Imaging.from_fits(
        data_path=path / "data.fits",
        psf_path=path / "psf.fits",
        noise_map_path=path / "noise_map.fits",
        pixel_scales=0.1,
    )
    mask = al.Mask2D.circular(
        shape_native=dataset.shape_native,
        pixel_scales=dataset.pixel_scales,
        radius=mask_radius,
    )
    return dataset.apply_mask(mask=mask)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--part", choices=("chain", "slam", "all"), default="all")
    p.add_argument("--output-root", type=Path, default=Path("./output").resolve())
    p.add_argument("--dataset-root", type=Path, required=True,
                   help="Path containing simple/ and simple__no_lens_light/ subdirs")
    p.add_argument("--repo-root", type=Path, required=True,
                   help="Path to Learning_to_Autolens (for slam_v2026 import)")
    p.add_argument("--n-live-s1", type=int, default=100)
    p.add_argument("--n-live-s2", type=int, default=150)
    p.add_argument("--slam-n-live", type=int, default=100)
    args = p.parse_args()

    sys.path.insert(0, str(args.repo_root.resolve()))
    args.output_root.mkdir(parents=True, exist_ok=True)

    import autolens as al
    print(f"PyAutoLens: {al.__version__}", flush=True)
    print(f"Output root: {args.output_root}", flush=True)
    print(f"Dataset root: {args.dataset_root}", flush=True)
    print(f"Part: {args.part}", flush=True)
    print(f"SLURM_JOB_ID: {os.environ.get('SLURM_JOB_ID', '(none)')}", flush=True)
    print(f"SLURM_CPUS_PER_TASK: {os.environ.get('SLURM_CPUS_PER_TASK', '(none)')}",
          flush=True)

    t_start = time.time()

    if args.part in ("chain", "all"):
        dataset_chain = load_dataset(args.dataset_root, "simple__no_lens_light")
        build_chain(dataset_chain, "simple__no_lens_light",
                    args.output_root, args.n_live_s1, args.n_live_s2)

    if args.part in ("slam", "all"):
        dataset_slam = load_dataset(args.dataset_root, "simple")
        build_slam(dataset_slam, "simple", args.output_root, args.slam_n_live)

    print(f"\nTotal wall time: {(time.time()-t_start)/3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
