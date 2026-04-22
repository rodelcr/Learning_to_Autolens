"""
fit_example_compound_lens.py — Standalone Python for Examples/compound_lens/.

Fits the simulated two-deflector compound lens (lens at z=0.5, lens at z=0.8,
source at z=1.7). Native multi-plane via `al.Tracer` — no custom analysis
class. Two fit parts are implemented:

    --part direct           Single Nautilus fit of all 3 galaxies simultaneously
                            (mirrors Examples/compound_lens/01_compound_direct_fit.ipynb).
                            ~29 free parameters, n_live=250.

    --part slam_effective   Stock slam_v2026 pipeline, single effective Isothermal
                            + shear at z=0.5. Treats the secondary's contribution as
                            extra shear (Keeton & Zabludoff 2004 regime).

    --part slam_staged      Three inline Nautilus searches:
                              stage_1_primary     — z=0.5 lens + source, no secondary
                              stage_2_add_secondary — primary fixed, add z=0.8 lens
                              stage_3_joint       — both free, refined priors from 1+2

    --part all              All three, sequentially.

Usage (Cannon):
    python fit_example_compound_lens.py --part direct \
        --repo-root    /path/to/Learning_to_Autolens \
        --dataset-root /path/to/Examples/compound_lens/mocks \
        --output-root  /path/to/output

`output-root` is written as a scratch tree (Nautilus hashes, checkpoints,
intermediate FITS). The `export_results.py` post-step pulls the small
git-trackable artifacts into `Examples/compound_lens/results/<stage>/`.

Nautilus auto-resumes from any existing checkpoint.hdf5 inside output-root,
so a requeued or re-submitted job continues from where it left off.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


def _force_visualize(analysis, result, tag: str = ""):
    """Force autolens to regenerate image/fit.fits after a resumed search.

    Otherwise a finished-but-resumed search can leave fit.fits absent,
    which nulls chi² fields in summary.json. See fit_module04.py for
    the full backstory.
    """
    try:
        analysis.visualize(
            paths=result.paths,
            instance=result.max_log_likelihood_instance,
            during_analysis=False,
        )
    except Exception as e:
        print(f"[CL]   warning: post-fit visualize {tag} failed: {e}",
              flush=True)


def load_dataset(dataset_root: Path, mask_radius: float = 2.7):
    """Load the mock_1 compound-lens imaging with the standard 2.7″ mask."""
    import autolens as al

    dataset = al.Imaging.from_fits(
        data_path      = dataset_root / "mock_1_image.fits",
        noise_map_path = dataset_root / "mock_1_noise.fits",
        psf_path       = dataset_root / "mock_psf.fits",
        pixel_scales   = 0.05,
    )
    mask = al.Mask2D.circular(
        shape_native = dataset.shape_native,
        pixel_scales = dataset.pixel_scales,
        radius       = mask_radius,
    )
    return dataset.apply_mask(mask=mask)


# -----------------------------------------------------------------------
# Part 1: direct free-fit
# -----------------------------------------------------------------------

def build_direct_fit(dataset, output_root: Path, n_live: int = 250):
    """Direct Nautilus free-fit of lens_0 + lens_1 + source in one search."""
    import autofit as af
    import autolens as al

    # ---- Primary lens (z=0.5) with shear --------------------------------
    bulge_0 = af.Model(al.lp.Sersic)
    mass_0  = af.Model(al.mp.Isothermal)
    mass_0.centre.centre_0  = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
    mass_0.centre.centre_1  = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
    bulge_0.centre.centre_0 = mass_0.centre.centre_0
    bulge_0.centre.centre_1 = mass_0.centre.centre_1
    bulge_0.ell_comps.ell_comps_0 = af.TruncatedGaussianPrior(
        mean=0.33, sigma=0.15, lower_limit=-1.0, upper_limit=1.0)
    bulge_0.ell_comps.ell_comps_1 = af.TruncatedGaussianPrior(
        mean=0.00, sigma=0.15, lower_limit=-1.0, upper_limit=1.0)
    bulge_0.effective_radius = af.TruncatedGaussianPrior(
        mean=1.0, sigma=0.5, lower_limit=0.01, upper_limit=5.0)
    bulge_0.sersic_index     = af.TruncatedGaussianPrior(
        mean=3.0, sigma=0.8, lower_limit=0.5, upper_limit=6.0)
    mass_0.ell_comps.ell_comps_0 = af.TruncatedGaussianPrior(
        mean=0.33, sigma=0.2, lower_limit=-1.0, upper_limit=1.0)
    mass_0.ell_comps.ell_comps_1 = af.TruncatedGaussianPrior(
        mean=0.00, sigma=0.2, lower_limit=-1.0, upper_limit=1.0)
    mass_0.einstein_radius       = af.TruncatedGaussianPrior(
        mean=1.6, sigma=0.2, lower_limit=0.3, upper_limit=3.0)
    shear = af.Model(al.mp.ExternalShear)
    lens_0 = af.Model(al.Galaxy, redshift=0.5,
                      bulge=bulge_0, mass=mass_0, shear=shear)

    # ---- Secondary lens (z=0.8), no shear -------------------------------
    bulge_1 = af.Model(al.lp.Sersic)
    mass_1  = af.Model(al.mp.Isothermal)
    mass_1.centre.centre_0  = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
    mass_1.centre.centre_1  = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
    bulge_1.centre.centre_0 = mass_1.centre.centre_0
    bulge_1.centre.centre_1 = mass_1.centre.centre_1
    bulge_1.effective_radius = af.TruncatedGaussianPrior(
        mean=1.0, sigma=0.5, lower_limit=0.01, upper_limit=5.0)
    bulge_1.sersic_index     = af.TruncatedGaussianPrior(
        mean=3.0, sigma=0.8, lower_limit=0.5, upper_limit=6.0)
    mass_1.einstein_radius   = af.TruncatedGaussianPrior(
        mean=1.0, sigma=0.2, lower_limit=0.1, upper_limit=3.0)
    lens_1 = af.Model(al.Galaxy, redshift=0.8,
                      bulge=bulge_1, mass=mass_1)

    # ---- Source (z=1.7) -------------------------------------------------
    bulge_src = af.Model(al.lp.SersicCore)
    bulge_src.centre.centre_0  = af.UniformPrior(lower_limit=-0.2, upper_limit=0.2)
    bulge_src.centre.centre_1  = af.UniformPrior(lower_limit=-0.2, upper_limit=0.2)
    bulge_src.effective_radius = af.TruncatedGaussianPrior(
        mean=0.2, sigma=0.15, lower_limit=0.01, upper_limit=1.5)
    bulge_src.sersic_index     = af.TruncatedGaussianPrior(
        mean=2.0, sigma=0.8, lower_limit=0.5, upper_limit=6.0)
    source = af.Model(al.Galaxy, redshift=1.7, bulge=bulge_src)

    model = af.Collection(galaxies=af.Collection(
        lens_0=lens_0, lens_1=lens_1, source=source))
    print(f"[CL/direct] total free parameters: {model.total_free_parameters}",
          flush=True)

    # Positions likelihood rejects rotational mirrors that fit one arc
    # system but miss the other. Two image systems → one PositionsLH
    # constructed from both.
    positions_A = al.Grid2DIrregular(
        values=[(0.93, 0.17), (0.87, 0.35), (0.27, -1.0), (-0.86, 0.26)])
    positions_B = al.Grid2DIrregular(
        values=[(1.08, 1.83), (-1.3, -0.6)])
    positions_lh = al.PositionsLH(
        positions=[positions_A, positions_B], threshold=0.1)

    analysis = al.AnalysisImaging(
        dataset=dataset,
        positions_likelihood_list=[positions_lh],
        use_jax=False,
    )

    search = af.Nautilus(
        path_prefix           = output_root / "compound_lens",
        name                  = "compound_direct_fit",
        unique_tag            = "mock_1",
        n_live                = n_live,
        n_batch               = 50,
        iterations_per_update = 30000,
        number_of_cores       = int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
    )

    print("[CL/direct] Nautilus starting...", flush=True)
    t0 = time.time()
    result = search.fit(model=model, analysis=analysis)
    print(f"[CL/direct] done in {(time.time()-t0)/60:.1f} min", flush=True)
    _force_visualize(analysis, result, tag="direct")
    print(result.info, flush=True)
    return result


# -----------------------------------------------------------------------
# Part 2: single-effective-deflector SLaM (Track A)
# -----------------------------------------------------------------------

def build_slam_effective(dataset, output_root: Path, slam_n_live: int = 100):
    """slam_v2026 unchanged; one Isothermal + shear at z=0.5."""
    import autofit as af
    import autolens as al
    from slam_v2026 import source_lp, source_pix, light_lp, mass_total

    settings_search = af.SettingsSearch(
        path_prefix=output_root / "compound_lens" / "slam_effective",
        unique_tag="mock_1",
        number_of_cores=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
    )

    mass_lp = af.Model(al.mp.Isothermal)
    mass_lp.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.1)
    mass_lp.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.1)
    mass_lp.einstein_radius = af.UniformPrior(lower_limit=0.5, upper_limit=3.0)
    # ell_comps: the mock has primary ell_comps ~ (0.33, 0) according to
    # the first-attempt priors in the source notebook. Seed there to block
    # the rotational mirror.
    mass_lp.ell_comps.ell_comps_0 = af.GaussianPrior(mean=0.33, sigma=0.15)
    mass_lp.ell_comps.ell_comps_1 = af.GaussianPrior(mean=0.0,  sigma=0.15)

    shear_lp = af.Model(al.mp.ExternalShear)

    source_bulge = af.Model(al.lp.Sersic)
    source_bulge.effective_radius = af.UniformPrior(lower_limit=0.01, upper_limit=1.0)
    source_bulge.sersic_index     = af.UniformPrior(lower_limit=0.5, upper_limit=4.0)
    source_bulge.centre.centre_0  = af.GaussianPrior(mean=0.0, sigma=0.3)
    source_bulge.centre.centre_1  = af.GaussianPrior(mean=0.0, sigma=0.3)

    lens_bulge_lp = af.Model(al.lp.Sersic)
    lens_bulge_lp.effective_radius = af.GaussianPrior(mean=1.0, sigma=0.5)
    lens_bulge_lp.sersic_index     = af.GaussianPrior(mean=3.0, sigma=0.8)

    print("[CL/slam_eff] SOURCE LP...", flush=True)
    t0 = time.time()
    source_lp_result = source_lp.run(
        settings_search=settings_search,
        dataset=dataset,
        lens_bulge=lens_bulge_lp,
        lens_disk=None,
        mass=mass_lp,
        shear=shear_lp,
        source_bulge=source_bulge,
        redshift_lens=0.5,
        redshift_source=1.7,
    )
    print(f"[CL/slam_eff] SOURCE LP done in {(time.time()-t0)/60:.1f} min; "
          f"θ_E = {source_lp_result.instance.galaxies.lens.mass.einstein_radius:.3f}\"",
          flush=True)

    print("[CL/slam_eff] SOURCE PIX run_1...", flush=True)
    t0 = time.time()
    source_pix_result_1 = source_pix.run_1(
        settings_search=settings_search,
        dataset=dataset,
        source_lp_result=source_lp_result,
    )
    print(f"[CL/slam_eff] SOURCE PIX run_1 done in {(time.time()-t0)/60:.1f} min",
          flush=True)

    print("[CL/slam_eff] SOURCE PIX run_2...", flush=True)
    t0 = time.time()
    source_pix_result_2 = source_pix.run_2(
        settings_search=settings_search,
        dataset=dataset,
        source_lp_result=source_lp_result,
        source_pix_result_1=source_pix_result_1,
    )
    print(f"[CL/slam_eff] SOURCE PIX run_2 done in {(time.time()-t0)/60:.1f} min",
          flush=True)

    print("[CL/slam_eff] LIGHT LP...", flush=True)
    t0 = time.time()
    light_result = light_lp.run(
        settings_search=settings_search,
        dataset=dataset,
        source_result_for_lens=source_pix_result_1,
        source_result_for_source=source_pix_result_2,
        lens_bulge=af.Model(al.lp.Sersic),
    )
    print(f"[CL/slam_eff] LIGHT LP done in {(time.time()-t0)/60:.1f} min",
          flush=True)

    print("[CL/slam_eff] MASS TOTAL...", flush=True)
    t0 = time.time()
    mass_result = mass_total.run(
        settings_search=settings_search,
        dataset=dataset,
        source_result_for_lens=source_pix_result_1,
        source_result_for_source=source_pix_result_2,
        light_result=light_result,
        mass=af.Model(al.mp.Isothermal),
    )
    print(f"[CL/slam_eff] MASS TOTAL done in {(time.time()-t0)/60:.1f} min",
          flush=True)
    print(mass_result.info, flush=True)
    return mass_result


# -----------------------------------------------------------------------
# Part 3: staged two-deflector chain (Track B)
# -----------------------------------------------------------------------

def build_slam_staged(dataset, output_root: Path):
    """Three inline Nautilus searches for the two-deflector system.

    stage_1_primary      — z=0.5 lens + source, no z=0.8 (absorbs secondary into shear)
    stage_2_add_secondary — primary at MAP, add z=0.8 lens
    stage_3_joint        — both lenses + source, all free, stage 1/2 posteriors → priors
    """
    import autofit as af
    import autolens as al

    stage_root = output_root / "compound_lens" / "slam_staged"
    ncores = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))

    # Image positions shared across all three stages.
    positions_A = al.Grid2DIrregular(
        values=[(0.93, 0.17), (0.87, 0.35), (0.27, -1.0), (-0.86, 0.26)])
    positions_B = al.Grid2DIrregular(
        values=[(1.08, 1.83), (-1.3, -0.6)])
    positions_lh = al.PositionsLH(
        positions=[positions_A, positions_B], threshold=0.1)
    analysis = al.AnalysisImaging(
        dataset=dataset,
        positions_likelihood_list=[positions_lh],
        use_jax=False,
    )

    # --------------- Stage 1: primary + source ---------------------------
    bulge_0 = af.Model(al.lp.Sersic)
    mass_0  = af.Model(al.mp.Isothermal)
    mass_0.centre.centre_0  = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
    mass_0.centre.centre_1  = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
    bulge_0.centre.centre_0 = mass_0.centre.centre_0
    bulge_0.centre.centre_1 = mass_0.centre.centre_1
    mass_0.einstein_radius = af.TruncatedGaussianPrior(
        mean=1.6, sigma=0.3, lower_limit=0.3, upper_limit=3.0)
    bulge_0.effective_radius = af.TruncatedGaussianPrior(
        mean=1.0, sigma=0.5, lower_limit=0.01, upper_limit=5.0)
    bulge_0.sersic_index = af.TruncatedGaussianPrior(
        mean=3.0, sigma=0.8, lower_limit=0.5, upper_limit=6.0)

    shear_stg1 = af.Model(al.mp.ExternalShear)
    lens_0_stg1 = af.Model(al.Galaxy, redshift=0.5,
                           bulge=bulge_0, mass=mass_0, shear=shear_stg1)

    bulge_src = af.Model(al.lp.SersicCore)
    bulge_src.centre.centre_0 = af.UniformPrior(lower_limit=-0.2, upper_limit=0.2)
    bulge_src.centre.centre_1 = af.UniformPrior(lower_limit=-0.2, upper_limit=0.2)
    source_stg1 = af.Model(al.Galaxy, redshift=1.7, bulge=bulge_src)

    model_1 = af.Collection(galaxies=af.Collection(
        lens_0=lens_0_stg1, source=source_stg1))
    print(f"[CL/slam_staged] Stage 1 params: {model_1.total_free_parameters}",
          flush=True)

    search_1 = af.Nautilus(
        path_prefix=stage_root,
        name="stage_1_primary",
        unique_tag="mock_1",
        n_live=200, n_batch=50, iterations_per_update=30000,
        number_of_cores=ncores,
    )
    t0 = time.time()
    result_1 = search_1.fit(model=model_1, analysis=analysis)
    print(f"[CL/slam_staged] stage 1 done in {(time.time()-t0)/60:.1f} min",
          flush=True)
    _force_visualize(analysis, result_1, tag="stage_1")

    # --------------- Stage 2: primary fixed, add secondary ---------------
    lens_0_fixed = result_1.instance.galaxies.lens_0

    bulge_1 = af.Model(al.lp.Sersic)
    mass_1  = af.Model(al.mp.Isothermal)
    mass_1.centre.centre_0  = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
    mass_1.centre.centre_1  = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
    bulge_1.centre.centre_0 = mass_1.centre.centre_0
    bulge_1.centre.centre_1 = mass_1.centre.centre_1
    mass_1.einstein_radius = af.TruncatedGaussianPrior(
        mean=1.0, sigma=0.3, lower_limit=0.1, upper_limit=3.0)
    bulge_1.effective_radius = af.TruncatedGaussianPrior(
        mean=1.0, sigma=0.5, lower_limit=0.01, upper_limit=5.0)
    bulge_1.sersic_index = af.TruncatedGaussianPrior(
        mean=3.0, sigma=0.8, lower_limit=0.5, upper_limit=6.0)
    lens_1 = af.Model(al.Galaxy, redshift=0.8, bulge=bulge_1, mass=mass_1)

    source_stg2 = result_1.model.galaxies.source

    model_2 = af.Collection(galaxies=af.Collection(
        lens_0=lens_0_fixed, lens_1=lens_1, source=source_stg2))
    print(f"[CL/slam_staged] Stage 2 params: {model_2.total_free_parameters}",
          flush=True)

    search_2 = af.Nautilus(
        path_prefix=stage_root,
        name="stage_2_add_secondary",
        unique_tag="mock_1",
        n_live=150, n_batch=50, iterations_per_update=30000,
        number_of_cores=ncores,
    )
    t0 = time.time()
    result_2 = search_2.fit(model=model_2, analysis=analysis)
    print(f"[CL/slam_staged] stage 2 done in {(time.time()-t0)/60:.1f} min",
          flush=True)
    _force_visualize(analysis, result_2, tag="stage_2")

    # --------------- Stage 3: joint refinement ---------------------------
    model_3 = af.Collection(galaxies=af.Collection(
        lens_0 = result_1.model.galaxies.lens_0,
        lens_1 = result_2.model.galaxies.lens_1,
        source = result_2.model.galaxies.source,
    ))
    print(f"[CL/slam_staged] Stage 3 params: {model_3.total_free_parameters}",
          flush=True)

    search_3 = af.Nautilus(
        path_prefix=stage_root,
        name="stage_3_joint",
        unique_tag="mock_1",
        n_live=250, n_batch=50, iterations_per_update=30000,
        number_of_cores=ncores,
    )
    t0 = time.time()
    result_3 = search_3.fit(model=model_3, analysis=analysis)
    print(f"[CL/slam_staged] stage 3 done in {(time.time()-t0)/60:.1f} min",
          flush=True)
    _force_visualize(analysis, result_3, tag="stage_3")
    print(result_3.info, flush=True)
    return result_3


# -----------------------------------------------------------------------
# main
# -----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--part",
                   choices=("direct", "slam_effective", "slam_staged", "all"),
                   default="direct")
    p.add_argument("--output-root", type=Path,
                   default=Path("./output").resolve())
    p.add_argument("--dataset-root", type=Path, required=True,
                   help="Path containing mock_1_image.fits, mock_1_noise.fits, "
                        "mock_psf.fits")
    p.add_argument("--repo-root", type=Path, required=True,
                   help="Path to Learning_to_Autolens (for slam_v2026 import)")
    p.add_argument("--n-live", type=int, default=250,
                   help="n_live for the direct-fit Nautilus search")
    p.add_argument("--slam-n-live", type=int, default=100)
    args = p.parse_args()

    sys.path.insert(0, str(args.repo_root.resolve()))
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

    if args.part in ("direct", "all"):
        build_direct_fit(dataset, args.output_root, n_live=args.n_live)

    if args.part in ("slam_effective", "all"):
        build_slam_effective(dataset, args.output_root,
                             slam_n_live=args.slam_n_live)

    if args.part in ("slam_staged", "all"):
        build_slam_staged(dataset, args.output_root)

    print(f"\nTotal wall time: {(time.time()-t_start)/3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
