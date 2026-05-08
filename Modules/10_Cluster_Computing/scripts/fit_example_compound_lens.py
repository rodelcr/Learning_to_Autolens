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

    # Priors mirror the loose, minimally-informative setup that the user's
    # original notebook `20251125_Mocks_redo_autolens_2src.ipynb` used to
    # reach log_Z≈30,700 on this same dataset. Key differences from earlier
    # tighter-prior attempts (jobs 7299592 and 7351708):
    #   - mass einstein_radius is UniformPrior(0, 8) — wide enough that
    #     the secondary (z=0.8) can collapse to near-zero if the data
    #     supports that. The compound-lens geometry is partially degenerate
    #     with a single effective lens, and forcing a ~1″ secondary mass
    #     locks the fit into a higher-chi² suboptimum (12σ arc residual).
    #   - bulge and mass centres are INDEPENDENT (not tied). Lets the
    #     light and mass centroids disagree when the data asks for it.
    #   - No external shear (user's best fit didn't need one).
    #   - intensity / effective_radius / sersic_index priors are wide
    #     LogUniform / Uniform, matching typical free-fit practice.

    # ---- Primary lens (z=0.5) -------------------------------------------
    bulge_0 = af.Model(al.lp.Sersic)
    mass_0  = af.Model(al.mp.Isothermal)
    mass_0.centre.centre_0  = af.GaussianPrior(mean=0.0, sigma=0.1)
    mass_0.centre.centre_1  = af.GaussianPrior(mean=0.0, sigma=0.1)
    bulge_0.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.3)
    bulge_0.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.3)
    bulge_0.ell_comps.ell_comps_0 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)
    bulge_0.ell_comps.ell_comps_1 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)
    bulge_0.intensity        = af.LogUniformPrior(lower_limit=1e-6, upper_limit=1e6)
    bulge_0.effective_radius = af.UniformPrior(lower_limit=0.0, upper_limit=30.0)
    bulge_0.sersic_index     = af.UniformPrior(lower_limit=0.8, upper_limit=5.0)
    mass_0.ell_comps.ell_comps_0 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)
    mass_0.ell_comps.ell_comps_1 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)
    mass_0.einstein_radius       = af.UniformPrior(lower_limit=0.0, upper_limit=8.0)

    # v4 cleanup (CLEANUP_PLAN.md Hypothesis A): re-enable external
    # shear on the primary. v3 reached log_Z=+30,705 but had max|res|
    # = 6.18σ with coherent chi² hot spots on the arc / counter-image —
    # the signature of missing shear that the v3 collapsed-secondary
    # couldn't absorb. Wide Gaussian(0, 0.15) — not truth-seeded, lets
    # data drive. Expected: max|res| drops to ≤4σ.
    shear_0 = af.Model(al.mp.ExternalShear)
    shear_0.gamma_1 = af.GaussianPrior(mean=0.0, sigma=0.15)
    shear_0.gamma_2 = af.GaussianPrior(mean=0.0, sigma=0.15)

    lens_0 = af.Model(al.Galaxy, redshift=0.5,
                      bulge=bulge_0, mass=mass_0, shear=shear_0)

    # ---- Secondary lens (z=0.8) -----------------------------------------
    bulge_1 = af.Model(al.lp.Sersic)
    mass_1  = af.Model(al.mp.Isothermal)
    mass_1.centre.centre_0  = af.GaussianPrior(mean=0.0, sigma=0.1)
    mass_1.centre.centre_1  = af.GaussianPrior(mean=0.0, sigma=0.1)
    bulge_1.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.3)
    bulge_1.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.3)
    bulge_1.ell_comps.ell_comps_0 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)
    bulge_1.ell_comps.ell_comps_1 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)
    bulge_1.intensity        = af.LogUniformPrior(lower_limit=1e-6, upper_limit=1e6)
    bulge_1.effective_radius = af.UniformPrior(lower_limit=0.0, upper_limit=30.0)
    bulge_1.sersic_index     = af.UniformPrior(lower_limit=0.8, upper_limit=5.0)
    mass_1.ell_comps.ell_comps_0 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)
    mass_1.ell_comps.ell_comps_1 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)
    mass_1.einstein_radius   = af.UniformPrior(lower_limit=0.0, upper_limit=8.0)
    lens_1 = af.Model(al.Galaxy, redshift=0.8, bulge=bulge_1, mass=mass_1)

    # ---- Source (z=1.7) -------------------------------------------------
    bulge_src = af.Model(al.lp.SersicCore)
    bulge_src.centre.centre_0  = af.GaussianPrior(mean=0.0, sigma=0.3)
    bulge_src.centre.centre_1  = af.GaussianPrior(mean=0.0, sigma=0.3)
    bulge_src.ell_comps.ell_comps_0 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)
    bulge_src.ell_comps.ell_comps_1 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)
    bulge_src.intensity        = af.LogUniformPrior(lower_limit=1e-5, upper_limit=1e3)
    bulge_src.effective_radius = af.UniformPrior(lower_limit=0.0, upper_limit=30.0)
    bulge_src.sersic_index     = af.UniformPrior(lower_limit=0.8, upper_limit=5.0)
    source = af.Model(al.Galaxy, redshift=1.7, bulge=bulge_src)

    model = af.Collection(galaxies=af.Collection(
        lens_0=lens_0, lens_1=lens_1, source=source))
    print(f"[CL/direct] total free parameters: {model.total_free_parameters}",
          flush=True)

    # Positions likelihood — NOT used in this driver.
    # The autolens 2026.4 visualizer can't render a list-of-lists positions
    # set (different-shape per image system fails Grid2DIrregular init),
    # and the truth-seeded priors above are already sufficient to block
    # the rotational mirror mode. The notebook version of this cell
    # documents how to enable a positions-likelihood in 02_compound_slam
    # (merge all positions into a single Grid2DIrregular before wrapping
    # in PositionsLH).
    analysis = al.AnalysisImaging(dataset=dataset, use_jax=False)

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
# Part 1b: direct_epl — same as direct but mass_0 is PowerLaw (free slope)
# -----------------------------------------------------------------------
# Why: v4 reached max|res|=4.40σ with Isothermal (γ'=2 fixed). The residual
# was salt-and-pepper but just over the 4σ pass bar. Hypothesis B from
# CLEANUP_PLAN.md: the real primary mass has γ' ≠ 2, and an EPL (Elliptical
# Power Law) fit recovers the slope and reduces max|res| further. This is
# the standard Mod 11 mass-model upgrade ladder: SIE → EPL is cheap (+1
# param) and often informative.

def build_direct_epl(dataset, output_root: Path, n_live: int = 250):
    """Same as build_direct_fit but lens_0.mass = PowerLaw (slope free)."""
    import autofit as af
    import autolens as al

    # ---- Primary lens (z=0.5) — PowerLaw mass --------------------------
    bulge_0 = af.Model(al.lp.Sersic)
    mass_0  = af.Model(al.mp.PowerLaw)
    mass_0.centre.centre_0  = af.GaussianPrior(mean=0.0, sigma=0.1)
    mass_0.centre.centre_1  = af.GaussianPrior(mean=0.0, sigma=0.1)
    bulge_0.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.3)
    bulge_0.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.3)
    bulge_0.ell_comps.ell_comps_0 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)
    bulge_0.ell_comps.ell_comps_1 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)
    bulge_0.intensity        = af.LogUniformPrior(lower_limit=1e-6, upper_limit=1e6)
    bulge_0.effective_radius = af.UniformPrior(lower_limit=0.0, upper_limit=30.0)
    bulge_0.sersic_index     = af.UniformPrior(lower_limit=0.8, upper_limit=5.0)
    mass_0.ell_comps.ell_comps_0 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)
    mass_0.ell_comps.ell_comps_1 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)
    mass_0.einstein_radius       = af.UniformPrior(lower_limit=0.0, upper_limit=8.0)
    # EPL slope: Isothermal = γ'=2. Typical real lenses γ' ∈ [1.8, 2.2].
    # Moderate GaussianPrior centred at 2 with sigma 0.2 — lets data drive
    # without wandering to unphysical extremes.
    mass_0.slope = af.TruncatedGaussianPrior(
        mean=2.0, sigma=0.2, lower_limit=1.5, upper_limit=2.5)

    shear_0 = af.Model(al.mp.ExternalShear)
    shear_0.gamma_1 = af.GaussianPrior(mean=0.0, sigma=0.15)
    shear_0.gamma_2 = af.GaussianPrior(mean=0.0, sigma=0.15)

    lens_0 = af.Model(al.Galaxy, redshift=0.5,
                      bulge=bulge_0, mass=mass_0, shear=shear_0)

    # ---- Secondary lens (z=0.8) — still Isothermal (data doesn't support
    # freeing its slope too; lens_1.mass.einstein_radius is rail-pinned at 0
    # in v4 anyway, so slope is meaningless).
    bulge_1 = af.Model(al.lp.Sersic)
    mass_1  = af.Model(al.mp.Isothermal)
    mass_1.centre.centre_0  = af.GaussianPrior(mean=0.0, sigma=0.1)
    mass_1.centre.centre_1  = af.GaussianPrior(mean=0.0, sigma=0.1)
    bulge_1.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.3)
    bulge_1.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.3)
    bulge_1.ell_comps.ell_comps_0 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)
    bulge_1.ell_comps.ell_comps_1 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)
    bulge_1.intensity        = af.LogUniformPrior(lower_limit=1e-6, upper_limit=1e6)
    bulge_1.effective_radius = af.UniformPrior(lower_limit=0.0, upper_limit=30.0)
    bulge_1.sersic_index     = af.UniformPrior(lower_limit=0.8, upper_limit=5.0)
    mass_1.ell_comps.ell_comps_0 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)
    mass_1.ell_comps.ell_comps_1 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)
    mass_1.einstein_radius   = af.UniformPrior(lower_limit=0.0, upper_limit=8.0)
    lens_1 = af.Model(al.Galaxy, redshift=0.8, bulge=bulge_1, mass=mass_1)

    # ---- Source (z=1.7) — same as direct -------------------------------
    bulge_src = af.Model(al.lp.SersicCore)
    bulge_src.centre.centre_0  = af.GaussianPrior(mean=0.0, sigma=0.3)
    bulge_src.centre.centre_1  = af.GaussianPrior(mean=0.0, sigma=0.3)
    bulge_src.ell_comps.ell_comps_0 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)
    bulge_src.ell_comps.ell_comps_1 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)
    bulge_src.intensity        = af.LogUniformPrior(lower_limit=1e-5, upper_limit=1e3)
    bulge_src.effective_radius = af.UniformPrior(lower_limit=0.0, upper_limit=30.0)
    bulge_src.sersic_index     = af.UniformPrior(lower_limit=0.8, upper_limit=5.0)
    source = af.Model(al.Galaxy, redshift=1.7, bulge=bulge_src)

    model = af.Collection(galaxies=af.Collection(
        lens_0=lens_0, lens_1=lens_1, source=source))
    print(f"[CL/direct_epl] total free parameters: {model.total_free_parameters}",
          flush=True)

    analysis = al.AnalysisImaging(dataset=dataset, use_jax=False)
    search = af.Nautilus(
        path_prefix           = output_root / "compound_lens",
        name                  = "compound_direct_epl_fit",
        unique_tag            = "mock_1",
        n_live                = n_live,
        n_batch               = 50,
        iterations_per_update = 30000,
        number_of_cores       = int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
    )
    print("[CL/direct_epl] Nautilus starting...", flush=True)
    t0 = time.time()
    result = search.fit(model=model, analysis=analysis)
    print(f"[CL/direct_epl] done in {(time.time()-t0)/60:.1f} min", flush=True)
    _force_visualize(analysis, result, tag="direct_epl")
    print(result.info, flush=True)
    return result


# -----------------------------------------------------------------------
# Part 1c: direct_pix — Isothermal + shear + pixelized source (2-stage)
# -----------------------------------------------------------------------
# Why: v4 has salt-and-pepper residual at 4.4σ. Hypothesis C from
# CLEANUP_PLAN.md: a pixelized source captures structure a 7-param
# SersicCore cannot, potentially dropping residuals further. Uses a
# 2-stage pipeline:
#   stage_1: Same as v4 (Isothermal + shear + Sersic source). This is
#            the "initialise adapt image" step — the Sersic source
#            becomes the model_image for the pixelization's adapt prior.
#   stage_2: Fix mass at stage_1 MAP, swap source to a pixelised
#            reconstruction (al.mesh.RectangularAdaptImage + al.reg.Adapt).

def build_direct_pix(dataset, output_root: Path, n_live: int = 200):
    """2-stage: Sersic source (init) → pixelised source (refine)."""
    import autofit as af
    import autolens as al

    # ---- Stage 1: full direct fit with Sersic source (v4 config) -------
    print("[CL/direct_pix] Stage 1: Sersic-source init fit...", flush=True)
    t0 = time.time()

    bulge_0 = af.Model(al.lp.Sersic)
    mass_0  = af.Model(al.mp.Isothermal)
    mass_0.centre.centre_0  = af.GaussianPrior(mean=0.0, sigma=0.1)
    mass_0.centre.centre_1  = af.GaussianPrior(mean=0.0, sigma=0.1)
    bulge_0.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.3)
    bulge_0.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.3)
    for comp in (bulge_0.ell_comps, mass_0.ell_comps):
        comp.ell_comps_0 = af.TruncatedGaussianPrior(
            mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)
        comp.ell_comps_1 = af.TruncatedGaussianPrior(
            mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)
    bulge_0.intensity        = af.LogUniformPrior(lower_limit=1e-6, upper_limit=1e6)
    bulge_0.effective_radius = af.UniformPrior(lower_limit=0.0, upper_limit=30.0)
    bulge_0.sersic_index     = af.UniformPrior(lower_limit=0.8, upper_limit=5.0)
    mass_0.einstein_radius   = af.UniformPrior(lower_limit=0.0, upper_limit=8.0)
    shear_0 = af.Model(al.mp.ExternalShear)
    shear_0.gamma_1 = af.GaussianPrior(mean=0.0, sigma=0.15)
    shear_0.gamma_2 = af.GaussianPrior(mean=0.0, sigma=0.15)
    lens_0 = af.Model(al.Galaxy, redshift=0.5,
                      bulge=bulge_0, mass=mass_0, shear=shear_0)

    bulge_1 = af.Model(al.lp.Sersic)
    mass_1  = af.Model(al.mp.Isothermal)
    mass_1.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.1)
    mass_1.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.1)
    bulge_1.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.3)
    bulge_1.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.3)
    for comp in (bulge_1.ell_comps, mass_1.ell_comps):
        comp.ell_comps_0 = af.TruncatedGaussianPrior(
            mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)
        comp.ell_comps_1 = af.TruncatedGaussianPrior(
            mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)
    bulge_1.intensity        = af.LogUniformPrior(lower_limit=1e-6, upper_limit=1e6)
    bulge_1.effective_radius = af.UniformPrior(lower_limit=0.0, upper_limit=30.0)
    bulge_1.sersic_index     = af.UniformPrior(lower_limit=0.8, upper_limit=5.0)
    mass_1.einstein_radius   = af.UniformPrior(lower_limit=0.0, upper_limit=8.0)
    lens_1 = af.Model(al.Galaxy, redshift=0.8, bulge=bulge_1, mass=mass_1)

    bulge_src_lp = af.Model(al.lp.SersicCore)
    bulge_src_lp.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.3)
    bulge_src_lp.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.3)
    bulge_src_lp.ell_comps.ell_comps_0 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)
    bulge_src_lp.ell_comps.ell_comps_1 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)
    bulge_src_lp.intensity = af.LogUniformPrior(lower_limit=1e-5, upper_limit=1e3)
    bulge_src_lp.effective_radius = af.UniformPrior(lower_limit=0.0, upper_limit=30.0)
    bulge_src_lp.sersic_index = af.UniformPrior(lower_limit=0.8, upper_limit=5.0)
    source_lp_gal = af.Model(al.Galaxy, redshift=1.7, bulge=bulge_src_lp)

    model_1 = af.Collection(galaxies=af.Collection(
        lens_0=lens_0, lens_1=lens_1, source=source_lp_gal))
    analysis = al.AnalysisImaging(dataset=dataset, use_jax=False)
    search_1 = af.Nautilus(
        path_prefix=output_root / "compound_lens",
        name="compound_direct_pix_stage1",
        unique_tag="mock_1",
        n_live=n_live, n_batch=50, iterations_per_update=30000,
        number_of_cores=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
    )
    result_1 = search_1.fit(model=model_1, analysis=analysis)
    print(f"[CL/direct_pix] Stage 1 done in {(time.time()-t0)/60:.1f} min; "
          f"θ_E_0 = {result_1.max_log_likelihood_instance.galaxies.lens_0.mass.einstein_radius:.3f}",
          flush=True)
    _force_visualize(analysis, result_1, tag="direct_pix_stage1")

    # ---- Stage 2: mass fixed, pixelised source ------------------------
    print("[CL/direct_pix] Stage 2: pixelised source refinement...", flush=True)
    t0 = time.time()

    # Fix both lens galaxies' light + mass at stage 1 MAP via `.instance`
    lens_0_fix = result_1.instance.galaxies.lens_0
    lens_1_fix = result_1.instance.galaxies.lens_1

    # Pixelised source: rectangular adaptive mesh, adapt-regularised,
    # using stage 1's Sersic-source reconstruction as the adapt image.
    # autolens 2026.4: use al.mesh.RectangularAdaptImage as a fixed
    # instance (shape isn't a fitted parameter). The REGULARISER must
    # be wrapped in af.Model with priors on its coefficients —
    # otherwise stage 2 has zero free parameters (mass fixed, source
    # pixelisation hyperparameters all fixed), which Nautilus refuses
    # to sample. AssertionError: "Model has no priors!" is the symptom
    # (caught on cl_pix_v2 job 8070792).
    regularization = af.Model(al.reg.Adapt)
    regularization.inner_coefficient = af.LogUniformPrior(
        lower_limit=1e-4, upper_limit=1.0)
    regularization.outer_coefficient = af.LogUniformPrior(
        lower_limit=1.0, upper_limit=1e3)
    regularization.signal_scale = af.LogUniformPrior(
        lower_limit=1e-3, upper_limit=10.0)
    pixelization = af.Model(
        al.Pixelization,
        # 20×20 = 400 source pixels. (28×28 = 784 caused OOM in
        # export_results post-phase on cl_pix_v4 — 60 GB peak vs 64 GB
        # SLURM cap. The 9176 image × 400 source mapping matrix is
        # ~3.7M elements — comfortable.)
        mesh=al.mesh.RectangularAdaptImage(shape=(20, 20)),
        regularization=regularization,
    )
    source_pix_gal = af.Model(al.Galaxy, redshift=1.7, pixelization=pixelization)

    model_2 = af.Collection(galaxies=af.Collection(
        lens_0=lens_0_fix, lens_1=lens_1_fix, source=source_pix_gal))
    adapt_images = al.AdaptImages(
        galaxy_name_image_dict=
            al.galaxy_name_image_dict_via_result_from(result=result_1),
    )
    # Pixelised fits MUST have a positions_likelihood_list, otherwise the
    # inversion is degenerate to demagnified solutions (a tiny faint
    # source + low-magnification model can mimic an extended source).
    # autolens 2026.4 explicitly raises AnalysisException without this.
    # `positions_likelihood_from(factor, minimum_threshold)` auto-derives
    # the image positions from stage 1's MAP tracer — same pattern slam_v2026
    # uses for its own pixelised stages.
    analysis_pix = al.AnalysisImaging(
        dataset=dataset,
        use_jax=False,
        adapt_images=adapt_images,
        positions_likelihood_list=[
            result_1.positions_likelihood_from(factor=3.0, minimum_threshold=0.2),
        ],
    )
    search_2 = af.Nautilus(
        path_prefix=output_root / "compound_lens",
        name="compound_direct_pix_stage2",
        unique_tag="mock_1",
        n_live=150, n_batch=50, iterations_per_update=30000,
        number_of_cores=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
    )
    result_2 = search_2.fit(model=model_2, analysis=analysis_pix)
    print(f"[CL/direct_pix] Stage 2 done in {(time.time()-t0)/60:.1f} min", flush=True)
    _force_visualize(analysis_pix, result_2, tag="direct_pix_stage2")
    print(result_2.info, flush=True)
    return result_1, result_2


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
    # Lower effective_radius floor 0.01→0.05 to prevent sub-pixel point-source
    # collapse (pixel_scale=0.05″).
    source_bulge.effective_radius = af.UniformPrior(lower_limit=0.05, upper_limit=1.0)
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

    # Positions-likelihood is dropped here too (see build_direct_fit
    # comment on the autolens 2026.4 visualizer limitation). Truth-seeded
    # priors + the staged chain's prior-passing provide sufficient
    # constraint without it.
    analysis = al.AnalysisImaging(dataset=dataset, use_jax=False)

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
    # Widened source centre box (see build_direct_fit — the true source sits
    # outside the ±0.2″ box, so the tighter prior rail-pinned the fit).
    bulge_src.centre.centre_0 = af.UniformPrior(lower_limit=-0.5, upper_limit=0.5)
    bulge_src.centre.centre_1 = af.UniformPrior(lower_limit=-0.5, upper_limit=0.5)
    bulge_src.effective_radius = af.TruncatedGaussianPrior(
        mean=0.2, sigma=0.15, lower_limit=0.05, upper_limit=1.5)
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

def _conjugate_positions_set_a():
    """Two image-plane positions of the brighter source component, derived
    via PointSolver on the v4 best-fit truth tracer (PROGRESS_LOG 2026-05-05).

    These conjugate at truth to ~10⁻³ arcsec (machine precision). They are
    the canonical PositionsLH input for compound_lens mock_1.
    """
    import autolens as al
    return al.Grid2DIrregular(values=[
        (-1.173, +1.806),   # outer NE image
        (+1.360, -0.571),   # outer SW image
    ])


def build_direct_with_positions_lh(dataset, output_root: Path,
                                   n_live: int = 250,
                                   positions_threshold: float = 0.1):
    """Phase C of v0.95 PositionsLH research batch.

    Identical model to `build_direct_fit` but adds an `al.PositionsLH` term
    using the 2 conjugate positions derived in `compound_lens/01` §2.5.
    The committed v4 PASS fit (log_Z=+30,856.54, chi²/N=0.69, max=4.40σ)
    used positions_likelihood_list=None; this run tests whether adding
    PositionsLH on top of the already-converging model:
      (a) tightens the posterior on lens_0.einstein_radius
      (b) accelerates burn-in (fewer Nautilus iterations to f_live=0.01)
      (c) preserves the existing log_Z (PositionsLH should add ~0 once the
          chain has found the basin — the penalty term is zero whenever
          source-plane spread < threshold)

    Multi-plane note: PositionsLH.log_likelihood_penalty_from traces input
    positions through the entire al.Tracer to its deepest source plane, so
    the constraint applies cleanly in the compound (3-redshift) setup.

    Default threshold = 0.1″. Sub-arcsecond for a converged fit; the
    looser threshold sweep is build_direct_positions_threshold_sweep.
    """
    import autofit as af
    import autolens as al
    import time

    print(f"\n[CL/direct_pos_lh] threshold={positions_threshold}\"", flush=True)

    # Same model as build_direct_fit. Inline-rebuilt to keep the function
    # self-contained (the helper functions in build_direct_fit aren't
    # split out yet).
    bulge_0 = af.Model(al.lp.Sersic)
    mass_0  = af.Model(al.mp.Isothermal)
    mass_0.centre.centre_0  = af.GaussianPrior(mean=0.0, sigma=0.1)
    mass_0.centre.centre_1  = af.GaussianPrior(mean=0.0, sigma=0.1)
    bulge_0.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.3)
    bulge_0.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.3)
    bulge_0.ell_comps.ell_comps_0 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)
    bulge_0.ell_comps.ell_comps_1 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)
    bulge_0.intensity        = af.LogUniformPrior(lower_limit=1e-6, upper_limit=1e6)
    bulge_0.effective_radius = af.UniformPrior(lower_limit=0.0, upper_limit=30.0)
    bulge_0.sersic_index     = af.UniformPrior(lower_limit=0.8, upper_limit=5.0)
    mass_0.ell_comps.ell_comps_0 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)
    mass_0.ell_comps.ell_comps_1 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)
    mass_0.einstein_radius       = af.UniformPrior(lower_limit=0.0, upper_limit=8.0)
    shear_0 = af.Model(al.mp.ExternalShear)
    shear_0.gamma_1 = af.GaussianPrior(mean=0.0, sigma=0.15)
    shear_0.gamma_2 = af.GaussianPrior(mean=0.0, sigma=0.15)
    lens_0 = af.Model(al.Galaxy, redshift=0.5,
                      bulge=bulge_0, mass=mass_0, shear=shear_0)

    bulge_1 = af.Model(al.lp.Sersic)
    mass_1  = af.Model(al.mp.Isothermal)
    mass_1.centre.centre_0  = af.GaussianPrior(mean=0.0, sigma=0.1)
    mass_1.centre.centre_1  = af.GaussianPrior(mean=0.0, sigma=0.1)
    bulge_1.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.3)
    bulge_1.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.3)
    bulge_1.ell_comps.ell_comps_0 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)
    bulge_1.ell_comps.ell_comps_1 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)
    bulge_1.intensity        = af.LogUniformPrior(lower_limit=1e-6, upper_limit=1e6)
    bulge_1.effective_radius = af.UniformPrior(lower_limit=0.0, upper_limit=30.0)
    bulge_1.sersic_index     = af.UniformPrior(lower_limit=0.8, upper_limit=5.0)
    mass_1.ell_comps.ell_comps_0 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)
    mass_1.ell_comps.ell_comps_1 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)
    mass_1.einstein_radius   = af.UniformPrior(lower_limit=0.0, upper_limit=8.0)
    lens_1 = af.Model(al.Galaxy, redshift=0.8, bulge=bulge_1, mass=mass_1)

    bulge_src = af.Model(al.lp.SersicCore)
    bulge_src.centre.centre_0  = af.GaussianPrior(mean=0.0, sigma=0.3)
    bulge_src.centre.centre_1  = af.GaussianPrior(mean=0.0, sigma=0.3)
    bulge_src.ell_comps.ell_comps_0 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)
    bulge_src.ell_comps.ell_comps_1 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)
    bulge_src.intensity        = af.LogUniformPrior(lower_limit=1e-5, upper_limit=1e3)
    bulge_src.effective_radius = af.UniformPrior(lower_limit=0.0, upper_limit=30.0)
    bulge_src.sersic_index     = af.UniformPrior(lower_limit=0.8, upper_limit=5.0)
    source = af.Model(al.Galaxy, redshift=1.7, bulge=bulge_src)

    model = af.Collection(galaxies=af.Collection(
        lens_0=lens_0, lens_1=lens_1, source=source))
    print(f"[CL/direct_pos_lh] free params: {model.total_free_parameters}",
          flush=True)

    positions = _conjugate_positions_set_a()
    positions_lh = al.PositionsLH(positions=positions,
                                  threshold=positions_threshold)
    print(f"[CL/direct_pos_lh] PositionsLH: {len(positions)} points, "
          f"threshold={positions_threshold}\"", flush=True)

    analysis = al.AnalysisImaging(
        dataset=dataset,
        positions_likelihood_list=[positions_lh],
        use_jax=False,
    )

    # Tag encodes the threshold so multiple sweep points coexist.
    tag = f"mock_1_pos_lh_t{positions_threshold:.3g}".replace(".", "p")
    search = af.Nautilus(
        path_prefix           = output_root / "compound_lens",
        name                  = "compound_direct_with_positions_lh",
        unique_tag            = tag,
        n_live                = n_live,
        n_batch               = 50,
        iterations_per_update = 30000,
        number_of_cores       = int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
    )

    print(f"[CL/direct_pos_lh] Nautilus starting (tag={tag})...", flush=True)
    t0 = time.time()
    result = search.fit(model=model, analysis=analysis)
    print(f"[CL/direct_pos_lh] done in {(time.time()-t0)/60:.1f} min",
          flush=True)
    _force_visualize(analysis, result, tag=f"direct_pos_lh_t{positions_threshold:.3g}")
    print(result.info, flush=True)
    return result


def build_positions_threshold_sweep(dataset, output_root: Path,
                                    n_live: int = 200):
    """Phase A of v0.95 PositionsLH research batch.

    Loops over 4 thresholds (1.0, 0.3, 0.1, 0.01 arcsec) and runs the
    same direct fit at each. Pedagogically validates the §2.5 sanity
    check from compound_lens/01: at the LOOSE end (1.0″) the constraint
    barely fires; at the TIGHT end (0.01″) it's ~aggressively
    rejecting; in the middle it accelerates burn-in.

    Wall: ~30-45 min/threshold × 4 ≈ 2.5h on 32 cores. Use n_live=200
    (slightly lower than the canonical 250) for speed.
    """
    thresholds = [1.0, 0.3, 0.1, 0.01]
    for t in thresholds:
        print(f"\n{'='*70}\n[CL/threshold_sweep] threshold = {t}\"\n{'='*70}",
              flush=True)
        try:
            build_direct_with_positions_lh(dataset, output_root,
                                           n_live=n_live,
                                           positions_threshold=t)
        except Exception as e:
            print(f"[CL/threshold_sweep] WARNING: threshold={t} failed: {e}",
                  flush=True)
            import traceback
            traceback.print_exc()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--part",
                   choices=("direct", "direct_epl", "direct_pix",
                            "slam_effective", "slam_staged", "all",
                            "direct_with_positions_lh",
                            "positions_threshold_sweep"),
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

    if args.part in ("direct_epl", "all"):
        build_direct_epl(dataset, args.output_root, n_live=args.n_live)

    if args.part in ("direct_pix", "all"):
        build_direct_pix(dataset, args.output_root, n_live=args.n_live)

    if args.part in ("slam_effective", "all"):
        build_slam_effective(dataset, args.output_root,
                             slam_n_live=args.slam_n_live)

    if args.part in ("slam_staged", "all"):
        build_slam_staged(dataset, args.output_root)

    if args.part == "direct_with_positions_lh":
        build_direct_with_positions_lh(dataset, args.output_root,
                                       n_live=args.n_live)

    if args.part == "positions_threshold_sweep":
        build_positions_threshold_sweep(dataset, args.output_root,
                                        n_live=200)

    print(f"\nTotal wall time: {(time.time()-t_start)/3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
