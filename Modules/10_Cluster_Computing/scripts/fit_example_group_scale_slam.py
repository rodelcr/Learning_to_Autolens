"""
fit_example_group_scale_slam.py — SLaM pipeline for the group_scale mock.

Adapts autolens_workspace_latest/scripts/group/slam.py to our 1-BGG +
3-satellites + 1-source mock. The previous freely-fit attempts
(LEARNING_LOG 2026-04-24) and the staged_satellites v1/v2 attempts all
stalled in burn-in for many hours without convergence — the 30+ free
parameter joint landscape is too large for Nautilus n_live=200 to
explore in useful time.

The SLaM pipeline solves both problems:
  1. **MGE light** for every galaxy. Replaces 7 nonlinear Sersic params
     per galaxy with ~3 nonlinear (Gaussian σ controls + ell_comps) plus
     N=30 linear amplitudes inverted at zero Nautilus cost. For BGG +
     3 satellites: 28 → 12 nonlinear light params.
  2. **Stage decomposition.** Each search fits one component at a time
     with the others fixed:
       Stage 0 (source_lp_0): light only (BGG + 3 sat MGE bulges, no
              source, no mass)
       Stage 1 (source_lp_1): mass + parametric source. Light fixed from
              stage 0. BGG mass = Iso, satellites = Iso with
              luminosity-bounded Einstein radius prior.
       Stage 2 (mass_total): final mass refinement. Light + source fixed
              from stage 1.
     This is the canonical group-scale SLaM 3-stage shortened pipeline
     (skips the pixelized source pipeline since our mock source is a
     simple Sersic).

Usage:
    sbatch --export=ALL,EXAMPLE=group_scale,FIT_EXTRA_ARGS=--part=slam \\
        Modules/10_Cluster_Computing/scripts/submit_cannon.slurm

Reference:
    autolens_workspace_latest/scripts/group/slam.py — full 6-stage
        cluster-scale SLaM pipeline with MGE bulges + extra/scaling
        galaxies + pixelized source. We strip pixelization +
        scaling-galaxies (we only have 3 satellites — individual modeling
        is fine).
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
        print(f"[GROUP_SLAM]   warning: post-fit visualize {tag} failed: {e}",
              flush=True)


def load_dataset(dataset_root: Path, mask_radius: float = 3.5):
    import autolens as al
    dataset = al.Imaging.from_fits(
        data_path=dataset_root / "mock_image.fits",
        noise_map_path=dataset_root / "mock_noise.fits",
        psf_path=dataset_root / "mock_psf.fits",
        pixel_scales=0.05,
    )
    mask = al.Mask2D.circular(
        shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales,
        radius=mask_radius,
    )
    return dataset.apply_mask(mask=mask)


# =============================================================================
# Stage 0: SOURCE LP — light only (MGE bulges, no source, no mass)
# =============================================================================
def slam_source_lp_0(dataset, output_root: Path, truth: dict,
                     mask_radius: float = 3.5, n_batch: int = 50):
    """Stage 0 of the SLaM pipeline.

    Light-only fit. Every galaxy (BGG + 3 satellites) gets a free MGE
    bulge. No mass, no source — just lens-light subtraction. Output is
    a clean light model for every galaxy.

    The MGE handles arbitrary morphology so we don't need to wrestle
    with Sersic-n choice. ~12 nonlinear light params + ~120 linear
    amplitudes inverted for free.
    """
    import autofit as af
    import autolens as al

    z_l = truth["redshifts"]["lens"]
    bgg_centre = (truth["bgg"]["bulge"]["centre"][0],
                  truth["bgg"]["bulge"]["centre"][1])

    # BGG MGE bulge (centre seeded near truth, slight Gaussian prior)
    bgg_bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius, total_gaussians=30, gaussian_per_basis=2,
        centre=bgg_centre, centre_prior_is_uniform=False, centre_sigma=0.05,
    )
    bgg = af.Model(al.Galaxy, redshift=z_l, bulge=bgg_bulge)

    # Satellite MGE bulges (centres fixed at photometric truth)
    satellites = []
    for s in truth["satellites"]:
        sat_centre = (s["centre"][0], s["centre"][1])
        sat_bulge = al.model_util.mge_model_from(
            mask_radius=mask_radius, total_gaussians=10,
            centre=sat_centre, centre_prior_is_uniform=False, centre_sigma=0.05,
            ell_comps_prior_is_uniform=True,
        )
        satellites.append(af.Model(al.Galaxy, redshift=z_l, bulge=sat_bulge))

    extra_galaxies = af.Collection(satellites)

    # n_live scaled to model complexity per workspace recipe
    n_live = 100 + 30 * 1 + 30 * len(satellites)  # 100 + 30 + 90 = 220

    model = af.Collection(
        galaxies=af.Collection(bgg=bgg),
        extra_galaxies=extra_galaxies,
    )
    print(f"[GROUP_SLAM/source_lp_0] {model.total_free_parameters} free params "
          f"(BGG + {len(satellites)} sats, MGE light only)", flush=True)

    analysis = al.AnalysisImaging(dataset=dataset, use_jax=False)
    search = af.Nautilus(
        path_prefix=output_root / "group_scale_slam",
        name="source_lp_0",
        unique_tag="mock_1",
        n_live=n_live, n_batch=n_batch, iterations_per_update=15000,
        number_of_cores=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
        n_like_max=1000000,
    )

    print(f"[GROUP_SLAM/source_lp_0] Nautilus starting (n_live={n_live})...",
          flush=True)
    t0 = time.time()
    result = search.fit(model=model, analysis=analysis)
    print(f"[GROUP_SLAM/source_lp_0] done in {(time.time()-t0)/60:.1f} min, "
          f"log_Z={result.samples.log_evidence:.2f}", flush=True)
    _force_visualize(analysis, result, tag="source_lp_0")
    return result


# =============================================================================
# Stage 1: SOURCE LP — light fixed, mass + parametric source free
# =============================================================================
def slam_source_lp_1(dataset, output_root: Path, truth: dict,
                     source_lp_0_result, mask_radius: float = 3.5,
                     n_batch: int = 50):
    """Stage 1 of the SLaM pipeline.

    Light fixed from stage 0. Add Isothermal mass on BGG + each satellite
    (with luminosity-bounded Einstein radius prior). Add a single-MGE
    parametric source.

    Per workspace pattern: extra-galaxy upper limit on θ_E is
        min(5 * 0.5 * total_luminosity^0.6, 5.0)
    which prevents unphysical mass for low-luminosity satellites.
    """
    import autofit as af
    import autolens as al
    import numpy as np

    z_l = truth["redshifts"]["lens"]
    z_s = truth["redshifts"]["source"]
    pixel_scale = truth["pixel_scales"]

    # ---- Image positions for PositionsLH ----
    # Truth positions: derived from the dataset arc topology (cf.
    # group_scale/00_climb_to_group.ipynb §2 photometric anchoring).
    # For the mock the source is at (0.12, 0.08) and lensed by BGG
    # θ_E=1.5″ — 4 image-like positions form an extended arc.
    positions = al.Grid2DIrregular(values=[
        (+0.7, +1.4), (+1.4, -0.5), (-1.0, -1.2), (-1.4, +0.6),
    ])

    tracer_lp = (
        source_lp_0_result.max_log_likelihood_fit.tracer_linear_light_profiles_to_light_profiles
    )

    # ---- BGG: light fixed from stage 0, mass = Isothermal free + shear ----
    lp0_bgg = source_lp_0_result.instance.galaxies.bgg

    bgg_mass = af.Model(al.mp.Isothermal)
    bgg_mass.centre = lp0_bgg.bulge.centre
    bgg_mass.einstein_radius = af.UniformPrior(lower_limit=0.0, upper_limit=3.0)
    bgg_mass.ell_comps.ell_comps_0 = af.GaussianPrior(mean=0.0, sigma=0.3)
    bgg_mass.ell_comps.ell_comps_1 = af.GaussianPrior(mean=0.0, sigma=0.3)

    bgg = af.Model(
        al.Galaxy, redshift=z_l, bulge=lp0_bgg.bulge,
        mass=bgg_mass, shear=af.Model(al.mp.ExternalShear),
    )

    # ---- Satellites: light fixed, Isothermal mass with luminosity-bounded prior ----
    satellite_models = []
    n_satellites = len(truth["satellites"])
    for i in range(n_satellites):
        lp0_sat = source_lp_0_result.instance.extra_galaxies[i]

        sat_mass = af.Model(al.mp.IsothermalSph)
        sat_mass.centre = lp0_sat.bulge.centre

        # Luminosity-bounded prior on θ_E (workspace recipe)
        try:
            luminosity_per_g = [
                2 * np.pi * g.sigma**2 / g.axis_ratio() * g.intensity
                for g in tracer_lp.galaxies[1 + i].bulge.profile_list
            ]
            total_lum = float(np.sum(luminosity_per_g) / pixel_scale**2)
            theta_E_upper = min(5 * 0.5 * total_lum**0.6, 5.0)
        except (AttributeError, TypeError, IndexError):
            theta_E_upper = 1.0  # fallback
        sat_mass.einstein_radius = af.UniformPrior(
            lower_limit=0.0, upper_limit=theta_E_upper)

        satellite_models.append(af.Model(
            al.Galaxy, redshift=z_l, bulge=lp0_sat.bulge, mass=sat_mass,
        ))

    extra_galaxies = af.Collection(satellite_models)

    # ---- Source: parametric MGE bulge ----
    source_bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius, total_gaussians=30,
        centre=(0.0, 0.0), centre_prior_is_uniform=False, centre_sigma=0.6,
    )
    source = af.Model(al.Galaxy, redshift=z_s, bulge=source_bulge)

    n_live = 150 + 30 * 1 + 30 * n_satellites  # 150 + 30 + 90 = 270

    model = af.Collection(
        galaxies=af.Collection(bgg=bgg, source=source),
        extra_galaxies=extra_galaxies,
    )
    print(f"[GROUP_SLAM/source_lp_1] {model.total_free_parameters} free params "
          f"(light fixed; BGG mass + {n_satellites} sat masses + parametric source)",
          flush=True)

    analysis = al.AnalysisImaging(
        dataset=dataset,
        positions_likelihood_list=[al.PositionsLH(positions=positions, threshold=1.0)],
        use_jax=False,
    )
    search = af.Nautilus(
        path_prefix=output_root / "group_scale_slam",
        name="source_lp_1",
        unique_tag="mock_1",
        n_live=n_live, n_batch=n_batch, iterations_per_update=15000,
        number_of_cores=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
        n_like_max=200000,
    )

    print(f"[GROUP_SLAM/source_lp_1] Nautilus starting (n_live={n_live})...",
          flush=True)
    t0 = time.time()
    result = search.fit(model=model, analysis=analysis)
    print(f"[GROUP_SLAM/source_lp_1] done in {(time.time()-t0)/60:.1f} min, "
          f"log_Z={result.samples.log_evidence:.2f}", flush=True)
    _force_visualize(analysis, result, tag="source_lp_1")
    return result


# =============================================================================
# Stage 2: MASS TOTAL — final mass refinement
# =============================================================================
def slam_mass_total(dataset, output_root: Path, truth: dict,
                    source_lp_1_result, mask_radius: float = 3.5,
                    n_batch: int = 50):
    """Stage 2 (final). Light + source fixed from stage 1; mass refines
    to PowerLaw on BGG (slope free) + Iso on satellites with priors
    centred on stage-1 posterior.
    """
    import autofit as af
    import autolens as al

    z_l = truth["redshifts"]["lens"]
    z_s = truth["redshifts"]["source"]

    lp1_bgg = source_lp_1_result.instance.galaxies.bgg
    lp1_source = source_lp_1_result.instance.galaxies.source
    lp1_extra = source_lp_1_result.instance.extra_galaxies

    # Promote BGG mass from Iso to PowerLaw, seeded near stage-1 best-fit
    bgg_mass = af.Model(al.mp.PowerLaw)
    bgg_mass.centre = lp1_bgg.mass.centre
    bgg_mass.einstein_radius = af.GaussianPrior(
        mean=lp1_bgg.mass.einstein_radius, sigma=0.1)
    bgg_mass.slope = af.GaussianPrior(mean=2.0, sigma=0.3)
    bgg_mass.ell_comps.ell_comps_0 = af.GaussianPrior(
        mean=lp1_bgg.mass.ell_comps[0], sigma=0.1)
    bgg_mass.ell_comps.ell_comps_1 = af.GaussianPrior(
        mean=lp1_bgg.mass.ell_comps[1], sigma=0.1)

    bgg = af.Model(
        al.Galaxy, redshift=z_l, bulge=lp1_bgg.bulge,
        mass=bgg_mass, shear=lp1_bgg.shear,
    )

    # Satellite masses — Gaussian priors centred on stage-1 posterior
    satellite_models = []
    for i, sat_inst in enumerate(lp1_extra):
        sat_mass = af.Model(al.mp.IsothermalSph)
        sat_mass.centre = sat_inst.mass.centre
        # Tight Gaussian centred on stage-1 posterior
        sat_mass.einstein_radius = af.GaussianPrior(
            mean=sat_inst.mass.einstein_radius, sigma=0.1)
        satellite_models.append(af.Model(
            al.Galaxy, redshift=z_l, bulge=sat_inst.bulge, mass=sat_mass,
        ))
    extra_galaxies = af.Collection(satellite_models)

    # Source fixed from stage 1
    source = af.Model(al.Galaxy, redshift=z_s, bulge=lp1_source.bulge)

    model = af.Collection(
        galaxies=af.Collection(bgg=bgg, source=source),
        extra_galaxies=extra_galaxies,
    )
    print(f"[GROUP_SLAM/mass_total] {model.total_free_parameters} free params "
          f"(BGG PowerLaw + sat masses + source fixed)", flush=True)

    analysis = al.AnalysisImaging(dataset=dataset, use_jax=False)
    search = af.Nautilus(
        path_prefix=output_root / "group_scale_slam",
        name="mass_total",
        unique_tag="mock_1",
        n_live=200, n_batch=n_batch, iterations_per_update=15000,
        number_of_cores=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
    )

    print("[GROUP_SLAM/mass_total] Nautilus starting...", flush=True)
    t0 = time.time()
    result = search.fit(model=model, analysis=analysis)
    print(f"[GROUP_SLAM/mass_total] done in {(time.time()-t0)/60:.1f} min, "
          f"log_Z={result.samples.log_evidence:.2f}", flush=True)
    _force_visualize(analysis, result, tag="mass_total")
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root",    type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--output-root",  type=Path, required=True)
    args = p.parse_args()

    sys.path.insert(0, str(args.repo_root))
    truth = json.loads((args.dataset_root / "mock_truth.json").read_text())
    dataset = load_dataset(args.dataset_root, mask_radius=3.5)
    print(f"[GROUP_SLAM] dataset pixels_in_mask={dataset.mask.pixels_in_mask}",
          flush=True)

    print("\n=== Stage 0: SOURCE LP — light only ===", flush=True)
    r0 = slam_source_lp_0(dataset, args.output_root, truth)

    print("\n=== Stage 1: SOURCE LP — mass + source ===", flush=True)
    r1 = slam_source_lp_1(dataset, args.output_root, truth, r0)

    print("\n=== Stage 2: MASS TOTAL — final ===", flush=True)
    r2 = slam_mass_total(dataset, args.output_root, truth, r1)

    print(f"\n[GROUP_SLAM] all three stages done. Final log_Z = "
          f"{r2.samples.log_evidence:.2f}", flush=True)


if __name__ == "__main__":
    main()
