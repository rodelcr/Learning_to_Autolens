"""fit_example_group_scale.py — Cannon driver for Examples/group_scale/.

A BGG (brightest group galaxy) + three satellites at the SAME redshift
(z=0.4) deflecting one source at z=1.8. All galaxies share the lens
plane — al.Tracer sums their deflection fields (no multi-plane).

Three fit parts for the pedagogical comparison:

Part 1: bgg_shear_only — BGG Sersic + Isothermal + ExternalShear + source.
         Satellite mass is ignored (absorbed into shear). ~15 params.

Part 2: bgg_plus_satellites — Same BGG model + 3 satellites with
         FIXED centres (at their photometric positions from the mock
         truth) and FREE einstein_radius per satellite. Each satellite's
         light is also modelled as a Sersic. ~25 params.

Part 3: truth_anchored — Same architecture as bgg_plus_satellites but
         ALL priors set as tight Gaussians centred on the truth values
         from mocks/mock_truth.json. Tests whether PyAutoLens's model
         space *can* fit this group-scale system when constrained near
         truth — establishes a chi^2 / log_Z ceiling for the freely-fit
         variants. If truth_anchored converges cleanly while v1-v3
         freely-fit attempts stalled in burn-in (LEARNING_LOG.md
         2026-04-24), the bottleneck is search-space exploration, not
         model-space representability.

Compare log_Z across all three. If Part 2 >> Part 1, satellites are
resolvable. If comparable, shear is doing the work. If Part 3 >> Part 2,
the freely-fit search misses the global optimum near truth — chart a
SLaM-style staged chain to walk the chain there.

Usage (Cannon):
    sbatch --export=ALL,EXAMPLE=group_scale,FIT_EXTRA_ARGS=--part=truth_anchored \
        submit_cannon.slurm
    sbatch --export=ALL,EXAMPLE=group_scale,FIT_EXTRA_ARGS=--part=all \
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
        print(f"[GROUP] warning: post-fit visualize {tag} failed: {e}",
              flush=True)


def load_dataset(dataset_root: Path, mask_radius: float = 3.5):
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


# Satellite photometric positions from mock_truth.json (in practice, these
# would come from a light-only Sersic fit of the lens-plane galaxies).
_SAT_POS = [(1.8, 0.7), (-1.5, -1.2), (0.5, -2.0)]


def _satellite_light_only_model(y, x):
    """Satellite as LIGHT-ONLY (Sersic), NO mass. For bgg_shear_only fit —
    we still need to subtract each satellite's photometric flux, otherwise
    the unmodelled bright spots dominate chi² and the 'shear absorbs
    satellite mass?' question becomes unanswerable."""
    import autofit as af
    import autolens as al
    sat_bulge = af.Model(al.lp.Sersic)
    sat_bulge.centre.centre_0 = y  # fixed
    sat_bulge.centre.centre_1 = x  # fixed
    sat_bulge.ell_comps.ell_comps_0 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)
    sat_bulge.ell_comps.ell_comps_1 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)
    # Tightened vs v1 (was LogUniformPrior 0.05-5): truth intensities are
    # 0.3-0.5 so 0.1-2 covers the full range comfortably and cuts the
    # 3-satellite joint prior volume by 9× (2^9 vs previous 5^9 ratio).
    sat_bulge.intensity        = af.LogUniformPrior(lower_limit=0.1, upper_limit=2.0)
    sat_bulge.effective_radius = af.TruncatedGaussianPrior(
        mean=0.25, sigma=0.10, lower_limit=0.05, upper_limit=0.6)
    sat_bulge.sersic_index     = af.TruncatedGaussianPrior(
        mean=2.0, sigma=0.6, lower_limit=0.8, upper_limit=4.0)
    return af.Model(al.Galaxy, redshift=0.4, bulge=sat_bulge)


def _bgg_model():
    """BGG: Sersic bulge + Isothermal mass (full ellipticity)."""
    import autofit as af
    import autolens as al
    bulge = af.Model(al.lp.Sersic)
    mass  = af.Model(al.mp.Isothermal)
    bulge.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.1)
    bulge.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.1)
    mass.centre.centre_0  = bulge.centre.centre_0
    mass.centre.centre_1  = bulge.centre.centre_1
    bulge.ell_comps.ell_comps_0 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)
    bulge.ell_comps.ell_comps_1 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)
    bulge.intensity        = af.LogUniformPrior(lower_limit=0.1, upper_limit=10.0)
    bulge.effective_radius = af.TruncatedGaussianPrior(
        mean=0.9, sigma=0.3, lower_limit=0.1, upper_limit=3.0)
    bulge.sersic_index     = af.TruncatedGaussianPrior(
        mean=4.0, sigma=1.0, lower_limit=0.8, upper_limit=5.0)
    mass.ell_comps.ell_comps_0 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)
    mass.ell_comps.ell_comps_1 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)
    mass.einstein_radius = af.TruncatedGaussianPrior(
        mean=1.5, sigma=0.2, lower_limit=0.5, upper_limit=3.0)
    return bulge, mass


def _source_model(z=1.8):
    import autofit as af
    import autolens as al
    b = af.Model(al.lp.SersicCore)
    b.centre.centre_0  = af.GaussianPrior(mean=0.12, sigma=0.15)
    b.centre.centre_1  = af.GaussianPrior(mean=0.08, sigma=0.15)
    b.ell_comps.ell_comps_0 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)
    b.ell_comps.ell_comps_1 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)
    b.intensity        = af.LogUniformPrior(lower_limit=0.3, upper_limit=30.0)
    b.effective_radius = af.TruncatedGaussianPrior(
        mean=0.08, sigma=0.03, lower_limit=0.02, upper_limit=0.3)
    b.sersic_index     = af.TruncatedGaussianPrior(
        mean=1.4, sigma=0.5, lower_limit=0.8, upper_limit=4.0)
    return af.Model(al.Galaxy, redshift=z, bulge=b)


def build_bgg_shear_only(dataset, output_root: Path, n_live: int = 200):
    """BGG + shear + source. Satellite mass ignored."""
    import autofit as af
    import autolens as al

    bulge, mass = _bgg_model()
    shear = af.Model(al.mp.ExternalShear)
    shear.gamma_1 = af.GaussianPrior(mean=0.0, sigma=0.1)
    shear.gamma_2 = af.GaussianPrior(mean=0.0, sigma=0.1)

    bgg = af.Model(al.Galaxy, redshift=0.4,
                   bulge=bulge, mass=mass, shear=shear)

    # Include satellite LIGHT (no mass) so the photometric flux at
    # satellite positions is subtracted from the data. Without this the
    # bright unmodelled galaxy cores dominate chi² and the comparison
    # against bgg_plus_satellites becomes apples-to-oranges.
    galaxies_dict = {"bgg": bgg}
    for i, (y, x) in enumerate(_SAT_POS):
        galaxies_dict[f"satellite_light_{i+1}"] = _satellite_light_only_model(y, x)

    source = _source_model()
    galaxies_dict["source"] = source
    model = af.Collection(galaxies=af.Collection(**galaxies_dict))
    print(f"[GROUP/shear_only] total free parameters: {model.total_free_parameters}",
          flush=True)

    analysis = al.AnalysisImaging(dataset=dataset, use_jax=False)
    search = af.Nautilus(
        path_prefix           = output_root / "group_scale",
        name                  = "bgg_shear_only_fit",
        unique_tag            = "mock_1",
        n_live                = n_live,
        n_batch               = 50,
        iterations_per_update = 30000,
        number_of_cores       = int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
    )
    print("[GROUP/shear_only] Nautilus starting...", flush=True)
    t0 = time.time()
    result = search.fit(model=model, analysis=analysis)
    print(f"[GROUP/shear_only] done in {(time.time()-t0)/60:.1f} min", flush=True)
    _force_visualize(analysis, result, tag="shear_only")
    print(result.info, flush=True)
    return result


def build_bgg_plus_satellites(dataset, output_root: Path, n_live: int = 200):
    """BGG + 3 satellites with fixed centres + source."""
    import autofit as af
    import autolens as al

    bulge, mass = _bgg_model()
    # No external shear — satellites should do that job now.
    bgg = af.Model(al.Galaxy, redshift=0.4, bulge=bulge, mass=mass)

    galaxies_dict = {"bgg": bgg}
    # Satellites: fixed centres at photometric positions, SIS mass (1 free
    # param each), Sersic light (5 free params) with moderate priors.
    # Satellite *light* is constrained by the photometric data; its MASS
    # is the pedagogical test. Priors match _satellite_light_only_model
    # so the two variants differ only in the presence of satellite mass.
    for i, (y, x) in enumerate(_SAT_POS):
        sat_bulge = af.Model(al.lp.Sersic)
        sat_bulge.centre.centre_0 = y  # fixed
        sat_bulge.centre.centre_1 = x  # fixed
        sat_bulge.ell_comps.ell_comps_0 = af.TruncatedGaussianPrior(
            mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)
        sat_bulge.ell_comps.ell_comps_1 = af.TruncatedGaussianPrior(
            mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)
        sat_bulge.intensity        = af.LogUniformPrior(lower_limit=0.1, upper_limit=2.0)
        sat_bulge.effective_radius = af.TruncatedGaussianPrior(
            mean=0.25, sigma=0.10, lower_limit=0.05, upper_limit=0.6)
        sat_bulge.sersic_index     = af.TruncatedGaussianPrior(
            mean=2.0, sigma=0.6, lower_limit=0.8, upper_limit=4.0)

        sat_mass = af.Model(al.mp.IsothermalSph)
        sat_mass.centre.centre_0 = y  # fixed — satellite mass at photometric position
        sat_mass.centre.centre_1 = x
        sat_mass.einstein_radius = af.UniformPrior(lower_limit=0.0, upper_limit=1.0)

        sat = af.Model(al.Galaxy, redshift=0.4, bulge=sat_bulge, mass=sat_mass)
        galaxies_dict[f"satellite_{i+1}"] = sat

    source = _source_model()
    galaxies_dict["source"] = source
    model = af.Collection(galaxies=af.Collection(**galaxies_dict))
    print(f"[GROUP/satellites] total free parameters: {model.total_free_parameters}",
          flush=True)

    analysis = al.AnalysisImaging(dataset=dataset, use_jax=False)
    search = af.Nautilus(
        path_prefix           = output_root / "group_scale",
        name                  = "bgg_plus_satellites_fit",
        unique_tag            = "mock_1",
        n_live                = n_live,
        n_batch               = 50,
        iterations_per_update = 30000,
        number_of_cores       = int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
    )
    print("[GROUP/satellites] Nautilus starting...", flush=True)
    t0 = time.time()
    result = search.fit(model=model, analysis=analysis)
    print(f"[GROUP/satellites] done in {(time.time()-t0)/60:.1f} min", flush=True)
    _force_visualize(analysis, result, tag="satellites")
    print(result.info, flush=True)
    return result


def _load_mock_truth(dataset_root: Path) -> dict:
    """Load mock_truth.json from the dataset_root."""
    import json
    return json.loads((dataset_root / "mock_truth.json").read_text())


# =============================================================================
# Staged-chain helper: BGG + source only, no shear, no satellites
# =============================================================================
def _bgg_only_galaxy_model():
    """BGG (Sersic + Isothermal) at z=0.4, no shear. For Stage 1 of the
    staged-satellites chain, where satellite regions are masked out."""
    import autofit as af
    import autolens as al
    bulge, mass = _bgg_model()
    return af.Model(al.Galaxy, redshift=0.4, bulge=bulge, mass=mass)


def build_staged_satellites(dataset_root: Path, output_root: Path,
                            n_live: int = 200):
    """Two-stage SLaM-style chain for the group-scale system.

    Stage 1: BGG + source ONLY, with a tight mask (radius=1.7") that
             excludes the three satellite regions (closest satellite is at
             r=1.92"). Fast — ~14 params, no satellite light to confuse
             chi^2. Gets a strong posterior on BGG mass / light + source.

    Stage 2: Full mask (radius=3.5"). BGG + source priors centred on Stage 1
             posterior. 3 satellites with FIXED centres, FREE Sersic light
             + IsothermalSph mass. The new free parameters are limited to
             the 3 satellite light + mass blocks (~19 params), which is the
             scale Nautilus handles cleanly.

    Tests whether the freely-fit failure (LEARNING_LOG 2026-04-24) was
    really a search-exploration problem in the joint 30+ param landscape.
    If Stage 1 -> Stage 2 converges cleanly, yes — the resume path is
    this staged chain. If Stage 2 still stalls, the problem is the joint
    BGG+satellite degeneracy structure, not the search.
    """
    import autofit as af
    import autolens as al
    import time

    print("[GROUP/staged] starting 2-stage chain", flush=True)

    # ---- Stage 1: BGG + source, mask=1.7" ----
    dataset_s1 = load_dataset(dataset_root, mask_radius=1.7)
    print(f"[GROUP/staged] Stage 1 dataset: "
          f"pixels_in_mask={dataset_s1.mask.pixels_in_mask}", flush=True)

    bgg = _bgg_only_galaxy_model()
    src = _source_model()
    s1_model = af.Collection(galaxies=af.Collection(bgg=bgg, source=src))
    print(f"[GROUP/staged] Stage 1 free params: "
          f"{s1_model.total_free_parameters}", flush=True)

    s1_search = af.Nautilus(
        path_prefix=output_root / "group_scale",
        name="staged_satellites_stage1_bgg_source",
        unique_tag="mock_1_staged",
        n_live=n_live, n_batch=50, iterations_per_update=30000,
        number_of_cores=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
    )
    print("[GROUP/staged] Stage 1 Nautilus starting...", flush=True)
    t0 = time.time()
    s1_analysis = al.AnalysisImaging(dataset=dataset_s1, use_jax=False)
    s1_result = s1_search.fit(model=s1_model, analysis=s1_analysis)
    print(f"[GROUP/staged] Stage 1 done in {(time.time()-t0)/60:.1f} min, "
          f"log_Z={s1_result.samples.log_evidence:.2f}", flush=True)
    _force_visualize(s1_analysis, s1_result, tag="staged_stage1")

    # ---- Stage 2: full mask, BGG/source from Stage 1, free satellites ----
    dataset_s2 = load_dataset(dataset_root, mask_radius=3.5)
    print(f"[GROUP/staged] Stage 2 dataset: "
          f"pixels_in_mask={dataset_s2.mask.pixels_in_mask}", flush=True)

    # BGG and source: priors centred on Stage 1 posterior
    galaxies_dict = {
        "bgg": s1_result.model.galaxies.bgg,
    }

    # Satellites: FIXED centres at photometric positions, FREE light + mass
    # Same priors as build_bgg_plus_satellites for direct comparability.
    for i, (y, x) in enumerate(_SAT_POS):
        sat_bulge = af.Model(al.lp.Sersic)
        sat_bulge.centre.centre_0 = y
        sat_bulge.centre.centre_1 = x
        sat_bulge.ell_comps.ell_comps_0 = af.TruncatedGaussianPrior(
            mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)
        sat_bulge.ell_comps.ell_comps_1 = af.TruncatedGaussianPrior(
            mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)
        sat_bulge.intensity = af.LogUniformPrior(lower_limit=0.1, upper_limit=2.0)
        sat_bulge.effective_radius = af.TruncatedGaussianPrior(
            mean=0.25, sigma=0.10, lower_limit=0.05, upper_limit=0.6)
        sat_bulge.sersic_index = af.TruncatedGaussianPrior(
            mean=2.0, sigma=0.6, lower_limit=0.8, upper_limit=4.0)

        sat_mass = af.Model(al.mp.IsothermalSph)
        sat_mass.centre.centre_0 = y
        sat_mass.centre.centre_1 = x
        sat_mass.einstein_radius = af.UniformPrior(lower_limit=0.0, upper_limit=1.0)

        galaxies_dict[f"satellite_{i+1}"] = af.Model(
            al.Galaxy, redshift=0.4, bulge=sat_bulge, mass=sat_mass)

    galaxies_dict["source"] = s1_result.model.galaxies.source

    s2_model = af.Collection(galaxies=af.Collection(**galaxies_dict))
    print(f"[GROUP/staged] Stage 2 free params: "
          f"{s2_model.total_free_parameters}", flush=True)

    s2_search = af.Nautilus(
        path_prefix=output_root / "group_scale",
        name="staged_satellites_stage2_full",
        unique_tag="mock_1_staged",
        n_live=n_live, n_batch=50, iterations_per_update=30000,
        number_of_cores=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
    )
    print("[GROUP/staged] Stage 2 Nautilus starting...", flush=True)
    t0 = time.time()
    s2_analysis = al.AnalysisImaging(dataset=dataset_s2, use_jax=False)
    s2_result = s2_search.fit(model=s2_model, analysis=s2_analysis)
    print(f"[GROUP/staged] Stage 2 done in {(time.time()-t0)/60:.1f} min, "
          f"log_Z={s2_result.samples.log_evidence:.2f}", flush=True)
    _force_visualize(s2_analysis, s2_result, tag="staged_stage2")
    print(s2_result.info, flush=True)
    return s2_result


def build_truth_anchored(dataset, output_root: Path, dataset_root: Path,
                         n_live: int = 200):
    """All-truth-anchored fit: BGG + 3 satellites + source, all priors as
    tight Gaussians centred on mock_truth.json values. Tests whether
    PyAutoLens's model space *can* fit this group-scale system."""
    import autofit as af
    import autolens as al

    truth = _load_mock_truth(dataset_root)

    bgg_t = truth["bgg"]
    src_t = truth["source"]["bulge"]
    sats_t = truth["satellites"]
    z_l = truth["redshifts"]["lens"]
    z_s = truth["redshifts"]["source"]

    # ---- BGG: Sersic light + Isothermal mass, tight on truth ----
    b = af.Model(al.lp.Sersic)
    b.centre.centre_0 = af.GaussianPrior(mean=bgg_t["bulge"]["centre"][0], sigma=0.05)
    b.centre.centre_1 = af.GaussianPrior(mean=bgg_t["bulge"]["centre"][1], sigma=0.05)
    b.intensity        = af.LogUniformPrior(lower_limit=1e-2, upper_limit=1e2)
    b.effective_radius = af.GaussianPrior(mean=bgg_t["bulge"]["effective_radius"], sigma=0.1)
    b.sersic_index     = af.GaussianPrior(mean=bgg_t["bulge"]["sersic_index"], sigma=0.5)
    b.ell_comps.ell_comps_0 = af.TruncatedGaussianPrior(
        mean=bgg_t["bulge"]["ell_comps"][0], sigma=0.05, lower_limit=-1.0, upper_limit=1.0)
    b.ell_comps.ell_comps_1 = af.TruncatedGaussianPrior(
        mean=bgg_t["bulge"]["ell_comps"][1], sigma=0.05, lower_limit=-1.0, upper_limit=1.0)

    m = af.Model(al.mp.Isothermal)
    m.centre.centre_0  = b.centre.centre_0
    m.centre.centre_1  = b.centre.centre_1
    m.einstein_radius  = af.GaussianPrior(mean=bgg_t["mass"]["einstein_radius"], sigma=0.05)
    m.ell_comps.ell_comps_0 = af.TruncatedGaussianPrior(
        mean=bgg_t["mass"]["ell_comps"][0], sigma=0.05, lower_limit=-1.0, upper_limit=1.0)
    m.ell_comps.ell_comps_1 = af.TruncatedGaussianPrior(
        mean=bgg_t["mass"]["ell_comps"][1], sigma=0.05, lower_limit=-1.0, upper_limit=1.0)

    bgg = af.Model(al.Galaxy, redshift=z_l, bulge=b, mass=m)
    galaxies_dict = {"bgg": bgg}

    # ---- Satellites: tight on truth (centre fixed, mass + light tight) ----
    for i, sat in enumerate(sats_t):
        cy, cx = sat["centre"]
        sb = af.Model(al.lp.Sersic)
        sb.centre.centre_0 = cy  # fixed at truth
        sb.centre.centre_1 = cx
        sb.intensity        = af.LogUniformPrior(lower_limit=1e-2, upper_limit=1e2)
        sb.effective_radius = af.GaussianPrior(
            mean=sat["effective_radius"], sigma=0.05)
        sb.sersic_index     = af.GaussianPrior(
            mean=sat["sersic_index"], sigma=0.4)
        sb.ell_comps.ell_comps_0 = af.TruncatedGaussianPrior(
            mean=0.0, sigma=0.1, lower_limit=-1.0, upper_limit=1.0)
        sb.ell_comps.ell_comps_1 = af.TruncatedGaussianPrior(
            mean=0.0, sigma=0.1, lower_limit=-1.0, upper_limit=1.0)

        sm = af.Model(al.mp.IsothermalSph)
        sm.centre.centre_0 = cy
        sm.centre.centre_1 = cx
        sm.einstein_radius = af.GaussianPrior(
            mean=sat["einstein_radius"], sigma=0.05)

        galaxies_dict[f"satellite_{i+1}"] = af.Model(
            al.Galaxy, redshift=z_l, bulge=sb, mass=sm)

    # ---- Source: tight on truth ----
    src = af.Model(al.lp.SersicCore)
    src.centre.centre_0 = af.GaussianPrior(mean=src_t["centre"][0], sigma=0.05)
    src.centre.centre_1 = af.GaussianPrior(mean=src_t["centre"][1], sigma=0.05)
    src.intensity        = af.LogUniformPrior(lower_limit=1e-2, upper_limit=1e2)
    src.effective_radius = af.GaussianPrior(mean=src_t["effective_radius"], sigma=0.02)
    src.sersic_index     = af.GaussianPrior(mean=src_t["sersic_index"], sigma=0.3)
    src.ell_comps.ell_comps_0 = af.TruncatedGaussianPrior(
        mean=src_t["ell_comps"][0], sigma=0.05, lower_limit=-1.0, upper_limit=1.0)
    src.ell_comps.ell_comps_1 = af.TruncatedGaussianPrior(
        mean=src_t["ell_comps"][1], sigma=0.05, lower_limit=-1.0, upper_limit=1.0)
    galaxies_dict["source"] = af.Model(al.Galaxy, redshift=z_s, bulge=src)

    model = af.Collection(galaxies=af.Collection(**galaxies_dict))
    print(f"[GROUP/truth_anchored] total free parameters: "
          f"{model.total_free_parameters}", flush=True)

    analysis = al.AnalysisImaging(dataset=dataset, use_jax=False)
    search = af.Nautilus(
        path_prefix           = output_root / "group_scale",
        name                  = "truth_anchored_fit",
        unique_tag            = "mock_1",
        n_live                = n_live,
        n_batch               = 50,
        iterations_per_update = 30000,
        number_of_cores       = int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
    )
    print("[GROUP/truth_anchored] Nautilus starting...", flush=True)
    t0 = time.time()
    result = search.fit(model=model, analysis=analysis)
    print(f"[GROUP/truth_anchored] done in {(time.time()-t0)/60:.1f} min", flush=True)
    _force_visualize(analysis, result, tag="truth_anchored")
    print(result.info, flush=True)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--part",
                   choices=("bgg_shear_only", "bgg_plus_satellites",
                            "truth_anchored", "staged_satellites", "all"),
                   default="all")
    p.add_argument("--repo-root",    type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--output-root",  type=Path, required=True)
    p.add_argument("--n-live",       type=int, default=200)
    args = p.parse_args()

    sys.path.insert(0, str(args.repo_root))
    dataset = load_dataset(args.dataset_root, mask_radius=3.5)

    if args.part in ("bgg_shear_only", "all"):
        build_bgg_shear_only(dataset, args.output_root, n_live=args.n_live)
    if args.part in ("bgg_plus_satellites", "all"):
        build_bgg_plus_satellites(dataset, args.output_root, n_live=args.n_live)
    if args.part == "truth_anchored":
        # Not in 'all' — has different architecture (no shear) so isn't
        # directly comparable; run separately as a validation diagnostic.
        build_truth_anchored(dataset, args.output_root, args.dataset_root,
                             n_live=args.n_live)
    if args.part == "staged_satellites":
        # 2-stage chain: builds its own datasets at different mask radii.
        build_staged_satellites(args.dataset_root, args.output_root,
                                n_live=args.n_live)


if __name__ == "__main__":
    main()
