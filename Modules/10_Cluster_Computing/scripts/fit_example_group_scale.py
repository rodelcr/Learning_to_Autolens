"""fit_example_group_scale.py — Cannon driver for Examples/group_scale/.

A BGG (brightest group galaxy) + three satellites at the SAME redshift
(z=0.4) deflecting one source at z=1.8. All galaxies share the lens
plane — al.Tracer sums their deflection fields (no multi-plane).

Two fit parts for the pedagogical comparison:

Part 1: bgg_shear_only — BGG Sersic + Isothermal + ExternalShear + source.
         Satellite mass is ignored (absorbed into shear). ~15 params.

Part 2: bgg_plus_satellites — Same BGG model + 3 satellites with
         FIXED centres (at their photometric positions from the mock
         truth) and FREE einstein_radius per satellite. Each satellite's
         light is also modelled as a Sersic. ~25 params.

Compare log_Z. If Part 2 >> Part 1, satellites are resolvable. If
comparable, shear is doing the work and the survey pipeline can skip
satellite masses.

Usage (Cannon):
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
    source = _source_model()
    model = af.Collection(galaxies=af.Collection(bgg=bgg, source=source))
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
    # is the pedagogical test.
    for i, (y, x) in enumerate(_SAT_POS):
        sat_bulge = af.Model(al.lp.Sersic)
        sat_bulge.centre.centre_0 = y  # fixed
        sat_bulge.centre.centre_1 = x  # fixed
        sat_bulge.ell_comps.ell_comps_0 = af.TruncatedGaussianPrior(
            mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)
        sat_bulge.ell_comps.ell_comps_1 = af.TruncatedGaussianPrior(
            mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)
        sat_bulge.intensity        = af.LogUniformPrior(lower_limit=0.05, upper_limit=5.0)
        sat_bulge.effective_radius = af.TruncatedGaussianPrior(
            mean=0.25, sigma=0.15, lower_limit=0.05, upper_limit=1.0)
        sat_bulge.sersic_index     = af.TruncatedGaussianPrior(
            mean=2.0, sigma=0.8, lower_limit=0.8, upper_limit=4.0)

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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--part", choices=("bgg_shear_only", "bgg_plus_satellites", "all"),
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


if __name__ == "__main__":
    main()
