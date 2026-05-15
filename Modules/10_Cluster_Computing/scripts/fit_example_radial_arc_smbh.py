"""
fit_example_radial_arc_smbh.py — Einstein-spiral + SMBH methodology bridge (v0.96).

Fits the synthetic Einstein-spiral mock from
`Examples/radial_arc_smbh/mocks/generate_mock.py`. Methodology mirrors the
Shajib et al. and Ferrami et al. AGEL spiral-lens papers: PowerLaw +
ExternalShear + central PointMass + Sersic source, with stellar kinematic
constraint as the γ′–M_BH degeneracy-breaker.

CLI parts (--part choices):

    direct           Power-law + shear + source. γ′ free, NO point mass.
                     Establishes the imaging-only inner-slope posterior.
                     ~9 free params, n_live=200.

    no_pointmass     Sanity baseline: γ′ pinned at 2.0 (isothermal). Same
                     model otherwise. Bayes-factor reference.
                     ~8 free, n_live=150.

    with_pointmass   Adds al.mp.PointMass at the lens centre. Free M_BH
                     (via einstein_radius_BH parameter). Tests the
                     joint (γ′, M_BH) posterior — the γ′–M_BH degeneracy
                     should be visible. ~10 free, n_live=250.

    with_kinematics  Joint AnalysisImaging + AnalysisKinematics (Jeans
                     σ_v likelihood) via af.FactorGraphModel. The Jeans
                     analysis is a custom subclass we share with the
                     kinematic_h0_break driver (Examples/quad_time_delay/).
                     STATUS in v0.96: STUB — the custom Analysis class
                     ships in v0.97. Raises NotImplementedError with
                     pointer.

    all              Sequential 1 -> 2 -> 3 (skip 4 until v0.97).

Usage (Cannon):
    sbatch --account=siag_lab --partition=siag \\
           --time=8:00:00 --mem=192G --cpus-per-task=32 \\
           --job-name=rarc_direct \\
           --export=ALL,EXAMPLE=radial_arc_smbh,FIT_EXTRA_ARGS='--part=direct' \\
           Modules/10_Cluster_Computing/scripts/submit_cannon.slurm

Nautilus auto-resumes from any existing checkpoint.hdf5.

Memory budget per 2026-05-11 CLUSTER_WORKFLOW_NOTES: --mem=192G minimum
on 32 cores for any single-galaxy autolens fit.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


def load_dataset(dataset_root: Path, mask_radius: float = 1.8):
    """Load the radial_arc_smbh mock + apply circular mask."""
    import autolens as al

    dataset = al.Imaging.from_fits(
        data_path=dataset_root / "image.fits",
        noise_map_path=dataset_root / "noise_map.fits",
        psf_path=dataset_root / "psf.fits",
        pixel_scales=0.05,
    )
    mask = al.Mask2D.circular(
        shape_native=dataset.shape_native,
        pixel_scales=dataset.pixel_scales,
        radius=mask_radius,
    )
    return dataset.apply_mask(mask=mask)


def _build_lens_galaxy_model(slope_pinned: float | None = None,
                              include_pointmass: bool = False):
    """Shared lens-model factory.

    slope_pinned: if not None, the PowerLaw slope is fixed at this value
        (used by --part=no_pointmass to pin γ′=2 for the isothermal baseline).
        Otherwise the slope is a free Uniform(1.6, 2.4) prior.
    include_pointmass: if True, attach an al.mp.PointMass with free
        einstein_radius_BH ∈ Uniform(0.001, 0.20) at the lens centre.
    """
    import autofit as af
    import autolens as al

    # Lens light Sersic n=4 (de Vaucouleurs-ish elliptical)
    bulge = af.Model(al.lp.Sersic)
    bulge.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.2)
    bulge.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.2)
    bulge.intensity = af.UniformPrior(lower_limit=0.0, upper_limit=3.0)
    bulge.effective_radius = af.UniformPrior(lower_limit=0.3, upper_limit=2.0)
    bulge.sersic_index = af.UniformPrior(lower_limit=2.0, upper_limit=6.0)

    # Power-law mass
    mass = af.Model(al.mp.PowerLaw)
    mass.centre = bulge.centre  # tied to bulge
    mass.einstein_radius = af.UniformPrior(lower_limit=0.5, upper_limit=2.0)
    if slope_pinned is not None:
        mass.slope = slope_pinned
    else:
        mass.slope = af.UniformPrior(lower_limit=1.6, upper_limit=2.4)

    # External shear
    shear = af.Model(al.mp.ExternalShear)
    shear.gamma_1 = af.GaussianPrior(mean=0.0, sigma=0.05)
    shear.gamma_2 = af.GaussianPrior(mean=0.0, sigma=0.05)

    kwargs = dict(redshift=0.7, bulge=bulge, mass=mass, shear=shear)
    if include_pointmass:
        smbh = af.Model(al.mp.PointMass)
        smbh.centre = bulge.centre
        smbh.einstein_radius = af.UniformPrior(lower_limit=0.001,
                                                upper_limit=0.20)
        kwargs["smbh"] = smbh

    return af.Model(al.Galaxy, **kwargs)


def _build_source_galaxy_model():
    """Source galaxy: Sersic n=1.5 disc."""
    import autofit as af
    import autolens as al

    src_bulge = af.Model(al.lp.Sersic)
    src_bulge.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.3)
    src_bulge.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.3)
    src_bulge.intensity = af.UniformPrior(lower_limit=0.0, upper_limit=2.0)
    src_bulge.effective_radius = af.UniformPrior(lower_limit=0.05,
                                                  upper_limit=0.5)
    src_bulge.sersic_index = af.UniformPrior(lower_limit=0.5, upper_limit=4.0)

    return af.Model(al.Galaxy, redshift=1.5, bulge=src_bulge)


def build_direct_fit(dataset, output_root: Path, n_live: int = 200,
                     tag_suffix: str = ""):
    """--part=direct: PowerLaw + shear + source, NO BH."""
    import autofit as af
    import autolens as al

    lens = _build_lens_galaxy_model(slope_pinned=None, include_pointmass=False)
    source = _build_source_galaxy_model()
    model = af.Collection(galaxies=af.Collection(lens=lens, source=source))
    print(f"[RARC/direct] free params: {model.prior_count}", flush=True)

    analysis = al.AnalysisImaging(dataset=dataset)
    search = af.Nautilus(
        path_prefix=output_root,
        name="rarc_direct",
        unique_tag=f"rarc_direct{tag_suffix}",
        n_live=n_live,
        n_batch=50,
        iterations_per_update=5000,
        number_of_cores=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
    )
    return search.fit(model=model, analysis=analysis)


def build_no_pointmass_fit(dataset, output_root: Path, n_live: int = 150,
                           tag_suffix: str = ""):
    """--part=no_pointmass: isothermal baseline (γ′=2 pinned, no BH)."""
    import autofit as af
    import autolens as al

    lens = _build_lens_galaxy_model(slope_pinned=2.0, include_pointmass=False)
    source = _build_source_galaxy_model()
    model = af.Collection(galaxies=af.Collection(lens=lens, source=source))
    print(f"[RARC/no_pointmass] free params: {model.prior_count}", flush=True)

    analysis = al.AnalysisImaging(dataset=dataset)
    search = af.Nautilus(
        path_prefix=output_root,
        name="rarc_no_pointmass",
        unique_tag=f"rarc_no_pointmass{tag_suffix}",
        n_live=n_live,
        n_batch=50,
        iterations_per_update=5000,
        number_of_cores=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
    )
    return search.fit(model=model, analysis=analysis)


def build_with_pointmass_fit(dataset, output_root: Path, n_live: int = 250,
                             tag_suffix: str = ""):
    """--part=with_pointmass: PowerLaw + shear + source + central PointMass."""
    import autofit as af
    import autolens as al

    lens = _build_lens_galaxy_model(slope_pinned=None, include_pointmass=True)
    source = _build_source_galaxy_model()
    model = af.Collection(galaxies=af.Collection(lens=lens, source=source))
    print(f"[RARC/with_pointmass] free params: {model.prior_count}", flush=True)

    analysis = al.AnalysisImaging(dataset=dataset)
    search = af.Nautilus(
        path_prefix=output_root,
        name="rarc_with_pointmass",
        unique_tag=f"rarc_with_pointmass{tag_suffix}",
        n_live=n_live,
        n_batch=50,
        iterations_per_update=5000,
        number_of_cores=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
    )
    return search.fit(model=model, analysis=analysis)


def build_with_kinematics_fit(dataset, output_root: Path, n_live: int = 250,
                              tag_suffix: str = ""):
    """--part=with_kinematics: STUB. Ships in v0.97 once the shared Jeans
    AnalysisKinematics class lands. See V095_PIPELINE_PLAN §4.3."""
    raise NotImplementedError(
        "with_kinematics requires a custom al.AnalysisKinematics subclass\n"
        "wrapping a Jeans-σ_v likelihood (Module 13 theory). Status:\n"
        "  - The shared analysis ships as Modules/10_Cluster_Computing/\n"
        "    scripts/_jeans_sigma_v.py in v0.97\n"
        "  - Consumed here AND by fit_example_quad_time_delay.py\n"
        "    --part=joint_fit_h0_kin (V095_PIPELINE_PLAN Stage 4 #3).\n"
        "  - Tracked as task #122 in the v0.96 design.\n"
        "\nFor v0.96 use --part=with_pointmass — γ′ + M_BH joint posterior\n"
        "with the degeneracy visible (the headline pedagogical result).\n"
        "When v0.97 lands, this --part will demonstrate the kinematic break."
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--part",
                   choices=("direct", "no_pointmass", "with_pointmass",
                            "with_kinematics", "all"),
                   default="all")
    p.add_argument("--output-root", type=Path,
                   default=Path("./output").resolve())
    p.add_argument("--dataset-root", type=Path, required=True,
                   help="Path containing image.fits + noise_map.fits + psf.fits")
    p.add_argument("--repo-root", type=Path, required=False,
                   help="Path to Learning_to_Autolens (slurm-driver compat).")
    p.add_argument("--mask-radius", type=float, default=1.8)
    p.add_argument("--n-live", type=int, default=200)
    args = p.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)

    import autolens as al
    print(f"PyAutoLens: {al.__version__}", flush=True)
    print(f"Output root:  {args.output_root}", flush=True)
    print(f"Dataset root: {args.dataset_root}", flush=True)
    print(f"Part:         {args.part}", flush=True)
    print(f"SLURM_JOB_ID: {os.environ.get('SLURM_JOB_ID', '(none)')}",
          flush=True)
    print(f"SLURM_CPUS_PER_TASK: "
          f"{os.environ.get('SLURM_CPUS_PER_TASK', '(none)')}", flush=True)

    t_start = time.time()
    dataset = load_dataset(args.dataset_root, mask_radius=args.mask_radius)
    print(f"Loaded dataset: {dataset.shape_native}, "
          f"{dataset.mask.pixels_in_mask} masked pixels", flush=True)

    if args.part in ("direct", "all"):
        build_direct_fit(dataset, args.output_root, n_live=args.n_live)
    if args.part in ("no_pointmass", "all"):
        build_no_pointmass_fit(dataset, args.output_root, n_live=150)
    if args.part in ("with_pointmass", "all"):
        build_with_pointmass_fit(dataset, args.output_root, n_live=250)
    if args.part == "with_kinematics":
        build_with_kinematics_fit(dataset, args.output_root)

    print(f"\nTotal wall time: {(time.time()-t_start)/3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
