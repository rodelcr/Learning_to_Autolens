"""
fit_example_compound_lens_zoo_climb.py — Cannon driver for the compound-lens
ladder beyond R2 (the canonical zoo fit). Implements R3 (multi-plane), R2_2src
(single-plane + 2-source), and R5 (multi-plane + 2-source) per the ladder
analysis in Examples/compound_lens_zoo/02_compound_lens_ladder.ipynb.

Pedagogical anchor:
    R2  (canonical zoo)         — fit_example_compound_lens_zoo.py
    R3  (multi-plane)           — THIS SCRIPT, --rung R3
    R2_2src (1-plane + 2src)    — THIS SCRIPT, --rung R2_2src
    R5  (multi-plane + 2src)    — THIS SCRIPT, --rung R5

R3 model:
    PowerLaw primary (slope free) at z_l1
  + Isothermal secondary (slope=2 fixed, centre seeded near truth) at z_l2
  + ExternalShear on primary plane
  + Sersic lens light, SersicCore source.
    => 19 free parameters.

R2_2src model:
    PowerLaw primary (slope free) at z_l1
  + ExternalShear
  + Sersic lens light
  + Two SersicCore source components.
    => 21 free parameters (single-plane).

R5 model:
    PowerLaw primary (slope free) at z_l1
  + Isothermal secondary at z_l2 (centre seeded near truth, einstein_radius
    Uniform(0, 1.0) — widened from R3's Uniform(0, 0.4) to avoid the Pattern-A
    rail-pinning observed on mock_4 R3 at 0.40)
  + ExternalShear on primary plane
  + Sersic lens light
  + Two SersicCore source components.
    => 26 free parameters (multi-plane).

The R5 rung is the post-climb diagnosis (notebook §11) for mocks 3 and 4:
mock_3 R3 hit Pattern E (lens_2.theta_E -> 0) and mock_4 R3 hit Pattern A
(prior rails). Both diagnoses point to the dominant residual being the
missing 2nd source, not the secondary deflector. R5 frees both at once.

Usage:
    python fit_example_compound_lens_zoo_climb.py --rung R5 --mock 3 \\
        --dataset-root /path/to/Examples/compound_lens_zoo/mocks \\
        --output-root  /path/to/output

    --mock <N>   — fit one mock at a time
    --rung R3 | R2_2src | R5

Cosmology held fixed at FlatLambdaCDM(70, 0.30) (matches truth for mocks 3,4,6;
mocks 2,5 have non-standard cosmologies and are NOT addressed by this driver).
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
        print(f"[CLIMB]   warning: post-fit visualize {tag} failed: {e}",
              flush=True)


def load_dataset(dataset_root: Path, mock_index: int, mask_radius: float = 2.7):
    import autolens as al
    dataset = al.Imaging.from_fits(
        data_path      = dataset_root / f"lenstronomy_mock_{mock_index}_image.fits",
        noise_map_path = dataset_root / f"lenstronomy_mock_{mock_index}_noise.fits",
        psf_path       = dataset_root / f"lenstronomy_mock_psf.fits",
        pixel_scales   = 0.05,
    )
    mask = al.Mask2D.circular(
        shape_native = dataset.shape_native,
        pixel_scales = dataset.pixel_scales,
        radius       = mask_radius,
    )
    return dataset.apply_mask(mask=mask)


def _load_truth(dataset_root: Path, mock_index: int) -> dict:
    return json.loads((dataset_root / f"truths_mock_{mock_index}.json").read_text())


def _find_secondary_truth(truths: dict) -> dict:
    """Pick out the secondary EPL from the truths kwargs_lens list.

    The list is [primary EPL, shear, secondary EPL]. The secondary is the
    second EPL by enumeration order (it's also the smaller theta_E one).
    """
    epls = [(i, kw) for i, (m, kw)
            in enumerate(zip(truths["lens_model_list"], truths["kwargs_lens"]))
            if m == "EPL"]
    if len(epls) < 2:
        raise ValueError(f"No secondary EPL found in truths (found {len(epls)} EPLs)")
    return epls[1][1]


# =============================================================================
# R3: multi-plane PowerLaw + Isothermal (secondary at z_l2) + ExternalShear
# =============================================================================
def build_R3_model(truths: dict):
    """R3: multi-plane PowerLaw primary + Isothermal secondary + ExternalShear."""
    import autofit as af
    import autolens as al

    z_l1 = truths["redshifts"]["lens_primary"]
    z_l2 = truths["redshifts"]["lens_secondary"]
    z_s  = truths["redshifts"]["source"]
    secondary_truth = _find_secondary_truth(truths)
    cx, cy = secondary_truth["center_x"], secondary_truth["center_y"]

    # Primary deflector (PowerLaw + Sersic light + ExternalShear)
    bulge = af.Model(al.lp.Sersic)
    bulge.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.5)
    bulge.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.5)
    bulge.intensity        = af.LogUniformPrior(lower_limit=1e-6, upper_limit=1e6)
    bulge.effective_radius = af.UniformPrior(lower_limit=0.1, upper_limit=5.0)
    bulge.sersic_index     = af.UniformPrior(lower_limit=0.5, upper_limit=8.0)
    bulge.ell_comps.ell_comps_0 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)
    bulge.ell_comps.ell_comps_1 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)

    mass = af.Model(al.mp.PowerLaw)
    mass.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.3)
    mass.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.3)
    mass.einstein_radius = af.UniformPrior(lower_limit=0.5, upper_limit=3.5)
    mass.slope           = af.TruncatedGaussianPrior(
        mean=2.0, sigma=0.3, lower_limit=1.5, upper_limit=2.7)
    mass.ell_comps.ell_comps_0 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)
    mass.ell_comps.ell_comps_1 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)

    shear = af.Model(al.mp.ExternalShear)
    shear.gamma_1 = af.GaussianPrior(mean=0.0, sigma=0.1)
    shear.gamma_2 = af.GaussianPrior(mean=0.0, sigma=0.1)

    lens_1 = af.Model(al.Galaxy, redshift=z_l1, bulge=bulge, mass=mass, shear=shear)

    # Secondary deflector — Isothermal, centre seeded near truth, theta_E free.
    mass_2 = af.Model(al.mp.Isothermal)
    mass_2.centre.centre_0 = af.GaussianPrior(mean=cx, sigma=0.15)
    mass_2.centre.centre_1 = af.GaussianPrior(mean=cy, sigma=0.15)
    mass_2.einstein_radius = af.UniformPrior(lower_limit=0.0, upper_limit=0.4)
    mass_2.ell_comps.ell_comps_0 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)
    mass_2.ell_comps.ell_comps_1 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)
    lens_2 = af.Model(al.Galaxy, redshift=z_l2, mass=mass_2)

    # Source — single SersicCore at z_s.
    src = af.Model(al.lp.SersicCore)
    src.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.5)
    src.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.5)
    src.intensity        = af.LogUniformPrior(lower_limit=1e-3, upper_limit=1e3)
    src.effective_radius = af.UniformPrior(lower_limit=0.0, upper_limit=2.0)
    src.sersic_index     = af.UniformPrior(lower_limit=0.5, upper_limit=4.0)
    src.ell_comps.ell_comps_0 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)
    src.ell_comps.ell_comps_1 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)

    source = af.Model(al.Galaxy, redshift=z_s, bulge=src)

    return af.Collection(galaxies=af.Collection(
        lens_1=lens_1, lens_2=lens_2, source=source))


# =============================================================================
# R2_2src: single-plane PowerLaw + shear + Sersic light + TWO SersicCore sources
# =============================================================================
def build_R2_2src_model(truths: dict):
    """R2_2src: single-plane PowerLaw + shear, two source components."""
    import autofit as af
    import autolens as al

    z_l1 = truths["redshifts"]["lens_primary"]
    z_s  = truths["redshifts"]["source"]

    bulge = af.Model(al.lp.Sersic)
    bulge.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.5)
    bulge.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.5)
    bulge.intensity        = af.LogUniformPrior(lower_limit=1e-6, upper_limit=1e6)
    bulge.effective_radius = af.UniformPrior(lower_limit=0.1, upper_limit=5.0)
    bulge.sersic_index     = af.UniformPrior(lower_limit=0.5, upper_limit=8.0)
    bulge.ell_comps.ell_comps_0 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)
    bulge.ell_comps.ell_comps_1 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)

    mass = af.Model(al.mp.PowerLaw)
    mass.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.3)
    mass.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.3)
    mass.einstein_radius = af.UniformPrior(lower_limit=0.5, upper_limit=3.5)
    mass.slope           = af.TruncatedGaussianPrior(
        mean=2.0, sigma=0.3, lower_limit=1.5, upper_limit=2.7)
    mass.ell_comps.ell_comps_0 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)
    mass.ell_comps.ell_comps_1 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)

    shear = af.Model(al.mp.ExternalShear)
    shear.gamma_1 = af.GaussianPrior(mean=0.0, sigma=0.1)
    shear.gamma_2 = af.GaussianPrior(mean=0.0, sigma=0.1)

    lens = af.Model(al.Galaxy, redshift=z_l1, bulge=bulge, mass=mass, shear=shear)

    # Two source components, both SersicCore, distinct centres.
    def _src(seed_centre):
        s = af.Model(al.lp.SersicCore)
        s.centre.centre_0 = af.GaussianPrior(mean=seed_centre[0], sigma=0.5)
        s.centre.centre_1 = af.GaussianPrior(mean=seed_centre[1], sigma=0.5)
        s.intensity        = af.LogUniformPrior(lower_limit=1e-3, upper_limit=1e3)
        s.effective_radius = af.UniformPrior(lower_limit=0.0, upper_limit=2.0)
        s.sersic_index     = af.UniformPrior(lower_limit=0.5, upper_limit=4.0)
        s.ell_comps.ell_comps_0 = af.TruncatedGaussianPrior(
            mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)
        s.ell_comps.ell_comps_1 = af.TruncatedGaussianPrior(
            mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)
        return s

    # Seed two source-plane centres at offset positions (truth-informed, but
    # widened to a 0.5" Gaussian so the chain can find them).
    src_kwargs_list = truths.get("kwargs_source", [])
    seeds = []
    for kw in src_kwargs_list[:2]:
        seeds.append((kw.get("center_x", 0.0), kw.get("center_y", 0.0)))
    while len(seeds) < 2:
        seeds.append((0.0, 0.0))

    src_a = _src(seeds[0])
    src_b = _src(seeds[1])
    source = af.Model(al.Galaxy, redshift=z_s, bulge=src_a, disk=src_b)

    return af.Collection(galaxies=af.Collection(lens=lens, source=source))


# =============================================================================
# R5: multi-plane PowerLaw + Isothermal secondary + ExternalShear + 2 sources
# =============================================================================
def build_R5_model(truths: dict):
    """R5: multi-plane (R3) + 2-source (R2_2src). Designed for mocks 3, 4.

    Differences from R3:
      - Two SersicCore source components (from R2_2src).
      - lens_2.einstein_radius prior widened to Uniform(0, 1.0) so the chain
        is free to either pin near zero (Pattern E) or land at truth without
        bumping a rail (Pattern A).
    """
    import autofit as af
    import autolens as al

    z_l1 = truths["redshifts"]["lens_primary"]
    z_l2 = truths["redshifts"]["lens_secondary"]
    z_s  = truths["redshifts"]["source"]
    secondary_truth = _find_secondary_truth(truths)
    cx, cy = secondary_truth["center_x"], secondary_truth["center_y"]

    # Primary deflector (PowerLaw + Sersic light + ExternalShear)
    bulge = af.Model(al.lp.Sersic)
    bulge.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.5)
    bulge.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.5)
    bulge.intensity        = af.LogUniformPrior(lower_limit=1e-6, upper_limit=1e6)
    bulge.effective_radius = af.UniformPrior(lower_limit=0.1, upper_limit=5.0)
    bulge.sersic_index     = af.UniformPrior(lower_limit=0.5, upper_limit=8.0)
    bulge.ell_comps.ell_comps_0 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)
    bulge.ell_comps.ell_comps_1 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)

    mass = af.Model(al.mp.PowerLaw)
    mass.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.3)
    mass.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.3)
    mass.einstein_radius = af.UniformPrior(lower_limit=0.5, upper_limit=3.5)
    mass.slope           = af.TruncatedGaussianPrior(
        mean=2.0, sigma=0.3, lower_limit=1.5, upper_limit=2.7)
    mass.ell_comps.ell_comps_0 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)
    mass.ell_comps.ell_comps_1 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)

    shear = af.Model(al.mp.ExternalShear)
    shear.gamma_1 = af.GaussianPrior(mean=0.0, sigma=0.1)
    shear.gamma_2 = af.GaussianPrior(mean=0.0, sigma=0.1)

    lens_1 = af.Model(al.Galaxy, redshift=z_l1, bulge=bulge, mass=mass, shear=shear)

    # Secondary deflector — widened theta_E prior (R5 vs R3 caveat).
    mass_2 = af.Model(al.mp.Isothermal)
    mass_2.centre.centre_0 = af.GaussianPrior(mean=cx, sigma=0.15)
    mass_2.centre.centre_1 = af.GaussianPrior(mean=cy, sigma=0.15)
    mass_2.einstein_radius = af.UniformPrior(lower_limit=0.0, upper_limit=1.0)
    mass_2.ell_comps.ell_comps_0 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)
    mass_2.ell_comps.ell_comps_1 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)
    lens_2 = af.Model(al.Galaxy, redshift=z_l2, mass=mass_2)

    # Two source components, both SersicCore, distinct centres seeded near truth.
    def _src(seed_centre):
        s = af.Model(al.lp.SersicCore)
        s.centre.centre_0 = af.GaussianPrior(mean=seed_centre[0], sigma=0.5)
        s.centre.centre_1 = af.GaussianPrior(mean=seed_centre[1], sigma=0.5)
        s.intensity        = af.LogUniformPrior(lower_limit=1e-3, upper_limit=1e3)
        s.effective_radius = af.UniformPrior(lower_limit=0.0, upper_limit=2.0)
        s.sersic_index     = af.UniformPrior(lower_limit=0.5, upper_limit=4.0)
        s.ell_comps.ell_comps_0 = af.TruncatedGaussianPrior(
            mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)
        s.ell_comps.ell_comps_1 = af.TruncatedGaussianPrior(
            mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)
        return s

    src_kwargs_list = truths.get("kwargs_source", [])
    seeds = []
    for kw in src_kwargs_list[:2]:
        seeds.append((kw.get("center_x", 0.0), kw.get("center_y", 0.0)))
    while len(seeds) < 2:
        seeds.append((0.0, 0.0))

    src_a = _src(seeds[0])
    src_b = _src(seeds[1])
    source = af.Model(al.Galaxy, redshift=z_s, bulge=src_a, disk=src_b)

    return af.Collection(galaxies=af.Collection(
        lens_1=lens_1, lens_2=lens_2, source=source))


# =============================================================================
# Driver
# =============================================================================
def build_fit(dataset, output_root: Path, mock_index: int, truths: dict,
              rung: str, n_live: int = 250):
    import autofit as af
    import autolens as al

    print(f"\n[CLIMB/mock_{mock_index}/{rung}] starting fit",
          flush=True)
    print(f"  z_l1={truths['redshifts']['lens_primary']}, "
          f"z_l2={truths['redshifts'].get('lens_secondary', '—')}, "
          f"z_s={truths['redshifts']['source']}", flush=True)

    if rung == "R3":
        model = build_R3_model(truths)
        unique_tag = f"mock_{mock_index}_R3_powerlaw_iso_shear_multiplane"
    elif rung == "R2_2src":
        model = build_R2_2src_model(truths)
        unique_tag = f"mock_{mock_index}_R2_2src_powerlaw_shear_sersic_2srcs"
    elif rung == "R5":
        model = build_R5_model(truths)
        unique_tag = f"mock_{mock_index}_R5_powerlaw_iso_shear_multiplane_2srcs"
    else:
        raise ValueError(f"unknown rung: {rung!r}")

    analysis = al.AnalysisImaging(dataset=dataset, use_jax=False)

    search = af.Nautilus(
        path_prefix=output_root,
        name=f"mock_{mock_index}_{rung}",
        unique_tag=unique_tag,
        n_live=n_live,
        n_batch=50,
        iterations_per_update=15000,
        number_of_cores=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
    )

    t0 = time.time()
    result = search.fit(model=model, analysis=analysis)
    print(f"[CLIMB/mock_{mock_index}/{rung}] done in {(time.time()-t0)/60:.1f} min",
          flush=True)
    _force_visualize(analysis, result, tag=f"mock_{mock_index}_{rung}")
    print(result.info, flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rung", choices=["R3", "R2_2src", "R5"], required=True)
    p.add_argument("--mock", type=str, default="all",
                   help="2, 3, 4, 5, 6, or 'all' (filtered per --rung default)")
    p.add_argument("--output-root", type=Path,
                   default=Path("./output").resolve())
    p.add_argument("--dataset-root", type=Path, required=True,
                   help="Path containing lenstronomy_mock_*.fits and truths_*.json")
    p.add_argument("--repo-root", type=Path, required=False,
                   help="Path to Learning_to_Autolens (slurm-driver compat)")
    p.add_argument("--n-live", type=int, default=250)
    args = p.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)

    # Defaults per rung:
    #   R3      -> mocks 3, 4 (multi-plane signal mocks)
    #   R2_2src -> mock 6     (single-deflector + 2-source mock)
    #   R5      -> mocks 3, 4 (multi-plane + 2-source — post-climb diagnosis)
    if args.mock == "all":
        if args.rung in ("R3", "R5"):
            mocks_to_fit = [3, 4]
        else:
            mocks_to_fit = [6]
    else:
        mocks_to_fit = [int(args.mock)]

    import autolens as al
    print(f"PyAutoLens: {al.__version__}", flush=True)
    print(f"Rung:           {args.rung}", flush=True)
    print(f"Mocks to fit:   {mocks_to_fit}", flush=True)
    print(f"Output root:    {args.output_root}", flush=True)
    print(f"Dataset root:   {args.dataset_root}", flush=True)

    t_start = time.time()
    for n in mocks_to_fit:
        truths = _load_truth(args.dataset_root, mock_index=n)
        dataset = load_dataset(args.dataset_root, mock_index=n)
        print(f"\nLoaded mock_{n}: shape={dataset.shape_native}, "
              f"pixels_in_mask={dataset.mask.pixels_in_mask}", flush=True)
        build_fit(dataset, args.output_root, mock_index=n, truths=truths,
                  rung=args.rung, n_live=args.n_live)

    print(f"\nTotal wall time: {(time.time()-t_start)/3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
