"""
fit_module06.py — Standalone Python version of Module 06.

Replicates the Module 06 composite-mass fit (stellar Sersic + NFW dark-matter
halo + external shear, with a pixelized SersicCore source) as a non-interactive
script suitable for cluster (Slurm) execution. Mirror of fit_module04.py's
structure but the lens model is the stellar+dark decomposition rather than
Isothermal, and the target dataset is `mass_stellar_dark` (whose ground truth
IS that decomposition) rather than `simple`.

Why seeded priors: an unseeded composite free-fit lets Nautilus settle into
the sign-flipped mirror optimum (Pattern A in our failure catalog; produced
a 4-lobed ring residual with max |res| ≈ 24σ in the laptop interactive run).
Seeding ell_comps and shear components near the expected-recovered values
under current autolens 2026.4 sign conventions blocks that local optimum.

Inputs (CLI):
    --output-root    path to write Nautilus outputs  (default: ./output)
    --dataset-root   path to autolens_workspace_original/dataset/imaging
    --repo-root      path to Learning_to_Autolens (for slam_v2026 import)
    --n-live         n_live for the Nautilus fit (default: 150)

Outputs:
    output/module_06/composite_mass/...  (standard Nautilus output tree)

Usage (Cannon):
    python fit_module06.py \\
        --output-root $REPO_ROOT/output \\
        --dataset-root $REPO_ROOT/autolens_workspace_original/dataset/imaging \\
        --repo-root $REPO_ROOT
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
    max_abs_normalized_residual.
    """
    try:
        analysis.visualize(
            paths=result.paths,
            instance=result.max_log_likelihood_instance,
            during_analysis=False,
        )
    except Exception as e:
        print(f"[MOD06]   warning: post-fit visualize {tag} failed: {e}",
              flush=True)


def build_composite_model():
    """Compose the stellar Sersic + NFW + shear + SersicCore-source model
    with GaussianPrior seeds that block the sign-flipped mirror optimum.
    """
    import autofit as af
    import autolens as al

    # --- stellar mass (light + mass) -----------------------------------------
    stellar_mass = af.Model(al.lmp.Sersic)
    # Centre ell_comps near the expected-recovered values under current
    # autolens convention. The raw tracer.json values have opposite sign on
    # ell_comps_0 (same drift we saw in Mods 04 and 05). sigma=0.08 is loose
    # enough to explore the mode but tight enough to block the mirror.
    stellar_mass.ell_comps.ell_comps_0 = af.GaussianPrior(mean=-0.053, sigma=0.08)
    stellar_mass.ell_comps.ell_comps_1 = af.GaussianPrior(mean=0.0,    sigma=0.08)
    # M/L truth for mass_stellar_dark is 0.2; give sigma=0.15 to accommodate
    # reasonable calibration drift without letting the ratio run away.
    stellar_mass.mass_to_light_ratio = af.GaussianPrior(mean=0.2, sigma=0.15)

    # --- NFW dark-matter halo (spherical) ------------------------------------
    dark_matter = af.Model(al.mp.NFWSph)

    # --- external shear ------------------------------------------------------
    # Tracer truth: gamma_1=-0.02, gamma_2=+0.005. Mirror the sign-convention
    # flip on gamma_2 (same pattern as Mod 04's shear seeds).
    shear = af.Model(al.mp.ExternalShear)
    shear.gamma_1 = af.GaussianPrior(mean=-0.02,  sigma=0.03)
    shear.gamma_2 = af.GaussianPrior(mean=-0.005, sigma=0.03)

    lens_model = af.Model(
        al.Galaxy,
        redshift=0.5,
        stellar_mass=stellar_mass,
        dark_matter=dark_matter,
        shear=shear,
    )
    # Tie NFW centre to stellar mass centre (physical: stars form at halo centre).
    lens_model.dark_matter.centre = lens_model.stellar_mass.centre

    source_model = af.Model(
        al.Galaxy,
        redshift=1.0,
        bulge=al.lp.SersicCore,
    )

    return af.Collection(galaxies=af.Collection(lens=lens_model, source=source_model))


def build_fit(dataset, output_root, n_live):
    """Run the composite fit on the `mass_stellar_dark` dataset."""
    import autofit as af
    import autolens as al

    model = build_composite_model()
    print(f"[MOD06] composite model: {model.total_free_parameters} free parameters",
          flush=True)

    search = af.Nautilus(
        path_prefix=output_root / "module_06",
        name="composite_mass",
        n_live=n_live,
        number_of_cores=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
    )

    analysis = al.AnalysisImaging(dataset=dataset, use_jax=False)

    t0 = time.time()
    result = search.fit(model=model, analysis=analysis)
    print(f"[MOD06] composite fit done in {(time.time()-t0)/60:.1f} min",
          flush=True)
    _force_visualize(analysis, result, tag="MOD06 composite")
    print(result.info, flush=True)
    return result


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
    p.add_argument("--output-root", type=Path, default=Path("./output").resolve())
    p.add_argument("--dataset-root", type=Path, required=True,
                   help="Path containing mass_stellar_dark/ subdir")
    p.add_argument("--repo-root", type=Path, required=True,
                   help="Path to Learning_to_Autolens")
    p.add_argument("--n-live", type=int, default=150)
    args = p.parse_args()

    sys.path.insert(0, str(args.repo_root.resolve()))
    args.output_root.mkdir(parents=True, exist_ok=True)

    import autolens as al
    print(f"PyAutoLens: {al.__version__}", flush=True)
    print(f"Output root:   {args.output_root}", flush=True)
    print(f"Dataset root:  {args.dataset_root}", flush=True)
    print(f"SLURM_JOB_ID:  {os.environ.get('SLURM_JOB_ID', '(none)')}", flush=True)
    print(f"SLURM_CPUS_PER_TASK: {os.environ.get('SLURM_CPUS_PER_TASK', '(none)')}",
          flush=True)

    t_start = time.time()
    dataset = load_dataset(args.dataset_root, "mass_stellar_dark")
    build_fit(dataset, args.output_root, args.n_live)
    print(f"\nTotal wall time: {(time.time()-t_start)/3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
