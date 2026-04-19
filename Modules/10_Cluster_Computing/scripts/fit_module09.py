"""
fit_module09.py — Standalone Python version of Module 09 (MGE + full SLaM).

Five-stage production SLaM pipeline using MGE light profiles on the v2026
`simple` dataset:

    Stage 1 (SOURCE LP):   MGE lens (2x20) + MGE source (1x20) + Isothermal+shear
    Stage 2 (SOURCE PIX 1): RectangularAdaptDensity(28,28) + reg.Adapt
    Stage 3 (SOURCE PIX 2): RectangularAdaptImage(28,28)    + reg.Adapt
    Stage 4 (LIGHT LP):     re-fit MGE lens light with mass/source fixed
    Stage 5 (MASS TOTAL):   PowerLaw mass + shear (source/light fixed)

This is the v2026 canonical pattern from
`autolens_workspace_latest/scripts/guides/modeling/slam_start_here.py`.
Uses `SafeAnalysisImaging` on pixelized stages to trap ill-conditioned
inversions.

Dataset note: Module 09 uses the v2026 `autolens_workspace_latest` copy
(not `_original`). That path needs to be reachable from --dataset-root.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


def build(dataset_root, output_root, dataset_name, mask_radius, n_live, ncores):
    import numpy as np
    import autofit as af
    import autolens as al

    path = dataset_root / dataset_name
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
    dataset = dataset.apply_mask(mask=mask)

    class SafeAnalysisImaging(al.AnalysisImaging):
        def log_likelihood_function(self, instance):
            try:
                return super().log_likelihood_function(instance)
            except (np.linalg.LinAlgError, Exception) as e:
                if "singular" in str(e).lower() or "positive definite" in str(e).lower():
                    return -1.0e99
                raise

    slam_prefix = output_root / "module_09" / "slam"

    def _nautilus(name, nl):
        return af.Nautilus(path_prefix=slam_prefix, name=name,
                           n_live=nl, number_of_cores=ncores)

    # ---- Stage 1: SOURCE LP --------------------------------------------------
    print("[MOD09] Stage 1: SOURCE LP", flush=True)
    lens_bulge_1 = al.model_util.mge_model_from(
        mask_radius=mask_radius, total_gaussians=20,
        gaussian_per_basis=2, centre_prior_is_uniform=True,
    )
    source_bulge_1 = al.model_util.mge_model_from(
        mask_radius=mask_radius, total_gaussians=20,
        gaussian_per_basis=1, centre_prior_is_uniform=False,
    )
    model_1 = af.Collection(galaxies=af.Collection(
        lens=af.Model(al.Galaxy, redshift=0.5,
                      bulge=lens_bulge_1, disk=None,
                      mass=af.Model(al.mp.Isothermal),
                      shear=af.Model(al.mp.ExternalShear)),
        source=af.Model(al.Galaxy, redshift=1.0, bulge=source_bulge_1),
    ))
    t0 = time.time()
    source_lp_result = _nautilus("source_lp[1]", n_live["s1"]).fit(
        model=model_1, analysis=al.AnalysisImaging(dataset=dataset, use_jax=False))
    print(f"[MOD09] Stage 1 done in {(time.time()-t0)/60:.1f} min; "
          f"θ_E = {source_lp_result.instance.galaxies.lens.mass.einstein_radius:.3f}",
          flush=True)

    # ---- Stage 2: SOURCE PIX 1 -----------------------------------------------
    print("[MOD09] Stage 2: SOURCE PIX 1", flush=True)
    adapt_images_1 = al.AdaptImages(
        galaxy_name_image_dict=al.galaxy_name_image_dict_via_result_from(
            result=source_lp_result)
    )
    mass_2 = al.util.chaining.mass_from(
        mass=source_lp_result.model.galaxies.lens.mass,
        mass_result=source_lp_result.model.galaxies.lens.mass,
        unfix_mass_centre=True,
    )
    model_2 = af.Collection(galaxies=af.Collection(
        lens=af.Model(al.Galaxy, redshift=0.5,
                      bulge=source_lp_result.instance.galaxies.lens.bulge,
                      disk=None, mass=mass_2,
                      shear=source_lp_result.model.galaxies.lens.shear),
        source=af.Model(al.Galaxy, redshift=1.0,
                        pixelization=af.Model(al.Pixelization,
                            mesh=af.Model(al.mesh.RectangularAdaptDensity, shape=(28, 28)),
                            regularization=al.reg.Adapt)),
    ))
    t0 = time.time()
    source_pix_result_1 = _nautilus("source_pix[1]", n_live["s2"]).fit(
        model=model_2,
        analysis=SafeAnalysisImaging(
            dataset=dataset, adapt_images=adapt_images_1,
            positions_likelihood_list=[source_lp_result.positions_likelihood_from(
                factor=3.0, minimum_threshold=0.2)],
            use_jax=False))
    print(f"[MOD09] Stage 2 done in {(time.time()-t0)/60:.1f} min", flush=True)

    # ---- Stage 3: SOURCE PIX 2 -----------------------------------------------
    print("[MOD09] Stage 3: SOURCE PIX 2", flush=True)
    adapt_images_2 = al.AdaptImages(
        galaxy_name_image_dict=al.galaxy_name_image_dict_via_result_from(
            result=source_pix_result_1)
    )
    model_3 = af.Collection(galaxies=af.Collection(
        lens=af.Model(al.Galaxy, redshift=0.5,
                      bulge=source_lp_result.instance.galaxies.lens.bulge,
                      disk=None,
                      mass=source_pix_result_1.instance.galaxies.lens.mass,
                      shear=source_pix_result_1.instance.galaxies.lens.shear),
        source=af.Model(al.Galaxy, redshift=1.0,
                        pixelization=af.Model(al.Pixelization,
                            mesh=af.Model(al.mesh.RectangularAdaptImage, shape=(28, 28)),
                            regularization=al.reg.Adapt)),
    ))
    t0 = time.time()
    source_pix_result_2 = _nautilus("source_pix[2]", n_live["s3"]).fit(
        model=model_3,
        analysis=SafeAnalysisImaging(
            dataset=dataset, adapt_images=adapt_images_2, use_jax=False))
    print(f"[MOD09] Stage 3 done in {(time.time()-t0)/60:.1f} min", flush=True)

    # ---- Stage 4: LIGHT LP ---------------------------------------------------
    print("[MOD09] Stage 4: LIGHT LP", flush=True)
    lens_bulge_4 = al.model_util.mge_model_from(
        mask_radius=mask_radius, total_gaussians=20,
        gaussian_per_basis=2, centre_prior_is_uniform=True,
    )
    source_4 = al.util.chaining.source_custom_model_from(
        result=source_pix_result_2, source_is_model=False)
    model_4 = af.Collection(galaxies=af.Collection(
        lens=af.Model(al.Galaxy, redshift=0.5,
                      bulge=lens_bulge_4, disk=None,
                      mass=source_pix_result_1.instance.galaxies.lens.mass,
                      shear=source_pix_result_1.instance.galaxies.lens.shear),
        source=source_4,
    ))
    adapt_images_4 = al.AdaptImages(
        galaxy_name_image_dict=al.galaxy_name_image_dict_via_result_from(
            result=source_pix_result_1)
    )
    t0 = time.time()
    light_result = _nautilus("light[1]", n_live["s4"]).fit(
        model=model_4,
        analysis=al.AnalysisImaging(
            dataset=dataset, adapt_images=adapt_images_4, use_jax=False))
    print(f"[MOD09] Stage 4 done in {(time.time()-t0)/60:.1f} min", flush=True)

    # ---- Stage 5: MASS TOTAL -------------------------------------------------
    print("[MOD09] Stage 5: MASS TOTAL (PowerLaw)", flush=True)
    mass_5 = al.util.chaining.mass_from(
        mass=af.Model(al.mp.PowerLaw),
        mass_result=source_pix_result_1.model.galaxies.lens.mass,
        unfix_mass_centre=True,
    )
    source_5 = al.util.chaining.source_from(result=source_pix_result_2)
    model_5 = af.Collection(galaxies=af.Collection(
        lens=af.Model(al.Galaxy, redshift=0.5,
                      bulge=light_result.instance.galaxies.lens.bulge,
                      disk=None, mass=mass_5,
                      shear=source_pix_result_1.model.galaxies.lens.shear),
        source=source_5,
    ))
    adapt_images_5 = al.AdaptImages(
        galaxy_name_image_dict=al.galaxy_name_image_dict_via_result_from(
            result=source_pix_result_1)
    )
    t0 = time.time()
    mass_result = _nautilus("mass_total[1]", n_live["s5"]).fit(
        model=model_5,
        analysis=SafeAnalysisImaging(
            dataset=dataset, adapt_images=adapt_images_5,
            positions_likelihood_list=[source_pix_result_2.positions_likelihood_from(
                factor=3.0, minimum_threshold=0.2)],
            use_jax=False))
    print(f"[MOD09] Stage 5 done in {(time.time()-t0)/60:.1f} min", flush=True)
    print(mass_result.info, flush=True)
    return mass_result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, required=True,
                   help="Parent of simple/ (typically "
                        "autolens_workspace_latest/dataset/imaging)")
    p.add_argument("--output-root", type=Path, default=Path("./output").resolve())
    p.add_argument("--dataset-name", default="simple")
    p.add_argument("--mask-radius", type=float, default=3.0)
    # n_live per stage — defaults match Module 09 notebook
    p.add_argument("--n-live-s1", type=int, default=200)
    p.add_argument("--n-live-s2", type=int, default=150)
    p.add_argument("--n-live-s3", type=int, default=100,
                   help="was 75 in notebook; bumped to 100 to avoid LinAlgError")
    p.add_argument("--n-live-s4", type=int, default=150)
    p.add_argument("--n-live-s5", type=int, default=150)
    args = p.parse_args()

    sys.path.insert(0, str(args.repo_root.resolve()))
    args.output_root.mkdir(parents=True, exist_ok=True)

    import autolens as al
    ncores = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    print(f"PyAutoLens: {al.__version__}", flush=True)
    print(f"SLURM_JOB_ID: {os.environ.get('SLURM_JOB_ID', '(none)')}", flush=True)
    print(f"Cores: {ncores}", flush=True)
    print(f"Dataset: {args.dataset_root / args.dataset_name}", flush=True)
    print(f"Output:  {args.output_root}", flush=True)

    n_live = {"s1": args.n_live_s1, "s2": args.n_live_s2,
              "s3": args.n_live_s3, "s4": args.n_live_s4,
              "s5": args.n_live_s5}
    t0 = time.time()
    build(args.dataset_root, args.output_root, args.dataset_name,
          args.mask_radius, n_live, ncores)
    print(f"\nTotal wall time: {(time.time()-t0)/3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
