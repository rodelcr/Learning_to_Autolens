"""
fit_module05.py — Standalone Python version of Module 05.

Two-search pixelized-source pipeline on simple__no_lens_light.

    Search 1 (parametric):  SIE + shear + Sersic source   (n_live=100)
    Search 2 (pixelized):   SIE + shear + RectangularAdaptDensity(40x40)
                            + Constant regularization       (n_live=80)

Note: the Mod 05 notebook uses mesh=(100,100) for pedagogical impact, but a
Nautilus search re-evaluates the pixelized inversion thousands of times and
the cost is O(N_pix^3) per eval. 100x100 = 10 000 pixels is tractable for a
single forward pass in the notebook but makes the cluster fit run for >2 h
with no progress (memory-bound thrashing on a 32 GB node). 40x40 = 1600
pixels is ~240x cheaper per eval, still demonstrates the "high-res adaptive"
idea from the tutorial, and matches the order of magnitude that Mod 09's
MGE SLaM uses (28x28 in slam_v2026). Original config preserved in git log.

Search 2 uses `SafeAnalysisImaging` (defined inline) to demote LinAlgErrors from
ill-conditioned inversions to finite bad-likelihoods instead of crashes.

Usage (see Module 10 notebook for full walk-through):
    python fit_module05.py \\
        --repo-root    /path/to/Learning_to_Autolens \\
        --dataset-root /path/to/autolens_workspace_original/dataset/imaging \\
        --output-root  $SCRATCH/learning_to_autolens/output

Nautilus auto-resumes from any existing checkpoint.hdf5 inside output-root.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


def build(dataset_root, output_root, dataset_name,
          n_live_1, n_live_2, ncores):
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
        radius=3.0,
    )
    dataset = dataset.apply_mask(mask=mask)

    class SafeAnalysisImaging(al.AnalysisImaging):
        """Catches LinAlgError from ill-conditioned source inversions."""
        def log_likelihood_function(self, instance):
            try:
                return super().log_likelihood_function(instance)
            except (np.linalg.LinAlgError, Exception) as e:
                if "singular" in str(e).lower() or "positive definite" in str(e).lower():
                    return -1.0e99
                raise

    # ---- Search 1: Parametric source -----------------------------------------
    print("[MOD05] Search 1: SIE+shear + Sersic source", flush=True)
    model_1 = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(al.Galaxy, redshift=0.5,
                          mass=al.mp.Isothermal, shear=al.mp.ExternalShear),
            source=af.Model(al.Galaxy, redshift=1.0, bulge=al.lp.Sersic),
        )
    )
    model_1.galaxies.lens.mass.einstein_radius = af.UniformPrior(
        lower_limit=0.5, upper_limit=3.0)

    search_1 = af.Nautilus(
        path_prefix=output_root / "module_05",
        name="search1_parametric_source",
        n_live=n_live_1,
        number_of_cores=ncores,
    )
    analysis_1 = al.AnalysisImaging(dataset=dataset, use_jax=False)
    t0 = time.time()
    result_1 = search_1.fit(model=model_1, analysis=analysis_1)
    print(f"[MOD05] Search 1 done in {(time.time()-t0)/60:.1f} min; "
          f"θ_E = {result_1.instance.galaxies.lens.mass.einstein_radius:.3f}\"",
          flush=True)

    # ---- Search 2: Pixelized source -----------------------------------------
    print("[MOD05] Search 2: pixelized source (RectangularAdaptDensity 100x100)",
          flush=True)
    model_2 = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(al.Galaxy, redshift=0.5,
                          mass=result_1.model.galaxies.lens.mass,
                          shear=result_1.model.galaxies.lens.shear),
            source=af.Model(
                al.Galaxy, redshift=1.0,
                pixelization=af.Model(
                    al.Pixelization,
                    mesh=al.mesh.RectangularAdaptDensity(shape=(40, 40)),
                    regularization=al.reg.Constant,
                ),
            ),
        )
    )

    positions = al.Grid2DIrregular(
        al.from_json(file_path=path / "positions.json")
    )
    positions_lh = al.PositionsLH(positions=positions, threshold=1.0)

    search_2 = af.Nautilus(
        path_prefix=output_root / "module_05",
        name="search2_pixelized_source",
        n_live=n_live_2,
        number_of_cores=ncores,
    )
    analysis_2 = SafeAnalysisImaging(
        dataset=dataset,
        positions_likelihood_list=[positions_lh],
        settings=al.Settings(
            use_border_relocator=True,
            # NNLS (positive-only) is both faster than the signed solve and
            # physically correct (source flux is non-negative). Mod 09 uses
            # True by default via slam_v2026; the original False here was a
            # holdover from an earlier notebook draft.
            use_positive_only_solver=True,
        ),
        use_jax=False,
    )
    t0 = time.time()
    result_2 = search_2.fit(model=model_2, analysis=analysis_2)
    print(f"[MOD05] Search 2 done in {(time.time()-t0)/60:.1f} min", flush=True)
    print(result_2.info, flush=True)
    return result_1, result_2


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, required=True,
                   help="Parent of simple__no_lens_light/ (typically "
                        "autolens_workspace_original/dataset/imaging)")
    p.add_argument("--output-root", type=Path, default=Path("./output").resolve())
    p.add_argument("--dataset-name", default="simple__no_lens_light")
    p.add_argument("--n-live-1", type=int, default=100)
    p.add_argument("--n-live-2", type=int, default=80)
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

    t0 = time.time()
    build(args.dataset_root, args.output_root, args.dataset_name,
          args.n_live_1, args.n_live_2, ncores)
    print(f"\nTotal wall time: {(time.time()-t0)/3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
