"""
fit_template.py — Starter script for converting any Learning_to_Autolens
notebook into a standalone, cluster-ready fit.

Intended workflow:

    1. Copy this file:
         cp fit_template.py fit_module<NN>.py        # NN = your module number

    2. Fill in the `build(...)` function with your model + search calls. The
       structure is already scaffolded; you just need to replace the
       example model with your own and add more searches if you're chaining
       or running SLaM.

    3. Wire it into `submit_cannon.slurm` by adding a `MODULE=<NN>` case.

    4. Include your module's dataset dir in `push_to_cannon.sh` and its
       results dir in `pull_from_cannon.sh`.

    5. In your module notebook, add a "Viewing pre-computed results from
       the Cannon cluster" cell (copy from Modules 04 / 05 / 09 notebooks).

Conventions (all three shipped fit_module*.py follow these):

    - Arguments: `--repo-root`, `--dataset-root`, `--output-root` (required),
      plus any per-search `--n-live-*` you want to expose.
    - Read `SLURM_CPUS_PER_TASK` for Nautilus `number_of_cores`.
    - Every `print(..., flush=True)` so Slurm logs show progress live.
    - Call `_force_visualize(analysis, result, tag=...)` after every
      `search.fit()` so export_results.py can read image/fit.fits on
      a resumed search.
    - Let Nautilus auto-resume: same `path_prefix`, `name`, `unique_tag`,
      and identical model hash → existing checkpoint.hdf5 is picked up.
    - Fail fast if PYAUTOFIT_TEST_MODE is set (inherited from
      slam_v2026 if you import it; check_install.py also blocks it).

Canonical submit:

    sbatch --export=ALL,MODULE=<NN> submit_cannon.slurm

Then after the job emails you that it's done:

    bash pull_from_cannon.sh --go
    # ...and open Modules/<NN>_<name>/<NN>_<snake>.ipynb; the results/
    # viewer cell will load Modules/<NN>_<name>/results/<search>/
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
#  Helper: force-regenerate image/fit.fits after a completed search
# ---------------------------------------------------------------------------
# When a search is resumed from a Nautilus checkpoint, autolens often skips
# the visualization regeneration step — leaving image/fit.fits absent.
# export_results.py pulls chi_squared_per_pixel and max_abs_normalized_residual
# out of that file, so missing it → null residual fields in summary.json.
# Call this after every search.fit() to guarantee the file is written.
def _force_visualize(analysis, result, tag: str = ""):
    try:
        analysis.visualize(
            paths=result.paths,
            instance=result.max_log_likelihood_instance,
            during_analysis=False,
        )
    except Exception as e:
        print(f"[MOD??]   warning: post-fit visualize {tag} failed: {e}",
              flush=True)


# ---------------------------------------------------------------------------
#  Dataset loading (mirror of the pattern in every module notebook)
# ---------------------------------------------------------------------------
def load_dataset(dataset_root: Path, name: str, mask_radius: float = 3.0,
                 pixel_scales: float = 0.1):
    """Load a masked imaging dataset from a workspace dataset directory.

    Args:
        dataset_root: Parent directory containing the named subdir (e.g.
            autolens_workspace_original/dataset/imaging/).
        name: Dataset subdir name (e.g. 'simple__no_lens_light').
        mask_radius: Circular mask radius in arcsec. Should be larger than
            the expected Einstein radius by ~1.5–2× so arcs aren't clipped.
        pixel_scales: Pixel scale in arcsec (0.1 for the tutorial datasets;
            HST ACS = 0.05, JWST NIRCam short = 0.031).
    """
    import autolens as al
    path = dataset_root / name
    dataset = al.Imaging.from_fits(
        data_path=path / "data.fits",
        psf_path=path / "psf.fits",
        noise_map_path=path / "noise_map.fits",
        pixel_scales=pixel_scales,
    )
    mask = al.Mask2D.circular(
        shape_native=dataset.shape_native,
        pixel_scales=dataset.pixel_scales,
        radius=mask_radius,
    )
    return dataset.apply_mask(mask=mask)


# ---------------------------------------------------------------------------
#  Build the fit — replace the example with your model + search(es)
# ---------------------------------------------------------------------------
def build(dataset_root: Path, output_root: Path, dataset_name: str, n_live: int):
    """Run your fit pipeline here.

    The example below is a single SIE + shear + SersicCore source fit
    (same as Module 03). Replace it with whatever your notebook does —
    a two-search chain, a SLaM pipeline, a multi-component mass model, etc.

    For chained fits, pass `result_prev.model.galaxies.lens.mass` etc.
    into the next model to promote priors. For SLaM, import from slam_v2026
    (mirror fit_module04.py or fit_module09.py).
    """
    import autofit as af
    import autolens as al

    dataset = load_dataset(dataset_root, dataset_name)
    ncores = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))

    # ---- Model (example — replace with yours) --------------------------------
    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy, redshift=0.5,
                mass=al.mp.Isothermal,
                shear=al.mp.ExternalShear,
            ),
            source=af.Model(
                al.Galaxy, redshift=1.0,
                bulge=al.lp.SersicCore,
            ),
        )
    )
    # Example prior tightening:
    model.galaxies.lens.mass.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.1)
    model.galaxies.lens.mass.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.1)
    model.galaxies.lens.mass.einstein_radius = af.UniformPrior(
        lower_limit=0.5, upper_limit=3.0)

    # ---- Search --------------------------------------------------------------
    # path_prefix + name + unique_tag + model hash → the Nautilus output
    # directory. An existing checkpoint.hdf5 at that path auto-resumes.
    search = af.Nautilus(
        path_prefix=output_root / "module_template",
        name="my_fit",
        unique_tag=dataset_name,
        n_live=n_live,
        number_of_cores=ncores,
        iterations_per_update=2500,
    )

    analysis = al.AnalysisImaging(dataset=dataset, use_jax=False)

    print(f"[TMPL] Fitting {model.total_free_parameters}-param model on "
          f"{dataset_name} ({n_live} live points, {ncores} cores)", flush=True)
    t0 = time.time()
    result = search.fit(model=model, analysis=analysis)
    print(f"[TMPL] Done in {(time.time()-t0)/60:.1f} min; "
          f"logZ = {result.samples.log_evidence:.2f}", flush=True)

    # Always force fit.fits regeneration so the export step gets real
    # residual metrics even after a resumed search.
    _force_visualize(analysis, result, tag="my_fit")

    return result


# ---------------------------------------------------------------------------
#  CLI entry point
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", type=Path, required=True,
                   help="Path to Learning_to_Autolens (for slam_v2026 / "
                        "shared-helper imports). Absolute path on Cannon.")
    p.add_argument("--dataset-root", type=Path, required=True,
                   help="Parent of the dataset subdir — typically "
                        "autolens_workspace_{original,latest}/dataset/imaging")
    p.add_argument("--output-root", type=Path, default=Path("./output").resolve(),
                   help="Where Nautilus will write its output tree. On Cannon, "
                        "set via submit_cannon.slurm's OUTPUT_ROOT env var.")
    p.add_argument("--dataset-name", default="simple__no_lens_light",
                   help="Subdir under --dataset-root (e.g. 'simple', "
                        "'simple__no_lens_light', or your AGEL target's name)")
    p.add_argument("--n-live", type=int, default=100,
                   help="Nautilus live points. 100 = minimum for ~14-dim "
                        "models; bump to 150–200 for SLaM final stages or "
                        "when an earlier run crashed with LinAlgError.")
    args = p.parse_args()

    sys.path.insert(0, str(args.repo_root.resolve()))
    args.output_root.mkdir(parents=True, exist_ok=True)

    import autolens as al
    print(f"PyAutoLens: {al.__version__}", flush=True)
    print(f"SLURM_JOB_ID:  {os.environ.get('SLURM_JOB_ID', '(none)')}", flush=True)
    print(f"CPUS_PER_TASK: {os.environ.get('SLURM_CPUS_PER_TASK', '(none)')}",
          flush=True)
    print(f"Dataset root:  {args.dataset_root}", flush=True)
    print(f"Output root:   {args.output_root}", flush=True)

    t0 = time.time()
    build(args.dataset_root, args.output_root, args.dataset_name, args.n_live)
    print(f"\nTotal wall time: {(time.time()-t0)/3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
