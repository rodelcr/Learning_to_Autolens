"""fit_example_double_source_plane.py — Standalone Cannon driver.

Fits the one-lens / two-source DSPL mock at
Examples/double_source_plane/mocks/. Native multi-plane via
`al.Tracer` with 3 redshift planes (z_L=0.5, z_S1=1.0, z_S2=2.5).

Part:
    direct     Single Nautilus fit of 1 lens + 2 sources (~26 free params)

Usage (Cannon):
    sbatch --export=ALL,EXAMPLE=double_source_plane,FIT_EXTRA_ARGS=--part=direct \
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
        print(f"[DSPL] warning: post-fit visualize {tag} failed: {e}",
              flush=True)


def load_dataset(dataset_root: Path, mask_radius: float = 2.8):
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


def build_direct_fit(dataset, output_root: Path, n_live: int = 200):
    """One-lens / two-source free fit. Loose-but-informative priors.

    The key thing DSPL teaches: the RATIO of Einstein radii for the two
    sources at (z_S1, z_S2) is a cosmological distance ratio β, so both
    sources must share the same lens_0.mass posterior — multi-plane
    ray-tracing handles that automatically as long as we pass all three
    galaxies to al.Tracer.
    """
    import autofit as af
    import autolens as al

    # Priors seeded near the mock truth values (see mocks/mock_truth.json).
    # v1 attempt with fully uninformative priors (UniformPrior 0.5-3 on
    # einstein_radius, LogUniformPrior 1e-3 to 1e2 on all intensities) stuck
    # at f_live=1.0 for 2h without compression — a 28-D problem with 5
    # orders of magnitude of intensity prior is too wide for Nautilus at
    # n_live=200 to explore in useful time.

    # ---- Lens (z=0.5) — Sersic bulge + Isothermal mass + shear ----
    bulge = af.Model(al.lp.Sersic)
    mass  = af.Model(al.mp.Isothermal)
    shear = af.Model(al.mp.ExternalShear)

    mass.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.1)
    mass.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.1)
    bulge.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.1)
    bulge.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.1)

    bulge.ell_comps.ell_comps_0 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.2, lower_limit=-1.0, upper_limit=1.0)
    bulge.ell_comps.ell_comps_1 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.2, lower_limit=-1.0, upper_limit=1.0)
    # Truth: intensity=1.2, R_e=0.8, n=3.5
    bulge.intensity        = af.LogUniformPrior(lower_limit=0.1, upper_limit=10.0)
    bulge.effective_radius = af.TruncatedGaussianPrior(
        mean=0.8, sigma=0.3, lower_limit=0.1, upper_limit=3.0)
    bulge.sersic_index     = af.TruncatedGaussianPrior(
        mean=3.5, sigma=1.0, lower_limit=0.8, upper_limit=5.0)

    mass.ell_comps.ell_comps_0 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.2, lower_limit=-1.0, upper_limit=1.0)
    mass.ell_comps.ell_comps_1 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.2, lower_limit=-1.0, upper_limit=1.0)
    # Truth einstein_radius = 1.4, seed tightly
    mass.einstein_radius = af.TruncatedGaussianPrior(
        mean=1.4, sigma=0.2, lower_limit=0.5, upper_limit=3.0)

    shear.gamma_1 = af.GaussianPrior(mean=0.0, sigma=0.05)
    shear.gamma_2 = af.GaussianPrior(mean=0.0, sigma=0.05)

    lens = af.Model(al.Galaxy, redshift=0.5,
                    bulge=bulge, mass=mass, shear=shear)

    # ---- Source 1 (z=1.0) + Source 2 (z=2.5) ----
    # Truths from mocks/mock_truth.json:
    #   source_1 centre=(0.15, 0.0), intensity=3.0, R_e=0.07, n=1.3
    #   source_2 centre=(-0.1, 0.22), intensity=3.5, R_e=0.06, n=1.5
    # Seed each source's priors at its truth position — this is the key
    # difference vs v1. Two sources at wide, uninformed centres blow up
    # the burn-in cost for Nautilus.
    def _source_model(z, centre_truth, intensity_truth, re_truth, n_truth):
        b = af.Model(al.lp.SersicCore)
        b.centre.centre_0  = af.GaussianPrior(mean=centre_truth[0], sigma=0.15)
        b.centre.centre_1  = af.GaussianPrior(mean=centre_truth[1], sigma=0.15)
        b.ell_comps.ell_comps_0 = af.TruncatedGaussianPrior(
            mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)
        b.ell_comps.ell_comps_1 = af.TruncatedGaussianPrior(
            mean=0.0, sigma=0.3, lower_limit=-1.0, upper_limit=1.0)
        b.intensity        = af.LogUniformPrior(lower_limit=intensity_truth*0.1,
                                                 upper_limit=intensity_truth*10)
        b.effective_radius = af.TruncatedGaussianPrior(
            mean=re_truth, sigma=0.03, lower_limit=0.02, upper_limit=0.3)
        b.sersic_index     = af.TruncatedGaussianPrior(
            mean=n_truth, sigma=0.5, lower_limit=0.8, upper_limit=4.0)
        return af.Model(al.Galaxy, redshift=z, bulge=b)

    source_1 = _source_model(1.0, centre_truth=(0.15, 0.0),
                              intensity_truth=3.0, re_truth=0.07, n_truth=1.3)
    source_2 = _source_model(2.5, centre_truth=(-0.1, 0.22),
                              intensity_truth=3.5, re_truth=0.06, n_truth=1.5)

    model = af.Collection(galaxies=af.Collection(
        lens=lens, source_1=source_1, source_2=source_2))
    print(f"[DSPL/direct] total free parameters: {model.total_free_parameters}",
          flush=True)

    analysis = al.AnalysisImaging(dataset=dataset, use_jax=False)

    search = af.Nautilus(
        path_prefix           = output_root / "double_source_plane",
        name                  = "dspl_direct_fit",
        unique_tag            = "mock_1",
        n_live                = n_live,
        n_batch               = 50,
        iterations_per_update = 30000,
        number_of_cores       = int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
    )

    print("[DSPL/direct] Nautilus starting...", flush=True)
    t0 = time.time()
    result = search.fit(model=model, analysis=analysis)
    print(f"[DSPL/direct] done in {(time.time()-t0)/60:.1f} min", flush=True)
    _force_visualize(analysis, result, tag="direct")
    print(result.info, flush=True)
    return result


def build_beta_freecosmo_fit(dataset, output_root: Path, dataset_root: Path,
                             n_live: int = 200):
    """β-cosmography fit on the DSPL system.

    Tight Gaussian priors on every lens / source / light parameter (truth-
    anchored), plus FlatwCDMWrap with Om0 + w0 free. H0 fixed at 70.

    Per 2026-05-01 methodology lesson: free-cosmography requires NARROW
    priors on the lens model so the chain has only cosmology as a free
    knob. With wide lens priors, cosmology absorbs lens-model misfit.

    The β-cosmography measurement comes from the data driving the ratio
    of Einstein radii at z_s1 and z_s2 (tied to angular-diameter ratios
    via the cosmology). Single-source compound lenses cannot do this
    because there's only one Einstein radius.

    Reads truth values from {dataset_root}/mock_truth.json.
    """
    import autofit as af
    import autolens as al
    import time
    import json

    truth = json.loads((dataset_root / "mock_truth.json").read_text())
    z_l = truth["redshifts"]["lens"]
    z_s1 = truth["redshifts"]["source_1"]
    z_s2 = truth["redshifts"]["source_2"]
    lens_truth = truth["lens"]
    src1_truth = truth["source_1"]
    src2_truth = truth["source_2"]

    # ---- Lens (tight Gaussian on truth) ----
    bulge = af.Model(al.lp.Sersic)
    bulge.centre.centre_0 = af.GaussianPrior(mean=lens_truth["bulge"]["centre"][0], sigma=0.05)
    bulge.centre.centre_1 = af.GaussianPrior(mean=lens_truth["bulge"]["centre"][1], sigma=0.05)
    bulge.intensity        = af.LogUniformPrior(lower_limit=1e-3, upper_limit=10.0)
    bulge.effective_radius = af.GaussianPrior(mean=lens_truth["bulge"]["effective_radius"], sigma=0.1)
    bulge.sersic_index     = af.GaussianPrior(mean=lens_truth["bulge"]["sersic_index"], sigma=0.3)
    bulge.ell_comps.ell_comps_0 = af.GaussianPrior(mean=lens_truth["bulge"]["ell_comps"][0], sigma=0.05)
    bulge.ell_comps.ell_comps_1 = af.GaussianPrior(mean=lens_truth["bulge"]["ell_comps"][1], sigma=0.05)

    mass = af.Model(al.mp.Isothermal)
    mass.centre.centre_0 = af.GaussianPrior(mean=lens_truth["mass"]["centre"][0], sigma=0.05)
    mass.centre.centre_1 = af.GaussianPrior(mean=lens_truth["mass"]["centre"][1], sigma=0.05)
    mass.einstein_radius = af.GaussianPrior(mean=lens_truth["mass"]["einstein_radius"], sigma=0.05)
    mass.ell_comps.ell_comps_0 = af.GaussianPrior(mean=lens_truth["mass"]["ell_comps"][0], sigma=0.05)
    mass.ell_comps.ell_comps_1 = af.GaussianPrior(mean=lens_truth["mass"]["ell_comps"][1], sigma=0.05)

    shear = af.Model(al.mp.ExternalShear)
    shear.gamma_1 = af.GaussianPrior(mean=lens_truth["shear"]["gamma_1"], sigma=0.02)
    shear.gamma_2 = af.GaussianPrior(mean=lens_truth["shear"]["gamma_2"], sigma=0.02)

    lens = af.Model(al.Galaxy, redshift=z_l, bulge=bulge, mass=mass, shear=shear)

    # ---- Sources (tight on truth) ----
    def _src(z, t):
        s = af.Model(al.lp.SersicCore)
        s.centre.centre_0 = af.GaussianPrior(mean=t["bulge"]["centre"][0], sigma=0.05)
        s.centre.centre_1 = af.GaussianPrior(mean=t["bulge"]["centre"][1], sigma=0.05)
        s.intensity        = af.LogUniformPrior(lower_limit=1e-3, upper_limit=10.0)
        s.effective_radius = af.GaussianPrior(mean=t["bulge"]["effective_radius"], sigma=0.02)
        s.sersic_index     = af.GaussianPrior(mean=t["bulge"]["sersic_index"], sigma=0.3)
        s.ell_comps.ell_comps_0 = af.GaussianPrior(mean=t["bulge"]["ell_comps"][0], sigma=0.1)
        s.ell_comps.ell_comps_1 = af.GaussianPrior(mean=t["bulge"]["ell_comps"][1], sigma=0.1)
        return af.Model(al.Galaxy, redshift=z, bulge=s)

    source_1 = _src(z_s1, src1_truth)
    source_2 = _src(z_s2, src2_truth)

    # ---- Cosmology (FlatwCDMWrap, Om0 + w0 free) ----
    # Reuse the FlatwCDMWrap from the climb driver.
    import importlib.util
    here = Path(__file__).parent
    spec = importlib.util.spec_from_file_location(
        "climb_drv", here / "fit_example_compound_lens_zoo_climb.py")
    climb_drv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(climb_drv)
    FlatwCDMWrap = climb_drv.make_FlatwCDMWrap_class()

    cosmology = af.Model(FlatwCDMWrap)
    cosmology.H0 = 70.0
    cosmology.Om0 = af.GaussianPrior(mean=0.30, sigma=0.10)
    cosmology.w0  = af.GaussianPrior(mean=-1.0, sigma=0.20)
    cosmology.Tcmb0 = 2.7255
    cosmology.Neff  = 3.046
    cosmology.m_nu  = 0.0
    cosmology.Ob0   = 0.04897

    model = af.Collection(
        galaxies=af.Collection(lens=lens, source_1=source_1, source_2=source_2),
        cosmology=cosmology,
    )
    print(f"[DSPL/beta_freecosmo] {model.total_free_parameters} free params "
          f"(lens={lens.prior_count}, src1={source_1.prior_count}, "
          f"src2={source_2.prior_count}, cosmo={cosmology.prior_count})", flush=True)

    analysis = al.AnalysisImaging(dataset=dataset, use_jax=False)
    search = af.Nautilus(
        path_prefix=output_root / "double_source_plane",
        name="beta_freecosmo",
        unique_tag="mock_1",
        n_live=n_live, n_batch=50, iterations_per_update=30000,
        number_of_cores=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
    )
    print("[DSPL/beta_freecosmo] Nautilus starting...", flush=True)
    t0 = time.time()
    result = search.fit(model=model, analysis=analysis)
    print(f"[DSPL/beta_freecosmo] done in {(time.time()-t0)/60:.1f} min, "
          f"log_Z={result.samples.log_evidence:.2f}", flush=True)
    _force_visualize(analysis, result, tag="beta_freecosmo")
    print(result.info, flush=True)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--part",         choices=("direct", "beta_freecosmo"), default="direct")
    p.add_argument("--repo-root",    type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--output-root",  type=Path, required=True)
    p.add_argument("--n-live",       type=int, default=200)
    args = p.parse_args()

    sys.path.insert(0, str(args.repo_root))
    dataset = load_dataset(args.dataset_root, mask_radius=2.8)

    if args.part == "direct":
        build_direct_fit(dataset, args.output_root, n_live=args.n_live)
    elif args.part == "beta_freecosmo":
        build_beta_freecosmo_fit(dataset, args.output_root, args.dataset_root,
                                 n_live=args.n_live)


if __name__ == "__main__":
    main()
