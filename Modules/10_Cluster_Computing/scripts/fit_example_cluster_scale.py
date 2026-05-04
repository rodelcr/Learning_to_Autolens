"""
fit_example_cluster_scale.py — Cannon driver for the synthetic cluster mock.

Direct fit on the Examples/cluster_scale/ system: BCG + 10 cluster
member galaxies (FJ-scaled mass) + 2 sources at different redshifts.
~19 free parameters using the scaling-relation API (10 members share
a single theta_E_star amplitude).

Usage:
    sbatch --time=24:00:00 --export=ALL,EXAMPLE=cluster_scale \\
        Modules/10_Cluster_Computing/scripts/submit_cannon.slurm

Reference:
    Examples/cluster_scale/01_cluster_scale_fit.ipynb has the same model
    construction, plus the geometry exposition.
    autolens_workspace_latest/scripts/imaging/features/scaling_relation/modeling.py
    is the canonical scaling-relation API source.
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
        print(f"[CLUSTER]   warning: post-fit visualize {tag} failed: {e}",
              flush=True)


def load_dataset(dataset_root: Path, mask_radius: float = 4.5):
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


def build_direct_fit(dataset, output_root: Path, truth: dict,
                     n_live: int = 200, n_batch: int = 50):
    """Direct fit: BCG (Sersic + Iso) + 10 FJ-scaled members + 2 sources.

    Mirrors Examples/cluster_scale/01_cluster_scale_fit.ipynb cell
    structure.
    """
    import autofit as af
    import autolens as al
    import numpy as np

    z_l  = truth["redshifts"]["lens"]
    z_s1 = truth["redshifts"]["source_1"]
    z_s2 = truth["redshifts"]["source_2"]

    # ---- BCG: Sersic light + Isothermal mass, centre fixed at origin ----
    bcg_bulge = af.Model(al.lp.SersicSph)
    bcg_bulge.centre = (0.0, 0.0)
    bcg_bulge.intensity        = af.LogUniformPrior(lower_limit=1e-2, upper_limit=10.0)
    bcg_bulge.effective_radius = af.UniformPrior(lower_limit=0.5, upper_limit=4.0)
    bcg_bulge.sersic_index     = af.UniformPrior(lower_limit=2.0, upper_limit=6.0)

    bcg_mass = af.Model(al.mp.IsothermalSph)
    bcg_mass.centre = (0.0, 0.0)
    bcg_mass.einstein_radius = af.UniformPrior(lower_limit=2.0, upper_limit=6.0)

    bcg = af.Model(al.Galaxy, redshift=z_l, bulge=bcg_bulge, mass=bcg_mass)

    # ---- FJ scaling: 1 shared theta_E_star, 10 derived satellite masses ----
    theta_E_star = af.UniformPrior(lower_limit=0.1, upper_limit=2.0)
    L_star = 1.0
    members_dict = {}
    for i, m in enumerate(truth["members"]):
        cy, cx = m["centre"]
        L_i = m["luminosity"]
        member_mass = af.Model(al.mp.IsothermalSph)
        member_mass.centre = (cy, cx)
        member_mass.einstein_radius = theta_E_star * float(np.sqrt(L_i / L_star))
        members_dict[f"member_{i+1}"] = af.Model(
            al.Galaxy, redshift=z_l, mass=member_mass)

    # ---- Two sources at different z ----
    def _src_model(z):
        s = af.Model(al.lp.SersicCore)
        s.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.5)
        s.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.5)
        s.intensity        = af.LogUniformPrior(lower_limit=1e-2, upper_limit=10.0)
        s.effective_radius = af.UniformPrior(lower_limit=0.05, upper_limit=0.3)
        s.sersic_index     = af.UniformPrior(lower_limit=0.5, upper_limit=3.0)
        s.ell_comps.ell_comps_0 = af.GaussianPrior(mean=0.0, sigma=0.3)
        s.ell_comps.ell_comps_1 = af.GaussianPrior(mean=0.0, sigma=0.3)
        return af.Model(al.Galaxy, redshift=z, bulge=s)

    source_1 = _src_model(z_s1)
    source_2 = _src_model(z_s2)

    model = af.Collection(
        galaxies=af.Collection(bcg=bcg, source_1=source_1, source_2=source_2),
        extra_galaxies=af.Collection(**members_dict),
    )
    print(f"[CLUSTER/direct] {model.total_free_parameters} free params "
          f"(BCG=4, shared FJ θ_E_star=1, source_1=7, source_2=7)", flush=True)

    analysis = al.AnalysisImaging(dataset=dataset, use_jax=False)
    search = af.Nautilus(
        path_prefix=output_root / "cluster_scale",
        name="direct_fit",
        unique_tag="mock_1",
        n_live=n_live, n_batch=n_batch, iterations_per_update=15000,
        number_of_cores=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
    )
    print("[CLUSTER/direct] Nautilus starting...", flush=True)
    t0 = time.time()
    result = search.fit(model=model, analysis=analysis)
    print(f"[CLUSTER/direct] done in {(time.time()-t0)/60:.1f} min, "
          f"log_Z={result.samples.log_evidence:.2f}", flush=True)
    _force_visualize(analysis, result, tag="direct")
    print(result.info, flush=True)
    return result


def build_truth_anchored(dataset, output_root: Path, truth: dict,
                         n_live: int = 200, n_batch: int = 50):
    """Truth-anchored variant: every BCG/satellite/source param tightly
    constrained on its truth value.

    Per the 2026-04-29 protocol established for compound and group_scale:
    when freely-fit fails (cluster `direct_fit` job 9675524 hit χ²/N=22.6,
    theta_E_star railed to 0.10, BCG mass ballooned to 5.95"), build a
    tight-prior variant to test whether the model space CAN reach the
    truth basin. If yes, the freely-fit failure is search-space exploration
    (resolve via prior anchoring or staged chain). If no, the model space
    is misspecified.

    For cluster_scale specifically, the wide-prior failure was the BCG
    vs theta_E_star degeneracy collapse. Truth-anchoring fixes the BCG
    mass at truth±0.05 and theta_E_star at truth±0.05 — the FJ scaling
    becomes a single free knob (the amplitude) but constrained to its
    truth, breaking the degeneracy.
    """
    import autofit as af
    import autolens as al
    import numpy as np

    z_l  = truth["redshifts"]["lens"]
    z_s1 = truth["redshifts"]["source_1"]
    z_s2 = truth["redshifts"]["source_2"]
    bcg_truth = truth["bcg"]
    fj_truth = truth["fj_relation"]

    # ---- BCG: tight Gaussians on truth ----
    bcg_bulge = af.Model(al.lp.SersicSph)
    bcg_bulge.centre = (0.0, 0.0)
    bcg_bulge.intensity        = af.LogUniformPrior(lower_limit=1e-2, upper_limit=10.0)
    bcg_bulge.effective_radius = af.GaussianPrior(
        mean=bcg_truth["effective_radius"], sigma=0.1)
    bcg_bulge.sersic_index     = af.GaussianPrior(
        mean=bcg_truth["sersic_index"], sigma=0.3)

    bcg_mass = af.Model(al.mp.IsothermalSph)
    bcg_mass.centre = (0.0, 0.0)
    bcg_mass.einstein_radius = af.GaussianPrior(
        mean=bcg_truth["einstein_radius"], sigma=0.05)

    bcg = af.Model(al.Galaxy, redshift=z_l, bulge=bcg_bulge, mass=bcg_mass)

    # ---- FJ scaling: theta_E_star tight Gaussian on truth ----
    # Each member contributes BOTH a SersicSph LIGHT profile (truth has
    # one per satellite) AND an IsothermalSph mass (FJ-scaled).
    # 2026-05-04: omitting the member light was the cluster_truth_v2 bug
    # — even at literal truth values the residuals were 30σ because the
    # 10 satellite light profiles were unmodeled. With member light
    # restored, chi^2 at truth drops to ~1.0.
    theta_E_star = af.GaussianPrior(
        mean=fj_truth["theta_E_star"], sigma=0.05)
    L_star = fj_truth["luminosity_star"]
    members_dict = {}
    for i, m in enumerate(truth["members"]):
        cy, cx = m["centre"]
        L_i = m["luminosity"]

        # Member light: SersicSph with intensity, R_eff, n tightly
        # anchored on truth (centre fixed at photometric).
        member_bulge = af.Model(al.lp.SersicSph)
        member_bulge.centre = (cy, cx)
        member_bulge.intensity        = af.GaussianPrior(
            mean=m["intensity"], sigma=max(0.02, 0.1 * m["intensity"]))
        member_bulge.effective_radius = af.GaussianPrior(
            mean=m["effective_radius"], sigma=0.05)
        member_bulge.sersic_index     = af.GaussianPrior(
            mean=m["sersic_index"], sigma=0.3)

        # Member mass: FJ-scaled from theta_E_star (single shared param).
        member_mass = af.Model(al.mp.IsothermalSph)
        member_mass.centre = (cy, cx)
        member_mass.einstein_radius = theta_E_star * float(np.sqrt(L_i / L_star))

        members_dict[f"member_{i+1}"] = af.Model(
            al.Galaxy, redshift=z_l, bulge=member_bulge, mass=member_mass)

    # ---- Sources: tight Gaussians on truth ----
    def _src_truth(z, t):
        s = af.Model(al.lp.SersicCore)
        s.centre.centre_0 = af.GaussianPrior(mean=t["centre"][0], sigma=0.05)
        s.centre.centre_1 = af.GaussianPrior(mean=t["centre"][1], sigma=0.05)
        s.intensity        = af.LogUniformPrior(lower_limit=1e-2, upper_limit=20.0)
        s.effective_radius = af.GaussianPrior(
            mean=t["effective_radius"], sigma=0.02)
        s.sersic_index     = af.GaussianPrior(
            mean=t["sersic_index"], sigma=0.3)
        # Convert truth (axis_ratio, angle_deg) -> ell_comps and seed
        # a Gaussian prior centred on the truth ellipticity.
        # CRITICAL: truth sources have axis_ratio=0.7/0.85 (non-circular);
        # a Gaussian(0, 0.1) prior would prevent the chain from fitting
        # the ellipticity, producing log_Z ~ -268k (cluster_truth job
        # 9727867 TIMEOUT pathology).
        truth_ell = al.convert.ell_comps_from(
            axis_ratio=t["axis_ratio"], angle=t["angle_deg"])
        s.ell_comps.ell_comps_0 = af.GaussianPrior(mean=truth_ell[0], sigma=0.05)
        s.ell_comps.ell_comps_1 = af.GaussianPrior(mean=truth_ell[1], sigma=0.05)
        return af.Model(al.Galaxy, redshift=z, bulge=s)

    source_1 = _src_truth(z_s1, truth["source_1"])
    source_2 = _src_truth(z_s2, truth["source_2"])

    model = af.Collection(
        galaxies=af.Collection(bcg=bcg, source_1=source_1, source_2=source_2),
        extra_galaxies=af.Collection(**members_dict),
    )
    print(f"[CLUSTER/truth_anchored] {model.total_free_parameters} free params",
          flush=True)

    analysis = al.AnalysisImaging(dataset=dataset, use_jax=False)
    search = af.Nautilus(
        path_prefix=output_root / "cluster_scale",
        name="truth_anchored",
        unique_tag="mock_1",
        n_live=n_live, n_batch=n_batch, iterations_per_update=15000,
        number_of_cores=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
    )
    print("[CLUSTER/truth_anchored] Nautilus starting...", flush=True)
    t0 = time.time()
    result = search.fit(model=model, analysis=analysis)
    print(f"[CLUSTER/truth_anchored] done in {(time.time()-t0)/60:.1f} min, "
          f"log_Z={result.samples.log_evidence:.2f}", flush=True)
    _force_visualize(analysis, result, tag="truth_anchored")
    print(result.info, flush=True)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--part",         choices=("direct", "truth_anchored"),
                   default="direct")
    p.add_argument("--repo-root",    type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--output-root",  type=Path, required=True)
    p.add_argument("--n-live",       type=int, default=200)
    args = p.parse_args()

    sys.path.insert(0, str(args.repo_root))
    truth = json.loads((args.dataset_root / "mock_truth.json").read_text())
    dataset = load_dataset(args.dataset_root, mask_radius=4.5)
    print(f"[CLUSTER] dataset pixels_in_mask={dataset.mask.pixels_in_mask}",
          flush=True)

    if args.part == "direct":
        build_direct_fit(dataset, args.output_root, truth, n_live=args.n_live)
    elif args.part == "truth_anchored":
        build_truth_anchored(dataset, args.output_root, truth, n_live=args.n_live)


if __name__ == "__main__":
    main()
