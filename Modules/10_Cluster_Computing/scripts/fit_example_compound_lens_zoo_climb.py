"""
fit_example_compound_lens_zoo_climb.py — Cannon driver for the compound-lens
ladder beyond R2 (the canonical zoo fit). Implements R3 (multi-plane), R2_2src
(single-plane + 2-source), and R5 (multi-plane + 2-source) per the ladder
analysis in Examples/compound_lens_zoo/02_compound_lens_ladder.ipynb.

Pedagogical anchor:
    R2  (canonical zoo)              — fit_example_compound_lens_zoo.py
    R3  (multi-plane)                — THIS SCRIPT, --rung R3
    R2_2src (1-plane + 2src)         — THIS SCRIPT, --rung R2_2src
    R5  (multi-plane + 2src)         — THIS SCRIPT, --rung R5
    R5_truth (truth-anchored R5)     — THIS SCRIPT, --rung R5_truth
    R5_staged (R2_2src -> R5 chain)  — THIS SCRIPT, --rung R5_staged

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

R5_truth model:
    Same architecture as R5 but with ALL priors set as tight Gaussians
    centred on the lenstronomy truth values (mass, light, sources). Tests
    whether PyAutoLens's R5 model space *can* represent the truth lens
    system when the chain is constrained near it. If R5_truth converges to
    good chi^2 with clean residuals, the freely-fit R5 failures (Pattern E
    on mock_3, Pattern A on mock_4) are local-optimum issues, NOT
    fundamental model-space limitations. Sigma chosen with ~3 sigma room
    so the chain can absorb autolens-vs-lenstronomy convention drift on
    ell_comps / centre / shear axes (project_fit_failure_patterns Pattern B).

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
# R5_truth: same architecture as R5, all priors TIGHTLY anchored on truth
# =============================================================================
def _truth_sources(truths: dict) -> list:
    """Return up to two source kwargs from the lenstronomy truths."""
    src_kwargs_list = truths.get("kwargs_source", [])
    out = []
    for kw in src_kwargs_list[:2]:
        out.append(kw)
    while len(out) < 2:
        out.append({})
    return out


def _truth_lens_light(truths: dict) -> dict:
    """Return the primary (Sersic) lens light kwargs from lenstronomy truths."""
    ll_list = truths.get("kwargs_lens_light", [])
    return ll_list[0] if ll_list else {}


def _truth_primary(truths: dict) -> dict:
    """First EPL = primary deflector."""
    epls = [(i, kw) for i, (m, kw)
            in enumerate(zip(truths["lens_model_list"], truths["kwargs_lens"]))
            if m == "EPL"]
    if not epls:
        raise ValueError("No primary EPL in truths")
    return epls[0][1]


def _truth_shear(truths: dict) -> tuple[float, float]:
    """Return (gamma_1, gamma_2) Cartesian from lenstronomy SHEAR_GAMMA_PSI.

    lenstronomy convention: gamma_1 = gamma_ext * cos(2 psi),
                            gamma_2 = gamma_ext * sin(2 psi).
    Returned in the same convention; the truth-anchored R5 uses TGaussian
    priors around these values with sigma large enough to absorb any
    autolens 2026.4 sign convention drift (see project_fit_failure_patterns
    Pattern B).
    """
    import math
    for m, kw in zip(truths["lens_model_list"], truths["kwargs_lens"]):
        if m in ("SHEAR_GAMMA_PSI", "SHEAR"):
            if m == "SHEAR_GAMMA_PSI":
                gx = kw.get("gamma_ext", 0.0)
                psi = kw.get("psi_ext", 0.0)
                return (gx * math.cos(2 * psi), gx * math.sin(2 * psi))
            else:
                return (kw.get("gamma1", 0.0), kw.get("gamma2", 0.0))
    return (0.0, 0.0)


def build_R5_truth_model(truths: dict):
    """R5_truth: same architecture as R5 with tight truth-anchored priors.

    Validation question: can PyAutoLens's R5 model space *represent* the
    truth lens system? If yes (good chi^2, clean residuals when chain is
    constrained near truth), then the freely-fit R5 failures (Pattern E
    on mock_3, Pattern A on mock_4) are local-optimum issues, not
    fundamental model-space limitations.

    All priors are TruncatedGaussian(mean=truth, sigma=tight) with the
    sigma chosen to (a) anchor the chain near truth and (b) leave enough
    room (~3 sigma) for autolens-vs-lenstronomy sign / convention drift
    on ell_comps / centre / shear axes.
    """
    import autofit as af
    import autolens as al

    z_l1 = truths["redshifts"]["lens_primary"]
    z_l2 = truths["redshifts"]["lens_secondary"]
    z_s  = truths["redshifts"]["source"]

    primary = _truth_primary(truths)
    secondary = _find_secondary_truth(truths)
    lens_light = _truth_lens_light(truths)
    sources = _truth_sources(truths)
    shear_g1, shear_g2 = _truth_shear(truths)

    # ---- Primary lens light (Sersic) — tight on truth ---------------------
    bulge = af.Model(al.lp.Sersic)
    bulge.centre.centre_0 = af.GaussianPrior(
        mean=lens_light.get("center_x", 0.0), sigma=0.05)
    bulge.centre.centre_1 = af.GaussianPrior(
        mean=lens_light.get("center_y", 0.0), sigma=0.05)
    bulge.intensity        = af.LogUniformPrior(lower_limit=1e-6, upper_limit=1e6)
    bulge.effective_radius = af.GaussianPrior(
        mean=lens_light.get("R_sersic", 1.0), sigma=0.3)
    bulge.sersic_index     = af.GaussianPrior(
        mean=lens_light.get("n_sersic", 4.0), sigma=0.5)
    bulge.ell_comps.ell_comps_0 = af.TruncatedGaussianPrior(
        mean=lens_light.get("e1", 0.0), sigma=0.1,
        lower_limit=-1.0, upper_limit=1.0)
    bulge.ell_comps.ell_comps_1 = af.TruncatedGaussianPrior(
        mean=lens_light.get("e2", 0.0), sigma=0.1,
        lower_limit=-1.0, upper_limit=1.0)

    # ---- Primary lens mass (PowerLaw) — tight on truth -------------------
    mass = af.Model(al.mp.PowerLaw)
    mass.centre.centre_0 = af.GaussianPrior(
        mean=primary.get("center_x", 0.0), sigma=0.05)
    mass.centre.centre_1 = af.GaussianPrior(
        mean=primary.get("center_y", 0.0), sigma=0.05)
    mass.einstein_radius = af.GaussianPrior(
        mean=primary.get("theta_E", 1.0), sigma=0.1)
    mass.slope           = af.TruncatedGaussianPrior(
        mean=primary.get("gamma", 2.0), sigma=0.1,
        lower_limit=1.5, upper_limit=2.7)
    mass.ell_comps.ell_comps_0 = af.TruncatedGaussianPrior(
        mean=primary.get("e1", 0.0), sigma=0.1,
        lower_limit=-1.0, upper_limit=1.0)
    mass.ell_comps.ell_comps_1 = af.TruncatedGaussianPrior(
        mean=primary.get("e2", 0.0), sigma=0.1,
        lower_limit=-1.0, upper_limit=1.0)

    # ---- ExternalShear — tight on Cartesian truth ------------------------
    shear = af.Model(al.mp.ExternalShear)
    shear.gamma_1 = af.GaussianPrior(mean=shear_g1, sigma=0.05)
    shear.gamma_2 = af.GaussianPrior(mean=shear_g2, sigma=0.05)

    lens_1 = af.Model(al.Galaxy, redshift=z_l1, bulge=bulge, mass=mass, shear=shear)

    # ---- Secondary deflector (PowerLaw, slope free) — tight on truth -----
    # Use PowerLaw (not Isothermal) so the slope can match mock_4's truth=2.1.
    mass_2 = af.Model(al.mp.PowerLaw)
    mass_2.centre.centre_0 = af.GaussianPrior(
        mean=secondary.get("center_x", 0.0), sigma=0.05)
    mass_2.centre.centre_1 = af.GaussianPrior(
        mean=secondary.get("center_y", 0.0), sigma=0.05)
    mass_2.einstein_radius = af.GaussianPrior(
        mean=secondary.get("theta_E", 0.1), sigma=0.05)
    mass_2.slope           = af.TruncatedGaussianPrior(
        mean=secondary.get("gamma", 2.0), sigma=0.1,
        lower_limit=1.5, upper_limit=2.7)
    mass_2.ell_comps.ell_comps_0 = af.TruncatedGaussianPrior(
        mean=secondary.get("e1", 0.0), sigma=0.1,
        lower_limit=-1.0, upper_limit=1.0)
    mass_2.ell_comps.ell_comps_1 = af.TruncatedGaussianPrior(
        mean=secondary.get("e2", 0.0), sigma=0.1,
        lower_limit=-1.0, upper_limit=1.0)
    lens_2 = af.Model(al.Galaxy, redshift=z_l2, mass=mass_2)

    # ---- Two source components — tight on truth --------------------------
    def _src_truth(seed: dict):
        s = af.Model(al.lp.SersicCore)
        s.centre.centre_0 = af.GaussianPrior(
            mean=seed.get("center_x", 0.0), sigma=0.05)
        s.centre.centre_1 = af.GaussianPrior(
            mean=seed.get("center_y", 0.0), sigma=0.05)
        # Source intensity normalization differs (lenstronomy amp vs autolens
        # intensity), so leave wide LogUniform.
        s.intensity        = af.LogUniformPrior(lower_limit=1e-3, upper_limit=1e3)
        s.effective_radius = af.GaussianPrior(
            mean=seed.get("R_sersic", 0.2), sigma=0.05)
        s.sersic_index     = af.GaussianPrior(
            mean=seed.get("n_sersic", 1.0), sigma=0.3)
        s.ell_comps.ell_comps_0 = af.TruncatedGaussianPrior(
            mean=seed.get("e1", 0.0), sigma=0.1,
            lower_limit=-1.0, upper_limit=1.0)
        s.ell_comps.ell_comps_1 = af.TruncatedGaussianPrior(
            mean=seed.get("e2", 0.0), sigma=0.1,
            lower_limit=-1.0, upper_limit=1.0)
        return s

    src_a = _src_truth(sources[0])
    src_b = _src_truth(sources[1])
    source = af.Model(al.Galaxy, redshift=z_s, bulge=src_a, disk=src_b)

    return af.Collection(galaxies=af.Collection(
        lens_1=lens_1, lens_2=lens_2, source=source))


# =============================================================================
# R5_staged: 2-stage chain (R2_2src -> R5 with prior passing)
# =============================================================================
def build_R5_staged_chain(dataset, output_root: Path, mock_index: int,
                          truths: dict, n_live: int = 250):
    """Two-stage SLaM-style chain on the same dataset:

    Stage 1: R2_2src (single-plane PowerLaw + shear + 2 SersicCore sources)
             — gets a strong posterior on primary lens + sources before
             multi-plane geometry is introduced.

    Stage 2: R5 (multi-plane + 2 sources). Primary lens, shear, lens light,
             and BOTH source components are passed from Stage 1 as priors
             centred on the posterior. Secondary lens (z_l2) parameters are
             the only ones with the original wide priors.

    Tests whether walking the chain through R2_2src first (which we know
    converges cleanly for mock_6) lands the secondary in the truth basin
    rather than the Pattern E (mock_3) / Pattern A (mock_4) freely-fit
    optima.
    """
    import autofit as af
    import autolens as al

    print(f"\n[STAGED/mock_{mock_index}] starting 2-stage chain", flush=True)

    analysis = al.AnalysisImaging(dataset=dataset, use_jax=False)

    # ---- Stage 1: R2_2src ----
    print(f"[STAGED/mock_{mock_index}] Stage 1: R2_2src", flush=True)
    s1_model = build_R2_2src_model(truths)
    s1_search = af.Nautilus(
        path_prefix=output_root,
        name=f"mock_{mock_index}_R5_staged_stage1_R2_2src",
        unique_tag=f"mock_{mock_index}_R5_staged",
        n_live=n_live, n_batch=50, iterations_per_update=15000,
        number_of_cores=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
    )
    t0 = time.time()
    s1_result = s1_search.fit(model=s1_model, analysis=analysis)
    print(f"[STAGED/mock_{mock_index}] Stage 1 done in "
          f"{(time.time()-t0)/60:.1f} min, log_Z={s1_result.samples.log_evidence:.2f}",
          flush=True)
    _force_visualize(analysis, s1_result, tag=f"mock_{mock_index}_staged_stage1")

    # ---- Stage 2: R5 with priors from Stage 1 ----
    print(f"[STAGED/mock_{mock_index}] Stage 2: R5 with prior passing", flush=True)
    s2_model = build_R5_model(truths)

    # Pass primary lens posterior (mass + shear + bulge) — R2_2src uses key
    # `lens`, R5 uses key `lens_1`. Both have PowerLaw mass + Sersic light
    # + ExternalShear, so the per-component priors are directly portable.
    s2_model.galaxies.lens_1.mass  = s1_result.model.galaxies.lens.mass
    s2_model.galaxies.lens_1.shear = s1_result.model.galaxies.lens.shear
    s2_model.galaxies.lens_1.bulge = s1_result.model.galaxies.lens.bulge
    # Both source components
    s2_model.galaxies.source.bulge = s1_result.model.galaxies.source.bulge
    s2_model.galaxies.source.disk  = s1_result.model.galaxies.source.disk
    # lens_2 keeps its original wide priors (no Stage 1 counterpart)

    s2_search = af.Nautilus(
        path_prefix=output_root,
        name=f"mock_{mock_index}_R5_staged_stage2_R5",
        unique_tag=f"mock_{mock_index}_R5_staged",
        n_live=n_live, n_batch=50, iterations_per_update=15000,
        number_of_cores=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
    )
    t0 = time.time()
    s2_result = s2_search.fit(model=s2_model, analysis=analysis)
    print(f"[STAGED/mock_{mock_index}] Stage 2 done in "
          f"{(time.time()-t0)/60:.1f} min, log_Z={s2_result.samples.log_evidence:.2f}",
          flush=True)
    _force_visualize(analysis, s2_result, tag=f"mock_{mock_index}_staged_stage2")
    print(s2_result.info, flush=True)
    return s2_result


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
    elif rung == "R5_truth":
        model = build_R5_truth_model(truths)
        unique_tag = f"mock_{mock_index}_R5_truth_anchored"
    elif rung == "R5_staged":
        # Special case — runs two Nautilus searches with prior passing,
        # bypassing the single-search build_fit return.
        return build_R5_staged_chain(dataset, output_root, mock_index, truths,
                                     n_live=n_live)
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
    p.add_argument("--rung",
                   choices=["R3", "R2_2src", "R5", "R5_truth", "R5_staged"],
                   required=True)
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
    #   R3        -> mocks 3, 4 (multi-plane signal mocks)
    #   R2_2src   -> mock 6     (single-deflector + 2-source mock)
    #   R5        -> mocks 3, 4 (multi-plane + 2-source — post-climb diagnosis)
    #   R5_truth  -> mocks 3, 4 (truth-anchored validation of R5 model space)
    #   R5_staged -> mocks 3, 4 (2-stage chain: R2_2src -> R5 with prior pass)
    if args.mock == "all":
        if args.rung in ("R3", "R5", "R5_truth", "R5_staged"):
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
