"""generate_mock.py — Einstein-spiral mock with embedded SMBH.

The methodology-bridge B example for v0.96. Synthetic Einstein-spiral
geometry mirroring the AGEL spiral-lens program (Shajib et al., Ferrami et
al. DESJ0206): source near the radial+tangential caustic intersection,
producing both a tangential arc AND a radial-arc feature near the centre.
Lens mass: PowerLaw + ExternalShear + central al.mp.PointMass.

Truth values chosen so the M_BH posterior recovery is the headline result:
  - lens: PowerLaw at z_l=0.7 with theta_E=1.0", slope=1.95 (sub-isothermal,
          matches AGEL spiral findings), ell_comps=(0.1, 0.05)
  - shear: ExternalShear with gamma_ext ~ 0.03
  - SMBH: PointMass with theta_E_BH = 0.08" (an unrealistically-large
    pedagogical BH so the imaging-only constraint is non-degenerate.
    Real AGEL UMBHs at z=0.7 have theta_E_BH ~ 0.02"; in the v0.97+
    real-target follow-up the BH is closer to the noise floor and
    kinematics is required to detect it. The Module 15 §4 narrative
    addresses this scale issue.)
  - lens light: Sersic n=4, R_e=0.8" (de Vaucouleurs-ish elliptical)
  - source: Sersic at z_s=1.5, n=1.5, R_e=0.18", offset slightly inside
    the radial caustic so both tangential and radial features form

Imaging: 80x80 px @ 0.05"/px (HST WFC3-IR-like), exp_time=1000 s,
bg_rms=0.01. PSF: Gaussian FWHM=0.12".

Auxiliary mock: synthetic sigma_v aperture measurement at R_eff with
sigma(sigma_v) = 10 km/s, written to sigma_v_dataset.json. Consumed by
the with_kinematics part of the driver via the al.AnalysisKinematics
custom subclass (Module 13 Jeans theory).

Self-consistency check: chi2-at-truth <= 1.5, max|res| <= 5.0 sigma.

Run from repo root:
    python Examples/radial_arc_smbh/mocks/generate_mock.py
"""

from __future__ import annotations

import json
from pathlib import Path

import autogalaxy as ag
import autolens as al
import autolens.plot as aplt
import numpy as np

OUT = Path(__file__).parent

# --- Geometry & cosmology --------------------------------------------------
shape = (80, 80)
pixel_scale = 0.05
exp_time = 1000.0
bg_rms = 0.01
z_l = 0.7
z_s = 1.5
H0, Om0 = 70.0, 0.30
cosmology = ag.cosmo.FlatLambdaCDM(H0=H0, Om0=Om0)

print(f"[mock] geometry: {shape[0]}x{shape[1]} @ {pixel_scale}\"/px, "
      f"exp={exp_time}s, bg_rms={bg_rms}")
print(f"[mock] redshifts: z_l={z_l}, z_s={z_s}")
print(f"[mock] cosmology: FlatLambdaCDM(H0={H0}, Om0={Om0})")

# --- Truth model -----------------------------------------------------------
# Lens mass: PowerLaw + ExternalShear + central PointMass (the SMBH)
theta_E_PL = 1.0
slope_truth = 1.95           # sub-isothermal, matches AGEL findings
theta_E_BH = 0.08            # pedagogical, NOT physical (see file docstring)
gamma_1_truth = 0.025
gamma_2_truth = -0.015

lens = al.Galaxy(
    redshift=z_l,
    bulge=al.lp.Sersic(
        centre=(0.0, 0.0),
        ell_comps=(0.10, 0.05),
        intensity=1.0,
        effective_radius=0.8,
        sersic_index=4.0,
    ),
    mass=al.mp.PowerLaw(
        centre=(0.0, 0.0),
        ell_comps=(0.10, 0.05),
        einstein_radius=theta_E_PL,
        slope=slope_truth,
    ),
    smbh=al.mp.PointMass(
        centre=(0.0, 0.0),
        einstein_radius=theta_E_BH,
    ),
    shear=al.mp.ExternalShear(gamma_1=gamma_1_truth, gamma_2=gamma_2_truth),
)
print(f"[mock] lens: PowerLaw theta_E={theta_E_PL} slope={slope_truth} "
      f"+ PointMass theta_E_BH={theta_E_BH} "
      f"+ ExternalShear (g1,g2)=({gamma_1_truth}, {gamma_2_truth}) "
      f"+ Sersic n=4 R_e=0.8\"")

# Source: small Sersic offset slightly to produce both tangential AND radial
# features. The (0.06, 0.03) offset places it inside both caustics.
source = al.Galaxy(
    redshift=z_s,
    bulge=al.lp.Sersic(
        centre=(0.06, 0.03),
        ell_comps=(0.05, 0.02),
        intensity=0.5,
        effective_radius=0.18,
        sersic_index=1.5,
    ),
)
print("[mock] source: Sersic n=1.5, R_e=0.18\", offset=(0.06, 0.03)")

# --- Build tracer + simulate ----------------------------------------------
tracer = al.Tracer(galaxies=[lens, source], cosmology=cosmology)
grid = al.Grid2D.uniform(shape_native=shape, pixel_scales=pixel_scale,
                         over_sample_size=4)

# Gaussian PSF, FWHM = 0.12"
fwhm = 0.12
sigma_psf = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
psf_shape = (11, 11)
yy, xx = np.indices(psf_shape) - (psf_shape[0] - 1) / 2.0
psf_arr = np.exp(-(yy**2 + xx**2) * pixel_scale**2 / (2.0 * sigma_psf**2))
psf_arr /= psf_arr.sum()
psf_kernel = al.Array2D.no_mask(values=psf_arr, pixel_scales=pixel_scale)
psf = al.Convolver(
    image=al.Array2D.no_mask(values=np.zeros(shape), pixel_scales=pixel_scale),
    kernel=psf_kernel,
)

simulator = al.SimulatorImaging(
    exposure_time=exp_time,
    psf=psf,
    background_sky_level=0.0,
    add_poisson_noise_to_data=True,
    noise_seed=42,
    noise_if_add_noise_false=bg_rms,
)
print("[mock] simulating...")
dataset = simulator.via_tracer_from(tracer=tracer, grid=grid)
print(f"[mock] simulated: data shape {dataset.data.shape_native}")

# --- Write FITS artifacts -------------------------------------------------
img_path = OUT / "image.fits"
noise_path = OUT / "noise_map.fits"
psf_path = OUT / "psf.fits"
aplt.fits_imaging(
    dataset=dataset,
    data_path=img_path,
    psf_path=psf_path,
    noise_map_path=noise_path,
    overwrite=True,
)
print(f"[mock] wrote {img_path.relative_to(OUT.parents[2])}")
print(f"[mock] wrote {noise_path.relative_to(OUT.parents[2])}")
print(f"[mock] wrote {psf_path.relative_to(OUT.parents[2])}")

# --- Synthetic sigma_v aperture measurement -------------------------------
# Module 13 Jeans theory: sigma_v(R_eff) is computed by spherical isotropic
# Jeans solver in `_jeans_sigma_v.py` (v0.97 Phase 3 deliverable). For a
# power-law M(<r) ~ r^(3-gamma) + point mass M_BH and a Sersic n=4 tracer
# of R_eff=0.8", aperture-weighted sigma_v ~ 1.10 × sigma_SIS for this
# geometry. Compute self-consistently here so the fit-time AnalysisKinematics
# class — which uses the SAME solver — can recover truth at the σ_v_err
# precision level.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[3]
                         / "Modules" / "10_Cluster_Computing" / "scripts"))
from _jeans_sigma_v import sigma_v_aperture_isotropic

_cosmo_truth = al.cosmo.FlatLambdaCDM(H0=70.0, Om0=0.30)
_D_l = float(_cosmo_truth.angular_diameter_distance_kpc_z1z2(0.0, z_l))
_D_s = float(_cosmo_truth.angular_diameter_distance_kpc_z1z2(0.0, z_s))
_D_ls = float(_cosmo_truth.angular_diameter_distance_kpc_z1z2(z_l, z_s))
sigma_v_truth = sigma_v_aperture_isotropic(
    theta_E_arcsec=theta_E_PL, slope=slope_truth,
    R_eff_arcsec=0.8, sersic_index=4.0,
    R_aperture_arcsec=0.8,
    D_l_kpc=_D_l, D_s_kpc=_D_s, D_ls_kpc=_D_ls,
    theta_E_BH_arcsec=theta_E_BH,
)
sigma_v_err = 10.0
np.random.seed(99)
sigma_v_obs = sigma_v_truth + np.random.normal(0, sigma_v_err)

sigma_v_dataset = {
    "framework": "synthetic Jeans aperture measurement",
    "R_eff_arcsec": 0.8,
    "aperture_kind": "Reff-circular",
    "sigma_v_obs_kms": float(sigma_v_obs),
    "sigma_v_err_kms": float(sigma_v_err),
    "sigma_v_truth_kms": float(sigma_v_truth),
    "notes": "Truth sigma_v computed self-consistently via "
             "_jeans_sigma_v.sigma_v_aperture_isotropic with the same "
             "PowerLaw+PointMass+Sersic parameters used to simulate the "
             "imaging. Real-data pipeline: ppxf on KCWI/LLAMAS IFU "
             "spectra, aperture matched to R_eff or Sersic kernel.",
}
sigma_v_path = OUT / "sigma_v_dataset.json"
sigma_v_path.write_text(json.dumps(sigma_v_dataset, indent=2))
print(f"[mock] wrote {sigma_v_path.relative_to(OUT.parents[2])}")
print(f"[mock] sigma_v_obs = {sigma_v_obs:.1f} +/- {sigma_v_err:.1f} km/s "
      f"(truth = {sigma_v_truth} km/s)")

# --- Truth JSON ------------------------------------------------------------
truths = {
    "framework": "PyAutoLens 2026.4.13.6 (native simulator)",
    "scenario": "Einstein-spiral analog with embedded SMBH (depth B)",
    "generated_date": "2026-05-14",
    "imaging": {
        "shape_native": list(shape),
        "pixel_scales_arcsec": pixel_scale,
        "exp_time_s": exp_time,
        "background_rms": bg_rms,
        "psf_fwhm_arcsec": fwhm,
    },
    "redshifts": {"lens": z_l, "source": z_s},
    "cosmology": {"H0": H0, "Om": Om0, "w": -1.0},
    "lens": {
        "bulge": {
            "type": "al.lp.Sersic",
            "centre": [0.0, 0.0],
            "ell_comps": [0.10, 0.05],
            "intensity": 1.0,
            "effective_radius": 0.8,
            "sersic_index": 4.0,
        },
        "mass": {
            "type": "al.mp.PowerLaw",
            "centre": [0.0, 0.0],
            "ell_comps": [0.10, 0.05],
            "einstein_radius": theta_E_PL,
            "slope": slope_truth,
        },
        "smbh": {
            "type": "al.mp.PointMass",
            "centre": [0.0, 0.0],
            "einstein_radius": theta_E_BH,
            "note": "Pedagogically-large BH for imaging-only detection. "
                    "Real AGEL UMBHs have theta_E_BH ~ 0.02; v0.97+ "
                    "real-data follow-up requires kinematics.",
        },
        "shear": {
            "type": "al.mp.ExternalShear",
            "gamma_1": gamma_1_truth,
            "gamma_2": gamma_2_truth,
        },
    },
    "source": {
        "bulge": {
            "type": "al.lp.Sersic",
            "centre": [0.06, 0.03],
            "ell_comps": [0.05, 0.02],
            "intensity": 0.5,
            "effective_radius": 0.18,
            "sersic_index": 1.5,
        },
    },
    "kinematics": sigma_v_dataset,
}
truths_path = OUT / "truths.json"
truths_path.write_text(json.dumps(truths, indent=2))
print(f"[mock] wrote {truths_path.relative_to(OUT.parents[2])}")

# --- Self-consistency check (chi^2-at-truth) ------------------------------
print("[mock] running chi^2-at-truth self-consistency check...")
mask = al.Mask2D.circular(
    shape_native=shape, pixel_scales=pixel_scale, radius=1.8,
)
masked = dataset.apply_mask(mask=mask)
fit = al.FitImaging(dataset=masked, tracer=tracer)
chi2_total = float(fit.chi_squared)
n_unmasked = int(np.sum(~mask))
chi2_per_pix = chi2_total / max(n_unmasked, 1)
max_norm_res = float(np.nanmax(np.abs(fit.normalized_residual_map)))
print(f"[mock] chi^2-at-truth: chi^2/N = {chi2_per_pix:.3f} (target <= 1.5)")
print(f"[mock] chi^2-at-truth: max|norm res| = {max_norm_res:.2f} sigma "
      f"(target <= 5.0 sigma)")
assert chi2_per_pix <= 1.5, (
    f"chi^2/N = {chi2_per_pix:.3f} > 1.5 — mock is NOT self-consistent."
)
assert max_norm_res <= 5.0, (
    f"max|res| = {max_norm_res:.2f} sigma > 5.0 sigma — mock is NOT self-consistent."
)
print("[mock] chi^2-at-truth: PASS — mock is self-consistent in autolens.")
