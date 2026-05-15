"""
_jeans_sigma_v.py — shared isotropic-Jeans σ_v module + AnalysisKinematics.

Phase 3 deliverable (task #122). Single file consumed by:

  - fit_example_radial_arc_smbh.py --part=with_kinematics
      (γ′–M_BH degeneracy break via stellar σ_v)

  - fit_example_quad_time_delay.py --part=joint_fit_h0_kin
      (TDCOSMO H0 + MSD break via kinematics, Birrer+2020 IV methodology)

Both drivers wrap an `AnalysisKinematics` instance in `af.AnalysisFactor` and
combine it with the imaging/point analyses via `af.FactorGraphModel`. The
joint log-likelihood is the natural sum (independent measurements).

Scope (v0.97 starter — isotropic spherical Jeans, β=0):

  - PowerLaw mass profile with optional central PointMass (SMBH)
  - Sersic tracer for the stellar light, deprojected via the Lima Neto,
    Gerbal & Marquez (1999) approximation [eq. 11 of LGM99]
  - Circular aperture σ_v, light-weighted
  - Returns scalar σ_v in km/s for a gaussian likelihood comparison

References:
  - Binney & Tremaine 2008, eq. 4.215 (isotropic spherical Jeans)
  - Lima Neto, Gerbal & Marquez 1999 (LGM99) — analytic Sersic 3D approx
  - Mamon & Łokas 2005 — anisotropy-kernel extension (v0.98+ scope)
  - Bolton+2008 SLACS V eq. 5, Sonnenfeld+2013 SL2S — observational baseline

Anisotropy (β ≠ 0) is NOT in this v0.97 starter. The hooks are there
(see `beta` arg) but only β=0 is exercised. The Mamon & Łokas 2005
kernel for β=const can be slotted into `_jeans_inner_integral` without
disturbing the API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


# Physical constants
_G_KPC_KMS2_PER_MSUN = 4.30091e-6        # G in (km/s)² · kpc / M_sun
_C_KMS = 2.99792458e5                    # speed of light in km/s


def _b_n(n: float) -> float:
    """Sersic b_n from Ciotti & Bertin 1999 5th-order expansion."""
    return (2.0 * n - 1.0 / 3.0 + 4.0 / (405.0 * n)
            + 46.0 / (25515.0 * n ** 2)
            + 131.0 / (1148175.0 * n ** 3)
            - 2194697.0 / (30690717750.0 * n ** 4))


def _p_n(n: float) -> float:
    """LGM99 deprojection slope: ρ_3D(r) ∝ (r/R_e)^(-p_n) exp(-b_n (r/R_e)^(1/n))."""
    return 1.0 - 0.6097 / n + 0.05463 / n ** 2


def _nu_sersic_3d(r_over_Re: np.ndarray, n: float) -> np.ndarray:
    """LGM99 3D Sersic luminosity density (relative units; only ratios matter).

    ρ_3D(r) = (r/R_e)^(-p_n) exp(-b_n (r/R_e)^(1/n))

    Good to ~1% within R_e for n ∈ [0.5, 8]; degrades at extreme radii but
    the Jeans integrals are dominated by 0.1 R_e ≲ r ≲ few R_e.
    """
    bn, pn = _b_n(n), _p_n(n)
    x = np.clip(r_over_Re, 1e-30, None)
    return x ** (-pn) * np.exp(-bn * x ** (1.0 / n))


def _sigma_sersic_2d(R_over_Re: np.ndarray, n: float) -> np.ndarray:
    """2D Sersic surface brightness profile (Ciotti & Bertin convention).

    I(R) = I_0 exp(-b_n (R/R_e)^(1/n))
    """
    bn = _b_n(n)
    x = np.clip(R_over_Re, 0.0, None)
    return np.exp(-bn * x ** (1.0 / n))


def _powerlaw_mass_3d_from_thetaE(r_kpc: np.ndarray,
                                  theta_E_kpc: float,
                                  slope: float,
                                  M_einstein: float) -> np.ndarray:
    """3D enclosed mass for an axisymmetric power-law deflector.

    Convention (autolens / SLACS): slope γ′ in 2D, ρ_3D(r) ∝ r^(-γ′).
    M_3D(<r) = M_einstein × (r / θ_E)^(3 - γ′)
    where M_einstein is the projected mass within the Einstein radius,
    M_einstein = π Σ_cr θ_E²  (in physical units).

    For γ′=2 (isothermal), M_3D(<r) ∝ r — the cumulative mass grows
    linearly with radius, as expected for an SIS.
    """
    return M_einstein * (np.clip(r_kpc, 1e-30, None) / theta_E_kpc) ** (3.0 - slope)


def _sigma_critical_Msun_per_kpc2(D_l_kpc: float,
                                  D_s_kpc: float,
                                  D_ls_kpc: float) -> float:
    """Critical surface density Σ_cr in M_sun / kpc².

    Σ_cr = c² / (4 π G) × D_s / (D_l D_ls)
    """
    return (_C_KMS ** 2) / (4.0 * np.pi * _G_KPC_KMS2_PER_MSUN) * D_s_kpc / (D_l_kpc * D_ls_kpc)


def _arcsec_to_kpc(arcsec: float, D_kpc: float) -> float:
    """Small-angle conversion: 1 arcsec = D × (π/180/3600) kpc."""
    return arcsec * D_kpc * np.pi / 180.0 / 3600.0


def _M_pointmass_from_thetaE_BH(theta_E_BH_arcsec: float,
                                D_l_kpc: float,
                                D_s_kpc: float,
                                D_ls_kpc: float) -> float:
    """Point-mass M_BH in M_sun from autolens PointMass.einstein_radius.

    M_BH = c² θ_E_BH² D_l D_s / (4 G D_ls)   (point-mass lens equation)
    """
    theta_E_BH_rad = theta_E_BH_arcsec * np.pi / 180.0 / 3600.0
    return ((_C_KMS ** 2) * theta_E_BH_rad ** 2 * D_l_kpc * D_s_kpc
            / (4.0 * _G_KPC_KMS2_PER_MSUN * D_ls_kpc))


def sigma_v_aperture_isotropic(
    *,
    theta_E_arcsec: float,
    slope: float,
    R_eff_arcsec: float,
    sersic_index: float,
    R_aperture_arcsec: float,
    D_l_kpc: float,
    D_s_kpc: float,
    D_ls_kpc: float,
    theta_E_BH_arcsec: float = 0.0,
    n_r: int = 120,
    r_max_factor: float = 50.0,
    beta: float = 0.0,
) -> float:
    """Aperture-integrated isotropic σ_v (km/s) for PowerLaw lens + Sersic tracer.

    Solves the spherical Jeans equation with β=0 numerically:

        σ_r²(r) = (G / ν(r)) ∫_r^∞ ν(r') M(<r') / r'² dr'

    projects to line-of-sight at each R via

        σ_los²(R) = (2 / Σ(R)) ∫_R^∞ ν(r) σ_r²(r) r / √(r² − R²) dr

    and aperture-averages with the 2D Sersic light weighting

        σ_ap² = ∫_0^Rap σ_los²(R) Σ(R) 2πR dR / ∫_0^Rap Σ(R) 2πR dR.

    The SMBH contribution is added as M_3D += M_BH (Dirac at r=0).

    Args:
        theta_E_arcsec, slope: PowerLaw lens parameters.
        R_eff_arcsec, sersic_index: Sersic tracer (lens light) parameters.
        R_aperture_arcsec: Circular aperture radius for the σ_v measurement.
        D_l_kpc, D_s_kpc, D_ls_kpc: angular-diameter distances (physical kpc).
        theta_E_BH_arcsec: central PointMass Einstein radius (0 if no BH).
        n_r: grid resolution (>= 80 for ~0.5% σ_v convergence).
        r_max_factor: outer-truncation radius in units of R_eff_arcsec.
        beta: anisotropy parameter (UNUSED in v0.97 starter — isotropic only).

    Returns:
        σ_v in km/s (scalar).
    """
    if beta != 0.0:
        raise NotImplementedError(
            "Anisotropy (β ≠ 0) is a v0.98 hook. v0.97 starter is isotropic only."
        )

    # All physical computations in kpc + M_sun.
    theta_E_kpc = _arcsec_to_kpc(theta_E_arcsec, D_l_kpc)
    R_eff_kpc = _arcsec_to_kpc(R_eff_arcsec, D_l_kpc)
    R_ap_kpc = _arcsec_to_kpc(R_aperture_arcsec, D_l_kpc)

    Sigma_cr = _sigma_critical_Msun_per_kpc2(D_l_kpc, D_s_kpc, D_ls_kpc)
    M_einstein = np.pi * Sigma_cr * theta_E_kpc ** 2

    M_BH = _M_pointmass_from_thetaE_BH(
        theta_E_BH_arcsec, D_l_kpc, D_s_kpc, D_ls_kpc
    ) if theta_E_BH_arcsec > 0 else 0.0

    # 3D radial grid spanning [0.01 R_eff, r_max_factor * R_eff]
    r = np.logspace(np.log10(0.01 * R_eff_kpc),
                    np.log10(r_max_factor * R_eff_kpc),
                    n_r)

    nu = _nu_sersic_3d(r / R_eff_kpc, sersic_index)

    M_pl = _powerlaw_mass_3d_from_thetaE(r, theta_E_kpc, slope, M_einstein)
    M_enc = M_pl + M_BH  # PointMass contributes uniformly inside

    # Inner Jeans integral: I(r) = ∫_r^∞ ν(r') M(<r') / r'² dr'
    # Reverse cumulative trapezoidal (numpy <2 has no cumulative_trapezoid).
    integrand_jeans = nu * M_enc / r ** 2
    dr = np.diff(r)
    avg = 0.5 * (integrand_jeans[1:] + integrand_jeans[:-1])
    seg = avg * dr                       # per-segment integral
    I_r = np.empty_like(r)
    I_r[-1] = 0.0
    I_r[:-1] = np.flip(np.cumsum(np.flip(seg)))

    sigma_r_sq = _G_KPC_KMS2_PER_MSUN * I_r / np.clip(nu, 1e-300, None)

    # Aperture grid in projected radius R ∈ (0, R_ap_kpc]
    R_grid = np.linspace(0.001 * R_eff_kpc, R_ap_kpc, n_r)
    sigma_los_sq = np.empty_like(R_grid)
    Sigma_R = np.empty_like(R_grid)

    for i, R in enumerate(R_grid):
        m = r > R
        rp = r[m]
        denom = np.sqrt(rp ** 2 - R ** 2)
        Sigma_R[i] = 2.0 * np.trapz(nu[m] * rp / denom, rp)
        num_int = 2.0 * np.trapz(nu[m] * sigma_r_sq[m] * rp / denom, rp)
        sigma_los_sq[i] = num_int / max(Sigma_R[i], 1e-300)

    # Aperture-light-weighted average
    weight = Sigma_R * R_grid  # 2D light × Jacobian
    sigma_ap_sq = (np.trapz(weight * sigma_los_sq, R_grid)
                   / max(np.trapz(weight, R_grid), 1e-300))

    return float(np.sqrt(max(sigma_ap_sq, 0.0)))


# ============================================================================
# autofit Analysis subclass
# ============================================================================

@dataclass
class KinematicDataset:
    """Single-aperture σ_v measurement + spec.

    Mirrors the JSON-on-disk format used by Examples/*/mocks/sigma_v_dataset.json.
    """
    sigma_v_obs_kms: float
    sigma_v_err_kms: float
    R_aperture_arcsec: float
    aperture_kind: str = "circular"   # only "circular" / "Reff-circular" in v0.97

    @classmethod
    def from_json(cls, path) -> "KinematicDataset":
        import json
        from pathlib import Path
        d = json.loads(Path(path).read_text())
        return cls(
            sigma_v_obs_kms=float(d["sigma_v_obs_kms"]),
            sigma_v_err_kms=float(d["sigma_v_err_kms"]),
            R_aperture_arcsec=float(d.get("R_eff_arcsec",
                                          d.get("R_aperture_arcsec"))),
            aperture_kind=d.get("aperture_kind", "circular"),
        )


def _get_powerlaw_params(galaxy):
    """Pull (θ_E, γ′, R_eff, n) from an autolens Galaxy instance.

    Assumes mass is PowerLaw / Isothermal (γ′=2) and bulge is Sersic.
    Raises AttributeError if the model shape is unexpected — callers
    should catch and report.
    """
    mass = galaxy.mass
    bulge = galaxy.bulge
    theta_E = float(mass.einstein_radius)
    # Isothermal has no `slope` attr; treat as γ′=2.
    slope = float(getattr(mass, "slope", 2.0))
    R_eff = float(bulge.effective_radius)
    n_s = float(bulge.sersic_index)
    return theta_E, slope, R_eff, n_s


def _get_pointmass_thetaE(galaxy) -> float:
    """Return θ_E_BH (arcsec) if a PointMass is present, else 0."""
    smbh = getattr(galaxy, "smbh", None)
    if smbh is None:
        return 0.0
    return float(getattr(smbh, "einstein_radius", 0.0))


class AnalysisKinematics:
    """autofit Analysis for a single σ_v aperture measurement.

    Joint use:
        analysis_imaging   = al.AnalysisImaging(dataset=imaging_dataset)
        analysis_kinematic = AnalysisKinematics(dataset=kin, z_lens=0.7, z_source=1.5)
        af_im  = af.AnalysisFactor(prior_model=model, analysis=analysis_imaging,   name="imaging")
        af_kin = af.AnalysisFactor(prior_model=model, analysis=analysis_kinematic, name="kinematic")
        factor_graph = af.FactorGraphModel(af_im, af_kin)
        result_list = search.fit(model=factor_graph.global_prior_model, analysis=factor_graph)

    The class is a plain Python object (not subclassing `af.Analysis`); autofit's
    AnalysisFactor only requires a `log_likelihood_function(instance)` method.
    Keeping it framework-agnostic avoids the subtle issues we hit in
    fit_example_quad_time_delay.py with autolens-vs-autofit class hierarchy
    (`_make_robust_analysis_point`).
    """

    def __init__(self,
                 dataset: KinematicDataset,
                 z_lens: float,
                 z_source: float,
                 cosmology=None,
                 lens_galaxy_index: int = 0,
                 use_smbh: bool = True,
                 n_r: int = 120):
        self.dataset = dataset
        self.z_lens = z_lens
        self.z_source = z_source
        self.lens_galaxy_index = lens_galaxy_index
        self.use_smbh = use_smbh
        self.n_r = n_r

        if cosmology is None:
            import autolens as al
            cosmology = al.cosmo.FlatLambdaCDM(H0=70.0, Om0=0.30)
        self.cosmology = cosmology

        # Pre-compute fixed-cosmology distances once. If cosmology is a
        # free parameter in the model (TDCOSMO H0 case), call sites must
        # use AnalysisKinematicsFreeCosmology below instead.
        self._D_l_kpc, self._D_s_kpc, self._D_ls_kpc = self._distances(cosmology)

    def _distances(self, cosmology):
        """Return (D_l, D_s, D_ls) in physical kpc.

        autolens FlatLambdaCDM exposes `angular_diameter_distance_kpc_z1z2(z1, z2)`
        which returns a float in kpc. For source distance use z1=0.
        """
        D_l = float(cosmology.angular_diameter_distance_kpc_z1z2(0.0, self.z_lens))
        D_s = float(cosmology.angular_diameter_distance_kpc_z1z2(0.0, self.z_source))
        D_ls = float(cosmology.angular_diameter_distance_kpc_z1z2(self.z_lens, self.z_source))
        return D_l, D_s, D_ls

    def log_likelihood_function(self, instance) -> float:
        """Gaussian σ_v likelihood at this model instance."""
        galaxy = instance.galaxies[self.lens_galaxy_index]

        try:
            theta_E, slope, R_eff, n_sersic = _get_powerlaw_params(galaxy)
        except (AttributeError, TypeError) as e:
            # Model shape mismatch — penalise but don't crash the chain.
            return -1.0e9

        theta_E_BH = _get_pointmass_thetaE(galaxy) if self.use_smbh else 0.0

        try:
            sigma_v_pred = sigma_v_aperture_isotropic(
                theta_E_arcsec=theta_E,
                slope=slope,
                R_eff_arcsec=R_eff,
                sersic_index=n_sersic,
                R_aperture_arcsec=self.dataset.R_aperture_arcsec,
                D_l_kpc=self._D_l_kpc,
                D_s_kpc=self._D_s_kpc,
                D_ls_kpc=self._D_ls_kpc,
                theta_E_BH_arcsec=theta_E_BH,
                n_r=self.n_r,
            )
        except (ValueError, FloatingPointError, ZeroDivisionError):
            return -1.0e9

        if not np.isfinite(sigma_v_pred):
            return -1.0e9

        residual = (sigma_v_pred - self.dataset.sigma_v_obs_kms) / self.dataset.sigma_v_err_kms
        # Drop the constant -0.5 log(2π σ²) term — autofit doesn't care.
        return float(-0.5 * residual ** 2)


class AnalysisKinematicsFreeCosmology(AnalysisKinematics):
    """Variant that reads the cosmology *from the model instance* each call.

    Use this when the joint model has `cosmology` as a free af.Model parameter
    (e.g. TDCOSMO H0 break: `instance.cosmology` is FlatLambdaCDM(H0=free, Om0=0.3)).
    """

    def log_likelihood_function(self, instance) -> float:
        if not hasattr(instance, "cosmology"):
            return super().log_likelihood_function(instance)

        cosmology = instance.cosmology
        D_l, D_s, D_ls = self._distances(cosmology)

        # Replicate the parent body with the model-derived distances.
        galaxy = instance.galaxies[self.lens_galaxy_index]
        try:
            theta_E, slope, R_eff, n_sersic = _get_powerlaw_params(galaxy)
        except (AttributeError, TypeError):
            return -1.0e9

        theta_E_BH = _get_pointmass_thetaE(galaxy) if self.use_smbh else 0.0

        try:
            sigma_v_pred = sigma_v_aperture_isotropic(
                theta_E_arcsec=theta_E, slope=slope,
                R_eff_arcsec=R_eff, sersic_index=n_sersic,
                R_aperture_arcsec=self.dataset.R_aperture_arcsec,
                D_l_kpc=D_l, D_s_kpc=D_s, D_ls_kpc=D_ls,
                theta_E_BH_arcsec=theta_E_BH, n_r=self.n_r,
            )
        except (ValueError, FloatingPointError, ZeroDivisionError):
            return -1.0e9

        if not np.isfinite(sigma_v_pred):
            return -1.0e9

        residual = (sigma_v_pred - self.dataset.sigma_v_obs_kms) / self.dataset.sigma_v_err_kms
        return float(-0.5 * residual ** 2)


# ============================================================================
# Self-test / driver-truth sanity check
# ============================================================================

def _self_test():
    """Sanity check: σ_v(truth) ≈ truth for radial_arc_smbh mock.

    Truth: θ_E=1.0″, γ′=1.95, R_eff=0.8″, n=4.0, R_ap=0.8″,
           θ_E_BH=0.08″, z_l=0.7, z_s=1.5, σ_v_truth_kms=280.0
    """
    import autolens as al

    cosmo = al.cosmo.FlatLambdaCDM(H0=70.0, Om0=0.30)
    D_l = float(cosmo.angular_diameter_distance_kpc_z1z2(0.0, 0.7))
    D_s = float(cosmo.angular_diameter_distance_kpc_z1z2(0.0, 1.5))
    D_ls = float(cosmo.angular_diameter_distance_kpc_z1z2(0.7, 1.5))

    sigma_v = sigma_v_aperture_isotropic(
        theta_E_arcsec=1.0, slope=1.95,
        R_eff_arcsec=0.8, sersic_index=4.0,
        R_aperture_arcsec=0.8,
        D_l_kpc=D_l, D_s_kpc=D_s, D_ls_kpc=D_ls,
        theta_E_BH_arcsec=0.08,
    )
    print(f"σ_v(truth, isotropic Jeans, with BH) = {sigma_v:.2f} km/s")
    print(f"  expected (mock truth):              280.00 km/s")
    print(f"  ratio:                              {sigma_v/280.0:.3f}")
    return sigma_v


if __name__ == "__main__":
    _self_test()
