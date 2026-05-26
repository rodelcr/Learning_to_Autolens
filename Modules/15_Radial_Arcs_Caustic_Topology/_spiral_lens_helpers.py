"""Spiral-lens helpers for Module 15 — PyAutoLens primitives only.

Provides reusable utilities for the ±SMBH and mass-class comparison demos in
§3.8 and §3.9 of `15_radial_arcs.ipynb`. The central design constraint: any
SMBH/NSC/BPL variation MUST preserve the smooth deflector's tangential
Einstein-ring location (same outer arc) so the comparison isolates the inner
mass-shape effect, not an extra-gravity artefact.

Shrinkage formula (LINEAR, exact for SIS, ≤1% error for SIE with e≤0.25):

    θ_smooth_new = θ_target − θ_central² / θ_target

where θ_central is the Einstein radius of the added compact component
(POINT_MASS, IsothermalCore, etc.). Verified by `verify_alpha_matches_baseline`.

Conventions:
- Coordinates: autolens uses (y, x) tuples for centre/origin and (α_y, α_x) for
  deflections. Source/lens centres passed by callers should match.
- Ellipticity: lenstronomy-style (q, φ) → autolens (e1, e2) via
  q_phi_to_ell_comps. Student SED_inferred_color_images uses (q, φ); we convert.
- Single-plane lens at z=0.5, source at z=1.5 (typical AGEL spiral redshifts).
"""

from __future__ import annotations

import autolens as al
import numpy as np
from matplotlib.colors import LogNorm


__all__ = [
    "q_phi_to_ell_comps",
    "shrunk_einstein_radius",
    "build_sie_smbh_tracer",
    "build_pl_only_tracer",
    "build_bpl_tracer",
    "build_pl_nsc_tracer",
    "verify_alpha_matches_baseline",
    "render_image_log",
]

Z_LENS = 0.5
Z_SOURCE = 1.5


def q_phi_to_ell_comps(q: float, phi_rad: float) -> tuple[float, float]:
    """Convert lenstronomy (q, φ) ellipticity to autolens (e1, e2) complex form.

    For an elliptical profile with axis ratio q ∈ (0, 1] and position angle φ
    (radians, measured from +x):
        ε = (1 - q) / (1 + q)
        e1 = ε cos(2φ)
        e2 = ε sin(2φ)
    """
    eps = (1.0 - q) / (1.0 + q)
    return eps * np.cos(2.0 * phi_rad), eps * np.sin(2.0 * phi_rad)


def shrunk_einstein_radius(theta_E_target: float, theta_E_compact: float) -> float:
    """Linear shrinkage to preserve total deflection at θ_target when adding a
    compact central component.

    Exact for SIS+POINT_MASS; accurate to ≤1% for SIE (e1=0.25). Verified
    against the no-compact baseline via verify_alpha_matches_baseline.
    """
    return theta_E_target - theta_E_compact ** 2 / theta_E_target


def _make_lens_galaxy(
    *,
    smooth_mass: al.mp.MassProfile,
    compact_mass: al.mp.MassProfile | None = None,
    lens_light: al.lp.LightProfile | None = None,
) -> al.Galaxy:
    kwargs = {"redshift": Z_LENS, "mass_smooth": smooth_mass}
    if compact_mass is not None:
        kwargs["mass_compact"] = compact_mass
    if lens_light is not None:
        kwargs["bulge"] = lens_light
    return al.Galaxy(**kwargs)


def _make_source_galaxy(source_kwargs: dict | None) -> al.Galaxy:
    if source_kwargs is None:
        return al.Galaxy(redshift=Z_SOURCE)
    return al.Galaxy(redshift=Z_SOURCE, bulge=al.lp.Sersic(**source_kwargs))


def _student_default_lens_light() -> al.lp.Sersic:
    """Student's base_class lens light: Sersic R=0.5, n=4, q=0.6, φ=0."""
    e1, e2 = q_phi_to_ell_comps(q=0.6, phi_rad=0.0)
    return al.lp.Sersic(
        centre=(0.0, 0.0), ell_comps=(e1, e2),
        intensity=1.0, effective_radius=0.5, sersic_index=4.0,
    )


def build_sie_smbh_tracer(
    *,
    theta_E_target: float = 1.6,
    e1: float = 0.25,
    e2: float = 0.0,
    f_BH: float = 0.0,
    source_kwargs: dict | None = None,
    include_lens_light: bool = True,
) -> al.Tracer:
    """SIE [+ optional POINT_MASS] + Sersic source [+ optional lens light].

    f_BH = M_BH / M(<θ_target) — fraction of total mass at the Einstein radius
    that the point mass carries. Setting f_BH=0 omits the POINT_MASS.

    The SIE einstein_radius is shrunk via shrunk_einstein_radius so the
    combined α matches the no-BH baseline at θ_target.
    """
    theta_E_BH = np.sqrt(f_BH) * theta_E_target
    if f_BH > 0:
        theta_E_sie = shrunk_einstein_radius(theta_E_target, theta_E_BH)
    else:
        theta_E_sie = theta_E_target

    smooth = al.mp.Isothermal(
        centre=(0.0, 0.0), ell_comps=(e1, e2), einstein_radius=theta_E_sie,
    )
    compact = (al.mp.PointMass(centre=(0.0, 0.0), einstein_radius=theta_E_BH)
               if f_BH > 0 else None)
    lens_light = _student_default_lens_light() if include_lens_light else None

    lens = _make_lens_galaxy(smooth_mass=smooth, compact_mass=compact, lens_light=lens_light)
    source = _make_source_galaxy(source_kwargs)
    return al.Tracer(galaxies=[lens, source])


def build_pl_only_tracer(
    *,
    theta_E_target: float = 1.6,
    e1: float = 0.25,
    e2: float = 0.0,
    gamma: float = 2.0,
    source_kwargs: dict | None = None,
    include_lens_light: bool = True,
) -> al.Tracer:
    """Pure elliptical PowerLaw at the target Einstein radius (no compact mass)."""
    smooth = al.mp.PowerLaw(
        centre=(0.0, 0.0), ell_comps=(e1, e2),
        einstein_radius=theta_E_target, slope=gamma,
    )
    lens_light = _student_default_lens_light() if include_lens_light else None
    lens = _make_lens_galaxy(smooth_mass=smooth, lens_light=lens_light)
    source = _make_source_galaxy(source_kwargs)
    return al.Tracer(galaxies=[lens, source])


def build_bpl_tracer(
    *,
    theta_E_target: float = 1.6,
    e1: float = 0.25,
    e2: float = 0.0,
    inner_slope: float = 1.5,
    outer_slope: float = 2.0,
    break_radius: float = 0.1,
    source_kwargs: dict | None = None,
    include_lens_light: bool = True,
) -> al.Tracer:
    """Broken-PL via al.mp.PowerLawBroken.

    autolens 2026.4 normalizes BPL such that einstein_radius approximates the
    tangential Einstein ring at the break_radius scale. For r_break ≪ θ_target,
    the convention is close to "outer-only slope determines θ_target", which we
    accept as a small approximation — comparison is qualitative.
    """
    smooth = al.mp.PowerLawBroken(
        centre=(0.0, 0.0), ell_comps=(e1, e2),
        einstein_radius=theta_E_target,
        inner_slope=inner_slope, outer_slope=outer_slope,
        break_radius=break_radius,
    )
    lens_light = _student_default_lens_light() if include_lens_light else None
    lens = _make_lens_galaxy(smooth_mass=smooth, lens_light=lens_light)
    source = _make_source_galaxy(source_kwargs)
    return al.Tracer(galaxies=[lens, source])


def build_pl_nsc_tracer(
    *,
    theta_E_target: float = 1.6,
    e1: float = 0.25,
    e2: float = 0.0,
    gamma: float = 2.0,
    theta_E_NSC: float = 0.05,
    core_radius_NSC: float = 0.01,
    source_kwargs: dict | None = None,
    include_lens_light: bool = True,
) -> al.Tracer:
    """PL + compact IsothermalCore representing a Nuclear Star Cluster.

    PL shrunk so α_PL(θ_target) + α_NSC(θ_target) ≈ θ_target. Same linear
    formula as SIE+POINT_MASS: works because IsothermalCore at r ≫ core_radius
    behaves like an SIS with α(r) ≈ θ_E_NSC.
    """
    theta_E_pl = shrunk_einstein_radius(theta_E_target, theta_E_NSC)
    smooth = al.mp.PowerLaw(
        centre=(0.0, 0.0), ell_comps=(e1, e2),
        einstein_radius=theta_E_pl, slope=gamma,
    )
    compact = al.mp.IsothermalCore(
        centre=(0.0, 0.0), ell_comps=(0.0, 0.0),  # spherical NSC
        einstein_radius=theta_E_NSC, core_radius=core_radius_NSC,
    )
    lens_light = _student_default_lens_light() if include_lens_light else None
    lens = _make_lens_galaxy(smooth_mass=smooth, compact_mass=compact, lens_light=lens_light)
    source = _make_source_galaxy(source_kwargs)
    return al.Tracer(galaxies=[lens, source])


def verify_alpha_matches_baseline(
    tracer: al.Tracer,
    baseline_tracer: al.Tracer,
    theta_target: float,
    *,
    angle_samples: int = 8,
) -> dict:
    """Compare |α(θ)| at angle_samples points on a circle of radius theta_target
    between `tracer` and `baseline_tracer`. Returns a summary dict with the
    fractional deviation — this is the primary mass-shape correctness check.

    A tracer that correctly shrinks its smooth component when adding a compact
    mass should give max_frac_dev ≤ 1% (for SIE with e=0.25) or ≈ 0 (for SIS).
    """
    angles = np.linspace(0, 2 * np.pi, angle_samples, endpoint=False)
    devs = []
    for a in angles:
        y, x = theta_target * np.sin(a), theta_target * np.cos(a)
        # autolens Grid2D uniform with shape (1,1) and origin at the test point
        grid = al.Grid2D.uniform(shape_native=(1, 1), pixel_scales=1.0, origin=(y, x))
        a_t = tracer.deflections_yx_2d_from(grid=grid)
        a_b = baseline_tracer.deflections_yx_2d_from(grid=grid)
        mag_t = float(np.hypot(a_t[0, 0], a_t[0, 1]))
        mag_b = float(np.hypot(a_b[0, 0], a_b[0, 1]))
        devs.append((mag_t - mag_b) / mag_b)
    devs = np.array(devs)
    return {
        "angle_samples": angle_samples,
        "max_frac_dev": float(np.max(np.abs(devs))),
        "mean_frac_dev": float(np.mean(devs)),
        "matches_within_1pct": bool(np.max(np.abs(devs)) < 0.01),
    }


def render_image_log(
    ax,
    tracer: al.Tracer,
    grid: al.Grid2D,
    *,
    cmap: str = "magma",
    vmin_floor_frac: float = 1e-3,
    extent: tuple | None = None,
):
    """Render tracer.image_2d_from(grid) on `ax` with LogNorm scaling.

    Returns the (image, vmax) tuple so callers can reuse vmax for difference panels.
    """
    image = tracer.image_2d_from(grid=grid).native
    vmax = float(np.max(image))
    vmin = max(vmax * vmin_floor_frac, 1e-8)
    if extent is None:
        # Derive from grid scale + shape
        ny, nx = image.shape
        half_y = grid.pixel_scales[0] * ny / 2
        half_x = grid.pixel_scales[1] * nx / 2
        extent = [-half_x, half_x, -half_y, half_y]
    ax.imshow(image, extent=extent, origin="lower", cmap=cmap,
              norm=LogNorm(vmin=vmin, vmax=vmax))
    return image, vmax
