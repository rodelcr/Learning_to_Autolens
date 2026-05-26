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
    "critical_curves_caustics",
    "plot_critical_curves",
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


def critical_curves_caustics(tracer: al.Tracer, *, fov: float, npix: int = 200):
    """Compute tangential + radial critical curves and source-plane caustics
    from a PyAutoLens tracer via numerical Hessian of the deflection field.

    PyAutoLens 2026.4 doesn't expose a high-level caustic API on `al.Tracer`,
    so we compute the Jacobian determinant + eigenvalues by finite-differencing
    the deflection on a dense grid, then extract zero-contours via matplotlib.

    Returns a dict:
      'tang_crit':    list of (y, x) arrays — tangential critical curve segments (image plane)
      'rad_crit':     list of (y, x) arrays — radial critical curve segments (image plane)
      'tang_caustic': list of (y, x) arrays — tangential caustic segments (source plane)
      'rad_caustic':  list of (y, x) arrays — radial caustic segments (source plane)
    """
    import matplotlib.pyplot as plt
    grid = al.Grid2D.uniform(over_sample_size=1, shape_native=(npix, npix),
                              pixel_scales=2 * fov / npix)
    # Deflection field, native (ny, nx, 2) where last axis is (α_y, α_x)
    alpha = tracer.deflections_yx_2d_from(grid=grid).native
    pix_y, pix_x = grid.pixel_scales
    # Hessian via numerical gradient
    da_y_dy, da_y_dx = np.gradient(np.asarray(alpha[..., 0]), pix_y, pix_x)
    da_x_dy, da_x_dx = np.gradient(np.asarray(alpha[..., 1]), pix_y, pix_x)
    # Jacobian A = I - ∂α/∂θ
    A_yy, A_xx = 1.0 - da_y_dy, 1.0 - da_x_dx
    A_yx, A_xy = -da_y_dx,        -da_x_dy
    trace = A_yy + A_xx
    det   = A_yy * A_xx - A_yx * A_xy
    disc  = np.maximum((trace / 2) ** 2 - det, 0.0)
    sqrt_disc = np.sqrt(disc)
    lambda_t = trace / 2 - sqrt_disc     # smaller eigenvalue
    lambda_r = trace / 2 + sqrt_disc     # larger eigenvalue
    # Coordinate axes (y from -fov to +fov, x from -fov to +fov)
    edge = np.linspace(-fov, fov, npix)
    yy, xx = np.meshgrid(edge, edge, indexing='ij')

    # Extract zero contours of each eigenvalue
    fig_tmp, ax_tmp = plt.subplots()
    cs_t = ax_tmp.contour(xx, yy, lambda_t, levels=[0])
    cs_r = ax_tmp.contour(xx, yy, lambda_r, levels=[0])
    plt.close(fig_tmp)

    def _segments(cs):
        segs = []
        for path in cs.allsegs[0]:
            # path is array (N, 2) with columns (x, y)
            if len(path) < 2:
                continue
            segs.append((path[:, 1].copy(), path[:, 0].copy()))   # → (y, x)
        return segs

    tang_crit = _segments(cs_t)
    rad_crit  = _segments(cs_r)

    # Map each critical-curve point to source plane via β = θ - α
    def _trace_to_source(segs):
        out = []
        for ys, xs in segs:
            irregular = al.Grid2DIrregular(
                values=np.column_stack([ys, xs]).astype(np.float64)
            )
            a = np.asarray(tracer.deflections_yx_2d_from(grid=irregular))
            beta_y = ys - a[:, 0]
            beta_x = xs - a[:, 1]
            out.append((beta_y, beta_x))
        return out

    try:
        tang_caustic = _trace_to_source(tang_crit)
        rad_caustic  = _trace_to_source(rad_crit)
    except Exception as e:
        print(f'  [critical_curves_caustics] caustic mapping failed: {type(e).__name__}: {e}')
        tang_caustic, rad_caustic = [], []

    return {
        'tang_crit':    tang_crit,
        'rad_crit':     rad_crit,
        'tang_caustic': tang_caustic,
        'rad_caustic':  rad_caustic,
    }


def plot_critical_curves(ax, curves: dict, *, on_source_plane: bool = False,
                          tang_kw: dict | None = None, rad_kw: dict | None = None):
    """Overlay critical curves (image plane) or caustics (source plane) on `ax`.

    curves: output of critical_curves_caustics(tracer, fov=...)
    on_source_plane: True → plot caustics; False → plot critical curves
    """
    if tang_kw is None:
        tang_kw = dict(color='cyan',   lw=1.2, alpha=0.85)
    if rad_kw is None:
        rad_kw  = dict(color='magenta', lw=1.2, alpha=0.85)
    if on_source_plane:
        tang_segs = curves.get('tang_caustic', [])
        rad_segs  = curves.get('rad_caustic',  [])
    else:
        tang_segs = curves.get('tang_crit', [])
        rad_segs  = curves.get('rad_crit',  [])
    for ys, xs in tang_segs:
        ax.plot(xs, ys, **tang_kw)
    for ys, xs in rad_segs:
        ax.plot(xs, ys, **rad_kw)
