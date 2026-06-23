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
    "build_pl_smbh_tracer",
    "build_spiral_source_galaxy",
    "build_clumpy_spiral_source",
    "spiral_source",
    "analytic_axisym_caustics",
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


def build_pl_smbh_tracer(
    *,
    theta_E_target: float = 1.0,
    e1: float = 0.10,
    e2: float = 0.05,
    gamma: float = 1.8,
    theta_E_BH: float = 0.08,
    source_galaxy: "al.Galaxy | None" = None,
    source_kwargs: dict | None = None,
    include_lens_light: bool = False,
) -> al.Tracer:
    """SUB-ISOTHERMAL PowerLaw [+ optional central PointMass] + source.

    The Einstein-spiral base model: a sub-isothermal (gamma < 2) total mass —
    physically the DM-dominated, shallow-inner-slope environment where radial
    critical curves exist (Cerny+2025) — plus an optional central point mass
    (SMBH). The PowerLaw einstein_radius is shrunk via `shrunk_einstein_radius`
    so the combined deflection at theta_E_target matches the no-BH baseline
    (same tangential ring; the comparison isolates the INNER mass shape).

    Pass a prebuilt `source_galaxy` (e.g. from `build_spiral_source_galaxy`) to
    use a structured source; otherwise `source_kwargs` builds a single Sersic.
    `include_lens_light` defaults False — figures expose the arcs cleanly.
    """
    if theta_E_BH > 0:
        theta_E_pl = shrunk_einstein_radius(theta_E_target, theta_E_BH)
    else:
        theta_E_pl = theta_E_target
    smooth = al.mp.PowerLaw(
        centre=(0.0, 0.0), ell_comps=(e1, e2),
        einstein_radius=theta_E_pl, slope=gamma,
    )
    compact = (al.mp.PointMass(centre=(0.0, 0.0), einstein_radius=theta_E_BH)
               if theta_E_BH > 0 else None)
    lens_light = _student_default_lens_light() if include_lens_light else None
    lens = _make_lens_galaxy(smooth_mass=smooth, compact_mass=compact,
                             lens_light=lens_light)
    if source_galaxy is None:
        source_galaxy = _make_source_galaxy(source_kwargs)
    return al.Tracer(galaxies=[lens, source_galaxy])


def build_spiral_source_galaxy(
    *,
    centre: tuple[float, float] = (0.0, 0.0),
    bulge_Re: float = 0.05,
    arm_Re: float = 0.06,
    arm_offset: float = 0.30,
    bulge_I: float = 0.8,
    arm_I: float = 0.6,
) -> al.Galaxy:
    """Structured 'spiral' source: a central bulge + two off-axis arm blobs.

    Built so the source straddles the radial caustic (arms reach |beta| ~
    arm_offset) AND samples the centre — producing a tangential arc/ring from
    the outer parts and a radial arc from the inner parts of the SAME source.
    Returns an `al.Galaxy` with `bulge` + `arm_1`/`arm_2` SersicSph profiles.
    """
    cy, cx = centre
    return al.Galaxy(
        redshift=Z_SOURCE,
        bulge=al.lp.SersicSph(centre=(cy, cx), intensity=bulge_I,
                              effective_radius=bulge_Re, sersic_index=1.0),
        arm_1=al.lp.SersicSph(centre=(cy + arm_offset, cx + 0.6 * arm_offset),
                              intensity=arm_I, effective_radius=arm_Re,
                              sersic_index=1.0),
        arm_2=al.lp.SersicSph(centre=(cy - 0.9 * arm_offset, cx + 0.4 * arm_offset),
                              intensity=arm_I, effective_radius=arm_Re,
                              sersic_index=1.0),
    )


def build_clumpy_spiral_source(
    *,
    centre: tuple[float, float] = (0.0, 0.16),
    disk_Re: float = 0.13,
    disk_q: float = 0.45,
    disk_phi: float = 0.2,
    disk_I: float = 0.20,
    arm_len: float = 0.40,
    arm_curl: float = 2.6,
    arm_phi0: float = 0.0,
    n_knot: int = 5,
    knot_Re: float = 0.11,
    knot_I: float = 0.6,
) -> al.Galaxy:
    """A clumpy, star-forming 'Einstein-spiral' source à la AGEL0206 / DESJ0206.

    The REAL AGEL0206 source is not a smooth Sersic — it is a clumpy blue
    star-forming galaxy whose HII-region knots trace spiral arms. Lensed by a
    sub-isothermal (γ<2) deflector this gives the characteristic barred-spiral
    Einstein-ring IMAGE: a bright central (radial) arc + a long clumpy
    tangential arc wrapping around the lens. Reproduces that morphology with an
    elliptical disk + a log-spiral chain of compact knots straddling the
    radial caustic. Returns an `al.Galaxy` (disk + knot_0..knot_{n-1}).
    """
    cy, cx = centre
    e1, e2 = q_phi_to_ell_comps(q=disk_q, phi_rad=disk_phi)
    comps = {"disk": al.lp.Sersic(centre=(cy, cx), ell_comps=(e1, e2),
                                  intensity=disk_I, effective_radius=disk_Re,
                                  sersic_index=1.2)}
    for k in range(n_knot):
        t = k / (n_knot - 1)
        ang = arm_phi0 + arm_curl * t
        r = arm_len * t
        comps[f"knot_{k}"] = al.lp.SersicSph(
            centre=(cy + r * np.sin(ang), cx + r * np.cos(ang)),
            intensity=knot_I * (1.0 - 0.35 * t),
            effective_radius=knot_Re, sersic_index=1.0)
    return al.Galaxy(redshift=Z_SOURCE, **comps)


def spiral_source(beta_x, beta_y):
    """Spiral-like extended source brightness on a source-plane grid.

    Bulge + two off-axis blobs (arms); total extent ~0.7", straddles
    |beta_r| ~ 0.6. Lifted from the 15b notebook so the figure script and the
    notebook share one definition. For PyAutoLens tracers prefer
    `build_spiral_source_galaxy` (returns an al.Galaxy).
    """
    bulge = np.exp(-((beta_x) ** 2 + (beta_y) ** 2) / (2 * 0.10 ** 2))
    arm_n = 0.6 * np.exp(-((beta_x - 0.35) ** 2 + (beta_y - 0.20) ** 2) / (2 * 0.08 ** 2))
    arm_s = 0.6 * np.exp(-((beta_x + 0.30) ** 2 + (beta_y - 0.10) ** 2) / (2 * 0.08 ** 2))
    return bulge + arm_n + arm_s


def analytic_axisym_caustics(gamma: float, theta_E: float = 1.0) -> dict:
    """Analytic radial/tangential critical radii + caustics for a circular
    power law rho ∝ r^(-gamma), gamma < 2.

    Robust replacement for the numerical Hessian extractor, which misses the
    radial critical curve of SINGULAR power laws (found 2026-06-16: the
    contour finder returns no radial caustic for cusped PLs at any gamma,
    though it is real). For a circular PL with deflection
    alpha(theta) = theta_E^(gamma-1) theta^(2-gamma):
      - tangential crit: theta_t = theta_E  (alpha/theta = 1)
      - radial crit:     theta_r = theta_E (2-gamma)^(1/(gamma-1))   (dalpha/dtheta = 1)
      - radial caustic:  beta_r = |theta_r - alpha(theta_r)|
    Returns radii in arcsec; beta_r is the source-plane radial-caustic radius.
    Ellipticity widens these into curves but the radii set the scale.
    """
    if gamma >= 2.0:
        return {"theta_t": theta_E, "theta_r": 0.0, "beta_r": 0.0}
    theta_r = theta_E * (2.0 - gamma) ** (1.0 / (gamma - 1.0))
    alpha_r = theta_E ** (gamma - 1.0) * theta_r ** (2.0 - gamma)
    return {"theta_t": theta_E, "theta_r": theta_r, "beta_r": abs(theta_r - alpha_r)}


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
    """Tangential + radial critical curves & source-plane caustics (dict of (y, x)
    segment lists; drop-in for ``plot_critical_curves``). Backward-compatible signature.

    Now delegates to the NATIVE autolens ``LensCalc`` (dual-scale) — robust for
    singular power laws, where the old finite-difference Hessian inflated the central
    shear and SILENTLY DROPPED the radial caustic. Passing a tracer WITH a central
    PointMass correctly shows the extra central radial-caustic structure the BH adds.
    `npix` is accepted for backward compatibility but the native routine sets its own
    (finer) resolution: a large tangential grid (the elongated tangential curve is not
    clipped) + a small fine radial grid. Original FD impl retained as
    ``_critical_curves_caustics_fd``.
    """
    n = max(int(npix), 180)                 # keep ~old-FD grid size for tutorial speed
    fov_r = min(float(fov), 0.9)
    return caustics_critical_native(
        tracer, fov_tangential=float(fov), fov_radial=fov_r,
        pixel_scale_tangential=2.0 * float(fov) / n,
        pixel_scale_radial=2.0 * fov_r / n)


def _critical_curves_caustics_fd(tracer: al.Tracer, *, fov: float, npix: int = 200):
    """[LEGACY finite-difference] Tangential + radial critical curves & caustics via a
    numerical Hessian of the deflection field. SUPERSEDED by the native-LensCalc
    ``critical_curves_caustics`` above — the FD version silently drops the radial
    caustic for singular power laws (κ→∞ centre inflates the numerical shear). Kept
    for provenance / cross-checks.
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


def caustics_critical_native(mass_obj, *, fov_tangential: float = 4.0,
                             fov_radial: float = 0.9,
                             pixel_scale_tangential: float = 0.01,
                             pixel_scale_radial: float = 0.0045):
    """Tangential + radial critical curves and caustics via the NATIVE autolens
    ``LensCalc`` (autogalaxy.operate.lens_calc), returning the same dict format as
    ``critical_curves_caustics`` (tang_crit / rad_crit / tang_caustic / rad_caustic
    — lists of (y, x) arrays) so it is a drop-in for ``plot_critical_curves``.

    Why this exists: the hand-rolled ``critical_curves_caustics`` finite-differences
    the deflection field, which BREAKS for singular power laws — the κ→∞ centre
    inflates the numerical shear so the radial eigenvalue never crosses zero and the
    radial caustic is silently dropped (and a central PointMass throws ±1.5" spurious
    segments). The native routine computes the eigenvalues from the analytic Jacobian
    + marching squares (the same code autolens uses for its own caustic plots) and is
    robust. In autolens 2026.4 these methods were refactored OFF ``Galaxy``/``Tracer``
    into the standalone ``LensCalc`` class — hence ``tracer.tangential_caustic_list_from``
    no longer exists; use ``LensCalc.from_mass_obj`` / ``LensCalc.from_tracer``.

    DUAL-SCALE EXTRACTION (important): for a shallow slope + ellipticity the
    tangential critical curve is a large, elongated figure-8/dumbbell (it traces the
    image positions, reaching several arcsec), while the radial critical curve is a
    tiny central oval (~0.3"). A single uniform grid cannot resolve both — a grid
    large enough for the tangential curve under-resolves the radial one (it vanishes),
    and a small grid CLIPS the tangential curve (making it look tiny). So we extract
    the tangential curve/caustic on a large grid (``fov_tangential``, match the image
    panel) and the radial curve/caustic on a small fine grid (``fov_radial``).

    Pass the SMOOTH mass component (e.g. the PowerLaw) — a central PointMass adds
    spurious extra radial-caustic segments from its own divergence; the macro caustic
    topology (dumbbell tangential + radial oval) is set by the smooth profile.
    """
    from autogalaxy.operate.lens_calc import LensCalc
    lc = LensCalc.from_mass_obj(mass_obj)
    grid_t = al.Grid2D.uniform(shape_native=(400, 400), pixel_scales=2 * fov_tangential / 400)
    grid_r = al.Grid2D.uniform(shape_native=(400, 400), pixel_scales=2 * fov_radial / 400)

    def _conv(lst):
        out = []
        for c in (lst or []):
            a = np.asarray(c)
            if a.ndim == 2 and a.shape[0] > 1:
                out.append((a[:, 0].copy(), a[:, 1].copy()))   # (y, x)
        return out

    return {
        'tang_crit':    _conv(lc.tangential_critical_curve_list_from(grid=grid_t, pixel_scale=pixel_scale_tangential)),
        'rad_crit':     _conv(lc.radial_critical_curve_list_from(grid=grid_r, pixel_scale=pixel_scale_radial)),
        'tang_caustic': _conv(lc.tangential_caustic_list_from(grid=grid_t, pixel_scale=pixel_scale_tangential)),
        'rad_caustic':  _conv(lc.radial_caustic_list_from(grid=grid_r, pixel_scale=pixel_scale_radial)),
    }
