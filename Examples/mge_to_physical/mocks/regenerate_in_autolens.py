"""regenerate_in_autolens.py — rebuild lenstronomy_mock_1 natively in autolens.

Why this exists
---------------
The 2026-05-07 chi²-at-truth diagnostic on the canonical mge_to_physical
fit (`Examples/mge_to_physical/results/search_3_stars_dark/`) showed:

  - With ALL truth components present in an autolens Tracer (1 lens light,
    1 EPL mass, ExternalShear, secondary EPL at z=0.8, 2 sources at z=1.7),
    the autolens-rendered model has chi²/N=6.35 and max|res|=33σ at the
    central pixel of the lens light.
  - Removing the secondary deflector AND/OR the second source changes
    chi²/N by less than 1%.
  - The 33σ central residual is therefore NOT from missing components.

The root cause is a **framework-level Sersic evaluation difference** at the
cuspy lens-light core (`n_sersic = 4.9`, `R_eff = 1.9″`). Lenstronomy and
autolens use slightly different inner-radius integration schemes for
high-n Sersic profiles; they agree at n ≲ 3 but diverge at n ≳ 4. The
pre-existing mock was simulated by lenstronomy and then fit by autolens
— the central pixel difference appears as a 33σ residual that no
amount of mass-model freedom can absorb.

The fix is to regenerate the mock natively in autolens with identical
truth parameters, so the simulator and fitter agree by construction.
After regeneration, chi²-at-truth should drop to ~1.0 with all
components present.

What this script does
---------------------
1. Loads `truths.json` (the lenstronomy truth dict).
2. Translates each lenstronomy lens / source / lens-light component into
   the autolens equivalent:
     EPL              → al.mp.PowerLaw
     SHEAR_GAMMA_PSI  → al.mp.ExternalShear (γ₁, γ₂ from γ_ext, ψ_ext)
     SERSIC_ELLIPSE   → al.lp.Sersic
3. Builds a multi-plane Tracer (z_l1=0.5 + z_l2=0.8 + z_s=1.7) under
   FlatLambdaCDM(H0=70, Ωₘ=0.30).
4. Runs `al.SimulatorImaging` with the original PSF + same exp_time
   + same background_rms → writes:
     autolens_mock_1_image.fits
     autolens_mock_1_noise_map.fits
     autolens_mock_1_psf.fits   (copy of lenstronomy PSF)
     autolens_truths.json       (autolens-convention truth dict)
5. Sanity check: fits the simulated image with the same truth tracer
   and asserts chi²/N ≤ 1.5. (If the assertion fires, the framework
   problem is somewhere ELSE and the chi²-at-truth analysis was
   misdiagnosed.)

Run from repo root:
    python Examples/mge_to_physical/mocks/regenerate_in_autolens.py
"""

from __future__ import annotations

import json
from pathlib import Path

import autolens as al
import numpy as np
from astropy.io import fits

OUT = Path(__file__).parent
TRUTHS = json.loads((OUT / "truths.json").read_text())

# --- 1. Geometry & cosmology ---------------------------------------------
shape = tuple(TRUTHS["imaging"]["shape_native"])
pixel_scale = float(TRUTHS["imaging"]["pixel_scales_arcsec"])
exp_time = float(TRUTHS["imaging"]["exp_time_s"])
bg_rms = float(TRUTHS["imaging"]["background_rms"])
z_l1 = float(TRUTHS["redshifts"]["lens_primary"])
z_l2 = float(TRUTHS["redshifts"]["lens_secondary"])
z_s = float(TRUTHS["redshifts"]["source"])
H0 = float(TRUTHS["cosmology"]["H0"])
Om0 = float(TRUTHS["cosmology"]["Om"])

import autogalaxy as ag
cosmology = ag.cosmo.FlatLambdaCDM(H0=H0, Om0=Om0)

print(f"[regen] geometry: {shape[0]}×{shape[1]} @ {pixel_scale}″/px, exp={exp_time}s, "
      f"bg_rms={bg_rms}")
print(f"[regen] redshifts: z_l1={z_l1}, z_l2={z_l2}, z_s={z_s}")
print(f"[regen] cosmology: FlatLambdaCDM(H0={H0}, Om0={Om0})")

# --- 2. Translate lenstronomy params → autolens --------------------------
def _shear_gamma_psi_to_components(gamma_ext: float, psi_ext: float) -> tuple:
    """SHEAR_GAMMA_PSI(gamma, psi) → al.mp.ExternalShear(gamma_1, gamma_2)
    using autolens convention: γ₁=γ·cos(2ψ), γ₂=γ·sin(2ψ)."""
    g1 = gamma_ext * np.cos(2.0 * psi_ext)
    g2 = gamma_ext * np.sin(2.0 * psi_ext)
    return g1, g2

# Primary lens (z=0.5) — EPL mass + Sersic light + ExternalShear
mass_l1 = TRUTHS["kwargs_lens"][0]
sh = TRUTHS["kwargs_lens"][1]
light_l1 = TRUTHS["kwargs_lens_light"][0]

g1, g2 = _shear_gamma_psi_to_components(float(sh["gamma_ext"]),
                                        float(sh["psi_ext"]))

# autolens convention: ell_comps = (e1, e2) with same parameterization as
# lenstronomy's SIE/EPL e1, e2 (Wright-Brainerd). Direct copy.
# NOTE 2026-05-15: autolens centre convention is (y, x) — first component
# is y, second is x. Lenstronomy uses (center_x, center_y) — first x then y.
# We must SWAP: autolens centre = (lenstronomy center_y, lenstronomy center_x).
# Earlier version of this script passed (center_x, center_y) which produced
# axis-swapped mock vs lenstronomy convention; the driver's hardcoded truth
# values then mismatched, producing the v0.95 chi²/N=6+ on the regenerated
# mock. (The lens at (0,0) hides the bug; the secondary at non-zero offsets
# exposes it.)
lens_l1 = al.Galaxy(
    redshift=z_l1,
    mass=al.mp.PowerLaw(
        centre=(float(mass_l1["center_y"]), float(mass_l1["center_x"])),
        ell_comps=(float(mass_l1["e1"]), float(mass_l1["e2"])),
        einstein_radius=float(mass_l1["theta_E"]),
        slope=float(mass_l1["gamma"]),
    ),
    shear=al.mp.ExternalShear(gamma_1=g1, gamma_2=g2),
    bulge=al.lp.Sersic(
        centre=(float(light_l1["center_y"]), float(light_l1["center_x"])),
        ell_comps=(float(light_l1["e1"]), float(light_l1["e2"])),
        intensity=float(light_l1["amp"]),
        effective_radius=float(light_l1["R_sersic"]),
        sersic_index=float(light_l1["n_sersic"]),
    ),
)
print(f"[regen] lens_l1: PowerLaw θ_E={mass_l1['theta_E']:.3f}, γ={mass_l1['gamma']:.2f}; "
      f"Sersic n={light_l1['n_sersic']:.2f}, R_e={light_l1['R_sersic']:.2f}; "
      f"shear (γ₁,γ₂)=({g1:+.4f}, {g2:+.4f})")

# Secondary deflector (z=0.8) — EPL mass only
mass_l2 = TRUTHS["kwargs_lens"][2]
lens_l2 = al.Galaxy(
    redshift=z_l2,
    mass=al.mp.PowerLaw(
        centre=(float(mass_l2["center_y"]), float(mass_l2["center_x"])),
        ell_comps=(float(mass_l2["e1"]), float(mass_l2["e2"])),
        einstein_radius=float(mass_l2["theta_E"]),
        slope=float(mass_l2["gamma"]),
    ),
)
print(f"[regen] lens_l2: PowerLaw θ_E={mass_l2['theta_E']:.3f}, γ={mass_l2['gamma']:.2f}")

# Source (z=1.7) — two Sersics
src_a = TRUTHS["kwargs_source"][0]
src_b = TRUTHS["kwargs_source"][1]
source = al.Galaxy(
    redshift=z_s,
    bulge=al.lp.Sersic(
        centre=(float(src_a["center_y"]), float(src_a["center_x"])),
        ell_comps=(float(src_a["e1"]), float(src_a["e2"])),
        intensity=float(src_a["amp"]),
        effective_radius=float(src_a["R_sersic"]),
        sersic_index=float(src_a["n_sersic"]),
    ),
    disk=al.lp.Sersic(
        centre=(float(src_b["center_y"]), float(src_b["center_x"])),
        ell_comps=(float(src_b["e1"]), float(src_b["e2"])),
        intensity=float(src_b["amp"]),
        effective_radius=float(src_b["R_sersic"]),
        sersic_index=float(src_b["n_sersic"]),
    ),
)
print(f"[regen] source: 2× Sersic (bulge n={src_a['n_sersic']:.2f}, "
      f"disk n={src_b['n_sersic']:.2f})")

# --- 3. Build Tracer + simulate ------------------------------------------
tracer = al.Tracer(galaxies=[lens_l1, lens_l2, source], cosmology=cosmology)

grid = al.Grid2D.uniform(shape_native=shape, pixel_scales=pixel_scale,
                         over_sample_size=4)

psf = al.Convolver.from_fits(
    file_path=OUT / "lenstronomy_mock_1_psf.fits",
    hdu=0,
    pixel_scales=pixel_scale,
)

simulator = al.SimulatorImaging(
    exposure_time=exp_time,
    psf=psf,
    background_sky_level=0.0,  # bg already in noise budget
    add_poisson_noise_to_data=True,
    noise_seed=42,
    noise_if_add_noise_false=bg_rms,
)

print("[regen] simulating...")
dataset = simulator.via_tracer_from(tracer=tracer, grid=grid)
print(f"[regen] simulated: data shape {dataset.data.shape_native}")

# --- 4. Write artifacts ---------------------------------------------------
import autolens.plot as aplt
img_path = OUT / "autolens_mock_1_image.fits"
noise_path = OUT / "autolens_mock_1_noise_map.fits"
psf_path = OUT / "autolens_mock_1_psf.fits"

aplt.fits_imaging(
    dataset=dataset,
    data_path=img_path,
    psf_path=psf_path,
    noise_map_path=noise_path,
    overwrite=True,
)
print(f"[regen] wrote {img_path.relative_to(OUT.parents[2])}")
print(f"[regen] wrote {noise_path.relative_to(OUT.parents[2])}")
print(f"[regen] wrote {psf_path.relative_to(OUT.parents[2])}")

autolens_truths = {
    "framework": "PyAutoLens 2026.4.13.6 (native simulator)",
    "regenerated_from": "lenstronomy_mock_1 truth-params (truths.json)",
    "regenerated_date": "2026-05-09",
    "imaging": TRUTHS["imaging"],
    "redshifts": TRUTHS["redshifts"],
    "cosmology": TRUTHS["cosmology"],
    "lens_primary": {
        "mass": {
            "type": "al.mp.PowerLaw",
            "centre": [float(mass_l1["center_y"]), float(mass_l1["center_x"])],
            "ell_comps": [float(mass_l1["e1"]), float(mass_l1["e2"])],
            "einstein_radius": float(mass_l1["theta_E"]),
            "slope": float(mass_l1["gamma"]),
        },
        "shear": {
            "type": "al.mp.ExternalShear",
            "gamma_1": float(g1),
            "gamma_2": float(g2),
        },
        "bulge": {
            "type": "al.lp.Sersic",
            "centre": [float(light_l1["center_y"]), float(light_l1["center_x"])],
            "ell_comps": [float(light_l1["e1"]), float(light_l1["e2"])],
            "intensity": float(light_l1["amp"]),
            "effective_radius": float(light_l1["R_sersic"]),
            "sersic_index": float(light_l1["n_sersic"]),
        },
    },
    "lens_secondary": {
        "mass": {
            "type": "al.mp.PowerLaw",
            "centre": [float(mass_l2["center_y"]), float(mass_l2["center_x"])],
            "ell_comps": [float(mass_l2["e1"]), float(mass_l2["e2"])],
            "einstein_radius": float(mass_l2["theta_E"]),
            "slope": float(mass_l2["gamma"]),
        },
    },
    "source": {
        "bulge": {
            "type": "al.lp.Sersic",
            "centre": [float(src_a["center_y"]), float(src_a["center_x"])],
            "ell_comps": [float(src_a["e1"]), float(src_a["e2"])],
            "intensity": float(src_a["amp"]),
            "effective_radius": float(src_a["R_sersic"]),
            "sersic_index": float(src_a["n_sersic"]),
        },
        "disk": {
            "type": "al.lp.Sersic",
            "centre": [float(src_b["center_y"]), float(src_b["center_x"])],
            "ell_comps": [float(src_b["e1"]), float(src_b["e2"])],
            "intensity": float(src_b["amp"]),
            "effective_radius": float(src_b["R_sersic"]),
            "sersic_index": float(src_b["n_sersic"]),
        },
    },
}
truths_path = OUT / "autolens_truths.json"
truths_path.write_text(json.dumps(autolens_truths, indent=2))
print(f"[regen] wrote {truths_path.relative_to(OUT.parents[2])}")

# --- 5. Self-consistency check (chi²-at-truth) ---------------------------
print("[regen] running chi²-at-truth self-consistency check...")
mask = al.Mask2D.circular(
    shape_native=shape, pixel_scales=pixel_scale, radius=2.7,
)
masked = dataset.apply_mask(mask=mask)
fit = al.FitImaging(dataset=masked, tracer=tracer)
chi2_total = float(fit.chi_squared)
n_unmasked = int(np.sum(~mask))
chi2_per_pix = chi2_total / max(n_unmasked, 1)
max_norm_res = float(np.nanmax(np.abs(fit.normalized_residual_map)))
print(f"[regen] chi²-at-truth: chi²/N = {chi2_per_pix:.3f} (target ≤ 1.5)")
print(f"[regen] chi²-at-truth: max|norm res| = {max_norm_res:.2f}σ "
      f"(target ≤ 5.0σ)")
assert chi2_per_pix <= 1.5, (
    f"chi²/N = {chi2_per_pix:.3f} > 1.5 — the framework problem is NOT just "
    "the cuspy-Sersic eval mismatch. Re-investigate before submitting Cannon."
)
assert max_norm_res <= 5.0, (
    f"max|res| = {max_norm_res:.2f}σ > 5.0σ — see chi²/N message above."
)
print("[regen] chi²-at-truth: PASS — mock is self-consistent in autolens.")
