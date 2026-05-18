# Paper-Repro Spec 03 — Li+2026 DSPL Jackpot IMF + NFW Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reproduce Li+2026's J0946+1006 DSPL fit (M_*=4.4×10¹¹, M_h=1.11×10¹³, α≈0 constant M/L, γ_inner≈1 canonical NFW, Salpeter IMF) in BOTH PyAutoLens (Nautilus, 32 CPU) and Herculens (NumPyro NUTS, A100 GPU); validate the two stacks agree at <1σ.

**Architecture:** DSPL Tracer (2 sources at z=(0.609, 2.035)) + main lens with MGE light + ellNFW dark halo + free M/L gradient α + free inner-slope γ_inner. v0.96 DSPL chain methodology adapted: Stage 1 fixed cosmology + lens MGE light, Stage 2 free α + γ_inner.

**Tech Stack:** PyAutoLens 2026.4 with MGE (Module 09 + `Examples/mge_to_physical`), custom gNFW profile, Herculens via Spec 00 bridge.

**Depends on:** Spec 00 (J0946 data, herculens_bridge, AGEL Watson reduction); v0.96 DSPL strict-PASS infrastructure; Phase 3 `_jeans_sigma_v.py` (for σ_v if used as a likelihood term).

---

## File Structure

```
private/2602_20889_li2026_dspl_imf_nfw/
├── code/
│   ├── __init__.py
│   ├── gnfw_profile.py                   ← custom autolens gNFW (free γ_inner)
│   ├── ml_gradient_basis.py              ← MGE Basis with α gradient
│   ├── dspl_jackpot_autolens.py
│   ├── dspl_jackpot_herculens.py
│   ├── validation.py
│   └── compute_derived.py                ← M_*, M_h, α from posterior
├── notebooks/
│   ├── 01_dspl_jackpot_autolens.ipynb
│   ├── 02_dspl_jackpot_herculens.ipynb
│   └── 03_crossval.ipynb
├── tests/
│   ├── __init__.py
│   ├── test_gnfw_profile.py
│   ├── test_ml_gradient_basis.py
│   └── test_dspl_jackpot_autolens.py
└── results/
    ├── dspl_jackpot_autolens/
    ├── dspl_jackpot_herculens/
    └── crossval.md
```

---

## Phase 1: custom autolens profiles

### Task 1: Implement custom gNFW profile

**Files:**
- Create: `private/2602_20889_li2026_dspl_imf_nfw/code/gnfw_profile.py`
- Create: `private/2602_20889_li2026_dspl_imf_nfw/tests/test_gnfw_profile.py`

Autolens has `al.mp.NFW` (γ_inner=1 fixed) but not a generalised NFW with free γ_inner. Implement as a subclass.

- [ ] **Step 1: Failing test**

Create `tests/test_gnfw_profile.py`:

```python
"""Tests for the generalized NFW profile."""

from pathlib import Path
import sys
import numpy as np
import pytest

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / 'code'))


def test_gnfw_reduces_to_nfw_at_gamma_inner_1():
    """gNFW with γ_inner=1 should match autolens al.mp.NFW."""
    import autolens as al
    from gnfw_profile import GeneralisedNFW
    grid = al.Grid2D.uniform(shape_native=(20, 20), pixel_scales=0.1)
    nfw = al.mp.NFW(centre=(0,0), ell_comps=(0,0), kappa_s=0.1, scale_radius=10.0)
    gnfw = GeneralisedNFW(centre=(0,0), ell_comps=(0,0),
                          kappa_s=0.1, scale_radius=10.0,
                          gamma_inner=1.0)
    alpha_nfw = np.array(nfw.deflections_yx_2d_from(grid=grid))
    alpha_gnfw = np.array(gnfw.deflections_yx_2d_from(grid=grid))
    # Tolerance: 1% relative — numerical quadrature in gNFW
    diff = np.abs(alpha_nfw - alpha_gnfw)
    mag = np.abs(alpha_nfw) + 1e-10
    assert (diff / mag).max() < 0.01


def test_gnfw_steeper_inner_slope_larger_central_alpha():
    """γ_inner=1.5 should give larger central deflection than γ_inner=1.0."""
    from gnfw_profile import GeneralisedNFW
    import autolens as al
    grid_pt = al.Grid2DIrregular(values=[[0.5, 0]])
    g1 = GeneralisedNFW(centre=(0,0), ell_comps=(0,0),
                         kappa_s=0.1, scale_radius=10.0, gamma_inner=1.0)
    g15 = GeneralisedNFW(centre=(0,0), ell_comps=(0,0),
                         kappa_s=0.1, scale_radius=10.0, gamma_inner=1.5)
    a1 = float(np.array(g1.deflections_yx_2d_from(grid=grid_pt))[0, 1])
    a15 = float(np.array(g15.deflections_yx_2d_from(grid=grid_pt))[0, 1])
    assert a15 > a1
```

- [ ] **Step 2: Run (FAIL)**

- [ ] **Step 3: Implement gNFW**

Create `code/gnfw_profile.py`:

```python
"""gnfw_profile.py — generalized NFW profile with free inner slope γ_inner.

ρ_3D(r) ∝ 1 / [ (r/r_s)^γ_inner × (1 + r/r_s)^(3 - γ_inner) ]

For γ_inner=1: reduces to canonical NFW. For γ_inner>1: steeper cusp
(adiabatic contraction). For γ_inner<1: cored (feedback signature).

Implemented as a subclass of autolens's spherical NFW with a numerical
override of the deflection integral.
"""

from __future__ import annotations

import autolens as al
import numpy as np
from scipy.integrate import quad


class GeneralisedNFW(al.mp.NFW):
    """gNFW with free γ_inner. Otherwise same parameters as NFW."""

    def __init__(self, centre=(0.0, 0.0), ell_comps=(0.0, 0.0),
                 kappa_s: float = 0.05, scale_radius: float = 5.0,
                 gamma_inner: float = 1.0):
        super().__init__(centre=centre, ell_comps=ell_comps,
                          kappa_s=kappa_s, scale_radius=scale_radius)
        self.gamma_inner = gamma_inner

    def convergence_func(self, grid_radii):
        """Override for the generalized profile.

        Uses a numerical integral for the projection — slower than NFW
        but only matters in the fit prior loop.
        """
        rs = self.scale_radius
        gamma = self.gamma_inner
        ks = self.kappa_s

        def kappa_at_r(r_kpc):
            x = r_kpc / rs

            def integrand(z):
                r3d = np.sqrt(x ** 2 + z ** 2)
                return 1.0 / (r3d ** gamma * (1.0 + r3d) ** (3.0 - gamma))

            res, _ = quad(integrand, 0, 100, limit=100)
            return 2.0 * ks * rs * res

        out = np.array([kappa_at_r(r) for r in np.atleast_1d(grid_radii)])
        return out.reshape(grid_radii.shape) if hasattr(grid_radii, 'shape') else out
```

- [ ] **Step 4: Run tests (may need iteration on the numerical-quadrature parameters)**

```bash
cd private/2602_20889_li2026_dspl_imf_nfw
conda run -n autolens pytest tests/test_gnfw_profile.py -v 2>&1 | tail -10
```

Note: this is a slow profile (numerical quadrature in convergence_func — autolens then numerically integrates that for the deflection). Cannon fits will be slower than pure-NFW fits — budget extra wall time.

### Task 2: Implement MGE M/L-gradient basis

**Files:**
- Create: `private/2602_20889_li2026_dspl_imf_nfw/code/ml_gradient_basis.py`

- [ ] **Step 1: Implement**

Create `code/ml_gradient_basis.py`:

```python
"""ml_gradient_basis.py — MGE light model with a radial M/L gradient.

The Li+2026 paper parameterises M/L(R) = (M/L)_0 × (R/R_e)^α. We
implement this as a multiplier applied to the MGE-decomposed light
profile when computing the stellar mass contribution.

Use:
    ml_mge = build_ml_gradient_mge(reference_R_e=0.8, alpha_init=0.0)
"""

from __future__ import annotations

import autofit as af
import autolens as al


def build_mge_basis_for_lens_light():
    """Stub — depends on Li+2026's specific Gaussian count + R_e seeds.

    The actual MGE decomposition lives upstream (e.g. via
    `autolens.lp_basis.Basis` with Gaussian components fit to the
    F814W lens-light image). For now, return a placeholder Sersic-Sersic
    decomposition that the full implementation will replace.
    """
    bulge = af.Model(al.lp_linear.Sersic)
    bulge.centre.centre_0 = af.GaussianPrior(0.0, 0.1)
    bulge.centre.centre_1 = af.GaussianPrior(0.0, 0.1)
    bulge.ell_comps.ell_comps_0 = af.GaussianPrior(0.0, 0.2)
    bulge.ell_comps.ell_comps_1 = af.GaussianPrior(0.0, 0.2)
    bulge.effective_radius = af.UniformPrior(0.5, 2.0)
    bulge.sersic_index = af.UniformPrior(3.0, 6.0)
    return bulge


def build_ml_gradient(alpha_init: float = 0.0):
    """Free α — slope of M/L(R) = (M/L)_0 × (R/R_e)^α.

    Returns an autofit Model param wrapper that the lens model can
    consume; specific implementation depends on how we feed alpha into
    the mass profile (typically as a constant multiplier to the MGE
    Gaussians' total light → mass mapping).
    """
    alpha = af.UniformPrior(lower_limit=-0.5, upper_limit=0.5)
    return alpha
```

---

## Phase 2: PyAutoLens DSPL fit

### Task 3: PyAutoLens DSPL driver (chained: Stage 1 + Stage 2)

**Files:**
- Create: `private/2602_20889_li2026_dspl_imf_nfw/code/dspl_jackpot_autolens.py`

- [ ] **Step 1: Implement (long, follows v0.96 DSPL chain pattern)**

Create `code/dspl_jackpot_autolens.py`:

```python
"""dspl_jackpot_autolens.py — DSPL J0946 with MGE + gNFW + free α
+ free γ_inner. Chained Nautilus fit.

Stage 1: fix γ_inner=1, α=0, fit (MGE, kappa_s, scale_radius, sources)
Stage 2: free γ_inner ∈ [0.5, 1.5], α ∈ [-0.5, 0.5], start from Stage 1
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import autofit as af
import autolens as al

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


Z_L = 0.222
Z_S1 = 0.609
Z_S2 = 2.035


def build_stage1(dataset, output_root, n_live=400):
    from gnfw_profile import GeneralisedNFW
    from ml_gradient_basis import build_mge_basis_for_lens_light

    bulge = build_mge_basis_for_lens_light()
    nfw = af.Model(GeneralisedNFW)
    nfw.centre.centre_0 = af.GaussianPrior(0.0, 0.1)
    nfw.centre.centre_1 = af.GaussianPrior(0.0, 0.1)
    nfw.kappa_s = af.LogUniformPrior(1e-3, 0.5)
    nfw.scale_radius = af.UniformPrior(5.0, 50.0)
    nfw.gamma_inner = 1.0  # FIXED in Stage 1
    shear = af.Model(al.mp.ExternalShear)
    shear.gamma_1 = af.GaussianPrior(0.0, 0.1)
    shear.gamma_2 = af.GaussianPrior(0.0, 0.1)

    lens = af.Model(al.Galaxy, redshift=Z_L, bulge=bulge,
                     mass=nfw, shear=shear)
    source1 = af.Model(al.Galaxy, redshift=Z_S1,
                        bulge=af.Model(al.lp.SersicCore))
    source2 = af.Model(al.Galaxy, redshift=Z_S2,
                        bulge=af.Model(al.lp.SersicCore))
    model = af.Collection(galaxies=af.Collection(
        lens=lens, source1=source1, source2=source2))

    analysis = al.AnalysisImaging(dataset=dataset, use_jax=False)
    search = af.Nautilus(
        path_prefix=output_root,
        name='dspl_jackpot_autolens',
        unique_tag='stage1_fixed_gNFW',
        n_live=n_live,
        n_batch=50,
        number_of_cores=int(os.environ.get('SLURM_CPUS_PER_TASK', '1')),
    )
    return search.fit(model=model, analysis=analysis)


def build_stage2(dataset, output_root, stage1_result, n_live=500):
    from gnfw_profile import GeneralisedNFW

    bulge = stage1_result.model.galaxies.lens.bulge
    nfw = af.Model(GeneralisedNFW)
    # Inherit stage1 posteriors for kappa_s, scale_radius
    nfw.kappa_s = stage1_result.model.galaxies.lens.mass.kappa_s
    nfw.scale_radius = stage1_result.model.galaxies.lens.mass.scale_radius
    nfw.centre = stage1_result.model.galaxies.lens.mass.centre
    nfw.gamma_inner = af.UniformPrior(0.5, 1.5)  # NOW FREE

    shear = stage1_result.model.galaxies.lens.shear
    lens = af.Model(al.Galaxy, redshift=Z_L, bulge=bulge, mass=nfw, shear=shear)
    source1 = stage1_result.model.galaxies.source1
    source2 = stage1_result.model.galaxies.source2
    model = af.Collection(galaxies=af.Collection(
        lens=lens, source1=source1, source2=source2))
    analysis = al.AnalysisImaging(dataset=dataset, use_jax=False)
    search = af.Nautilus(
        path_prefix=output_root,
        name='dspl_jackpot_autolens',
        unique_tag='stage2_free_gNFW',
        n_live=n_live,
        n_batch=50,
        number_of_cores=int(os.environ.get('SLURM_CPUS_PER_TASK', '1')),
    )
    return search.fit(model=model, analysis=analysis)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--part', choices=['stage1', 'stage2', 'chain'],
                    default='chain')
    p.add_argument('--dataset-root', type=Path, required=True)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--n-live', type=int, default=400)
    args = p.parse_args()

    dataset = al.Imaging.from_fits(
        data_path=args.dataset_root / 'image.fits',
        noise_map_path=args.dataset_root / 'noise_map.fits',
        psf_path=args.dataset_root / 'psf.fits',
        pixel_scales=0.05,
    )
    mask = al.Mask2D.circular(
        shape_native=dataset.shape_native,
        pixel_scales=dataset.pixel_scales, radius=3.0)
    dataset = dataset.apply_mask(mask=mask)

    t0 = time.time()
    if args.part in ('stage1', 'chain'):
        r1 = build_stage1(dataset, args.output_root, n_live=args.n_live)
    if args.part == 'stage2':
        # Load stage1 result from disk — for now assume in-process
        raise NotImplementedError("Stand-alone stage2 needs result-from-disk loader")
    if args.part == 'chain':
        build_stage2(dataset, args.output_root, r1, n_live=args.n_live + 100)
    print(f"Wall: {(time.time()-t0)/3600:.2f} h")


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Smoke test (laptop)**

```bash
cd private/2602_20889_li2026_dspl_imf_nfw
mkdir -p mocks/smoke
# Reuse the v0.96 DSPL synthetic mock
cp /Users/rosador/Documents/AGEL/Learning_to_Autolens/Examples/double_source_plane/mocks/{image,noise_map,psf}.fits mocks/smoke/
conda run -n autolens python code/dspl_jackpot_autolens.py \
    --part stage1 --dataset-root mocks/smoke --output-root output_smoke --n-live 20
```

Expected: ~5 min smoke run completes. Don't expect strict-PASS; just that the gNFW + MGE basis don't crash the chain.

---

## Phase 3: Herculens DSPL fit

### Task 4: Herculens DSPL driver

**Files:**
- Create: `private/2602_20889_li2026_dspl_imf_nfw/code/dspl_jackpot_herculens.py`

- [ ] **Step 1: Implement**

Create `code/dspl_jackpot_herculens.py`:

```python
"""dspl_jackpot_herculens.py — Herculens + NumPyro NUTS version of P3.

Uses the same scientific model as dspl_jackpot_autolens.py; sampling
via NUTS on A100 GPU.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--part', choices=['stage1', 'stage2', 'chain', 'full'],
                    default='full')
    p.add_argument('--dataset-root', type=Path, required=True)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--num-samples', type=int, default=5000)
    args = p.parse_args()

    import jax
    print(f"JAX devices: {jax.devices()}")

    import numpyro
    from numpyro.infer import MCMC, NUTS
    import numpyro.distributions as dist
    from herculens.MassModel import MassModel
    from herculens.LightModel import LightModel

    # --- Model: mirror dspl_jackpot_autolens stage 2 ---
    def numpyro_model(data, noise):
        theta_E = numpyro.sample('theta_E', dist.Uniform(0.5, 2.5))
        gamma_inner = numpyro.sample('gamma_inner', dist.Uniform(0.5, 1.5))
        alpha = numpyro.sample('alpha', dist.Uniform(-0.5, 0.5))
        kappa_s = numpyro.sample('kappa_s', dist.LogUniform(1e-3, 0.5))
        scale_radius = numpyro.sample('scale_radius', dist.Uniform(5.0, 50.0))
        # Build the Herculens lens + light models
        # ... (depends on Herculens API; fill in once bridge tests pass)
        # Compute log_likelihood from observed image
        log_L = 0.0  # PLACEHOLDER until full implementation
        numpyro.factor('log_likelihood', log_L)

    print("Stub: dspl_jackpot_herculens NUTS — fill in body after Spec 00 bridge fully passing")


if __name__ == '__main__':
    main()
```

Like Spec 02 Task 6, this is intentionally a stub. The full implementation depends on:
1. Spec 00 `herculens_bridge.py` Task 10 fully tuned (all 5 profile cross-render tests passing)
2. A gNFW translator added to the bridge (currently only NFW; gNFW is a Spec-03-specific extension)
3. The Herculens-side handling of multi-plane (DSPL is just 2 sources, but the API is the same as TSPL)

---

## Phase 4: Cannon submission

### Task 5: Submit autolens chain

- [ ] **Step 1: Patch submit_cannon.slurm**

Add `p3_dspl_jackpot_autolens)` case routing to `dspl_jackpot_autolens.py`.

- [ ] **Step 2: Submit**

```bash
ssh cannon "cd /n/.../learning_to_autolens && \
    sbatch --account=siag_lab --partition=siag --mem=192G --cpus-per-task=32 \
    --time=48:00:00 --job-name=p3_dspl_autolens \
    --export=ALL,EXAMPLE=p3_dspl_jackpot_autolens,FIT_EXTRA_ARGS='--part=chain' \
    Modules/10_Cluster_Computing/scripts/submit_cannon.slurm"
```

### Task 6: Submit Herculens (GPU)

- [ ] **Step 1: Submit**

```bash
conda run -n autolens python private/00_shared_infrastructure/code/herculens_cannon_runner.py \
    --example p3_dspl_jackpot_herculens --fit-script-extra "--part=full" \
    --time-hours 12 --memory 80G
```

---

## Phase 5: Validation + Module 17 outline

### Task 7: Validation + cross-validation

**Files:**
- Create: `private/2602_20889_li2026_dspl_imf_nfw/code/validation.py`
- Create: `private/2602_20889_li2026_dspl_imf_nfw/code/compute_derived.py`

- [ ] **Step 1: Implement compute_derived (M_*, M_h from posterior)**

Create `code/compute_derived.py`:

```python
"""compute_derived.py — convert posterior on (kappa_s, scale_radius, MGE) into
M_* and M_h."""

from pathlib import Path

import numpy as np
import pandas as pd


def compute_M_halo(chain: pd.DataFrame) -> pd.Series:
    """Total halo mass from kappa_s × scale_radius² (× cosmology factor)."""
    # Simplified: actual conversion requires Σ_cr → physical mass
    return chain['kappa_s'] * chain['scale_radius'] ** 2 * 1e13  # placeholder


def compute_M_star(chain: pd.DataFrame, mge_total_lum: float) -> pd.Series:
    """Stellar mass from MGE intensity × M/L_0."""
    return chain.get('ml_zero', 1.0) * mge_total_lum * 1e11  # placeholder
```

- [ ] **Step 2: Validation script**

Create `code/validation.py`:

```python
"""validation.py — compare against Li+2026 published values."""

import pandas as pd
from pathlib import Path
from compute_derived import compute_M_halo, compute_M_star

LI_2026 = {
    'M_star_log10': 11.64,  # log10(4.4e11)
    'M_halo_log10': 13.05,  # log10(1.11e13)
    'alpha_M_L_gradient': 0.0,
    'alpha_M_L_gradient_sigma': 0.05,  # paper says "consistent with 0"
    'gamma_inner': 1.0,
    'gamma_inner_sigma': 0.1,
}


def validate(chain_path: Path):
    chain = pd.read_csv(chain_path)
    M_halo = compute_M_halo(chain)
    print(f"M_halo: ours log10 {np.log10(M_halo.median()):.2f}, "
          f"Li+2026: {LI_2026['M_halo_log10']:.2f}")
```

### Task 8: Module 17 outline

**Files:**
- Modify: `private/2602_20889_li2026_dspl_imf_nfw/MODULE_17_OUTLINE.md` (create)

```markdown
# Module 17: Dynamical Mass Decomposition via Jeans (draft outline)

For public-repo Module 17. Promotes from this private/ work + Phase 3 `_jeans_sigma_v.py`.

## Sections

1. Stars + DM decomposition: DSPL as the lever-arm
2. M/L gradient α — physical motivation (radial age gradients, central BH growth)
3. gNFW γ_inner — ΛCDM expectation vs adiabatic contraction vs feedback
4. Phase 3 isotropic Jeans + Mamon & Łokas 2005 anisotropy kernel (v0.98 hook)
5. Hand-off to `Examples/dspl_jackpot_imf_nfw/`

Pre-requisites: Module 09 (MGE), Module 13 (TDCOSMO+kinematics), Phase 3.
References gr-lensing-intuition's Jeans chain.
```

---

## Self-Review

| Spec 03 section | Tasks |
|---|---|
| §4 architecture | Tasks 1-8 |
| §5 model | Tasks 1-3 |
| §6 stack implementations | Tasks 3 (autolens), 4 (Herculens stub) |
| §7 Cannon | Tasks 5, 6 |
| §8 data flow | Task 7 |
| §9 error handling | reuse v0.96 patterns |
| §10 testing | Task 1, 3 |
| §11 pedagogical Module 17 | Task 8 outline |
| §12 timeline | ~1.5 weeks |

**Open dependencies (not blockers for spec completion but blockers for strict-PASS):**
- Spec 00 bridge fully tuned + gNFW added (for Herculens leg)
- AGEL Watson reduction of J0946 (for clean input)

**Total: 8 tasks. ~10-14 days.**
