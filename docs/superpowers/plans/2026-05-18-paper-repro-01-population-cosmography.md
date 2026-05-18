# Paper-Repro Spec 01 — Li+2023 Population Cosmography Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reproduce w = −0.96 ± 0.46 on the Chen+2019 161-lens sample using a hierarchical Bayesian model in BOTH autofit-Nautilus and NumPyro-NUTS; cross-validate the two samplers agree at <0.1σ on (w, Ωₘ).

**Architecture:** Per-system Gaussian likelihood on (σ_v_obs, σ_v_pred) given (γ_eff_i, Ωₘ, w₀); population layer marginalises γ_eff against hyperprior N(μ_γ, σ_γ). Two parallel implementations: `population_model_autofit.py` (Nautilus, single CPU core, ~24h) and `population_model_numpyro.py` (NUTS on A100 if available, ~6-12h). Cross-validation via Spec 00 `crossval_framework.py`.

**Tech Stack:** Phase 3 `_jeans_sigma_v.sigma_v_aperture_isotropic` (existing) + `astropy.cosmology.FlatwCDM`; autofit (Nautilus); NumPyro + JAX (NUTS).

**Depends on:** Spec 00 (Chen+2019 catalogue, crossval_framework).

---

## File Structure

```
private/2307_09271_li2023_cosmography_population/
├── code/
│   ├── __init__.py
│   ├── per_system_likelihood.py            ← already exists (smoke-tested 2026-05-18)
│   ├── per_system_likelihood_jax.py        ← NEW (JAX port for NumPyro)
│   ├── population_model_autofit.py         ← NEW
│   ├── population_model_numpyro.py         ← NEW
│   ├── run_sampler_cannon.py               ← NEW (Cannon submit helper)
│   └── validation.py                       ← NEW
├── notebooks/
│   ├── 01_population_inference.ipynb       ← NEW
│   └── 02_sampler_crossval.ipynb           ← NEW
├── tests/
│   ├── __init__.py
│   ├── test_per_system_likelihood_jax.py
│   ├── test_population_model_autofit.py
│   └── test_population_model_numpyro.py
└── results/
    ├── nautilus_chain.csv                  ← Cannon output
    ├── numpyro_chain.csv                   ← Cannon output
    └── crossval_plot.png                   ← post-fit
```

---

## Phase 1: per-system likelihood ports

### Task 1: JAX port of `per_system_likelihood`

**Files:**
- Create: `private/2307_09271_li2023_cosmography_population/code/per_system_likelihood_jax.py`
- Create: `private/2307_09271_li2023_cosmography_population/tests/test_per_system_likelihood_jax.py`

The numpy-based `per_system_likelihood.py` works but is too slow for NumPyro NUTS (which traces JAX). Port the Jeans solver + log_L to JAX.

- [ ] **Step 1: Write the failing test**

Create `tests/test_per_system_likelihood_jax.py`:

```python
"""Tests for the JAX port: per_system_likelihood_jax.

Verifies (a) JIT compiles without errors, (b) agrees with the numpy
version at < 1e-3 relative tolerance over a 100-point parameter sweep.
"""

from pathlib import Path
import numpy as np
import pytest

import sys
HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / 'code'))
from per_system_likelihood import log_L_system as log_L_np
jax_module = pytest.importorskip('per_system_likelihood_jax')
log_L_jax = jax_module.log_L_system_jax


def test_jax_matches_numpy_at_truth():
    """SLACS-typical lens at γ_eff = truth. JAX should match numpy < 1e-3."""
    kwargs = dict(
        theta_E_arcsec=1.2, sigma_v_obs=250.0, sigma_v_err=20.0,
        z_l=0.2, z_s=0.6, R_eff_arcsec=2.0, n_sersic=4.0,
        gamma_eff=2.0,
    )
    ll_np = log_L_np(**kwargs)
    ll_jax = float(log_L_jax(**kwargs))
    assert abs(ll_np - ll_jax) < 1e-3 * max(abs(ll_np), 1.0)


@pytest.mark.parametrize('gamma_eff', [1.8, 1.95, 2.05, 2.2])
def test_jax_matches_numpy_off_truth(gamma_eff):
    kwargs = dict(
        theta_E_arcsec=1.2, sigma_v_obs=250.0, sigma_v_err=20.0,
        z_l=0.2, z_s=0.6, R_eff_arcsec=2.0, n_sersic=4.0,
        gamma_eff=gamma_eff,
    )
    ll_np = log_L_np(**kwargs)
    ll_jax = float(log_L_jax(**kwargs))
    assert abs(ll_np - ll_jax) < 1e-3 * max(abs(ll_np), 1.0)


def test_jax_jits():
    """jax.jit on the function should compile without errors."""
    import jax
    f = jax.jit(log_L_jax, static_argnames=[
        'theta_E_arcsec', 'z_l', 'z_s', 'R_eff_arcsec', 'n_sersic'])
    out = f(theta_E_arcsec=1.2, sigma_v_obs=250.0, sigma_v_err=20.0,
            z_l=0.2, z_s=0.6, R_eff_arcsec=2.0, n_sersic=4.0,
            gamma_eff=2.0, Om0=0.3, w0=-1.0, H0=70.0)
    assert np.isfinite(out)
```

- [ ] **Step 2: Run test (FAIL: module missing)**

```bash
cd private/2307_09271_li2023_cosmography_population
conda run -n herculens312 pytest tests/test_per_system_likelihood_jax.py -v 2>&1 | tail -10
```

Expected: import error.

- [ ] **Step 3: Implement JAX port**

Create `code/per_system_likelihood_jax.py`:

```python
"""per_system_likelihood_jax.py — JAX port of per_system_likelihood.py.

Implements the same isotropic spherical Jeans σ_v solver as
`_jeans_sigma_v.sigma_v_aperture_isotropic` (Phase 3, 2026-05-15) but
in pure JAX so NumPyro NUTS can trace it.

Public surface:
    log_L_system_jax(...) → jax scalar
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln


# Physical constants (same as Phase 3 _jeans_sigma_v.py)
_G_KPC_KMS2_PER_MSUN = 4.30091e-6
_C_KMS = 2.99792458e5


@jax.jit
def _b_n_jax(n: float) -> float:
    return (2.0 * n - 1.0 / 3.0 + 4.0 / (405.0 * n)
            + 46.0 / (25515.0 * n ** 2)
            + 131.0 / (1148175.0 * n ** 3)
            - 2194697.0 / (30690717750.0 * n ** 4))


def _p_n_jax(n: float) -> float:
    return 1.0 - 0.6097 / n + 0.05463 / n ** 2


def _nu_sersic_3d_jax(r_over_Re, n):
    bn = _b_n_jax(n)
    pn = _p_n_jax(n)
    x = jnp.clip(r_over_Re, 1e-30, None)
    return x ** (-pn) * jnp.exp(-bn * x ** (1.0 / n))


def _distance_kpc_z1z2(z1, z2, Om0, w0, H0):
    """Angular diameter distance from z1 to z2 in kpc, FlatwCDM."""
    # Trapezoidal numerical integration of comoving distance
    n_z = 100
    zs = jnp.linspace(z1 + 1e-6, z2, n_z)
    # E(z) for FlatwCDM with constant w0
    E = jnp.sqrt(Om0 * (1.0 + zs) ** 3
                 + (1.0 - Om0) * (1.0 + zs) ** (3.0 * (1.0 + w0)))
    integrand = 1.0 / E
    # Hubble distance c/H0 in Mpc; convert to kpc via *1000
    d_H_mpc = _C_KMS / H0
    d_C_mpc = d_H_mpc * jnp.trapezoid(integrand, zs)
    # Angular diameter from z1 perspective: D_A = D_C / (1 + z2)
    return d_C_mpc * 1000.0 / (1.0 + z2)


def _sigma_v_aperture_isotropic_jax(
    theta_E_arcsec, slope, R_eff_arcsec, sersic_index,
    R_aperture_arcsec, D_l_kpc, D_s_kpc, D_ls_kpc, n_r=80,
):
    """JAX port of the Jeans aperture σ_v from Phase 3."""
    # Convert arcsec → kpc
    arcsec_to_kpc = jnp.pi / 180.0 / 3600.0
    theta_E_kpc = theta_E_arcsec * D_l_kpc * arcsec_to_kpc
    R_eff_kpc = R_eff_arcsec * D_l_kpc * arcsec_to_kpc
    R_ap_kpc = R_aperture_arcsec * D_l_kpc * arcsec_to_kpc

    Sigma_cr = (_C_KMS ** 2) / (4.0 * jnp.pi * _G_KPC_KMS2_PER_MSUN) \
        * D_s_kpc / (D_l_kpc * D_ls_kpc)
    M_einstein = jnp.pi * Sigma_cr * theta_E_kpc ** 2

    # 3D radial grid
    r = jnp.logspace(jnp.log10(0.01 * R_eff_kpc),
                     jnp.log10(50.0 * R_eff_kpc), n_r)
    nu = _nu_sersic_3d_jax(r / R_eff_kpc, sersic_index)
    M_enc = M_einstein * (jnp.clip(r, 1e-30, None) / theta_E_kpc) ** (3.0 - slope)

    integrand_jeans = nu * M_enc / r ** 2
    # Reverse cumulative trapezoidal (JAX)
    dr = jnp.diff(r)
    avg = 0.5 * (integrand_jeans[1:] + integrand_jeans[:-1])
    seg = avg * dr
    seg_rev = jnp.flip(seg)
    I_r_inner = jnp.cumsum(seg_rev)
    I_r = jnp.concatenate([jnp.flip(I_r_inner), jnp.array([0.0])])
    sigma_r_sq = _G_KPC_KMS2_PER_MSUN * I_r / jnp.clip(nu, 1e-300, None)

    # Aperture projection grid in R
    R_grid = jnp.linspace(0.001 * R_eff_kpc, R_ap_kpc, n_r)

    def project_one(R):
        mask = r > R
        rp = jnp.where(mask, r, R + 1e-10)
        denom = jnp.sqrt(jnp.clip(rp ** 2 - R ** 2, 1e-12, None))
        nu_mask = jnp.where(mask, nu, 0.0)
        sigma_sq_mask = jnp.where(mask, sigma_r_sq, 0.0)
        Sigma_R = 2.0 * jnp.trapezoid(nu_mask * rp / denom, rp)
        num_int = 2.0 * jnp.trapezoid(nu_mask * sigma_sq_mask * rp / denom, rp)
        sigma_los_sq = num_int / jnp.clip(Sigma_R, 1e-300, None)
        return Sigma_R, sigma_los_sq

    Sigma_R, sigma_los_sq = jax.vmap(project_one)(R_grid)
    weight = Sigma_R * R_grid
    sigma_ap_sq = (jnp.trapezoid(weight * sigma_los_sq, R_grid)
                   / jnp.clip(jnp.trapezoid(weight, R_grid), 1e-300, None))
    return jnp.sqrt(jnp.clip(sigma_ap_sq, 0.0, None))


def log_L_system_jax(
    theta_E_arcsec, sigma_v_obs, sigma_v_err,
    z_l, z_s, R_eff_arcsec, n_sersic,
    gamma_eff,
    Om0=0.3, w0=-1.0, H0=70.0,
):
    """Per-system Gaussian log-likelihood in JAX."""
    D_l = _distance_kpc_z1z2(0.0, z_l, Om0, w0, H0)
    D_s = _distance_kpc_z1z2(0.0, z_s, Om0, w0, H0)
    D_ls = _distance_kpc_z1z2(z_l, z_s, Om0, w0, H0)

    sigma_v_pred = _sigma_v_aperture_isotropic_jax(
        theta_E_arcsec=theta_E_arcsec, slope=gamma_eff,
        R_eff_arcsec=R_eff_arcsec, sersic_index=n_sersic,
        R_aperture_arcsec=R_eff_arcsec,
        D_l_kpc=D_l, D_s_kpc=D_s, D_ls_kpc=D_ls,
    )
    residual = (sigma_v_pred - sigma_v_obs) / sigma_v_err
    return -0.5 * residual ** 2
```

- [ ] **Step 4: Run tests — iterate on JAX numerical agreement**

```bash
conda run -n herculens312 pytest tests/test_per_system_likelihood_jax.py -v 2>&1 | tail -10
```

Expected: tests pass at <1e-3. If they fail, the cumtrapz / vmap projection has a JAX-numerical-stability quirk (likely at the grid boundaries); inspect with `print(jnp.where(jnp.isnan(...)))` and fix.

- [ ] **Step 5: No commit (private/)**

---

## Phase 2: Population model — autofit / Nautilus

### Task 2: Write `population_model_autofit.py`

**Files:**
- Create: `private/2307_09271_li2023_cosmography_population/code/population_model_autofit.py`
- Create: `private/2307_09271_li2023_cosmography_population/tests/test_population_model_autofit.py`

The autofit population model marginalises 161 per-system γ_eff_i analytically via population hyperprior N(μ_γ, σ_γ); free parameters are (Ωₘ, w₀, μ_γ, σ_γ) — only 4. Nautilus n_live=400, fast convergence.

- [ ] **Step 1: Write the failing test**

Create `tests/test_population_model_autofit.py`:

```python
"""Tests for the autofit population model."""

from pathlib import Path
import sys

import pandas as pd
import pytest

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / 'code'))


def test_model_has_4_free_params():
    """Top-level model: (Om0, w0, mu_gamma, sigma_gamma)."""
    from population_model_autofit import build_population_model
    model = build_population_model()
    assert model.prior_count == 4


def test_analysis_returns_finite_loglike_at_truth():
    """Run analysis on a 5-lens subset of Chen+2019; truth-anchored Om0=0.3, w0=-1."""
    from population_model_autofit import (
        build_population_model, PopulationAnalysis, load_catalog_subset)
    catalog = load_catalog_subset(n=5)
    analysis = PopulationAnalysis(catalog=catalog)
    model = build_population_model()
    # Build a truth-anchored instance
    instance = model.instance_from_unit_vector([0.5, 0.5, 0.5, 0.5])
    ll = analysis.log_likelihood_function(instance)
    assert ll > -1e6  # finite, not crashed
```

- [ ] **Step 2: Run (FAIL: module missing)**

```bash
conda run -n autolens pytest tests/test_population_model_autofit.py -v 2>&1 | tail -5
```

- [ ] **Step 3: Implement**

Create `code/population_model_autofit.py`:

```python
"""population_model_autofit.py — hierarchical Li+2023 in autofit Nautilus.

Marginalises per-system γ_eff_i against population hyperprior
N(μ_γ, σ_γ). Top-level free params: Ωₘ, w₀, μ_γ, σ_γ (4 dims).

Public surface:
    build_population_model() → af.Collection
    PopulationAnalysis(catalog) → af.Analysis
    load_catalog_subset(n=5) → pd.DataFrame
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import autofit as af
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SHARED = HERE.parents[2] / '00_shared_infrastructure'
sys.path.insert(0, str(SHARED / 'code'))


def load_catalog(path: Path | None = None) -> pd.DataFrame:
    """Load the enriched catalogue (Chen+2019 + Vizier structural)."""
    if path is None:
        path = SHARED / 'data' / 'lens_catalogs' / 'catalogue_161.csv'
    return pd.read_csv(path)


def load_catalog_subset(n: int = 5) -> pd.DataFrame:
    """First n rows — for smoke tests."""
    return load_catalog().head(n)


def build_population_model() -> af.Collection:
    """4-d top-level model: (Om0, w0, mu_gamma, sigma_gamma)."""
    Om0 = af.UniformPrior(lower_limit=0.10, upper_limit=0.50)
    w0 = af.UniformPrior(lower_limit=-2.0, upper_limit=-0.3)
    mu_gamma = af.UniformPrior(lower_limit=1.80, upper_limit=2.30)
    sigma_gamma = af.HalfNormalPrior(sigma=0.2) if hasattr(af, 'HalfNormalPrior') \
        else af.UniformPrior(lower_limit=0.01, upper_limit=0.5)
    return af.Collection(Om0=Om0, w0=w0, mu_gamma=mu_gamma, sigma_gamma=sigma_gamma)


class PopulationAnalysis(af.Analysis):
    """Hierarchical Bayesian population analysis.

    For each system, marginalises γ_eff_i against N(μ_γ, σ_γ) via 21-point
    Gauss-Hermite quadrature: sum_k w_k × exp(log_L_i(γ_eff_k)), where
    γ_eff_k = μ_γ + σ_γ × √2 × node_k.
    """

    # Gauss-Hermite 21-point nodes + weights (precomputed)
    _GH_NODES, _GH_WEIGHTS = np.polynomial.hermite.hermgauss(21)

    def __init__(self, catalog: pd.DataFrame):
        from per_system_likelihood import log_L_system
        self._log_L_per_system = log_L_system
        self.catalog = catalog
        super().__init__()

    def log_likelihood_function(self, instance) -> float:
        import autolens as al
        try:
            cosmo = al.cosmo.FlatLambdaCDM(H0=70.0, Om0=instance.Om0)
        except Exception:
            return -1e9

        log_L_total = 0.0
        for _, row in self.catalog.iterrows():
            if pd.isna(row.get('R_eff_arcsec')) or pd.isna(row.get('n_sersic')):
                # No structural data — skip (paper does this via QC cut)
                continue
            # Marginalise γ_eff via GH quadrature
            gamma_nodes = instance.mu_gamma + instance.sigma_gamma * np.sqrt(2) * self._GH_NODES
            log_Ls = []
            for g in gamma_nodes:
                ll = self._log_L_per_system(
                    theta_E_arcsec=row['theta_E_arcsec'],
                    sigma_v_obs=row['sigma_v_kms'],
                    sigma_v_err=row['sigma_v_err_kms'],
                    z_l=row['z_l'], z_s=row['z_s'],
                    R_eff_arcsec=row['R_eff_arcsec'],
                    n_sersic=row['n_sersic'],
                    gamma_eff=g,
                    cosmology=cosmo,
                )
                log_Ls.append(ll)
            log_Ls = np.array(log_Ls)
            # log-sum-exp + GH weight normalisation; constants in Z drop out
            max_ll = log_Ls.max()
            marg_L = (self._GH_WEIGHTS / np.sqrt(np.pi)) * np.exp(log_Ls - max_ll)
            log_L_marg = max_ll + np.log(marg_L.sum() + 1e-300)
            log_L_total += log_L_marg

        return float(log_L_total)
```

- [ ] **Step 4: Run tests**

```bash
conda run -n autolens pytest tests/test_population_model_autofit.py -v 2>&1 | tail -10
```

Expected: 2 pass.

- [ ] **Step 5: No commit (private/)**

### Task 3: Cannon submit driver for autofit

**Files:**
- Create: `private/2307_09271_li2023_cosmography_population/code/run_sampler_cannon.py`

- [ ] **Step 1: Implement (autofit Cannon submit)**

Create `code/run_sampler_cannon.py`:

```python
"""run_sampler_cannon.py — Cannon submit driver for the hierarchical
population fit. Two stacks supported: autofit (Nautilus) and NumPyro (NUTS).

Usage:
    python run_sampler_cannon.py --stack autofit --n-live 400 --output-root ./output
    python run_sampler_cannon.py --stack numpyro --num-samples 5000 --output-root ./output
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def run_autofit(args):
    import autofit as af
    from population_model_autofit import (
        build_population_model, PopulationAnalysis, load_catalog)
    catalog = load_catalog()
    model = build_population_model()
    analysis = PopulationAnalysis(catalog=catalog)
    search = af.Nautilus(
        path_prefix=args.output_root,
        name='p1_population_autofit',
        unique_tag='chen2019_161',
        n_live=args.n_live,
        n_batch=50,
        iterations_per_update=2000,
    )
    result = search.fit(model=model, analysis=analysis)
    print(result.info, flush=True)


def run_numpyro(args):
    """NumPyro NUTS sampler. Runs on A100 GPU if available."""
    import jax
    print(f"jax devices: {jax.devices()}", flush=True)
    import numpyro
    from numpyro.infer import MCMC, NUTS
    from population_model_numpyro import population_model_numpyro, load_catalog_arrays
    catalog_arrays = load_catalog_arrays()
    nuts = NUTS(population_model_numpyro, target_accept_prob=0.8)
    mcmc = MCMC(nuts, num_warmup=1000, num_samples=args.num_samples,
                num_chains=1, progress_bar=True)
    rng_key = jax.random.PRNGKey(0)
    mcmc.run(rng_key, **catalog_arrays)
    out = args.output_root / 'numpyro_chain.csv'
    out.parent.mkdir(exist_ok=True, parents=True)
    samples = mcmc.get_samples()
    import pandas as pd
    pd.DataFrame({k: v for k, v in samples.items()}).to_csv(out, index=False)
    print(f"Wrote {out}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stack', choices=['autofit', 'numpyro'], required=True)
    p.add_argument('--n-live', type=int, default=400)
    p.add_argument('--num-samples', type=int, default=5000)
    p.add_argument('--output-root', type=Path, required=True)
    args = p.parse_args()
    args.output_root.mkdir(exist_ok=True, parents=True)

    if args.stack == 'autofit':
        run_autofit(args)
    else:
        run_numpyro(args)


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Smoke**

```bash
conda run -n autolens python code/run_sampler_cannon.py --help 2>&1 | tail -5
```

Expected: argparse help.

---

## Phase 3: Population model — NumPyro NUTS

### Task 4: Write `population_model_numpyro.py`

**Files:**
- Create: `private/2307_09271_li2023_cosmography_population/code/population_model_numpyro.py`
- Create: `private/2307_09271_li2023_cosmography_population/tests/test_population_model_numpyro.py`

NumPyro version uses vmap over all 161 systems and explicit (γ_eff_i) latent variables — no marginalisation; NUTS samples them directly.

- [ ] **Step 1: Write failing test**

Create `tests/test_population_model_numpyro.py`:

```python
"""Tests for the NumPyro population model."""

from pathlib import Path
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / 'code'))

numpyro = pytest.importorskip('numpyro')


def test_model_runs_short_chain():
    """3-system, 50-sample smoke."""
    import jax
    from numpyro.infer import MCMC, NUTS
    from population_model_numpyro import population_model_numpyro, load_catalog_arrays
    arrays = load_catalog_arrays(n=3)
    nuts = NUTS(population_model_numpyro)
    mcmc = MCMC(nuts, num_warmup=10, num_samples=50, progress_bar=False)
    rng_key = jax.random.PRNGKey(0)
    mcmc.run(rng_key, **arrays)
    samples = mcmc.get_samples()
    assert 'Om0' in samples
    assert 'w0' in samples
    assert samples['Om0'].shape == (50,)
```

- [ ] **Step 2: Run (FAIL)**

```bash
conda run -n herculens312 pytest tests/test_population_model_numpyro.py -v 2>&1 | tail -10
```

- [ ] **Step 3: Implement**

Create `code/population_model_numpyro.py`:

```python
"""population_model_numpyro.py — hierarchical Li+2023 in NumPyro NUTS.

Per-system γ_eff_i is an explicit latent variable; NUTS samples
(Om0, w0, mu_gamma, sigma_gamma, gamma_eff_1, ..., gamma_eff_N)
jointly. Total dimension = 4 + N.

Public surface:
    population_model_numpyro(theta_E_obs, sigma_v_obs, sigma_v_err,
                              z_l, z_s, R_eff, n_sersic)
    load_catalog_arrays(n=None) → dict of jax arrays
"""

from __future__ import annotations

from pathlib import Path
import sys

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd

HERE = Path(__file__).resolve().parent
SHARED = HERE.parents[2] / '00_shared_infrastructure'
sys.path.insert(0, str(HERE))  # for per_system_likelihood_jax
from per_system_likelihood_jax import log_L_system_jax


def load_catalog_arrays(n: int | None = None) -> dict:
    """Load the enriched catalog as jax arrays."""
    path = SHARED / 'data' / 'lens_catalogs' / 'catalogue_161.csv'
    df = pd.read_csv(path)
    # Drop rows with missing structural data — paper does this
    df = df.dropna(subset=['R_eff_arcsec', 'n_sersic'])
    if n is not None:
        df = df.head(n)
    return {
        'theta_E_obs': jnp.array(df['theta_E_arcsec'].values),
        'sigma_v_obs': jnp.array(df['sigma_v_kms'].values),
        'sigma_v_err': jnp.array(df['sigma_v_err_kms'].values),
        'z_l': jnp.array(df['z_l'].values),
        'z_s': jnp.array(df['z_s'].values),
        'R_eff_arcsec': jnp.array(df['R_eff_arcsec'].values),
        'n_sersic': jnp.array(df['n_sersic'].values),
    }


def population_model_numpyro(
    theta_E_obs, sigma_v_obs, sigma_v_err, z_l, z_s, R_eff_arcsec, n_sersic,
):
    """NumPyro generative model for Li+2023 hierarchical population."""
    N = theta_E_obs.shape[0]

    # Top-level cosmology + hyperparameters
    Om0 = numpyro.sample('Om0', dist.Uniform(0.10, 0.50))
    w0 = numpyro.sample('w0', dist.Uniform(-2.0, -0.3))
    mu_gamma = numpyro.sample('mu_gamma', dist.Uniform(1.80, 2.30))
    sigma_gamma = numpyro.sample('sigma_gamma', dist.HalfNormal(0.2))

    # Per-system gamma_eff
    with numpyro.plate('lenses', N):
        gamma_eff = numpyro.sample('gamma_eff',
                                    dist.Normal(mu_gamma, sigma_gamma))

    # Vmap per-system log-likelihood
    import jax
    log_Ls = jax.vmap(
        lambda tE, sv, se, zl, zs, re, ns, g: log_L_system_jax(
            theta_E_arcsec=tE, sigma_v_obs=sv, sigma_v_err=se,
            z_l=zl, z_s=zs, R_eff_arcsec=re, n_sersic=ns,
            gamma_eff=g, Om0=Om0, w0=w0, H0=70.0,
        )
    )(theta_E_obs, sigma_v_obs, sigma_v_err, z_l, z_s,
       R_eff_arcsec, n_sersic, gamma_eff)

    numpyro.factor('log_likelihood', log_Ls.sum())
```

- [ ] **Step 4: Run tests**

```bash
conda run -n herculens312 pytest tests/test_population_model_numpyro.py -v 2>&1 | tail -10
```

Expected: 1 pass. May take 30s for JIT compile on first run.

---

## Phase 4: Cannon submission

### Task 5: Push code + submit autofit run

- [ ] **Step 1: Push** (Spec 00's `push_to_cannon.sh` already shipped)

```bash
bash Modules/10_Cluster_Computing/scripts/push_to_cannon.sh --go 2>&1 | tail -5
```

- [ ] **Step 2: Submit autofit job**

```bash
ssh cannon "cd /n/holystore01/LABS/hernquist_lab/Lab/rcordova/learning_to_autolens && \
    sbatch --account=siag_lab --partition=siag --mem=64G --cpus-per-task=8 --time=24:00:00 \
    --job-name=p1_autofit --export=ALL,EXAMPLE=p1_population,FIT_EXTRA_ARGS='--stack autofit --n-live 400' \
    Modules/10_Cluster_Computing/scripts/submit_cannon.slurm"
```

Note: `submit_cannon.slurm` will need to route `EXAMPLE=p1_population` to call our new `private/.../run_sampler_cannon.py`. For now, this submit will fail because the existing slurm script doesn't know about `p1_population`. Either:
  (a) Add a `p1_population)` case to `submit_cannon.slurm` (preferred, ~5 lines)
  (b) Run the script manually via ssh

Let me patch `submit_cannon.slurm`:

- [ ] **Step 3: Patch submit_cannon.slurm to know about Spec 01 + 02 + 03 + 04**

Modify `Modules/10_Cluster_Computing/scripts/submit_cannon.slurm` to add cases for:
- `p1_population` → runs `private/2307_.../code/run_sampler_cannon.py`
- `p2_tspl_jackpot_autolens|_herculens` → Spec 02
- `p3_dspl_jackpot_autolens|_herculens` → Spec 03

Since `private/` is gitignored, these scripts won't be on the Cannon push by default. Workaround: explicit rsync of `private/` separately.

```bash
rsync -av --exclude='data/' --exclude='results/' --exclude='_provenance/' \
    private/ cannon:/n/holystore01/LABS/hernquist_lab/Lab/rcordova/learning_to_autolens/private/
```

(`--exclude='data/'` because the 9.8 GB HST + future catalogs shouldn't be on Cannon; Cannon re-downloads as needed.)

### Task 6: Submit NumPyro run on GPU

- [ ] **Step 1: Submit GPU job via herculens_cannon_runner.py**

```bash
conda run -n autolens python private/00_shared_infrastructure/code/herculens_cannon_runner.py \
    --example p1_population \
    --fit-script-extra="--stack numpyro --num-samples 5000" \
    --time-hours 12 --memory 80G
```

Expected: submits a GPU job to siag_lab; prints job ID.

---

## Phase 5: validation + Module 16 outline

### Task 7: Write `validation.py`

**Files:**
- Create: `private/2307_09271_li2023_cosmography_population/code/validation.py`

- [ ] **Step 1: Implement**

Create `code/validation.py`:

```python
"""validation.py — compare our (Om0, w0) posteriors against Li+2023."""

from pathlib import Path
import pandas as pd
import numpy as np

LI_2023 = {
    'w0_median': -0.96,
    'w0_sigma': 0.46,
    'Om0_median': 0.30,
    'Om0_sigma_upper_bound': 0.50,  # paper not specific; their forecast prior was U(0.1,0.5)
}


def validate(chain_path: Path) -> dict:
    df = pd.read_csv(chain_path)
    w0_med = float(df['w0'].quantile(0.5))
    w0_sigma = float((df['w0'].quantile(0.84) - df['w0'].quantile(0.16)) / 2)
    Om0_med = float(df['Om0'].quantile(0.5))
    Om0_sigma = float((df['Om0'].quantile(0.84) - df['Om0'].quantile(0.16)) / 2)

    w0_dist_from_truth = abs(w0_med - LI_2023['w0_median']) / LI_2023['w0_sigma']
    print(f"w0: ours {w0_med:.3f}±{w0_sigma:.3f} vs Li+2023 "
          f"{LI_2023['w0_median']}±{LI_2023['w0_sigma']}")
    print(f"  → ours is {w0_dist_from_truth:.2f}σ from their median")
    print(f"Om0: ours {Om0_med:.3f}±{Om0_sigma:.3f} (Li+2023 fixed Om0 implicit)")

    return {
        'w0_pass': w0_dist_from_truth < 1.0,
        'Om0_pass': Om0_med > 0.15 and Om0_med < 0.45,
    }
```

### Task 8: Write `01_population_inference.ipynb` walkthrough

**Files:**
- Create: `private/2307_09271_li2023_cosmography_population/notebooks/01_population_inference.ipynb`

- [ ] **Step 1: Write notebook (8 cells: intro, data, model, autofit run, NumPyro run, posteriors, crossval, validation)**

Use the same notebook structure as `Examples/cosmography_joint_posterior/01_joint_posterior.ipynb`. Cells:
1. Header markdown — context, paper, headline
2. Code: load catalog
3. Code: instantiate autofit model, render its `info`
4. Code: load autofit chain (if exists, else graceful)
5. Code: load NumPyro chain (if exists, else graceful)
6. Code: crossval via Spec 00 framework
7. Code: validate against Li+2023
8. Markdown: discussion + bridge to Module 16

### Task 9: Module 16 outline

**Files:**
- Modify: `private/2307_09271_li2023_cosmography_population/MODULE_16_OUTLINE.md` (create)

- [ ] **Step 1: Write outline**

Create `MODULE_16_OUTLINE.md`:

```markdown
# Module 16: Hierarchical Bayesian Cosmography (draft outline)

For public-repo Module 16 — promotes from this private/ reproduction.

## Sections

1. Why hierarchical: collapsing nuisance params
2. Per-system likelihood from spherical Jeans + thin-lens (refs gr-lensing-intuition §"cosmological distance relations")
3. Population layer + GH-quadrature marginalisation
4. The cosmographic punchline: D_ds/D_s sensitivity to (Ωₘ, w₀); per-system info budget ~0.1 nat
5. Dual-sampler demo: Nautilus vs NUTS, when they agree, where they diverge
6. Hand-off to `Examples/hierarchical_population_cosmography/`

Refs: gr-lensing-intuition's distance-ratio + cosmographic-degeneracy sections.
```

---

## Self-Review

**Spec coverage** — all Spec 01 sections covered:

| Spec section | Implementing task(s) |
|---|---|
| §4 architecture | Task 1-4 |
| §5 hierarchical model | Task 2 (autofit), Task 4 (NumPyro) |
| §6 stack implementations | Task 2 (Nautilus), Task 4 (NUTS) |
| §7 Cannon submission | Task 5, 6 |
| §8 data flow | Task 8 (notebook) |
| §9 error handling | Task 2 catches Om0≤0; Task 4 inherits NumPyro graceful priors |
| §10 testing | Tasks 1, 2, 4 |
| §11 pedagogical Module 16 | Task 9 outline; full module ships in Spec 04 |
| §12 timeline | ~1 week |

**Placeholder scan:** no TBDs. Open dependency: Chen+2019 full catalogue (Spec 00 STUB), inherits as a known limitation.

**Type consistency:** all per-system likelihood inputs use the same keyword names (`theta_E_arcsec`, `sigma_v_obs`, etc.) in both numpy and JAX versions. Catalogue schema same in autofit + NumPyro loaders.

---

## Total

9 tasks. ~3-5 days laptop + Cannon overnight runs.
