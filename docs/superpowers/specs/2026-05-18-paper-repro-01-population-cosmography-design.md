# Paper-Repro Spec 01 — P1: Hierarchical Population Cosmography (Li+2023)

**Date:** 2026-05-18
**Reproduction target:** arxiv 2307.09271 — Li, Collett, Krawczyk, Enzi 2023
**Headline result:** w = −0.96 ± 0.46 on 161 existing galaxy-galaxy lenses; forecast σ(w) = 0.11 for 10⁴ systems.
**Depends on:** Spec 00 (shared infra — uses `catalogue_161.csv`, `crossval_framework.py`).

---

## 1. Context

Li+2023 builds a hierarchical Bayesian model that combines Einstein-radius (θ_E) and stellar velocity dispersion (σ_v) measurements across a galaxy-lens population, marginalising over per-system lens density profile slope γ_eff and stellar anisotropy β. Population-level hyperpriors on (μ_γ, σ_γ, μ_β, σ_β) constrain cosmology via the σ_v(θ_E, z_l, z_s, γ_eff, β, cosmology) prediction.

The maths is unambiguous; the implementation is a sampling-stack choice. The paper uses NumPyro / NUTS. Our v0.96 stack uses autofit / Nautilus. **Reproducing in BOTH validates the result is sampler-independent** and exercises both toolchains on a non-trivial multilevel inference.

## 2. Goals

- Reproduce **w = −0.96 ± 0.46** on the 161-lens sample, with autofit-Nautilus AND NumPyro-NUTS, agreeing at < 0.1σ on (w, Ωₘ).
- Establish a **reusable hierarchical-cosmography template** for future AGEL multi-target analyses.
- Validate Phase 3 `_jeans_sigma_v` solver in a population context (per-system sensitivity is weak — Δlog_L ~ 0.4 per system — so 161-lens aggregation is the test).

## 3. Non-goals

- The 10⁴-lens forecast — paper's secondary result, not blocking
- Redshift-evolution mitigation (paper §4 extension) — v0.98+
- Anisotropy β(r) — isotropic-only on this spec; future spec extends `_jeans_sigma_v` to β ≠ 0 (Mamon & Łokas 2005 kernel)

## 4. Architecture

```
private/2307_09271_li2023_cosmography_population/
├── code/
│   ├── per_system_likelihood.py        ← DONE (smoke-tested 2026-05-18)
│   ├── per_system_likelihood_jax.py    ← JAX/NumPyro port of the above
│   ├── population_model_autofit.py     ← autofit hierarchical + Nautilus
│   ├── population_model_numpyro.py     ← NumPyro hierarchical + NUTS
│   ├── run_sampler_cannon.py           ← Cannon submit (CPU Nautilus, GPU NUTS)
│   └── validation.py                   ← against published median ± σ
├── notebooks/
│   ├── 01_population_inference.ipynb   ← end-to-end walkthrough; both samplers
│   └── 02_sampler_crossval.ipynb       ← Nautilus vs NUTS overlay
└── results/
    ├── nautilus_chain.csv
    ├── numpyro_chain.csv
    └── crossval_plot.png
```

## 5. The hierarchical model

**Per-system likelihood** (already implemented in `per_system_likelihood.py`):
$$
\log L_i = -\frac{1}{2}\left(\frac{\sigma_v^\text{pred}(\theta_E^i, z_l^i, z_s^i, R_\text{eff}^i, n_\text{Sersic}^i, \gamma_\text{eff}^i, \beta^i, \Omega_m, w_0) - \sigma_v^\text{obs,i}}{\sigma_{\sigma_v}^i}\right)^2
$$

where σ_v^pred is computed by `_jeans_sigma_v.sigma_v_aperture_isotropic` (isotropic, β = 0 for this spec).

**Population layer:**
$$
\gamma_\text{eff}^i \sim \mathcal{N}(\mu_\gamma, \sigma_\gamma), \quad \beta^i = 0 \text{ (fixed for v0.97)}
$$

**Top-level priors:**
- $\Omega_m \sim$ Uniform(0.1, 0.5)
- $w_0 \sim$ Uniform(−2.0, −0.3)
- $\mu_\gamma \sim$ Uniform(1.8, 2.3)
- $\sigma_\gamma \sim$ HalfNormal(0.2)

**Marginalisation strategy.** Per-system γ_eff_i is marginalised analytically against the population prior (since the per-system likelihood is approximately gaussian in γ_eff after the Jeans transformation — Li+2023 verifies this is a good approximation). This collapses the model from 161 + 4 = 165 dimensions to 4 (population + cosmology).

## 6. Stack implementations

### 6.1 autofit / Nautilus

- `population_model_autofit.py` wraps the 4-d marginalised model as an `af.Analysis`
- Single AnalysisFactor wrapping all 161 systems (returns scalar log_L_total)
- Nautilus sampler with n_live=400 (4-d so converges quickly)
- Wall: ~24h on Cannon 32-core single node

### 6.2 NumPyro / NUTS

- `population_model_numpyro.py` defines the `model()` function with NumPyro primitives
- All 161 per-system likelihoods vectorised via `jax.vmap`
- NUTS with adapt_step_size, target_accept_prob=0.8
- Wall: ~6-12h on Cannon A100 GPU (warming up may dominate)

### 6.3 Cannon submission

- `run_sampler_cannon.py --stack=autofit --part=population` → submits with `EXAMPLE=p1_population_autofit` to `submit_cannon.slurm`
- `run_sampler_cannon.py --stack=numpyro --part=population` → submits to GPU partition via `herculens_cannon_runner.py` pattern from Spec 00

## 7. Data flow

```
data/lens_catalogs/catalogue_161.csv (Spec 00)
   ↓
per_system_likelihood.py / _jax.py
   ↓
population_model_{autofit,numpyro}.py
   ↓
results/{nautilus,numpyro}_chain.csv
   ↓
crossval_framework.py (Spec 00) → results/crossval_plot.png + agreement table
   ↓
notebooks/02_sampler_crossval.ipynb (auto-renders the above)
   ↓
validation.py → assertion: w ∈ (−0.96 − 0.46, −0.96 + 0.46) at 1σ
```

## 8. Error handling

- Per-system likelihood returns -inf if cosmology gives non-physical distances (Ωₘ ≤ 0 already implemented)
- NumPyro `model()` uses `numpyro.handlers.scale` to gracefully downweight systems with missing R_eff or n_sersic
- Cannon submission validates GPU partition reachable before submit; falls back to CPU NumPyro with a warning if A100 unavailable

## 9. Testing

- **per_system_likelihood**: smoke-test already passing (γ_eff and Ωₘ sensitivity confirmed)
- **per_system_likelihood_jax**: assert pixel-level agreement with the numpy version at <1e-6 on 100 random parameter draws
- **population_model_autofit**: 10-lens subset, n_live=50 — should complete in <5 min and recover (μ_γ, w) at 2σ of injected truth
- **population_model_numpyro**: same 10-lens subset; 500 NUTS samples; same agreement target
- **Validation against publication**: when full 161-lens run lands, posterior median (w, Ωₘ) must lie within 1σ of (−0.96, 0.30)

## 10. Pedagogical extensions

### 10.1 NEW Learning_to_Autolens Module 16 — Hierarchical Bayesian Cosmography

Mirrors the layout of Module 12 (TDCOSMO + MSD):
1. Why hierarchical — collapsing nuisance parameters; pooling information across systems
2. Per-system likelihood derivation: σ_v(θ_E, cosmology) from spherical Jeans + thin-lens distance ratio
3. Population layer: hyperpriors on (μ_γ, σ_γ); marginalisation analytic vs numerical
4. The cosmographic punchline: D_ds/D_s sensitivity to (Ωₘ, w₀); per-system info budget = O(0.1 nat)
5. Dual-sampler demo: when do Nautilus and NUTS agree, where do they diverge?
6. Hand-off to `Examples/hierarchical_population_cosmography/` for the practical recipe

References gr-lensing-intuition's distance-ratio section, cosmographic-degeneracy section. References Module 12's TDCOSMO setup for cross-cosmography context.

### 10.2 Learning_to_Lens module `09_Galaxy_Lensing_Applications` extension

A new section `09.5_Hierarchical_Cosmography.wl` (Mathematica) derives:
1. The σ_v ↔ θ_E ↔ cosmology mapping for an SIE + isotropic Jeans tracer
2. The Fisher-information per system for (w, Ωₘ) — sets the σ(w) ~ N^(−1/2) scaling
3. Symbolic check that the Jeans aperture average for an isothermal lens reduces to the Bolton+08 formula

### 10.3 Promotion to public repo

NEW `Examples/hierarchical_population_cosmography/` (depth-B → depth-C):
- mocks/: 30-lens synthetic sample (depth-B baseline)
- code/: refactor of `population_model_autofit.py` to public-style namespace
- notebooks/01_*.ipynb: pedagogical walkthrough
- results/: synthetic-sample posterior + real-data 161-lens posterior side by side

Status promotion criteria: both samplers strict-PASS on the 30-lens synthetic; real-data result within 1σ of Li+2023; both Modules 16 + LtL-09.5 ship.

## 11. Timeline

- Day 1-2: extend `catalogue_161.csv` with BELLS sample (Spec 00 deliverable)
- Day 2-3: `per_system_likelihood_jax.py` + pixel-level test against numpy
- Day 3-5: `population_model_autofit.py` + 10-lens smoke test → full 161-lens Cannon submit
- Day 4-6: `population_model_numpyro.py` + 10-lens smoke + full A100 submit
- Day 6-7: cross-validation notebook + Module 16 draft + LtL extension
- Total: ~1 week + Cannon wall time
