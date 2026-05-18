# Paper-Repro Spec 03 — P3: DSPL Jackpot Salpeter+NFW (Li+2026)

**Date:** 2026-05-18
**Reproduction target:** arxiv 2602.20889 — Li, Collett, Krawczyk, Granata, Enzi, Ballard, Lines, Sainz de Murieta, Weisenbach, Ryczanowski 2026
**Headline result:** SDSSJ0946+1006 DSPL fit → M_* = 4.4×10¹¹ M☉, M_h = 1.11×10¹³ M☉, **constant M/L preferred** (α ≈ 0), **Salpeter IMF normalisation**, **canonical NFW** (γ_inner ≈ 1).
**Depends on:** Spec 00 (J0946 data, herculens_bridge); shares lens with Spec 02.

---

## 1. Context

Li+2026 is part methodology paper, part scientific result. The methodology contribution is **fast JAX-based DSPL inference via Herculens + NumPyro NUTS** on an A100 GPU. The scientific contribution is the canonical-IMF + canonical-NFW recovery on a single (very high-quality) DSPL system.

Our v0.96 ships an autolens-native DSPL example with strict-PASS chi²/N=0.99 on a synthetic mock. P3 is the **real-data version + methodology cross-validation**: can our autolens stack match Herculens's published result on real J0946 data?

If **yes**, our autolens DSPL infrastructure is validated against an external groups' result. If **no**, the disagreement is itself a publication-grade methodology finding (which sampler is right? which model parameterisation? where is the bias?).

## 2. Goals

- Reproduce **M_* = 4.4×10¹¹, M_h = 1.11×10¹³, α ≈ 0, γ_inner ≈ 1, Salpeter IMF** in BOTH stacks
- Establish PyAutoLens MGE + ellNFW + free-α + free-γ_inner as a public-curriculum-ready model recipe (depth C)
- Cross-validate autolens-Nautilus vs Herculens-NUTS on identical model + identical data → posterior agreement at <1σ on every parameter
- Document any disagreement as a tool-development finding (Spec 04 deliverable)

## 3. Non-goals

- Re-deriving the Jeans aperture σ_v on real KCWI/LLAMAS IFU data — use Sonnenfeld 2012 published σ_v directly
- Discovering new sources / new substructure (that's Spec 02 territory)
- Reproducing all of the Li+2026 Bayesian model-comparison ladder (5+ variants) — focus on the 3 most informative

## 4. Architecture

```
private/2602_20889_li2026_dspl_imf_nfw/
├── code/
│   ├── dspl_jackpot_autolens.py        ← MGE + ellNFW + free α, γ_inner; Cannon Nautilus
│   ├── dspl_jackpot_herculens.py       ← Herculens NumPyro NUTS on A100
│   ├── ml_gradient_model.py            ← α-parameterised M/L(R) = M/L_0 (R/R_e)^α
│   ├── gnfw_profile.py                 ← generalised NFW (free γ_inner) — autolens-side custom
│   └── validation.py
├── notebooks/
│   ├── 01_dspl_jackpot_autolens.ipynb  ← autolens-native run + audit
│   ├── 02_dspl_jackpot_herculens.ipynb ← Herculens NUTS run + audit
│   └── 03_dspl_crossval.ipynb          ← side-by-side posterior overlay
└── results/
    ├── dspl_jackpot_autolens/
    ├── dspl_jackpot_herculens/
    └── crossval_table.md
```

## 5. The model

**Mass profile**: SIE → generalised NFW with free inner slope γ_inner (the Salpeter vs Chabrier distinction lives in the M/L ratio normalisation, which the paper treats as a derived parameter from the stellar mass posterior).

**Stellar component**: MGE-decomposed light × M/L(R) where M/L(R) = (M/L)_0 × (R / R_e)^α, with α as the gradient (paper finds α ≈ 0 → constant M/L).

**Dark matter**: ellNFW with free (M_h, c) AND free γ_inner. Canonical NFW has γ_inner = 1; Li+2026 finds γ_inner posterior consistent with 1.

**Sources**: SersicCore at z_s1 = 0.609 + SersicCore at z_s2 = 2.035 (DSPL — only two sources here; the third source from Smith+2024 is too faint for Li+2026's DSPL methodology and is in P2 territory).

**Cosmology**: Fixed FlatLambdaCDM(H0=70, Ωₘ=0.30) per Li+2026's analysis choice.

**Free parameters** (~15):
- SIE: centre (2), ell_comps (2), θ_E (1)
- ExternalShear: γ_1, γ_2 (2)
- gNFW: γ_inner (1), c (1), M_h_log (1)  — centre tied to SIE
- M/L: α (1), (M/L)_0 (1)
- 2× SersicCore source: 7 params each shared structure (~ 6 free per source after fixing core radius/gamma)

## 6. Stack implementations

### 6.1 PyAutoLens (autolens-native)

- `gnfw_profile.py`: custom `al.mp.gNFW` if not in autolens; otherwise use `al.mp.NFWSph` and inject γ_inner via subclass
- `ml_gradient_model.py`: custom `al.lp.Sersic`-derived MGE basis with α-scaled normalisation
- `dspl_jackpot_autolens.py`: chains a Stage 1 (fixed cosmology, free lens + source MGE) → Stage 2 (free α + γ_inner) following our v0.96 DSPL chain pattern
- Sampler: Nautilus, n_live=400-500 (15 free params)
- Cannon: 32 cores, ~48h wall

### 6.2 Herculens (faithful reproduction)

- `dspl_jackpot_herculens.py`: same model spec via `herculens_bridge.py` (Spec 00 deliverable)
- NumPyro NUTS with `numpyro.handlers` for the hierarchical structure
- Cannon: 1 × A100, ~12h wall
- Built primarily to **reproduce Li+2026 by construction** — they wrote Herculens. If our Herculens fit deviates from their published numbers, we have a configuration bug.

### 6.3 Cross-validation

- `03_dspl_crossval.ipynb`: load both posteriors, render side-by-side corner; compute 1D KL per param via Spec 00's `crossval_framework.py`
- Acceptance: max-per-param-KL < 0.05 (well below 1σ separation) OR documented disagreement with hypothesis

## 7. Data flow

```
Spec 00 j0946_data_loader → HST cutouts + spec-z
                          → MGE decomposition (one-time, on lens light)
                          → dspl_jackpot_{stack}.py
   ↓
results/dspl_jackpot_{autolens,herculens}/
   ↓
crossval_framework → results/crossval_table.md + side-by-side corner
   ↓
notebooks/03_dspl_crossval.ipynb
```

## 8. Error handling

- gNFW with γ_inner < 0.5 or > 1.5 may be numerically unstable in either stack — clip via prior bounds
- α gradient bounded to [-0.5, +0.5] to avoid unphysical M/L variations
- Both stacks use the v0.96 `_make_robust_analysis_imaging` pattern for failure-safe priors

## 9. Testing

- Render the truth model in both stacks at Li+2026's published medians, assert pixel-level agreement at <0.5% RMS (sign + parameter-convention sanity)
- Smoke fit: 100 Nautilus + 100 NUTS samples on the autolens-native v0.96 synthetic DSPL mock, both stacks; assert (θ_E, slope) recovered within prior bounds in both
- Production fits: Cannon strict-PASS bars (chi²/N ≤ 1.3, max\|res\| ≤ 4.1σ) + literature-match (M_*, M_h, α, γ_inner within 1σ of Li+2026)

## 10. Pedagogical extensions

### 10.1 NEW Learning_to_Autolens Module 17 — Dynamical Mass Decomposition via Jeans

Mirrors Module 11 (Physical Mass Models) layout but emphasises:
1. The stars + dark matter decomposition via DSPL (different distance ratios constrain different mass profiles)
2. M/L gradient α: physical motivation (radial age gradients, central BH growth, accreted stars in halos)
3. gNFW γ_inner: ΛCDM expectation (γ_inner = 1) vs adiabatic contraction (γ_inner > 1) vs feedback (γ_inner < 1)
4. The Phase 3 `_jeans_sigma_v.py` + anisotropic β(r) extension via Mamon & Łokas 2005 kernel (v0.98 hook; spec'd here, deferred implementation)
5. Hand-off to `Examples/dspl_jackpot_imf_nfw/` for the real-data application

References gr-lensing-intuition's Jeans chain. Pre-requisites: Module 09 (MGE), Module 13 (TDCOSMO + kinematics), Phase 3 `_jeans_sigma_v.py`.

### 10.2 NEW Learning_to_Lens module 11 — Stellar Dynamics + Jeans Equation

Mathematica derivation:
1. The Jeans equation for a spherical isotropic stellar system: $\frac{d}{dr}(\nu \sigma_r^2) + \frac{2\beta}{r}\nu\sigma_r^2 = -\nu \frac{d\Phi}{dr}$
2. Aperture-averaged σ_v projection (Mamon & Łokas 2005 kernel) symbolically derived for power-law M(<r)
3. Cross-check against the Bolton+08 formula for SIE-isothermal at R_aperture = R_eff
4. Tracer deprojection (LGM99) — show symbolically that the 3D Sersic density approximation reproduces the 2D Σ(R) to <2% within R_e for n ∈ [0.5, 8]
5. Generalisation to anisotropic β(r) — Mamon & Łokas 2005 kernel decomposition

### 10.3 Promotion to public repo

NEW `Examples/dspl_jackpot_imf_nfw/` (depth-C real data):
- code/: refactor of `dspl_jackpot_autolens.py` + `dspl_jackpot_herculens.py`
- notebooks/01_*.ipynb (autolens), 02_*.ipynb (Herculens), 03_*.ipynb (crossval)
- README.md: physical context, Li+2026 + Sonnenfeld 2012 + Gavazzi 2008 references; cross-link to `Examples/double_source_plane/` for the methodology baseline

Status criteria: both stacks recover Li+2026 numbers; crossval KL < 0.05/param; Modules 17 + LtL-11 ship.

## 11. Timeline

- Day 1-3: Spec 00 deliverables in hand (J0946 data, herculens_bridge)
- Day 3-5: `gnfw_profile.py` + `ml_gradient_model.py` (autolens custom subclasses)
- Day 5-7: `dspl_jackpot_autolens.py` + Cannon Nautilus submit
- Day 5-7: `dspl_jackpot_herculens.py` + Cannon A100 submit (parallel with previous)
- Day 8: cross-validation + Module 17 draft + LtL-11 draft
- Total: ~1.5 weeks + Cannon wall time
