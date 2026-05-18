# Paper-Repro Spec 02 — P2: TSPL Jackpot Dark Subhalo (Ballard+2023)

**Date:** 2026-05-18
**Reproduction target:** arxiv 2309.04535 — Ballard, Enzi, Collett, Turner, Smith 2023
**Headline result:** 5.9σ detection of a perturber in SDSSJ0946+1006, ΛCDM-consistent (M_sub, c_sub); wandering BH presented as an alternative.
**Depends on:** Spec 00 (J0946 data, herculens_bridge, crossval_framework).

---

## 1. Context

SDSSJ0946+1006 is the canonical "Jackpot" lens (Gavazzi+2008). Initially a DSPL, the system was later shown to be a **triple-source-plane lens** (TSPL) — three sources at three different redshifts (Smith+2024). Minor+2021 reported a high-concentration dark perturber in disagreement with ΛCDM. Ballard+2023 re-models with the TSPL data using HST I+U-band + VLT-MUSE emission lines, recovering a perturber at 5.9σ but at ΛCDM-consistent concentration — and proposes a wandering supermassive black hole as an alternative.

**Why this paper is high-value for our curriculum:** TSPL extends DSPL methodology naturally; the subhalo Bayes-factor methodology generalizes the gravitational-imaging technique to any AGEL lens with imaging precision; the wandering-BH alternative is publication-grade pedagogical content for "always compare alternative model classes when claiming a detection."

**Cross-validation stake:** Multi-plane lensing is a strength of both stacks. Reproducing in BOTH PyAutoLens (Nautilus) and Herculens (NUTS on A100) tests whether the Bayes-factor methodology produces sampler-stable inferences — a question Ballard+2023 doesn't explicitly address.

**Data + code sources (audit 2026-05-18):**
- **No paper-specific code repo** — Ballard+2023 does not deposit code on github/zenodo. Reproduction is from scratch using our autolens + Herculens infrastructure.
- **Data Availability statement** (verbatim from MNRAS 528:4): *"Supporting research data are available on request from the corresponding author and from the HST and VLT archives."* — i.e., we download HST + MUSE directly from MAST + ESO; the posterior chains require an email to Ballard if a direct posterior comparison is desired.
- **Related follow-on**: Enzi+2025 ([MNRAS 540:1](https://academic.oup.com/mnras/article/540/1/247/8123410)) re-interprets the same J0946 subhalo as evidence for self-interacting dark matter (SIDM). Cite as a forward reference in our §11.1 Module 14 extension; out of scope for the primary reproduction.
- **TSPL geometry baseline**: Smith+2024 establishes the 3-source-plane redshifts ((z_s1, z_s2, z_s3) ≈ (0.609, 2.035, 5.975)) — that's the dataset our P2 fit consumes via Spec 00's `j0946_data_loader.py`.

## 2. Goals

- Reproduce 5.9σ perturber detection on J0946+1006 in BOTH stacks; (M_sub, c_sub) posteriors agree at <1σ
- Subhalo posterior is ΛCDM-consistent (low concentration) per Ballard+2023, not the high-c Minor+2021 result
- Wandering-BH alternative produces Bayes factor within the bounds reported in the paper
- Internal consistency with P3 (Spec 03): same J0946, both fits should converge on the same main-lens γ′, M_E, ellipticity at <2σ

## 3. Non-goals

- Reproducing the full Bayesian model-selection ladder of Ballard+2023 (~5 model variants) — focus on the 3 most informative
- Independent MUSE reduction — use ESO archive products
- New source-position detection — accept Ballard's source positions as input prior

## 4. Architecture

```
private/2309_04535_ballard2023_tspl_jackpot/
├── code/
│   ├── tspl_tracer_autolens.py         ← 3-plane al.Tracer setup
│   ├── tspl_tracer_herculens.py        ← Herculens multi_plane equivalent
│   ├── muse_position_likelihood.py     ← stack-agnostic Gaussian likelihood
│   ├── subhalo_fit_driver_autolens.py  ← Bayes-factor (with/without NFW perturber)
│   ├── subhalo_fit_driver_herculens.py ← same in Herculens
│   ├── wandering_bh_alternative.py     ← PointMass substitution; stack-agnostic config
│   └── validation.py
├── notebooks/
│   ├── 01_tspl_main_fit.ipynb          ← main lens + 3 sources; both stacks
│   ├── 02_subhalo_bayes_factor.ipynb   ← with/without perturber; both stacks
│   ├── 03_wandering_bh_alt.ipynb       ← PointMass alternative; both stacks
│   └── 04_crossval_summary.ipynb       ← all 6 fits, posterior agreement
└── results/
    ├── tspl_main_{autolens,herculens}/
    ├── tspl_subhalo_{autolens,herculens}/
    ├── tspl_wbh_{autolens,herculens}/
    └── crossval_table.md
```

## 5. The TSPL model

Three source planes at (z_s1, z_s2, z_s3) ≈ (0.609, 2.035, 5.975) per Smith+2024.

**Main lens**: SIE + ExternalShear at z_l = 0.222.
**Source 1**: Sersic at z_s1 (the original Gavazzi "first" source, brightest).
**Source 2**: SersicCore at z_s2 (the "second" source, fainter, redshift-confirmed by MUSE).
**Source 3**: compact / point-like at z_s3 (Smith+2024 discovery, very faint and high-z).

The lensing potential is recursive (Schneider+1992 §9, also see Learning_to_Lens module `04_Lens_Equation/`). For both stacks, the multi-plane Tracer iterates the lens equation backward through each plane.

**Perturber**: free position (within an arcsec of the main lens), free θ_E_sub, optional free concentration if using NFW.

**MUSE source-position likelihood**: for each MUSE-detected emission line peak, a Gaussian penalty on the model-source-plane position vs the MUSE-derived centroid (with covariance from the MUSE PSF).

## 6. Stack implementations

### 6.1 PyAutoLens

`al.Tracer(galaxies=[lens, src1, src2, src3], cosmology=...)` natively handles 3+ planes. Subhalo as `al.Galaxy(redshift=z_l, mass=al.mp.NFW)` with shared redshift but distinct centre. MUSE position likelihood as a custom `AnalysisMUSEPositions(af.Analysis)` subclass that joins via `af.FactorGraphModel` with the imaging analysis (same pattern as our Phase 3 AnalysisKinematics).

Sampler: Nautilus, n_live=400 (multi-plane is degenerate — needs many live points).

### 6.2 Herculens

`herculens.LensModel.multi_plane.MultiPlaneLensModel(['EPL', 'SHEAR', 'NFW'], lens_redshift_list=[z_l, z_l, z_l], source_redshift_list=[z_s1, z_s2, z_s3])` — supports the same 3-plane geometry. JAX vmap over the prior samples; NUTS sampler.

Translation handled by `herculens_bridge.py` (Spec 00 deliverable).

## 7. Cannon submission

Three Cannon jobs per stack:
1. `--part=tspl_main` — main fit, no perturber (baseline log_Z)
2. `--part=tspl_subhalo` — + NFW perturber (free position + mass + c)
3. `--part=tspl_wbh` — + PointMass at the perturber position (wandering BH alternative)

Per stack: 3 × ~24h × 32 cores (autolens) or 3 × ~12h × 1 A100 (Herculens with NUTS). Six fits total; ~6 Cannon-days budgeted.

## 8. Data flow

```
Spec 00 j0946_data_loader → tspl_tracer_{stack}.py → fit driver
                          → muse_position_likelihood.py → fit driver
   ↓
results/{stack}_{tspl_main, tspl_subhalo, tspl_wbh}/
   ↓
ΔlogZ between (main, subhalo) and (subhalo, wbh) per stack
   ↓
notebooks/02 (subhalo) + 03 (wbh)
   ↓
notebooks/04 → Spec 00 crossval_framework → autolens-vs-Herculens agreement table
```

## 9. Error handling

- Multi-plane Tracer can produce non-physical configurations under wide priors. Reuse v0.96 `_make_robust_analysis_imaging` + `_make_robust_analysis_point` patterns: catch + return -1e99.
- MUSE position likelihood: if a source emerges below the MUSE detection threshold in the model, return -1e9 (don't crash chain).
- Herculens NUTS occasional divergent chains: log + downweight, don't crash run.

## 10. Testing

- TSPL Tracer self-test (both stacks): render image at injected truth, assert chi²/N ≤ 1.5 on a noise-free realization
- Subhalo Bayes-factor sanity: inject a known subhalo, recover (M_sub, c_sub) within prior bounds; compare ΔlogZ to analytic Vegetti+2010 estimator
- Wandering-BH alternative: with sufficient injected M_BH, the alternative wins by a known ΔlogZ — verify
- Cross-validation: autolens and Herculens TSPL main posteriors must agree at <1σ on the main-lens (θ_E, γ′, ell)

## 11. Pedagogical extensions

### 11.1 Learning_to_Autolens Module 14 extension — TSPL Multi-Plane

Current Module 14 covers DSPL + the recursive multi-plane lens equation. Add §"TSPL: when β_jk distance-ratio cross-terms matter" with the J0946 system as the worked example. Discusses:
- Why three planes give 3 independent β_jk ratios (vs 1 for DSPL) — extra cosmographic lever-arms
- Multi-caustic topology (image multiplicity bound by Burke's theorem generalised to N planes)
- Gravitational imaging methodology (Vegetti+2010) generalised to TSPL

References gr-lensing-intuition's "image topology" + "Burke's theorem" + "cosmological distance relations".

### 11.2 Learning_to_Lens module `04_Lens_Equation` extension

Add `04.5_Multi_Plane_Recursive.wl` (Mathematica): derive the recursive lens equation, the β_jk distance ratio definitions, and the Jacobian structure for N source planes. Cross-link to module `08_Elliptical_Models_Caustics` for the multi-plane critical-curve topology.

### 11.3 Promotion to public repo

NEW `Examples/tspl_jackpot/` (depth C, real data):
- code/: refactored TSPL drivers
- mocks/: synthetic TSPL with injected perturber (depth-B baseline)
- notebooks/01_*.ipynb (TSPL main), 02_*.ipynb (subhalo Bayes factor), 03_*.ipynb (wandering BH alt)
- README.md: real-target context, Ballard+2023 + Smith+2024 + Minor+2021 references

Status: depth C ✓ when both stacks recover 5.9σ + Module 14 extension ships.

## 12. Timeline

- Day 1-2: Spec 00 deliverables in hand (J0946 data, herculens_bridge)
- Day 2-4: `tspl_tracer_*` + `muse_position_likelihood.py` + smoke tests with synthetic TSPL mock
- Day 4-5: subhalo + wandering BH drivers
- Day 5-7: Cannon submits (6 jobs, ~6 Cannon-days; can run in parallel waves)
- Day 8-9: cross-validation notebook + Module 14 extension + LtL extension
- Total: ~1.5 weeks + Cannon wall time
