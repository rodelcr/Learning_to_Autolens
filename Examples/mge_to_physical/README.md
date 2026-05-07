# Example: From MGE to Physical Mass Models

> **v0.92 ships:** the methodology (3-search MGE → stars-only → stars+dark chain) and the v2 fits with secondary deflector + 2 sources added. v2 reduces χ²/N from 3.0 → 1.87 — improvement, but max\|res\| ≈ 9.7σ remains (SUSPECT by `/autolens-fit-diagnostics`).
> 
> **Research in progress (NOT in v0.92):** publication-grade f_DM extraction. The remaining residuals trace to PSF / noise-correlation issues outside the model. Use this example as a *Bayes-factor-with-caveats* demo, not as a definitive stars+DM measurement.


## Status

◐ **In progress — methodology shipped, but canonical fit fails the physical bar**. The three-search chain runs end-to-end and produces a clean Bayes-factor ladder (Search 2 → Search 3 = +288 log units favouring stars+DM), but **both Search 2 and Search 3 fail the `/autolens-fit-diagnostics` physical bar** (~10σ coherent ring residuals visible in `results/search_3_stars_dark/fit_subplot.png`). The fit is mis-specified relative to the truth: it omits (a) the secondary deflector at z=0.8 (truth `θ_E=0.11"`) and (b) the second source component (truth has TWO Sersic sources offset by 0.4″). The +288 Bayes factor is *real as a number* but, per `bayesian_model_comparison/`, **a Bayes factor between two misspecified models picks the less-wrong one without validating either**. See §"Caveats" below.

## What this example is for

The other Examples in this collection fit lenses with **`Isothermal` or `PowerLaw` mass profiles** — total-mass parameterisations that don't separate stars from dark matter. This example shows the **next step in scientific maturity**: decompose the mass into a *stellar* component (constrained by the lens light) and a *dark-matter* component (unconstrained by light, modelled as NFW), and then audit the result for physical reasonableness.

The chain is:

```
1. Fit lens LIGHT alone with an MGE basis      → get the stellar light shape
2. Tie the MGE shape to a stellar mass profile → fit M/L ratio + NFW DM jointly
3. Compute f_DM(<θ_E) and γ'_stellar           → physical-bar audit
```

This is the workflow used in publication-grade analyses (Auger+10 SLACS, Sonnenfeld+13 SL2S, Treu+10) and in TDCOSMO-class systems where the stars/DM split matters for cosmology. Module 09 introduces the *MGE basis*; **Module 11** ships the *physical-bar audit* — the 6-panel residual walk, the Pattern A–F failure catalogue, and the `f_DM(<θ_E)` extraction recipe applied to *this fit's* `search_3_stars_dark/` artifacts. This example puts MGE light and physical-bar audit together end-to-end.

## Mock data

Adapted from `lenstronomy_AGEL_modules/tutorials_DB_2025_09/`. See `mocks/PROVENANCE.md` for the full provenance and `truths.json` for the structured truth values.

Key mock geometry (lenstronomy `mock_1`):

| Quantity | Value |
|---|---|
| Cosmology | `FlatLambdaCDM(H0=70, Om0=0.30)` |
| Primary lens | EPL @ z=0.5, θ_E=1.65″, γ′=2.15, ε≈(-0.13,-0.07) |
| Secondary lens | EPL @ z=0.8, θ_E=0.11″ (small perturber) |
| Source | Two Sersics @ z=1.7 |
| Lens light | Sersic, R_e=1.9″, n=4.9 (cuspy cD-like) |
| Image | 110×110 px @ 0.05″/px, exp_time=500 s, bg_RMS=0.004 |

The cuspy `n=4.9` Sersic light is the key — it's the regime where MGE's flexible basis demonstrably beats a single-Sersic fit. The secondary lens at θ_E=0.11″ is small enough to be absorbed into shear at first pass; an exercise re-introduces it.

## Methodology — three searches

### Search 1 — Lens light only (MGE)
Mask the arcs, fit the lens light with an MGE basis (`al.model_util.mge_model_from(total_gaussians=30, gaussian_per_basis=2)`). Two basis groups so the MGE captures both the bulge core and the extended envelope. **Output**: tightly-constrained MGE shape (centre, ellipticity, σ-distribution of Gaussians).

### Search 2 — Stars-only mass (MGE-light-as-mass)
Take the MGE shape from Search 1 and convert each Gaussian to a `LightAndMassProfile` (`al.lmp.Gaussian`) via `take_attributes()`. The mass distribution is now *forced to follow the light*. The single free parameter is the **mass-to-light ratio** (one global scaling). Add `ExternalShear` to absorb large-scale structure and the secondary deflector. Source: `SersicCore`. **Output**: how well does mass-follows-light alone fit the data?

### Search 3 — Stars + dark matter
Same MGE-light-as-stellar-mass as Search 2, but **add an `al.mp.NFW` dark-matter halo** centred on the bulge with free virial mass and concentration. M/L is also free. The total deflection is now *stars + DM*. **Output**: f_DM(<θ_E), the M/L_stellar that the data prefer, and the Bayes factor of (stars+DM) vs (stars-only) — the canonical test for whether the data demand dark matter.

### Comparison + audit (in the same notebook)
Compute Δlog_Z(stars+DM vs stars-only). Read off f_DM(<θ_E) from the joint posterior. Verify against the truth file: the truth EPL has γ′=2.15, but is built as a *single profile* — when we decompose into stars+DM the recovered M/L_stellar should land in a plausible range (3–8 M☉/L☉ for an old elliptical at z=0.5), and f_DM(<θ_E) should be in the SLACS-typical range (0.4–0.7 within R_eff/2).

## Notebook

`01_mge_to_physical.ipynb` — runs all three searches inline with skip-guards, plus the audit. Mirrors the `mass_stellar_dark/chaining.py` pattern from `autolens_workspace_latest`.

## Running on Cannon

```bash
sbatch --export=ALL,EXAMPLE=mge_to_physical,FIT_EXTRA_ARGS=--part=all \
       Modules/10_Cluster_Computing/scripts/submit_cannon.slurm
```

Parts: `light` (Search 1, MGE light only), `stars_only` (Search 2, MGE mass-follows-light + shear + source), `stars_dark` (Search 3, + NFW halo), `all` (the chain). Wall time: ~2–3 h on 32 cores per part for stars_only/stars_dark; light is ~30 min.

## Exercises

1. **Compound deflector check**. Add the z=0.8 secondary back as an `extra_galaxies` perturber with photometric centre fixed at the truth (`(-0.05, 0.02)″`) and free Einstein radius (`Uniform(0, 0.5)`). Does the recovered f_DM change? The truth secondary has θ_E=0.11″, so the posterior on its θ_E should peak near 0.11.
2. **Two-source decomposition**. Replace the single SersicCore source with two — using truth positions as Gaussian-prior centres. Does the MGE+NFW mass model shift? Source-substructure and DM substructure are degenerate; this is the flip side of Vegetti+09.
3. **MGE vs Sersic light prior**. Replace the MGE bulge in Search 1 with a single `al.lp.Sersic`. Compute Δlog_Z. The Bayes factor should favour MGE for this cuspy-light mock by ~50–200 log units (the lens light *is* multi-Sersic-like in truth).
4. **Power-law equivalent**. Run a parallel PowerLaw fit (mass not decomposed) on the same dataset, recover γ′, and check it matches the truth value γ′=2.15 to within 1σ. Compare its log_Z against (stars+DM) — when does decomposition earn its extra parameters?
5. **Concentration prior sensitivity**. Re-fit Search 3 with `al.mp.NFWMCRLudlowSph` (concentration tied to mass via the Ludlow+16 mass–concentration relation) instead of free `c`. Does f_DM(<θ_E) change? This is the prior-information vs. data-driven question for cluster lensing.

## Caveats — what to fix before this is publication-grade

The canonical Cannon fit (Searches 1–3) recovers a clean Bayes-factor ladder, but the fit's residual map shows ~10σ coherent ring structure. Three concrete improvements would fix this:

1. **Add the second source.** Truth has two `SERSIC_ELLIPSE` sources at (-0.05, 0.02)″ and (0.3, 0.22)″ — both unmodelled in the canonical fit. Adding a second `SersicCore` source with a centre prior near (0.3, 0.22)″ should drop the residuals dramatically.
2. **Add the secondary deflector** at z=0.8 (truth `θ_E=0.11″`, centre (-0.05, 0.02)″) as a fixed-centre `extra_galaxies` perturber. Per Pattern E (`compound_lens` story), a small enough secondary often gets absorbed into shear, but `θ_E=0.11″` is *just at the boundary* — for this mock it likely needs explicit modelling.
3. **Tighten the lens-light mask** if the MGE light fit is leaking flux into the arcs. Search 1's `chi²/N=3.23` already suggests the MGE light alone is leaving ~3σ residuals before any source is fit.

Until these are addressed, the recovered f_DM and M/L from Search 3 should be treated as *methodology demonstration* rather than physical inference. The Bayes-factor ladder still holds as a teaching example for `bayesian_model_comparison/` (it's exactly what §5.4 — *model misspecification* — warns about).

## References

- **Cappellari (2002)** MNRAS 333, 400 — original MGE for galaxy kinematics. arXiv:astro-ph/0201430.
- **Auger et al. (2010)** ApJ 724, 511 — the SLACS γ′ ≈ 2.08 ± 0.03 finding for ellipticals; explicitly stars+DM decomposition.
- **Sonnenfeld et al. (2013)** ApJ 777, 98 — SL2S, the canonical stars+DM analysis methodology.
- **Treu & Koopmans (2002)** MNRAS 337, L6 — early dynamics+lensing decomposition.
- **Suyu et al. (2014)** ApJ 788, L35 — composite mass model with NFW for time-delay cosmography.
- `autolens_workspace_latest/scripts/imaging/features/advanced/mass_stellar_dark/chaining.py` — the canonical PyAutoLens 3-search chain we mirror here.
- `autolens_workspace_latest/scripts/imaging/features/multi_gaussian_expansion/source_science.py` — for the related "source-science" flavour (intrinsic R_e, total flux, magnification of the *source*).
- Module 09 (MGE & linear light profiles) — the basis-function mechanics.
- **Module 11 (physical mass models)** — the audit methodology. ✓ shipped — see `Modules/11_Physical_Mass_Models/11_physical_mass_models.ipynb` for the 6-panel walk, Pattern A–F catalogue, f_DM and γ′ extraction. This example's `search_3_stars_dark/` is the canonical fit Module 11 §5 audits.