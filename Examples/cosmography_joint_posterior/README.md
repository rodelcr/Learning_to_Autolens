# cosmography_joint_posterior

**v0.96 ship.** The Birrer+2020 §4 combination — joint cosmographic posterior from two orthogonal strong-lensing probes:

1. **DSPL** (Examples/double_source_plane) — Ωₘ, w₀ via the β distance ratio
2. **TDCOSMO** (Examples/quad_time_delay) — H₀ via the time-delay distance D_Δt

## Status

- **Depth:** A (laptop-only notebook on existing chains; no Cannon dependency)
- **Notebook:** `01_joint_posterior.ipynb` (6 sections, <30 s execute)
- **Figures:** `figures/01_marginals.png`, `02_dspl_om0_w0.png`, `03_joint_om0_h0.png`

## v0.96 mock recovery

| Parameter | Truth | 1σ | 3σ | Verdict |
|---|---|---|---|---|
| Ωₘ | 0.30 | 0.30 ± 0.05 | (0.20, 0.40) | ✓ recovers truth |
| w₀ | −1.00 | −1.00 ± 0.05 | (−1.14, −0.89) | ✓ recovers truth |
| H₀ | 70.0 km/s/Mpc | 75.0 ± 2.6 | (67, 83) | ⚠ biased high (MSD effect) |

The H₀ bias is the **mass-sheet degeneracy** — TDCOSMO imaging+point-source alone cannot constrain λ_int. The v0.97 `joint_fit_h0_kin` rung adds Phase 3's `AnalysisKinematics` to break this and recover H₀ at truth.

## What this notebook is

- A clean laptop-side combination of two pre-computed Nautilus chains.
- Independence approximation for the Ωₘ-H₀ joint (the TDCOSMO chain was sampled at Ωₘ=0.30 fixed).
- Bridge documentation to the v0.97 fully-joint-fit implementation.

## What this notebook is NOT

- A FactorGraphModel joint-likelihood sampler (that's v0.97).
- An MSD-broken H₀ measurement (that requires kinematics → v0.97 `joint_fit_h0_kin`).
- A real-data cosmographic constraint (requires ≳ 5 strict-PASS DSPL systems + an MSD-broken TDCOSMO chain).

## v0.97 follow-ons

- `joint_fit_h0_kin` in `fit_example_quad_time_delay.py` — adds Phase 3 AnalysisKinematics. Should recover H₀ at 70 ± few.
- DSPL × TDCOSMO single-likelihood fit via FactorGraphModel — removes the independence approximation.

## References

- Birrer+2020 ("TDCOSMO IV: hierarchical analysis with kinematic constraints")
- Collett & Bacon 2014 (DSPL methodology)
- Wong+2020 (H0LiCOW, TDCOSMO baseline)
- Module 12 (TDCOSMO + MSD theory)
- Module 14 (multi-plane / DSPL theory)
