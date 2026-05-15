# Example: Position Modeling in PyAutoLens

## Status

✓ **Shipped 2026-05-14** — standalone tutorial consolidating scattered position-modeling coverage. Executes <60s on laptop, no Cannon dependency.

## Purpose

Strong-lensing fits can use **position-level constraints** from the conjugate image positions in addition to (or instead of) the pixel-level imaging likelihood. Before this tutorial, the position-modeling API + pedagogy was scattered across:

- `Examples/compound_lens/01_compound_direct_fit.ipynb` §2 + §2.5 (API contract + sanity check)
- `Examples/quad_time_delay/` (the 3-rung H0 chain: pos-only / image-only / joint)
- Various `--part=positions_only` driver branches

This notebook brings it into one place — the canonical reference for new lens modelers learning when and how to add positions to a fit.

## Topics

1. **Why positions?** Complementarity between pixel and position constraints.
2. **The PyAutoLens API** — `al.Grid2DIrregular` → `al.PositionsLH` → threshold.
3. **Sanity check** — read the source-plane spread without running a fit.
4. **Threshold sensitivity** — empirical 4-point sweep landed in `Examples/compound_lens/results/pos_lh_sweep_*/`.
5. **Cosmography 3-rung chain** — Refsdal/H0LiCOW demonstration from `Examples/quad_time_delay/`.
6. **When NOT to use PositionsLH** — over-constraining, mismatched conjugacy, pixelization conflicts.

## Reuses landed results

This notebook runs no new fits. It consumes:

- `Examples/compound_lens/results/pos_lh_sweep_{t1,t0p3,t0p1,t0p01}/summary.json`
- `Examples/quad_time_delay/results/{phase_4_positions_only_v2, phase_3_h0_free_tight, joint_h0_free}/model_results.txt`
- `Examples/quad_time_delay/figures/h0_chain_overlay.png` (referenced)

All produced by the v0.95 PositionsLH research batch.

## When to read this

After Module 03 (first lens model). Before any compound / DSPL / TDCOSMO fit where you have compact-feature conjugate images available.

## References

- `Examples/compound_lens/01_compound_direct_fit.ipynb` §2 + §2.5 — extended API derivation
- `Examples/quad_time_delay/README.md` — 3-rung H0 chain
- `Modules/12_Time_Delay_Cosmography_MSD/` — point-source likelihoods in cosmography
