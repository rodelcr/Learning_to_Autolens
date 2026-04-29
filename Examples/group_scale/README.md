# Example: Group-Scale Lens

## Status

◐ **In progress** — mock generated, driver and notebook scaffolding committed, **but freely-fit Cannon attempts do not converge**. Three attempts (v1: n_live=200 wide priors, v2: satellite-light-only fix, v3: n_live=400 + tightened priors) all stalled in burn-in over 5+ hours each — see `LEARNING_LOG.md` 2026-04-24 ("group_scale doesn't converge with reasonable priors") for the diagnostic.

**2026-04-29 — truth-anchored validation submitted.** A new `--part=truth_anchored` was added to `fit_example_group_scale.py` (Cannon job 9204948). All priors are tight Gaussians on `mock_truth.json` values: 4 lens galaxies + source. If this converges cleanly, the freely-fit failures are search-space-exploration issues (not model representability), and the resume path is a SLaM-style staged chain that walks the search there. If it stalls too, the problem is structural — the model architecture or PSF/exposure can't represent this group system.

Resume options after the truth_anchored result lands:
- (a) If truth_anchored PASS: build a 2-stage SLaM chain — fit BGG-with-satellite-mask first, then add satellites with photometric priors derived from stage 1.
- (b) If truth_anchored FAIL: investigate input PSF / noise model / extended-source assumption. May need a different mock generation.
- (c) Original fallback: run only `--part=bgg_plus_satellites` and let satellite einstein_radii collapse to zero per Pattern E if data supports it.

## Problem

A **brightest group galaxy (BGG)** lenses a background source, with multiple **satellite galaxies** and an **extended dark-matter envelope** also contributing to the deflection. All deflectors are at roughly the same redshift (unlike `compound_lens/`), so this is single-plane but multi-body — intermediate between galaxy-scale and cluster-scale lensing.

## Data source

Options:
- `autolens_workspace_latest/scripts/group/` has an existing canonical group-scale example at z=0.5 (all lenses same redshift). Fork the mock data from there.
- Real-world target: one of the Gavazzi+08 "SLACS-like but group-scale" lenses if we want to use real HST data.

## Method hint

- **BGG + satellites as `extra_galaxies`** with fixed centres (from imaging), fitted masses. See `autolens_workspace_latest/scripts/imaging/features/extra_galaxies/modeling.py` lines 336–409.
- **Common NFW halo** for the group's dark matter, centred on the BGG (or left free with a tight prior).
- **External shear** to absorb residual large-scale structure.

## Exercises

1. Compare a single-NFW model against a single-Isothermal total model. Is the BGG stellar mass recoverable?
2. Satellite-galaxy ablation: drop each satellite in turn, re-fit, measure the ΔlogZ. Which satellite(s) are actually needed for the fit?
3. Dark-matter fraction inside θ_E (Section 1 of Module 11).

## References

- Limousin et al. (2009), ApJ 696, 1771 — group-scale strong lensing fundamentals.
- `autolens_workspace_latest/scripts/group/start_here.py`.
- `autolens_workspace_latest/scripts/imaging/features/extra_galaxies/`.

## To build this out

Mirror `Examples/compound_lens/`: two notebooks, mocks, cluster driver, slurm routing.
