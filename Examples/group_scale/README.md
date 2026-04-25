# Example: Group-Scale Lens

## Status

◐ **In progress** — mock generated, driver and notebook scaffolding committed, **but Cannon fits do not yet converge**. Three attempts (v1: n_live=200 wide priors, v2: satellite-light-only fix, v3: n_live=400 + tightened priors) all stalled in burn-in over 5+ hours each. The bgg_shear_only model appears fundamentally unable to fit data with 4 mass-perturbing galaxies at the same z — see `LEARNING_LOG.md` ("group_scale doesn't converge with reasonable priors") for the diagnostic and proposed next steps.

To resume: try (a) running only `--part=bgg_plus_satellites` and let the satellite einstein_radii collapse to zero per Pattern E if data supports it, or (b) a 2-stage SLaM-style chain that fits BGG-with-satellite-mask first then adds satellites with photometric priors.

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
