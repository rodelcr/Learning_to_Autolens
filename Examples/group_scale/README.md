# Example: Group-Scale Lens

> **v0.92 ships:** the `truth_anchored` Cannon result (job 9204948, 1h05m, χ²/N=1.025, max\|res\|=4.50σ) — clean PASS that validates the model space can fit this group system.
> 
> **Research in progress (NOT in v0.92):** the freely-fit, staged_satellites, and SLaM 3-stage variants all FAILED to converge (search-space exploration is the bottleneck). Documented as approaches #1, #3, #4 in the Status section + `Modules/10_Cluster_Computing/HANDOFF_2026_05_02.md` §1.


## Status

◐ **In progress** — three approaches investigated:

1. **Freely-fit (v1, v2, v3): FAIL.** Three attempts (LEARNING_LOG 2026-04-24) at varying `n_live` and prior tightness all stalled in burn-in for 5+ hours. The 30+ free-parameter joint landscape is too large for Nautilus to explore in useful wall time.
2. **Truth-anchored (`--part=truth_anchored`, 2026-04-29 job 9204948): PASS** in 1h 05m — log_Z=44699.80, χ²/N=1.025, max\|res\|=4.50σ, clean white-noise residuals. Confirms the model space *can* fit this system. The freely-fit problem is search-space exploration, not model representability.
3. **Iterative-mask staged satellites (`--part=staged_satellites`, 2026-04-29 mask=1.7" cancelled, 2026-04-30 mask=1.85" v2): cancelled at 18.5h** with f_live=1.0 still. The 17-param BGG+source Stage 1 didn't compress. The mask widening didn't fix the search-space stall.
4. **SLaM pipeline (`--part=slam` via `fit_example_group_scale_slam.py`, 2026-05-01): in flight.** Three-stage SLaM ported from `autolens_workspace_latest/scripts/group/slam.py`: source_lp_0 (light only, all MGE bulges) → source_lp_1 (mass + parametric source, light fixed) → mass_total (final PowerLaw refinement). MGE replaces Sersic for every galaxy (saves ~16 nonlinear params), and the staged decomposition fits one component at a time. This is the canonical group-scale SLaM approach.

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