# Example (stub): Double Source-Plane Lens

## Status

◯ **Planned** — no notebook yet.

## Problem

A single lens galaxy at z_L deflects **two sources at different redshifts** — typically z_S1 ∼ 0.5–1.0 and z_S2 ∼ 1.5–3.0. The ratio of Einstein radii for the two source planes constrains the cosmological distance ratio β = (D_ds₁ / D_s₁) / (D_ds₂ / D_s₂), which is sensitive to dark-energy equation of state without time delays.

This is **not** a compound lens (see [`../compound_lens/`](../compound_lens/)) — there's only *one* deflector, but two source planes.

## Data source

Options for mocks:
- Simulate with `al.Tracer` by stacking two sources at different z in the galaxy list.
- Borrow from DSPL mocks at `/Users/rosador/Documents/AGEL/DSPL_training/` (real DSPL parameter templates exist there).
- The `autolens_workspace_latest/scripts/imaging/features/double_source_plane/` directory has canonical examples.

## Method hint

- Direct fit is natural: lens at z_L with Isothermal+shear, two independent source galaxies at z_S1 and z_S2. Priors: tight on lens centre/Einstein radius (from combined image positions), wider on the two source positions.
- The key parameter is the **ratio of Einstein radii** θ_E(z_S1) / θ_E(z_S2), which equals β. This quantity is directly cosmology-sensitive.

## Exercises

1. **Direct fit** recovering β with known cosmology; compare to truth.
2. **Sensitivity to z_S uncertainty**: marginalise over redshift priors; how much does β inflate?
3. **Compare β to Treu+10, Collett+14** published DSPL cosmology measurements.

## References

- Collett & Auger (2014), MNRAS 443, 969 — DSPL cosmography.
- Gavazzi et al. (2008), ApJ 677, 1046 — first DSPL (J0946+1006 / "Jackpot").
- Smith et al. (2021) on DESI-QSO DSPL observations.
- `autolens_workspace_latest/scripts/imaging/features/double_source_plane/modeling.py`.

## To build this out

1. Copy the template from `Examples/compound_lens/`: two notebooks + mocks subdir.
2. Add `fit_example_double_source_plane.py` to `Modules/10_Cluster_Computing/scripts/`.
3. Extend slurm routing.
4. Update the row in `Examples/README.md` status column to ✓.
