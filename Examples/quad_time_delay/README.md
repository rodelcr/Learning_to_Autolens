# Example (stub): Quad Lens with Time Delays

## Status

◯ **Planned** — no notebook yet.

## Problem

A **point-source quasar** at z_S ≳ 2 is quadruply imaged by a foreground galaxy. Because the source is variable, the **time delays** between the four images carry direct information about the time-delay distance D_Δt = (1 + z_L) D_L D_S / D_LS, which constrains H₀.

This example teaches the **point-source likelihood** (rather than the extended-source likelihood used everywhere else in the curriculum) and shows how time-delay observations break the mass-sheet degeneracy that cripples extended-source cosmography alone.

## Data source

- Simulate: `al.ps.PointSource` + `al.mp.PowerLaw+shear` lens; compute arrival-time surface, extract time delays between images.
- Real-world: any H0LiCOW/TDCOSMO target (HE0435-1223, RXJ1131-1231, etc.) has public delays + HST imaging.
- `autolens_workspace_latest/scripts/point/` has canonical point-source examples.

## Method hint

- **`al.AnalysisPoint`** instead of `AnalysisImaging`. Likelihood sums over image positions (χ²) + time delays (also χ²) with independent uncertainty estimates.
- **Mass model**: PowerLaw + shear is the TDCOSMO default (isothermal is too rigid for sub-percent H₀).
- Break the mass-sheet degeneracy explicitly by comparing recovered H₀ under fixed vs free external convergence κ_ext.

## Exercises

1. Recover H₀ using only image positions. How well-constrained is it?
2. Add time delays. Precision should improve by 3–5×.
3. Marginalise over κ_ext ∈ Uniform(−0.1, +0.1). Watch H₀ inflate — this *is* the mass-sheet degeneracy.
4. Add stellar-kinematics constraint (σ_v for the lens, from SDSS or Keck). Precision recovers.

## References

- Refsdal (1964), MNRAS 128, 307 — original time-delay H₀ proposal.
- H0LiCOW / TDCOSMO papers (Wong+20, Birrer+20).
- `autolens_workspace_latest/scripts/point/modeling.py`.

## To build this out

Will need a new `al.AnalysisPoint` wrapper in the `fit_example_*` driver pattern. The imaging-only slurm dataset-routing won't be sufficient.
