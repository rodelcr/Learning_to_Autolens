# Example: Cluster-Scale Lens

> **2026-05-04 diagnostic finding:** the cluster_truth_v2 Cannon job (9882023, 8h TIMEOUT) was diagnosed locally by evaluating the FitImaging likelihood at *literal truth values* from `mock_truth.json`. Result: χ²/pixel = 30.5, max\|res\| = 79σ, log_L = −312,687. **The mock data and the stored truth are internally inconsistent** — the truth-anchored point itself does not reproduce the data. This is a mock-generation bug (likely coordinate convention or PSF normalization mismatch in `mocks/generate_mock.py`), not a fit-search-space problem. No further Cannon runs are warranted on this mock until the generator is fixed.
>
> **v0.93 plan for cluster_scale:** regenerate the mock (audit `generate_mock.py` for the consistency bug, ideally add a built-in "fit at truth → chi²/N near 1" sanity check), then re-run direct + truth-anchored variants. Until then, this example stays research-in-progress.

## Status

◐ **Scaffolded.** Synthetic mock + minimal direct-fit notebook in place; no Cannon results yet. The natural architectural extension of [`../group_scale/`](../group_scale/): more deflectors (this scaffold uses 10 cluster members, real-world clusters have 20–100), and **multiple lensed sources at different redshifts** that constrain the mass distribution at multiple radii.

- ✓ `mocks/generate_mock.py` — synthetic 1 BCG + 10 members + 2 sources at z=(1.5, 2.8)
- ✓ `mocks/mock_image.fits` + `mock_truth.json` — committed mock data
- ✓ `01_cluster_scale_fit.ipynb` — minimal direct-fit notebook with FJ scaling-relation API demonstrated
- ◯ Cannon driver `Modules/10_Cluster_Computing/scripts/fit_example_cluster_scale.py` — TODO
- ◯ `02_multi_source_cosmography.ipynb` — TODO (β_12 cosmography over 2-source ratio)
- ◯ `03_substructure_sensitivity.ipynb` — TODO (Vegetti+ 2010 / Despali+ 2018 methodology)

## Problem

A **galaxy cluster** lenses *multiple* background sources at different redshifts. The cluster's mass is decomposed into:
- A central **BCG (brightest cluster galaxy)** with its own light + mass distribution
- An ensemble of **member galaxies** at the cluster's redshift, fixed at their photometric positions
- An **extended dark-matter halo** (typically NFW or dPIE) that dominates the lensing at $r > 50~\text{kpc}$
- Optional **substructure halos** to fit specific perturbations

This is the regime where strong-lensing cosmography on its own (without time delays or kinematics) constrains cluster mass profiles to ~5% accuracy at the Einstein radius (Caminha+22, Bergamini+23).

## Why this example matters in the curriculum

| You've seen | Architectural step |
|---|---|
| Single-deflector single-source (Mod 03–09) | 1 mass + 1 source |
| Compound 2-deflector (compound_lens, compound_lens_zoo) | 2 masses at different z + 1 source |
| Double source-plane (double_source_plane) | 1 mass + 2 sources at different z |
| Group-scale (group_scale) | BGG + N satellites, **all at same z** |
| **Cluster-scale (this example)** | **BCG + N members + halo + multiple sources at different z** |

It's the architectural *union* of `group_scale` (multi-deflector at one z) and `double_source_plane` (multi-source at different z). All techniques you've seen apply — iterative masking on the BCG region first, photometric centroid anchoring for the member galaxies, $\beta_{12}$-style cosmological distance ratios for the multi-source constraints — but the parameter space is much higher dimensional (typically 50–200 free parameters).

## Data source options

1. **Real cluster data** from the **Frontier Fields** (HST programs 13495, 13496, 13386, 13389, 13504, 14037, 14041) — six clusters with deep ACS + WFC3 imaging and many spectroscopically-confirmed lensed sources. MACS J1149.5+2223, MACS J0416.1−2403, Abell 2744, etc.
2. **Real cluster data** from the **BUFFALO** survey (HST GO 15117) — extension of the Frontier Fields with wide-area outskirts.
3. **Simulated mock** with `al.Tracer` building a BCG (Sersic + NFW or Sersic + dPIE) plus ~20 member galaxies at the BCG redshift, plus 2–3 sources at different redshifts. Generation script would mirror `Examples/group_scale/mocks/generate_mock.py`.

For a tutorial we recommend **starting with a simulated mock**: real cluster data has ~10× higher per-pixel S/N than typical galaxy-scale mocks plus drizzle-correlated noise plus mosaic artifacts, all of which obscure the methodology.

## Method hint

The standard pipeline (per Caminha+22 / Bergamini+23):

1. **Photometric pre-processing**: identify BCG + member galaxies in the F814W (or equivalent) image. Centroids fixed throughout.
2. **Common parametric mass profile** for the cluster halo — typically `dPIE` (dual pseudo-isothermal elliptical) but `NFW` is also valid. Centred on the BCG with a tight prior on offset.
3. **Member galaxies as `extra_galaxies`** with masses scaled to luminosity via Faber-Jackson:
   $$\sigma_v \propto L^{1/4} \quad\Rightarrow\quad \theta_{E,gal} \propto L^{1/2}$$
   so each member has ONE free parameter (a global $\sigma_*$ amplitude) rather than N independent parameters. PyAutoLens supports this via `extra_galaxies` scaling relations.
4. **Each lensed source** as a separate `Galaxy(redshift=z_s_i, ...)` in the `Tracer` collection. PyAutoLens auto-detects multi-plane.
5. **Iterative masking** in three passes:
   - **Pass 1**: BCG + brightest source only, mask radius ~10–20″ around BCG. Constrain the central halo.
   - **Pass 2**: Add the member galaxies (with FJ scaling) and the second-brightest source. Larger mask.
   - **Pass 3**: Full ensemble + all sources. Joint refinement with Pass-2-as-prior.

## Exercises

1. **Faber-Jackson scaling validation**: fit each member galaxy independently in a small mask around it; compare the recovered $\theta_E$ vs the FJ-predicted value. Does the empirical relation hold for cluster members, or do you need cluster-environment corrections?
2. **Multi-source cosmology**: with 3+ sources at different redshifts, the cosmology-sensitive ratio $\beta_{ij}$ between source pairs is over-determined. Compare $\beta$ from each source pair; the spread is the systematic error budget.
3. **Substructure detection**: residuals after Pass 3 may show coherent structure at locations of expected dark-matter subhalos. Run a sensitivity-mapping pass (similar to `Examples/subhalo_sensitivity/`) over the cluster mask to identify whether the data require substructure.

## References

- **Caminha et al. 2022**, A&A 666, A48 — modern cluster lensing pipeline (CLASH-VLT)
- **Bergamini et al. 2023**, A&A 670, A60 — multi-source cosmography in MACS J0416
- **Limousin, Kneib, Natarajan 2005** — original cluster-scale parametric framework
- **Kneib & Natarajan 2011**, A&AR 19, 47 — review of cluster lens modeling
- `autolens_workspace_latest/scripts/group/` — closest existing workspace example
- `autolens_workspace_latest/scripts/imaging/features/extra_galaxies/` — scaling-relation API for member galaxies
- **Frontier Fields**: https://frontierfields.org/ — public data
- **Lenstool**, **glafic**, **lenstronomy multiplane** — alternative tools for cluster modeling worth comparison

## Notes

- This stub is intentionally large in scope. A first version could fit only the BCG + 5 brightest members + 1 lensed arc on a Frontier Fields cluster, deferring multi-source cosmography and substructure to extensions. That alone would teach 90% of the cluster-scale methodology.
- Computationally: ~100k masked pixels × ~80 free parameters × Nautilus n_live=400 will be a multi-day Cannon job per fit. Resource planning matters here.

## To build this out

Mirror the `Examples/group_scale/` structure:
1. `mocks/generate_mock.py` for a synthetic cluster (BCG + ~10 members + 3 sources at z = 1.0, 2.0, 3.5)
2. `01_cluster_scale_fit.ipynb` — single-search direct fit at moderate scale
3. `00_climb_to_cluster.ipynb` — staged climb with iterative masking
4. `Modules/10_Cluster_Computing/scripts/fit_example_cluster_scale.py` — driver
5. Cluster-runtime budget needed: estimate at 10–20× group_scale wall time
