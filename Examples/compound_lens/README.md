# Example: Compound Two-Deflector Lens

## Problem

A single background source at **z = 1.7** is lensed by **two deflectors** on the line of sight:

| Plane | z | Role |
|---|---|---|
| 1 | 0.5 | Primary lens galaxy (bulge + Isothermal mass) |
| 2 | 0.8 | Secondary lens galaxy (bulge + Isothermal mass) |
| 3 | 1.7 | Source galaxy (SersicCore bulge) |

This is **not** a double-source-plane lens (one lens, two sources at different z). It is a **compound** (a.k.a. "two-plane" or "serial") deflector: rays from the source first pass through the z=0.8 lens, then the z=0.5 lens, and finally arrive at the observer. `al.Tracer` handles the recursive ray-tracing automatically once you pass it galaxies at different redshifts.

## Data

Simulated mock from `tutorials_DB_2025_09/mocks/images/mock_1_image.fits`:

- `mocks/mock_1_image.fits` — image (~98 KB, 0.05″/pixel)
- `mocks/mock_1_noise.fits` — noise map
- `mocks/mock_psf.fits` — PSF (~5.6 KB)
- `mocks/mock_1_redshifts.txt` — `redshift_list: [0.5, 0.8, 1.7]`

Mask: circular 2.7″ radius.

## Notebooks

1. **`01_compound_direct_fit.ipynb`** — A single free-fit with Nautilus + curated priors. This is the "build it explicitly and hand Nautilus a good starting point" approach. Fastest to understand, highest-dimensional search. Mirrors the method in `20251125_Mocks_redo_autolens_2src.ipynb` cells 14–24, cleaned up and narrated.

2. **`02_compound_slam.ipynb`** — A two-track comparison:
    - **Track A — single-effective-deflector** — a pedagogical simplification where both lenses collapse into one effective Isothermal + ExternalShear. When this works, compound lensing reduces to a vanilla SLaM problem.
    - **Track B — staged two-deflector chain** — a SLaM-spirit pipeline written inline (since `slam_v2026` assumes one lens body): fit primary first, then freeze and add secondary, then joint refinement.

## Key references

- Schneider, Ehlers & Falco (1992), Ch. 9 — multi-plane lensing theory.
- `autolens_workspace_latest/scripts/guides/advanced/multi_plane.py` — `al.Tracer` multi-plane API and `traced_grid_2d_list_from()`.
- `autolens_workspace_latest/scripts/imaging/features/extra_galaxies/modeling.py` — fixed-centre multi-galaxy modelling pattern (lines 336–409).
- `autolens_workspace_latest/scripts/howtolens/chapter_3_search_chaining/tutorial_4_x2_lens_galaxies.py` — staged chain across two lens bodies.

## Relation to the Learning-to-Autolens curriculum

Prerequisites: **Modules 03, 04, 05** (free-fit mechanics, SLaM pipeline, pixelized sources). The compound-lens problem extends Mod 04's single-SIE fit into the multi-plane regime without introducing any new autolens machinery — it's an exercise in prior placement and model composition.

After working through these notebooks, you've seen what **Module 11**'s physical-bar audit looks like on a significantly more complex architecture.
