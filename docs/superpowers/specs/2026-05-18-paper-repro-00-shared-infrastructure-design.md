# Paper-Repro Spec 00 — Shared Infrastructure

**Date:** 2026-05-18
**Author:** Rodrigo Córdova Rosado
**Scope:** Data assets, code utilities, environments, and cross-stack tooling shared by Specs 01–04.

---

## 1. Context

Three Portsmouth-group papers (arxiv 2307.09271, 2309.04535, 2602.20889) become reproductions in `private/`, eventually promoted to public Modules and Examples. Two of the three (P2 + P3) share the same target (SDSSJ0946+1006, the "Jackpot" lens); all three benefit from a unified validation + dual-stack (PyAutoLens + Herculens) cross-validation framework. This spec defines the shared substrate.

The user has confirmed siag_lab fairshare grants A100 GPU access — Herculens NUTS sampling can run on Cannon GPU, enabling a clean autolens-CPU vs Herculens-GPU performance comparison.

## 2. Goals

- **One canonical J0946+1006 data product** consumed by P2 and P3 alike (no per-paper re-downloads, no version drift).
- **One canonical 161-lens cosmographic catalogue** consumed by P1.
- **One canonical autolens ↔ Herculens model-spec bridge** so that the same scientific model is implemented in both stacks without re-derivation.
- **Reusable cross-validation framework** producing 1D KL divergences, joint Bayes factors, and posterior-overlay plots between any two posteriors (autolens-vs-Herculens, or in-stack alternative-model comparisons).
- **Cluster-deployed Herculens** via `herculens312` conda env on Cannon + an A100 GPU submit pattern.

## 3. Non-goals

- Building actual scientific results — that's Specs 01–04.
- Re-implementing PyAutoLens or Herculens internals — bridge only.
- Bidirectional bridge (Herculens → autolens) — only autolens → Herculens initially; bidirectional is v2 if needed.

## 4. Architecture

```
private/00_shared_infrastructure/
├── data/
│   ├── j0946/                              ← P2 + P3
│   │   ├── hst/                            ← MAST DRC products (ACS F814W, WFC3 F336W/F438W)
│   │   ├── muse_cube.fits                  ← ESO archive (manual download)
│   │   ├── spec_z.json                     ← curated source redshifts
│   │   └── README.md                       ← provenance, download date, program IDs
│   └── lens_catalogs/
│       ├── catalogue_161.csv               ← P1 unified table
│       ├── slacs_auger10.csv               ← SLACS source
│       ├── sl2s_sonnenfeld13.csv           ← SL2S source
│       ├── bells_brownstein12.csv          ← BELLS source (PDF-table extraction)
│       └── README.md
├── code/
│   ├── validation_framework.py             ← posterior-vs-published
│   ├── crossval_framework.py               ← autolens-vs-Herculens
│   ├── herculens_bridge.py                 ← autolens → Herculens model spec
│   ├── herculens_cannon_runner.py          ← Cannon-side Herculens submit helper
│   └── j0946_data_loader.py                ← canonical loader for both stacks
├── docs/
│   ├── herculens_cannon_setup.md           ← runbook
│   ├── dual_stack_conventions.md           ← gotchas + sign conventions
│   └── data_provenance.md                  ← every fits-file's origin
└── environment-herculens.yml               ← conda env spec
```

## 5. Data flow

```
MAST       → download_hst.py    → data/j0946/hst/*.fits          → j0946_data_loader.py → P2, P3
ESO archive → manual            → data/j0946/muse_cube.fits      → j0946_data_loader.py → P2 MUSE-position likelihood
SDSS-BOSS  → Vizier             → data/lens_catalogs/*.csv       → catalogue_161.csv     → P1 per-system likelihood

P1 autofit posterior  ┐
P1 NumPyro posterior  ┤→ crossval_framework.py → KL, Bayes factor, overlay PNG
                       ┘
P2 autolens posterior ┐
P2 Herculens posterior┤→ crossval_framework.py → same outputs
                       ┘
P3 autolens posterior ┐
P3 Herculens posterior┤→ crossval_framework.py → same outputs
                       ┘
P2 main-lens posterior┐
P3 main-lens posterior┤→ validation/j0946_consistency.ipynb → 4-way agreement table
                       ┘
                       └→ docs/tool_development_report.md (synthesis)
```

## 6. Components

### 6.1 `j0946_data_loader.py`

Loads HST + MUSE + spec-z from `data/j0946/` and returns:
- `(image_array, noise_map, psf, wcs)` per HST band
- `(muse_cube, muse_wcs, muse_var)` for the source-position likelihood
- `redshifts_dict` (lens + 2-3 sources)

Same interface used by both autolens-side (`al.Imaging.from_fits`) and Herculens-side (Herculens's data loader) — i.e., this function returns Python primitives, not framework-specific objects.

### 6.2 `herculens_bridge.py`

Mass-profile and light-profile translators:
- `al.mp.PowerLaw(centre, ell_comps, einstein_radius, slope)` → Herculens `EPL`
- `al.mp.ExternalShear(gamma_1, gamma_2)` → Herculens `SHEAR`
- `al.mp.PointMass(centre, einstein_radius)` → Herculens `POINT_MASS`
- `al.mp.NFW(centre, ell_comps, kappa_s, scale_radius)` → Herculens `NFW_ELLIPSE`
- `al.lp.Sersic(...)` → Herculens `SERSIC_ELLIPSE`
- `al.lp_basis.Basis([Sersic, Sersic, ...])` (MGE) → Herculens `MULTI_GAUSSIAN` / equivalent
- Multi-plane tracer (`al.Tracer(galaxies, cosmology)`) → Herculens `multi_plane.MultiPlaneLensModel`

For each profile, the bridge resolves the sign-convention / parameter-name differences between PyAutoLens's (y, x) centre convention and Herculens's (x, y) convention.

### 6.3 `crossval_framework.py`

Given two posterior chains (column-equivalent CSV or h5), produces:
- Per-param 1D KL divergence
- 2D joint Bayes factor over the union of the two posteriors' support
- Side-by-side corner plot (matplotlib + corner.py)
- Markdown summary table for inclusion in notebooks

Decoupled from any one paper — used by all of P1, P2, P3.

### 6.4 `herculens_cannon_runner.py`

Wraps the existing `submit_cannon.slurm` mechanism for Herculens jobs:
- Selects siag_lab GPU partition + A100 (or A100-80GB if available)
- Activates `herculens312` env instead of `autolens312`
- Skips `submit_cannon.slurm`'s default `autolens312` activation
- Writes Nautilus-equivalent output structure (`output/<example>/<unique_tag>/<run_name>/chain/`)

### 6.5 Catalogue assembly

`data/lens_catalogs/catalogue_161.csv` produced by extending today's working Vizier query (85 SLACS-Auger10 + 56 SL2S-Sonnenfeld13 = 141) with:
- BELLS Brownstein+2012 Table 3 (≥20 lenses) via PDF text extraction or a manual hand-curated CSV
- Cross-match dedupe by RA/Dec
- Quality cuts: σ_v error < 25 km/s, R_eff > 0, defined z_l + z_s
- Target: 161 lenses with complete (id, z_l, z_s, θ_E_arcsec, σ_v_kms, σ_v_err_kms, R_eff_arcsec, n_sersic, source_catalog)

## 7. Error handling

- HST/MUSE downloads use `astroquery` with retry-on-network-error
- Herculens bridge raises `ValueError` with the failing profile name if a translation lookup misses
- Cross-validation framework safe-returns NaN entries (not crashes) when posterior overlap is empty
- Cannon job submission validates that `herculens312` env exists on Cannon before submit; helpful error if not

## 8. Testing

- `j0946_data_loader.py`: smoke test loads each FITS file, asserts shape + non-nan
- `herculens_bridge.py`: for each profile, render an image in BOTH stacks at identical parameters; assert pixel-level agreement at <0.1% (sign conventions correctly applied)
- `crossval_framework.py`: pass two identical chains; assert KL ≈ 0 and Bayes factor ≈ 1
- `catalogue_161.csv`: assert ≥161 rows, no duplicate IDs, all required columns present + non-null per the quality cuts

## 9. Cross-references (gr-lensing-intuition + Learning_to_Lens)

- Sign-convention reconciliation references gr-lensing-intuition's section on "PyAutoLens centre = (y, x) convention" and our `feedback_mock_driver_consistency.md` memory
- `dual_stack_conventions.md` cites Learning_to_Lens `04_Lens_Equation/` for the canonical β = θ − α derivation; Herculens's API differences are diffed against this baseline

## 10. Timeline

- Day 1: data dir creation + HST download finish (already in-flight) + MUSE manual download
- Day 1-2: `j0946_data_loader.py` + smoke tests
- Day 2: catalogue assembly extension (BELLS)
- Day 2-3: Herculens env install on laptop + Cannon
- Day 3-5: `herculens_bridge.py` per-profile translators + pixel-level tests
- Day 5: `crossval_framework.py` + tests
- Day 5-6: Cannon Herculens runbook (GPU partition probing, env activation, output-tree confirmation)
- Total: ~1 week of focused work; runs concurrently with Specs 01-04 once `j0946_data_loader.py` lands
