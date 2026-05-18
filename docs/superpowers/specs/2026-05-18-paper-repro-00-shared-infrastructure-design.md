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
│       ├── chen2019_table1.csv             ← canonical 161-lens (θ_E, σ_v, z_l, z_s)
│       ├── catalogue_161.csv               ← enriched with R_eff + n_sersic from auxiliary sources
│       ├── auger2010_slacs_re.csv          ← R_eff / n_sersic enrichment (SLACS)
│       ├── sonnenfeld2013_sl2s_re.csv      ← R_eff / n_sersic enrichment (SL2S)
│       ├── brownstein2012_bells_re.csv     ← R_eff / n_sersic enrichment (BELLS)
│       └── README.md                       ← cites Chen+2019 as primary source
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

`data/lens_catalogs/catalogue_161.csv` produced from **Chen et al. 2019** (MNRAS 488:3, [arxiv 1809.09845](https://arxiv.org/abs/1809.09845), DOI 10.1093/mnras/stz1902 — "Assessing the effect of lens mass model in cosmological application with updated galaxy-scale strong gravitational lensing sample"). This is the canonical 161-lens compilation cited by Li+2023's data availability statement.

Acquisition steps:
1. Download Chen+2019 Table 1 / supplementary material from MNRAS supplementary materials portal
2. If only the published PDF is available: extract the table via `pdftotext` or `tabula-py`
3. Schema: (id, z_l, z_s, θ_E_arcsec, σ_v_kms, σ_v_err_kms) — Chen+2019 has all six fields per system
4. R_eff_arcsec + n_sersic enrichment: cross-match with Auger+2010 (SLACS structural decomp), Sonnenfeld+2013 (SL2S), Brownstein+2012 (BELLS) via Vizier lookups on (RA, Dec)
5. Quality cuts: σ_v error < 25 km/s, R_eff > 0; document any rows dropped

**Today's existing `build_catalogue.py`** (141 lenses from Vizier SLACS-Auger10 + SL2S-Sonnenfeld13) should be **refactored**: keep as the structural-data source (R_eff, n_sersic enrichment) but use Chen+2019 as the primary (θ_E, σ_v) source — not the union of catalogs.

### 6.6 Data discovery — per-paper code/data source audit (2026-05-18)

What each paper has published or pointed at:

| Paper | Code repo | Data source | Public posteriors | Per-paper notes |
|---|---|---|---|---|
| **P1 Li+2023** | None paper-specific; forecast lens population at [github.com/tcollett/LensPop](https://github.com/tcollett/LensPop) (Collett 2015) | **Chen+2019 161-lens parameters** (MNRAS 488:3) | On request from corresponding author | Forecast uses LensPop; real-data test uses Chen+2019 |
| **P2 Ballard+2023** | None | HST + VLT (MUSE) archives | "from corresponding author on request" | Manual email needed for posterior chains; Smith+2024 cited for the TSPL geometry |
| **P3 Li+2026** | [github.com/Herculens/herculens](https://github.com/Herculens/herculens) | HST + VLT archives | Data Availability section truncated in our fetch; check the MNRAS published version | Uses Herculens + NumPyro + Colossus + JamPy + pPXF v8.2.6 + XSL DR3 |

Related-work code worth knowing about for the dual-stack effort:
- [github.com/Herculens/herculens_workspace](https://github.com/Herculens/herculens_workspace) — official Herculens example notebooks (13 examples; none specifically reproduce J0946 but #4 NumPyro-VI and #6 dark-satellite detection are template-relevant)
- [github.com/lenstronomy/JAXtronomy](https://github.com/lenstronomy/JAXtronomy) — alternative JAX lensing stack (lenstronomy port; not in our spec but worth knowing)
- [Caustics](https://joss.theoj.org/papers/10.21105/joss.07081.pdf) — third JAX lensing tool (JOSS 2025); out of scope for this spec
- Follow-on to P2: Enzi+2025 ("self-interacting dark matter" interpretation of the J0946 subhalo, [MNRAS 540:1](https://academic.oup.com/mnras/article/540/1/247/8123410)) — cite in Spec 02 references

### 6.7 Catalogue assembly script changes

`code/build_chen2019_catalog.py` (NEW) — primary script for P1 input. Replaces the role of `build_catalogue.py` from today's prototype.

`code/build_structural_enrichment.py` (renamed from `build_catalogue.py`) — keeps Vizier-based R_eff / n_sersic enrichment by cross-match.

### 6.8 Standing protocol — code/data discovery for any future paper

Before scaffolding ANY new paper reproduction (this program or future), run the discovery checklist. The 2026-05-18 mistake here (assuming the 161-lens sample was a SLACS/SL2S/BELLS union when the published Data Availability statement points to Chen+2019) cost us a wrong-direction build of `build_catalogue.py`. Don't repeat it.

**Discovery checklist:**

1. Fetch the published-version (NOT arxiv preprint) page from the journal — for MNRAS, the Oxford Academic URL. Use WebFetch with a prompt focused on the **"Data Availability"** and **"Code Availability"** sections.
2. Verbatim-quote both statements into the per-paper `PAPER_NOTES.md`. Note any URLs (github.com, zenodo.org, figshare.com, doi.org, journal-specific portals).
3. If the paper cites a *different* paper for its input data, follow that citation chain to ground truth. Example: Li+2023 cites Chen+2019 for the 161-lens sample; **don't** rebuild Chen+2019's compilation work from scratch.
4. For each tool the paper names (lens-modelling code, samplers, spectral libraries, etc.), check whether it has a github repo. If not previously known to us, add to `private/00_shared_infrastructure/docs/dual_stack_conventions.md` with version + install notes.
5. If the Data Availability statement points to "from the corresponding author on request" (P2's case), draft an email to the corresponding author before starting the reproduction. Don't block on the email — proceed with public-data-only reproduction in parallel — but get the request in early.
6. Cross-reference the **Related Work / Citations** section: a paper often has a follow-on or companion paper that exposes more code (e.g., P2 Ballard+2023 → Enzi+2025 self-interacting DM follow-on).
7. Search arxiv for the same author's other papers — they often share a code repo across multiple papers (e.g., Collett's [LensPop](https://github.com/tcollett/LensPop) is the forecast-simulator dependency for Li+2023, but Collett uses it across his publications).

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
