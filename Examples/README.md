# Examples — Practice Problems by Lens Architecture

Standalone practice problems for learners who have worked through **Modules 01–11**. Each example is self-contained: a lens architecture, the data (simulated or real), a direct-fit notebook, a SLaM/staged-fit notebook, and a diagnostic audit.

## Philosophy

The `Modules/` tree is **linear curriculum**: each module builds on the one before it, and teaches a single new concept. The `Examples/` tree is a **practice gym**: each example is picked by *physical architecture*, not by *new autolens feature*, and can be tackled in any order by anyone who has finished the modules. The goal is to show that what you learned on the toy `simple` dataset generalises to realistic lens configurations.

Every example follows the same 4-file template:

```
Examples/<name>/
├── README.md                     # Problem statement + references
├── 00_climb_to_<name>.ipynb      # OPTIONAL: bridge from single-deflector to this architecture
├── 01_<name>_direct_fit.ipynb    # Single-search curated-prior fit
├── 02_<name>_slam.ipynb          # SLaM / staged-chain variant
├── mocks/                        # Data (simulated or real FITS)
└── results/                      # Cannon-produced lightweight artifacts
```

The optional **`00_climb_to_*.ipynb`** notebook is a *bridge* for students who finished Module 09 (single-deflector fitting) and now face a multi-galaxy or multi-source-plane system. It demonstrates two general techniques that recur across architectures: **iterative masking** (start with a tight mask that excludes nuisance objects, fit cleanly, expand the mask, hold the early-pass posterior fixed) and **position likelihoods** (`al.PositionsLH` to anchor mass models to image positions). Each climb notebook runs in `<60` seconds via `PYAUTOFIT_TEST_MODE=1` — it teaches the API and methodology, not the wall-clock fit.

## The roadmap

| # | Example | Architecture | Status | What it teaches |
|---|---|---|---|---|
| 1 | **[compound_lens/](compound_lens/)** | Two deflectors at different z, one source | ✓ shipped + 🪜 climb | Multi-plane `al.Tracer`, staged chaining across lens bodies, when to simplify to an effective deflector |
| 2 | **[double_source_plane/](double_source_plane/)** | One lens, two sources at different z | ✓ shipped + 🪜 climb | Cosmological distance ratios, joint multi-source posterior, β cosmography |
| 3 | [group_scale/](group_scale/) | BGG + satellites + extended envelope (same z) | ◐ in-progress + 🪜 climb | Cluster-lite multi-component mass; freely-fit attempts stall but `truth_anchored` PASSED (χ²/N=1.025, max\|res\|=4.50σ) — diagnostic confirms search-space exploration is the bottleneck, not model representability (see `group_scale/01_group_scale_fit.ipynb` §3.1) |
| 4 | **[disky_spiral_lens/](disky_spiral_lens/)** | Non-elliptical lens morphology | ✓ shipped | Two-component Sersic vs. single Sersic, mass/light PA misalignment, Bayes-factor model comparison |
| 5 | [quad_time_delay/](quad_time_delay/) | Point-source quasar, four images, time delays | ◐ in-progress | Point-source likelihood (`AnalysisPoint`), time-delay cosmography, H₀ recovery |
| 6 | [mge_to_physical/](mge_to_physical/) | Stars + dark matter decomposition (MGE light → lmp.Sersic + NFW) | ◐ in-progress | f_DM(<θ_E), M/L recovery, three-search chain mirroring `mass_stellar_dark/chaining.py` |
| 7 | [compound_lens_zoo/](compound_lens_zoo/) | Five compound mocks (mock_2–mock_6, lenstronomy origin) — same prior set, varying z, γ′, cosmology | ◐ in-progress + 🪜 climb | Methodology robustness; cosmology mis-specification effects on recovered (θ_E, γ′); the **production climb** at scale (R0→R5 ladder) |
| 8 | [bayesian_model_comparison/](bayesian_model_comparison/) | *Pedagogy* — log-evidence + Bayes-factor methodology with double-checked references | ◐ in-progress | Jeffreys & Kass-Raftery scales (and the factor-of-2 trap), Occam's razor, when log_Z misleads |
| 9 | [interferometer_basic/](interferometer_basic/) | Galaxy-scale lens on uv-plane (SMA) visibility data | ◐ in-progress | `al.AnalysisInterferometer` API, visibility-plane χ², dirty-image vs sky-plane reasoning |
| 10 | [agel_real_target/](agel_real_target/) | **Real** AGEL HST data — AGEL013322-125201A (DCLS0133-1252), ACS WFC F606W cutout | ◐ in-progress | What changes on real data: drizzle-correlated noise, placeholder vs empirical PSF, lens-light wing leakage, redshift marginalisation |

**Status legend.** ✓ shipped: at least one converged Cannon result committed and audited. ◐ in-progress: notebook + driver exist, results not yet finalised. 🪜 climb: a `00_climb_to_*.ipynb` bridge notebook is available, teaching the iterative-masking + position-likelihood techniques in the context of this architecture.

## Climb scaffolding

Four `00_climb_to_*.ipynb` notebooks bridge from a single-deflector fit to a multi-object architecture. They share a common template (problem inspection → step-by-step model elaboration → Cannon-result hand-off) and each runs in `<60` seconds with `PYAUTOFIT_TEST_MODE=1`.

| Climb notebook | Architecture | Featured techniques |
|---|---|---|
| [`compound_lens/00_climb_to_compound.ipynb`](compound_lens/00_climb_to_compound.ipynb) | 2 deflectors @ different z (minimal) | Multi-plane `Tracer` API as a 1-line upgrade over single-deflector |
| [`compound_lens_zoo/00_climb_to_compound.ipynb`](compound_lens_zoo/00_climb_to_compound.ipynb) | 2 deflectors @ different z (production) | Iterative masking, `al.PositionsLH`, `extra_galaxies` API, multi-plane Tracer, Pattern E diagnosis |
| [`double_source_plane/00_climb_to_dspl.ipynb`](double_source_plane/00_climb_to_dspl.ipynb) | 1 lens, 2 sources @ different z | Cosmological distance ratios via joint source-plane fit, β_12 derivation |
| [`group_scale/00_climb_to_group.ipynb`](group_scale/00_climb_to_group.ipynb) | BGG + 3 satellites @ same z | Iterative masking by photometric centroids, BGG ↔ satellite degeneracy, truth-anchored validation |

## Running an Example

1. Read its `README.md` to understand the physical problem and where the data came from.
2. Open `01_<name>_direct_fit.ipynb`. The heavy Nautilus fit is skip-guarded: on a laptop the committed Cannon result loads, no sampling happens. To force a local re-fit set `LTA_RUN_HEAVY=1` before starting Jupyter.
3. For a fresh cluster run, see `Modules/10_Cluster_Computing/scripts/fit_example_<name>.py` and the submit workflow in `Modules/10_Cluster_Computing/`.
4. After inspecting the results, go to `02_<name>_slam.ipynb` for the staged-chain variant and its comparison to the direct fit.
5. Work the exercises at the end of each notebook.

## What "audited" means in this collection

Every committed result is audited using the **`/autolens-fit-diagnostics`** skill — six mandatory panel checks plus the numerical threshold (chi²/N ≤ 1.3 pass, ≤ 2.0 suspect; max |res| ≤ 4σ pass, ≤ 6σ suspect). The verdict template from Module 11 §4 is filled out in each notebook's audit section. A result that numerically passes but fails a physical-bar panel (mesh collapse, unphysical caustic, un-subtracted lens light) is **failed**, regardless of chi².

## Extending the collection

To add a new architecture:

1. `mkdir Examples/<new_name>/` with the four canonical subfiles.
2. Add the mocks under `mocks/` or point at a shared dataset.
3. Create `fit_example_<new_name>.py` in `Modules/10_Cluster_Computing/scripts/`, mirroring the existing `fit_module04/05/06/09.py` pattern.
4. Extend `submit_cannon.slurm`'s dataset-routing case.
5. Add a row to the table above.

## See also

- **[Module 11: Physical Mass Models](../Modules/11_Physical_Mass_Models/)** — the physical-bar audit methodology these examples lean on.
- **Recurring failure patterns** — five modes keep showing up across Modules and Examples: rotational mirror-image optimum (SIE+shear without seeded priors), autolens sign-convention drift between versions, SLaM stage cascade (early stage fails quietly and contaminates downstream), doc/code dataset mismatch (narrative says one dataset, code loads another), and forced-compound suboptimum (tight priors on a secondary Einstein radius when the data supports a single effective lens). Pattern E was discovered while shipping the compound_lens example. The Mod 04 and 11 notebooks document the first four in their audit sections; Example 1's `02_compound_slam.ipynb` documents the fifth.
- **[`Modules/10_Cluster_Computing/`](../Modules/10_Cluster_Computing/)** — the Cannon submit/pull workflow that produces the committed artifacts.
