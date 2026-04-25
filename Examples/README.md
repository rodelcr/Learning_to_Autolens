# Examples — Practice Problems by Lens Architecture

Standalone practice problems for learners who have worked through **Modules 01–11**. Each example is self-contained: a lens architecture, the data (simulated or real), a direct-fit notebook, a SLaM/staged-fit notebook, and a diagnostic audit.

## Philosophy

The `Modules/` tree is **linear curriculum**: each module builds on the one before it, and teaches a single new concept. The `Examples/` tree is a **practice gym**: each example is picked by *physical architecture*, not by *new autolens feature*, and can be tackled in any order by anyone who has finished the modules. The goal is to show that what you learned on the toy `simple` dataset generalises to realistic lens configurations.

Every example follows the same 4-file template:

```
Examples/<name>/
├── README.md                     # Problem statement + references
├── 01_<name>_direct_fit.ipynb    # Single-search curated-prior fit
├── 02_<name>_slam.ipynb          # SLaM / staged-chain variant
├── mocks/                        # Data (simulated or real FITS)
└── results/                      # Cannon-produced lightweight artifacts
```

## The roadmap

| # | Example | Architecture | Status | What it teaches |
|---|---|---|---|---|
| 1 | **[compound_lens/](compound_lens/)** | Two deflectors at different z, one source | ✓ shipped | Multi-plane `al.Tracer`, staged chaining across lens bodies, when to simplify to an effective deflector |
| 2 | **[double_source_plane/](double_source_plane/)** | One lens, two sources at different z | ✓ shipped | Cosmological distance ratios, joint multi-source posterior, β cosmography |
| 3 | [group_scale/](group_scale/) | BGG + satellites + extended envelope (same z) | ◐ in-progress | Cluster-lite multi-component mass; 3 Cannon attempts stalled in burn-in (see `group_scale/README.md`) |
| 4 | **[disky_spiral_lens/](disky_spiral_lens/)** | Non-elliptical lens morphology | ✓ shipped | Two-component Sersic vs. single Sersic, mass/light PA misalignment, Bayes-factor model comparison |
| 5 | [quad_time_delay/](quad_time_delay/) | Point-source quasar, four images, time delays | ◯ stub | Point-source likelihood, time-delay cosmography, H₀ |
| 6 | [agel_real_target/](agel_real_target/) | A real AGEL survey lens on HST data | ◯ stub | Everything together on actual HST data |

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
