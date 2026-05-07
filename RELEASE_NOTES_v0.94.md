# Release Notes — v0.94-alpha (DRAFT)

**Status:** in-progress draft. Tag pending Cannon results from `dspl_beta_chain` (job 11214940) and `truth_fc_m3_v4` (job 11214941). Final classification of those two fits goes here once they land.

**Predecessor:** `v0.93-alpha` (2026-05-07)
**Author:** Rodrigo Córdova Rosado (rodrigo.cordova_rosado@cfa.harvard.edu, Harvard CfA)

---

## What's new in v0.94

v0.94 closes the two methodology debts from v0.93 and adds the canonical "physical bar" reference notebook that v0.92/v0.93 examples have all been pointing at without it existing.

1. **Module 11 (Physical Mass Models)** — pedagogical capstone shipped (~29 cells, executes <5s with no Cannon dependency). Six sections: 6-panel residual audit, numerical bar with Bonferroni-corrected noise floor, Pattern A-F failure catalogue, f_DM(<θ_E) extraction, γ′ slope recovery, decision flowchart. Solutions/SOLVED variant included. 6 cross-referencing READMEs updated from "Module 11 planned" to "Module 11 shipped".

2. **DSPL β-cosmography staged chain (task #110)** — replaces the v0.93 Pattern A stall. Stage 1 fits lens + sources at fixed FlatLambdaCDM(70, 0.30); Stage 2 frees Om0/w0 with **TruncatedGaussianPrior** bounds (Om0 in [0.05, 0.60], w0 in [-1.6, -0.4]) and inherits Stage 1 lens/source posteriors as priors. The truncated bounds prevent the autolens FlatwCDM angular-diameter integrator from being asked about Om0 ≤ 0 or extreme phantom-DE w0 < -1.5 — which crashed the integrator and produced f_live=1.0 indefinitely in v0.93.

3. **Nautilus checkpoint resume deadlock fix (task #111)** — same TruncatedGaussian fix applied to `build_R5_truth_freecosmo_model()` in the compound_lens_zoo climb driver. The v0.93 truth_fc trio resume deadlock was traced via `diagnose_nautilus_resume.py` to Pattern B3: a saved live point evaluating to a cosmology that crashed the autolens integrator on resume, hanging the worker. New `Modules/10_Cluster_Computing/CLUSTER_WORKFLOW_NOTES.md` "Checkpoint hygiene" section establishes the rule **"always assign a fresh `unique_tag` when prior bounds change"** to prevent recurrence.

4. **chi²-at-truth methodology applied to mge_to_physical (Track D)** — the diagnostic falsified the v0.92-stated diagnosis ("missing 2nd source + secondary deflector"). Removing those components changes χ²/N by <1%; the actual issue is a **framework-level Sersic evaluation difference** (lenstronomy simulator vs autolens fitter) at the cuspy `n=4.9` lens-light peak, producing 33σ at the central pixel even with all truth components present. README updated with corrected diagnosis. Fix is to regenerate the mock natively in autolens — deferred to v0.95+.

### New strict-PASS shipped fits

| Example | Result | Wall | log_Z | χ²/N | max\|res\| |
|---|---|---|---|---|---|
| `double_source_plane/beta_chain` | _PENDING_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| `compound_lens_zoo/mock_3_R5_truth_freecosmo` | _PENDING_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

### New methodology

- **`Modules/10_Cluster_Computing/scripts/diagnose_nautilus_resume.py`** — opens a Nautilus checkpoint.hdf5 with h5py and dumps bound count, n_dim, n_live, sampler attrs, and per-column live-point min/max. Used to discriminate model-hash-mismatch (B1) vs. version-format-mismatch (B2) vs. bad-sample-point (B3). The 2026-05-07 truth_fc deadlock was diagnosed as B3 in <30 minutes with this tool.
- **CLUSTER_WORKFLOW_NOTES.md "Checkpoint hygiene" section** — codifies the unique_tag rule for prior-bound changes. Prevents future deadlocks from the same root cause.
- **chi²-at-truth diagnostic now demonstrably falsifies a hypothesised model gap.** The mge_to_physical case is the second public application of the diagnostic (after cluster_scale's `f8471bb` fix); here it *prevented* a bad Cannon submit by showing the proposed fix wouldn't change χ². Methodology generalisation: ablation of truth components is the right next step when chi²-at-truth surfaces a structural-looking gap.

### Drivers + repo hygiene

- `fit_example_double_source_plane.py`: split into `build_beta_fixedcosmo_fit` (Stage 1, no cosmology) + `build_beta_freecosmo_v3_fit` (Stage 2 with TruncatedGaussian + optional prior passing) + `build_beta_chain` (orchestrates both). New `--part` choices: `beta_fixedcosmo`, `beta_freecosmo_v3`, `beta_chain` (recommended), and `beta_freecosmo` redirects to v3 for backward compat.
- `fit_example_compound_lens_zoo_climb.py`: `build_R5_truth_freecosmo_model` uses TruncatedGaussianPrior on Om0/w0.
- Both DSPL fresh-start runs use `unique_tag=mock_1_v0_94_{chain,standalone}` to avoid resuming the v0.93 deadlocked checkpoints.

---

## v0.94 ship-set tally (delta from v0.93)

### Modules — Module 11 newly shipped

| # | Module | v0.93 status | v0.94 status |
|---|---|---|---|
| 11 | Physical Mass Models | ◯ planned | **✓ ship** (full notebook + Solutions/SOLVED, executes <5s) |

### Examples — status changes since v0.93

| Example | v0.93 status | v0.94 status |
|---|---|---|
| `double_source_plane` | ◐ research-in-progress (Pattern A stall) | _TBD_ — pending Cannon `beta_chain` result |
| `compound_lens_zoo` (mock_3 truth_freecosmo) | ◐ research-in-progress (Nautilus deadlock) | _TBD_ — pending Cannon `truth_fc_m3_v4` result |
| `mge_to_physical` | ◐ research-in-progress (claimed missing components) | ◐ research-in-progress (corrected diagnosis: framework-level Sersic eval mismatch) |

All other examples retain their v0.93 ship/research-in-progress status.

---

## What's deferred to v0.95+

1. **mge_to_physical strict-PASS** — requires regenerating the mock natively in autolens to eliminate the lenstronomy↔autolens cuspy-Sersic evaluation difference.

2. **Compound zoo mocks 2 and 5** (truth_freecosmo) — only mock_3 was retried in v0.94 because the v0.93 chi²-at-truth diagnostic showed it was the most likely to converge (chi²/N=2.1 at literal truth, below the mock_4 known-good baseline of 4.9). Mocks 2/5 stay deferred unless mock_3 v4 ships clean and budget remains.

3. **Modules 12 (TDCOSMO + MSD), 13 (kinematic TDCOSMO), 14 (multi-plane)** — curriculum expansion, not blocked.

4. **`subhalo_sensitivity` full grid-search SLaM** (Vegetti+ 2010 / Despali+ 2018 methodology) — scaffolded only.

5. **`bayesian_model_comparison` empirical worked-examples populated.**

---

## Roadmap to v0.95

The v0.94 → v0.95 path is curriculum-expansion-led, not methodology-debt-led (assuming Cannon results land clean):

- Module 12 (TDCOSMO + MSD) — Refsdal time-delay cosmography, mass-sheet degeneracy derivation.
- mge_to_physical native-autolens mock regeneration + strict-PASS retry.
- `quad_time_delay` audited result (point-source likelihood pipeline).

---

## Getting started with v0.94

```bash
git clone https://github.com/rodelcr/Learning_to_Autolens.git
cd Learning_to_Autolens
git checkout v0.94-alpha       # if you want this exact tagged state
open START_HERE.md             # ← read this first
```

For students new to the audit methodology, the canonical reference is now:

```bash
open Modules/11_Physical_Mass_Models/11_physical_mass_models.ipynb
```

For the chi²-at-truth diagnostic recipe used to falsify mge_to_physical's v0.92 hypothesis, see `PROGRESS_LOG.md` 2026-05-05 / 2026-05-07 entries.

---

## Acknowledgements

v0.94's cleanest outcome is **chi²-at-truth as a *negative* evidence tool** — falsifying a stated model hypothesis (mge_to_physical) before any Cannon time was spent. The cluster_scale fix (`f8471bb`, v0.93) demonstrated the *positive* version of the same diagnostic. Together they establish the recipe: diagnose at truth before submitting; if χ² is bad at truth, the fit driver has a structural bug; if χ² is fine at truth, missing components are ruled out and the next failure mode is search-budget or framework-level. The Nautilus checkpoint hygiene rule is the same insight applied to the resume side: always rename when priors change, lest stale live points evaluate differently and hang the worker pool indefinitely.
