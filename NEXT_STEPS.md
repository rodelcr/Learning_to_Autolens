# Next Steps — Curriculum Gaps for New Lens Modelers

**Date drafted:** 2026-05-11
**Author:** Rodrigo Córdova Rosado
**Status:** roadmap; deliverables tracked individually below.

After 13 audited examples + 14 modules, the gap analysis for what a *new* lens modeler still needs:

## Tier 1 — biggest pedagogical gaps

### #1 — `galaxy_galaxy_single_arc` (NEW example, in progress)

A **clean SLACS-style single-lens / single-source ring/arc** — the canonical first fit. Every existing Example starts with compounds (`compound_lens`), DSPL, TDCOSMO, multi-plane, or real data; a new modeler going from Module 03's toy mocks straight to `agel_real_target` skips the textbook starting point. Auger+10 / Bolton+08 SLACS setup: 1 SIE+shear deflector, 1 Sersic source, ~0.5–2.0 arcsec Einstein ring. Single-search audited fit.

**Deliverables:**
- `Examples/galaxy_galaxy_single_arc/mocks/generate_mock.py` (autolens-native)
- `Examples/galaxy_galaxy_single_arc/01_galaxy_galaxy_single_arc.ipynb` (pedagogical walkthrough)
- `Examples/galaxy_galaxy_single_arc/README.md` + `Modules/10_Cluster_Computing/scripts/fit_example_galaxy_galaxy_single_arc.py`
- Single Cannon submit: `--part=direct`, ~2h on 32 cores → strict-PASS target

### #2 — `positions_modeling` (NEW example)

Currently position-likelihood coverage is **scattered**: 12 mentions in `Examples/compound_lens/01_compound_direct_fit.ipynb` §2 + §2.5 (API contract + sanity check) and the 3-rung H0 chain in `Examples/quad_time_delay/`. **No dedicated tutorial** walking through:
- Why positions complement imaging (resampling-from-source constraint vs pixel-level constraint)
- The API: `al.Grid2DIrregular` → `al.PositionsLH(positions=..., threshold=...)`
- How positions enter the plotter, the analysis, and the likelihood
- Threshold sensitivity (empirical: ≥0.1″ converges to v4 PASS basin; 0.01″ over-constrains)
- The 3-rung H0 chain (pos-only / image-only / joint) as the headline pedagogical demo

**Deliverables:**
- `Examples/positions_modeling/01_positions_tutorial.ipynb` — consolidates the §2.5 compound_lens sanity check + the qtd H0 chain into a standalone walkthrough.
- No new Cannon work — reuses landed posteriors from `compound_lens/results/pos_lh_sweep_*/` and `quad_time_delay/results/phase_4_*+phase_3_*+joint_h0_free`.

### #3 — `kinematic_h0_break` (extends `quad_time_delay`)

Stage 4 v0.95 deliverable from `V095_PIPELINE_PLAN.md`. Extend the joint TDCOSMO fit with a σ_v aperture-projected likelihood term using **Module 13's anisotropic Jeans theory** (already shipped). The current Track D joint fit lands H0 = 74.95 ± 2.3 with +5 bias from un-broken MSD; kinematics should close that.

**Deliverables:**
- New `--part=joint_fit_h0_kin` in `fit_example_quad_time_delay.py` — wraps `AnalysisPoint + AnalysisImaging + AnalysisKinematics` via `af.FactorGraphModel`
- Kinematic mock: σ_v(R_eff) = 280 km/s ± 10 km/s (mock truth from Module 13 stellar mass)
- Target: H0 bias drops from +5 → ~+1, σ stays ~2.3 km/s/Mpc
- Single Cannon submit, ~12h on 32 cores

## Tier 2 — completes existing scaffolds

### #4 — `bayesian_model_comparison` empirical numerics

Already scaffolded; ~20 audited fits with `log_evidence` values exist across the curriculum (compound_lens v4 / direct_epl / slam_*, mge_to_physical Search 2 vs Search 3, agel_real_target direct_clean, quad_time_delay Phase 3 vs Track D joint). Populate the worked-example tables in the existing notebook. Pure laptop work, ~2h, no Cannon.

**Deliverables:**
- `Examples/bayesian_model_comparison/01_bayesian_model_comparison.ipynb` — fill in the empirical log_Z table + Kass-Raftery classification (decisive / strong / substantial / barely-worth-mentioning) for each pair.

### #5 — `subhalo_sensitivity` full grid-search SLaM

The minimum-viable two-fit Bayes-factor demo exists; the **upgrade** is the Vegetti+ 2010 / Despali+ 2018 full grid-search SLaM. Walks a (M_sub, R_sub) grid and computes the per-cell Bayes factor → derives the (M_sub, R_sub) detection limit at a given confidence level.

**Deliverables:**
- New `--part=grid_search` in `fit_example_subhalo_sensitivity.py`
- ~50 Cannon submits (one per grid cell) at 4h each, total ~200 CPU-days. **Defer to v0.96** unless someone has a real subhalo claim to test.

### #5.5 — Multi-GPU JAX speedup test for MGE fits (task #127)

Per `project_multigpu_jax_idea.md` memory: single-GPU JAX was 4× *slower* than numpy on our typical MGE fits. Multi-GPU data-parallel + SLURM-array + per-process JAX have NOT been tested. Open question whether (a) `jax.pmap` across 2–4 GPUs on a `fasrc-cannon-gpu` node, or (b) a SLURM array with N tasks each on 1 GPU, gives a speedup relative to the numpy 32-core baseline.

**Deliverables:**
- Adapt `fit_example_mge_to_physical.py` with a `--use-jax-pmap` flag (or new driver `fit_example_mge_to_physical_gpu.py`).
- Benchmark wall + chi²/N landed vs numpy baseline on the autolens-native regenerated mock.
- Deferred priority — only valuable if numpy stops fitting in `--mem=192G` envelope or wall times exceed 24h. Currently mge fits complete in ~25min.

## Tier 3 — production AGEL-realistic

### #6 — `multi_band_joint_fit` (NEW example)

HST WFC3 (F814W) + JWST NIRCam (F200W) joint fit on the same lens. AGEL DR2 targets all have multi-band imaging. The autolens API supports multi-dataset fits via `af.FactorGraphModel` — same pattern as the TDCOSMO joint fit, but with two imaging analyses instead of imaging + point-source. Demonstrates wavelength-dependent source structure (older stellar populations in NIR, star-forming regions in optical).

**Deliverables:** mocks generator for 2 bands, driver, notebook, Cannon submit (~8h).

### #7 — `cosmography_joint_posterior` (NEW example)

Combine the **DSPL Stage 2** posterior on (Ωₘ, w₀) (when `dspl_beta_chain` lands) + the **TDCOSMO chain** posterior on H0 into a joint 3D constraint following **Birrer+ 2020 §4** methodology. Demonstrates the cross-link between Stages 1 and 4 of the pipeline plan — DSPL pins (Ωₘ, w₀) without H0; TDCOSMO pins H0 with weak (Ωₘ, w₀) dependency; jointly they're tighter than either alone.

**Deliverables:** notebook that loads the two posteriors and renders the joint corner. Pure laptop work once Cannon results land.

## Sequencing

Work order under auto-mode (no Cannon dependency for #1, #2, #4):

1. **#1 `galaxy_galaxy_single_arc`** (in progress today)
2. **#2 `positions_modeling`** (laptop only, reuses landed results)
3. **#4 `bayesian_model_comparison` empirical** (laptop only)
4. **#3 `kinematic_h0_break`** (depends on Module 13 Jeans theory consolidation + Cannon)
5. **#6 `multi_band_joint_fit`** (depends on mock design choices)
6. **#7 `cosmography_joint_posterior`** (depends on `dspl_beta_chain` Cannon result)
7. **#5 `subhalo_sensitivity` grid SLaM** (v0.96)

The first three close the biggest pedagogical gap for new modelers without any new Cannon CPU spend.
