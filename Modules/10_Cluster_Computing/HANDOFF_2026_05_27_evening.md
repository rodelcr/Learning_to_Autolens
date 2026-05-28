# Handoff — 2026-05-27 (evening session)

**Headline since the morning handoff** (`HANDOFF_2026_05_27.md`): a
massive parallel session that closed multiple infrastructure loops, shipped
two new modules, and queued ~6 new Cannon jobs. Paper-repro program
went from "Spec 03 mostly done" → "Specs 01, 02, 03 architecture all
wired + Cannon-submitted; Spec 05 has a publishable cross-class M_BH
posterior figure; Modules 13 + 15a + 15b all shipped".

## Commits this session

```
4392203  Module 13: ship joint factor-graph fit + curriculum status update
c0edc35  Notes: A1201 M_BH cross-class scoreboard + F390W anomaly
0ae3736  Module 15b §3: cite M_BH recovery empirical anchor
39dfd7e  Module 15 → 15a/15b split + bug fixes + M_BH recovery audit
2482923  HANDOFF + PROGRESS_LOG 2026-05-27 (morning handoff)
91a5212  Notes: SMBH from lensing lit review + BPL ↔ SMBH degeneracy addendum
```

## What shipped (in chronological order)

### Phase 1 — Module 15 split + bug fix + M_BH recovery anchor

- **Split** `Modules/15_Radial_Arcs_Caustic_Topology/15_radial_arcs.ipynb`
  (39 cells, mixed numpy + PyAutoLens) into:
  - `Modules/15a_Radial_Arcs_Analytic_Foundation/` (18 cells, pure
    numpy, 3.4s exec)
  - `Modules/15b_Radial_Arcs_PyAutoLens_Realistic/` (27 cells, PyAutoLens,
    2:37 exec)
  - Original retained with banner for provenance
- **Bug fixed**: §3 numerical d(β_r)/dγ at γ=2 used central difference
  straddling the γ=2 boundary where the radial caustic vanishes (λ_r≡1
  for γ≥2). Replaced with one-sided LEFT polyfit; now reports
  d(β_r)/dγ ≈ −2.58 per unit γ. Applied to both 15a and legacy.
- **PEDAGOGICAL_AUDIT_2026_05_27.md** caught a theory-first-code-second
  violation at the §3.7→§3.8 boundary; fix in 15b §3 header.
- **15b §3 cited** the empirical anchor: `Examples/radial_arc_smbh/`
  posterior recovers θ_BH at truth within 1σ; omitting SMBH biases θ_E
  to +4.5σ pull (smoking-gun diagnostic).

### Phase 2 — A1201 (Spec 05) deep dive

- **Pulled 7 completed A1201 fits** from Cannon (lp_v5, BPL ×3, F390W ×2,
  with_kin_v2).
- **`compute_bayes_matrix.py` updated** to include lp_v5, lp_f390w,
  with_kin_v2 rows.
- **Critical finding — lp_v5 is the WRONG baseline**: widened-prior
  baseline drifted to θ_E=1.67, γ=1.57, e1=-0.30 (different physical mode
  than lp_orig's θ_E=1.95, γ=2.13 — the N+23 canonical mode). The +624 lnZ
  improvement over lp_orig is the classic
  [[feedback-bayes-factor-vs-truth]] failure pattern.
- **Stage 2 v5chain (job 15969519, COMPLETED 3h52m)** confirmed in the
  same wrong mode: SMBH rail-pinned at 0.001 (1σ: 0.001–0.18). NOT a
  detection — but expected, given wrong-mode baseline.
- **3 v5chain bugs found + fixed**:
  - **Bug 1**: `chain_priors_from_lp._extract_param_dict` missed a
    second-level `{'type':'dict', 'arguments':...}` unwrap, returning
    `{}` (0 leaves seeded). Patched.
  - **Bug 2**: mask radius mismatch (v5chain used 3.5″, lp_v5 used 6.0″
    → 24024 vs 70688 pixels → ~3× lnZ ratio).
  - **Bug 3**: lp_v5 itself is the wrong mode (see above).
- **NOTES_v5chain_bugs_2026_05_27.md** documents all three bugs +
  expected outcomes of v5chain v2.
- **v5chain v2 (job 16377115)** queued with chain fix + mask 6.0″.
- **BPL extended-prior (job 16341076)** queued — tests whether r_break
  collapses to SMBH-scale when prior floor removed (DESJ0206 found
  r_break ≤ 10 pc). Currently RUNNING.
- **A1201 SMBH lit review §VI bis** (`Notes/SMBH_from_lensing_literature_review.md`)
  documents the BPL ↔ SMBH degeneracy + the 2026-05-27 cross-class
  scoreboard.

### Phase 3 — Spec 03 (Li+26 DSPL) end-to-end wiring

- **gNFW custom profile** shipped as `GeneralisedNFW(al.mp.gNFW)` thin
  wrapper. Key finding: `al.mp.gNFW` already ships in autolens 2026.4.13.6 —
  hand-rolled scipy.quad would be wrong; wrapper preserves all autolens
  infra. NFW-limit agreement at 3.6×10⁻⁵ arcsec (autolens internal MGE
  floor, not our code).
- **Herculens bridge** (`herculens_bridge.py`) shipped:
  - 4 converters (PowerLaw, Isothermal, ExternalShear, Sersic)
  - DSPL builder (`build_herculens_dspl_lens_image`)
  - NUTS layer (`numpyro_nuts_fit`)
- **4 convention divergences cataloged** (`NOTES_herculens_bridge_2026_05_27.md`):
  - Centre axis (y,x) vs (x,y) — swap
  - ell_comps axis-system rotated 90° (EPL only; SHEAR has NO swap)
  - theta_E geometric-mean vs major-axis — for γ=2 closed-form factor
    `2√q/(1+q)`; for γ≠2 numerical calibration flag
  - Sersic amp normalization (~0.89 constant, absorbed by free amp)
- **Cross-validation**:
  - Single-plane SIE+Shear: 2×10⁻¹³ arcsec agreement
  - Multi-plane DSPL: autolens-at-source-2 = Herculens-raw to 1.1×10⁻¹⁶
  - Cross-plane scaling = analytic D_LS/D_S to 1×10⁻⁵
- **DSPL J0946 mock** generated (`generate_mock_dspl_jackpot.py`) for
  end-to-end testing before HST data lands.
- **DSPL driver wired** (`dspl_jackpot_autolens.py` already existed; today
  verified end-to-end). 28 → 29 free params (Stage 1 fixes γ_inner=1;
  Stage 2 frees it).
- **Architecture check passes** for both stages.
- **Cannon job 16403803** submitted: DSPL Stage 1 on mock, 24h budget.

### Phase 4 — Spec 02 (Ballard+23 TSPL) parallel scaffold

- Mirrored the Spec 03 pattern: `tspl_jackpot_autolens.py`,
  `generate_mock_tspl_jackpot.py`, `tspl_jackpot_architecture_check.py`,
  `submit_tspl_jackpot.slurm`.
- **z_s3 = 5.96** (canonical Ballard+23 / Smith+24 value, not spec's
  5.975 — transcription drift).
- **NFW perturber** (`al.mp.SphIsothermal` doesn't exist in 2026.4.13.6;
  NFW is the correct ΛCDM perturber anyway).
- **`use_jax=False`** applied (same JAX-scan crash on PointMass per
  `feedback_autolens_2026_4_bugs.md`).
- **Architecture check passes** for all 3 stages (Stage 1: 35 params,
  Stage 2 +NFW: 39 params, WBH +PointMass: 38 params).
- **Cannon job 16404972** submitted: TSPL smoke with PART=stage1, n_live=50.

### Phase 5 — Spec 01 (Li+23 population cosmography) scaffold

- Different architecture from 02/03: population-level Bayesian fit, not
  image-plane. Extended in-tree `per_system_likelihood.py` +
  `population_model_autofit.py` (3 tests passing, Gauss-Hermite γ_eff
  marginalisation) with joint (Ω_m, w_0) sampling via astropy `FlatwCDM`.
- N=10 mock lenses, truth = (Ω_m=0.30, w_0=−1.00, μ_γ=2.05, σ_γ=0.12).
- **Architecture check PASSES** — 4 free top-level params, log_L at
  truth = −3.10 > random draw = −11.29.
- **Cannon job 16475828** submitted: smoke with n_live=50.
- **Blocker**: production fit (w_0 = −0.96 ± 0.46 on 161-lens
  Chen+2019) needs the enriched `catalogue_161.csv` which is a Spec 00
  deliverable.

### Phase 6 — Module 13 (TDCOSMO + Kinematics) ship

- **Critical correction**: `AnalysisKinematics` was ALREADY production-
  wired (hardened through 6+ A1201 Cannon jobs since 2026-05-18). The
  CLAUDE.md "stubbed" status was outdated.
- **Notebook §6 "Worked joint fit"** added (15 cells): mock σ_v + imaging
  → FIT 1 (imaging-only Nautilus, 44.7s) → FIT 2 (imaging+kinematic
  factor-graph Nautilus, 50.7s) → posterior comparison → 2-panel viz +
  honest "what this shows / doesn't show" markdown.
- **Result**: lens-only γ′=2.0484±0.0093 → lens+kin γ′=2.0492±0.0085
  (1.09× tighter, both bracket truth 2.05). Honest single-system result;
  exercise 5 points to TDCOSMO IV hierarchical for dramatic-tightening.
- **Notebook exec time**: 138s end-to-end. CLAUDE.md curriculum table
  updated.

### Phase 7 — A1201 M_BH Fig 5-style comparison

- **Figure shipped**:
  `private/2303_15514_nightingale2023_abell1201/results/figures/fig5_mbh_comparison_2026_05_27.{pdf,png}`
- **`extract_mbh.py` patched**: weighted percentiles (unweighted gave
  ~2% bias; Nautilus weights are concentrated), FactorGraphModel `1.`
  prefix handling, --batch mode, 95% upper-limit for rail-pinned.
- **Cross-class scoreboard**:
  - PL+SMBH (orig): **(7.71 +0.60 −0.51) × 10⁹ M_sun**, ΔlnZ=+21.3 (−1.2σ vs N+23)
  - Decomp+SMBH: **(9.82 +0.64 −0.59) × 10⁹ M_sun**, ΔlnZ=+29.9 (in N+23 1σ)
  - F390W non-detection (95% UL <4×10⁸, ΔlnZ=+2) — **OPPOSITE of N+23
    (+100); real anomaly to investigate**
  - σ_v Jeans Stage 3 ALSO rail-pinned — kinematics didn't break
    degeneracy in our parametric-source baseline
  - BPL+SMBH consistent with zero, as N+23 predicts for tied-centre BPL
  - v5chain (wrong-mode): 95% UL <5×10⁸, ΔlnZ unreliable due to mask mismatch

## Cannon state at handover (14 jobs in flight)

### Today's submissions

| Job | Purpose | State |
|---|---|---|
| 16341076 | A1201 BPL extended-prior (this morning) | RUNNING (9h, 15h left) |
| 16376586 | A1201 adapt_3s resubmit (TIMEOUT recovery) | RUNNING (6h, 65h left) |
| 16376589 | A1201 bpl_smbh_free_v2 resubmit | PENDING (3d budget) |
| 16377115 | A1201 v5chain v2 (chain bug fixed + mask=6.0″) | PENDING (1d budget) |
| 16403803 | Spec 03 DSPL Stage 1 on mock (gNFW γ_inner=1) | PENDING (1d budget) |
| 16404972 | Spec 02 TSPL smoke | PENDING (1d budget) |
| 16475828 | Spec 01 cosmo smoke (N=10 mock) | PENDING (1d budget) |

### Yesterday's RUNNING

| Job | Purpose | TimeLeft |
|---|---|---|
| 15715446 | A1201 decomp_smbh_3s_f390w | 22min — IMMINENT |
| 15715447 | A1201 bpl_smbh_3s_f390w | 21h |
| 15715540 | A1201 adapt_3s_f390w | 21h |
| 15717750 | A1201 decomp_adapt_3s | 22h |
| 15717768 | A1201 decomp_adapt_3s_f390w | 24h |

### FAILED this session — need investigation next session

- `16350240` herculens_nuts_smoke v2 — used system `pip` (Python 3.10
  Mambaforge) instead of `python -m pip` (target env's Python 3.12)
- `16377459` herculens_nuts_smoke v3 — `python -m pip install -U
  'jax[cuda12]'` succeeded but jax devices still CPU only. **Next
  session debug**: explicitly install jaxlib cuda12 variant via
  `python -m pip install -U "jax[cuda12]==0.10.1"` with the cuda-extra
  index URL, OR uninstall jax + jaxlib first, then reinstall fresh.

## First actions next session

1. **Re-2FA Cannon** if socket expired: `! ssh cannon "echo ok"`.
2. **Pull artifacts** for any landed jobs:
   ```bash
   bash Modules/10_Cluster_Computing/scripts/pull_from_cannon.sh --go
   # Plus direct rsync for private/2303_*/output and private/{2602,2309,2307}_*/results
   ```
3. **Audit the landed fits** using `/autolens-fit-diagnostics`. Special attention:
   - **`16377115` v5chain v2**: does the chain seeding now bracket the
     wrong-mode or the N+23 mode? If wrong-mode → confirms our methodology
     decision; if N+23 mode → SMBH posterior is the headline.
   - **`16341076` BPL extended-prior**: does r_break collapse to ≤10 pc
     (DESJ0206 finding)? If yes → BPL is a disguised SMBH at A1201.
   - **`16403803` Spec 03 Stage 1**: first DSPL gNFW fit on bridge mock —
     does it recover κ_s=0.05, r_s=12″ truth?
   - **`16404972` Spec 02 TSPL**: first 4-source-plane fit — does it work?
   - **`16475828` Spec 01 cosmo**: first population fit — does mock
     recover Ω_m=0.30, w_0=−1.00?
4. **Investigate F390W anomaly** — our F390W gives ΔlnZ=+2 vs N+23's
   ~+100. Compare our F390W noise map + PSF + central-BCG mask to N+23
   published equivalents.
5. **Investigate why σ_v Jeans didn't break the degeneracy** — likely
   needs converged source first (adapt_3sersic, job 16376586, running);
   wait for that to land + retry kinematics.
6. **Fix NUTS GPU install** — explicit jaxlib cuda12 variant.
7. **Submit Stage 2 chain for DSPL** once Stage 1 lands:
   `sbatch --export=ALL,PART=chain` on the same slurm.

## What's blocked / waiting

- **Spec 01 production** ← blocked on Spec 00 deliverable
  (`catalogue_161.csv`). Subagent currently scaffolding (status: see
  task #94).
- **Spec 02/03 real-data fits** ← blocked on HST data download
  (`download_j0946.sh` needs Ballard+23 P2 paper-repro data pipeline).
- **NUTS production on Herculens** ← blocked on Cannon GPU jax install
  debug.

## Memory updates

- [[project-herculens-state]] updated with bridge shipped + 4 conventions
- [[project-a1201-first-mbh-detection]] amended with lp_v5 wrong-mode finding
- [[memory MEMORY.md]] index updated

## Files shipped today (new + modified)

### Public-track (committed)

```
CLAUDE.md  (Module 13 + 15a/15b curriculum rows updated)
Notes/SMBH_from_lensing_literature_review.md  (+§VI bis A1201 scoreboard)
Modules/13_TDCOSMO_Kinematics_MSD/13_tdcosmo_kinematics_msd.ipynb  (§6 joint fit added)
Modules/15_Radial_Arcs_Caustic_Topology/15_radial_arcs.ipynb  (banner + nan fix)
Modules/15_Radial_Arcs_Caustic_Topology/PEDAGOGICAL_AUDIT_2026_05_27.md  (new)
Modules/15a_Radial_Arcs_Analytic_Foundation/15a_radial_arcs_analytic.ipynb  (new)
Modules/15b_Radial_Arcs_PyAutoLens_Realistic/15b_radial_arcs_pyautolens.ipynb  (new)
Modules/15b_Radial_Arcs_PyAutoLens_Realistic/_spiral_lens_helpers.py  (new, copy of original)
Examples/radial_arc_smbh/compute_mbh_recovery.py  (new)
Examples/radial_arc_smbh/results/M_BH_RECOVERY_2026_05_27.md  (new)
```

### Private-track (gitignored, but on disk)

```
private/2602_20889_li2026_dspl_imf_nfw/
  code/gnfw_profile.py  (rewritten)
  code/gnfw_smoke.py  (new)
  code/herculens_bridge.py  (new)
  code/herculens_bridge_smoke.py  (new)
  code/herculens_dspl_bridge_smoke.py  (new)
  code/herculens_nuts_smoke.py  (new)
  code/generate_mock_dspl_jackpot.py  (new)
  code/dspl_jackpot_architecture_check.py  (new)
  code/dspl_jackpot_smoke.py  (new)
  tests/test_gnfw_profile.py  (rewritten)
  data/mock/{image,noise_map,psf}.fits + mock_truth.json  (new)
  submit_dspl_jackpot.slurm  (new)
  submit_nuts_smoke.slurm  (new)
  NOTES_herculens_bridge_2026_05_27.md  (new)
  PROGRESS_2026_05_27.md  (new)

private/2309_04535_ballard2023_tspl_jackpot/
  code/tspl_jackpot_autolens.py  (new)
  code/generate_mock_tspl_jackpot.py  (new)
  code/tspl_jackpot_architecture_check.py  (new)
  data/mock/{image,noise_map,psf}.fits + mock_truth.json  (new)
  submit_tspl_jackpot.slurm  (new)

private/2307_09271_li2023_cosmography_population/
  code/cosmography_li23_autolens.py  (new)
  code/generate_mock_population.py  (new)
  code/cosmography_architecture_check.py  (new)
  code/per_system_likelihood.py  (modified — Om0/w0 wiring)
  code/population_model_autofit.py  (modified — Om0/w0 instance wiring)
  submit_cosmography_li23.slurm  (new)

private/2303_15514_nightingale2023_abell1201/
  code/a1201_lens_model.py  (modified — --extended-break-prior flag)
  code/chain_priors_from_lp.py  (patched — 2nd-level dict unwrap)
  code/compute_bayes_matrix.py  (modified — lp_v5/lp_f390w/with_kin_v2 rows)
  code/extract_mbh.py  (patched — weighted percentiles, FactorGraph prefix, --batch, 95% UL)
  code/build_fig5_mbh_comparison.py  (new)
  submit_a1201.slurm  (modified — EXTENDED_BREAK_PRIOR flag)
  results/figures/fig5_mbh_comparison_2026_05_27.{pdf,png}  (new)
  results/M_BH_COMPARISON_TABLE_2026_05_27.md  (new)
  results/mbh_batch_2026_05_27.json  (new)
  NOTES_v5chain_bugs_2026_05_27.md  (new)
  + 8 pulled output dirs (lp_v5, with_smbh_v5chain, BPL ×3, F390W ×2, with_kin_v2)
```

## Net progress on paper-repro plan checkboxes

| Spec | Before today | After today |
|---|---|---|
| 00 | 0/N | scaffolding in flight (subagent) |
| 01 | 0/23 | architecture wired, smoke queued, blocked on Spec 00 for production |
| 02 | 0/25 | architecture wired, smoke queued |
| 03 | 0/13 | 5+/8 done, Stage 1 queued |
| 05 | 2/63 | 3 bugs fixed, Fig 5 shipped, v5chain v2 queued, BPL extbreak queued |

## Curriculum state

| Module | Before | After |
|---|---|---|
| 13 | "in progress" (stubbed) | ✓ shipped |
| 14 | "in progress" | unchanged |
| 15 (combined) | shipped (legacy) | retained for provenance |
| 15a (analytic) | did not exist | ✓ shipped |
| 15b (PyAutoLens) | did not exist | ✓ shipped |

## Open scientific questions

1. **F390W non-detection vs N+23's +100 ΔlnZ** — data-prep mismatch?
   Compare noise map + PSF + central-BCG masking to N+23's published
   pipeline outputs.
2. **Why kinematics didn't break the γ′-M_BH degeneracy** in our
   parametric-source baseline — likely needs pixelised+Adapt source
   first; `adapt_3sersic` (job 16376586) is the test when it lands.
3. **BPL r_break behavior with extended prior** — does it collapse to
   ≤10 pc (DESJ0206 pattern, confirming BPL is a disguised SMBH at
   A1201)? `16341076` test.
4. **Does v5chain v2 (with chain bug fixed) land in N+23 mode or
   wrong-mode** — confirms or refutes the methodology decision to retire
   the lp_v5 baseline.
