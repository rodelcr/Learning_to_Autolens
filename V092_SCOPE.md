# v0.92-alpha Scope

> **Discipline:** v0.92 ships only material that has been **audited PASS or borderline-SUSPECT-but-clean** (residuals at the noise floor, no coherent structure in the residual map, parameters reasonable, students can reproduce). Everything else moves to "research-in-progress" status — visible in the repo but flagged so students don't expect to reproduce it.

> **On the "PASS" threshold.** The autolens-fit-diagnostics strict bar is `max|res| ≤ 4σ` AND `χ²/N ≤ 1.3`. That bar was calibrated on the curriculum's small test mocks (~3000 unmasked pixels). For larger fits (9k+ pixels), the **Bonferroni-corrected expected max under pure white noise** is √(2·ln(N)) ≈ 4.3σ at 9000 pixels — so a max|res| of 4-5σ is consistent with the noise floor, not a fit failure. v0.92 ships fits in the 4-5σ band as **borderline-PASS** when the residual map is visually clean (no coherent ring/cross structure). Strict-PASS (≤4σ) and borderline-PASS (4-5σ) are both shipped; everything ≥5σ is research-in-progress unless visually clean.

## What ships in v0.92

### Curriculum modules (Modules/)

| # | Module | Status | Why it ships |
|---|---|---|---|
| 01 | Basics: Grids, Galaxies, Ray-Tracing | ✓ ship | Foundation; runs clean on laptop in seconds |
| 02 | Simulating Lens Data | ✓ ship | Foundation; runs clean |
| 03 | Your First Lens Model | ✓ ship | Audited; converges in <2 min on laptop |
| 04 | Search Chaining & SLaM Pipeline | ✓ ship | Cannon-ready; SLaM/source_pix audited PASS |
| 05 | Pixelized Source Reconstructions | ✓ ship | search2_pixelized: χ²/N=1.02, max\|res\|=5.73σ (borderline-PASS at 9000-pixel scale) |
| 06 | Multi-Component Mass Models | ✓ ship | composite_mass: χ²/N=1.05, max\|res\|=4.54σ (borderline-PASS) |
| 07 | Real Data: FITS to Model | ✓ ship | Loader + masker validated |
| 08 | Results, Diagnostics & Figures | ✓ ship | All cells render |
| 09 | MGE & Linear Light Profiles | ✓ ship | source_pix[2]: χ²/N=0.87, max\|res\|=3.75σ (strict-PASS) |
| 10 | Cluster Computing on Cannon | ✓ ship | + STUDENT_QUICKSTART.md + RECIPES.md |

**Mods 11-14 (curriculum-listed)**: 11=Physical Mass Models is **planned**; 12-14 (TDCOSMO, kinematic TDCOSMO, multi-plane) are **roadmap-only**. → All four **defer**.

### Bridge climb notebooks (`00_climb_to_*`)

All four ship — they're skip-guarded API walkthroughs that render in <60s, no Cannon dependency:
- `Examples/compound_lens/00_climb_to_compound.ipynb`
- `Examples/compound_lens_zoo/00_climb_to_compound.ipynb`
- `Examples/double_source_plane/00_climb_to_dspl.ipynb`
- `Examples/group_scale/00_climb_to_group.ipynb`

### Recipe notebooks (v0.92's main pedagogical addition)

- `Modules/05_Pixelized_Source_Reconstructions/06_pixelization_recipe.ipynb` ✓
- `Modules/09_MGE_Linear_Light_Profiles/05_mge_recipe.ipynb` ✓
- `Examples/compound_lens_zoo/03_slam_recipe.ipynb` ✓

### Student-facing cluster docs

- `Modules/10_Cluster_Computing/SETUP_NEW_USER.md` ✓ (one-time)
- `Modules/10_Cluster_Computing/STUDENT_QUICKSTART.md` ✓ (daily loop)
- `Modules/10_Cluster_Computing/RECIPES.md` ✓ (sbatch one-liners)

### Examples — converged subset

| Example | Ships in v0.92 | What's NOT shipping |
|---|---|---|
| **`compound_lens`** | ✓ all 3 notebooks + compound_direct_fit (χ²/N=0.69, max\|res\|=4.40σ — borderline-PASS) + 3 SLaM staged fits | — |
| **`compound_lens_zoo`** | ✓ R0/R2/R3/R5 ladder *as a pedagogical exercise* — `02_compound_lens_ladder.ipynb` §1-12 documents the climb across 5 mocks, including which mocks converge cleanly (mock_3 borderline SUSPECT at R2; mocks 4-6 FAIL at R2 → improve up the ladder but mock_4/6 still SUSPECT/FAIL at R5). The pedagogical value is the *negative-result analysis* (Pattern A/E catalogue), not a "5-of-5 PASS" claim. | §13-§15 (truth + staged + freecosmo): research-in-progress |
| **`double_source_plane`** | ✓ direct fit (χ²/N=0.99, max\|res\|=3.90σ — strict-PASS) + climb notebook | β-cosmography (in flight as `9727096 dspl_beta_v2`) |
| **`disky_spiral_lens`** | ✓ Bayes-factor demo + bulge_disk_fit (χ²/N=1.00, max\|res\|=4.21σ — borderline-PASS) + the deliberately-failed single_sersic fit as the comparison case | — |
| **`group_scale`** | ✓ truth_anchored result (χ²/N=1.025, max\|res\|=4.50σ — borderline-PASS) + climb notebook | freely-fit + staged_satellites + SLaM all FAIL — research-in-progress |
| `mge_to_physical` | ◐ pedagogy only | v2 fits show improvement (1.87 χ²/N) but max\|res\| 9.7σ still SUSPECT — flag as "stars+DM Bayes-factor demo with caveats" |
| `agel_real_target` | TBD | Pending hot-pixel fix; if residuals clean up after data-prep, ships |

### Examples — DEFERRED to v0.93+

| Example | Why deferred |
|---|---|
| `cluster_scale` | direct FAIL; truth_anchored attempt 1 had source-prior bug (now fixed); attempt 2 in flight as `9882023 cluster_truth_v2`. Architecture is in the roadmap with mock + minimal notebook + driver, but no audited result yet. |
| `quad_time_delay` | Notebook + driver scaffolded; results pending Cannon audit. |
| `subhalo_sensitivity` | Notebook + 2 fits committed (smooth + with_perturber) but not audited against Vegetti+10 baseline. |
| `interferometer_basic` | Notebook + driver; no audited result. |
| `bayesian_model_comparison` | Pedagogy-only notebook with placeholder tables — needs P3+P4 results to fill. |

## Status banners (added to each in-progress README)

Every ◐ example's README gets a banner at the top:

```markdown
> **v0.92 ships:** [what's in]
> **Research in progress (NOT in v0.92):** [what's deferred + link to its HANDOFF entry]
```

This keeps the *visible* surface honest: students see only what works as a learning experience; the rest is clearly labelled as research-frontier material that may or may not work on their machine.

## What I'm running on Cannon to close v0.92

Currently in flight (as of 2026-05-03):
- `9882023 cluster_truth_v2` — second attempt at cluster truth-anchored after fixing source ell_comps prior. If PASS, cluster_scale moves into v0.92's "shipping" list. If FAIL, it stays deferred.
- `9727090/92/94 truth_fc_m{3,2,5}_v2` — R5_truth_freecosmo on compound mocks (48h budget). Their results inform the §15 cosmology narrative but don't gate v0.92 — that section is already labeled "research in progress" for the ladder notebook.
- `9727096 dspl_beta_v2` — DSPL β-cosmography (48h). If PASS, ships as `02_beta_cosmography.ipynb`'s converged result. If TIMEOUT, stays in research-in-progress.

After these land + the hot-pixel fix on AGEL, v0.92 is taggable.

## Tagging discipline

When tagging:
1. The git tag message lists the ship-set above, with audit verdicts inline
2. The ◐ items are NOT touched in the tag — they continue in the working branch as research-in-progress
3. The next release (v0.93) is the next time we cross a threshold of converged additions

## Estimated days to v0.92 tag

- 0 days: cluster docs + recipe notebooks (DONE)
- 0 days: status banner edits (this session, after this doc)
- 1 day: cluster_truth + dspl_beta land
- 2 days: truth_freecosmo trio land
- 0.5 day: AGEL data-prep cleanup
- 0.5 day: final audit pass + tag

**ETA: 2026-05-05 / 2026-05-06.**
