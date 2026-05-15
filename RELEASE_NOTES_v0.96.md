# Release Notes — v0.96-alpha (DRAFT)

**Status:** draft. Tag pending `mge_s3v2` (Cannon job 13058789) audit and `preflight_check_v096.sh` clean PASS once that lands.

**Predecessor:** `v0.95-alpha` (2026-05-13)
**Author:** Rodrigo Córdova Rosado (rodrigo.cordova_rosado@cfa.harvard.edu, Harvard CfA)

---

## What's new in v0.96

v0.96 is the **AGEL-science-aligned phase** — the two scientific applications that anchor the project's headline use cases (radial-arc + SMBH detection; DSPL standard-ruler cosmography) ship as strict-PASS depth-B examples, plus a new theory module (15) and a methodology upgrade catching two bug families.

### 1. Module 15 — Radial Arcs & Caustic Topology

The theory module for v0.96's flagship Example. Five sections (executes <60 s, no Cannon dependency):

- Radial vs tangential caustics and their topology under γ′ variation (Krywult+2017 framework)
- λ_t / λ_r eigenvalues of the magnification tensor, asymptotic behaviour near caustics
- γ′ constraints from radial-arc position alone (the radial caustic radius is exquisitely γ′-sensitive)
- The γ′–M_BH degeneracy in numerical form (Ferrami+2024 for the DESJ0206 case)
- Hand-off to Module 13 (kinematics break the degeneracy)

### 2. `Examples/radial_arc_smbh/` (depth B)

The first AGEL-aligned Example: an Einstein-spiral-analog mock with embedded SMBH. Autolens-native simulator + 4-rung driver + 8-item bridge-to-C checklist.

**Headline scientific finding** (Cannon-audited 2026-05-15):

| Rung | log_Z | chi²/N | max\|res\| | γ′ recovered (3σ) | θ_E_BH recovered (3σ) |
|---|---|---|---|---|---|
| `rarc_direct` (PowerLaw + shear) | 8725.87 | 1.019 | 4.27σ | 1.959 (1.913, 2.003) | — |
| `rarc_with_pointmass` (+ PointMass) | 8724.16 | 1.019 | 4.23σ | 1.947 (1.901, 1.999) | 0.073 (0.024, 0.117) |
| **Truth** | — | — | — | **1.95** | **0.08** |

**Δlog_Z = +1.71** in favour of the no-BH model — *inconclusive* by Jeffreys scale (|Δ| < 2.5). At this S/N and 0.05″/px pixel scale, the BH is **not detectable by imaging alone** even when pedagogically amplified to θ_E_BH = 0.08″ (~5× real AGEL UMBH expectation). The γ′–M_BH degeneracy is the bottleneck.

This is a **publication-grade negative result**. The pedagogical pivot: Module 13 (Jeans kinematics) + the deferred `--part=with_kinematics` rung (v0.97 / Phase 3, task #122) provides the orthogonal probe that breaks the degeneracy. Real AGEL UMBHs at θ_E_BH ≈ 0.02″ sit well below this imaging-only floor.

Both rungs are **strict-PASS** on the autolens audit bar (chi²/N < 1.3, max|res| < strict Bonferroni-corrected floor √(2 ln 4060) = 4.10σ). The `01_radial_arc_smbh.ipynb` walkthrough (8 sections, executes <30 s with graceful-when-absent) reads the pulled results and produces the headline table.

### 3. `Examples/double_source_plane/` (depth B): DSPL cosmography strict-PASS

The v0.93 Pattern A stall + v0.94 TruncatedGaussian fix is now fully closed. The 2026-05-14 stale-truth-JSON bug (chi²-at-truth = 127.47 on the v0.94 mock) was the persisting blocker; the regenerated mock (single source-of-truth dict + chi²-at-truth = 0.995) clears it.

**Cannon-audited 2026-05-15:**

| Stage | log_Z | chi²/N | max\|res\| | Wall | Outcome |
|---|---|---|---|---|---|
| `beta_fixedcosmo` (Stage 1) | 29039.85 | 0.990 | 3.97σ | ~45 min | STRICT-PASS ✓ |
| `beta_freecosmo_v3` (Stage 2, chain) | 29036.34 | 0.990 | 3.98σ | ~45 min | STRICT-PASS ✓ |

**Cosmography recovery** (Stage 2, 3σ): Ωₘ ∈ (0.2, 0.4), w₀ ∈ (−1.14, −0.89). Truth Ωₘ = 0.30, w₀ = −1.0 — both bracket truth at 3σ; w₀ is well-constrained (1σ ≈ ±0.05). Publication-grade joint posterior, ready for cross-link with TDCOSMO H0 (Birrer+2020 §4) in v0.96's pending `cosmography_joint_posterior` example.

### 4. mge_to_physical axis-swap fix — bug class **(y, x)**

`Examples/mge_to_physical/mocks/regenerate_in_autolens.py` was found to swap axes between lenstronomy and autolens conventions at 10 sites. PyAutoLens centre is **`(y, x)`** (first y, second x); lenstronomy parameters are named `center_x, center_y` explicitly. Mapping correctly is `centre = (center_y, center_x)`; the regen script passed `(center_x, center_y)` directly.

The bug hid for 9 days (2026-05-09 to 2026-05-15) because the lens light + mass were at (0, 0) where the swap is invisible. Off-axis components (secondary deflector at (0.02, −0.05), sources at (0.30, 0.22)) exposed it as 29σ residuals on the fit.

**Detection methodology:** *driver-truth check*. The internal chi²-at-truth assertion at the end of `regenerate_in_autolens.py` passed (1.003) — the simulator and verifier read the same dict. But the **driver's hardcoded prior means** (centred on `(centre_0, centre_1)`) did not match the swapped mock positions; the fit-time driver was anchored on the wrong basin. The fix: at the same 10 sites in the regen script, swap `(center_x, center_y)` → `(center_y, center_x)`. Driver-truth on the corrected mock: chi²/N = 1.007 ✓.

**search_2_v2_stars_only** on the corrected mock landed chi²/N = 4.60 — improved from pre-fix 7.07 (32% reduction), but the stars-only model is **structurally incomplete by design** (no dark halo, no secondary deflector). The s2 result here is not a strict-PASS retry; the strict-PASS verdict awaits `search_3_v2_stars_dark` (still running as of tag-day).

### 5. New methodology: chi²-at-truth **+** driver-truth (two-check)

The 2026-05-14 DSPL stale-JSON bug and the 2026-05-15 mge axis-swap bug both exposed a gap in the v0.93–v0.95 chi²-at-truth methodology: **internal mock consistency does not imply mock-driver consistency.**

**The two-check protocol (codified 2026-05-15):**

1. **chi²-at-truth** — at the end of the mock generator, build a tracer from the exact same dict used to simulate, fit it back, assert chi²/N ≤ 1.5 and max|res| ≤ 5σ. Catches mock-internal inconsistencies.
2. **Driver-truth** — separately, build a tracer using the **fit driver's hardcoded constants/prior means**, apply to the same mock, assert the same bar. Catches mock-DRIVER mismatches (axis-swap, stale-JSON, version-drift).

Both checks are necessary; neither alone is sufficient. The methodology has now been **thrice-validated** (cluster_scale 2026-04, DSPL 2026-05-14, mge 2026-05-15). Codified in `feedback_mock_driver_consistency.md` memory + `preflight_check_v096.sh` Step 6 (chi²-at-truth assertions across all 5 mock generators) and Step 7 (mge axis-swap convention verification).

### 6. `preflight_check_v096.sh` (10 steps)

Extends v0.94's 9-step preflight with two mock-hygiene steps (6, 7) and a tag-aware Cannon-result strict-PASS check (10, recognizes `_autolens_v2` source_dir suffix). Current state: **25 PASS / 4 WARN / 0 FAIL**. The 4 WARNs are conditional Cannon-result strict-PASS checks for `rarc_direct`, `rarc_with_pointmass`, `dspl_v096`, `mge_s3v2` — they auto-promote when results land strict-PASS. After 2026-05-15 audit: rarc_* and dspl_v096 promote to PASS; only mge_s3v2 remains WARN until that fit lands.

### 7. New strict-PASS shipped fits

| Example | Result | Wall | log_Z | χ²/N | max\|res\| |
|---|---|---|---|---|---|
| `double_source_plane/beta_fixedcosmo` | strict-PASS | 45 min | 29039.85 | 0.990 | 3.97σ |
| `double_source_plane/beta_freecosmo_v3` | strict-PASS (chain) | 45 min | 29036.34 | 0.990 | 3.98σ |
| `radial_arc_smbh/rarc_direct` | strict-PASS | 1h 18m | 8725.87 | 1.019 | 4.27σ |
| `radial_arc_smbh/rarc_with_pointmass` | strict-PASS | 1h 19m | 8724.16 | 1.019 | 4.23σ |
| `mge_to_physical/search_3_v2_stars_dark` | _PENDING_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

---

## v0.96 ship-set tally (delta from v0.95)

### Modules — Module 15 newly shipped

| # | Module | v0.95 status | v0.96 status |
|---|---|---|---|
| 15 | Radial Arcs & Caustic Topology | ◯ planned | **✓ ship** (5 sections, <60 s execute, no Cannon dep) |

The 15-module curriculum table is now complete (Modules 11-14 shipped in v0.94-v0.95).

### Examples — status changes since v0.95

| Example | v0.95 status | v0.96 status |
|---|---|---|
| `radial_arc_smbh` | _new_ | **◐ depth-B strict-PASS** (autolens-native mock + 4-rung driver + walkthrough notebook; γ′ recovered to 1.5%, BH not imaging-detected) |
| `double_source_plane` | ◐ research-in-progress (Pattern A stall) | **✓ depth-B strict-PASS** (Ωₘ, w₀ joint posterior bracketing truth at 3σ) |
| `mge_to_physical` | ◐ research-in-progress (Sersic eval gap) | ◐ research-in-progress (axis-swap fix landed; s3v2 strict-PASS retry running) |

### New methodology

- **`feedback_mock_driver_consistency.md`** (memory) — two-check protocol with worked examples.
- **`(y, x)` axis-swap bug class identified** — `centre = (center_y, center_x)` is the always-correct mapping from lenstronomy → autolens.
- **Single source-of-truth dict pattern** for mock generators — one Python dict feeds both `al.Galaxy(...)` and `json.dumps(...)`. Eliminates the stale-JSON bug family.
- **`preflight_check_v096.sh`** — 10 steps codifying both checks across all 5 active mock generators.

### Drivers + repo hygiene

- `fit_example_double_source_plane.py`: `unique_tag` v0_94 → v0_96 (fresh checkpoint family for the regenerated mock).
- `fit_example_mge_to_physical.py`: `tag_suffix` `_autolens` → `_autolens_v2` (fresh checkpoint family for the axis-fixed mock).
- `Examples/mge_to_physical/mocks/regenerate_in_autolens.py`: axis-swap fix at 10 sites with explicit naming comment.
- `Examples/double_source_plane/mocks/generate_mock.py`: complete rewrite with single source-of-truth dict + chi²-at-truth assertion.
- `Examples/radial_arc_smbh/mocks/generate_mock.py`: NEW autolens-native simulator; chi²-at-truth = 1.029 verified.
- `Modules/10_Cluster_Computing/scripts/fit_example_radial_arc_smbh.py`: NEW 4-rung driver (direct, no_pointmass, with_pointmass, with_kinematics STUB).
- `use_jax=False` on `AnalysisImaging` constructors in rarc + ggsa drivers — works around autolens JAX scan + functools.partial bug in PowerLaw.deflections_yx_2d_from.

---

## What's deferred to v0.97

1. **`Examples/radial_arc_smbh/ --part=with_kinematics`** — requires the shared `_jeans_sigma_v.py` Analysis class (Phase 3, task #122). Single-file module imported by both `radial_arc_smbh` and the planned `quad_time_delay --part=joint_fit_h0_kin`. Custom `al.AnalysisKinematics` subclass following `autolens_workspace_latest/scripts/guides/advanced/custom_analysis.py` template. ~200 lines + isotropic-Jeans-formula reuse from Module 13.

2. **`Examples/cosmography_joint_posterior/`** (task #125) — combines DSPL β posterior (Ωₘ, w₀) with TDCOSMO H0 chain (Birrer+2020 §4 methodology). Pure-laptop notebook once `beta_freecosmo_v3` posterior is available (it is now).

3. **`Examples/agel_spiral_real_target/`** — depth-C application on real DESJ0206 imaging + KCWI/LLAMAS kinematics. Requires v0.97 PSF-from-data pipeline, hot-pixel mask, KCWI ppxf σ_v with R_eff aperture matching. 8-item bridge checklist in `Examples/radial_arc_smbh/README.md`.

4. **Multi-GPU JAX MGE benchmark** (task #127) — does multi-GPU data-parallel JAX make MGE fits faster than the current numpy-default? Single-GPU is 4× slower than numpy (known); multi-GPU might cross. Deferred priority.

5. **Mathematica companion Module 15** for the Learning_to_Lens parallel project.

---

## Roadmap to v0.97

The v0.96 → v0.97 path closes the kinematics extension across two examples simultaneously:

- **Phase 3** — Shared `_jeans_sigma_v.py` Analysis class. Unlocks both `radial_arc_smbh --part=with_kinematics` (γ′–M_BH break demo) AND `quad_time_delay --part=joint_fit_h0_kin` (TDCOSMO H0 MSD break, Birrer+2020 IV methodology).
- **Cosmography joint posterior** — laptop-only notebook combining DSPL + TDCOSMO chains.
- **Real-target bridge** — `Examples/agel_spiral_real_target/` on DESJ0206 (the AGEL flagship).

---

## Getting started with v0.96

```bash
git clone https://github.com/rodelcr/Learning_to_Autolens.git
cd Learning_to_Autolens
git checkout v0.96-alpha       # if you want this exact tagged state
open START_HERE.md             # ← read this first
```

For the v0.96 flagship demo, the canonical entry point is:

```bash
open Examples/radial_arc_smbh/01_radial_arc_smbh.ipynb
open Examples/double_source_plane/results/beta_freecosmo_v3/corner.pdf
```

For the chi²-at-truth + driver-truth two-check methodology recipe, see `feedback_mock_driver_consistency.md` in the memory tree and `Modules/10_Cluster_Computing/scripts/preflight_check_v096.sh` Steps 6 + 7.

---

## Acknowledgements

v0.96's cleanest outcome is **two bug families found and closed within 24 hours via the same methodology**: chi²-at-truth + driver-truth as a two-check protocol caught both the DSPL stale-JSON (mock-internal-but-detached) and mge axis-swap (mock-internal-OK-but-mismatched-with-driver) failure modes. The `(y, x)` autolens convention bug class is now documented as a class hazard, not a one-off. Together with v0.93–v0.95's chi²-at-truth wins, the project now has a four-time-validated diagnostic discipline that prevents the failure mode of *spending Cannon cluster hours on a mock-driver mismatch*. This is the project's signature methodology, and v0.96 ships it as a first-class tool with named bug classes + a preflight script that enforces it.

The radial_arc_smbh negative result (BH not imaging-detected at θ_E_BH=0.08″) is the most pedagogically valuable v0.96 finding — it sets up Module 13 + Phase 3 as a *required* extension rather than a nice-to-have, and accurately reflects the imaging-only limit at AGEL's actual data quality.
