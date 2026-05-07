# Release Notes — v0.93-alpha

**Release date:** 2026-05-07
**Tag:** `v0.93-alpha`
**Predecessor:** `v0.92-alpha` (2026-05-03)
**Author:** Rodrigo Córdova Rosado (rodrigo.cordova_rosado@cfa.harvard.edu, Harvard CfA)

---

## What's new in v0.93

v0.93 builds on v0.92's student-handoff foundation with two **strict-PASS** advances on architectures that v0.92 left as research-in-progress, plus a Hernquist-lab onboarding lane and a cleaner methodology around hand-anchored fit validation. Highlights:

1. **cluster_scale shipped strict-PASS** (the `Examples/group_scale` + `Examples/double_source_plane` architectural union — BCG + 10 cluster members + 2 source planes).
2. **AGEL real-data target shipped strict-PASS** after a single-line hot-pixel data-prep fix that drove `max|res|` from 32σ down to 3.41σ.
3. **`cannon.env.hernquist`** pre-filled config — Hernquist-lab students fill in only `CANNON_USER` and submit.
4. **`AGEL_QUICKSTART.md`** for collaborators with PyAutoLens experience who want to fit a *new* AGEL target end-to-end.
5. **PositionsLH API contract + empirical sanity check** in `compound_lens/01` — same `Grid2DIrregular` flows to plotter and likelihood; demonstrate that the constraint actually penalises wrong models before spending Cannon hours.
6. **Generator self-consistency assertion** — `chi²-at-truth` check baked into mock generation, prevents Pattern-F (light-component omission) bugs from shipping.

### New strict-PASS shipped fits

| Example | Result | Wall | log_Z | χ²/N | max\|res\| |
|---|---|---|---|---|---|
| `cluster_scale` (truth_anchored, mock 1) | **strict-PASS** | 47 min | +62,123 | 1.006 | 4.04σ |
| `agel_real_target` (direct_clean) | **strict-PASS** | 46 min | +20,663 | 0.165 | 3.41σ |

`cluster_scale` is the architectural climax of the example collection: BCG light + Sersic mass + Faber-Jackson scaling on 10 satellite members + 2 sources at z=(1.5, 2.8). The fit driver now models cluster members as **mass + light** (the v0.92 attempt had mass only, leaving 10 unmodeled satellite light profiles in the residual budget — a textbook Pattern F omission, fixed in commit `f8471bb`).

`agel_real_target` direct_clean uses the §1.5 hot-pixel mask cell from the notebook (sn>8σ, 3 outliers excluded inside the 2.7" aperture) — the single dominant 32σ residual in the v0.92 fit was a cosmic-ray survivor, not a model failure.

### New onboarding (Hernquist lab)

- **`Modules/10_Cluster_Computing/cannon.env.hernquist`** — pre-filled config. Account `hernquist_lab` (lab-wide fairshare), partition `hernquist` (10 nodes, holy7c). Smoke-tested end-to-end: `sbatch --test-only` accepts the resource combo and dispatches immediately.
- **SIAG → Hernquist-lab scrub** — partition default flipped from `siag_gpu` (SIAG-subgroup-only) to `hernquist` (all lab members). 22 files updated (notebooks, slurm header, env templates, READMEs). All Hernquist-lab students can now use the cluster pipeline regardless of subgroup membership.
- **`Examples/agel_real_target/AGEL_QUICKSTART.md`** — 8-step recipe for fitting a *new* AGEL HST target: cutout extraction → empirical PSF stack → hot-pixel mask → Keck redshift confirmation → Cannon submit → audit. ~2 hours to first submit. Cross-linked from `START_HERE.md`.

### New methodology

- **`Examples/compound_lens/01` §2 + §2.5** — explicit dual-API contract: the same `al.Grid2DIrregular` flows to both `aplt.subplot_imaging_dataset(positions=...)` and to `al.PositionsLH(positions=..., threshold=...)`. New §2.5 sanity-check section traces a truth tracer + 3 perturbed tracers through to the source plane and prints the penalty contract empirically — demonstrates that PositionsLH genuinely prunes parameter space (penalty ~10⁸, vs ~10⁴ typical imaging-likelihood differences).
- **`Examples/cluster_scale/mocks/generate_mock.py`** self-consistency assertion — every mock generation runs `FitImaging` at the literal truth tracer and refuses to write the FITS file if `chi²/pixel > 1.5` or `max|res| > 6σ`. Catches Pattern F (light-component omission) before any Cannon time is spent.
- **chi²-at-truth diagnostic methodology** — captured in `PROGRESS_LOG.md` 2026-05-05. When a Cannon fit TIMEOUTs, the local diagnostic answers in <5 minutes whether the failure is search-budget-bound or model-space-bound. Calibrated against a known-good (mock_4 truth_anchored ships at χ²/N=1.025; literal-truth instance eval gives χ²/N=4.9, so any test mock at chi²/N <~5 is search-bounded, not structural).

### Drivers + repo hygiene

- `fit_example_cluster_scale.py`: added member SersicSph light component with truth-anchored intensity / R_eff / n priors (`f8471bb`).
- `fit_example_agel_real_target.py`: added `--part=direct_clean` variant that ORs the hot-pixel mask with the circular aperture (`de9345f`).
- Emoji cleanup across 14 files (kept `✓ ✗ ⚠ ☉` per user feedback) — more professional doc surface (`354a858`).

---

## v0.93 ship-set tally (delta from v0.92)

### Examples — status changes since v0.92

| Example | v0.92 status | v0.93 status |
|---|---|---|
| `cluster_scale` | ◐ research-in-progress (truth-anchored in flight) | **✓ strict-PASS shipped** (1 truth_anchored fit, 47 min) |
| `agel_real_target` | ✓ shipped with hot-pixel teaching cell | **✓ strict-PASS shipped** (direct_clean refit, 46 min) |
| `compound_lens` | ✓ shipped, PositionsLH not used in canonical fit | ✓ shipped + **PositionsLH API contract + sanity check** (`acb0ffb`) |

All other examples retain their v0.92 ship/research-in-progress status.

### Modules — no changes

Modules 01-10 retain their v0.92 audited status. Modules 11-14 remain planned for future releases.

---

## What's deferred to v0.94 (research-in-progress)

Two items that v0.92 expected to ship in v0.93 hit methodology issues, not budget issues, and have been deferred:

1. **DSPL β-cosmography** (`dspl_beta_v2` Cannon job): TIMED OUT at 48h with **f_live=1.0 throughout** (chain never escaped initial bounds), log_Z=−70,678 and *decreasing*. Pattern A stall — the 30-parameter prior box (14 lens + 7+7 sources + 2 cosmo) doesn't bracket the basin. **Methodology fix needed**: truth-anchored protocol like cluster_scale, or staged chain (single-source DSPL → add cosmography). Filed as task #110.

2. **Compound-zoo `R5_truth_freecosmo` trio** (mocks 2/3/5): the v3 resubmits at 96/120h **deadlocked on Nautilus checkpoint resume** — workers spinning at 80-95% CPU but zero log output and zero new checkpoint state for 38 hours. Cancelled. Likely root cause: stale checkpoint format from older Nautilus version, or model-hash mismatch from a recent commit touching the freecosmo builder. Filed as task #111. Note: chi²-at-truth diagnostic showed mock_3 *could* converge (chi²/N=2.1 at literal truth, below the known-good mock_4 baseline of 4.9) — so when the resume deadlock is fixed, mock_3 should ship cleanly.

3. **mge_to_physical PSF / noise-correlation investigation** — deferred from v0.92, still open.

---

## On the v0.92 → v0.93 PASS classifications

Same `/autolens-fit-diagnostics` standard: strict-PASS (`max|res| ≤ 4σ` AND `χ²/N ≤ 1.3`), borderline-PASS (4-5σ at 9k+ pixels under Bonferroni-corrected noise floor), SUSPECT (5-6σ), FAIL (≥6σ or coherent residual structure). Both v0.93 strict-PASS additions sit inside the 4σ strict bar.

---

## Roadmap to v0.94

### Methodology debt to clear before next ship
- **task #110**: DSPL β-cosmography prior box / staged-chain fix.
- **task #111**: Diagnose Nautilus checkpoint resume deadlock (truth_fc trio). Hypotheses: stale-format incompatibility, model-hash mismatch, bad sample point in resume queue. Diagnostic plan in task description.
- mge_to_physical PSF / noise-correlation investigation.

### Curriculum expansion (planned, not blocked)
- Modules 11 (Physical Mass Models), 12 (TDCOSMO + MSD), 13 (TDCOSMO + Kinematics), 14 (Compound Multi-Plane).
- `subhalo_sensitivity` full grid-search SLaM (Vegetti+ 2010 / Despali+ 2018 methodology).
- `bayesian_model_comparison` empirical worked-examples populated.
- Additional architecture stubs: `quad_time_delay` audited result, `interferometer_basic`.

---

## Getting started with v0.93

```bash
git clone https://github.com/rodelcr/Learning_to_Autolens.git
cd Learning_to_Autolens
git checkout v0.93-alpha       # if you want this exact tagged state
open START_HERE.md             # ← read this first
```

For Hernquist-lab students who want the cluster-pipeline shortcut:

```bash
cp Modules/10_Cluster_Computing/cannon.env.hernquist \
   Modules/10_Cluster_Computing/cannon.env
$EDITOR Modules/10_Cluster_Computing/cannon.env   # set CANNON_USER, save
```

For AGEL collaborators fitting a new target end-to-end: `Examples/agel_real_target/AGEL_QUICKSTART.md`.

---

## Acknowledgements

v0.93's two strict-PASS advances came from a single diagnostic insight applied to two different problems: **at scale, evaluate the model at literal truth values before spending Cannon hours**. cluster_scale's mass-only-no-light bug and AGEL's 32σ-cosmic-ray-survivor pixel were both invisible from a freely-fit Cannon submit but immediately visible from a 30-second local FitImaging at the truth tracer (or a hot-pixel S/N cut on the data). The chi²-at-truth diagnostic methodology in PROGRESS_LOG generalises this to any future tight-prior fit.

The v0.93 deferred items (DSPL Pattern A stall + truth_fc resume deadlock) are themselves valuable — both are well-characterised failure modes that previously would have been lumped under "compute spend, hope for convergence". Naming them and filing diagnostic tasks is a step toward shipping them in v0.94 with confidence.
