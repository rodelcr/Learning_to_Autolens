# Paper-Repro Spec 04 — Cross-Validation + Promotion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When Specs 01-03 land, weave the per-paper results together: 4-way J0946 consistency (P2 + P3 × both stacks), per-paper autolens-vs-Herculens cross-validation, tool-development synthesis, then promote `private/code/` derivatives into public Modules + Examples and tag `v0.98-alpha`.

**Architecture:** Mostly notebook + docs work (no new Cannon fits required). One synthesis doc (`tool_development_findings.md`), two new public Modules (16, 17), one module extension (Module 14 §TSPL), two Learning_to_Lens additions (09.5, 11), three new public Examples (one per paper).

**Tech Stack:** pandas + matplotlib + corner.py for analysis notebooks; nbconvert for execution gates; Wolfram Mathematica for Learning_to_Lens; standard git workflow for the public-repo promotion.

**Depends on:** Specs 00, 01, 02, 03 substantially complete (per-paper strict-PASS or documented research-in-progress).

---

## File Structure

```
private/04_crossval_promotion/
├── validation/
│   ├── j0946_main_lens_consistency.ipynb       ← 4-way comparison
│   ├── autolens_herculens_crossval.ipynb       ← per-paper summary
│   ├── p1_sampler_crossval.ipynb               ← Nautilus vs NUTS for P1
│   └── tool_development_findings.md            ← synthesis
└── promotion/
    ├── promotion_plan.md
    ├── module_16_outline.md                    ← Hierarchical Bayesian Cosmography
    ├── module_17_outline.md                    ← Dynamical Mass Decomposition
    ├── module_14_tspl_extension_outline.md     ← TSPL Extension to Module 14
    ├── ltl_09_5_outline.md                     ← Hierarchical Cosmography Mathematica
    ├── ltl_11_outline.md                       ← Stellar Dynamics + Jeans Mathematica
    └── HERCULENS_INTEGRATION_draft.md          ← public repo-level doc

# Public-repo additions (promoted from private/):
Modules/16_Hierarchical_Bayesian_Cosmography/
    16_hierarchical_cosmography.ipynb
Modules/17_Dynamical_Mass_Decomposition/
    17_dynamical_mass_decomposition.ipynb
Examples/hierarchical_population_cosmography/
    01_population_inference.ipynb
    README.md
    code/
    mocks/                                       ← synthetic 30-lens for depth-B
    results/
Examples/tspl_jackpot/
    01_tspl_main_fit.ipynb
    02_subhalo_bayes_factor.ipynb
    03_wandering_bh_alt.ipynb
    README.md
    code/
    results/
Examples/dspl_jackpot_imf_nfw/
    01_dspl_jackpot_autolens.ipynb
    02_dspl_jackpot_herculens.ipynb
    03_crossval.ipynb
    README.md
    code/
    results/
HERCULENS_INTEGRATION.md                          ← public repo-level
RELEASE_NOTES_v0.98.md
~/Documents/Learning_to_Lens/Mathematica/09_Galaxy_Lensing_Applications/09.5_Hierarchical_Cosmography.wl
~/Documents/Learning_to_Lens/Mathematica/11_Stellar_Dynamics_Jeans/
    11a_Jeans_Equation.wl
    11b_Aperture_Projection.wl
    11c_Anisotropy_Kernel.wl
```

---

## Phase 1: Cross-paper validation

### Task 1: 4-way J0946 main-lens consistency

**Files:**
- Create: `private/04_crossval_promotion/validation/j0946_main_lens_consistency.ipynb`

The notebook compares main-lens parameters (θ_E, γ′ if free, ell_comps, shear γ_1/γ_2) across four fits:
- P2-autolens (`tspl_main_autolens`)
- P2-herculens (`tspl_main_herculens`)
- P3-autolens (`dspl_jackpot_autolens` Stage 2)
- P3-herculens (`dspl_jackpot_herculens`)

- [ ] **Step 1: Create notebook scaffold**

```bash
mkdir -p private/04_crossval_promotion/validation
mkdir -p private/04_crossval_promotion/promotion
```

- [ ] **Step 2: Write notebook (10 cells)**

Cells (markdown + code interleaved):
1. **Header**: context, goal
2. **Setup**: load all four posterior chains (graceful when absent)
3. **Schema**: list which parameters are present in each posterior (params differ across fits)
4. **4-panel comparison**: for each main-lens param, a 4-way violin plot
5. **Summary table**: median ± 1σ from each chain
6. **Pass/fail verdict**: <2σ agreement on all main-lens params → PASS
7. **Discussion**: where the four disagree, what does it tell us about (a) stack drift, (b) model mis-specification in one paper vs the other, (c) TSPL geometry effects on the main-lens posterior

Concrete code outline for cell 2:

```python
import pandas as pd
from pathlib import Path

REPO = Path('.').resolve()
while not (REPO / 'requirements.txt').exists() and REPO != REPO.parent:
    REPO = REPO.parent

P2A = REPO / 'private' / '2309_04535_ballard2023_tspl_jackpot' / 'results' / 'tspl_main_autolens' / 'samples.csv'
P2H = REPO / 'private' / '2309_04535_ballard2023_tspl_jackpot' / 'results' / 'tspl_main_herculens' / 'samples.csv'
P3A = REPO / 'private' / '2602_20889_li2026_dspl_imf_nfw' / 'results' / 'dspl_jackpot_autolens' / 'stage2' / 'samples.csv'
P3H = REPO / 'private' / '2602_20889_li2026_dspl_imf_nfw' / 'results' / 'dspl_jackpot_herculens' / 'samples.csv'

chains = {}
for label, p in [('P2-autolens', P2A), ('P2-herculens', P2H),
                  ('P3-autolens', P3A), ('P3-herculens', P3H)]:
    if p.exists():
        chains[label] = pd.read_csv(p)
        print(f'{label}: {len(chains[label])} samples')
    else:
        print(f'{label}: MISSING ({p.relative_to(REPO)})')
```

The notebook executes <60s; gracefully handles missing fits by reporting which combinations are present.

### Task 2: Per-paper autolens-vs-Herculens crossval

**Files:**
- Create: `private/04_crossval_promotion/validation/autolens_herculens_crossval.ipynb`

- [ ] **Step 1: Notebook with 6 sections (one per paper × stack pair)**

For each paper (P1, P2, P3), use Spec 00's `crossval_framework.crossval_report` to produce:
- 1D KL divergences per shared parameter
- Joint Bayes factor (marginal-product proxy)
- Side-by-side corner plot
- Markdown summary table

Concrete code outline for one section:

```python
import sys; sys.path.insert(0, '00_shared_infrastructure/code')
from crossval_framework import crossval_report
import pandas as pd

# P2 subhalo crossval
a = pd.read_csv('2309_04535_ballard2023_tspl_jackpot/results/tspl_subhalo_autolens/samples.csv')
h = pd.read_csv('2309_04535_ballard2023_tspl_jackpot/results/tspl_subhalo_herculens/samples.csv')
shared_params = ['perturber.mass.centre.centre_0', 'perturber.mass.centre.centre_1',
                  'perturber.mass.kappa_s', 'perturber.mass.scale_radius']
crossval_report(a, h, params=shared_params,
                labels=('autolens-Nautilus', 'Herculens-NUTS'),
                out_path='p2_subhalo_crossval.md',
                plot_corner=True)
```

Run for all 3 papers; render each report inline.

### Task 3: P1 sampler crossval

**Files:**
- Create: `private/04_crossval_promotion/validation/p1_sampler_crossval.ipynb`

Compare autofit Nautilus vs NumPyro NUTS on the SAME hierarchical model (Spec 01 Tasks 2 + 4). Verify (Ωₘ, w₀, μ_γ, σ_γ) agree at < 0.1σ between samplers.

- [ ] **Step 1: Notebook**

```python
import pandas as pd, sys
sys.path.insert(0, '00_shared_infrastructure/code')
from crossval_framework import crossval_report

nautilus = pd.read_csv('2307_09271_li2023_cosmography_population/results/nautilus_chain.csv')
numpyro = pd.read_csv('2307_09271_li2023_cosmography_population/results/numpyro_chain.csv')
crossval_report(nautilus, numpyro,
                params=['Om0', 'w0', 'mu_gamma', 'sigma_gamma'],
                labels=('autofit-Nautilus', 'NumPyro-NUTS'),
                out_path='p1_sampler_crossval.md', plot_corner=True)
```

### Task 4: tool_development_findings.md synthesis

**Files:**
- Create: `private/04_crossval_promotion/validation/tool_development_findings.md`

- [ ] **Step 1: Write the synthesis doc**

Template (filled with actual numbers from Specs 01-03 once those land):

```markdown
# Tool Development Findings (private/)

Distilled from the cross-paper, cross-stack reproductions of Li+2023, Ballard+2023, Li+2026.

## 1. Performance

| Paper | Stack | Hardware | Wall time | Wall × cores |
|---|---|---|---|---|
| P1 hierarchical (autofit) | CPU | 8 cores | TBD | TBD |
| P1 hierarchical (NumPyro NUTS) | A100 | 1 GPU | TBD | TBD |
| P2 TSPL main (autolens) | CPU | 32 cores | TBD | TBD |
| P2 TSPL main (Herculens NUTS) | A100 | 1 GPU | TBD | TBD |
| P3 DSPL (autolens chain) | CPU | 32 cores | TBD | TBD |
| P3 DSPL (Herculens NUTS) | A100 | 1 GPU | TBD | TBD |

**Expected:** Herculens-on-A100 ~3-10× faster than autolens-on-32-CPU for likelihood-bottlenecked DSPL/TSPL fits. Less advantage for point-source-bottlenecked TSPL (multi-image solver is the slow step).

## 2. Convention drift

Per Spec 00 `dual_stack_conventions.md`, the bridge handles these:
- centre: PyAutoLens (y, x) ↔ Herculens (center_x, center_y) — separate kwargs
- ell_comps: PyAutoLens (ell_y, ell_x) ↔ Herculens (e1=ell_x, e2=ell_y)
- NFW: kappa_s+scale_radius ↔ alpha_Rs+Rs (factor of ~4×scale_radius)
- SHEAR: gamma_1/2 ↔ gamma1/2 + ra_0/dec_0=0

[Additional drift found during Spec 01-03 implementation goes here.]

## 3. API gaps

Features in one stack but not the other (found during implementation):
- PyAutoLens has `lp_basis.Basis` for MGE; Herculens uses MULTI_GAUSSIAN with separate amplitudes
- Herculens has built-in NumPyro integration; PyAutoFit uses Nautilus
- PyAutoFit has FactorGraphModel for joint analyses; Herculens uses NumPyro `factor`

## 4. Sampler stability

[P1 Nautilus vs NUTS: where they agreed, where they drifted. Filled in from Task 3 results.]

## 5. Documentation gaps (PR-targets for upstream)

[Empty until found during implementation.]
```

---

## Phase 2: Public Module promotions

### Task 5: Module 16 (Hierarchical Bayesian Cosmography)

**Files:**
- Create: `Modules/16_Hierarchical_Bayesian_Cosmography/16_hierarchical_cosmography.ipynb`
- Create: `Modules/16_Hierarchical_Bayesian_Cosmography/README.md`
- Modify: `CLAUDE.md` (curriculum table — add row 16)
- Modify: `README.md` (curriculum table)
- Modify: `START_HERE.md`
- Create: `Solutions/16_hierarchical_cosmography_SOLVED.ipynb`

- [ ] **Step 1: Write the notebook (8 cells)**

```bash
mkdir -p Modules/16_Hierarchical_Bayesian_Cosmography
```

Notebook content (Module 16):
1. **What hierarchical Bayes is** (markdown — refs gr-lensing-intuition's distance-ratio + cosmographic-degeneracy sections)
2. **The per-system likelihood** (markdown + code: simplified σ_v from PowerLaw lens, Sersic tracer)
3. **The population layer** (markdown: hyperprior N(μ_γ, σ_γ) on γ_eff_i, marginalisation strategies)
4. **GH-quadrature vs explicit-latent NUTS** — tradeoffs (markdown)
5. **The cosmographic punchline** — per-system info budget (~0.1 nat for SLACS-like lens); N=10^4 needed for σ(w) ≈ 0.11
6. **Demo on 20-lens synthetic sample** (code — pulls private/'s `population_model_autofit.py`)
7. **Hand-off to `Examples/hierarchical_population_cosmography/`** for real-data (Chen+2019) application
8. **Exercises** — 4-5 questions

Executes <60s with synthetic data; no Cannon dependency.

- [ ] **Step 2: README + Solutions stub**

Module 16 README + Solutions follow the v0.96 pattern (see Module 15 for the template).

- [ ] **Step 3: Update curriculum tables**

In `CLAUDE.md`, add row 16:

```markdown
| 16 | Hierarchical Bayesian Cosmography | ✓ ship (v0.98) — population-level (θ_E, σ_v) hierarchical model, GH-quadrature marginalisation, Nautilus vs NUTS dual-sampler demo, hand-off to Examples/hierarchical_population_cosmography for real-data application | Chen+2019, Li+2023 |
```

Same in `README.md`'s curriculum table.

### Task 6: Module 17 (Dynamical Mass Decomposition)

**Files:**
- Create: `Modules/17_Dynamical_Mass_Decomposition/17_dynamical_mass_decomposition.ipynb`
- Create: `Solutions/17_dynamical_mass_decomposition_SOLVED.ipynb`
- Modify: `CLAUDE.md`, `README.md`, `START_HERE.md`

- [ ] **Step 1: Notebook content**

8-10 cells:
1. Stars + DM decomposition via DSPL (markdown)
2. MGE light + ellNFW dark — what they constrain together
3. M/L gradient α — physical motivation
4. gNFW γ_inner — ΛCDM expectation vs adiabatic contraction vs feedback
5. Phase 3 `_jeans_sigma_v.py` recap; β=0 isotropic vs Mamon & Łokas 2005 anisotropy
6. Demo on synthetic mock; recover (M_*, M_h)
7. Hand-off to `Examples/dspl_jackpot_imf_nfw/`
8. Exercises

### Task 7: Module 14 § TSPL Extension

**Files:**
- Modify: `Modules/14_Compound_Multi_Plane_Lensing/14_compound_multi_plane.ipynb`

- [ ] **Step 1: Add a section to the existing notebook**

Append section "§ TSPL Extension" with 4-5 new cells:
1. From DSPL (2 planes) to TSPL (3 planes): how recursive lens eq generalises
2. β_jk distance ratios: 1 for DSPL, 3 for TSPL
3. Multi-caustic topology (Burke's theorem generalisation)
4. J0946 as the worked example: pointers to `Examples/tspl_jackpot/`
5. Cross-link to Module 16 (cosmography from TSPL via β_jk)

### Task 8: Learning_to_Lens additions

**Files:**
- Create: `~/Documents/Learning_to_Lens/Mathematica/09_Galaxy_Lensing_Applications/09.5_Hierarchical_Cosmography.wl`
- Create: `~/Documents/Learning_to_Lens/Mathematica/11_Stellar_Dynamics_Jeans/{11a,11b,11c}_*.wl`

These are Mathematica scripts; symbolic derivations rather than code.

- [ ] **Step 1: Write the Mathematica scripts (skeletons; full derivations are by hand)**

09.5: symbolic derivation of the SIE + isotropic Jeans + thin-lens distance ratio chain that gives σ_v(θ_E, z_l, z_s, cosmology). Fisher information per system.

11a-c:
- 11a — Jeans equation derivation (B&T 2008 §4.215)
- 11b — Aperture-averaged σ_v projection (Mamon & Łokas 2005)
- 11c — Anisotropy kernel β(r) generalisation

---

## Phase 3: Example promotions

### Task 9: `Examples/hierarchical_population_cosmography/` (NEW)

**Files:**
- Create: `Examples/hierarchical_population_cosmography/README.md`
- Create: `Examples/hierarchical_population_cosmography/01_population_inference.ipynb`
- Create: `Examples/hierarchical_population_cosmography/code/` (refactored from `private/2307_…/code/`)
- Create: `Examples/hierarchical_population_cosmography/mocks/` (synthetic 30-lens for depth-B baseline)

- [ ] **Step 1: Refactor private/ code → public-style**

The private code lives at `private/2307_09271_li2023_cosmography_population/code/`. Refactor for public:
- Drop the paper-specific naming (`per_system_likelihood.py` is fine as-is)
- Add docstring linking to Li+2023 + Chen+2019 for citation
- Sanitize any restricted-data hardcoded paths

- [ ] **Step 2: Write 30-lens synthetic mock generator**

The depth-B baseline: a synthetic 30-lens sample with known truth. Generate via `mocks/generate_population.py`:
- 30 SLACS-like lenses with known (θ_E, σ_v, z_l, z_s)
- Hierarchical truth: μ_γ = 2.0, σ_γ = 0.1
- Inject Gaussian σ_v errors
- Save to `mocks/population_30.csv`

- [ ] **Step 3: Walkthrough notebook**

Cells:
1. Header
2. Load the 30-lens synthetic + Chen+2019 161-lens (if available)
3. Build the autofit model
4. Run on synthetic (laptop, ~2 min): recover (μ_γ, σ_γ, w_0, Ω_m) within 1σ
5. Load the 161-lens result if Cannon-pulled; compare to Li+2023
6. Cross-validation: Nautilus vs NUTS posteriors (overlay)
7. Bridge: this is the depth-C application from `private/` work; for the AGEL DR2 ≥5-target population, see §Future Work

### Task 10: `Examples/tspl_jackpot/` (NEW)

**Files:**
- Create: `Examples/tspl_jackpot/README.md`
- Create: `Examples/tspl_jackpot/{01_tspl_main_fit,02_subhalo_bayes_factor,03_wandering_bh_alt}.ipynb`
- Create: `Examples/tspl_jackpot/code/`
- Create: `Examples/tspl_jackpot/mocks/` (synthetic TSPL injection)
- Create: `Examples/tspl_jackpot/results/` (refactored from private/)

- [ ] **Step 1: Refactor private/ TSPL code → public**

Drop the paper-name prefix; clean up code organization. Note: the J0946 HST + MUSE data products themselves likely STAY in private/ (data are not redistributable). The Examples/ shows the methodology on a SYNTHETIC mock + points at private/ for the real-data application.

- [ ] **Step 2: Three pedagogical notebooks**

1. `01_tspl_main_fit.ipynb`: TSPL methodology on synthetic injection
2. `02_subhalo_bayes_factor.ipynb`: with vs without NFW perturber Bayes factor
3. `03_wandering_bh_alt.ipynb`: wandering-BH alternative; compare ΔlogZ

### Task 11: `Examples/dspl_jackpot_imf_nfw/` (NEW)

**Files:**
- Create: `Examples/dspl_jackpot_imf_nfw/README.md` + `code/` + `notebooks/01-03_*.ipynb`

Same refactoring pattern as Tasks 9, 10.

---

## Phase 4: Repo-level docs + release

### Task 12: HERCULENS_INTEGRATION.md

**Files:**
- Create: `HERCULENS_INTEGRATION.md` (repo root, public)

- [ ] **Step 1: Write the doc**

```markdown
# Herculens Integration

PyAutoLens is the curriculum's primary lens-modelling stack. Starting
with v0.98 (2026-06-ish), Herculens (github.com/Herculens/herculens) is
the secondary stack used for:

1. **Cross-validation**: every paper-reproduction Example in v0.98 lands
   in BOTH PyAutoLens (Nautilus on CPU) AND Herculens (NumPyro NUTS on
   A100 GPU). Posterior agreement is a methodology consistency check.
2. **GPU acceleration**: where likelihood evaluation dominates (DSPL,
   TSPL with extended sources), Herculens-on-A100 lands ~3-10× faster
   than PyAutoLens-on-32-CPU.

## Setup

See `private/00_shared_infrastructure/environment-herculens.yml` for
the conda env spec. Install pattern:

    conda env create -f private/00_shared_infrastructure/environment-herculens.yml
    conda activate herculens312
    python -c "import herculens, numpyro, jax; print('OK')"

On Cannon:
    salloc --account=siag_lab --partition=gpu --gres=gpu:a100:1
    source activate herculens312

## When to use which stack

- **PyAutoLens**: pedagogical (curriculum's primary teaching stack);
  best for SLaM-style staged chains; Nautilus is robust on extended
  source fits; matches reference Modules 01-15.
- **Herculens**: when GPU available, for fast NUTS on differentiable
  models; for paper-reproduction where Herculens is the authors' tool
  (P3 Li+2026 case).

## Cross-validation reports

See per-paper validation notebooks at `Examples/*/03_crossval.ipynb`.
```

### Task 13: RELEASE_NOTES_v0.98.md

**Files:**
- Create: `RELEASE_NOTES_v0.98.md`

- [ ] **Step 1: Write release notes**

Headline ships:
- 3 new Examples (hierarchical_population_cosmography, tspl_jackpot, dspl_jackpot_imf_nfw)
- 2 new Modules (16, 17)
- 1 module extension (Module 14 § TSPL)
- 1 new repo doc (HERCULENS_INTEGRATION.md)
- 3 paper reproductions (Li+2023, Ballard+2023, Li+2026)
- Dual-stack (PyAutoLens + Herculens) methodology established

### Task 14: Tag v0.98-alpha

- [ ] **Step 1: Final preflight**

Run a preflight script analogous to `preflight_check_v096.sh`:
- All 3 new Examples have README + main notebook + results dir
- Modules 16, 17 ship strict-PASS notebooks (execute <60s)
- Module 14 has the TSPL § extension
- HERCULENS_INTEGRATION.md exists
- RELEASE_NOTES_v0.98.md committed

- [ ] **Step 2: Tag**

```bash
git tag -a v0.98-alpha -m "v0.98-alpha — dual-stack paper-reproduction phase"
git push origin main v0.98-alpha
```

---

## Self-Review

| Spec 04 section | Tasks |
|---|---|
| §4 architecture | Tasks 1-14 |
| §5 4-way J0946 consistency | Task 1 |
| §6 tool development synthesis | Task 4 |
| §7 promotion plan: 3 Examples | Tasks 9, 10, 11 |
| §7 promotion plan: 2 Modules | Tasks 5, 6 |
| §7 promotion plan: Module 14 ext | Task 7 |
| §7 promotion plan: LtL additions | Task 8 |
| §7 promotion plan: HERCULENS_INTEGRATION | Task 12 |
| §8 cross-references gr-lensing-intuition | embedded in Module 16, 17 outlines |
| §9 sequencing | full plan ~2 weeks |
| §10 risks | docs-only spec, low risk |
| §11 timeline | 14 days |

**Total: 14 tasks. ~2 weeks of laptop work + git ops.**

**Pre-req:** Specs 01-03 must be substantially complete (per-paper strict-PASS or documented research-in-progress) before this spec starts.
