# Audit & Refine — Session Handoff (2026-04-20)

Start-of-session brief for the next Claude instance resuming the notebook audit.
This is a snapshot of state, priorities, and gotchas **as of the compaction
point**. Git HEAD at snapshot: `94b342b notebooks: clear TEST_MODE-tainted
cached outputs + fix two real bugs`.

---

## 1. Load these skills first

Both auto-trigger on the paths we'll touch, but invoke explicitly to be safe:

- **`autolens-fit-diagnostics`** (user-level). Core rule:
  *"numbers lie without pictures — always open `fit_subplot.png` before
  pronouncing a fit PASS."*  Triggers on `results/`, `output/`,
  `fit_subplot*.png`, `corner.pdf`, `samples.csv`, `summary.json`, SLaM stage
  dirs, notebooks containing `SLaM` / `SOLVED` / `pixelized` / `mge`.
- **`gr-lensing-intuition`** (user-level). Use for physical-sanity checks —
  e.g. the Mod 06 bulge-halo conspiracy, NFW scale radii, θ_E vs enclosed
  mass, image-topology / caustic counts.

---

## 2. The task you were in the middle of

User request (verbatim): *"double check the modules and module solutions to
make sure we catch all these incorrect plots"*.

An enumeration pass found **92 embedded PNG cells** across module + solution
notebooks. High-priority fit-subplot cells were extracted to
`/tmp/all_nb_outputs/` (that tmp dir is gone now — re-extract as needed).

### Already viewed in the previous session

| Cell | File | Verdict |
|---|---|---|
| idx 14 | `Modules/03_First_Lens_Model/03_first_lens_model.ipynb` | viewed, not yet written up |
| idx 13 | `Modules/06_Multi_Component_Mass_Models/06_multi_component_mass.ipynb` | viewed, not yet written up |
| idx 7  | `Solutions/09_mge_linear_light_profiles_SOLVED.ipynb` | viewed, **suspected bug** (see §3) |

### Still to audit — ranked by priority

**A. High (fits likely to have bugs):**
- `Solutions/09_mge_linear_light_profiles_SOLVED.ipynb` — cells 4, 7,
  13, 16, 18, 39. Cell 7 is the prime suspect: check whether it hardcodes
  `ell_comps=(0.05, 0.0)` for the truth tracer. Module 09 cell 4 had the same
  bug and was fixed to `(0.05263, 0.0)` in commit `94b342b`. The SOLVED copy
  was not touched in that commit. **Grep the file for `ell_comps=(0.05, 0.0)`
  and `ell_comps=(0.096, -0.056)` and fix to `(0.05263, 0.0)` /
  `(0.09622, -0.05556)` if present.**
- `Modules/03_First_Lens_Model/03_first_lens_model.ipynb` — idx 17, 18, 19
  (fit residuals, corner, tracer).
- `Solutions/03_first_lens_model_SOLVED.ipynb` — idx 14, 17, 18, 19.
- `Solutions/05_pixelized_source_SOLVED.ipynb` — idx 3, 4, 11, 13, 18,
  19, 28. Idx 28 is the *intentionally bad* "WRONG MASS MODEL → SOURCE
  ARTIFACTS" demo — confirm it is still clearly framed as a teaching
  anti-example, not a real result.
- `Modules/06_Multi_Component_Mass_Models/06_multi_component_mass.ipynb` —
  idx 13, 14. Narrative around bulge-halo split was rewritten in `94b342b`;
  cached outputs may still reflect the old NFW tuning. If so, the fix is to
  clear outputs (the parameters in the code cells are already correct).
- `Solutions/06_multi_component_mass_SOLVED.ipynb` — same cells.

**B. Medium (real data — likely safe, quick scan):**
- `Modules/07_Real_Data_FITS_to_Model/07_real_data_fits_to_model.ipynb` —
  idx 5, 9, 16.

**C. Low (data visualizations, not fits):**
- `Modules/01_Basics_Grids_Galaxies_RayTracing/*.ipynb`
- `Modules/02_Simulating_Lens_Data/*.ipynb`
- `Solutions/01_*.ipynb`, `Solutions/02_*.ipynb`

---

## 3. Known bug pattern — hardcoded rounded `ell_comps`

In Mod 09 cell 4 (fixed in `94b342b`), the "truth" tracer was built with
`ell_comps=(0.05, 0.0)` while the stored `tracer.json` from Mod 02 has
`(0.05263, 3e-18)`. 5 % error in ellipticity → **18σ ring residual** in the
"truth − data" plot, which reads exactly like a model-fit residual.

Canonical values for the Mod 02 simulated-lens tracer (trust these, they come
from `autolens_workspace_original`):

| Component | Field | Correct value | Wrong (rounded) |
|---|---|---|---|
| lens bulge | `ell_comps` | `(0.05263, 0.0)` | `(0.05, 0.0)` |
| lens mass  | `ell_comps` | `(0.05263, 0.0)` | `(0.05, 0.0)` |
| source bulge | `ell_comps` | `(0.09622, -0.05556)` | `(0.096, -0.056)` |

Any "truth" plot in the tutorials using the Mod 02 simulation **must** use the
left column. Search for the rounded values before fixing.

---

## 4. What NOT to touch

- **`Modules/04_Search_Chaining_SLaM/results/`** — currently tracks a PASS
  HEAD state from an older Cannon run. A re-run with new priors (free
  `mass_lp.centre`, tight source `R_e`) degraded `mass_total[1]` from
  χ²/pix 0.91 → 2.06 and pushed max|res| from 3.86σ → 6.33σ. Root cause not
  yet understood (the 21-parameter model seems to destabilize the downstream
  chain). **Ship the old results; investigate separately.**
- The committed Mod 04 `mass_total[1]` numerically PASSes
  (χ²/pix 0.91, max|res| 3.86σ) but the `fit_subplot.png` shows (a) a central
  blob in the lens-light-subtracted panel, (b) scattered bright source-plane
  pixels far from the caustic, (c) a χ² peak of 14.87 at the lens centre.
  This is a *documented known issue*, not a new finding to fix.
- Mod 04 `source_lp` / `source_pix[1,2]` intermediate stages have visible ring
  residuals — expected behavior of early SLaM stages, do not "fix".

---

## 5. Environment guards in place (do not remove)

Five-layer defence against `PYAUTOFIT_TEST_MODE=1` leaks, in order from
outermost to innermost:

1. `check_install.py` — prints a warning if set.
2. `Modules/10_Cluster_Computing/scripts/submit_cannon.slurm` — refuses to
   launch.
3. `slam_v2026.py` — raises at import time.
4. Every module notebook's imports cell (Mods 03 / 05 / 06 / 08).
5. Every corresponding Solutions notebook.

`PYAUTOFIT_TEST_MODE` makes PyAutoFit skip sampling and return a random prior
draw; the April 17 leak is what poisoned the Mod 04/05/09 cached notebook
outputs that were cleared in `94b342b`.

---

## 6. Cluster workflow — one command

```bash
# From laptop, from repo root:
bash Modules/10_Cluster_Computing/scripts/submit_to_cannon.sh 04
# pulls in cannon.env → push_to_cannon.sh --go → SHA256-verify local vs remote
# fit_module04.py → ssh cannon sbatch. Uses the `cannon` ssh alias so Duo
# prompts once per session, not three times.
```

Supporting files:

- `Modules/10_Cluster_Computing/cannon.env` (gitignored, user's defaults)
- `Modules/10_Cluster_Computing/cannon.env.example` (tracked template)
- `Modules/10_Cluster_Computing/SETUP_NEW_USER.md` (10-step onboarding)
- `Modules/10_Cluster_Computing/CANNON_HANDOFF.md` (for a Claude running on Cannon)
- `Modules/10_Cluster_Computing/CLUSTER_WORKFLOW_NOTES.md` (retro + backlog)

Not needed for today's audit, but keep in mind that the audit may reveal plots
that need regenerating — in which case the cluster path is ready.

---

## 7. Concrete first move for the next session

1. Invoke `/autolens-fit-diagnostics` and `/gr-lensing-intuition`.
2. Run:
   ```
   Grep on Solutions/09_mge_linear_light_profiles_SOLVED.ipynb for
     ell_comps=(0.05, 0.0)
     ell_comps=(0.096, -0.056)
   ```
   If either appears, fix to `(0.05263, 0.0)` / `(0.09622, -0.05556)` and
   clear the notebook's cached outputs for that cell.
3. Work down the priority list in §2. For each notebook: view the flagged
   PNG → apply the skill's PASS/SUSPECT/FAIL thresholds → file a fix or
   move on.
4. End-of-session: append to `PROGRESS_LOG.md` and commit.

---

## 2026-04-21 update — §1 complete

Done in the post-midnight session:

- **Mod 09** `09_mge_linear_light_profiles.ipynb` cell `4454484a`
  ("MANUAL MGE: 60 LINEAR GAUSSIANS"): mass `Isothermal` → `(0.05263, 0.0)`;
  source `SersicCore` → `(0.09622, -0.05556)`. MGE Gaussians kept at
  `(0.05, 0.0)` with updated narrative explaining the pedagogical point
  (MGE is flexible enough to fit well even when the bulge geometry is
  slightly off; mass must be exact for the residuals to isolate light vs
  source modelling error). Stale `(0.1, 0.05)` comment rewritten.

- **`Solutions/09_mge_linear_light_profiles_SOLVED.ipynb`** (three cells,
  `23787c6d` "QUICK FIT", `9945b7c3` "MANUAL MGE", `1d84ee24`
  "SOLUTION 1: GAUSSIAN COUNT EXPERIMENT"): same policy — mass + source
  fixed to exact truth, MGE Gaussians kept as `(0.05, 0.0)` with an
  explanatory comment. The API-demo cell (`8146ffe9`, lines 241–255)
  was intentionally left alone since `ell_comps` is arbitrary in that
  context. Outputs cleared on all three edited cells.

- **Paths corrected everywhere in this file**: `Modules/Solutions/...` →
  `Solutions/...` (the compact summary had the wrong path; actual location
  is repo-top `Solutions/`).

§2 remaining (audit PNGs) has not yet been re-checked now that the
underlying ell_comps bugs are resolved — the cached "suspected bug"
outputs on Solutions/09 idx 7 in particular should re-render clean
after re-execution, not from a fresh edit pass.
