# Post-midnight run plan — 2026-04-21

Authoritative instructions for the scheduled wake-up chain. Each firing of
`ScheduleWakeup` should re-read this file and act from it — the wake prompt
itself just says "read POST_MIDNIGHT_PLAN.md and follow it."

## 0. Sleep chain

- Current time < midnight EDT → `ScheduleWakeup` again. Pick the tightest legal
  delay: `min(3600, seconds_to_midnight - 10)` with a minimum of 60. Reason
  field should say "chained wake toward midnight". Prompt: "Read
  `/Users/rosador/Documents/AGEL/Learning_to_Autolens/Modules/10_Cluster_Computing/POST_MIDNIGHT_PLAN.md` and follow it."
- Current time ≥ midnight EDT → start §1.

## 1. Finish the in-flight ell_comps fixes

Start here — these are cheap and unblock the re-runs in §2 from reproducing the
2026-04-20 bug discoveries.

### Task #17 — Mod 09 `09_mge_linear_light_profiles.ipynb`

Use **NotebookEdit** (plain `Edit` errors on `.ipynb`). Cell id `4454484a`
("MANUAL MGE: 60 LINEAR GAUSSIANS, FIXED GEOMETRY"). Changes:

- Stale code comment currently says `ell_comps=(0.1, 0.05)` → rewrite to
  reflect the actual `(0.05, 0.0)` the cell uses, and add a line that the mass
  below uses exact truth.
- Mass `al.mp.Isothermal` ell_comps: `(0.05, 0.0)` → `(0.05263, 0.0)`.
- Source `al.lp_linear.SersicCore` ell_comps: `(0.096, -0.056)` →
  `(0.09622, -0.05556)`.
- **Keep** MGE Gaussian `ell_comps=(0.05, 0.0)` (comment "Close to true lens
  ellipticity" — pedagogical approximation to show MGE flexibility).
- **Do NOT touch** cell `d445493d` (lines ~200–220) — pure API-comparison
  demo, ell_comps is arbitrary.

### Task #18 — `Solutions/09_mge_linear_light_profiles_SOLVED.ipynb`

(Solutions live at **repo-top `Solutions/`**, not `Modules/Solutions/` — the
compact summary had the wrong path.)

Run `Grep ell_comps` on the file and apply the same fix policy as §1/Task #17:

- "QUICK FIT" truth tracer (lines ~307–327 in the raw file): mass + source must
  be exact truth. 94b342b only fixed the Modules copy; Solutions still has the
  rounded values here — verify, fix if so.
- MGE basis "true lens" fit (lines ~689–715): mass + source exact truth; MGE
  Gaussians keep `(0.05, 0.0)` with comment.
- Later truth-comparison block (lines ~1820–1834): same policy.
- API-demo cell (lines ~240–253): leave alone.

### Task #19 — Fix `AUDIT_HANDOFF.md`

That file (written 2026-04-20) points to `Modules/Solutions/*SOLVED.ipynb`.
Actual path is repo-top `Solutions/`. Grep-replace `Modules/Solutions/` →
`Solutions/` in `Modules/10_Cluster_Computing/AUDIT_HANDOFF.md`.

## 2. Run every locally-runnable notebook

**Scope of "runnable locally":** notebooks whose fits will hit existing
Nautilus caches under `Modules/*/output/` and complete in seconds to minutes.
Fresh heavy fits (new priors, new model) that would take >30 min on the laptop
are **not** in scope — those go to Cannon (the user has said "Prefer cluster
over local for heavy fits").

### Order (cheap → heavy)

1. `Modules/01_Basics_Grids_Galaxies_RayTracing/01_grids_galaxies_raytracing.ipynb`
2. `Modules/02_Simulating_Lens_Data/02_simulating_lens_data.ipynb`
3. `Modules/03_First_Lens_Model/03_first_lens_model.ipynb`
4. `Modules/04_Search_Chaining_SLaM/04_search_chaining_slam.ipynb` — hits the
   Cannon-pulled `results/` loader, not a fresh fit. Should be fast.
5. `Modules/05_Pixelized_Source_Reconstructions/05_pixelized_sources.ipynb` — same.
6. `Modules/06_Multi_Component_Mass_Models/06_multi_component_mass.ipynb` — the
   NFW tuning was rewritten in 94b342b; outputs are cleared and need fresh
   execution to render correctly.
7. `Modules/07_Real_Data_FITS_to_Model/07_real_data_fits_to_model.ipynb`
8. `Modules/08_Results_Diagnostics_Figures/08_results_diagnostics_figures.ipynb` —
   reads Mod 04 `results/` artifacts; should execute fast.
9. `Modules/09_MGE_Linear_Light_Profiles/09_mge_linear_light_profiles.ipynb` —
   cell `4454484a` does a non-search NNLS fit (~seconds); cell `4bc1c309` runs
   Nautilus (`n_live=75`) that should hit cache. Cell `f1998bc3` is the full
   5-stage SLaM pipeline — **skip executing this in-notebook**, defer to the
   `show_result("mass_total[1]")` cell that reads `results/`.
10. `Modules/10_Cluster_Computing/10_cluster_computing.ipynb` — documentation
    / launchers; just execute.
11. Solutions/*.ipynb in the same 1→10 order.

### Execution mechanics

For each notebook:

```bash
PYAUTOFIT_TEST_MODE=   # explicit unset — the 5 guards will trip if set
unset PYAUTOFIT_TEST_MODE

jupyter nbconvert \
    --to notebook \
    --execute \
    --ExecutePreprocessor.timeout=1800 \
    --ExecutePreprocessor.kernel_name=autolens \
    --inplace \
    "$NB"
```

1800s (30 min) per-cell timeout = hard ceiling to catch non-cached heavy fits.
If a cell times out, the skill rule applies: it's a cluster job. Do *not* bump
the timeout — revert the notebook (`git checkout HEAD -- "$NB"` if pre-dirty,
or just accept the partial execute state), document the heavy cell in
`PROGRESS_LOG.md`, and move on.

### Debug-as-needed loop

On execution failure:

1. Read the traceback directly from nbconvert stderr.
2. Classify:
   - **Import / env bug** (autolens / autofit / nautilus mismatch) → fix in
     `requirements.txt` or `check_install.py`.
   - **Path bug** (dataset path, results/ path) → fix the path in-notebook.
   - **API rename** (autolens monthly releases) → check `pip index versions
     autolens`; if current env is behind, update; if API genuinely changed,
     update the notebook.
   - **`PYAUTOFIT_TEST_MODE` leak** (guards raise) → fix environment, never
     remove the guard.
   - **Heavy fit with no cache** (>30min timeout) → do NOT run; mark for
     Cannon in `PROGRESS_LOG.md`.
3. Re-run the notebook from scratch after the fix.
4. Commit fixes per-notebook with a clear message so the run can be reverted
   granularly if later review disagrees.

### Apply the fit-diagnostics skill

After each fit-bearing notebook executes, inspect newly-rendered
`fit_subplot` panels and residual maps per
`~/.claude/skills/autolens-fit-diagnostics/SKILL.md`. The core rule stands:
**numbers lie without pictures**. Any SUSPECT/FAIL verdict on a freshly
re-rendered cell → stop, document in `PROGRESS_LOG.md`, either fix inline or
flag for the next audit session. Do not silently accept ring residuals.

## 3. What NOT to do

- Do NOT launch a fresh cluster job (`submit_to_cannon.sh`) without explicit
  user authorization. Pulling existing results is fine.
- Do NOT bump the 1800s nbconvert timeout.
- Do NOT remove any `PYAUTOFIT_TEST_MODE` guard.
- Do NOT touch `Modules/04_Search_Chaining_SLaM/results/` — committed state
  is the PASS HEAD and the new-priors re-run produced regressions.
- Do NOT rebuild `Modules/04/results/mass_total[1]/corner.pdf` etc.

## 4. End of shift

When §2 is complete (or blocked by a cluster-required fit):

1. Append a dated session entry to `PROGRESS_LOG.md` listing every notebook
   executed, verdict per fit, timings, and any issues surfaced.
2. `git status` → commit fixes in logical groups (ell_comps fixes, NotebookEdit
   fixes, handoff path fixes, notebook outputs). Use clear messages.
3. Do NOT push. Do NOT create PRs.
4. Tell the user via end-of-turn summary what ran, what passed/failed, and
   what's left for them.
