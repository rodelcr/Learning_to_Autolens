# Cluster workflow notes — pain points & improvement roadmap

A retrospective of what's been painful about the laptop ↔ Cannon loop
and the specific upgrades worth making. Written during the 2026-04-20
session where a stale-hash silent resume wasted a full audit round.

---

## Pain points we've actually hit

### 1. Stale-hash silent resume (⭐ the expensive one)

**What happened:** A re-submitted job completed in minutes instead of
hours. The pulled `results/` had `chi_squared_per_pixel` values
**identical to four decimals** with the previous run; the only visible
difference was that `fit_subplot.png` was newly rendered (same content,
new timestamp).

**Why:** `fit_module04.py` on Cannon was out of sync with the laptop
version (forgot to `push_to_cannon.sh` before `sbatch`). Nautilus found
`.completed` markers at the existing model hash, skipped sampling, and
only the post-job `export_results.py` ran — re-rendering PNGs from the
already-finished search output.

**Cost:** 1 wasted cluster allocation + a full audit cycle to realize
the fit hadn't actually re-run. Without the `autolens-fit-diagnostics`
skill catching that numbers were suspiciously identical, we'd have
advanced to the next step thinking the priors had been refit.

**Fixed by:**
- `submit_to_cannon.sh` — one-command wrapper that pushes, verifies
  `sha256sum` matches on both sides, then sbatches. Refuses to submit
  on mismatch.
- `submit_cannon.slurm` now echoes `fit_module${MODULE}.py` SHA256 and
  `git HEAD`/`git status` at job start. Any future "why did this run
  produce wrong numbers?" debug starts from the slurm log.

**Not yet fixed:**
- No post-job "diff the new results against previous" check. A
  `diff_results.py` utility that compares numeric fields of every
  `summary.json` between the current HEAD and the working tree would
  catch "stale-hash rerun" instantly — if every number is identical to
  the last commit, the run didn't actually do anything new.

### 2. "Forgot to push before sbatch"

**What happened:** Tried to re-run Mod 04 with updated priors; hit
problem #1 because the new script never left the laptop.

**Fixed by:** `submit_to_cannon.sh` bundles push + sbatch. Direct
`sbatch` is still allowed — the wrapper just adds a verification layer
for users who want it.

### 3. `image/fit.fits` missing on resumed searches

**What happened:** Mod 05 `search1_parametric_source/summary.json` had
null `chi_squared_per_pixel`, `max_abs_normalized_residual`,
`n_unmasked_pixels` despite a valid fit.

**Why:** `export_results.py` reads those metrics from
`<search>/image/fit.fits`. On a resumed search, autolens's default
config (`force_visualize_overwrite: false`) skips regenerating the
visualization artifacts. The CSV, samples_summary, and log files are
fine; only the FITS cube is absent.

**Fixed by:**
- `_force_visualize(analysis, result, ...)` helper added to
  `fit_template.py`, `fit_module04.py`, and `fit_module05.py`. Called
  after every `search.fit()`, it explicitly invokes
  `analysis.visualize(paths, instance, during_analysis=False)` which
  guarantees `image/fit.fits` is written.
- `export_results.py` now emits a `chi_squared_status` field in
  `summary.json` when the FITS cube is missing, naming the fix
  rather than silently nulling the metrics.

**Not yet fixed:**
- The config-level fix — setting `force_visualize_overwrite: true` in
  `autolens_workspace_{original,latest}/config/general.yaml` — would be
  simpler and more robust. We haven't applied it because neither config
  tree is unambiguously the one autolens loads (Mods 04/05 use
  `_original`, Mod 09 uses `_latest`), and overriding both risks drift.

### 4. `PYAUTOFIT_TEST_MODE` leak

**What happened:** Module 08's cached results_summary.json reported
`chi²_red = 44.8`, `θ_E` off by 15%, `log_evidence = −54970`. Fit was
real-looking but every number was garbage.

**Why:** `PYAUTOFIT_TEST_MODE=1` (a documented integration-testing flag
in `autolens_workspace_latest/CLAUDE.md`) was set in the shell that
launched Jupyter. PyAutoFit's `abstract_search.py` checks this flag
and, if set, returns a random prior draw instead of sampling. The
search completes, writes valid-looking artifacts, and everything
downstream believes it.

**Fixed by:** 5-layer guard:
1. `check_install.py` — FAIL if set
2. `submit_cannon.slurm` — refuse to submit
3. `slam_v2026.py` — raise at import (covers Mods 04, 09)
4. Every module notebook's imports cell — raise on load
5. Every SOLVED notebook's imports cell — same

### 5. Numbers lie without pictures

**What happened:** An initial audit (by a general-purpose subagent
reading only `summary.json` files) pronounced all Mod 04/05/09 cluster
results "all good." The user pushed back; opening the actual
`fit_subplot.png` images revealed that Mod 04 `source_lp[1]`,
`source_pix[1]`, and `source_pix[2]` had coherent Einstein-ring
residuals despite `chi²/pix ≈ 1` and `max|res| ≈ 5σ` — a failed fit
masquerading as acceptable.

**Fixed by:** `autolens-fit-diagnostics` skill auto-triggers on any
`results/` path, mandates opening `fit_subplot.png`, and refuses to
pass a fit without both the numbers and the image.

**Not yet fixed:**
- A static-site renderer that aggregates every `results/**/fit_subplot.png`
  into one browsable page would make "look at every image" cheap.
  Currently the human (or Claude) has to open files one by one.

### 6. Dataset workspace split (`_original` vs `_latest`)

**What happened:** Mods 04/05 use `autolens_workspace_original/`
(v2025.11) datasets; Mod 09 uses `autolens_workspace_latest/`
(v2026.2). The submit script has to dispatch based on module number.
Dataset paths differ; config paths differ; adding a new module requires
picking the right workspace.

**Consequence:** Fit scripts carry a hard dataset path dependency; a
student copying `fit_template.py` for a new module has to know which
workspace to point at.

**Not yet fixed:**
- Pin to one workspace. `_latest` (v2026.2) supersedes `_original` for
  all current APIs. Migrating Mods 04/05 datasets to `_latest` would
  eliminate the dispatch. Blocker: `_original` has the `simple` and
  `simple__no_lens_light` datasets that `_latest` may have renamed.

### 7. `pull_from_cannon.sh` pulled from the wrong path

**What happened:** The script pulled `$CANNON_SCRATCH/...` but
`export_results.py` writes to `${REPO_ROOT}/Modules/*/results/`
(under `holystore`, not `holyscratch`). Users had to manually rsync.

**Fixed by:** two-step pull — (1) always pull `Modules/*/results/**`
from the repo root, (2) optional `--include-raw` for the scratch output.

### 8. Username / host inconsistency

**What happened:** `push_to_cannon.sh` defaulted `CANNON_USER=rcordova`;
`pull_from_cannon.sh` defaulted `CANNON_USER=rcordovarosado`. Either
worked but they drifted out of sync; `submit_to_cannon.sh` (new) uses
the push-side convention.

**Fixed by:** unified defaults.

### 9. SSH + Duo 2FA blocks automation

**What happened:** Claude can't complete the 2FA challenge, so any ssh
call from an agent session fails. All cluster interaction has to be
laptop-side, bouncing results back through files on disk.

**Not yet fixed (probably can't be):**
- SSH keys + Kerberos tickets reduce the 2FA frequency per FASRC docs,
  but Duo remains mandatory on login. Accept that Claude can't submit
  jobs directly; the `submit_to_cannon.sh` wrapper is the substitute.

### 10. No provenance in slurm logs

**What happened:** Debugging "why did run N produce different numbers
than run N-1?" required reading the laptop git log and hoping the
Cannon file state matched at submit time. No log evidence linked a
specific slurm job to a specific script version.

**Fixed by:** `submit_cannon.slurm` now echoes `fit_module*.py`
SHA256, mtime, `git HEAD`, `git branch`, and whether the repo is
dirty at job start. The log file is self-contained for forensics.

---

## Medium-term improvements (worth doing next)

### A. Git-backed sync instead of rsync

**Current:** `rsync` over ssh. Works, but:
- No version history on the Cannon side
- No easy way to roll back a bad script change
- No way to reference a run by commit hash

**Proposal:** Treat the Cannon repo as a git working tree.

1. Initialize the Cannon copy as a git clone once:
   `git clone <github-url> ${REPO_ROOT}`
2. `push_to_cannon.sh` becomes a `git pull` on Cannon (invoked via ssh),
   not an rsync. User pushes to a remote first.
3. `submit_cannon.slurm`'s provenance echo (already in place) shows the
   commit hash → any log file is trivially reproducible.
4. Datasets, outputs, and checkpoints stay out of git
   (gitignore); they're still rsynced.

**Tradeoffs:**
- Requires a git remote (GitHub private repo, FASRC gitlab, or a bare
  repo on a shared filesystem).
- User must commit + push before submitting — an additional step, but
  one that forces versioning discipline. `submit_to_cannon.sh` can
  chain `git push` + `ssh git pull` + `ssh sbatch` automatically.

**Blocker:** The repo contains the `autolens_workspace_{original,latest}/`
trees (large, third-party). Those shouldn't be in a shared git remote.
Either make them git submodules pointing at upstream, or keep rsync
for those dirs only and git for everything else.

### B. Pre-submit dry-run

**Proposal:** Add a `--dry-run` mode to `submit_to_cannon.sh` that:
- Pushes (or checks git state is pushed)
- Verifies SHA256 matches
- Dispatches an `salloc -t 5:00` interactive session
- Runs `python fit_module${MODULE}.py --part dry-run` (new flag)
- Imports autolens, builds the model, prints `model.total_free_parameters`
  and the first 30 lines of `model.info`, then exits **before**
  calling `search.fit()`.

Catches model-definition errors, missing dependencies, or
PYAUTOFIT_TEST_MODE leaks in < 5 min instead of in a scheduled job
that might wait in the queue for hours. Ideal before a new SLaM
pipeline's first full run.

### C. Results-diff tool

**Proposal:** `scripts/diff_results.py <from-commit> <to-commit>` that:
- Walks every `Modules/*/results/*/summary.json` in both refs
- Emits a table of fields that changed (or are unchanged to 4 dp)
- Flags "100% of numeric fields unchanged" as a stale-hash signal

Running this as part of `pull_from_cannon.sh` (compare `git HEAD` vs.
pulled working-tree) would catch problem #1 automatically.

### D. Structured slurm output

**Proposal:** Every `print(..., flush=True)` in `fit_module*.py` emits
a JSON line alongside the human-readable line, e.g.:

```
[SLaM] SOURCE LP done in 47.2 min; θ_E = 1.603"
{"event": "stage_complete", "stage": "source_lp", "wall_min": 47.2, "theta_E": 1.603}
```

A downstream parser (maybe the same `diff_results.py`) can extract
per-stage timing, θ_E, log_evidence, chi²/pix from any slurm log
without human reading. Useful for cross-run trend plots (e.g.
"did source_lp get faster after switching to n_live=150?").

### E. Auto-triggered post-pull diagnostic

**Proposal:** `pull_from_cannon.sh` ends with `autolens-fit-diagnostics`
applied to every newly-changed `results/` directory, and prints a
PASS/SUSPECT/FAIL table before exiting.

Requires the skill to be callable as a script (it currently fires
through Claude Code's runtime). A standalone Python implementation
of the thresholds from `references/thresholds.md` would be a
50-line script — worth writing.

### F. Consolidate `fit_module*` duplication

**Current:** `fit_module04.py`, `fit_module05.py`, `fit_module09.py`,
and `fit_template.py` duplicate ~40% of their code (`load_dataset`,
`_force_visualize`, CLI parsing, the print-banner block).

**Proposal:** Move the shared code to `Modules/10_Cluster_Computing/scripts/_cluster_utils.py`.
Each fit script imports from it and only contains the model + search
definitions. Makes the "copy template and edit build()" story a
one-cell edit.

### G. Explicit Nautilus non-resume mode for forced reruns

**Current:** To force a fresh SOURCE LP (e.g., after changing priors
in a way that unexpectedly produces the same model hash), the user
has to `rm -rf ${REPO_ROOT}/output/module_04/slam/simple/source_lp\[1\]`.
Error-prone.

**Proposal:** Add a `--force-fresh-stage=source_lp` CLI flag to each
`fit_module*.py` that deletes the named stage's output directory
before calling `search.fit()`. Idempotent via date-stamped backup:
`mv source_lp source_lp.bak.2026-04-20T12-00Z`.

### H. Cluster runbook in-notebook

**Current:** `CANNON_HANDOFF.md` (new today) lives as a separate file.
A student or new Claude instance would need to know to look for it.

**Proposal:** Add a markdown cell to Mod 10 that `%%bash`-invokes
`cat CANNON_HANDOFF.md` inline, so the handoff is part of the
notebook's rendered flow.

---

## Long-term improvements

### I. Webhook reporting

After every `sbatch` job completion, have a post-step script POST a
summary JSON to a URL. If it's a Slack webhook: instant "Mod 04 job
{JOB_ID} finished: PASS / SUSPECT / FAIL: mass_total χ²/pix=0.913,
max|res|=3.86σ, logZ=5555.9". Keeps the user out of email-ping-then-ssh
loops.

### J. Pre-merge CI

If the repo ends up on GitHub, a CI workflow on every PR that:
- Runs `check_install.py` in a fresh conda env
- Runs a 5-minute smoke fit (the Mod 03 model with n_live=25)
- Applies the `autolens-fit-diagnostics` thresholds
- Comments on the PR with the summary

Would catch API drift between autolens releases automatically — a
known pain point with a monthly-release package.

### K. Cross-session memory for Claude

`~/.claude/projects/.../memory/` already caches laptop-side lessons
learned. The Cannon handoff doc extends this to the cluster side but
isn't automatically loaded. A future iteration could:
- Auto-load `CANNON_HANDOFF.md` when Claude detects it's running in
  `/n/holystore01/.../learning_to_autolens/`
- Keep a cluster-side memory dir at `${REPO_ROOT}/.claude/cannon-memory/`
  so cluster Claude sessions preserve lessons separately from laptop ones

### L. Unified environment via nix / devcontainer

Rebuild the laptop/cluster parity on top of a devcontainer (`.devcontainer.json`)
or nix flake. Both sides activate the same hermetic env; no more
`autolens` vs. `autolens312` vs. `_original` vs. `_latest` matrix.
Large effort; pays off once there are 3+ modules with differing env
needs.

---

## Summary table

| Pain point | Fixed in this session? | What's next |
|---|---|---|
| Stale-hash silent resume | ✅ submit_to_cannon.sh + slurm provenance | Results-diff tool (C) |
| Forgot to push | ✅ submit_to_cannon.sh | — |
| Missing fit.fits | ✅ _force_visualize in all fit scripts | Config `force_visualize_overwrite: true` (optional) |
| PYAUTOFIT_TEST_MODE leak | ✅ 5-layer guard | — |
| Numbers lie without pictures | ✅ autolens-fit-diagnostics skill | Auto-triggered post-pull diagnostic (E) |
| Two-workspace split | — | Consolidate to `_latest` (long-term) |
| Wrong pull path | ✅ pull_from_cannon.sh fixed | Git-backed sync (A) |
| Username inconsistency | ✅ unified | — |
| SSH / 2FA blocks automation | ⚠️ working around | Webhook reporting (I) |
| No slurm provenance | ✅ hash + git echo in submit_cannon.slurm | Structured JSON output (D) |
| Script duplication | — | _cluster_utils.py (F) |
| Forced-rerun ergonomics | — | `--force-fresh-stage=` flag (G) |
| Handoff discoverability | ✅ CANNON_HANDOFF.md | Inline in Mod 10 notebook (H) |

Priorities if we do only three more things:

1. **Git-backed sync (A)** — eliminates most of §7 and reframes
   provenance around commits instead of SHA256s.
2. **Results-diff tool (C)** — detects stale-hash reruns automatically
   instead of relying on humans noticing "too quick".
3. **Auto-triggered post-pull diagnostic (E)** — moves the
   `autolens-fit-diagnostics` skill from manual invocation to
   pull-time, so every return trip lands with a verdict in hand.
