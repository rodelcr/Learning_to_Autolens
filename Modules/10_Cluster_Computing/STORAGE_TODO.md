# Storage TODO — Cannon output path

## Current state (2026-04-22)

Cluster jobs write Nautilus output to **netscratch**:

```
OUTPUT_ROOT = /n/netscratch/hernquist_lab/Lab/rcordova/learning_to_autolens/output
```

passed to `submit_cannon.slurm` via
`--export=ALL,MODULE=XX,OUTPUT_ROOT=/n/netscratch/.../output` at `sbatch` time.

## Why — temporary escape valve, not a durable solution

The default path in `submit_cannon.slurm`
(`${REPO_ROOT}/output` on holystore01 lab storage) started hitting
**`RuntimeError: Disk quota exceeded`** mid-job during `SOURCE PIX`
checkpoint writes. `lfs quota -hg hernquist_lab /n/holystore01` reports
**600T used / 600T limit** — the lab quota is saturated. Our own usage is
trivial (~1.3 GB total; 65 MB of SLaM output), but any new write fails
because the lab pool is full.

## Why netscratch is not a real fix

- `/n/netscratch/` has a **90-day purge policy**. Anything we write gets
  deleted on its own timer. The lightweight artifacts
  (`Modules/*/results/`) are git-tracked so they survive, but the raw
  Nautilus output tree (`output/module_XX/slam/.../hashes/`) — which
  `--include-raw` pulls, and which you'd need to resume/re-export —
  disappears automatically.
- If you re-submit a job past the purge window, Nautilus can't auto-resume
  from a checkpoint that's been deleted. You start from scratch.
- Cross-user sharing on scratch is less durable than lab storage.

## Options we should evaluate

1. **Clean up existing lab-storage content.** `du` on Cannon today:
   - `HSC+DESI` — 773 GB (rcordova's largest item)
   - `osage_llm` — 183 GB
   - `desi_env` — 5.5 GB
   - `HSC+DESI_2026` — 1.4 GB

   These predate the Learning_to_Autolens project. Audit what's still
   active, archive or delete what isn't. Even a 20% cleanup of HSC+DESI
   would free more than this project will ever use.

2. **Request lab-quota increase from FASRC.** hernquist_lab is at 100%
   utilisation on a 600T allocation; either the PI asks for more, or
   members consolidate.

3. **Migrate bulk data to a different lab path.** Some labs at FASRC have
   secondary allocations on `holylfs*` that may not be saturated. Worth
   `df -h /n/holylfs*/LABS/hernquist*` to check.

4. **Purchase scratch allocation.** FASRC sells per-TB-per-year storage
   with no purge — a 1 TB purchase would comfortably hold the full raw
   Nautilus output tree for this project and a few others.

## Implication for this project specifically

The **committed** lightweight artifacts in `Modules/*/results/` are
unaffected — they live in the git repo, are ~MB-scale each, and the
results-viewer cells in every notebook read them directly. Whoever clones
the repo still gets the publication-grade summaries.

The **raw Nautilus output tree** on netscratch will get purged after 90
days. After that, re-running a module requires a fresh Cannon fit; you
can't resume an old checkpoint. For the tutorial suite this is fine (the
committed summaries are the canonical record). For ongoing *research*
work — where you expect to re-sample the posterior or try new priors on
existing chains — we need a durable fix.

## Action items

- [ ] Audit the 773 GB `HSC+DESI` content on Cannon — identify what's archivable
- [ ] Ask the PI about lab-quota utilisation; request increase if needed
- [ ] Verify `/n/holylfs*/LABS/hernquist*` availability (secondary allocations)
- [ ] Consider FASRC storage-purchase option
- [ ] When a durable path is chosen, update `Modules/10_Cluster_Computing/cannon.env.example`
      to document the `OUTPUT_ROOT` convention, and drop the netscratch
      override from `submit_to_cannon.sh` invocations

Until one of the above lands, every Cannon submission needs
`OUTPUT_ROOT=/n/netscratch/hernquist_lab/Lab/${USER}/learning_to_autolens/output`
passed explicitly.
