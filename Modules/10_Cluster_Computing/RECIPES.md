# Cannon Submission Recipes

Copy-pasteable `sbatch` invocations for every fit in this repo, with realistic wall-time budgets. **Pre-flight:** `bash Modules/10_Cluster_Computing/scripts/push_to_cannon.sh --go` first. Each recipe assumes you `ssh cannon` and `cd $CANNON_REPO_ROOT` first (or run via `ssh cannon "cd ... && sbatch ..."` as one line).

For the workflow context see [`STUDENT_QUICKSTART.md`](STUDENT_QUICKSTART.md).

---

## Curriculum modules

| Recipe | What it fits | Wall-time |
|---|---|---|
| `MODULE=04` | Search chaining + SLaM on simple Iso lens | ~1-2 h |
| `MODULE=05` | Pixelized source reconstruction | ~3-4 h |
| `MODULE=06` | Multi-component mass (Sersic + NFW) | ~2-3 h |
| `MODULE=09` | MGE + linear light profiles | ~1-2 h |

```bash
# Module 04 SLaM (the canonical first example)
sbatch --time=4:00:00 --export=ALL,MODULE=04 \
  Modules/10_Cluster_Computing/scripts/submit_cannon.slurm

# Module 05 pixelized source
sbatch --time=8:00:00 --export=ALL,MODULE=05 \
  Modules/10_Cluster_Computing/scripts/submit_cannon.slurm
```

---

## Examples — galaxy-scale single-deflector

### `compound_lens` — 2-deflector simple mock (z=0.5, 0.8, source z=1.7)

| Recipe | What | Time |
|---|---|---|
| `--part=direct` | Single-search free fit | ~3-5 h |
| `--part=slam` | Two-track SLaM (effective vs staged) | ~6-10 h |
| `--part=advanced` | PowerLaw + pixelized variants | ~10-15 h |

```bash
sbatch --time=8:00:00 --export=ALL,EXAMPLE=compound_lens,FIT_EXTRA_ARGS=--part=direct \
  Modules/10_Cluster_Computing/scripts/submit_cannon.slurm
```

### `disky_spiral_lens` — Bulge+disk Bayes-factor demo

| Recipe | What | Time |
|---|---|---|
| `--part=single_sersic` | Wrong-physics single-Sersic baseline | ~1 h |
| `--part=bulge_disk` | Two-component bulge+disk fit | ~2 h |
| `--part=all` | Both, sequentially | ~3 h |

```bash
sbatch --time=4:00:00 --export=ALL,EXAMPLE=disky_spiral_lens,FIT_EXTRA_ARGS=--part=all \
  Modules/10_Cluster_Computing/scripts/submit_cannon.slurm
```

### `mge_to_physical` — Stars + dark-matter decomposition (3-search chain)

| Recipe | What | Time |
|---|---|---|
| `--part=light` | Search 1 only — MGE light profile | ~30 min |
| `--part=stars_only` | Search 2 — stars-only mass (MGE-as-mass) | ~2 h |
| `--part=stars_dark` | Search 3 — stars + NFW dark matter | ~3 h |
| `--part=all` | 1 → 2 → 3 sequentially | ~5-6 h |
| `--part=all_v2` | Same chain but with secondary deflector + 2 sources (matches truth) | ~6-8 h |

```bash
# v2 chain (corrects the model misspecification documented in the README)
sbatch --time=8:00:00 --export=ALL,EXAMPLE=mge_to_physical,FIT_EXTRA_ARGS=--part=all_v2 \
  Modules/10_Cluster_Computing/scripts/submit_cannon.slurm
```

### `agel_real_target` — Real HST cutout of AGEL013322-125201A

| Recipe | What | Time |
|---|---|---|
| `--part=direct` | Single PowerLaw + Sersic source on real HST data | ~2-4 h |

```bash
sbatch --time=4:00:00 --export=ALL,EXAMPLE=agel_real_target,FIT_EXTRA_ARGS=--part=direct \
  Modules/10_Cluster_Computing/scripts/submit_cannon.slurm
```

---

## Examples — multi-deflector / multi-source architectures

### `compound_lens_zoo` — 5-mock zoo (mocks 2-6) with the R0→R5 ladder

| Recipe | What | Time |
|---|---|---|
| `EXAMPLE=compound_lens_zoo` | Canonical R2 fit on all 5 mocks | ~4-6 h |
| `EXAMPLE=compound_lens_zoo_climb FIT_EXTRA_ARGS='--rung R3 --mock 3'` | R3 multi-plane on one mock | ~1-2 h |
| `--rung R5` | R5 multi-plane + 2-source (mocks 3, 4) | ~13 h |
| `--rung R5_truth` | Truth-anchored R5 (PowerLaw, validation) | ~24-48 h |
| `--rung R5_truth_iso` | Same but Iso primary (cheaper) | ~12 h |
| `--rung R5_staged` | 2-stage chain (R2_2src → R5) | ~3-10 h |
| `--rung R5_freecosmo` | Wide lens + free Om₀, w₀ (compound zoo §15) | ~12 h |
| `--rung R5_truth_freecosmo` | Tight lens + free Om₀, w₀ (correct cosmography) | ~24 h |

```bash
# R5 climb on mock 3
sbatch --time=24:00:00 \
  --export=ALL,EXAMPLE=compound_lens_zoo_climb,FIT_EXTRA_ARGS="--rung R5 --mock 3" \
  --job-name=mock3_R5 \
  Modules/10_Cluster_Computing/scripts/submit_cannon.slurm

# Truth-anchored cosmography (the methodologically-correct setup)
sbatch --time=24:00:00 \
  --export=ALL,EXAMPLE=compound_lens_zoo_climb,FIT_EXTRA_ARGS="--rung R5_truth_freecosmo --mock 3" \
  --job-name=truth_freecosmo_m3 \
  Modules/10_Cluster_Computing/scripts/submit_cannon.slurm
```

**Important:** for `EXAMPLE=compound_lens_zoo_climb`, the slurm post-process export will fail (no `Examples/compound_lens_zoo_climb/` directory). Manually export with the redirect:

```bash
ssh cannon "source ~/.bashrc && conda activate autolens312 && \
  cd \$CANNON_REPO_ROOT && \
  python Modules/10_Cluster_Computing/scripts/export_results.py \
    --output-root output/<unique_tag_dir> \
    --example compound_lens_zoo --repo-root \$(pwd)"
```

For staged (multi-stage) outputs, use `--search-dir` + `--dest`:
```bash
python Modules/10_Cluster_Computing/scripts/export_results.py \
  --search-dir output/<output_root>/<stage_dir>/<hash_dir> \
  --dest Examples/compound_lens_zoo/results/<stage_dir> \
  --repo-root $(pwd)
```

### `double_source_plane` — 1 lens + 2 sources at z=1.0, 2.5

| Recipe | What | Time |
|---|---|---|
| `--part=direct` | Cosmology-fixed direct fit | ~3-4 h |
| `--part=beta_freecosmo` | Tight lens priors + free Om₀, w₀ | ~12-24 h |

```bash
sbatch --time=24:00:00 \
  --export=ALL,EXAMPLE=double_source_plane,FIT_EXTRA_ARGS=--part=beta_freecosmo \
  Modules/10_Cluster_Computing/scripts/submit_cannon.slurm
```

### `group_scale` — BGG + 3 satellites at z=0.4

| Recipe | What | Time |
|---|---|---|
| `--part=truth_anchored` | Tight Gaussians on `mock_truth.json` (working baseline, χ²/N=1.025 PASS) | ~1-2 h |
| `--part=staged_satellites` | 2-stage BGG-then-satellites chain | (DOES NOT CONVERGE — known issue, see README) |
| `EXAMPLE=group_scale_slam` | 3-stage SLaM pipeline (MGE bulges + Iso mass) | ~8-9 h (converges but FAIL on residuals — see README) |

```bash
# The working baseline
sbatch --time=4:00:00 \
  --export=ALL,EXAMPLE=group_scale,FIT_EXTRA_ARGS=--part=truth_anchored \
  Modules/10_Cluster_Computing/scripts/submit_cannon.slurm
```

### `cluster_scale` — BCG + 10 FJ-scaled members + 2 sources at z=1.5, 2.8

| Recipe | What | Time |
|---|---|---|
| `EXAMPLE=cluster_scale` | Direct fit (~19 free params via FJ scaling) | ~12-18 h |

```bash
sbatch --time=24:00:00 --export=ALL,EXAMPLE=cluster_scale \
  Modules/10_Cluster_Computing/scripts/submit_cannon.slurm
```

---

## Examples — single-source-plane variants

### `quad_time_delay` — Point-source quasar with time delays

| Recipe | What | Time |
|---|---|---|
| `--part=phase_1_cosmology_fixed` | Lens fit at fixed Planck15 | ~1-2 h |
| `--part=phase_2_h0_free` | Same model, H₀ free | ~2-3 h |

```bash
sbatch --time=4:00:00 --export=ALL,EXAMPLE=quad_time_delay,FIT_EXTRA_ARGS=--part=phase_2_h0_free \
  Modules/10_Cluster_Computing/scripts/submit_cannon.slurm
```

### `subhalo_sensitivity` — Substructure detection demo

| Recipe | What | Time |
|---|---|---|
| `--part=smooth` | Smooth-mass baseline | ~1 h |
| `--part=with_perturber` | Same + a fixed-position perturber | ~1 h |
| `--part=both` | Both (Bayes-factor comparison) | ~2 h |

```bash
sbatch --time=4:00:00 --export=ALL,EXAMPLE=subhalo_sensitivity,FIT_EXTRA_ARGS=--part=both \
  Modules/10_Cluster_Computing/scripts/submit_cannon.slurm
```

### `interferometer_basic` — Galaxy-scale lens on uv-plane data

| Recipe | What | Time |
|---|---|---|
| `EXAMPLE=interferometer_basic` | Visibility-plane direct fit | ~1-2 h |

```bash
sbatch --time=4:00:00 --export=ALL,EXAMPLE=interferometer_basic \
  Modules/10_Cluster_Computing/scripts/submit_cannon.slurm
```

---

## Common patterns

### Resuming a TIMED-OUT fit
Nautilus auto-checkpoints to `output/<...>/search_internal/`. Resubmit with the same `unique_tag` and a longer `--time` and it picks up where it left off:

```bash
sbatch --time=48:00:00 --export=ALL,EXAMPLE=<same as before>,FIT_EXTRA_ARGS=<same as before> \
  --job-name=<descriptive_v2> \
  Modules/10_Cluster_Computing/scripts/submit_cannon.slurm
```

### Submitting an array (multiple mocks at once)
Use a shell loop, one `sbatch` per:
```bash
for MOCK in 2 3 4 5 6; do
  sbatch --time=24:00:00 \
    --export=ALL,EXAMPLE=compound_lens_zoo_climb,FIT_EXTRA_ARGS="--rung R5 --mock $MOCK" \
    --job-name=mock${MOCK}_R5 \
    Modules/10_Cluster_Computing/scripts/submit_cannon.slurm
done
```

### Cancelling a running job
```bash
ssh cannon "scancel <JOBID>"
```

### Cleaning up after a run (if you want to recover Cannon disk)
The `output/` tree on Cannon can balloon to many GB. After exporting + pulling artifacts, you can prune intermediate state:
```bash
ssh cannon "rm -rf \$CANNON_REPO_ROOT/output/<unique_tag>/.../search_internal"
```
**Be careful** — without `search_internal/` Nautilus can't resume. Only do this for fits you're sure are done.
