# Paper-Repro Spec 05 — Nightingale+2023 Abell 1201 UMBH Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reproduce Nightingale+2023's **M_BH = (3.27 ± 2.12) × 10¹⁰ M_sun**
detection (Abstract; reconfirmed §4.2 / Table 4) in the Abell 1201 BCG.
**Two-track plan**: a paper-faithful reproduction track using N+23's actual
methodology (counter-image suppression in a pixelised AdaptiveBrightness
source — Stage 4 below) AND a methodology-extension track that adds a Jeans
σ_v factor on top (Stage 3) as OUR contribution beyond the paper. See the
design doc §6 (Methodology correction) for the verified-via-ar5iv reading
of how N+23 actually breaks the γ′-M_BH degeneracy — it is **NOT via σ_v
kinematics** (paper §4.2 + §4.3 use counter-image morphology + mass-light
coaxiality, both imaging-only arguments).

**Architecture:** 4-stage chained PyAutoLens fit. Each stage submitted as a
chained slurm job with `--dependency=afterok`. PyAutoLens-only — no Herculens
cross-validation (paper is PyAutoLens-native).

- **Stage 1 (`lp`)** — parametric Sersic source + PowerLaw lens + ExternalShear, no SMBH. Smoke / θ_E baseline.
- **Stage 2 (`with_smbh`)** — adds central `al.mp.PointMass`. **First M_BH posterior**; Bayes-factor vs Stage 1 gives our first detection-significance estimate (N+23 threshold ΔlnZ = 4.5 = 3σ per §3.9; their F814W triple-Sersic = ΔlnZ ≈ 3 marginal).
- **Stage 3 (`with_kin`)** — joint AnalysisImaging + Jeans σ_v factor via FactorGraphModel. **OUR EXTENSION** — N+23 does NOT use σ_v in their likelihood (§1 mention only).
- **Stage 4 (`adapt`)** — pixelised Voronoi source + `al.reg.Adapt` (autolens 2026.4's rename of N+23's `al.reg.AdaptiveBrightness`). **This is the paper-faithful stage** — only the pixelised source can resolve the counter-image whose morphology drives the M_BH detection in N+23 §4.2 + Table 4 (3-Sersic light + F390W → ΔlnZ = +100.58).

For the publication-grade match: combine the **multi-Sersic light refactor**
(driver flag `--n-light={1,2,3}` shipped 2026-05-19, matches N+23 Table 1's
ladder; their 3-Sersic wins the Bayes comparison) with **F390W primary band**
(7150 s exposure; F814W's 1009 s is N+23's marginal-detection cross-check).

**Tech Stack:** PyAutoLens 2026.4 + autofit Nautilus + Phase 3
`Modules/10_Cluster_Computing/scripts/_jeans_sigma_v.py` (`AnalysisKinematics`
+ `KinematicDataset`) + AGEL Watson HST reduction pipeline (Spec 00 §6.9) for
the paper-grade F814W drizzle.

**Depends on:**
- Spec 00 §6.9 (Watson pipeline) — *optional for Stage-0 HLA pass, required
  for paper-grade rerun*
- v0.96 Module 15 + `Examples/radial_arc_smbh` (methodology validated on synthetic mock)
- v0.97 Phase 3 `_jeans_sigma_v.AnalysisKinematics` (kinematic factor;
  `save_results` no-op stub committed in `cdce73f`)
- v0.97 `Examples/radial_arc_smbh/ --part=with_kinematics` strict-PASS Cannon
  result audited (Cannon job `rarc_kin3` completed 2026-05-18; needs subplot
  regen — see HANDOFF_2026_05_18 follow-on)

---

## File Structure

```
private/2303_15514_nightingale2023_abell1201/
├── PAPER_NOTES.md                                ← shipped 2026-05-18
├── data/
│   ├── download_a1201.py                         ← shipped — HST proposal 14886 fetch
│   ├── raw_flt/mastDownload/                     ← already populated (~700 MB)
│   ├── hla/mastDownload/                         ← already populated (HLA drizzles)
│   ├── fit_ready/                                ← already populated (image/noise/psf)
│   └── agel_reduced/                             ← NEW: Watson-pipeline outputs (Stage-1+)
├── code/
│   ├── __init__.py                               ← shipped
│   ├── a1201_lens_model.py                       ← shipped; needs --part=adapt added (Phase 4)
│   ├── prep_a1201_dataset.py                     ← shipped; needs empirical-PSF step (Task 2)
│   ├── extract_mbh.py                            ← shipped
│   ├── extract_psf_from_field.py                 ← NEW (Task 2): empirical PSF from stars
│   └── chain_priors_from_lp.py                   ← NEW (Task 8): Stage 1 → Stage 2 chaining
├── submit_a1201.slurm                            ← NEW (Task 6): chained 4-stage submit
├── notebooks/
│   ├── 01_a1201_data_prep.ipynb                  ← NEW (Task 4)
│   ├── 02_a1201_local_smoke.ipynb                ← NEW (Task 5)
│   └── 03_a1201_mbh_recovery.ipynb               ← NEW (Task 15) — the headline notebook
├── tests/
│   ├── test_a1201_lens_model.py                  ← shipped (4 tests)
│   ├── test_a1201_data_prep.py                   ← NEW (Task 4)
│   ├── test_extract_psf.py                       ← NEW (Task 2)
│   └── test_chain_priors.py                      ← NEW (Task 8)
├── output_local/                                 ← Phase 2 smoke-fit output (gitignored)
└── results/                                      ← exported lightweight artifacts (Cannon)
    ├── a1201_lp/                                 ← Stage 1
    ├── a1201_with_smbh/                          ← Stage 2
    ├── a1201_with_kin/                           ← Stage 3
    └── a1201_adapt/                              ← Stage 4 (AdaptiveBrightness)
```

---

## Phase 1: Data prep (laptop, ~½ day)

### Task 1: Verify download + cutout state

- [x] `data/download_a1201.py` shipped 2026-05-18; raw FLT pulled to `data/raw_flt/mastDownload/`.
- [x] `code/prep_a1201_dataset.py` shipped 2026-05-18; `data/fit_ready/{image,noise_map,psf}.fits` built from HLA F814W DRC.

- [ ] **Step 1: Confirm raw + HLA drizzles intact**

```bash
cd private/2303_15514_nightingale2023_abell1201
du -sh data/
find data/raw_flt -name '*flt.fits' -o -name '*flc.fits' | wc -l   # expect ≥ 4
find data/hla -name '*f814w*drc*.fits' | head -3                   # expect ≥ 1
```

Expected: ~700 MB total, ≥ 4 FLT files, ≥ 1 F814W HLA DRC.

- [ ] **Step 2: Confirm the fit-ready cutout exists and is sane**

```bash
python -c "
from astropy.io import fits
for p in ['image', 'noise_map', 'psf']:
    h = fits.open(f'data/fit_ready/{p}.fits')[0]
    print(f'{p:10s} shape={h.data.shape} min={h.data.min():.3g} max={h.data.max():.3g}')
"
```

Expected: `image` is 200×200 and centred on the BCG (peak ≥ ~10×background);
`noise_map` is 200×200 strictly positive; `psf` is 21×21 normalised to ~1.

- [ ] **Step 3: Commit if not yet committed (gitignored anyway, but log progress)**

Append a line to `private/PROGRESS_2026_05_18.md`:
```
2026-05-XX [Spec 05] Phase 1 Task 1 verified — raw FLT + HLA DRC + fit_ready trio.
```

### Task 2: Replace Gaussian PSF with empirical WFC3/UVIS PSF

The shipped `prep_a1201_dataset.py` uses a 21×21 σ=0.04″ Gaussian PSF as a
placeholder. Nightingale+2023 uses an empirical PSF from foreground stars
in the field — this is critical for the radial image, which lives within
~1″ of the bright BCG core and is PSF-shape-dependent.

**Files:**
- Create: `private/2303_15514_nightingale2023_abell1201/code/extract_psf_from_field.py`
- Create: `private/2303_15514_nightingale2023_abell1201/tests/test_extract_psf.py`
- Modify: `private/2303_15514_nightingale2023_abell1201/code/prep_a1201_dataset.py`
  (call the new function instead of the Gaussian stub)

- [ ] **Step 1: Failing test — empirical PSF should be normalised and peaked at centre**

```python
# tests/test_extract_psf.py
from pathlib import Path
import sys
HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "code"))

import numpy as np


def test_empirical_psf_is_normalised_and_centred():
    from extract_psf_from_field import build_psf_from_dummy_star
    psf = build_psf_from_dummy_star(shape=(21, 21), fwhm_pix=2.5, seed=0)
    assert psf.shape == (21, 21)
    assert abs(psf.sum() - 1.0) < 1e-6, f"PSF should sum to 1, got {psf.sum()}"
    peak_y, peak_x = np.unravel_index(np.argmax(psf), psf.shape)
    assert peak_y == 10 and peak_x == 10, f"Peak at ({peak_y}, {peak_x}), expected (10, 10)"


def test_empirical_psf_from_field_returns_correct_shape():
    """Extracting from real field: just verify the API contract on a dummy DRC."""
    from extract_psf_from_field import extract_psf_from_field
    import numpy as np
    from astropy.io import fits
    # Inject 3 fake stars at known positions in a 500x500 dummy image
    img = np.zeros((500, 500), dtype=np.float32)
    yy, xx = np.mgrid[:21, :21]
    star = np.exp(-((yy - 10) ** 2 + (xx - 10) ** 2) / (2 * 2.0 ** 2))
    for cy, cx in [(100, 100), (250, 300), (400, 150)]:
        img[cy - 10:cy + 11, cx - 10:cx + 11] += star
    psf = extract_psf_from_field(img, star_positions=[(100, 100), (250, 300), (400, 150)],
                                  cutout_size=21)
    assert psf.shape == (21, 21)
    assert abs(psf.sum() - 1.0) < 1e-6
```

- [ ] **Step 2: Run test (FAIL — module not yet written)**

```bash
cd private/2303_15514_nightingale2023_abell1201
pytest tests/test_extract_psf.py -v
```
Expected: `ModuleNotFoundError: No module named 'extract_psf_from_field'`.

- [ ] **Step 3: Implement `extract_psf_from_field.py`**

```python
"""extract_psf_from_field.py — build an empirical PSF from isolated stars
in the same WFC3/UVIS frame as the BCG cutout.

Process:
  1. For each star at (y_pix, x_pix), cut out an N×N stamp.
  2. Centroid-shift each stamp to sub-pixel-align the peaks.
  3. Median-stack across stars (robust to bad pixels / cosmic rays).
  4. Normalise so sum = 1.

For first pass, the caller can either:
  - Pass `star_positions=[(y1, x1), ...]` extracted manually from the DRC,
  - OR call `build_psf_from_dummy_star(...)` to get a Gaussian fallback
    that matches the API for smoke testing.

Long-term (Stage-1+): swap in a tinytim WFC3/UVIS F814W model PSF.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple

import numpy as np


def build_psf_from_dummy_star(shape: Tuple[int, int] = (21, 21),
                              fwhm_pix: float = 2.5,
                              seed: int = 0) -> np.ndarray:
    """Single-star Gaussian PSF for tests / smoke fits."""
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    cy, cx = (shape[0] - 1) / 2.0, (shape[1] - 1) / 2.0
    sigma = fwhm_pix / 2.3548
    psf = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2))
    return (psf / psf.sum()).astype(np.float32)


def _centroid_shift(stamp: np.ndarray) -> np.ndarray:
    """Sub-pixel-align the peak of `stamp` to its geometric centre."""
    from scipy.ndimage import shift
    cy, cx = (stamp.shape[0] - 1) / 2.0, (stamp.shape[1] - 1) / 2.0
    # First-moment centroid (intensity-weighted)
    yy, xx = np.mgrid[:stamp.shape[0], :stamp.shape[1]]
    s = stamp.sum()
    py = (stamp * yy).sum() / s
    px = (stamp * xx).sum() / s
    return shift(stamp, (cy - py, cx - px), order=3, mode="constant", cval=0.0)


def extract_psf_from_field(image: np.ndarray,
                           star_positions: Sequence[Tuple[int, int]],
                           cutout_size: int = 21) -> np.ndarray:
    """Stack PSF from a list of (y_pix, x_pix) stellar centroids in `image`."""
    half = cutout_size // 2
    stamps = []
    for (cy, cx) in star_positions:
        y0, y1 = cy - half, cy + half + 1
        x0, x1 = cx - half, cx + half + 1
        if y0 < 0 or x0 < 0 or y1 > image.shape[0] or x1 > image.shape[1]:
            continue
        stamps.append(_centroid_shift(image[y0:y1, x0:x1].astype(np.float64)))
    if not stamps:
        raise ValueError("No valid stars (all positions out of bounds).")
    stack = np.median(np.stack(stamps, axis=0), axis=0)
    return (stack / stack.sum()).astype(np.float32)
```

- [ ] **Step 4: Run tests (PASS)**

```bash
pytest tests/test_extract_psf.py -v
```
Expected: 2 passed.

- [ ] **Step 5: Add a `--use-empirical-psf` flag to `prep_a1201_dataset.py`**

Modify `prep_a1201_dataset.py` so that when `--use-empirical-psf
"y1,x1;y2,x2;..."` is passed, it pulls the DRC, extracts stamps at those
pixel coords (use SAOImage DS9 to identify isolated stars in the field
beforehand), stacks them, and writes that as `psf.fits` instead of the
Gaussian fallback. The default (no flag) retains the Gaussian for fast smoke.

- [ ] **Step 6: Regenerate `psf.fits` with empirical PSF (one-off)**

```bash
# Pick 3-5 isolated stars from the DRC via DS9 first; record their (y, x) pixels.
python code/prep_a1201_dataset.py \
    --hla-path data/hla/mastDownload/HST/<...>/hst_*f814w_drc.fits \
    --output-dir data/fit_ready/ \
    --use-empirical-psf "<y1>,<x1>;<y2>,<x2>;<y3>,<x3>"
```

Expected: `data/fit_ready/psf.fits` updated; mtime newer than other fits-ready files.

- [ ] **Step 7: Commit**

```bash
git add private/2303_15514_nightingale2023_abell1201/code/extract_psf_from_field.py \
        private/2303_15514_nightingale2023_abell1201/tests/test_extract_psf.py \
        private/2303_15514_nightingale2023_abell1201/code/prep_a1201_dataset.py
# Note: private/ is gitignored — this commit goes to the public-facing wrapper docs only.
# Update private/PROGRESS_2026_05_18.md with the regeneration record.
```

---

## Phase 2: Local smoke fit (`--part=lp`) — verify driver runs end-to-end

### Task 3: Run a 20-live-point local smoke fit

**Files:**
- Modify: nothing (driver `a1201_lens_model.py` is shipped)

- [ ] **Step 1: Run the smoke**

```bash
cd /Users/rosador/Documents/AGEL/Learning_to_Autolens
mkdir -p private/2303_15514_nightingale2023_abell1201/output_local
conda run -n autolens python \
    private/2303_15514_nightingale2023_abell1201/code/a1201_lens_model.py \
    --part=lp --n-live=20 --mask-radius=3.0 \
    --dataset-root=private/2303_15514_nightingale2023_abell1201/data/fit_ready \
    --output-root=private/2303_15514_nightingale2023_abell1201/output_local
```

Expected: ~10–15 min wall on laptop; exit code 0; output tree at
`output_local/a1201_lp/<hash>/files/` containing `samples.csv`,
`samples_summary.json`, `model.results`.

- [ ] **Step 2: Run the shipped tests to confirm they still pass after Task 2's PSF refactor**

```bash
cd private/2303_15514_nightingale2023_abell1201
pytest tests/ -v
```
Expected: 4 (Task 0) + 2 (Task 2) = 6 passed.

- [ ] **Step 3: Eyeball the residual map**

Open `output_local/.../image/fit_subplot.png` and verify:
- Lens light (Sersic) approximately fits the BCG core
- Lensed arc geometry visible in the model image (model_image panel)
- Residual map shows obvious unmodelled structure near the lens centre
  (this is the radial image — Stage 2 is what catches it)

If the fit is catastrophic (chi²/N > 100, model is featureless), STOP and
re-check the cutout centring, mask radius, and PSF.

### Task 4: Data-prep walkthrough notebook

**Files:**
- Create: `private/2303_15514_nightingale2023_abell1201/notebooks/01_a1201_data_prep.ipynb`
- Create: `private/2303_15514_nightingale2023_abell1201/tests/test_a1201_data_prep.py`

- [ ] **Step 1: Failing test — data-prep module loads its `BCG_RA_DEG`/`BCG_DEC_DEG` from PAPER_NOTES**

```python
# tests/test_a1201_data_prep.py
from pathlib import Path
import sys
HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "code"))


def test_prep_a1201_uses_smith2017_centre():
    """The BCG centre must match Smith+2017 KCWI position (recorded in PAPER_NOTES)."""
    import prep_a1201_dataset as p
    assert abs(p.BCG_RA_DEG - 168.2270925) < 1e-6
    assert abs(p.BCG_DEC_DEG - 13.43582777778) < 1e-6


def test_fit_ready_trio_exists():
    """Smoke check: the fit-ready files exist with the expected shapes."""
    from astropy.io import fits
    root = HERE.parents[1] / "data" / "fit_ready"
    img = fits.open(root / "image.fits")[0].data
    noise = fits.open(root / "noise_map.fits")[0].data
    psf = fits.open(root / "psf.fits")[0].data
    assert img.shape == noise.shape, f"image {img.shape} vs noise {noise.shape}"
    assert psf.shape[0] == psf.shape[1] and psf.shape[0] % 2 == 1, "PSF must be odd-square"
```

- [ ] **Step 2: Run (PASS — these are integration tests against shipped state)**

```bash
pytest tests/test_a1201_data_prep.py -v
```

- [ ] **Step 3: Build the walkthrough notebook**

`notebooks/01_a1201_data_prep.ipynb` content (5 cells):

  1. Markdown — Title, summary of source data (HST proposal 14886, F814W),
     Smith+2017 spec-z, link to PAPER_NOTES.md.
  2. Code — `from astropy.io import fits; load and display image/noise_map/psf` with imshow.
  3. Markdown — Cutout strategy: 200×200 at ~0.04″/pix = 8″ × 8″ window
     centred on Smith+2017 BCG centroid.
  4. Code — Overlay the circular mask (radius 3.0″) on the image; verify
     no contaminant galaxies inside the mask.
  5. Markdown — Hand-off to `02_a1201_local_smoke.ipynb`.

- [ ] **Step 4: Run the notebook headlessly to confirm cells don't error**

```bash
jupyter nbconvert --to notebook --execute \
    --output 01_a1201_data_prep_executed.ipynb \
    --ExecutePreprocessor.kernel_name=autolens \
    notebooks/01_a1201_data_prep.ipynb
```

Expected: exits 0 with no exception cells.

- [ ] **Step 5: Commit (PROGRESS log + private/ note; notebook itself is gitignored)**

### Task 5: Local-smoke walkthrough notebook

**Files:**
- Create: `private/2303_15514_nightingale2023_abell1201/notebooks/02_a1201_local_smoke.ipynb`

- [ ] **Step 1: Build the notebook**

6 cells:
  1. Markdown — purpose: smoke-fit the lp driver; verify the model class wiring.
  2. Code — invoke `subprocess.run(['python', 'code/a1201_lens_model.py', '--part=lp', '--n-live=20', ...])`
     (or call `build_lp_fit` directly with a fresh n_live=20 model).
  3. Code — load `samples_summary.json` from `output_local/.../files/`; print the max-log-likelihood
     `instance` for `lens.mass.einstein_radius`, `lens.mass.slope`, `lens.shear.gamma_1/2`,
     `source.bulge.effective_radius`.
  4. Code — load `output_local/.../image/fit_subplot.png` and display inline.
  5. Markdown — sanity bar: θ_E should land in (3.0, 5.0)″ (BCG-scale lens at z=0.169 with
     z_s=0.451 — N+23 reports θ_E ≈ 3.8″); slope should be in (1.8, 2.2).
     Residual should still show un-modelled radial image structure inside ~1″ — that's
     Stage 2's job to absorb.
  6. Markdown — hand-off to Cannon: Phase 3.

- [ ] **Step 2: Execute headlessly + commit progress note**

```bash
jupyter nbconvert --to notebook --execute \
    --output 02_a1201_local_smoke_executed.ipynb \
    --ExecutePreprocessor.kernel_name=autolens \
    notebooks/02_a1201_local_smoke.ipynb
```

---

## Phase 3: Cannon strict-PASS chain (Stages 1–3 parametric source)

### Task 6: Write the chained `submit_a1201.slurm`

**Files:**
- Create: `private/2303_15514_nightingale2023_abell1201/submit_a1201.slurm`

The shipped `Modules/10_Cluster_Computing/scripts/submit_cannon.slurm` is
module-oriented and doesn't fit the Spec-05 directory layout. Write a
self-contained slurm that takes a `STAGE` env var and routes to the right
`--part`. Sourcing pattern follows the official cannon.env defaults (do not
hard-code a personal miniforge path — that's what killed `p1_pop_autofit`
job 13690686 today; see `CANNON_HANDOFF.md` §"Conda environment").

- [ ] **Step 1: Write the slurm**

```bash
#!/bin/bash
#SBATCH --job-name=a1201
#SBATCH --account=siag_lab
#SBATCH --partition=siag
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=rodrigo.cordova_rosado@cfa.harvard.edu

# =============================================================================
# Abell 1201 (Nightingale+2023) chained 4-stage submit.
#
# Pick the stage via STAGE env var at sbatch time:
#   sbatch --export=ALL,STAGE=lp        submit_a1201.slurm     (Stage 1)
#   sbatch --export=ALL,STAGE=with_smbh submit_a1201.slurm     (Stage 2)
#   sbatch --export=ALL,STAGE=with_kin  submit_a1201.slurm     (Stage 3)
#   sbatch --export=ALL,STAGE=adapt     submit_a1201.slurm     (Stage 4; Phase 4)
#
# Chained submission (Task 7 wraps this in --dependency=afterok):
#   STAGE1=$(sbatch --parsable --export=ALL,STAGE=lp        submit_a1201.slurm)
#   STAGE2=$(sbatch --parsable --dependency=afterok:$STAGE1 \
#                   --export=ALL,STAGE=with_smbh submit_a1201.slurm)
#   STAGE3=$(sbatch --parsable --dependency=afterok:$STAGE2 \
#                   --export=ALL,STAGE=with_kin  submit_a1201.slurm)
# =============================================================================

set -euo pipefail

STAGE="${STAGE:?must export STAGE=lp|with_smbh|with_kin|adapt}"

REPO_ROOT="/n/holystore01/LABS/hernquist_lab/Lab/${USER}/learning_to_autolens"
PROJECT_ROOT="${REPO_ROOT}/private/2303_15514_nightingale2023_abell1201"
DATASET_ROOT="${PROJECT_ROOT}/data/fit_ready"
OUTPUT_ROOT="${PROJECT_ROOT}/output/${STAGE}"

# Per CANNON_HANDOFF.md — Miniforge path, NOT a personal one.
source /n/sw/Miniforge3-25.3.1-0/etc/profile.d/conda.sh
conda activate autolens312

export PYTHONUNBUFFERED=1

mkdir -p "${REPO_ROOT}/logs" "${OUTPUT_ROOT}"
cd "${REPO_ROOT}"

# Per-stage n_live (heavier samplers for richer models)
case "${STAGE}" in
    lp)        N_LIVE=200 ;;
    with_smbh) N_LIVE=250 ;;
    with_kin)  N_LIVE=300 ;;
    adapt)     N_LIVE=400 ;;
    *) echo "ERROR: unknown STAGE=${STAGE}" >&2 ; exit 2 ;;
esac

echo "=== Job $SLURM_JOB_ID on $(hostname) at $(date -Is) ==="
echo "STAGE=${STAGE}  N_LIVE=${N_LIVE}  CPUs=${SLURM_CPUS_PER_TASK}"
echo "Python: $(which python) ($(python --version 2>&1))"
echo "Output: ${OUTPUT_ROOT}"
echo "================================================================"

srun python "${PROJECT_ROOT}/code/a1201_lens_model.py" \
     --part="${STAGE}" \
     --n-live="${N_LIVE}" \
     --mask-radius=3.0 \
     --dataset-root="${DATASET_ROOT}" \
     --output-root="${OUTPUT_ROOT}"

# Lightweight artifact export (so pull_from_cannon.sh has something small to ship)
python "${REPO_ROOT}/Modules/10_Cluster_Computing/scripts/export_results.py" \
    --output-root "${OUTPUT_ROOT}" \
    --results-root "${PROJECT_ROOT}/results/a1201_${STAGE}"

echo "=== Done at $(date -Is) ==="
```

- [ ] **Step 2: Validate the slurm is sane (no execution; just lint)**

```bash
bash -n private/2303_15514_nightingale2023_abell1201/submit_a1201.slurm
```
Expected: exit 0 (syntax OK).

- [ ] **Step 3: Commit the slurm to the public-facing wrapper docs trail (the slurm
  is under `private/` so itself gitignored; just log the addition in PROGRESS).**

### Task 7: Push + submit Stage 1 (parametric lp)

- [ ] **Step 1: Push current state to Cannon**

```bash
bash Modules/10_Cluster_Computing/scripts/push_to_cannon.sh --go
```
Expected: `private/2303_15514_nightingale2023_abell1201/` tree synced (including
the new `submit_a1201.slurm` + `code/extract_psf_from_field.py` + tests).

- [ ] **Step 2: Verify dataset is on Cannon (seed only if missing)**

```bash
ssh cannon "ls /n/holystore01/LABS/hernquist_lab/Lab/rcordova/learning_to_autolens/private/2303_15514_nightingale2023_abell1201/data/fit_ready/"
```
Expected: `image.fits`, `noise_map.fits`, `psf.fits`.

If missing (Stage-0 seeding), rsync just that subtree:
```bash
rsync -avh --progress \
    private/2303_15514_nightingale2023_abell1201/data/fit_ready/ \
    cannon:/n/holystore01/LABS/hernquist_lab/Lab/rcordova/learning_to_autolens/private/2303_15514_nightingale2023_abell1201/data/fit_ready/
```

- [ ] **Step 3: Submit Stage 1**

```bash
ssh cannon "cd /n/holystore01/LABS/hernquist_lab/Lab/rcordova/learning_to_autolens && \
            sbatch --parsable --export=ALL,STAGE=lp \
                   --job-name=a1201_lp \
                   private/2303_15514_nightingale2023_abell1201/submit_a1201.slurm"
```
Expected: prints a job ID; `squeue --me` shows it pending/running.

- [ ] **Step 4: Monitor**

```bash
ssh cannon "squeue --me --format='%.10i %.20j %.10P %.10T %.12M %.12L %R' \
            && tail -30 /n/holystore01/LABS/hernquist_lab/Lab/rcordova/learning_to_autolens/logs/a1201_lp_${JOBID}.out"
```

Expected wall: ~6 h on 32 cores at n_live=200.

### Task 8: Audit Stage 1 + write Stage-1→Stage-2 prior chainer

The next stage initialises from Stage 1's posterior. PyAutoLens offers two
options: `af.Model.from_samples(...)` (uses the Nautilus posterior as a prior
KDE), or hand-rolled `GaussianPrior(mean, sigma)` initialised from the
posterior summary. We use the latter for explicitness and to keep
provenance in the slurm log.

**Files:**
- Create: `private/2303_15514_nightingale2023_abell1201/code/chain_priors_from_lp.py`
- Create: `private/2303_15514_nightingale2023_abell1201/tests/test_chain_priors.py`

- [ ] **Step 1: Pull Stage 1 artifacts back**

```bash
bash Modules/10_Cluster_Computing/scripts/pull_from_cannon.sh --go
```
Expected: `private/2303_15514_nightingale2023_abell1201/results/a1201_lp/<search>/`
populated with `samples_summary.json`, `model.results`, `fit_subplot.png`,
`corner.pdf`.

- [ ] **Step 2: Audit (open `fit_subplot.png` + run `/autolens-fit-diagnostics`)**

Per `feedback_open_fit_png.md` memory: do not pass a fit on JSON numbers
alone. Open the subplot. The Stage-1 strict-PASS bar is **soft** (a
parametric Sersic source CANNOT absorb the radial image — residuals there
are expected):

  - chi²/N ≤ 2.0 on the masked region (lenient — Stage 1 is incomplete by design)
  - max|res| ≤ 6σ outside the central 0.5″
  - θ_E ∈ (3.0, 5.0)″, slope ∈ (1.7, 2.3)
  - Visible un-modelled flux at the BCG centre (the radial image)

If chi²/N > 2.0 OR θ_E lands outside (3.0, 5.0)″: STOP. Likely root causes:
mask radius wrong, dataset cutout off-centred, or PSF mismatch (re-do Task 2
with empirical PSF if not already).

- [ ] **Step 3: Failing test for chainer**

```python
# tests/test_chain_priors.py
from pathlib import Path
import sys, json, tempfile
HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "code"))


def test_chain_priors_loads_from_samples_summary(tmp_path):
    """Given a stub samples_summary.json, produce GaussianPriors centred on the medians."""
    from chain_priors_from_lp import build_chained_priors_from_summary

    summary_stub = {
        "median_pdf_sample": {
            "galaxies.lens.mass.einstein_radius": 3.85,
            "galaxies.lens.mass.slope": 2.02,
            "galaxies.lens.mass.ell_comps.ell_comps_0": -0.04,
            "galaxies.lens.mass.ell_comps.ell_comps_1":  0.07,
            "galaxies.lens.shear.gamma_1": 0.012,
            "galaxies.lens.shear.gamma_2": -0.008,
        },
        "errors_at_sigma_3_sample": {
            "galaxies.lens.mass.einstein_radius": [3.78, 3.92],
            "galaxies.lens.mass.slope": [1.95, 2.09],
        },
    }
    p = tmp_path / "samples_summary.json"
    p.write_text(json.dumps(summary_stub))

    priors = build_chained_priors_from_summary(p)
    assert "einstein_radius" in priors
    assert abs(priors["einstein_radius"]["mean"] - 3.85) < 1e-6
    # 3σ width should be ~(3.92 - 3.78)/2 → 1σ ≈ 0.0233
    assert abs(priors["einstein_radius"]["sigma"] - 0.0233) < 1e-3
```

- [ ] **Step 4: Run (FAIL)**

```bash
pytest tests/test_chain_priors.py -v
```

- [ ] **Step 5: Implement `chain_priors_from_lp.py`**

```python
"""chain_priors_from_lp.py — emit GaussianPriors initialised from Stage 1 posterior.

The Stage 2 driver imports `build_chained_priors_from_summary` and uses the
returned dict to override the default UniformPriors on the parametric-lens
parameters that survive Stage 1 → 2.

The radial-image structure that motivates the PointMass is in the RESIDUAL of
Stage 1, so Stage 2's lens-light + shear + mass parameters can be tightened
substantially (factor 5-10 on θ_E, 3-5 on slope, 3-5 on ell_comps) without
biasing the new PointMass parameter — which retains a wide LogUniform prior
because it's the new degree of freedom.

Key keys returned (used by `_apply_chained_priors` in a1201_lens_model.py):
    mass_einstein_radius, mass_slope, mass_ell_comps_0, mass_ell_comps_1,
    mass_centre_0, mass_centre_1, shear_gamma_1, shear_gamma_2,
    bulge_centre_0, bulge_centre_1, bulge_ell_comps_0, bulge_ell_comps_1,
    bulge_effective_radius, bulge_sersic_index
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict


# (full_dot_path_in_summary, short_key_used_by_driver)
_LP_TO_CHAIN = [
    ("galaxies.lens.mass.einstein_radius",            "mass_einstein_radius"),
    ("galaxies.lens.mass.slope",                      "mass_slope"),
    ("galaxies.lens.mass.ell_comps.ell_comps_0",      "mass_ell_comps_0"),
    ("galaxies.lens.mass.ell_comps.ell_comps_1",      "mass_ell_comps_1"),
    ("galaxies.lens.mass.centre.centre_0",            "mass_centre_0"),
    ("galaxies.lens.mass.centre.centre_1",            "mass_centre_1"),
    ("galaxies.lens.shear.gamma_1",                   "shear_gamma_1"),
    ("galaxies.lens.shear.gamma_2",                   "shear_gamma_2"),
    ("galaxies.lens.bulge.centre.centre_0",           "bulge_centre_0"),
    ("galaxies.lens.bulge.centre.centre_1",           "bulge_centre_1"),
    ("galaxies.lens.bulge.ell_comps.ell_comps_0",     "bulge_ell_comps_0"),
    ("galaxies.lens.bulge.ell_comps.ell_comps_1",     "bulge_ell_comps_1"),
    ("galaxies.lens.bulge.effective_radius",          "bulge_effective_radius"),
    ("galaxies.lens.bulge.sersic_index",              "bulge_sersic_index"),
]


def build_chained_priors_from_summary(summary_path: Path) -> Dict[str, Dict[str, float]]:
    """Return {short_param_name: {"mean": ..., "sigma": ...}} from samples_summary.json.

    Width is 1σ derived from the 3σ symmetric width if available, else falls
    back to 10% of the median (conservative for poorly-constrained parameters).
    """
    summary = json.loads(Path(summary_path).read_text())
    medians = summary.get("median_pdf_sample", {})
    err3 = summary.get("errors_at_sigma_3_sample", {})

    out: Dict[str, Dict[str, float]] = {}
    for full_name, short_key in _LP_TO_CHAIN:
        if full_name not in medians:
            continue
        mean = float(medians[full_name])
        if full_name in err3:
            lo, hi = err3[full_name]
            sigma_3 = (hi - lo) / 2.0
            sigma = max(sigma_3 / 3.0, 1e-4)
        else:
            sigma = max(abs(mean) * 0.10, 1e-3)
        out[short_key] = {"mean": mean, "sigma": sigma}
    return out
```

- [ ] **Step 6: Run (PASS)**

```bash
pytest tests/test_chain_priors.py -v
```

- [ ] **Step 7: Wire `build_with_smbh_fit()` to consume chained priors**

Modify `a1201_lens_model.py`'s `build_with_smbh_fit(...)` and `main()` to
accept an optional `--chain-from <summary_path>` argument; if present, call
`build_chained_priors_from_summary(...)` and override the relevant
`GaussianPrior` means/sigmas on the lens mass + shear + bulge before
building the model.

```python
# add inside main():
p.add_argument("--chain-from", type=Path, default=None,
               help="samples_summary.json from prior stage to seed priors")

# add helper used by build_with_smbh_fit / build_with_kin_fit:
def _apply_chained_priors(model, priors):
    import autofit as af
    # model is af.Collection(galaxies=...) — walk it and overwrite GaussianPriors
    # for each key in `priors` that matches a leaf parameter name.
    for short_name, spec in priors.items():
        try:
            param_path = _RESOLVE_SHORT_TO_FULL[short_name]  # mapping table
            target = model
            for attr in param_path:
                target = getattr(target, attr)
            target.replace_with(af.GaussianPrior(mean=spec["mean"], sigma=spec["sigma"]))
        except (AttributeError, KeyError):
            pass  # parameter not in this model variant (e.g. smbh in Stage 1)
    return model
```

- [ ] **Step 8: Commit**

### Task 9: Submit Stage 2 (with_smbh) chained from Stage 1

- [ ] **Step 1: Push the updated driver + chainer to Cannon**

```bash
bash Modules/10_Cluster_Computing/scripts/push_to_cannon.sh --go
```

- [ ] **Step 2: Submit Stage 2 with --chain-from**

```bash
ssh cannon "cd /n/holystore01/LABS/hernquist_lab/Lab/rcordova/learning_to_autolens && \
            STAGE1_SUMMARY=\$(ls private/2303_15514_nightingale2023_abell1201/results/a1201_lp/*/files/samples_summary.json | head -1) && \
            sbatch --parsable --export=ALL,STAGE=with_smbh,CHAIN_FROM=\$STAGE1_SUMMARY \
                   --job-name=a1201_with_smbh \
                   private/2303_15514_nightingale2023_abell1201/submit_a1201.slurm"
```

(Edit `submit_a1201.slurm` to pass `${CHAIN_FROM}` through to the python driver
as `--chain-from "${CHAIN_FROM}"` when set.)

Expected wall: ~12 h on 32 cores at n_live=250.

- [ ] **Step 3: Pull + audit**

```bash
bash Modules/10_Cluster_Computing/scripts/pull_from_cannon.sh --go
```

Audit bar for Stage 2:
- chi²/N ≤ 1.4 (the PointMass should largely absorb the radial-image residual)
- max|res| ≤ 5σ inside the masked region
- **ΔlogZ vs Stage 1 ≥ +3** (Nightingale+2023's "≥3σ detection" criterion — Jeffreys "decisive")
- M_BH posterior median in range; full headline extraction deferred to Task 11

If ΔlogZ < +3: open `corner.pdf`; the M_BH parameter (`smbh.einstein_radius`)
may have railed against the lower prior (LogUniform(1e-3, 0.5)) — a near-zero
posterior means imaging alone can't break γ′–M_BH. Proceed to Stage 3 regardless;
the kinematic factor is what finishes the job.

### Task 10: Submit Stage 3 (with_kin — joint imaging + Jeans σ_v) — OUR METHODOLOGY EXTENSION

**Citation note** (added 2026-05-19): this Stage is **not** part of N+23's
likelihood — that paper's degeneracy-breaker is imaging-only (counter-image
suppression §4.2 + mass-light coaxiality §4.3). Stage 3 here adds the
Smith+2017 σ_v = 285 ± 5 km/s as an independent Jeans factor on top of the
Stage 2 imaging fit, and is **our methodological contribution beyond the
paper**. The paper-faithful match is Stage 4 (`adapt`), not Stage 3.

If our Stage 3 +σ_v variant tightens the M_BH posterior consistent with the
N+23 headline, it validates the kinematic-break demonstration our v0.96
`Examples/radial_arc_smbh` synthetic example shipped. If it conflicts, the
N+23 imaging-only value remains the canonical answer and we report the
tension as a methodology finding.

- [ ] **Step 1: Sanity check on Cannon: `_jeans_sigma_v.py` + `AnalysisKinematics` import OK**

```bash
ssh cannon "cd /n/holystore01/LABS/hernquist_lab/Lab/rcordova/learning_to_autolens && \
            source /n/sw/Miniforge3-25.3.1-0/etc/profile.d/conda.sh && conda activate autolens312 && \
            python -c 'from Modules.\"10_Cluster_Computing\".scripts._jeans_sigma_v import AnalysisKinematics, KinematicDataset; print(\"OK\")'"
```

(Note: the dotted path with a space won't import directly; use the
`sys.path.insert` pattern already in `a1201_lens_model.build_kinematic_factor`.
Just verify the source file exists and is non-empty.)

Expected: `OK`, OR an import path issue you fix before submitting.

- [ ] **Step 2: Submit Stage 3 chained from Stage 2**

```bash
ssh cannon "cd /n/holystore01/LABS/hernquist_lab/Lab/rcordova/learning_to_autolens && \
            STAGE2_SUMMARY=\$(ls private/2303_15514_nightingale2023_abell1201/results/a1201_with_smbh/*/files/samples_summary.json | head -1) && \
            sbatch --parsable --export=ALL,STAGE=with_kin,CHAIN_FROM=\$STAGE2_SUMMARY \
                   --job-name=a1201_with_kin \
                   private/2303_15514_nightingale2023_abell1201/submit_a1201.slurm"
```

Expected wall: ~24 h on 32 cores at n_live=300 (kinematic factor adds ~2× per
likelihood call due to the Jeans integration).

- [ ] **Step 3: Pull + audit**

Audit bar for Stage 3:
- chi²_imaging/N ≤ 1.4 (unchanged from Stage 2; the kinematic factor shouldn't
  hurt the imaging chi²)
- σ_v_predicted at max-log-likelihood: 285 ± few km/s (within Smith+2017's σ)
- **M_BH posterior tightens** vs Stage 2; the γ′–M_BH degeneracy direction is
  visibly cut in `corner.pdf`
- ΔlogZ vs Stage 2 ≥ +1 (the kinematic factor adds 1 datum; the Bayes factor
  should still favour the joint fit if σ_v is consistent)

### Task 11: Extract the M_BH headline number

**Files:**
- Modify: `private/2303_15514_nightingale2023_abell1201/code/extract_mbh.py`
  (use directly; no edit needed unless schema changed)

- [ ] **Step 1: Run extract_mbh on Stage 3 samples**

```bash
cd /Users/rosador/Documents/AGEL/Learning_to_Autolens
python private/2303_15514_nightingale2023_abell1201/code/extract_mbh.py \
    --samples-csv $(ls private/2303_15514_nightingale2023_abell1201/results/a1201_with_kin/*/files/samples.csv | head -1) \
    --output-json private/2303_15514_nightingale2023_abell1201/results/mbh_with_kin.json
```

Expected output:
```
M_BH median: X.XXe+10
M_BH 16-84% (1σ): [X.XXe+10, X.XXe+10]
M_BH  2.5-97.5% (2σ): [X.XXe+10, X.XXe+10]
Published (Nightingale+2023): (3.27 ± 2.12) × 1e+10 — overlap with our 1σ: yes/no
```

- [ ] **Step 2: Verification gate**

**Strict-PASS:** our 1σ interval contains Nightingale+2023's 3.27 × 10¹⁰ M_sun.
**Soft-PASS:** our 2σ contains it.
**FAIL:** our 2σ excludes it → mass-model-too-simple or systematics elsewhere;
proceed to Phase 4 (AdaptiveBrightness) which is the documented escape route.

---

## Phase 4: AdaptiveBrightness pixelised source (Stage 4 — stretch, v0.99)

This is Nightingale+2023's actual methodological contribution: replace the
parametric Sersic source with a Voronoi mesh + `al.reg.Adapt`
regularization that gives the pixelised source adaptive degrees of freedom
concentrated at bright source-plane regions. A parametric Sersic can't
resolve the radial image — Stage 4 is what makes the M_BH detection
publication-grade.

Promoting Spec 05 from "stretch — v0.99" to "v0.98 ship-set" requires
Phase 4 to land strict-PASS.

### Task 12: Add `--part=adapt` to the driver

**Files:**
- Modify: `private/2303_15514_nightingale2023_abell1201/code/a1201_lens_model.py`
  (add `build_adapt_fit` + add `"adapt"` to argparse choices)
- Modify: `private/2303_15514_nightingale2023_abell1201/tests/test_a1201_lens_model.py`

- [ ] **Step 1: Failing test — `--part=adapt` is a valid CLI choice**

```python
# Append to tests/test_a1201_lens_model.py
def test_argparse_includes_adapt_part():
    """The driver must expose --part=adapt for Phase 4."""
    import subprocess, sys
    res = subprocess.run(
        [sys.executable, "code/a1201_lens_model.py", "--part=adapt",
         "--dataset-root=/nonexistent", "--output-root=/nonexistent",
         "--help"],
        capture_output=True, text=True,
    )
    # --help short-circuits before the missing-path check
    assert "--part {lp,with_smbh,with_kin,adapt}" in res.stdout or \
           "--part" in res.stdout and "adapt" in res.stdout


def test_build_adapt_fit_returns_search():
    """build_adapt_fit should construct a Nautilus search with a pixelized source."""
    from a1201_lens_model import build_adapt_fit
    # Smoke construction only — don't actually fit.
    # The function should accept dataset, output_root, n_live; return a Nautilus search obj.
    assert callable(build_adapt_fit)
    import inspect
    sig = inspect.signature(build_adapt_fit)
    for arg in ("dataset", "output_root", "n_live"):
        assert arg in sig.parameters
```

- [ ] **Step 2: Run (FAIL)**

```bash
cd private/2303_15514_nightingale2023_abell1201
pytest tests/test_a1201_lens_model.py::test_argparse_includes_adapt_part tests/test_a1201_lens_model.py::test_build_adapt_fit_returns_search -v
```

- [ ] **Step 3: Implement `build_adapt_fit` + register the argparse choice**

```python
# Add to a1201_lens_model.py after build_with_kin_fit:

def _build_source_galaxy_model_pixelized():
    """Voronoi mesh + Adapt regularization, per Nightingale+2023 §3."""
    import autofit as af
    import autolens as al

    mesh = af.Model(al.mesh.Voronoi)
    # Per Mod 05/09 audit memory: shape parameter is the source-pixel-count
    # control; (30, 30) is the default that audited PASS in Mod 05/SLAM.
    mesh.shape = (30, 30)

    # AdaptiveBrightness analogue in autolens 2026.4+ is reg.Adapt (renamed from
    # AdaptiveBrightness; confirmed via `pip index versions autolens` + grep
    # in autoarray/inversion/regularization/).
    reg = af.Model(al.reg.Adapt)
    reg.inner_coefficient = af.UniformPrior(lower_limit=0.001, upper_limit=10.0)
    reg.outer_coefficient = af.UniformPrior(lower_limit=0.001, upper_limit=100.0)
    reg.signal_scale = af.UniformPrior(lower_limit=0.001, upper_limit=10.0)

    pixelization = af.Model(al.Pixelization, mesh=mesh, regularization=reg)
    return af.Model(al.Galaxy, redshift=Z_SOURCE, pixelization=pixelization)


def build_adapt_fit(dataset, output_root: Path, n_live: int = 400):
    """Stage 4: pixelised AdaptiveBrightness source + lens model from Stage 3."""
    import autofit as af
    import autolens as al
    lens = _build_lens_galaxy_model(include_pointmass=True)
    source = _build_source_galaxy_model_pixelized()
    model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

    # Pixelized source REQUIRES position-likelihood constraints to avoid the
    # demagnified-central-solution failure mode (Mod 05 §7). Build from the
    # Stage 1 fit's predicted positions; for first pass, hard-code from N+23
    # (radial-image + tangential-arc positions in pixels).
    positions = al.Grid2DIrregular(values=[
        # TODO at execution time: replace with Stage 3 max-log-L predicted positions
        (0.0, 0.0),    # radial image (BCG core)
        (3.5, 0.0),    # tangential arc (approximate from N+23 Fig 1)
    ])
    analysis = al.AnalysisImaging(
        dataset=dataset,
        positions_likelihood=al.PositionsLHPenalty(positions=positions, threshold=0.3),
        use_jax=False,  # pixelized inversions incompatible with JAX tracing — see README §JAX
    )

    search = af.Nautilus(
        path_prefix=str(output_root),
        name="a1201_adapt", unique_tag="a1201_adapt",
        n_live=n_live,
        number_of_cores=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
    )
    return search.fit(model=model, analysis=analysis)


# In main(), change the choices list and add the dispatch branch:
#     choices=["lp", "with_smbh", "with_kin", "adapt"]
# ...
#     elif args.part == "adapt":
#         build_adapt_fit(dataset, args.output_root, n_live=args.n_live)
```

- [ ] **Step 4: Run (PASS)**

```bash
pytest tests/test_a1201_lens_model.py -v
```
Expected: all 6 + 2 new tests pass (8 total).

- [ ] **Step 5: Commit**

### Task 13: Local smoke fit `--part=adapt`

Before committing 24 h of Cannon time, verify the adapt model runs at all
locally (low n_live + small mask).

- [ ] **Step 1: Local smoke**

```bash
conda run -n autolens python \
    private/2303_15514_nightingale2023_abell1201/code/a1201_lens_model.py \
    --part=adapt --n-live=30 --mask-radius=2.5 \
    --dataset-root=private/2303_15514_nightingale2023_abell1201/data/fit_ready \
    --output-root=private/2303_15514_nightingale2023_abell1201/output_local
```

Expected: completes in ~30–60 min; no crash; output contains
`samples_summary.json` and `image/fit_subplot.png`.

Common failure modes:
- **Mesh collapse**: all source-plane pixels at one point. Symptom: chi²/N
  worsens vs Stage 1, samples_summary shows tiny `signal_scale`. Fix: tighten
  the `signal_scale` prior lower bound.
- **Position-likelihood penalty firing**: the positions argument keeps
  rejecting samples → ~zero acceptance. Symptom: `search.log` shows
  `acceptance_rate ≈ 0`. Fix: widen the threshold (0.3 → 0.5) or replace
  hard-coded positions with Stage 3 max-log-L predicted positions.
- **JAX tracing error**: confirm `use_jax=False` in `AnalysisImaging`.

- [ ] **Step 2: Eyeball residuals**

Open `output_local/.../image/fit_subplot.png`. The pixelised source should
have **resolved** the radial image near the BCG core — the bright residual
spot present in Stage 1 should be gone.

### Task 14: Cannon submit Stage 4 (chained from Stage 3) + audit

- [ ] **Step 1: Push + submit**

```bash
bash Modules/10_Cluster_Computing/scripts/push_to_cannon.sh --go
ssh cannon "cd /n/holystore01/LABS/hernquist_lab/Lab/rcordova/learning_to_autolens && \
            STAGE3_SUMMARY=\$(ls private/2303_15514_nightingale2023_abell1201/results/a1201_with_kin/*/files/samples_summary.json | head -1) && \
            sbatch --parsable --export=ALL,STAGE=adapt,CHAIN_FROM=\$STAGE3_SUMMARY \
                   --job-name=a1201_adapt --time=36:00:00 \
                   private/2303_15514_nightingale2023_abell1201/submit_a1201.slurm"
```

Expected wall: ~24–36 h on 32 cores at n_live=400. Pixelised inversions are
~2× slower per likelihood call than parametric.

- [ ] **Step 2: Monitor for mesh-collapse / pos-penalty failures**

Mid-run, when the job has been running ≥ 1 h:

```bash
ssh cannon "tail -50 /n/holystore01/LABS/hernquist_lab/Lab/rcordova/learning_to_autolens/logs/a1201_adapt_<jobid>.out | grep -E 'acceptance|signal_scale|mesh'"
```

If acceptance < 1% sustained, kill the job and revisit Task 13's failure-mode list.

- [ ] **Step 3: Pull + audit**

Audit bar for Stage 4:
- chi²/N ≤ 1.2 (pixelised source should drive it lower than parametric Stage 3)
- max|res| ≤ 4σ on the masked region (radial image residual now resolved)
- M_BH posterior 1σ contains Nightingale+2023's 3.27 × 10¹⁰ M_sun → **strict-PASS**
- σ_v_predicted: still 285 ± few km/s
- ΔlogZ vs Stage 3 ≥ +5 (pixelised source + same data, much better fit)

If Stage 4 lands strict-PASS, Spec 05 is **complete**.

---

## Phase 5: Promotion + headline notebook + Module 15 update

### Task 15: Build the M_BH headline notebook `03_a1201_mbh_recovery.ipynb`

**Files:**
- Create: `private/2303_15514_nightingale2023_abell1201/notebooks/03_a1201_mbh_recovery.ipynb`

- [ ] **Step 1: Build the notebook (8 cells)**

  1. Markdown — Reproduction summary, comparison target (3.27 ± 2.12 × 10¹⁰ M_sun).
  2. Code — Load `samples.csv` for all 4 stages from `results/a1201_*/`.
  3. Code — Run `extract_mbh.compute_mbh_from_samples()` on each stage's samples
     where `smbh.einstein_radius` exists (Stages 2/3/4); plot the M_BH
     posterior overlay (parametric Stage 2 vs +kinematic Stage 3 vs
     pixelised Stage 4).
  4. Code — Overlay vertical bands at Nightingale+2023 1σ (3.27 ± 2.12) and 2σ.
  5. Markdown — Verdict: STRICT-PASS / SOFT-PASS / FAIL per the rule in Task 11.
  6. Code — Side-by-side residual maps: Stage 1 (lp) vs Stage 4 (adapt) — visual
     demonstration that the pixelized source absorbed the radial image.
  7. Code — Corner plot of (γ′, M_BH, σ_v) showing the kinematic break of the
     γ′–M_BH degeneracy.
  8. Markdown — Methodology lessons, deferred questions, link to Module 15 update.

- [ ] **Step 2: Execute headlessly**

```bash
jupyter nbconvert --to notebook --execute \
    --output 03_a1201_mbh_recovery_executed.ipynb \
    --ExecutePreprocessor.kernel_name=autolens \
    notebooks/03_a1201_mbh_recovery.ipynb
```

- [ ] **Step 3: Export the headline figure to `Figures/` (committed)**

```bash
cp notebooks/03_a1201_mbh_recovery_files/figure_mbh_overlay.pdf \
   Figures/spec05_a1201_mbh_overlay.pdf
git add Figures/spec05_a1201_mbh_overlay.pdf
```

### Task 16: Promote Module 15 with the Abell 1201 cap-stone

**Files:**
- Modify: `Modules/15_Radial_Arcs_Caustic_Topology/15_radial_arcs.ipynb`
  (add §"Real-data: Abell 1201 UMBH detection (Nightingale+2023)")
- Modify: `Notes/15_Radial_Arcs/15_radial_arcs_theory.tex`
  (add an aside on Nightingale+2023 as the published-result instance)

- [ ] **Step 1: Add §6 to the Module 15 notebook**

New section after the existing §5 "Hand-off to Module 13":
  - Markdown — Real-data instance: Nightingale+2023 detected M_BH = 3.27 ×
    10¹⁰ M☉ via the radial-image methodology this module derives. Pixelised
    AdaptiveBrightness source was the methodological contribution.
  - Code — Load `Figures/spec05_a1201_mbh_overlay.pdf` (or the headline corner
    PNG from the Spec 05 notebook).
  - Markdown — Reproduction note: our depth-C reproduction lands at M_BH =
    X.XX × 10¹⁰ (1σ); overlap with published value: yes/no/marginal.

- [ ] **Step 2: Re-execute Module 15 headlessly to keep ship-set in sync**

```bash
jupyter nbconvert --to notebook --execute \
    --output Modules/15_Radial_Arcs_Caustic_Topology/15_radial_arcs.ipynb \
    --ExecutePreprocessor.kernel_name=autolens \
    Modules/15_Radial_Arcs_Caustic_Topology/15_radial_arcs.ipynb
```

- [ ] **Step 3: Update LaTeX theory note**

Add to `Notes/15_Radial_Arcs/15_radial_arcs_theory.tex` a sub-section
\subsection{Real-data: the Nightingale+2023 Abell 1201 UMBH} citing the
arxiv ID and our reproduction's M_BH value.

```bash
cd Notes/15_Radial_Arcs && pdflatex 15_radial_arcs_theory.tex
# verify the PDF rebuilt with the new section
```

### Task 17: Update Examples/radial_arc_smbh + RELEASE_NOTES + handoff

**Files:**
- Modify: `Examples/radial_arc_smbh/README.md`
- Create: `RELEASE_NOTES_v0.98.md` (if Phase 4 strict-PASS) OR append to v0.97
- Create: `Modules/10_Cluster_Computing/HANDOFF_<YYYY-MM-DD>.md`

- [ ] **Step 1: Update `Examples/radial_arc_smbh/README.md`**

In the §"Bridge to depth-C" section, replace the "FUTURE TODO" reference
to Abell 1201 with a "DONE — see Spec 05 reproduction" pointer and link to
the headline figure.

- [ ] **Step 2: Release notes entry**

If Stage 4 strict-PASS landed: this is a v0.98 ship. Create
`RELEASE_NOTES_v0.98.md` with the structure of `RELEASE_NOTES_v0.96.md`:
headline ship-set, what closed, what defers, the chi²/M_BH/σ_v table from
the headline notebook.

If only Stages 1–3 landed strict-PASS (parametric): append to v0.97 notes
as a "research-in-progress" entry naming Phase 4 (adapt) as the v0.99 gate.

- [ ] **Step 3: Handoff doc**

`Modules/10_Cluster_Computing/HANDOFF_<YYYY-MM-DD>.md` per the cadence in
the project CLAUDE.md "Handoff cadence" section. Include the Spec 05 verdict,
the M_BH headline number with 1σ + 2σ, comparison verdict vs Nightingale+2023,
and the next priorities.

- [ ] **Step 4: Update project memory**

Add to `~/.claude/projects/-Users-rosador-Documents-AGEL-Learning-to-Autolens/memory/`:
- `project_spec05_a1201_outcome.md` — the M_BH number, verdict, lesson re:
  pixelised vs parametric source for radial-image detection
- Update `MEMORY.md` index

---

## Verification (full Spec 05 ship)

End-to-end strict-PASS requires ALL of:

1. **Stage 1 (lp)**: chi²/N ≤ 2.0, max|res| ≤ 6σ outside central 0.5″,
   θ_E ∈ (3.0, 5.0)″, slope ∈ (1.7, 2.3) → soft bar; radial-image residual visible
2. **Stage 2 (with_smbh)**: chi²/N ≤ 1.4, max|res| ≤ 5σ, ΔlogZ vs Stage 1 ≥ +3
3. **Stage 3 (with_kin)**: chi²_imaging/N ≤ 1.4, σ_v_pred = 285 ± few km/s,
   M_BH posterior tightens (γ′–M_BH degeneracy visibly cut in corner.pdf)
4. **Stage 4 (adapt, stretch)**: chi²/N ≤ 1.2, max|res| ≤ 4σ,
   **M_BH 1σ contains Nightingale+2023's 3.27 × 10¹⁰ M☉**, ΔlogZ vs Stage 3 ≥ +5
5. **Reproduction headline figure** committed to `Figures/`
6. **Module 15 §"Real-data: Abell 1201"** ships in the next tagged release
7. **`Examples/radial_arc_smbh/`** README depth-C bridge updated to "DONE"

**Strict-PASS = (1)+(2)+(3)+(4) all met → v0.98 candidate**
**Soft-PASS = (1)+(2)+(3) met, (4) FAIL → v0.97 research-in-progress, (4) is the v0.99 gate**
**FAIL = any of (1)+(2)+(3) FAIL → systematics investigation (PSF, mask, cutout)**

---

## Sequencing

- **Week 1** (Phase 1 + 2): data prep audit (~½ day), empirical PSF (~½ day),
  local smoke fit (1 day), tests + walkthrough notebooks (1 day). Mostly done; just close.
- **Week 2** (Phase 3 stages 1+2): Stage 1 submit + audit + chainer (~1 day +
  6 h Cannon wall); Stage 2 submit + audit (~1 day + 12 h wall).
- **Week 3** (Phase 3 stage 3 + Phase 4 if proceeding): Stage 3 (~1 day + 24 h);
  Stage 4 driver + smoke + Cannon (~3 days including 24-36 h wall).
- **Week 4** (Phase 5): headline notebook + Module 15 update + promotion (~2 days).

Total: ~4 weeks elapsed; ~3.5 Cannon-days of compute. Depends on siag_lab
queue priority (currently waiting on `rarc_kin3`, `qtd_h0_kin2`, `p1_pop_autofit`
to land first per fairshare policy in CLAUDE.md).

---

## Out-of-scope

- Independent kinematic reduction (KCWI / VIMOS); use Smith+2017 σ_v=285±5 km/s
- Cluster-scale lensing (BCG-scale only; the lensing arc within ~5″ of the BCG)
- Multi-band joint fit (F390W + F814W); F814W only, per N+23 §3 primary fit
- Cosmography-marginalised M_BH (fix to Planck18 H0=67.4 or FlatLambdaCDM(70, 0.3) — N+23 uses 67.4)
- Watson AGEL pipeline drizzle (deferred to v0.99 paper-grade rerun; Stage-0 uses HLA defaults)

---

## Self-review notes (2026-05-18 rewrite)

Plan rewritten 2026-05-18 from the 259-line / 10-step skeleton. Changes:
- Phases 1+2 marked `[x]` where already shipped; only the genuine gaps (Task 2
  empirical PSF, Tasks 4+5 walkthrough notebooks) have new steps.
- Phase 3 expanded from 1 task / 3 steps to 6 tasks / 20+ steps with concrete
  slurm + chained-prior implementation + per-stage audit bars.
- Phase 4 promoted from "no tasks, just discussion" to 3 fully-fleshed tasks
  (12-14) with adapt driver implementation, local smoke, and Cannon submit
  + audit.
- Phase 5 added — promotion, notebook, Module 15 update, release notes,
  memory hygiene.
- Slurm template explicitly does NOT use a personal miniforge path (lesson
  from `p1_pop_autofit` 2026-05-18 crash at line 23) — uses Cannon-canonical
  `/n/sw/Miniforge3-25.3.1-0/...`.
- Account/partition routing follows the policy codified in CLAUDE.md and
  `feedback_cannon_account_routing.md` memory: siag_lab / siag (primary analysis).
- Total: 17 tasks, ~55 steps, ~700 lines. In line with Specs 01-03 grain.
