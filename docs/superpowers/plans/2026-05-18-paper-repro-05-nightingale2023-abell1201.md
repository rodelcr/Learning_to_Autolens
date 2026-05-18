# Paper-Repro Spec 05 — Nightingale+2023 Abell 1201 UMBH Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reproduce Nightingale+2023's **M_BH = (3.27 ± 2.12) × 10¹⁰ M_sun**
detection in Abell 1201 BCG via a radial-image signature in HST WFC3/UVIS
imaging. Use the exact `Examples/radial_arc_smbh` (v0.96) methodology adapted
to real data with σ_v=285±5 km/s as a kinematic factor.

**Architecture:** 3-stage chained PyAutoLens fit (parametric Sersic source → +
central PointMass → + Jeans σ_v factor), then Stage 4 promote source to
AdaptiveBrightness pixelisation (Nightingale's contribution). PyAutoLens-only;
no Herculens cross-validation (paper is PyAutoLens-native).

**Tech Stack:** PyAutoLens 2026.4 + autofit Nautilus + Phase 3 `_jeans_sigma_v.py`
+ AGEL Watson HST reduction pipeline (Spec 00 §6.9) for clean F814W drizzle.

**Depends on:** Spec 00 (Watson pipeline), v0.96 Module 15 + radial_arc_smbh,
v0.97 Phase 3 _jeans_sigma_v + AnalysisKinematics (save_results stub).

---

## File Structure

```
private/2303_15514_nightingale2023_abell1201/
├── PAPER_NOTES.md                           ← already shipped 2026-05-18
├── data/
│   ├── download_a1201.py                    ← already shipped 2026-05-18
│   ├── raw_flt/                             ← 4 FLT/FLC files (~700 MB)
│   ├── hla/                                 ← HLA drizzles (fallback)
│   └── agel_reduced/                        ← Watson-pipeline drizzles
├── code/
│   ├── a1201_lens_model.py                  ← already shipped 2026-05-18
│   ├── prep_a1201_dataset.py                ← NEW: drizzle → fit-ready FITS
│   └── run_chain.py                         ← NEW: Cannon submit driver
├── notebooks/
│   ├── 01_a1201_data_prep.ipynb             ← walkthrough of drizzle/mask
│   ├── 02_a1201_lp_fit.ipynb                ← Stage 1 results
│   └── 03_a1201_mbh_recovery.ipynb          ← M_BH posterior + comparison
├── tests/
│   ├── test_a1201_lens_model.py
│   └── test_a1201_data_prep.py
└── results/
    ├── a1201_lp/                            ← Stage 1 Cannon output
    ├── a1201_with_smbh/                     ← Stage 2
    └── a1201_with_kin/                      ← Stage 3 (γ′–M_BH break via σ_v)
```

---

## Phase 1: Data prep (laptop, ~1 day)

### Task 1: Download raw FLT (already in progress)

- [x] `data/download_a1201.py` written; running in background 2026-05-18 PM.
  Pulls all FLT/FLC products for proposal 14886 (~700 MB).

- [ ] **Step 1: Verify download completed**

```bash
du -sh private/2303_15514_nightingale2023_abell1201/data/
find private/2303_15514_nightingale2023_abell1201/data -name '*flt.fits' -o -name '*flc.fits' | wc -l
```

Expected: ~700 MB total, ≥4 FLT files (one per visit).

### Task 2: Drizzle FLT → fit-ready FITS via Watson pipeline

**Files:**
- Create: `private/2303_15514_nightingale2023_abell1201/code/prep_a1201_dataset.py`

The Watson pipeline (notebooks `2-Drizzler_Rewritten`, `3-Build_cutouts`,
`4-Pix_scale_change_reprojection` in `private/00_shared_infrastructure/code/agel_hst_reduction/`)
takes raw FLT and produces drizzled DRC at 0.03″/pix with AGEL-tuned CR
rejection. For A1201 first pass we can use the HLA default drizzles
instead; that's a Stage-0 acceptable.

- [ ] **Step 1: Locate / verify HLA F814W drizzled product**

```bash
find private/2303_15514_nightingale2023_abell1201/data -name '*f814w*drc*.fits' | head -3
```

If found: use that. If not: run Watson notebook 2 on the raw FLT manually.

- [ ] **Step 2: Build the cutout + noise_map + psf**

Write `prep_a1201_dataset.py`:

```python
"""prep_a1201_dataset.py — build image.fits + noise_map.fits + psf.fits
from the HLA F814W drizzle product (or Watson AGEL drizzle).

Output:
    image.fits      — 200x200 cutout around the BCG, 0.04"/pix
    noise_map.fits  — RMS from the DRC weight map + readnoise
    psf.fits        — 21x21 Gaussian (sigma=0.04") for first pass
                      (or empirical from a foreground star)
"""
```

Cutout box: 200×200 pixels at 0.04″/pix = 8″ × 8″ — large enough to capture
the lensed arcs + radial image but excluding cluster-scale neighbours.

- [ ] **Step 3: Smoke test**

```bash
python private/2303_15514_nightingale2023_abell1201/code/prep_a1201_dataset.py \
    --hla-path private/2303_15514_nightingale2023_abell1201/data/hla/<DRC.fits> \
    --output-dir private/2303_15514_nightingale2023_abell1201/data/fit_ready/
```

Expected: writes image/noise/psf trio.

---

## Phase 2: Local smoke fit (`--part=lp`)

### Task 3: Local smoke fit on Abell 1201 (parametric Sersic source)

**Files:**
- modify: `private/2303_15514_nightingale2023_abell1201/code/a1201_lens_model.py`
  (already shipped) — verify it works on real data.

- [ ] **Step 1: Run a 20-live-point smoke fit locally**

```bash
mkdir -p private/2303_15514_nightingale2023_abell1201/output_local
conda run -n autolens python \
    private/2303_15514_nightingale2023_abell1201/code/a1201_lens_model.py \
    --part=lp --n-live=20 --mask-radius=3.0 \
    --dataset-root=private/2303_15514_nightingale2023_abell1201/data/fit_ready \
    --output-root=private/2303_15514_nightingale2023_abell1201/output_local
```

Expected: ~10 min wall, fit completes without crash. Posterior obviously
won't be strict-PASS at n_live=20 — this is a smoke gate.

- [ ] **Step 2: Eyeball the residual map**

Open `output_local/<...>/files/fit_subplot.png` and check that the lens
mass + parametric source produces a model image broadly matching the BCG
+ arc geometry. Expect residuals at the radial-image position (this is
what Stage 2 is for).

---

## Phase 3: Cannon strict-PASS chain

### Task 4: Cannon submit script + 3-stage chain driver

**Files:**
- Create: `private/2303_15514_nightingale2023_abell1201/submit_a1201.slurm`
- Create: `private/2303_15514_nightingale2023_abell1201/code/run_chain.py`

Three-stage chain:
1. `--part=lp` (n_live=200, ~6h on 32 CPUs)
2. `--part=with_smbh` initialised from Stage 1 posterior (n_live=250, ~12h)
3. `--part=with_kin` joint imaging + σ_v Jeans (n_live=300, ~24h)

Total Cannon budget: ~42 h on a single 32-core node = 1.75 days. Run as
a chained sbatch (3 jobs with --dependency=afterok).

- [ ] **Step 1: Write the slurm script**

Pattern: same as `private/2307_09271_li2023_cosmography_population/submit_p1_autofit.slurm`
but with `--cpus-per-task=32 --time=2-00:00:00 --mem=64G` and three stages
chained via `--dependency=afterok`.

- [ ] **Step 2: Push + submit**

```bash
bash Modules/10_Cluster_Computing/scripts/push_to_cannon.sh --go
ssh cannon "cd .../learning_to_autolens && sbatch private/2303_.../submit_a1201.slurm"
```

- [ ] **Step 3: Audit the with_kin posterior on theta_E_BH → M_BH**

After Stage 3 lands, extract the M_BH posterior:

```python
from astropy.cosmology import FlatLambdaCDM
import autolens as al, numpy as np, pandas as pd

samples = pd.read_csv('<output>/samples.csv')
theta_E_BH_arcsec = samples['galaxies.lens.smbh.einstein_radius']
cosmo = FlatLambdaCDM(H0=70.0, Om0=0.30)
D_l = cosmo.angular_diameter_distance(0.169).to('kpc').value
D_s = cosmo.angular_diameter_distance(0.451).to('kpc').value
D_ls = cosmo.angular_diameter_distance_z1z2(0.169, 0.451).to('kpc').value
# M_BH = c^2 theta_E^2 D_l D_s / (4 G D_ls)
C_KMS, G = 2.99792458e5, 4.30091e-6  # kpc/Msun (km/s)^2
theta_rad = theta_E_BH_arcsec * np.pi / 180 / 3600
M_BH = C_KMS**2 * theta_rad**2 * D_l * D_s / (4 * G * D_ls)
print(f"M_BH median: {np.median(M_BH):.2e}, 16-84%: "
      f"{np.percentile(M_BH, [16, 84])}")
```

**Verification:** `M_BH ∈ [3.27 ± 2.12] × 10^10 M_sun` (1σ Nightingale+2023).
Strict-PASS = our 1σ contains the published median; soft-PASS = our 2σ contains it.

---

## Phase 4: AdaptiveBrightness pixelised source (stretch — v0.99)

Nightingale's specific methodological contribution is the AdaptiveBrightness
regularisation. PyAutoLens 2026.4 exposes this via `al.reg.Adapt` (renamed
from `AdaptiveBrightness` in earlier versions — confirmed via
`grep -rn "Adapt" /opt/anaconda3/envs/autolens/lib/python3.12/site-packages/autoarray/inversion/regularization/`).

Replace parametric Sersic source with:

```python
import autolens as al
source = af.Model(
    al.Galaxy, redshift=Z_SOURCE,
    pixelization=af.Model(al.Pixelization,
                          mesh=af.Model(al.mesh.Voronoi),
                          regularization=af.Model(al.reg.Adapt)),
)
```

This is the closest reproduction of Nightingale's headline pipeline. Defer
to v0.99 — not blocking the M_BH detection (parametric Sersic plus PointMass
already exposes the central deflection).

---

## Verification (full Spec 05 ship)

1. **Stage 1 lp fit**: chi²/N ≤ 2.0, max\|res\| ≤ 6σ on the cutout
2. **Stage 2 with_smbh**: positive ΔlogZ vs Stage 1 (≥+3 = 95% confidence
   for a black hole)
3. **Stage 3 with_kin**: M_BH posterior 1σ overlaps Nightingale+2023's
   3.27 × 10¹⁰ M_sun ± 2.12 × 10¹⁰
4. **σ_v_pred at MaxLike**: 285 ± a few km/s (matches Smith+2017 within
   the kinematic likelihood's own constraint)

---

## Sequencing

- **Week 1**: Phase 1 + Phase 2 (data prep + local lp smoke fit). Currently
  in flight 2026-05-18 PM.
- **Week 2**: Phase 3 (3-stage Cannon chain) — depends on v0.97 jobs (rarc_kin3,
  qtd_h0_kin2) landing first so fairshare frees up.
- **Week 3+**: Phase 4 (AdaptiveBrightness, optional).

---

## Out-of-scope

- Independent kinematic reduction (KCWI / VIMOS); use Smith+2017 published
  σ_v = 285 ± 5 km/s
- Cluster-scale lensing (this paper is BCG-scale only)
- The full radial-magnification analysis at the radial-image position
  (Nightingale+2023 Fig 3) — qualitatively reproduced via the radial-arc
  signature; quantitatively this needs the Stage 4 pixelisation
