# AGEL Data Quickstart

**Audience:** a student who already knows PyAutoLens (you've worked through Modules 01-09 and at least one example) and now wants to fit a **new AGEL target** end-to-end.

**This is not a PyAutoLens tutorial.** It's an AGEL-data-specific recipe — what changes about your workflow when you move from clean simulated mocks to the real HST cutouts AGEL ships.

**Time budget:** ~2 hours from "I have an AGEL target name" to "I'm submitting a Cannon fit." Plus 2-12h Cannon wall time depending on the target.

---

## What's different about AGEL data vs. mocks

You've been working with simulated mocks where:
- Truth parameters are known
- The PSF is a clean Gaussian or exact instrumental
- Noise is uncorrelated Gaussian
- Redshifts are exact
- No cosmic-ray survivors, no foreground galaxies, no diffraction spikes

AGEL gives you **real HST imaging** where every one of those assumptions breaks. The good news: PyAutoLens handles all of it once you know which knobs to turn. This guide walks you through every knob.

---

## Step 0 — Pick a target

AGEL DR2 catalog of confirmed strong lenses lives at `~/Documents/AGEL/20250910-keerthi-Keck-AGELDR2-main/` (or wherever Keerthi's repo is on your laptop). The catalog gives you:

- `target_name` (e.g. `AGEL013322-125201A`)
- `RA, Dec` (J2000)
- `z_lens, z_source` (spectroscopic, with uncertainties)
- HST proposal ID + filename
- `axis_class` (single-deflector / compound / DSPL / group-scale) — picks your example template

**This guide uses `AGEL013322-125201A`** (also catalogued as `DCLS0133-1252`) as a worked example. It's a galaxy-scale single-deflector lens at z_L=0.30, z_S=1.6, observed in HST/ACS F606W (proposal 17307).

---

## Step 1 — Get the HST data on disk

The AGEL collaboration has a shared HST archive. Drizzled images (`*_drc.fits`) live at one of:

```
~/Documents/AGEL/<target>_HST_ACS_F606W/   # individual target dirs
/n/holystore01/.../AGEL_HST/                # if you're on Cannon already
```

You want the `_drc.fits` (drizzled, calibrated, geometric-distortion-corrected) **plus** the corresponding `_point-cat.ecsv` and `_segment-cat.ecsv` (HST drizzle pipeline source catalogs). You'll use the catalogs for both target identification AND empirical PSF construction.

If your target is missing locally: either pull from the AGEL S3 bucket (Keerthi has the credentials) or pull from MAST directly with:
```bash
astroquery.mast.Observations.download_products(...)
```

---

## Step 2 — Extract the cutout

PyAutoLens fits 100×100 to 200×200 pixel stamps, not full 5000×5000 HST frames. The extraction recipe:

```bash
# Adapt Examples/agel_real_target/data/extract_cutout.py to your target
cp Examples/agel_real_target/data/extract_cutout.py /tmp/extract_<TARGET>.py
$EDITOR /tmp/extract_<TARGET>.py
# Edit:
#   SRC = path to your target's _drc.fits
#   LENS_PIXEL = (x, y) of the lens in the full frame  
#   CUTOUT_HALF = 100 (gives a 200×200 stamp, ~10″×10″ at 0.05″/px)
python /tmp/extract_<TARGET>.py
```

To find `LENS_PIXEL`: open the `_segment-cat.ecsv` catalog, filter to RA/Dec near your target's coordinates, the X-Center / Y-Center columns are the pixel coordinates. Or use `aplpy` to load the FITS + click on the lens.

The script writes `image.fits`, `noise_map.fits` (= `1 / sqrt(WHT)`), and a `metadata.json`. If you're following the `agel_real_target` example pattern, drop these into `Examples/<your_target>/data/`.

---

## Step 3 — Build an empirical PSF

**The placeholder Gaussian PSF is wrong for HST.** Drizzle resampling broadens the PSF beyond the diffraction limit; instrumental wings + spatial dependence matter at sub-arcsec lensing scales.

Use the recipe from `Examples/agel_real_target/data/build_empirical_psf.py`:

```bash
python Examples/agel_real_target/data/build_empirical_psf.py
```

This:
1. Reads your `_point-cat.ecsv`
2. Filters for clean point sources (Flags=0, S/N>50, CI in [1.05, 1.25])
3. Drops blended sources (any neighbor within 60 px)
4. Drops sources within 200 px of the lens (avoid drizzle-position bias)
5. Extracts 51×51 stamps, sub-pixel re-centres each, sigma-clips, median-stacks
6. Writes `data/psf.fits` (FWHM ≈ 0.113″ for ACS F606W)

If you only have <3 isolated stars after filters: relax the isolation cut to 40 px or use a TinyTim model instead. AGEL collaborator Keerthi has TinyTim configs.

**Backup the placeholder Gaussian PSF first** (the script does this if `data/psf.fits` already exists).

---

## Step 4 — Identify + mask hot pixels

This is the lesson Module 09's mocks don't teach you. Real HST data has cosmic-ray survivors that drizzle didn't reject — they show up as ~30σ residuals after a fit and contaminate parameter posteriors.

The recipe lives in `Examples/agel_real_target/01_agel_real_target.ipynb` §1.5:

```python
ds_raw = al.Imaging.from_fits(...)
sn = np.abs(ds_raw.data.native) / ds_raw.noise_map.native
outliers = sn > 8.0   # 8σ; well above the Bonferroni floor of ~4.3σ at 9k pixels
print(f"Found {outliers.sum()} hot pixels.")

base_mask = al.Mask2D.circular(...)
combined = np.array(base_mask) | outliers
mask_clean = al.Mask2D(mask=combined, pixel_scales=ds_raw.pixel_scales)
dataset = ds_raw.apply_mask(mask=mask_clean)
```

For AGEL013322 this catches 44 pixels. Do this BEFORE running the fit — it shaves ~hours off your debugging if a single bright pixel was driving a parameter.

---

## Step 5 — Get the redshifts (and their uncertainties)

The AGEL DR2 catalog gives spectroscopic z_L and z_S. These have **finite uncertainties** — typically ~0.1% on z_L (galaxy-scale spectra are well-measured) and 1-3% on z_S (the source spectra are faint).

For most fits: hold the redshifts **fixed** at the catalog values. They're tight enough that fitting them adds dimensions for nothing.

For cosmography work (TDCOSMO, β-cosmography): marginalize over redshift with a Gaussian prior centered on the catalog value, sigma = the catalog uncertainty.

You'll find the redshifts in the Keck spectroscopic file:
```
~/Documents/AGEL/20250910-keerthi-Keck-AGELDR2-main/<your_target>/spectra/redshift.txt
```

Or in the AGEL DR2 master catalog CSV. Cross-check both — Keerthi's per-target dir has the latest fit.

---

## Step 6 — Pick an example template + compose your model

Match your target's `axis_class` to one of the existing `Examples/`:

| Your target's class | Template to fork |
|---|---|
| Single deflector + 1 source | `Examples/agel_real_target/` (this is the canonical real-data template) |
| Two deflectors at different z | `Examples/compound_lens/` |
| One deflector + 2 sources at different z | `Examples/double_source_plane/` |
| BGG + visible satellites (group) | `Examples/group_scale/` |
| Cluster member with arc | `Examples/cluster_scale/` (research-in-progress; use with care) |

Copy the relevant `fit_example_<template>.py` to a new `fit_example_<your_target>.py`, edit:
- Dataset path → `Examples/<your_target>/data/`
- `z_l`, `z_s` → from your catalog
- Mass model: start with `al.mp.PowerLaw` (slope free)
- Lens light: start with `al.lp.Sersic`. **If residuals show structure at the lens centre, swap to MGE** (see `Modules/09_MGE_Linear_Light_Profiles/05_mge_recipe.ipynb`)
- Source: start with `al.lp.SersicCore`. **If residuals show arc structure, swap to pixelization** (see `Modules/05_Pixelized_Source_Reconstructions/06_pixelization_recipe.ipynb`)

For a fresh target, **start small**: `n_live=80`, mass=`Isothermal` (not PowerLaw), source=single SersicCore. Make sure it converges before adding complexity. The SLaM recipe (`Examples/compound_lens_zoo/03_slam_recipe.ipynb`) shows the right way to ladder up.

---

## Step 7 — Submit on Cannon

Add slurm routing for your new target. Edit `Modules/10_Cluster_Computing/scripts/submit_cannon.slurm`:

```bash
        agel_<your_target>)
            DATASET_ROOT="${REPO_ROOT}/Examples/agel_<your_target>/data"
            FIT_SCRIPT="${REPO_ROOT}/Modules/10_Cluster_Computing/scripts/fit_example_agel_<your_target>.py"
            ;;
```

Then:
```bash
bash Modules/10_Cluster_Computing/scripts/push_to_cannon.sh --go

ssh cannon "cd \$CANNON_REPO_ROOT && \
  sbatch --time=8:00:00 \
    --export=ALL,EXAMPLE=agel_<your_target>,FIT_EXTRA_ARGS=--part=direct \
    --job-name=agel_<your_target> \
    Modules/10_Cluster_Computing/scripts/submit_cannon.slurm"
```

Wall-time expectation for a single PowerLaw + Sersic-source fit: **2-4 hours** on the `hernquist` partition (32 cores). Larger if you go to MGE light or pixelized source.

---

## Step 8 — Audit

After the job lands:
```bash
bash Modules/10_Cluster_Computing/scripts/pull_from_cannon.sh --go
```

Open `Examples/agel_<your_target>/results/<fit_name>/fit_subplot.png` and apply the bar from `Modules/10_Cluster_Computing/STUDENT_QUICKSTART.md`:

| Metric | strict-PASS | borderline-PASS | SUSPECT | FAIL |
|---|---|---|---|---|
| `chi_squared_per_pixel` | ≤1.3 | ≤1.3 | 1.3-2.0 | ≥2.0 |
| `max_abs_normalized_residual` | ≤4σ | 4-5σ at 9k+ pixels | 5-6σ | ≥6σ |
| Visual residual map | white noise | white noise + isolated outliers | clipped coherent structure | ring/cross/arc |

**For real data, expect borderline-PASS or SUSPECT on the first try.** The most common failure modes:

| Symptom | Likely cause | Fix |
|---|---|---|
| Single >20σ residual at one pixel | Hot pixel survived the mask | Re-run §1.5 hot-pixel detection at lower threshold |
| Coherent ring residual | Mass model too rigid (Iso when it should be PowerLaw, or wrong ellipticity prior) | Free `slope`, widen `ell_comps` priors |
| Lens-centre residual structure | Sersic light profile not flexible enough | Swap to MGE basis (`05_mge_recipe.ipynb`) |
| Arc structure on the lensed source | Source morphology too complex for SersicCore | Swap to pixelization (`06_pixelization_recipe.ipynb`) |
| `chi_squared_per_pixel` < 0.5 | Noise map overestimated (drizzle WHT inverse-variance is conservative) | Multiply `noise_map` by 0.7-0.8 and refit; cosmetically lower chi² but parameter posteriors don't change |
| Burn-in stall (f_live=1.0 for hours) | 30+ free param landscape too wide | Use SLaM staging (`03_slam_recipe.ipynb`); start with tight Gaussian priors |

---

## Common AGEL-specific gotchas

1. **Drizzle correlated noise.** `noise_map = 1/sqrt(WHT)` treats each pixel independently, but drizzle correlates adjacent pixels. For high-precision substructure work, inflate σ by 1.3-1.6 (the drizzle correlation length). For most fits this isn't a problem — the lens dominates over noise.

2. **Foreground stars / galaxies.** Real HST cutouts often have unrelated objects in the mask. Use `data_preparation/optional/mask_extra_galaxies.py` from the autolens_workspace to build a custom mask that excludes them.

3. **Lens-light wing leakage.** AGEL lens galaxies have faint envelopes that bleed into the arc region. If your single Sersic leaves residuals along the arc, the lens light is the problem, not the source. Always check by looking at the "Lens Light Subtracted" panel of `fit_subplot.png`.

4. **Redshift catalog refresh.** The AGEL DR2 catalog redshifts get updated as Keerthi finalizes the spectroscopy. Check the per-target dir for the latest before starting a multi-day Cannon fit — it would be sad to re-run because z_S shifted by 0.05.

5. **Cosmology choice.** Most AGEL papers use FlatLambdaCDM(70, 0.30). Match this in your driver:
   ```python
   from autolens.cosmology import FlatLambdaCDM
   cosmology = FlatLambdaCDM(H0=70.0, Om0=0.30)
   model = af.Collection(galaxies=..., cosmology=cosmology)
   ```
   For **β-cosmography** (multi-source), see `Examples/double_source_plane/02_beta_cosmography.ipynb`.

6. **The 32σ residual is a feature, not a bug** (sometimes). A single bright residual on real HST data usually means a cosmic-ray survivor. The hot-pixel masking recipe handles it. **What's NOT a feature**: a coherent residual ring that gets bigger when you add more parameters — that's a Pattern A/E lens-cosmology degeneracy and means your fit is in a wrong basin.

---

## Quick checklist for a new AGEL target

```
[ ] Pick target from AGEL DR2 catalog
[ ] Download _drc.fits + _point-cat.ecsv + _segment-cat.ecsv
[ ] Find lens (X, Y) in full frame
[ ] Extract 200×200 cutout → image.fits + noise_map.fits + metadata.json
[ ] Build empirical PSF from ~10-15 isolated stars in the same frame
[ ] Run hot-pixel detection (8σ threshold), update mask
[ ] Confirm spectroscopic z_L + z_S from per-target Keck spectra
[ ] Pick example template based on architecture
[ ] Copy + edit fit_example_<template>.py
[ ] Add slurm routing for the new target
[ ] Push to Cannon, submit with --time=8:00:00
[ ] Pull results, audit fit_subplot.png against the bar
[ ] If SUSPECT/FAIL: open the failure-mode table above and ladder up appropriately
```

---

## When you're stuck

1. Re-read `Examples/agel_real_target/01_agel_real_target.ipynb` end-to-end — every gotcha above is encoded as a cell.
2. Check `Examples/agel_real_target/README.md` §Caveats for deeper PSF / noise / mask discussion.
3. The SLaM recipe (`Examples/compound_lens_zoo/03_slam_recipe.ipynb`) is the right path when a single direct fit won't converge.
4. For redshift / spectroscopy questions: Keerthi (Keck DR2 lead) is your contact.
5. For cluster issues: `Modules/10_Cluster_Computing/STUDENT_QUICKSTART.md` "When in doubt" section.
6. For autolens API changes after a version bump: `pip index versions autolens` against your env's Python.
7. Anything else: open a repo issue or email rodrigo.cordova_rosado@cfa.harvard.edu.

---

*Learning to Autolens v0.92-alpha — AGEL Quickstart*
*Rodrigo Córdova Rosado, Harvard CfA*
