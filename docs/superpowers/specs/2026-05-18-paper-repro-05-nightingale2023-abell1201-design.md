# Paper-Repro Spec 05 — Nightingale+2023 Abell 1201 UMBH (FUTURE TODO)

**Date drafted:** 2026-05-18 (added after specs 00-04 were already approved)
**Reproduction target:** arxiv [2303.15514](https://arxiv.org/abs/2303.15514) — Nightingale, Smith, Qin, He, Amvrosiadis, Bharath, Cole, Frenk, Li, Massey, Robertson 2023
**MNRAS:** 521:3, 3298 (March 2023)
**Headline result:** Detection of an **ultramassive black hole (M_BH ≈ 3.3 × 10¹⁰ M☉)** in the central galaxy of the cluster Abell 1201 (z = 0.169), via a **radial-image** signature in the strong-lensing imaging. Uses PyAutoLens with the AdaptiveBrightness regularization scheme.

**Status:** FUTURE TODO — added to this program 2026-05-18 at user's request. NOT in the active v0.97-v0.98 implementation window. Spec'd here so the program documentation is complete and future-session entry is unambiguous.

---

## 1. Context

The Nightingale+2023 Abell 1201 result is **directly downstream of the radial-arc + SMBH methodology** we've spent v0.96 + v0.97 building (Module 15 + `Examples/radial_arc_smbh`):
- A radial image (= radial arc near the central point of the lens) sets a sharp constraint on inner mass slope
- A central UMBH steepens the inner mass profile beyond what a pure power-law would predict
- The combination of (radial image position + magnification + brightness) constrains M_BH directly from imaging when the BH is massive enough — exactly the methodology our `Examples/radial_arc_smbh` operationalises on a synthetic mock.

**Pedagogical value:** Nightingale+2023 is the **published-result counterpart** of our synthetic-mock demonstration. Reproducing it would:
1. Validate our `Examples/radial_arc_smbh` methodology on real published data
2. Land a strict-PASS PyAutoLens fit on a paper-cited UMBH result
3. Become the depth-C application of Module 15 + Module 17 (when those ship)

**Stack:** PyAutoLens native (Nightingale IS the PyAutoLens lead author). The autolens-vs-Herculens cross-validation question is **not relevant** here since the paper itself uses PyAutoLens — we test whether we can reproduce HIS PyAutoLens result.

## 2. Goals

Verified against the N+23 body via ar5iv on 2026-05-19 — see §6 (Methodology
correction) below. The goal set was rewritten after we discovered that N+23
does NOT use σ_v in their likelihood.

### 2.1 Faithful-reproduction track

- Reproduce N+23's headline **M_BH = 3.27 ± 2.12 × 10¹⁰ M☉** (Abstract /
  §4.2) at >3σ confidence (their detection threshold ΔlnZ = 4.5 per §3.9;
  their actual headline ΔlnZ = +100.58 for 3-Sersic + SMBH F390W per Table 4).
- Reproduce the **imaging-only methodology** N+23 uses to break the
  γ′–M_BH degeneracy:
    1. **Counter-image flux suppression** (§4.2, Figs 6–7) — pixelised
       Voronoi + AdaptiveBrightness source resolves a counter-image that
       suppresses cleanly when an SMBH is added.
    2. **Mass–light coaxiality argument** (§4.3) — alternative high-density
       BPL with no SMBH fits only at ≥100 pc mass-light offsets (unphysical).
- Use the result as the **depth-C application** of `Examples/radial_arc_smbh`
  — flipping that example from synthetic-mock-only to real-target-validated.

### 2.2 Methodology-extension track (NOT in the paper)

- Add the **Phase 3 σ_v Jeans factor** (`_jeans_sigma_v.AnalysisKinematics`)
  as an INDEPENDENT cross-check using Smith+2017 KCWI σ_v = 285 ± 5 km/s.
  This is **our methodological contribution beyond N+23** — the paper itself
  does not use kinematics in the likelihood (§1 mention only). If our +σ_v
  variant tightens the M_BH posterior consistent with N+23's value, it
  validates the kinematic-break demonstration our `radial_arc_smbh`
  synthetic example shipped in v0.96.

## 3. Non-goals

- Independent KCWI / VIMOS / MUSE kinematic reduction — use published σ_v
  (only relevant for our §2.2 extension track, not the paper-faithful run).
- Cluster-scale lensing fit (just the BCG + radial image; galaxy-scale only).
- BPL alternative mass-model variant — N+23 ran this as a ΛCDM-vs-alternative
  control (§4.3); we defer it to v0.99 unless our M_BH posterior fails to
  recover N+23's value, in which case the BPL alternative becomes diagnostic.

## 4. Architecture

```
private/2303_15514_nightingale2023_abell1201/
├── PAPER_NOTES.md                                  ← claims, methods, data sources
├── data/
│   ├── download_a1201.py                           ← MAST HST WFC3-UVIS (high S/N imaging)
│   ├── agel_pipeline_reduction.py                  ← Watson pipeline on raw FLT files
│   └── README.md
├── code/
│   ├── a1201_lens_model.py                         ← PyAutoLens with AdaptiveBrightness
│   ├── radial_image_likelihood.py                  ← per the paper's modelling
│   └── run_chain.py                                ← Cannon Nautilus driver
├── notebooks/
│   ├── 01_a1201_radial_image.ipynb                 ← walkthrough
│   └── 02_mbh_recovery.ipynb                       ← validation against published M_BH
└── results/
    └── a1201_main_fit/
```

## 5. Data

**HST WFC3/UVIS** imaging of Abell 1201 BCG region. Nightingale+2023 used new observations (high spatial resolution, deep). Proposal IDs to be identified — search MAST for "Abell 1201" or "A1201" or directly RA/Dec.

**Source spec-z** for the radial-image lensed source: extracted from Nightingale+2023 directly (or from earlier Smith+2017 paper if needed).

**Kinematics**: Smith, Lucey & Edge 2017b KCWI σ_v = 285 ± 5 km/s (BCG
aperture). **Note**: N+23 cites this in §1 (Introduction) as motivating
background ONLY — it is NOT a likelihood input in their analysis. Our use
of it is the §2.2 methodology-extension track, not a reproduction of N+23.

**Reduction:** Use Spec 00 §6.9 (Watson pipeline) to re-drizzle the raw FLT files — even more critical here than for J0946 because the radial image is FAINT and lives within ~1″ of the bright BCG core, so cosmic-ray cleanliness directly limits detectability.

## 6. Methodology correction (added 2026-05-19 after paper-body verification)

Direct inspection of the N+23 body via ar5iv on 2026-05-19 corrected several
assumptions encoded in the original draft of this spec. The verified
methodology:

### 6.1 Modelling chain (§3.8 SLaM pipelines)

N+23 runs a **3-pipeline** SLaM chain, not the 4-stage chain claimed in
earlier drafts:

1. **Source pipeline** (§3.4): pixelised source via Voronoi + `al.reg.AdaptiveBrightness`
   + total mass (PL, γ=2 fixed) + double-Sersic lens light.
2. **Light pipeline** (§3.1, Table 1): Bayesian comparison of 5 lens-light
   models (single/double/triple-Sersic ladder + others). **3-Sersic wins**
   and drives downstream.
3. **Mass pipeline** (§3.8 / §4.2 / §4.3): each candidate mass model
   (decomposed Sersic+NFW+shear, total PL, total BPL) fit **with and
   without** a central `al.mp.PointMass`. Bayes-factor between +SMBH and
   −SMBH variants produces the headline detection.

### 6.2 Detection driver (Table 4)

| Model variant | F390W ΔlnZ (SMBH vs no-SMBH) | F814W ΔlnZ |
|---|---|---|
| 3-Sersic light, PL mass | **+100.58** ← headline (Table 4) | ~+3 (marginal) |
| BPL alternative | matches PL+SMBH ΔlnZ but requires ≥100 pc mass-light offset (§4.3) → unphysical, rejected |

N+23's §3.9 states the 3σ detection threshold is ΔlnZ = 4.5. The headline
ΔlnZ = 100 is decisive. **F390W (7150 s) drives the detection**; F814W
(1009 s) alone is marginal.

### 6.3 How the γ′–M_BH degeneracy is broken (NOT via kinematics)

Two **imaging-based** arguments (§4.2–§4.3):
- **Counter-image flux suppression** (§4.2, Figs 6–7): models without SMBH
  leave extraneous reconstructed-source flux that the SMBH variant cleans.
- **Mass–light coaxiality** (§4.3): the alternative high-density BPL model
  fits only at ≥100 pc mass-light centre offsets, ruled out physically.

σ_v from Smith+2017 appears ONLY in §1 (Introduction) as background
motivation, not in the likelihood. Our σ_v Jeans factor (§2.2) is OUR
methodology contribution.

## 7. Tools

**Faithful-reproduction track** (PyAutoLens-native, matching N+23):
- `al.mesh.Voronoi` mesh + `al.reg.Adapt` regularization (the autolens
  2026.4 rename of N+23's `al.reg.AdaptiveBrightness`; verified in Module 09
  pattern, lines 1501-1518 of `09_mge_linear_light_profiles.ipynb`).
- `al.mp.PowerLaw` + `al.mp.PointMass` lens mass.
- Multi-Sersic (1/2/3) lens light via our `_build_lens_galaxy_model(..., n_light_components=N)`
  refactor (shipped 2026-05-19, tied centres across all components).
- SLaM-style chained Cannon submissions per Stage (via the shipped
  `submit_a1201.slurm` + `chain_priors_from_lp.py`).
- F390W band primary, F814W as cross-check (per Table 4).

**Methodology-extension track** (our §2.2 contribution):
- Phase 3 `_jeans_sigma_v.AnalysisKinematics` for σ_v constraint via
  `af.FactorGraphModel(imaging, kinematic)`. NOT used by N+23.

## 8. Computational budget

Single-target, single-fit, ~24h on Cannon (siag_lab CPU partition, 32 cores, --mem=192G). Cheaper than P2 / P3 because no multi-plane geometry. Total: ~1 Cannon-day.

## 9. Pedagogical promotion

When this lands:

- **Update `Examples/radial_arc_smbh/`** README: add §"Real-target validation: Abell 1201" pointing at this paper as the canonical published result.
- **Update Module 15 (Radial Arcs)**: add §"Real-data: Abell 1201 UMBH detection (Nightingale+2023)" with the actual posterior corner from this reproduction as the cap stone figure.
- **New `Examples/abell1201_umbh/`** OR extend `Examples/radial_arc_smbh/` with a `--target=abell1201` rung. (Decide based on how much new architecture vs reuse — likely a sister-Example since the lens is at z=0.169 not z=0.7, different physical regime.)

## 10. Timeline (when activated)

- Day 1: Run Spec 00 §6.9 Watson pipeline on the Abell 1201 raw FLT files
- Day 2: Build the autolens model in `a1201_lens_model.py`; smoke fit on laptop
- Day 3: Cannon submit; ~24h wall
- Day 4: Audit + reconcile M_BH posterior against Nightingale+2023's published value
- Day 5: Module 15 §"Real-data" update + Examples promotion
- Total: ~1 week + Cannon wall time

## 11. Pre-requisites

Before activating this spec:
- Specs 00-04 should be substantially complete (or at least Spec 00's Watson pipeline integration shipped)
- Phase 3 `_jeans_sigma_v.AnalysisKinematics` strict-PASS on Cannon (already in flight 2026-05-18)
- `Examples/radial_arc_smbh/ --part=with_kinematics` strict-PASS Cannon result audited (Cannon job 13649652 running)

When those pre-reqs are met, file this spec for re-activation.

## 12. Notes on the related "Probing Dark Matter Substructures with Free-Form Modelling: A Case Study of the 'Jackpot' Strong Lens" ([arxiv 2504.19177](https://arxiv.org/html/2504.19177v1))

A 2025 free-form modelling re-analysis of the Jackpot lens — relevant to Spec 02 (P2 TSPL Jackpot). Not part of this spec but worth tracking when the program reaches the eventual cross-validation discussion. Document as a "see also" in Spec 02 references.
