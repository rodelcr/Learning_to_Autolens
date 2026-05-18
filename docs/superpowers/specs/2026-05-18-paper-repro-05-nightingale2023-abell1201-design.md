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

- Reproduce **M_BH ≈ 3.3 × 10¹⁰ M☉** at the published significance with PyAutoLens (our v0.96+ infrastructure)
- Validate that the Phase 3 `_jeans_sigma_v.py` + kinematics infrastructure (already shipped) reproduces the σ_v constraint Nightingale+2023 uses to break the γ′–M_BH degeneracy in a real-data setting
- Use the result as the depth-C application of `Examples/radial_arc_smbh` — flipping `radial_arc_smbh` from synthetic-mock-only to real-target-validated

## 3. Non-goals

- Independent KCWI / VIMOS / MUSE kinematic reduction — use published σ_v
- Cluster-scale lensing fit (just the central galaxy + radial image; this is galaxy-scale, NOT the full cluster)

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

**Kinematics**: published σ_v profile or central σ_v aperture from one of Smith's prior papers (Smith+2017 KCWI? VIMOS?).

**Reduction:** Use Spec 00 §6.9 (Watson pipeline) to re-drizzle the raw FLT files — even more critical here than for J0946 because the radial image is FAINT and lives within ~1″ of the bright BCG core, so cosmic-ray cleanliness directly limits detectability.

## 6. Tools

PyAutoLens native — no Herculens cross-validation needed (paper is PyAutoLens).

Key autolens features used:
- `al.pix.Mesh` + `al.reg.AdaptiveBrightness` regularization (Nightingale+2023's contribution)
- `al.mp.PowerLaw` lens + `al.mp.PointMass` central BH
- SLaM staged chain methodology (already in our Module 04 + Examples)
- Phase 3 `_jeans_sigma_v.AnalysisKinematics` for σ_v constraint

## 7. Computational budget

Single-target, single-fit, ~24h on Cannon (siag_lab CPU partition, 32 cores, --mem=192G). Cheaper than P2 / P3 because no multi-plane geometry. Total: ~1 Cannon-day.

## 8. Pedagogical promotion

When this lands:

- **Update `Examples/radial_arc_smbh/`** README: add §"Real-target validation: Abell 1201" pointing at this paper as the canonical published result.
- **Update Module 15 (Radial Arcs)**: add §"Real-data: Abell 1201 UMBH detection (Nightingale+2023)" with the actual posterior corner from this reproduction as the cap stone figure.
- **New `Examples/abell1201_umbh/`** OR extend `Examples/radial_arc_smbh/` with a `--target=abell1201` rung. (Decide based on how much new architecture vs reuse — likely a sister-Example since the lens is at z=0.169 not z=0.7, different physical regime.)

## 9. Timeline (when activated)

- Day 1: Run Spec 00 §6.9 Watson pipeline on the Abell 1201 raw FLT files
- Day 2: Build the autolens model in `a1201_lens_model.py`; smoke fit on laptop
- Day 3: Cannon submit; ~24h wall
- Day 4: Audit + reconcile M_BH posterior against Nightingale+2023's published value
- Day 5: Module 15 §"Real-data" update + Examples promotion
- Total: ~1 week + Cannon wall time

## 10. Pre-requisites

Before activating this spec:
- Specs 00-04 should be substantially complete (or at least Spec 00's Watson pipeline integration shipped)
- Phase 3 `_jeans_sigma_v.AnalysisKinematics` strict-PASS on Cannon (already in flight 2026-05-18)
- `Examples/radial_arc_smbh/ --part=with_kinematics` strict-PASS Cannon result audited (Cannon job 13649652 running)

When those pre-reqs are met, file this spec for re-activation.

## 11. Notes on the related "Probing Dark Matter Substructures with Free-Form Modelling: A Case Study of the 'Jackpot' Strong Lens" ([arxiv 2504.19177](https://arxiv.org/html/2504.19177v1))

A 2025 free-form modelling re-analysis of the Jackpot lens — relevant to Spec 02 (P2 TSPL Jackpot). Not part of this spec but worth tracking when the program reaches the eventual cross-validation discussion. Document as a "see also" in Spec 02 references.
