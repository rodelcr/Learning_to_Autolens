# Next Steps — v0.97 Roadmap

**Date drafted:** 2026-05-18 (post v0.96-alpha tag)
**Author:** Rodrigo Córdova Rosado
**Predecessor:** the v0.95→v0.96 roadmap, fully closed. v0.96-alpha tagged 2026-05-15.

What v0.96 closed: galaxy_galaxy_single_arc (Tier-1 #1), positions_modeling (#2), kinematic_h0_break driver shipped via Phase 3 `_jeans_sigma_v.py` + `AnalysisKinematics` (#3), bayesian_model_comparison empirical fill-in (#4), cosmography_joint_posterior (#7), Module 15 (Radial Arcs), Examples/radial_arc_smbh (depth B), DSPL Stage 1+2 STRICT-PASS cosmography, mge axis-swap fix, chi²-at-truth + driver-truth two-check methodology, preflight_check_v096.sh.

---

## Tier 1 — v0.97 substantial deliverables

### #1 — `Examples/radial_arc_smbh/ --part=with_kinematics` Cannon ship

The driver is wired against `_jeans_sigma_v.py` and the smoke-tested AnalysisKinematics class. Need a single Cannon submit (~12 h, `--mem=192G`, 32 cores) to produce the joint imaging + Jeans σ_v posterior. The pedagogical payoff: imaging-only Δlog_Z = +1.71 in favour of no-BH → joint should land **with kinematics breaking the γ′–M_BH degeneracy** so M_BH posterior tightens to truth.

**Deliverables:** Cannon result → strict-PASS audit → update `01_radial_arc_smbh.ipynb` §5 with the kinematic-break headline table.

### #2 — `fit_example_quad_time_delay.py --part=joint_fit_h0_kin`

Same `_jeans_sigma_v.AnalysisKinematicsFreeCosmology` instance + the existing AnalysisPoint + AnalysisImaging via `af.FactorGraphModel`. Adds Jeans σ_v to the TDCOSMO joint fit. **Goal:** breaks the +5 km/s/Mpc H₀ bias seen in v0.96 (`joint_h0_free`: 75.0 ± 2.6 vs truth 70.0). Birrer+ 2020 IV methodology.

**Deliverables:** new `--part=joint_fit_h0_kin` rung in the driver; Cannon submit; expected H₀ recovered at 70 ± few; update `Examples/cosmography_joint_posterior/` to remove the independence approximation.

### #3 — `mge_to_physical` lp_linear.Sersic + MGE light follow-on

v0.96 axis-fix landed 50% chi²/N reduction (6.44 → 3.16) but `search_3_v2_stars_dark` still shows coherent ring-shaped residuals. Suspected root cause: `lp_linear.Sersic` source decomposition convention vs simple Sersic, OR untreated 2-source-plane multiplicity. Replace simple Sersic with `al.lp_linear.Sersic` + MGE light per Module 09 methodology.

**Deliverables:** updated driver, Cannon retry, strict-PASS audit. Promotes mge_to_physical from research-in-progress to ✓ shipped.

### #4 — DSPL × TDCOSMO single-likelihood fully-joint-fit

`Examples/cosmography_joint_posterior/` v0.96 used the independence approximation. The full Birrer+ 2020 §4 implementation runs `af.FactorGraphModel(DSPL_imaging, TDCOSMO_imaging, TDCOSMO_point, TDCOSMO_kinematic)` with a shared cosmology model. Removes the approximation.

**Deliverables:** new combined-fit driver; Cannon ship (~24 h); single posterior on (Ωₘ, w₀, H₀) without independence.

---

## Tier 2 — depth-C real-data application

### #5 — `Examples/agel_spiral_real_target/` (NEW example, depth C)

Apply the v0.96 `radial_arc_smbh` methodology to a real AGEL Einstein-spiral target. Baseline candidate: DESJ0206 (Ferrami+ 2024 ApJL). The 8-item bridge checklist is in `Examples/radial_arc_smbh/README.md` §"Bridge to depth-C":

- [ ] HST imaging cutout from MAST (ACS WFC F606W + WFC3-IR F125W minimum)
- [ ] Empirical PSF (not Gaussian — `tinytim` or stars-in-field)
- [ ] Real `tinytim`/empirical PSF kernel pipeline
- [ ] Lens-light decomposition: MGE basis vs simple Sersic ablation
- [ ] Hot-pixel + cosmic-ray cleanup (reuse `Examples/agel_real_target` pipeline)
- [ ] σ_v from KCWI / LLAMAS IFU via ppxf, aperture-matched to lensing R_eff
- [ ] Photometric or spectroscopic z_l, z_s with marginalisation
- [ ] Multi-band joint fit if both HST and JWST available

This is the v0.97 stretch goal — production-grade real-target paper-ready fit.

### #6 — `Examples/multi_band_joint_fit/` (NEW example)

HST WFC3 (F814W) + JWST NIRCam (F200W) on the same lens. AGEL DR2 targets have multi-band imaging. The autolens API supports multi-dataset fits via `af.FactorGraphModel` — same pattern as TDCOSMO joint but with two AnalysisImaging factors. Demonstrates wavelength-dependent source structure.

**Deliverables:** 2-band mocks generator, driver, notebook, Cannon submit (~8 h).

---

## Tier 3 — methodology / infrastructure

### #7 — Cannon submit script: prevent cross-example output pollution

Root cause of the 124-untracked-dir cleanup on 2026-05-18: `submit_cannon.slurm` writes outputs to `Examples/<EXAMPLE>/results/<search_name>/` based on `EXAMPLE` env var, but the rsync pulls EVERY `Examples/*/results/<search_name>/` dir from Cannon. When several drivers wrote search_names that collide across examples (e.g. `composite_mass` exists under multiple Example output trees because Module 06 fits write there), the rsync replicates them under every example's local results tree.

**Investigate:** is the issue in `submit_cannon.slurm`, in `export_results.py`, or in `pull_from_cannon.sh`'s rsync include rules? Cannon-side fix preferred over client-side cleanup script.

**Deliverables:** documented root cause + Cannon-side fix + regression test that pulls fresh and asserts zero cross-example duplication.

### #8 — Multi-GPU JAX MGE benchmark (task #127)

Per `project_multigpu_jax_idea.md` memory: single-GPU JAX was 4× *slower* than numpy. Multi-GPU data-parallel + SLURM-array have not been tested. Open question.

**Deliverables:** adapt `fit_example_mge_to_physical.py` with `--use-jax-pmap` flag, benchmark wall + chi²/N on 2–4 GPUs vs numpy 32-core baseline. Deferred priority — only valuable if numpy stops fitting in `--mem=192G` envelope.

### #9 — Anisotropy β(r) extension to `_jeans_sigma_v.py`

Phase 3 starter is isotropic only (β=0). Mamon & Łokas 2005 provides an anisotropy-kernel extension that slots into `_jeans_inner_integral` without API change. v0.98 scope. Adds 1 free param to AnalysisKinematics; β prior from Cappellari+13 OM model.

---

## Sequencing

Work order assuming v0.97 completes within ~6 weeks:

1. **Week 1**: #7 Cannon submit root cause (1-2 days) + #1 radial_arc_smbh with_kinematics submit (~12 h Cannon)
2. **Week 2**: #2 joint_fit_h0_kin driver + Cannon submit; #1 audit + notebook update
3. **Week 3**: #3 mge_to_physical lp_linear retry; #2 audit + cosmography update
4. **Week 4-5**: #5 agel_spiral_real_target (DESJ0206) or #4 DSPL × TDCOSMO single-likelihood
5. **Week 6**: #6 multi_band_joint_fit or #8 Multi-GPU JAX benchmark
6. Tag v0.97-alpha when #1, #2, #3, #7 land strict-PASS + #5 or #4 lands depth-C

Critical path: #7 first (prevents recurring cleanup overhead). Then #1, #2, #3 unlock the kinematic-break + lp_linear methodology shipping. Then real-target application (#5).

---

## What's deferred to v0.98+

- #9 anisotropic Jeans
- `subhalo_sensitivity` full Vegetti+10 grid SLaM
- Mathematica companion Module 15 in Learning_to_Lens
- AGEL DR2 multi-target stack: ≥ 5 DSPL targets for competitive cosmographic precision
