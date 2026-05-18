# Example: Radial Arc + SMBH (Einstein-Spiral Methodology Bridge)

## Status

✓ **Shipped v0.96 (2026-05-15)** — mock + driver + 4-rung dispatcher + tutorial notebook + STRICT-PASS Cannon results all in tree.

- `--part=direct` (PowerLaw + shear, no BH): chi²/N = 1.019, max\|res\| = 4.27σ, γ′ recovered 1.959 ± 0.015 (truth 1.95) ✓ strict-PASS
- `--part=with_pointmass` (+ central PointMass): chi²/N = 1.019, θ_E_BH = 0.073 (-0.05, +0.04) 3σ, **Δlog_Z = +1.71 favouring no-BH** — γ′–M_BH degeneracy demonstrated; BH NOT detected by imaging alone (the headline pedagogical result).
- `--part=with_kinematics` (joint AnalysisImaging + AnalysisKinematics via FactorGraphModel): driver wired in v0.96 against the new `_jeans_sigma_v.py` (Phase 3). Cannon submit deferred to v0.97 ship cycle.
- See `01_radial_arc_smbh.ipynb` for the 8-section walkthrough.

## Why this example exists

Module 15 (Radial Arcs & Caustic Topology) shipped the **theory** of why radial arcs constrain inner mass slope γ′ and how a central SMBH creates a γ′–M_BH degeneracy. This example takes that theory and operationalises the AGEL spiral-lens **methodology** on a synthetic mock:

> *Can you, from imaging alone, recover M_BH for an Einstein-spiral lens with an embedded UMBH? How well does stellar kinematics break the γ′–M_BH degeneracy?*

The mock geometry mirrors the AGEL spiral-lens targets (Shajib et al. *"the first Einstein spiral"*, Ferrami et al. DESJ0206 ApJL). The example is the **methodology bridge B** from `V095_PIPELINE_PLAN.md` — between Module 15's theory (depth A: pedagogical baseline) and a real-data application on an actual AGEL target (depth C: future v0.97+ work).

## Mock geometry

`mocks/generate_mock.py` renders the mock natively in autolens. Self-consistency at the end: **chi²-at-truth = 1.029, max|res| = 4.35σ** (well within noise floor).

| Quantity | Truth |
|---|---|
| z_lens | 0.7 |
| z_source | 1.5 |
| Lens mass — `PowerLaw` | θ_E=1.0″, slope=1.95 (sub-isothermal), ell_comps=(0.10, 0.05), centre=(0, 0) |
| Lens — `PointMass` (SMBH) | θ_E,BH=0.08″ (pedagogically large; see note) |
| Lens shear — `ExternalShear` | γ₁=0.025, γ₂=-0.015 |
| Lens light — `Sersic` (de Vaucouleurs) | n=4, R_e=0.8″, intensity=1.0 |
| Source — `Sersic` (disc) | n=1.5, R_e=0.18″, offset=(0.06, 0.03)″, intensity=0.5 |
| Cosmology | FlatLambdaCDM(70, 0.30) |
| Imaging | 80×80 px @ 0.05″/px, HST WFC3-IR-like (FWHM 0.12″) |
| σ_v measurement | 278.6 ± 10.0 km/s at R_eff (truth 280 km/s, synthetic ppxf-like) |

### Pedagogical note on M_BH scale

The truth `θ_E,BH = 0.08″` corresponds (at z_l=0.7, z_s=1.5) to **M_BH ≈ 3×10¹¹ M☉** — larger than any UMBH ever detected. This is *intentionally* pedagogical: the imaging-only `--part=with_pointmass` fit is sensitive at this scale so the (γ′, M_BH) joint posterior shows the degeneracy clearly. Real AGEL targets have θ_E,BH ~ 0.02″ — well below HST pixel scale, where the imaging-alone constraint is weak and **kinematics is required to detect the BH**. That's the v0.97+ depth-C scenario. Module 15 §4 discusses the scale issue.

## Method — four fit parts

`fit_example_radial_arc_smbh.py` builds:

1. **`--part=direct`** — `PowerLaw(γ′ free) + ExternalShear + Sersic source`. NO BH. Establishes the imaging-only inner-slope posterior. The fit will return γ_eff somewhere between truth (1.95) and apparent steepening due to the unmodeled SMBH. ~9 free params.
2. **`--part=no_pointmass`** — Same model, `γ′ = 2.0` pinned (isothermal baseline). Bayes-factor reference for the AGEL "sub-isothermal" finding. ~8 free.
3. **`--part=with_pointmass`** — Adds `al.mp.PointMass`. Free `einstein_radius_BH ∈ U(0.001, 0.20)`. Joint (γ′, M_BH) posterior. **Headline pedagogical result** — the γ′–M_BH degeneracy is visible here. ~10 free.
4. **`--part=with_kinematics`** — STUB. Joint `AnalysisImaging + AnalysisKinematics` via `af.FactorGraphModel`. The custom `AnalysisKinematics` is the shared Jeans-σ_v likelihood that also feeds `kinematic_h0_break` in `fit_example_quad_time_delay.py`. Ships in v0.97 (task #122).

## Headline results (to land when Cannon does)

Three audit checks, each measuring whether the fit recovers the physics:

| Part | Free params | Strict-PASS criterion |
|---|---|---|
| `direct` | 9 | chi²/N ≤ 1.3, max\|res\| ≤ 4.5σ, γ′ posterior brackets 1.95 within 1σ |
| `no_pointmass` | 8 | chi²/N ≤ 1.5 expected (model misspecified by the missing BH); Bayes factor `direct - no_pointmass` should be strongly positive |
| `with_pointmass` | 10 | chi²/N ≤ 1.3, max\|res\| ≤ 4.5σ, M_BH posterior brackets truth within ~2σ (degenerate with γ′) |
| `with_kinematics` (v0.97) | 10 + 1 σ_v | chi²/N ≤ 1.3, M_BH posterior brackets truth within 1σ, degeneracy broken |

## Running on Cannon

```bash
sbatch --account=siag_lab --partition=siag \
       --time=8:00:00 --mem=192G --cpus-per-task=32 \
       --job-name=rarc_direct \
       --export=ALL,EXAMPLE=radial_arc_smbh,FIT_EXTRA_ARGS='--part=direct' \
       Modules/10_Cluster_Computing/scripts/submit_cannon.slurm
```

Wall budgets: `direct` ~3h, `no_pointmass` ~2h, `with_pointmass` ~4h on 32 cores. The `all` part runs all three sequentially (~12h total).

## Bridge to real AGEL targets (depth C)

When v0.97 brings the kinematic-break working on this synthetic mock, the natural next step is application to a **real AGEL Einstein-spiral target** (depth C, future `Examples/agel_spiral_real_target/`). The methodological gap is small; the real-data systematics budget is large. **Concrete checklist** for that transition:

- [ ] **Empirical PSF.** Real lens fields have variable PSFs across the detector. Construct an empirical PSF by drizzling bright isolated stars in the same HST exposure as the target (the pattern is in `Examples/agel_real_target/data/`).
- [ ] **Neighbor light masking.** Einstein spirals sit at the centre of bright lens galaxies; neighbours within ~3″ contribute lens-light flux that must be masked OR jointly modeled. The radial-arc lives *inside* the lens light — the mask radius cannot exceed ~ R_eff.
- [ ] **Real σ_v measurement pipeline.** ppxf on IFU data (KCWI / LLAMAS). Aperture matching: lensing integrates a different scale than kinematics — handle aperture matching explicitly via the Jeans projection.
- [ ] **Spectroscopic vs photometric redshifts.** Real systems have measured z but with finite precision. Propagate z_l, z_s uncertainty into the M_BH posterior.
- [ ] **Multi-band joint fit** (HST WFC3 F814W + JWST NIRCam F200W typical). Source colour gradients matter at radial-arc scales. The JWST inner-resolution data also directly resolves the SMBH-dominance region.
- [ ] **Inner-slope systematics budget.** Compare γ′ posterior against PSF mismatch ablations, mask-radius variation, prior tightness ablations, multi-source consistency.
- [ ] **Reference comparison.** Compare M_BH posterior against the Shajib et al. and Ferrami et al. published recipes. Same physics, different mass-decomposition choices.
- [ ] **Target picks.** Suggested AGEL spiral targets in order of recommendation:
  - DESJ0206-0114 (Ferrami paper baseline; HST F814W + KCWI σ_v already in hand)
  - AGEL104041, AGEL143408, AGEL144640 — all have HST + IFU per the Observing Logs
  - The HST Cycle 34 spiral targets (10 newly imaged systems) when those data land

## Exercises

1. **Recover γ′ from `direct`.** Compare the posterior median to truth (1.95). It should bias *high* (toward isothermal) because the imaging is partially absorbing the unmodeled SMBH.
2. **Bayes-factor isothermal vs free.** Run `--part=no_pointmass` AND `--part=direct`; compute Δlog_Z. A strong positive favour for `direct` shows the data demand sub-isothermal — the AGEL spiral-lens signature.
3. **γ′–M_BH degeneracy visualisation.** From `--part=with_pointmass` posterior, plot the joint (γ′, einstein_radius_BH) corner. The two parameters should be anti-correlated — the degeneracy from Module 15 §4.
4. **What does M_BH=10¹⁰ M☉ look like?** Modify `generate_mock.py` to set `theta_E_BH = 0.02` (the realistic AGEL UMBH scale). Re-run all three parts. Show that the imaging-only constraint becomes non-detection.
5. **Future v0.97 exercise**: when `with_kinematics` ships, repeat exercise 4 with the kinematic constraint added. Demonstrate the kinematic break.

## References

- **Module 15 (Radial Arcs & Caustic Topology)** — the physics background. Required reading.
- **Module 11 (Physical Mass Models)** — the 6-panel residual audit applied to this fit.
- **Module 13 (TDCOSMO + Kinematics)** — Jeans theory the `with_kinematics` part will consume.
- **Shajib, Tran, Vasan, Córdova Rosado et al.** — *"An ultramassive black hole 7.1 Gyr ago in the first 'Einstein spiral' gravitational lens"* (in preparation).
- **Ferrami, O'Riordan, Córdova Rosado, et al.** — *"Detection of a quiescent billion solar mass black hole at z≈0.7 with gravitational lensing"* (DESJ0206 ApJL, in preparation).
- **Sonnenfeld+13** — γ′ recovery on SL2S (typical σ ~ 0.02 with both arc types).
- **Auger+10** — SLACS structural decomposition framework.

## What this example DOESN'T cover

- **The radial-caustic physics** — see Module 15 (the theory companion).
- **Pixelized source reconstruction** — see Module 05 and `compound_lens/02_compound_slam.ipynb`. The Sersic source is enough for this depth-B example.
- **Real-data systematics** — see the Bridge-to-C checklist above. v0.97+ work.
- **Joint with time delays** — radial-arc Einstein spirals don't typically have time-variable AGN; the TDCOSMO methodology is for quad quasars (see `quad_time_delay/`).
