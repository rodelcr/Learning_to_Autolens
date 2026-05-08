# v0.95 Full-Pipeline Plan — DSPL → MGE → Physical → Cosmology

**Date drafted:** 2026-05-08
**Author:** Rodrigo Córdova Rosado
**Scope:** Specify the four scientific stages needed for AGEL's full cosmography programme, audit what's already shipped, identify gaps, and verify cluster-runnability for each.

The four stages map onto distinct *physical* questions:

| Stage | Physics question | Constraint |
|---|---|---|
| 1. DSPL | What is β = (D_ls1/D_s1)/(D_ls2/D_s2)? | (Om0, w0) without H0 |
| 2. MGE | What is the lens-light surface brightness Σ★(θ)? | Stellar mass profile |
| 3. Physical mass | Decompose mass into stars (Υ·MGE) + dark (NFW) | f_DM(<θ_E), γ★ |
| 4. Cosmology | What is H0 from time delays + kinematics? | H0 with MSD broken |

Each later stage *depends on* the prior stage's output. End-to-end this is the AGEL
cosmography programme: DSPL pins (Om0, w0) cheap → MGE+Physical anchor lens-mass
profile → time delays + kinematics close H0 with the mass model fixed.

---

## Stage 1 — DSPL (Double Source-Plane Lensing)

### Physics
A single deflector at z_L with two background sources at z_s1 < z_s2. The ratio
of Einstein radii θ_E(z_s1)/θ_E(z_s2) traces β = (D_ls1·D_s2)/(D_ls2·D_s1),
which is **independent of H0** and constrains (Ωₘ, w₀) via the angular-diameter
distance integrand. Collett & Auger (2014) demonstrated this for SL2SJ02.
Lower-budget cosmography than time delays.

### Current state (2026-05-08)
- `Examples/double_source_plane/`
  - 3 notebooks ✓ (`00_climb_to_dspl`, `01_dspl_direct_fit`, `02_beta_cosmography`)
  - Mocks ✓ (`mocks/mock_image.fits` etc., `mock_truth.json`)
  - Results ◐ (research-in-progress; v0.93 v2 stalled, v0.94 staged chain in flight)
  - **README status: ◯ Planned (STALE — should be ◐ in progress)**
- `Modules/10_Cluster_Computing/scripts/fit_example_double_source_plane.py`
  - 5 `--part` choices: `direct`, `beta_fixedcosmo`, `beta_freecosmo_v3`, `beta_chain`, `beta_freecosmo`
  - v0.94 added `beta_chain` as the recommended pipeline (Stage 1 fixedcosmo + Stage 2 freecosmo with TruncatedGaussian on Om0/w0)

### Cluster status
- Job 11214940 `dspl_beta_chain` running (hour 18 of 96, log_Z improving)
- Will land within v0.94 budget if Stage 1 converges

### v0.95 deliverables
- [ ] Update README from ◯ Planned → ◐ In progress (notebooks ship, fit pending)
- [ ] When job 11214940 lands, audit Stage 2 posteriors on (Om0, w0) — bracketing truth (0.30, -1.0) is the strict-PASS criterion
- [ ] Document the staged-chain methodology in the `02_beta_cosmography.ipynb` results section

### Cluster command
```bash
sbatch --time=96:00:00 --mem=64G --cpus-per-task=32 \
       --job-name=dspl_beta_chain \
       --export=ALL,EXAMPLE=double_source_plane,FIT_EXTRA_ARGS=--part=beta_chain \
       Modules/10_Cluster_Computing/scripts/submit_cannon.slurm
```

---

## Stage 2 — MGE (Multi-Gaussian Expansion light)

### Physics
A Sersic profile is a single function with three parameters. Real lens galaxies
have isophote-twists, boxy/disky deviations, multiple stellar components — Sersic
under-fits these structurally. **MGE** (Cappellari 2002, Emsellem+ 1994) sums
N Gaussians with shared centre but free σᵢ, qᵢ, Iᵢ. With N ≥ 8 it captures the
true light profile to <0.5%, which is what the **physical** mass-model needs.

### Current state
- `Modules/09_MGE_Linear_Light_Profiles/` ✓ shipped (Module 09: 8 cells + recipe)
- `Examples/mge_to_physical/`
  - Mocks ✓ (lenstronomy_mock_1)
  - Driver ✓ (`fit_example_mge_to_physical.py`, 7 `--part` choices)
  - Results ◐ — canonical Search 3 fit has χ²/N=1.87, max|res|≈9.7σ
  - **2026-05-07 chi²-at-truth diagnostic verdict: framework-level Sersic eval
    mismatch.** Removing components changes χ² by <1%; the ~33σ residual at
    the central pixel persists with all truth components present. Root cause
    is lenstronomy↔autolens cuspy-Sersic (n=4.9) integration difference.

### Cluster status
Driver works; the science blocker is the framework mismatch, not the cluster.

### v0.95 deliverables
- [ ] **Regenerate `mocks/lenstronomy_mock_1_*.fits` natively in autolens** — 2-hour scripting + 24h Cannon. Rebuild the mock with `al.Galaxy(... bulge=al.lp.Sersic(n=4.9, ...))` simulator, save FITS + truths.json, retry the canonical Search 3 fit. Should drop max|res| < 5σ at truth.
- [ ] Update mge_to_physical README with the new mock + new chi²-at-truth = 1.0 baseline
- [ ] Re-run `--part=stars_dark_v2` on the regenerated mock, expect strict-PASS

### Cluster command
After regeneration:
```bash
sbatch --time=24:00:00 --mem=64G --cpus-per-task=32 \
       --job-name=mge_v3_remock \
       --export=ALL,EXAMPLE=mge_to_physical,FIT_EXTRA_ARGS=--part=stars_dark_v2 \
       Modules/10_Cluster_Computing/scripts/submit_cannon.slurm
```

---

## Stage 3 — Physical Mass Modeling

### Physics
Convert the Stage-2 MGE light into mass: M(R) = Υ★ × ∫Σ★(R) dR + ρ_NFW. The
mass-to-light ratio Υ★ is the free parameter (constrained ~0.5–5 in solar units
from stellar populations); the NFW halo is anchored by the cluster/group's
virial mass when external constraints exist. Recover **f_DM(<θ_E) and γ★** —
the dark-matter fraction within the Einstein radius and the stellar slope.
Auger+10 / Sonnenfeld+13 published these for SLACS/SL2S; Module 11's audit
framework is built around recovering these from a fit.

### Current state
- `Modules/11_Physical_Mass_Models/` ✓ shipped — 29 cells, 6 sections:
  - 6-panel residual audit
  - Bonferroni-corrected numerical bar
  - Pattern A–F failure catalogue
  - f_DM(<θ_E) extraction via Sersic incomplete-gamma + NFW Wright-Brainerd
  - γ★ recovery vs Auger+10
  - Decision flowchart
- **No standalone Cannon driver** — Stage 3 is *post-fit analysis* of Stage 2's
  output. The audit runs in <60s on the laptop with no fits.

### Cluster status
Not applicable — Stage 3 is laptop work that consumes Stage 2's output.

### v0.95 deliverables
- [ ] Once Stage 2 lands clean (regenerated mock + strict-PASS Search 3),
      re-execute Module 11 on those results. f_DM(<θ_E) should match
      truth within 1σ; γ★ should match truth within 1σ.
- [ ] Add a "results" markdown cell to Module 11 §4 quoting the actual
      recovered numbers vs truth — Module 11 currently shows the
      methodology applied to AGEL `direct_clean` (a strict-PASS imaging
      fit but NOT a stars+dark decomposition); after Stage 2 v0.95 it can
      cite the true f_DM recovery.

---

## Stage 4 — Cosmology (Time-Delay + Kinematics)

### Physics
Time delays measure D_Δt = (1+z_l)·D_l·D_s/D_ls, dominantly sensitive to H0.
But the mass-sheet degeneracy (κ → λκ + (1−λ)) preserves all *imaging*
observables and rescales delays by exactly λ → H0 by 1/λ. **The only way
to break MSD with imaging+delays alone is external information**:
1. Stellar kinematics (σ_v aperture-projected) — the canonical break
2. Multiple source planes (DSPL!) — the cross-link to Stage 1
3. Standardizable lensed magnifications (rare)

This is why Stages 1 and 4 are connected: **DSPL provides MSD-breaking
information that's independent of kinematics.** Birrer+ 2020 used DSPL
re-analysis to constrain MSD without σ_v.

### Current state
- `Modules/12_Time_Delay_Cosmography_MSD/` ✓ shipped (19 cells)
- `Modules/13_TDCOSMO_Kinematics_MSD/` ✓ shipped — anisotropic Jeans + σ_v
- `Modules/14_Compound_Multi_Plane_Lensing/` ✓ shipped — multi-plane recursion
- `Examples/quad_time_delay/`
  - Mock with extended host arc ✓
  - 7 `--part` choices on driver
  - **Track D joint fit STRICT-PASS** (chi²/N=1.05, max|res|=4.66σ)
  - **H0 chain 2026-05-08:** pos-only σ=26 → image-only σ=10 → joint σ=2.3,
    bias +12 → +5 km/s/Mpc
  - Figure: `Examples/quad_time_delay/figures/h0_chain_overlay.png`

### Cluster gaps
- **No kinematics driver yet.** Module 13 ships the theory (anisotropic Jeans
  + aperture projection). What's missing is a `fit_example_*` that combines
  AnalysisPoint + AnalysisImaging + a σ_v likelihood term that uses Module 13's
  Jeans solver to predict σ_v(θ_eff) from the model lens mass.
- **No DSPL→TDCOSMO cross-likelihood example.** The DSPL β posterior from
  Stage 1 could be used as an external λ_int prior for the TDCOSMO chain in
  Stage 4. Birrer+ 2020 §4 shows the math.

### v0.95 deliverables
- [ ] **Driver: `fit_example_quad_time_delay.py --part=joint_fit_h0_kin`** —
      adds a kinematic σ_v likelihood on top of the Track D joint fit.
      σ_v truth = 280 km/s (from Module 13 mock), σ(σ_v_obs) = 10 km/s.
      Expected: H0 bias drops from +5 → ~+1, σ stays ~2.3.
- [ ] After 11214940 lands: cross-link the DSPL β posterior into the TDCOSMO
      chain as an MSD-breaker (research in `Examples/quad_time_delay/03_dspl_msd_break.ipynb`).
- [ ] Tag v0.95 with the kinematic-augmented strict-PASS H0 fit.

### Cluster command (after kin-driver lands)
```bash
sbatch --time=24:00:00 --mem=64G --cpus-per-task=32 \
       --job-name=qtd_joint_h0_kin \
       --export=ALL,EXAMPLE=quad_time_delay_joint,FIT_EXTRA_ARGS=--part=joint_fit_h0_kin \
       Modules/10_Cluster_Computing/scripts/submit_cannon.slurm
```

---

## End-to-end pipeline runnability check

For each stage, the cluster driver must satisfy 3 invariants:

1. **`build_*_fit()` constructs without error** — model + priors + data load
2. **`--part=<name>` validates in argparse** — no silent typo failures
3. **A 1-min smoke test fit (n_live=20, 1 iteration) completes** — proves the
   likelihood is finite at the truth and Nautilus can step

The 1-min smoke test pattern (laptop):
```bash
LTA_RUN_HEAVY=0 PYAUTOFIT_TEST_MODE=1 \
   python Modules/10_Cluster_Computing/scripts/fit_example_<NAME>.py \
   --part=<PART>
```
PYAUTOFIT_TEST_MODE=1 short-circuits Nautilus to 1 iteration (used in CI).

A pipeline-wide smoke test runner is `Modules/10_Cluster_Computing/scripts/smoke_test_drivers.sh` — TBD if it exists.

---

## Tagging and shipping

v0.95 ships when all 4 stages have at least one strict-PASS exemplar:

| Stage | v0.94 | v0.95 target |
|---|---|---|
| DSPL | beta_chain in flight | strict-PASS Stage 2 with (Om0, w0) bracketing truth |
| MGE | research-in-progress (mock mismatch) | regenerated mock + strict-PASS Search 3 |
| Physical | pedagogical only | f_DM/γ★ recovery on regenerated MGE fit |
| Cosmology | Track D STRICT-PASS (joint), MSD un-broken | kinematic-augmented joint with H0 within 1σ |

Below this bar v0.95 still ships **methodology and curriculum** improvements
even if the strict-PASS exemplars don't land — same discipline as v0.92–0.94.
