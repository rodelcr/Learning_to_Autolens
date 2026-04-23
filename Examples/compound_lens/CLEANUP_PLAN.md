# compound_lens cleanup plan — pushing v4 below the 4σ bar

The v3 direct fit (job 7401556, committed commit `18d7842`) reached:

- log_Z = +30,705.01 (matches user's reference notebook to +7 log units)
- chi²/N = 0.727 (PASS)
- max |res| = **6.18σ** (borderline — the skill's pass bar is ≤ 4σ)

The `fit_subplot.png` residual map is mostly salt-and-pepper, but the
**chi² map** shows localized hot spots on the arc and at the counter-image,
and the **Lens Light Subtracted** panel shows a residual central peak
(the model's lens bulge is slightly under-subtracting the data's central
galaxy). So there is real, coherent-ish residual structure worth cleaning
up — the fit is not yet "pure noise."

This plan orders candidate changes by expected impact vs. compute cost.

---

## Hypothesis A — re-enable external shear (cheapest, highest probability)

**Why it might help.** v1/v2 had external shear on `lens_0`; v3 dropped it to
match the user's reference notebook. But the reference notebook treated
lens_1 as a real deflector with free θ_E, whose MAP actually absorbed the
asymmetry that shear would capture. In v3, `lens_1.einstein_radius
collapsed to 0`, so the asymmetry that shear would model has nowhere to go.
The hot-spot chi² on the counter-image is exactly the signature of a
missing shear term.

**What to change in `fit_example_compound_lens.py` `build_direct_fit`:**

```python
# Add back to lens_0 (z=0.5):
shear = af.Model(al.mp.ExternalShear)
shear.gamma_1 = af.GaussianPrior(mean=0.0, sigma=0.15)
shear.gamma_2 = af.GaussianPrior(mean=0.0, sigma=0.15)
lens_0 = af.Model(al.Galaxy, redshift=0.5,
                  bulge=bulge_0, mass=mass_0, shear=shear)
```

Wide GaussianPrior(0, 0.15) — not truth-seeded; lets the data drive shear
to whatever it needs. Expected outcome: |γ| ≈ 0.05–0.1, max |res| drops
from 6.2σ to ~4σ.

**Cost:** 1 Cannon job, ~1–2h wall clock. 33 free parameters (up from 31).

**Risk:** shear might absorb some of `lens_1`'s ell_comps freedom,
changing the collapse-to-zero story. Check after the fit whether
`lens_1.einstein_radius` still lands at 0 or drifts to a small non-zero
value. If drifts, that's fine — shear and a small real secondary are
physically distinguishable.

---

## Hypothesis B — upgrade `lens_0.mass` to PowerLaw (moderate, medium probability)

**Why it might help.** Isothermal (γ' = 2) is a common approximation but
real galaxy-scale lenses have γ' ∈ [1.8, 2.2] with ~0.1 scatter. If the
true slope for this mock is, say, 2.1, Isothermal leaves small but
coherent residuals on the arc's bright features.

**What to change:**

```python
mass_0 = af.Model(al.mp.PowerLaw)  # was al.mp.Isothermal
mass_0.slope = af.UniformPrior(lower_limit=1.5, upper_limit=2.5)
# ... rest of priors unchanged
```

**Cost:** 1 Cannon job, ~2h wall clock. 32 free parameters (+1 for slope).

**Risk:** PowerLaw slope is degenerate with source size and lens radial
extent. The posterior may broaden noticeably on θ_E and R_e(source).
Worth doing anyway because the slope itself is pedagogically
interesting — the user asked exactly this as Exercise 2 in
`01_compound_direct_fit.ipynb`.

---

## Hypothesis C — switch source to pixelized (most flexible, highest wall time)

**Why it might help.** SersicCore has 7 parameters. A real lensed source
can have spiral arms, clumps, or asymmetric profile that 7 parameters
cannot capture. `al.mesh.RectangularAdaptImage` + `al.reg.Adapt` gives the
source ~O(10²) effective degrees of freedom.

**What to change.** This requires a new driver part (`--part=direct_pix`)
that runs a 3-stage mini-pipeline:

1. **`direct_pix_stage1`** — Sersic source + Isothermal mass + external
   shear (same as Hypothesis A). Gets a clean initial mass model.
2. **`direct_pix_stage2`** — Fix mass, swap source for
   `al.mesh.RectangularAdaptDensity` (seeded by stage 1's Sersic
   max-likelihood image).
3. **`direct_pix_stage3`** — Free mass + `al.mesh.RectangularAdaptImage`
   (seeded by stage 2's adapt image). Joint refinement.

**Cost:** 3 Cannon stages, ~3–4h total wall clock. Risk of mesh collapse
if stage 2's adapt image is poor (seen in Mod 04 failures; mitigated by
the seeded-prior approach from commit `8656998`).

**Risk:** pixelization may over-fit noise features and claim log_Z
improvements that are not physical. Check chi² drops against the extra
effective parameters.

---

## Recommended order of execution

1. **A (shear)** first — cheapest, most likely to get us to the 4σ bar,
   and independently educational.
2. If A gets max |res| ≤ 4σ, stop. Update notebook narrative to note
   that shear was needed even in the collapsed-secondary regime.
3. If A helps but doesn't reach 4σ, run **B (PowerLaw)**. Record the
   recovered slope as an ingredient for the Exercise 2 answer.
4. Only do **C (pixelized)** if A and B both leave max |res| > 4σ.
   Pixelization is worth doing as an illustrative *comparison* anyway —
   the "`direct_pix` vs `direct_sersic`" panel in the notebook is
   pedagogical — but start with the cheaper changes.

---

## Audit criteria for v4

Declare success when:

- max |res| ≤ 4σ (full PASS on the numerical bar)
- Chi² map shows NO bright spots on the arc or at lens centres (pure
  noise)
- Lens Light Subtracted panel shows NO central residual peak
- log_Z ≥ +30,700 (within 100 log units of v3 — prior change shouldn't
  evict us from the good basin)

Document the winning hypothesis in the notebook's "What would change this
diagnosis" section; retire the v3 numbers in `model_results.txt` but keep
them referenced in the commit message for reproducibility.

---

## Notes on what NOT to change

- **Don't re-tighten `einstein_radius` priors.** `UniformPrior(0, 8)` on
  both lenses was the key fix that let v3 find the right basin; narrower
  priors would reintroduce Pattern E (forced-compound suboptimum).
- **Don't drop `lens_1` entirely** from the model. Even though its
  `einstein_radius` collapses to 0, its Sersic *light* is real and
  removing the galaxy would un-subtract visible flux. Keep the galaxy;
  let its mass collapse.
- **Don't change the mask** radius. 2.7″ matches the user's reference
  notebook; changing it would make the numeric comparison incommensurable.
