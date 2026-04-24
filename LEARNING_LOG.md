# Learning Log — what actually broke, what actually worked

Running notebook of engineering insights as we build out the curriculum
and Examples/ collection. Meant to be a "hard-won lessons" record that
complements the structural docs (`CLAUDE.md`, handoffs, memory files) —
things you'd have wanted to know *before* starting, collected *after*.

Format: dated entry, one insight per section, concrete enough that a
future session can jump to the fix without re-deriving.

---

## 2026-04-23 — Compound-lens work

### Loose priors beat tight truth-seeded priors when the geometry is degenerate

Compound_lens v1 (tight `TruncatedGaussianPrior` on both θ_E) and v2
(wider source box) both FAILED with 12σ coherent arc residuals. Only v3
— with `UniformPrior(0, 8)` on both Einstein radii, NO external shear,
centres un-tied between mass and bulge — matched the user's original
log_Z=+30,705 and produced salt-and-pepper residuals.

**Mechanism.** Tight informative priors on `lens_1.einstein_radius`
(say mean=1.0, sigma=0.3) force the fit to put real secondary mass.
But the data for that mock are *actually* consistent with a
single-effective-deflector at z=0.5 (Keeton & Zabludoff 2004 regime).
The tight prior locks Nautilus into a higher-chi² local optimum where
the forced secondary has to pretend to matter. Only the loose prior
lets the posterior collapse `lens_1.einstein_radius → 0`, which is the
data-driven answer.

**Generalisation.** When model has more freedom than the data resolves,
*loose priors with the zero boundary included* are more robust than
tight informed priors. Let the posterior tell you what the data
supports — don't decide in advance.

Catalogued as Pattern E in `project_fit_failure_patterns.md`.

### Sign-convention drift between autolens versions is axis-0 only

The user's November-2025 notebook (autolens 2025.x) and our 2026.4 v3
fit reached identical log_Z (30,698 vs 30,705) with mathematically
identical posteriors — but every `centre_0` and `ell_comps_0` had its
sign *flipped*. `centre_1`, `ell_comps_1`, θ_E, and shear γ carry
through unchanged.

The rendered `fit_subplot.png` looks mirrored about the y-axis, which
is what made the user notice. But the physics is identical.

**If you see a rendered fit that looks mirrored vs. a reference:** check
axis-0 quantities for a sign flip before assuming the fit converged to
a mirror-image local optimum (Pattern A).

### Single-GPU JAX is 4× slower than numpy at HST-cutout scale

Benchmarked on A100-40GB with `--gres=gpu:1`: `FitImaging.log_likelihood`
at 9,176 masked pixels is **49 ms on numpy** vs **190 ms on jax-GPU**.
Likelihoods match to 6 sig figs so the path is correct — just overhead-
dominated. Host↔device transfer + kernel launches > compute for this
problem size.

`use_jax=False` does what it claims (verified in
`autolens.analysis.analysis.lens.AnalysisLens._xp`): returns plain numpy
when False. JAX still gets imported as a side effect of some autolens
submodule, but isn't called into during the likelihood.

Production path = numpy + Nautilus multiprocessing on CPU. Open question
(tracked in memory): does multi-GPU data-parallel or SLURM-array ×
JAX-per-GPU change this picture? Worth testing on real HST cutouts where
per-call numpy latency is 500 ms+.

### Cannon login-node OOMs when rendering 20+ dim corner plots

`export_results.py --force` on a 31-D posterior with 42 MB of samples
gets OOM-killed on the Cannon login node. Login quota isn't ~20 GB.

**Workaround 1.** Let `export_results.py` skip corners that already
exist (it does by default without `--force`).

**Workaround 2.** Render focused subset corners (6 params) locally using
the downsampled samples.csv. The full 31-D corner is illegible anyway.

**Downsampling.** Weighted-resample samples.csv to 10k rows before
committing if the file is >20 MB. Preserves posterior shape, shrinks
files to 4-7 MB.

### Autolens post-fit viz can hang for 30+ min after Nautilus terminates

Jobs 7351708 (compound_lens v2) and 7431275 (compound_lens Track B) both
got stuck in "Fit Running: Updating results" for >30 min after Nautilus
had written `samples.csv`. Cancelling with `scancel` is safe — samples
are already on disk. `export_results.py` can be run manually afterwards
to produce the lightweight artifacts.

Not yet root-caused. Happens specifically on ~30-D models.

---

## 2026-04-23 (later) — DSPL + disky_spiral build cycle

### `al.Convolver.from_gaussian(...).kernel` is SLIM, not NATIVE

When building a mock PSF and writing to FITS:

```python
# WRONG — writes a (49,) flat array that autolens can't read back
psf = al.Convolver.from_gaussian(shape_native=(7,7), pixel_scales=0.05, sigma=0.08)
fits.PrimaryHDU(data=np.asarray(psf.kernel)).writeto("mock_psf.fits")
```

```python
# RIGHT — use .native for the (7, 7) 2D representation
fits.PrimaryHDU(data=np.asarray(psf.kernel.native)).writeto("mock_psf.fits")
```

Caught at Cannon runtime (3-min FAIL on both DSPL and disky) because
`al.Imaging.from_fits` requires `shape_native` for 1D arrays.

### Uninformative priors are computationally infeasible on 28-D problems

DSPL v1 with `UniformPrior(0.5, 3)` on θ_E, `LogUniformPrior(1e-3, 1e2)`
on all intensities, wide GaussianPrior on source centres: stuck at
f_live=1.0 for 1h52m with log_Z=-17k, never compressing.

DSPL v2 with the same geometry but priors **seeded near the mock truth**
(source centres at truth ± 0.15″, intensities within ×10 of truth,
einstein_radius GaussianPrior at truth ± 0.2): converged in 1h05m with
log_Z=+29,014, max|res|=3.90σ, every MAP value within 0.5% of truth.

**When this matters.** 28-D problems with 5 orders of magnitude of
prior volume per intensity param have O(10¹⁰) worth of prior volume
that Nautilus has to hunt through. It can do it on 10-15 parameter
problems (compound_lens v3 converged with loose priors), but not on
25+ dimensions with wide log-uniform intensities.

**Fix.** Always seed centres and characteristic scales near an external
estimate — image positions for centres, SIS approximation for θ_E,
typical galaxy values for n / R_e. For mocks, use the truth. For real
data, use whatever rough fit you've done first.

### Bayes factor can be e⁶⁰⁰⁰⁰ — watch the exponent

disky_spiral_lens: the two-Sersic fit beat the single-Sersic fit by
log_Z difference **61,969**. That's a Bayes factor of e^61,969,
which is not a number anyone should compute in floating-point — the
exponent is the physical quantity.

This happens when one model is fundamentally unable to fit the data
(single Sersic vs two-PA morphology) — the chi² gap is huge, and
multiplied by the number of data pixels (9176), log_Z differences
scale as O(N_pixel × Δχ²). Don't be alarmed by the number; read it
as "the simpler model is rejected at essentially infinite significance."

### mass_1.einstein_radius collapsing to 0 is informative, not pathological

See Pattern E catalogue for the full story. In short: in the compound_lens
mock, `lens_1.einstein_radius` posterior rail-pinned at its lower limit
of 0. Earlier iterations treated this as a problem. Actually it's a
finding: the data support a single effective deflector. The rail-pinned
posterior tells you "the data can't distinguish this lens from zero mass."

This changes how the `Lens Light Subtracted` panel should be read in
the audit. A "missing" centre residual from a collapsed secondary is
informative, not broken — cite this as an alternate outcome in any
audit template that looks for "two cleanly subtracted centres."

---

## 2026-04-24 — Group-scale build

### Group-scale mocks: the arc visibly remembers every satellite

Wrote a mock with a BGG (θ_E=1.5″) + 3 satellites (θ_E=0.25–0.45″) at
fixed offset positions, all at z=0.4. The Einstein ring is clearly
circular dominated by the BGG, but with **visible kinks and brightness
modulations precisely where each satellite sits**. The arcsinh×5
preview shows three local arc deformations corresponding to the three
satellite positions.

**Why this matters pedagogically.** Unlike compound_lens (where the
secondary's effect is degenerate with shear because it's at a different
redshift, partially absorbable), **satellites at the same z produce
localised, visually-identifiable perturbations** on the arc. A student
looking at the image can *point at* each satellite and predict which
part of the ring it's warping.

This should make the "is the satellite mass real or shear-absorbable?"
question much more visually compelling than the compound_lens version.

### `IsothermalSph` vs `Isothermal` for satellites

For satellites in the mock, I used `al.mp.IsothermalSph` (spherical,
no ell_comps) instead of `al.mp.Isothermal`. Three reasons:

1. **Fewer params per satellite** — SIS has 3 (centre + θ_E) vs SIE's 5.
   For a 3-satellite fit, that's 9 params saved, meaningful at ~30 total.
2. **Satellite ellipticity isn't resolvable on this mock** — the light
   from each satellite is faint, so even if we modelled it as SIE, the
   data couldn't constrain the ell_comps. Nautilus would explore a
   uniformly-posterior direction, wasting compute.
3. **It's the survey-scale default.** Real group-lens papers (e.g.
   cluster pipelines) typically use SIS for satellites and SIE only for
   the BGG, for exactly this reason.

The BGG still gets full SIE in the model — that's where the ellipticity
*is* constrained by the arc shape.

---

## 2026-04-24 — compound_lens v4 vindicates BOTH findings at once

v3 had achieved log_Z=+30,705 with `lens_1.einstein_radius → 0`
(single-effective-deflector) but `max|res|=6.18σ` with coherent chi²
hot spots on the arc and counter-image. v4 added external shear back
to the primary (`ExternalShear` + `GaussianPrior(0, 0.15)`) and:

- log_Z: +30,705 → **+30,856.5** (+151 log units, Bayes factor e^151)
- max|res|: 6.18σ → **4.40σ** (just over the 4σ pass bar but visually
  salt-and-pepper; chi² hot spots gone)
- `lens_1.einstein_radius`: **still 0.000** (still collapsed)
- Recovered shear: γ = (0.007, 0.023), |γ| ≈ 0.024 — small but
  well-separated from zero (3σ)

**The lesson:** the single-effective-deflector finding (lens_1 mass
collapses) and the missing-shear finding (asymmetry absorbed into a
γ≈0.024 shear) are **additive, not competing**. They describe the
same underlying data:

- Primary SIE at z=0.5 with θ_E ≈ 1.78″ and modest ell_comps.
- An extra ~2.4% external shear from whatever isn't modelled
  (large-scale structure, a true distant perturber, or a small
  asymmetry in the true mass profile).
- The z=0.8 "secondary" galaxy is real *as light* but contributes
  essentially zero mass deflection — consistent with Keeton &
  Zabludoff's (2004) degeneracy regime.

v3's architectural mistake was to *remove* shear when adopting the
loose-prior approach (because the user's reference notebook didn't
use shear, and we were matching it). But the reference notebook's
log_Z=30,698 vs v3's log_Z=30,705 is essentially the same value —
they were BOTH stuck in the no-shear basin for reasons that are
probably autolens-version-sign-convention-drift (Pattern B) more
than anything else. v4 is the genuinely better fit.

**For any future Bayes-factor hypothesis test**: adding a
seemingly-redundant term (external shear when you already have an
SIE) can still earn +100 log units if there's actual asymmetry the
other params can't capture. Don't drop terms just because they feel
redundant; let the posterior tell you.

---

## 2026-04-24 — Lesson while debugging group_scale

### "Simpler model" must still model the *light*, or you're comparing apples to oranges

group_scale job v1 stalled at log_Z = -270,000 on the `bgg_shear_only`
variant. Root cause: my variant model was `bgg + source` only. No
satellite light at all. The 3 bright unmodelled satellite cores in the
data were dominating chi² — the fit wasn't actually testing "is
satellite mass resolvable", it was testing "can we fit 3 unmodelled
galaxy cores with shear" (answer: no).

**The pedagogical question was supposed to be**: does modelling
satellite MASS (in addition to their light) do better than shear alone?
For that comparison, both variants need to include satellite LIGHT —
only vary whether mass is on or off.

**Fix.** In the `bgg_shear_only` variant, add each satellite as a
LIGHT-ONLY galaxy (Sersic bulge with fixed centre, no mass component).
In the `bgg_plus_satellites` variant, give each satellite BOTH light
AND a SIS mass. The delta is now purely the 3 einstein_radius params.

**Generalisation.** When comparing models that differ in a single
physical quantity, every *other* component must be present in both
models at the same level of fidelity. "Simpler" should mean "fewer
free parameters at the thing-we're-testing", not "entirely missing
unrelated components."

---

## Open questions to investigate later

- What is autolens's default policy on `centre_0` sign convention changes
  between versions? Is there a flag or shim? (Pattern B context)
- Multi-GPU / array-job JAX speedup — see `project_multigpu_jax_idea.md`
- Post-fit-viz hang root cause — reproducibly on ~30-D fits but not 15-D
- Compound_lens v4 (shear re-enable) — does it drop max|res| below 4σ?
  (CLEANUP_PLAN.md Hypothesis A, queued for next run)

---

*Maintained by Claude Code alongside the user as the curriculum grows.*
*Append as you learn; don't rewrite history. Oldest entries stay put.*
