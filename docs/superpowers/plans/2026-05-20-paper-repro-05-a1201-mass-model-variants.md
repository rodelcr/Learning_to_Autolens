# A1201 Mass-Model Variants Implementation Plan (Spec 05 Phase 4.5)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reproduce N+23's additional mass-model tests (§3.8, §4.3, Appendix D)
that establish the M_BH detection across multiple mass-model classes — total
BPL with/without SMBH, decomposed Sersic+NFW+shear with/without SMBH — and
reproduce §4.3's "100 pc offset" coaxiality argument that rules out the BPL
alternative explanation.

**Architecture:** Five new build functions in `a1201_lens_model.py`, four new
`--part` choices (`bpl`, `bpl_smbh`, `decomp`, `decomp_smbh`), plus a sub-variant
`bpl_free_centre` for the §4.3 reproduction. All variants chain from the
**v4 3-Sersic Stage 1 baseline** (Cannon output `output/lp_3sersic/...`) which
established a real Stage 1 fit. Submissions follow the Stage-2-pattern slurm
with distinct `OUTPUT_SUFFIX` per variant.

**Tech Stack:** PyAutoLens 2026.4 + autofit Nautilus.
- BPL mass: `al.mp.PowerLawBroken` (autolens 2026.4 name for the
  O'Riordan-style broken power law; verify exact class name in Task 0).
  Parameters per O'Riordan, Warren & Mortlock 2019/2020/2021 (cited in
  N+23 §3.4): `einstein_radius`, `inner_slope`, `outer_slope`, `break_radius`.
- Stellar mass component: `al.mp.Sersic` (mass-tracing version with free
  `mass_to_light_ratio`). N+23's full `Ψ + Γ` radial-gradient variant is
  deferred to v0.99 (`al.lmp.SersicGradient` exists but is complex; we
  ship the simpler constant-M/L version first).
- DM halo: `al.mp.NFW` (standard NFW per N+23 §3.4 — generalised gNFW is
  what Spec 03 / Li+26 uses, NOT N+23).
- Shear: `al.mp.ExternalShear` (matches PL stage).

**Depends on:**
- v4 Stage 1 baseline = `private/2303_.../output/lp_3sersic/a1201_lp/a1201_lp/<hash>/files/samples_summary.json` (Cannon job 14015488, lnZ=+174,904).
- Existing `chain_priors_from_lp.py` (chains shared LP parameters across mass models).
- Existing `compute_bayes_factor.py` (emits per-variant ΔlnZ).

---

## File Structure

```
private/2303_15514_nightingale2023_abell1201/
├── code/
│   ├── a1201_lens_model.py            ← MODIFY: add 4 build funcs + 4 part choices
│   ├── bpl_oriordan.py                ← NEW (Task 1): documents the BPL parameterization
│   ├── chain_priors_from_lp.py        ← MODIFY: extend LP_TO_CHAIN with BPL + decomposed params
│   ├── compute_bayes_matrix.py        ← NEW (Task 7): ΔlnZ matrix across all 6 mass variants × ±SMBH
│   └── audit_stage1.py                ← MODIFY: extend PRIOR_BOUNDS for multi-Sersic + BPL
├── submit_a1201.slurm                 ← already supports STAGE=<arbitrary> + OUTPUT_SUFFIX
├── tests/
│   └── test_a1201_lens_model.py       ← MODIFY: add 5 tests for BPL + decomposed + tied-centre
└── notebooks/
    └── 04_a1201_mass_model_matrix.ipynb  ← NEW (Task 9): ΔlnZ matrix headline figure
```

---

## Task 0: Verify autolens 2026.4 class names

The autolens API has evolved through 2026.4 with several renames. Before
writing the build functions, confirm the exact class names by inspecting
the env on Cannon (where autolens 2026.4.13.6 is installed).

**Files:** none (read-only inspection)

- [ ] **Step 1: Verify the BPL class name**

```bash
ssh cannon "source /n/sw/Miniforge3-25.3.1-0/etc/profile.d/conda.sh && conda activate autolens312 && python -c '
import autolens as al
import inspect
candidates = [name for name in dir(al.mp) if (\"PowerLaw\" in name or \"BrokenPower\" in name) and \"Sph\" not in name]
print(\"BPL candidate classes in al.mp:\", candidates)
for name in candidates:
    cls = getattr(al.mp, name)
    sig = inspect.signature(cls.__init__)
    print(f\"  {name}{sig}\")
'"
```

Expected: a class like `PowerLawBroken` (or `BrokenPowerLaw`) with constructor
parameters including `einstein_radius`, `inner_slope`, `outer_slope`,
`break_radius`, `centre`, `ell_comps`. Record the verified class name.

- [ ] **Step 2: Verify the NFW class name**

```bash
ssh cannon "source /n/sw/Miniforge3-25.3.1-0/etc/profile.d/conda.sh && conda activate autolens312 && python -c '
import autolens as al
nfw_candidates = [n for n in dir(al.mp) if \"NFW\" in n and \"Sph\" not in n and \"Generalize\" not in n.lower() and \"gNFW\" not in n.lower()]
print(\"NFW candidates:\", nfw_candidates)
import inspect
for n in nfw_candidates[:5]:
    cls = getattr(al.mp, n)
    print(f\"  {n}{inspect.signature(cls.__init__)}\")
'"
```

Expected: `NFW` or `NFWMCRDuffy` etc. with parameters including `centre`,
`ell_comps`, `kappa_s` (or `mass_at_200`), `scale_radius`. Record the
verified class name.

- [ ] **Step 3: Verify the Sersic mass-profile class name**

```bash
ssh cannon "source /n/sw/Miniforge3-25.3.1-0/etc/profile.d/conda.sh && conda activate autolens312 && python -c '
import autolens as al
print(\"al.mp.Sersic exists:\", hasattr(al.mp, \"Sersic\"))
if hasattr(al.mp, \"Sersic\"):
    import inspect
    print(\"signature:\", inspect.signature(al.mp.Sersic.__init__))
print(\"al.lmp.Sersic exists:\", hasattr(al, \"lmp\") and hasattr(al.lmp, \"Sersic\"))
'"
```

Expected: `al.mp.Sersic` exists with parameters including `centre`, `ell_comps`,
`intensity`, `effective_radius`, `sersic_index`, `mass_to_light_ratio`. Record
the verified call.

- [ ] **Step 4: Record verified class names**

Append a section to `private/2303_15514_nightingale2023_abell1201/PAPER_NOTES.md`:

```markdown
## autolens 2026.4 class-name verification (2026-05-20)

For the mass-model-variant reproduction (`docs/superpowers/plans/2026-05-20-
paper-repro-05-a1201-mass-model-variants.md`):

- BPL:        `al.mp.<VERIFIED>` — parameters: <list>
- NFW:        `al.mp.<VERIFIED>` — parameters: <list>
- Sersic-mass: `al.mp.<VERIFIED>` — parameters: <list>
```

Replace `<VERIFIED>` and `<list>` with the actual outputs from steps 1-3.

- [ ] **Step 5: Commit the PAPER_NOTES update**

(`private/` is gitignored, so no git commit here — just verify the file was
edited and saved.)

---

## Task 1: BPL parameterization reference doc

**Files:**
- Create: `private/2303_15514_nightingale2023_abell1201/code/bpl_oriordan.py`

Documents (as Python docstrings) the O'Riordan BPL parameterization N+23
uses, with the explicit functional form so future readers don't need to
chase the citation.

- [ ] **Step 1: Write the doc file**

```python
"""bpl_oriordan.py — O'Riordan broken power-law mass profile reference.

N+23 §3.4 uses the BPL parameterization of O'Riordan, Warren & Mortlock
(2019, 2020, 2021). The functional form:

    κ(R) = κ_0 × (R / R_B) ^ (-t1)      for R ≤ R_B
    κ(R) = κ_0 × (R / R_B) ^ (-t2)      for R > R_B

where:
    R    = elliptical-projected radius
    R_B  = break radius (free parameter)
    t1   = inner density slope (free; flatter ⇒ smaller t1)
    t2   = outer density slope (free; near 1 for isothermal-outer)
    κ_0  = normalisation tied to θ_E via integration condition

This is the parameterization that N+23 §4.3 tests as the alternative to
an SMBH: a sufficiently flexible inner slope (small t1, small R_B) can
mimic the central deflection of a point mass.

Autolens 2026.4 class (verified Task 0): al.mp.<see PAPER_NOTES update>

References:
  O'Riordan, Warren & Mortlock 2019, MNRAS 487:5060 — "Towards a unified
    parameterization of strong-lensing perturber profiles" (BPL introduced)
  O'Riordan, Warren & Mortlock 2020, MNRAS 496:3424 — BPL applied to subhalo
  O'Riordan, Warren & Mortlock 2021, MNRAS 501:3687 — BPL methodology paper

  Nightingale et al. 2023, arxiv 2303.15514, §3.4 (BPL citation) + §4.3
  (BPL alternative test) + Appendix D (tied-vs-free centre BPL comparison)
"""
```

- [ ] **Step 2: Verify**

```bash
conda run -n autolens python -c "import sys; sys.path.insert(0, 'private/2303_15514_nightingale2023_abell1201/code'); import bpl_oriordan; print(bpl_oriordan.__doc__[:200])"
```

Expected: prints the first 200 chars of the module docstring.

---

## Task 2: Add `build_bpl_fit` and `build_bpl_smbh_fit` to driver

**Files:**
- Modify: `private/2303_15514_nightingale2023_abell1201/code/a1201_lens_model.py`

Two new build functions. Both default to **mass.centre TIED to bulge.centre**
(matches our radial-arc-methodology canonical pattern), with a
`free_mass_centre` flag for the §4.3 reproduction.

- [ ] **Step 1: Write a failing test first**

Add to `tests/test_a1201_lens_model.py`:

```python
def test_build_bpl_fit_callable():
    """build_bpl_fit must exist and accept the standard kwargs."""
    from a1201_lens_model import build_bpl_fit
    import inspect
    assert callable(build_bpl_fit)
    sig = inspect.signature(build_bpl_fit)
    for arg in ("dataset", "output_root", "n_live", "chain_from",
                "n_light_components", "informed_priors", "free_mass_centre"):
        assert arg in sig.parameters, f"missing arg {arg!r}"


def test_build_bpl_smbh_fit_includes_smbh():
    """build_bpl_smbh_fit must construct a lens with `smbh` attribute."""
    from a1201_lens_model import _build_lens_galaxy_bpl
    lens = _build_lens_galaxy_bpl(include_pointmass=True, n_light_components=1,
                                    free_mass_centre=False)
    info = lens.info
    assert "smbh" in info
    assert "PowerLawBroken" in info or "BrokenPowerLaw" in info or "Broken" in info


def test_build_bpl_free_centre_breaks_tie():
    """When free_mass_centre=True, mass.centre is NOT the same Prior object as bulge.centre."""
    from a1201_lens_model import _build_lens_galaxy_bpl
    lens_tied = _build_lens_galaxy_bpl(include_pointmass=False, n_light_components=1,
                                         free_mass_centre=False)
    lens_free = _build_lens_galaxy_bpl(include_pointmass=False, n_light_components=1,
                                         free_mass_centre=True)
    assert lens_tied.mass.centre.centre_0 is lens_tied.bulge.centre.centre_0
    assert lens_free.mass.centre.centre_0 is not lens_free.bulge.centre.centre_0
```

- [ ] **Step 2: Run failing test**

```bash
cd private/2303_15514_nightingale2023_abell1201
conda run -n autolens pytest tests/test_a1201_lens_model.py::test_build_bpl_fit_callable -v
```
Expected: FAIL with `ImportError` for `build_bpl_fit`.

- [ ] **Step 3: Implement the BPL helpers in a1201_lens_model.py**

Append after `build_adapt_fit`:

```python
def _build_lens_galaxy_bpl(include_pointmass: bool = False,
                            n_light_components: int = 1,
                            free_mass_centre: bool = False):
    """Lens galaxy with O'Riordan BPL mass profile (N+23 §3.4 alternative).

    The §4.3 reproduction uses free_mass_centre=True: the BPL centre is
    unconstrained by the bulge centre, allowing the fit to drift to ≥100 pc
    offset (N+23 Appendix D shows tied-centre BPL has much lower lnZ).
    """
    import autofit as af
    import autolens as al

    # Reuse the multi-Sersic bulge / disk / envelope from _build_lens_galaxy_model.
    # Need a helper that returns just the light kwargs + bulge.centre anchor.
    from a1201_lens_model import _make_sersic_light_component   # re-use
    if n_light_components not in (1, 2, 3):
        raise ValueError(f"n_light_components must be 1, 2, or 3 (got {n_light_components})")
    bulge = _make_sersic_light_component(
        centre_anchor=None,
        effective_radius_bounds=(0.3, 3.0),
        sersic_index_bounds=(2.0, 6.0),
    )
    light_kwargs = dict(bulge=bulge)
    centre_anchor = bulge.centre
    if n_light_components >= 2:
        light_kwargs["disk"] = _make_sersic_light_component(
            centre_anchor=centre_anchor,
            effective_radius_bounds=(1.0, 6.0),
            sersic_index_bounds=(0.5, 3.0),
        )
    if n_light_components >= 3:
        light_kwargs["envelope"] = _make_sersic_light_component(
            centre_anchor=centre_anchor,
            effective_radius_bounds=(3.0, 15.0),
            sersic_index_bounds=(0.3, 2.0),
        )

    # BPL mass — class name TBD by Task 0 verification. Assume PowerLawBroken
    # for now; switch if Task 0 reveals a different name.
    mass = af.Model(al.mp.PowerLawBroken)
    if free_mass_centre:
        mass.centre.centre_0 = af.GaussianPrior(0.0, 0.2)
        mass.centre.centre_1 = af.GaussianPrior(0.0, 0.2)
    else:
        mass.centre = centre_anchor
    mass.ell_comps.ell_comps_0 = af.GaussianPrior(0.0, 0.3)
    mass.ell_comps.ell_comps_1 = af.GaussianPrior(0.0, 0.3)
    mass.einstein_radius = af.UniformPrior(0.5, 5.0)
    # BPL-specific: inner_slope, outer_slope, break_radius
    mass.inner_slope = af.UniformPrior(0.5, 2.5)
    mass.outer_slope = af.UniformPrior(0.5, 2.5)
    mass.break_radius = af.UniformPrior(0.05, 3.0)

    shear = af.Model(al.mp.ExternalShear)
    shear.gamma_1 = af.GaussianPrior(0.0, 0.15)
    shear.gamma_2 = af.GaussianPrior(0.0, 0.15)

    kwargs = dict(redshift=Z_LENS, mass=mass, shear=shear, **light_kwargs)
    if include_pointmass:
        smbh = af.Model(al.mp.PointMass)
        smbh.centre = centre_anchor   # tied to bulge regardless of free_mass_centre
        smbh.einstein_radius = af.LogUniformPrior(1e-3, 0.5)
        kwargs["smbh"] = smbh

    return af.Model(al.Galaxy, **kwargs)


def build_bpl_fit(dataset, output_root: Path, n_live: int = 250,
                    chain_from: Path | None = None,
                    n_light_components: int = 1,
                    informed_priors: bool = False,
                    free_mass_centre: bool = False):
    """N+23 §3.4 BPL alternative — no SMBH."""
    import autofit as af
    import autolens as al
    lens = _build_lens_galaxy_bpl(include_pointmass=False,
                                    n_light_components=n_light_components,
                                    free_mass_centre=free_mass_centre)
    if informed_priors:
        _apply_informed_priors(lens)
    _maybe_apply_chain(lens, chain_from)
    source = _build_source_galaxy_model_parametric()
    model = af.Collection(galaxies=af.Collection(lens=lens, source=source))
    analysis = al.AnalysisImaging(dataset=dataset, use_jax=False)
    tag = "a1201_bpl" + ("_free" if free_mass_centre else "")
    search = af.Nautilus(
        path_prefix=str(output_root), name=tag, unique_tag=tag,
        n_live=n_live,
        number_of_cores=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
    )
    return search.fit(model=model, analysis=analysis)


def build_bpl_smbh_fit(dataset, output_root: Path, n_live: int = 300,
                        chain_from: Path | None = None,
                        n_light_components: int = 1,
                        informed_priors: bool = False,
                        free_mass_centre: bool = False):
    """N+23 §3.4 BPL alternative WITH central PointMass — companion to build_bpl_fit."""
    import autofit as af
    import autolens as al
    lens = _build_lens_galaxy_bpl(include_pointmass=True,
                                    n_light_components=n_light_components,
                                    free_mass_centre=free_mass_centre)
    if informed_priors:
        _apply_informed_priors(lens)
    _maybe_apply_chain(lens, chain_from)
    source = _build_source_galaxy_model_parametric()
    model = af.Collection(galaxies=af.Collection(lens=lens, source=source))
    analysis = al.AnalysisImaging(dataset=dataset, use_jax=False)
    tag = "a1201_bpl_smbh" + ("_free" if free_mass_centre else "")
    search = af.Nautilus(
        path_prefix=str(output_root), name=tag, unique_tag=tag,
        n_live=n_live,
        number_of_cores=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
    )
    return search.fit(model=model, analysis=analysis)
```

- [ ] **Step 4: Run tests (should now pass)**

```bash
cd private/2303_15514_nightingale2023_abell1201
conda run -n autolens pytest tests/test_a1201_lens_model.py -v -k "bpl"
```
Expected: 3 BPL tests PASS.

- [ ] **Step 5: If Task 0 revealed a different BPL class name**, search-and-replace
  `al.mp.PowerLawBroken` in the implementation with the verified name. Re-run tests.

---

## Task 3: Add `build_decomposed_fit` and `build_decomposed_smbh_fit` to driver

**Files:**
- Modify: `private/2303_15514_nightingale2023_abell1201/code/a1201_lens_model.py`

Decomposed mass: each light Sersic component gets a stellar-mass twin
(`al.mp.Sersic` with free `mass_to_light_ratio`), tied morphologically to the
light component but with independent normalisation. Plus an elliptical NFW
halo (standard NFW per N+23 §3.4, NOT the gNFW that Spec 03 uses) + ExternalShear.

**Note on divergence from N+23**: N+23 uses Sersic-mass with both `Ψ`
(M/L scale) AND `Γ` (M/L radial gradient) per component. This first-pass
implementation uses constant M/L per component (only `Ψ`, no `Γ`). The
gradient extension lives behind an `al.lmp.SersicGradient`-style class
and is deferred to v0.99 — document in §3 of the comparison notebook.

- [ ] **Step 1: Add failing tests**

Add to `tests/test_a1201_lens_model.py`:

```python
def test_build_decomposed_fit_callable():
    from a1201_lens_model import build_decomposed_fit
    import inspect
    assert callable(build_decomposed_fit)
    sig = inspect.signature(build_decomposed_fit)
    for arg in ("dataset", "output_root", "n_live", "chain_from",
                "n_light_components", "informed_priors"):
        assert arg in sig.parameters


def test_decomposed_has_nfw_and_stellar_mass_components():
    from a1201_lens_model import _build_lens_galaxy_decomposed
    lens = _build_lens_galaxy_decomposed(include_pointmass=False,
                                           n_light_components=1)
    info = lens.info
    assert "stellar_mass" in info or "stellar" in info or "Sersic" in info
    assert "nfw" in info or "NFW" in info
    assert "shear" in info or "ExternalShear" in info


def test_decomposed_with_smbh_adds_pointmass():
    from a1201_lens_model import _build_lens_galaxy_decomposed
    lens = _build_lens_galaxy_decomposed(include_pointmass=True,
                                           n_light_components=1)
    assert "smbh" in lens.info
```

- [ ] **Step 2: Run failing tests**

```bash
conda run -n autolens pytest tests/test_a1201_lens_model.py -v -k "decomposed"
```
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement the decomposed helpers**

Append to `a1201_lens_model.py`:

```python
def _make_sersic_stellar_mass_component(centre_anchor, ell_anchor=None,
                                          effective_radius_anchor=None,
                                          sersic_index_anchor=None):
    """Stellar-mass Sersic — tied morphologically to the light Sersic.

    Per N+23 §3.4, each light Sersic component has a corresponding stellar-mass
    Sersic with the SAME shape (centre, ell, R_eff, n) but a free
    mass_to_light_ratio parameter Ψ. Here we tie the morphology to the
    pre-built light Sersic by reusing its Prior objects.
    """
    import autofit as af
    import autolens as al
    m = af.Model(al.mp.Sersic)
    m.centre = centre_anchor                       # tied to light centre
    if ell_anchor is not None:
        m.ell_comps = ell_anchor                   # tied to light ell
    else:
        m.ell_comps.ell_comps_0 = af.GaussianPrior(0.0, 0.3)
        m.ell_comps.ell_comps_1 = af.GaussianPrior(0.0, 0.3)
    if effective_radius_anchor is not None:
        m.effective_radius = effective_radius_anchor
    else:
        m.effective_radius = af.UniformPrior(0.3, 6.0)
    if sersic_index_anchor is not None:
        m.sersic_index = sersic_index_anchor
    else:
        m.sersic_index = af.UniformPrior(1.0, 6.0)
    # mass_to_light_ratio — wide LogUniform; this is what carries the stellar mass
    m.mass_to_light_ratio = af.LogUniformPrior(0.01, 50.0)
    return m


def _build_lens_galaxy_decomposed(include_pointmass: bool = False,
                                    n_light_components: int = 1):
    """Lens with decomposed stellar + dark mass — N+23 §3.4."""
    import autofit as af
    import autolens as al

    if n_light_components not in (1, 2, 3):
        raise ValueError(f"n_light_components must be 1, 2, or 3 (got {n_light_components})")

    # Light components (reusing the same pattern as _build_lens_galaxy_model)
    from a1201_lens_model import _make_sersic_light_component
    bulge = _make_sersic_light_component(
        centre_anchor=None,
        effective_radius_bounds=(0.3, 3.0),
        sersic_index_bounds=(2.0, 6.0),
    )
    light_kwargs = dict(bulge=bulge)
    centre_anchor = bulge.centre
    if n_light_components >= 2:
        light_kwargs["disk"] = _make_sersic_light_component(
            centre_anchor=centre_anchor,
            effective_radius_bounds=(1.0, 6.0),
            sersic_index_bounds=(0.5, 3.0),
        )
    if n_light_components >= 3:
        light_kwargs["envelope"] = _make_sersic_light_component(
            centre_anchor=centre_anchor,
            effective_radius_bounds=(3.0, 15.0),
            sersic_index_bounds=(0.3, 2.0),
        )

    # Stellar mass — one Sersic-mass per light component, sharing the
    # morphology Prior objects.
    stellar_mass = _make_sersic_stellar_mass_component(
        centre_anchor=light_kwargs["bulge"].centre,
        ell_anchor=light_kwargs["bulge"].ell_comps,
        effective_radius_anchor=light_kwargs["bulge"].effective_radius,
        sersic_index_anchor=light_kwargs["bulge"].sersic_index,
    )

    # Dark matter — standard elliptical NFW per N+23 §3.4 (NOT generalised)
    nfw = af.Model(al.mp.NFW)
    nfw.centre = centre_anchor                          # tied to BCG centre
    nfw.ell_comps.ell_comps_0 = af.GaussianPrior(0.0, 0.3)
    nfw.ell_comps.ell_comps_1 = af.GaussianPrior(0.0, 0.3)
    nfw.kappa_s = af.UniformPrior(0.001, 0.5)           # convergence at scale radius
    nfw.scale_radius = af.UniformPrior(1.0, 50.0)       # arcsec; large for cluster-scale halo

    shear = af.Model(al.mp.ExternalShear)
    shear.gamma_1 = af.GaussianPrior(0.0, 0.15)
    shear.gamma_2 = af.GaussianPrior(0.0, 0.15)

    # NOTE: this version uses ONE stellar-mass component anchored to the bulge.
    # For paper-faithful N+23 §3.4: each of n_light_components light Sersics
    # has its own stellar-mass twin. Multi-component stellar mass is a v0.99
    # extension; for now we test whether single-component stellar + NFW
    # suffices.
    kwargs = dict(redshift=Z_LENS, stellar_mass=stellar_mass, dark_matter=nfw,
                  shear=shear, **light_kwargs)
    if include_pointmass:
        smbh = af.Model(al.mp.PointMass)
        smbh.centre = centre_anchor
        smbh.einstein_radius = af.LogUniformPrior(1e-3, 0.5)
        kwargs["smbh"] = smbh

    return af.Model(al.Galaxy, **kwargs)


def build_decomposed_fit(dataset, output_root: Path, n_live: int = 300,
                          chain_from: Path | None = None,
                          n_light_components: int = 1,
                          informed_priors: bool = False):
    """N+23 §3.4 decomposed Sersic+NFW+shear mass model — no SMBH."""
    import autofit as af
    import autolens as al
    lens = _build_lens_galaxy_decomposed(include_pointmass=False,
                                           n_light_components=n_light_components)
    if informed_priors:
        _apply_informed_priors(lens)   # NB: this currently anchors PL.slope; may fail on decomposed lens — see Task 4 fix
    _maybe_apply_chain(lens, chain_from)
    source = _build_source_galaxy_model_parametric()
    model = af.Collection(galaxies=af.Collection(lens=lens, source=source))
    analysis = al.AnalysisImaging(dataset=dataset, use_jax=False)
    search = af.Nautilus(
        path_prefix=str(output_root), name="a1201_decomp", unique_tag="a1201_decomp",
        n_live=n_live,
        number_of_cores=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
    )
    return search.fit(model=model, analysis=analysis)


def build_decomposed_smbh_fit(dataset, output_root: Path, n_live: int = 350,
                                chain_from: Path | None = None,
                                n_light_components: int = 1,
                                informed_priors: bool = False):
    """N+23 §3.4 decomposed mass model WITH central PointMass."""
    import autofit as af
    import autolens as al
    lens = _build_lens_galaxy_decomposed(include_pointmass=True,
                                           n_light_components=n_light_components)
    if informed_priors:
        _apply_informed_priors(lens)
    _maybe_apply_chain(lens, chain_from)
    source = _build_source_galaxy_model_parametric()
    model = af.Collection(galaxies=af.Collection(lens=lens, source=source))
    analysis = al.AnalysisImaging(dataset=dataset, use_jax=False)
    search = af.Nautilus(
        path_prefix=str(output_root), name="a1201_decomp_smbh",
        unique_tag="a1201_decomp_smbh", n_live=n_live,
        number_of_cores=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
    )
    return search.fit(model=model, analysis=analysis)
```

- [ ] **Step 4: Run tests (should now pass)**

```bash
conda run -n autolens pytest tests/test_a1201_lens_model.py -v -k "decomposed"
```
Expected: 3 decomposed tests PASS.

---

## Task 4: Add `--part={bpl,bpl_smbh,decomp,decomp_smbh}` to driver argparse

**Files:**
- Modify: `private/2303_15514_nightingale2023_abell1201/code/a1201_lens_model.py:181-184`
  (the `--part` choices list)
- Modify: same file's `main()` dispatch + add `--free-mass-centre` flag

- [ ] **Step 1: Failing test**

Add to `tests/test_a1201_lens_model.py`:

```python
def test_argparse_includes_bpl_and_decomp_parts():
    """The driver must expose --part={bpl,bpl_smbh,decomp,decomp_smbh}."""
    import subprocess, sys
    from pathlib import Path
    code_dir = Path(__file__).resolve().parents[1] / "code"
    res = subprocess.run(
        [sys.executable, str(code_dir / "a1201_lens_model.py"), "--help"],
        capture_output=True, text=True,
    )
    for part in ["bpl", "bpl_smbh", "decomp", "decomp_smbh"]:
        assert part in res.stdout, f"--part={part} not in --help output"
    assert "--free-mass-centre" in res.stdout, "missing --free-mass-centre flag"
```

- [ ] **Step 2: Run failing test**

```bash
conda run -n autolens pytest tests/test_a1201_lens_model.py::test_argparse_includes_bpl_and_decomp_parts -v
```
Expected: FAIL — current `choices` list doesn't have these.

- [ ] **Step 3: Update `main()`**

```python
# In a1201_lens_model.main(), modify:
    p.add_argument("--part",
                   choices=["lp", "with_smbh", "with_kin", "adapt",
                            "bpl", "bpl_smbh", "decomp", "decomp_smbh"],
                   default="lp")
    # ... existing args ...
    p.add_argument("--free-mass-centre", action="store_true",
                   help="For BPL variants: free the mass centre from bulge.centre. "
                        "Used for N+23 §4.3 / Appendix D coaxiality reproduction.")
    args = p.parse_args()

    # ... after existing branches ...
    elif args.part == "bpl":
        build_bpl_fit(dataset, args.output_root, n_live=args.n_live,
                      chain_from=args.chain_from,
                      n_light_components=args.n_light,
                      informed_priors=args.informed_priors,
                      free_mass_centre=args.free_mass_centre)
    elif args.part == "bpl_smbh":
        build_bpl_smbh_fit(dataset, args.output_root, n_live=args.n_live,
                            chain_from=args.chain_from,
                            n_light_components=args.n_light,
                            informed_priors=args.informed_priors,
                            free_mass_centre=args.free_mass_centre)
    elif args.part == "decomp":
        build_decomposed_fit(dataset, args.output_root, n_live=args.n_live,
                             chain_from=args.chain_from,
                             n_light_components=args.n_light,
                             informed_priors=args.informed_priors)
    elif args.part == "decomp_smbh":
        build_decomposed_smbh_fit(dataset, args.output_root, n_live=args.n_live,
                                   chain_from=args.chain_from,
                                   n_light_components=args.n_light,
                                   informed_priors=args.informed_priors)
```

Also update the slurm `case "${STAGE}"` block to map the new STAGE values to
N_LIVE defaults — add to `submit_a1201.slurm` line ~57:

```bash
case "${STAGE}" in
    lp)           N_LIVE=200 ;;
    with_smbh)    N_LIVE=250 ;;
    with_kin)     N_LIVE=300 ;;
    adapt)        N_LIVE=400 ;;
    bpl)          N_LIVE=250 ;;
    bpl_smbh)     N_LIVE=300 ;;
    decomp)       N_LIVE=300 ;;
    decomp_smbh)  N_LIVE=350 ;;
    *) echo "ERROR: unknown STAGE=${STAGE}" >&2 ; exit 2 ;;
esac
```

And update the slurm to pass `--free-mass-centre` when `FREE_MASS_CENTRE=1` is in env:

```bash
FREE_MASS_CENTRE="${FREE_MASS_CENTRE:-0}"
# ... near INFORMED_ARG ...
FREE_CENTRE_ARG=""
if [[ "${FREE_MASS_CENTRE}" == "1" ]]; then
    FREE_CENTRE_ARG="--free-mass-centre"
fi
# ... in the srun command ...
${CHAIN_ARG} ${INFORMED_ARG} ${FREE_CENTRE_ARG}
```

- [ ] **Step 4: Run all tests**

```bash
conda run -n autolens pytest tests/test_a1201_lens_model.py -v
```
Expected: all tests (12 original + 6 new = 18) PASS.

- [ ] **Step 5: Lint the slurm**

```bash
bash -n private/2303_15514_nightingale2023_abell1201/submit_a1201.slurm
```
Expected: exit 0.

---

## Task 5: Push driver + slurm to Cannon, submit BPL variants

**Files:** none directly modified; uses already-edited driver + slurm.

- [ ] **Step 1: Push**

```bash
bash Modules/10_Cluster_Computing/scripts/push_to_cannon.sh --go
```
Expected: rsync exit 0.

- [ ] **Step 2: Submit `bpl` (tied centre, no SMBH) chained from v4**

```bash
ssh cannon 'cd /n/holystore01/LABS/hernquist_lab/Lab/rcordova/learning_to_autolens && \
            STAGE1_SUMMARY=$(ls /n/holystore01/LABS/hernquist_lab/Lab/rcordova/learning_to_autolens/private/2303_15514_nightingale2023_abell1201/output/lp_3sersic/a1201_lp/a1201_lp/*/files/samples_summary.json | head -1) && \
            sbatch --parsable --export=ALL,STAGE=bpl,MASK_RADIUS=6.0,N_LIGHT=3,OUTPUT_SUFFIX=_3sersic,CHAIN_FROM=$STAGE1_SUMMARY \
                   --job-name=a1201_bpl_3s \
                   private/2303_15514_nightingale2023_abell1201/submit_a1201.slurm'
```

Expected: prints a job ID. Record it.

- [ ] **Step 3: Submit `bpl_smbh` (tied centre, +SMBH)**

```bash
ssh cannon 'cd /n/holystore01/LABS/hernquist_lab/Lab/rcordova/learning_to_autolens && \
            STAGE1_SUMMARY=$(ls /n/holystore01/LABS/hernquist_lab/Lab/rcordova/learning_to_autolens/private/2303_15514_nightingale2023_abell1201/output/lp_3sersic/a1201_lp/a1201_lp/*/files/samples_summary.json | head -1) && \
            sbatch --parsable --export=ALL,STAGE=bpl_smbh,MASK_RADIUS=6.0,N_LIGHT=3,OUTPUT_SUFFIX=_3sersic,CHAIN_FROM=$STAGE1_SUMMARY \
                   --job-name=a1201_bpl_smbh_3s \
                   private/2303_15514_nightingale2023_abell1201/submit_a1201.slurm'
```

- [ ] **Step 4: Submit `bpl_smbh` with **FREE centre** — the §4.3 / App D reproduction**

```bash
ssh cannon 'cd /n/holystore01/LABS/hernquist_lab/Lab/rcordova/learning_to_autolens && \
            STAGE1_SUMMARY=$(ls /n/holystore01/LABS/hernquist_lab/Lab/rcordova/learning_to_autolens/private/2303_15514_nightingale2023_abell1201/output/lp_3sersic/a1201_lp/a1201_lp/*/files/samples_summary.json | head -1) && \
            sbatch --parsable --export=ALL,STAGE=bpl_smbh,MASK_RADIUS=6.0,N_LIGHT=3,OUTPUT_SUFFIX=_free_centre,CHAIN_FROM=$STAGE1_SUMMARY,FREE_MASS_CENTRE=1 \
                   --job-name=a1201_bpl_smbh_free \
                   private/2303_15514_nightingale2023_abell1201/submit_a1201.slurm'
```

- [ ] **Step 5: Submit `bpl` with **FREE centre** — for the §4.3 paired comparison**

```bash
ssh cannon 'cd /n/holystore01/LABS/hernquist_lab/Lab/rcordova/learning_to_autolens && \
            STAGE1_SUMMARY=$(ls /n/holystore01/LABS/hernquist_lab/Lab/rcordova/learning_to_autolens/private/2303_15514_nightingale2023_abell1201/output/lp_3sersic/a1201_lp/a1201_lp/*/files/samples_summary.json | head -1) && \
            sbatch --parsable --export=ALL,STAGE=bpl,MASK_RADIUS=6.0,N_LIGHT=3,OUTPUT_SUFFIX=_free_centre,CHAIN_FROM=$STAGE1_SUMMARY,FREE_MASS_CENTRE=1 \
                   --job-name=a1201_bpl_free \
                   private/2303_15514_nightingale2023_abell1201/submit_a1201.slurm'
```

- [ ] **Step 6: Verify all 4 BPL jobs are queued**

```bash
ssh cannon "squeue --me --format='%.10i %.25j %.10P %.10T %.12L %R'"
```
Expected: 4 a1201_bpl* jobs queued or running.

---

## Task 6: Submit decomposed variants (chained from v4)

- [ ] **Step 1: Submit `decomp` (decomposed, no SMBH)**

```bash
ssh cannon 'cd /n/holystore01/LABS/hernquist_lab/Lab/rcordova/learning_to_autolens && \
            STAGE1_SUMMARY=$(ls /n/holystore01/LABS/hernquist_lab/Lab/rcordova/learning_to_autolens/private/2303_15514_nightingale2023_abell1201/output/lp_3sersic/a1201_lp/a1201_lp/*/files/samples_summary.json | head -1) && \
            sbatch --parsable --export=ALL,STAGE=decomp,MASK_RADIUS=6.0,N_LIGHT=3,OUTPUT_SUFFIX=_3sersic,CHAIN_FROM=$STAGE1_SUMMARY \
                   --job-name=a1201_decomp_3s --time=24:00:00 \
                   private/2303_15514_nightingale2023_abell1201/submit_a1201.slurm'
```

- [ ] **Step 2: Submit `decomp_smbh`**

```bash
ssh cannon 'cd /n/holystore01/LABS/hernquist_lab/Lab/rcordova/learning_to_autolens && \
            STAGE1_SUMMARY=$(ls /n/holystore01/LABS/hernquist_lab/Lab/rcordova/learning_to_autolens/private/2303_15514_nightingale2023_abell1201/output/lp_3sersic/a1201_lp/a1201_lp/*/files/samples_summary.json | head -1) && \
            sbatch --parsable --export=ALL,STAGE=decomp_smbh,MASK_RADIUS=6.0,N_LIGHT=3,OUTPUT_SUFFIX=_3sersic,CHAIN_FROM=$STAGE1_SUMMARY \
                   --job-name=a1201_decomp_smbh_3s --time=36:00:00 \
                   private/2303_15514_nightingale2023_abell1201/submit_a1201.slurm'
```

- [ ] **Step 3: Verify all 6 variant jobs queued**

```bash
ssh cannon "squeue --me --format='%.10i %.25j %.10P %.10T %.12L %R'"
```
Expected: a1201_lp_3sersic STAGE 2 (with_smbh) running + 4 BPL variants + 2 decomposed = 7 jobs total (if Stage 2 hasn't landed) or 6 (if it has).

---

## Task 7: Bayes-factor matrix script

**Files:**
- Create: `private/2303_15514_nightingale2023_abell1201/code/compute_bayes_matrix.py`

Wraps `compute_bayes_factor.py` to produce a full ΔlnZ matrix across all
6 mass-model variants, mirroring N+23 Table 4 / §4.3.

- [ ] **Step 1: Write a failing test**

Add to `tests/test_a1201_lens_model.py`:

```python
def test_compute_bayes_matrix_callable_with_stub_dirs(tmp_path):
    """compute_bayes_matrix should aggregate over multiple variant output dirs."""
    import sys
    code_dir = HERE.parents[1] / "code"
    sys.path.insert(0, str(code_dir))
    from compute_bayes_matrix import build_matrix

    # Build stub output directories with samples_summary.json files
    import json
    variants = {
        "pl":         172904,
        "pl_smbh":    173500,
        "bpl":        172800,
        "bpl_smbh":   173400,
        "decomp":     173000,
        "decomp_smbh": 173600,
    }
    paths = {}
    for v, lz in variants.items():
        d = tmp_path / f"out_{v}" / "files"
        d.mkdir(parents=True)
        (d / "samples_summary.json").write_text(json.dumps({
            "type": "instance", "class_path": "stub",
            "arguments": {"log_evidence": float(lz)},
        }))
        paths[v] = tmp_path / f"out_{v}"

    matrix = build_matrix(paths)
    # Each "with_smbh" ΔlnZ should match the test data
    assert abs(matrix["pl"]["smbh_vs_no"] - 596) < 1e-3        # 173500 - 172904
    assert abs(matrix["bpl"]["smbh_vs_no"] - 600) < 1e-3       # 173400 - 172800
    assert abs(matrix["decomp"]["smbh_vs_no"] - 600) < 1e-3    # 173600 - 173000
```

- [ ] **Step 2: Run failing test**

```bash
conda run -n autolens pytest tests/test_a1201_lens_model.py::test_compute_bayes_matrix_callable_with_stub_dirs -v
```
Expected: FAIL.

- [ ] **Step 3: Write `compute_bayes_matrix.py`**

```python
"""compute_bayes_matrix.py — ΔlnZ matrix across A1201 mass-model variants.

Mirrors N+23 Table 4 / §4.3 — for each mass-model class (PL, BPL, decomposed),
emit ΔlnZ = lnZ(+SMBH) − lnZ(−SMBH) plus cross-class ΔlnZ comparisons.

Usage:
    python compute_bayes_matrix.py \\
        --pl path/to/output/lp_3sersic    --pl-smbh path/to/output/with_smbh_3sersic \\
        --bpl path/to/output/bpl_3sersic  --bpl-smbh path/to/output/bpl_smbh_3sersic \\
        --decomp path/to/output/decomp_3sersic --decomp-smbh path/to/output/decomp_smbh_3sersic \\
        [--bpl-free path/to/output/bpl_free_centre]
        [--bpl-smbh-free path/to/output/bpl_smbh_free_centre]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Optional


def _unwrap_instance(node):
    while isinstance(node, dict) and node.get("type") == "instance" and "arguments" in node:
        node = node["arguments"]
    return node


def _load_log_evidence(d: Path) -> Optional[float]:
    if d is None or not d.exists():
        return None
    hits = sorted(d.rglob("samples_summary.json"))
    if not hits:
        return None
    raw = json.loads(hits[0].read_text())
    args = _unwrap_instance(raw)
    return float(args.get("log_evidence", 0.0)) if isinstance(args, dict) else None


def build_matrix(paths: Dict[str, Path]) -> Dict[str, Dict[str, float]]:
    """Compute per-class +SMBH vs -SMBH ΔlnZ, and cross-class comparisons.

    paths keys: pl, pl_smbh, bpl, bpl_smbh, decomp, decomp_smbh,
                bpl_free, bpl_smbh_free (the latter two optional).
    """
    lz = {k: _load_log_evidence(p) for k, p in paths.items()}

    out: Dict[str, Dict[str, float]] = {}
    for cls in ("pl", "bpl", "decomp"):
        smbh_key = f"{cls}_smbh"
        if lz.get(cls) is None or lz.get(smbh_key) is None:
            continue
        out[cls] = {
            "lnZ_no_smbh":   lz[cls],
            "lnZ_with_smbh": lz[smbh_key],
            "smbh_vs_no":    lz[smbh_key] - lz[cls],
        }

    # §4.3 / Appendix D: tied-centre BPL vs free-centre BPL coaxiality argument
    if lz.get("bpl_smbh") is not None and lz.get("bpl_smbh_free") is not None:
        out["bpl_smbh_centre_coaxiality"] = {
            "lnZ_tied":  lz["bpl_smbh"],
            "lnZ_free":  lz["bpl_smbh_free"],
            "free_minus_tied": lz["bpl_smbh_free"] - lz["bpl_smbh"],
        }
    if lz.get("bpl") is not None and lz.get("bpl_free") is not None:
        out["bpl_centre_coaxiality"] = {
            "lnZ_tied":  lz["bpl"],
            "lnZ_free":  lz["bpl_free"],
            "free_minus_tied": lz["bpl_free"] - lz["bpl"],
        }
    return out


def _format(matrix: Dict[str, Dict[str, float]]) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append("A1201 Mass-Model Variant ΔlnZ Matrix")
    lines.append("=" * 70)
    lines.append(f"{'Class':<10s}  {'lnZ(-SMBH)':>12s}  {'lnZ(+SMBH)':>12s}  {'ΔlnZ':>10s}  {'σ-equiv':>8s}  Verdict")
    lines.append("-" * 70)
    for cls in ("pl", "bpl", "decomp"):
        if cls not in matrix:
            lines.append(f"{cls:<10s}  {'(pending)':>12s}")
            continue
        m = matrix[cls]
        d = m["smbh_vs_no"]
        sigma = math.sqrt(2 * abs(d)) * (1 if d >= 0 else -1)
        verdict = ("DECISIVE detection" if d >= 100 else
                   ">3σ detection"      if d >= 4.5 else
                   "marginal"           if d >= 2.0 else
                   "inconclusive"       if d >= -2 else
                   "PREFERS no-SMBH")
        lines.append(f"{cls:<10s}  {m['lnZ_no_smbh']:12.2f}  {m['lnZ_with_smbh']:12.2f}  {d:+10.2f}  {sigma:+8.2f}  {verdict}")
    if "bpl_centre_coaxiality" in matrix:
        lines.append("")
        lines.append("§4.3 / Appendix D — BPL centre coaxiality test (paired ΔlnZ at fixed model spec):")
        m = matrix["bpl_centre_coaxiality"]
        lines.append(f"  BPL (no SMBH):    tied-centre lnZ = {m['lnZ_tied']:.2f}   free-centre lnZ = {m['lnZ_free']:.2f}   free-tied = {m['free_minus_tied']:+.2f}")
        m = matrix.get("bpl_smbh_centre_coaxiality", {})
        if m:
            lines.append(f"  BPL (+ SMBH):     tied-centre lnZ = {m['lnZ_tied']:.2f}   free-centre lnZ = {m['lnZ_free']:.2f}   free-tied = {m['free_minus_tied']:+.2f}")
        lines.append("")
        lines.append("N+23 §4.3 prediction: tied-centre BPL−SMBH << +SMBH (BPL alone can't match SMBH).")
        lines.append("                       Free-centre BPL can match +SMBH but at ≥100 pc mass-light offset.")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    for key in ("pl", "pl-smbh", "bpl", "bpl-smbh", "decomp", "decomp-smbh",
                "bpl-free", "bpl-smbh-free"):
        p.add_argument(f"--{key}", type=Path, default=None,
                       help=f"Output directory for the {key} variant.")
    args = p.parse_args()

    paths = {
        "pl":             args.pl,
        "pl_smbh":        getattr(args, "pl_smbh"),
        "bpl":            args.bpl,
        "bpl_smbh":       getattr(args, "bpl_smbh"),
        "decomp":         args.decomp,
        "decomp_smbh":    getattr(args, "decomp_smbh"),
        "bpl_free":       getattr(args, "bpl_free"),
        "bpl_smbh_free":  getattr(args, "bpl_smbh_free"),
    }
    matrix = build_matrix(paths)
    print(_format(matrix))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the test**

```bash
conda run -n autolens pytest tests/test_a1201_lens_model.py::test_compute_bayes_matrix_callable_with_stub_dirs -v
```
Expected: PASS.

---

## Task 8: §4.3 reproduction — analyse BPL centre coaxiality

After the 4 BPL jobs + 2 decomposed jobs land (rough wall: 24-48h on Cannon),
pull and analyse.

- [ ] **Step 1: Pull all six variant outputs**

```bash
for variant in bpl_3sersic bpl_smbh_3sersic bpl_free_centre bpl_smbh_free_centre decomp_3sersic decomp_smbh_3sersic; do
    rsync -av --progress \
        cannon:/n/holystore01/LABS/hernquist_lab/Lab/rcordova/learning_to_autolens/private/2303_15514_nightingale2023_abell1201/output/${variant}/ \
        private/2303_15514_nightingale2023_abell1201/output/${variant}/
done
```

- [ ] **Step 2: Run the full Bayes-matrix**

```bash
python private/2303_15514_nightingale2023_abell1201/code/compute_bayes_matrix.py \
    --pl=private/2303_15514_nightingale2023_abell1201/output/lp_3sersic \
    --pl-smbh=private/2303_15514_nightingale2023_abell1201/output/with_smbh_3sersic \
    --bpl=private/2303_15514_nightingale2023_abell1201/output/bpl_3sersic \
    --bpl-smbh=private/2303_15514_nightingale2023_abell1201/output/bpl_smbh_3sersic \
    --bpl-free=private/2303_15514_nightingale2023_abell1201/output/bpl_free_centre \
    --bpl-smbh-free=private/2303_15514_nightingale2023_abell1201/output/bpl_smbh_free_centre \
    --decomp=private/2303_15514_nightingale2023_abell1201/output/decomp_3sersic \
    --decomp-smbh=private/2303_15514_nightingale2023_abell1201/output/decomp_smbh_3sersic \
    > private/2303_15514_nightingale2023_abell1201/results/bayes_matrix_3sersic.txt

cat private/2303_15514_nightingale2023_abell1201/results/bayes_matrix_3sersic.txt
```

- [ ] **Step 3: Audit the §4.3 coaxiality prediction**

Verification criteria (from N+23 §4.3 + Appendix D):
- **Strict-pass**: tied-centre BPL +SMBH ΔlnZ > +4.5 (3σ detection); tied-centre BPL −SMBH ΔlnZ < +1 (no detection alone).
- **§4.3 reproduction**: free-centre BPL −SMBH should approach (within ΔlnZ ≤ 2) the tied-centre BPL +SMBH evidence, BUT the free-centre BPL mass-centre posterior should land ≥100 pc offset from bulge.centre (≥ 0.04″ at z=0.169 — verify by computing the posterior centroid separation).

- [ ] **Step 4: Extract the BPL free-centre mass-light separation**

```python
python -c "
import sys, json
sys.path.insert(0, 'private/2303_15514_nightingale2023_abell1201/code')
from audit_stage1 import _unwrap_instance, _extract_param_dict
import numpy as np
from astropy.cosmology import FlatLambdaCDM

path = list((Path('private/2303_15514_nightingale2023_abell1201/output/bpl_smbh_free_centre').rglob('samples_summary.json')))[0]
args = _unwrap_instance(json.loads(path.read_text()))
med = _extract_param_dict(args['median_pdf_sample'])

mass_c0 = med.get('galaxies.lens.mass.centre.centre_0')
mass_c1 = med.get('galaxies.lens.mass.centre.centre_1')
bulge_c0 = med.get('galaxies.lens.bulge.centre.centre_0')
bulge_c1 = med.get('galaxies.lens.bulge.centre.centre_1')
sep_arcsec = ((mass_c0 - bulge_c0)**2 + (mass_c1 - bulge_c1)**2)**0.5

# Convert arcsec to kpc at z_l=0.169
cosmo = FlatLambdaCDM(H0=70.0, Om0=0.30)
D_l_kpc = cosmo.angular_diameter_distance(0.169).to('kpc').value
sep_kpc = sep_arcsec * np.pi / 180 / 3600 * D_l_kpc

print(f'BPL +SMBH free-centre mass-light separation:')
print(f'  arcsec: {sep_arcsec:.4f}\"')
print(f'  kpc:    {sep_kpc:.3f} kpc')
print(f'N+23 §4.3 threshold: ≥0.1 kpc (100 pc) for the unphysical mode')
print(f'  → reproduces §4.3?  {\"YES\" if sep_kpc >= 0.1 else \"NO\"}')
"
```

Expected: prints the separation. If ≥100 pc, reproduces N+23 §4.3 argument.

---

## Task 9: Mass-model matrix headline notebook

**Files:**
- Create: `private/2303_15514_nightingale2023_abell1201/notebooks/04_a1201_mass_model_matrix.ipynb`

Build via a separate `_build_04_notebook.py` script (mirroring the pattern
used for `03_a1201_mbh_recovery.ipynb`).

- [ ] **Step 1: Write the builder script**

```python
# private/2303_15514_nightingale2023_abell1201/notebooks/_build_04_notebook.py
from pathlib import Path
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell(
    "# A1201 — Mass-Model Variant ΔlnZ Matrix (N+23 §3.8 / §4.3 / Appendix D)\n\n"
    "Headline notebook for the **paper-faithful M_BH detection cross-check**: "
    "Bayes-factor comparison of M_BH detection across PL, BPL (O'Riordan), and "
    "decomposed (Sersic+NFW+shear) mass-model classes. Plus the §4.3 / "
    "Appendix D coaxiality test (tied vs free BPL centre).\n\n"
    "**Reproduction targets** (Verification note: anchored to PAPER_NOTES §Table 4 + §4.3 + Appendix D):\n"
    "- PL ± SMBH: F390W ΔlnZ = +100.58 (N+23 Table 4, 3-Sersic light)\n"
    "- BPL alone (tied centre, no SMBH) — ΔlnZ << SMBH case\n"
    "- BPL alone (free centre, no SMBH) — should approach SMBH case BUT at ≥100 pc offset\n"
    "- Decomposed ± SMBH (no separate ΔlnZ quoted, but consistent with PL detection per §3.8)"
))

cells.append(nbf.v4.new_code_cell(
    "# Cell 1 — imports + Bayes-matrix computation\n"
    "from pathlib import Path\n"
    "import sys\n"
    "PROJECT_ROOT = Path('..').resolve()\n"
    "sys.path.insert(0, str(PROJECT_ROOT / 'code'))\n"
    "from compute_bayes_matrix import build_matrix, _format\n\n"
    "paths = {\n"
    "    'pl':             PROJECT_ROOT / 'output' / 'lp_3sersic',\n"
    "    'pl_smbh':        PROJECT_ROOT / 'output' / 'with_smbh_3sersic',\n"
    "    'bpl':            PROJECT_ROOT / 'output' / 'bpl_3sersic',\n"
    "    'bpl_smbh':       PROJECT_ROOT / 'output' / 'bpl_smbh_3sersic',\n"
    "    'decomp':         PROJECT_ROOT / 'output' / 'decomp_3sersic',\n"
    "    'decomp_smbh':    PROJECT_ROOT / 'output' / 'decomp_smbh_3sersic',\n"
    "    'bpl_free':       PROJECT_ROOT / 'output' / 'bpl_free_centre',\n"
    "    'bpl_smbh_free':  PROJECT_ROOT / 'output' / 'bpl_smbh_free_centre',\n"
    "}\n"
    "matrix = build_matrix(paths)\n"
    "print(_format(matrix))"
))

cells.append(nbf.v4.new_markdown_cell(
    "## §1 Cross-class consistency check\n\n"
    "If we reproduce N+23, ALL THREE mass-model classes (PL, BPL, decomposed) "
    "should show a +SMBH preference (ΔlnZ > +4.5 = 3σ) when their mass centres "
    "are tied to the BCG light centre. The TIED-CENTRE BPL preference for SMBH "
    "is the key result: even with the BPL's extra flexibility, the data "
    "demands a point-mass component when mass and light are physically aligned.\n\n"
    "The FREE-CENTRE BPL test (§4.3 / Appendix D) checks whether the BPL can "
    "EVADE the SMBH detection by allowing the mass centre to drift. N+23 finds "
    "it CAN, but only at unphysical ≥100 pc offsets — which we reproduce below."
))

cells.append(nbf.v4.new_code_cell(
    "# Cell 2 — §4.3 coaxiality reproduction: mass-light centre separation\n"
    "import json\n"
    "import numpy as np\n"
    "from astropy.cosmology import FlatLambdaCDM\n"
    "from audit_stage1 import _unwrap_instance, _extract_param_dict\n\n"
    "cosmo = FlatLambdaCDM(H0=70.0, Om0=0.30)\n"
    "D_l_kpc = float(cosmo.angular_diameter_distance(0.169).to('kpc').value)\n\n"
    "for label, p in [('BPL +SMBH (free centre)', paths['bpl_smbh_free']),\n"
    "                 ('BPL alone (free centre)', paths['bpl_free'])]:\n"
    "    hits = list((p).rglob('samples_summary.json'))\n"
    "    if not hits:\n"
    "        print(f'{label}: (pending)')\n"
    "        continue\n"
    "    args = _unwrap_instance(json.loads(hits[0].read_text()))\n"
    "    med = _extract_param_dict(args['median_pdf_sample'])\n"
    "    mc0 = med.get('galaxies.lens.mass.centre.centre_0', 0.0)\n"
    "    mc1 = med.get('galaxies.lens.mass.centre.centre_1', 0.0)\n"
    "    bc0 = med.get('galaxies.lens.bulge.centre.centre_0', 0.0)\n"
    "    bc1 = med.get('galaxies.lens.bulge.centre.centre_1', 0.0)\n"
    "    sep_as = ((mc0-bc0)**2 + (mc1-bc1)**2)**0.5\n"
    "    sep_kpc = sep_as * np.pi/180/3600 * D_l_kpc\n"
    "    print(f'{label}: mass-light sep = {sep_as:.4f}\" = {sep_kpc*1000:.1f} pc')\n"
    "print()\n"
    "print('N+23 §4.3 prediction: ≥100 pc for the unphysical free-centre mode')"
))

cells.append(nbf.v4.new_markdown_cell(
    "## §2 Methodology divergences vs N+23\n\n"
    "| Aspect | N+23 (per ar5iv verified) | Us | Risk |\n"
    "|---|---|---|---|\n"
    "| Light model | 3-Sersic (Table 1 selected) | matches (--n-light=3) | none |\n"
    "| PL mass | al.mp.PowerLaw + ExternalShear | matches | none |\n"
    "| BPL mass | O'Riordan param (Eq from O'Riordan 2019-21) | al.mp.PowerLawBroken — verify name in Task 0 | may differ if autolens class deviates |\n"
    "| Stellar mass (decomposed) | sum of Sersic with M/L gradient (Ψ + Γ per component) | single Sersic stellar mass + constant Ψ | Γ-gradient deferred to v0.99 — bias unclear |\n"
    "| DM halo | standard elliptical NFW | matches al.mp.NFW | none |\n"
    "| Source | pixelised AdaptiveBrightness (paper's headline) | parametric Sersic (matches our Stages 1-3); Stage 4 adapt available | LIKELY underconstrains M_BH magnitude vs N+23 |\n"
    "| Sampler | Dynesty | Nautilus (autolens 2026.4 default) | converged posteriors should match |\n"
    "| Cosmology | not stated — probably Planck18 | FlatLambdaCDM(70, 0.3) | <2% effect on derived M_BH (M_sun) |"
))

cells.append(nbf.v4.new_markdown_cell(
    "## §3 Citation register\n\n"
    "**Anchored** (verified via ar5iv 2026-05-20):\n"
    "- BPL parameterization from O'Riordan, Warren & Mortlock 2019/2020/2021 — N+23 §3.4\n"
    "- §4.3 / Appendix D BPL centre coaxiality argument: mass-centre output ≥100 pc offset = unphysical\n"
    "- N+23 §3.8 SLaM Mass Pipeline runs PL, BPL, decomposed each ± SMBH\n"
    "- Table 4 = light-model ladder for PL ± SMBH; F390W 3-Sersic + SMBH ΔlnZ = +100.58\n\n"
    "**Aspirational** (not yet verified):\n"
    "- Exact BPL functional form (need O'Riordan 2019 §X / Eq Y direct check)\n"
    "- Decomposed-model M/L gradient parameterization (Ψ vs Γ) per component\n"
    "- BPL detection ΔlnZ for tied-centre case (Table 4 doesn't list — need Appendix D)\n"
    "- Decomposed-model SMBH-detection ΔlnZ"
))

nb['cells'] = cells

out = Path('private/2303_15514_nightingale2023_abell1201/notebooks/04_a1201_mass_model_matrix.ipynb')
import nbformat as nbf2
nbf2.write(nb, str(out))
print(f'Wrote {out} ({len(cells)} cells)')
```

- [ ] **Step 2: Run the builder**

```bash
conda run -n autolens python private/2303_15514_nightingale2023_abell1201/notebooks/_build_04_notebook.py
```

Expected: prints "Wrote ... (6 cells)".

---

## Task 10: Update PROGRESS_LOG + commit + memory

- [ ] **Step 1: Append to PROGRESS_LOG.md**

Add a new section under the 2026-05-18/19 paper-repro-program section:

```markdown
### 2026-05-20 — A1201 mass-model variant tests submitted

Reproducing N+23's §3.8 Mass Pipeline alternatives + the §4.3 / Appendix D
BPL coaxiality test. Six new Cannon submissions chained from v4 (the
3-Sersic Stage 1 baseline that landed lnZ = +174,904 at 6h09m wall):

| Variant | Chained from v4 | Note |
|---|---|---|
| `bpl` (tied centre, no SMBH) | yes | N+23 alternative test 1 |
| `bpl_smbh` (tied centre, + SMBH) | yes | N+23 alternative test 2 — should land ΔlnZ vs `bpl` |
| `bpl` (free centre, no SMBH) | yes | §4.3 reproduction control |
| `bpl_smbh` (free centre, + SMBH) | yes | §4.3 reproduction target — should show ≥100 pc offset |
| `decomp` (no SMBH) | yes | Sersic+NFW+shear, no BH |
| `decomp_smbh` (+ SMBH) | yes | Decomposed + BH — N+23's physically-motivated alternative |

Implementation: `_build_lens_galaxy_bpl`, `build_bpl_fit`,
`build_bpl_smbh_fit`, `_build_lens_galaxy_decomposed`, `build_decomposed_fit`,
`build_decomposed_smbh_fit` added to `a1201_lens_model.py`. New `--part=bpl`,
`bpl_smbh`, `decomp`, `decomp_smbh` and `--free-mass-centre` flag.
`compute_bayes_matrix.py` aggregates per-class ΔlnZ + §4.3 coaxiality test.
```

- [ ] **Step 2: Commit (no private/ — that's gitignored; only the plan + PROGRESS_LOG)**

```bash
git add docs/superpowers/plans/2026-05-20-paper-repro-05-a1201-mass-model-variants.md PROGRESS_LOG.md
git commit -m "$(cat <<'EOF'
plan: A1201 mass-model variants (Spec 05 Phase 4.5)

Reproduce N+23's Mass Pipeline alternative tests (§3.8): PL, BPL, decomposed
each ± SMBH, plus the §4.3 / Appendix D BPL-centre coaxiality test.

Driver additions (in private/, gitignored):
- _build_lens_galaxy_bpl + build_bpl_fit / build_bpl_smbh_fit (with
  free_mass_centre flag for §4.3 reproduction)
- _build_lens_galaxy_decomposed + build_decomposed_fit / build_decomposed_smbh_fit
- compute_bayes_matrix.py: ΔlnZ matrix across all 6 variants + coaxiality
- _build_04_notebook.py: headline notebook scaffold

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 3: Update memory**

Create `~/.claude/projects/.../memory/project_a1201_mass_model_variants.md`:

```markdown
---
name: project-a1201-mass-model-variants
description: Spec 05 Phase 4.5 — reproducing N+23's Mass Pipeline alternatives (PL/BPL/decomposed × ±SMBH) + §4.3 BPL coaxiality test
metadata:
  type: project
---

## Plan
docs/superpowers/plans/2026-05-20-paper-repro-05-a1201-mass-model-variants.md

## Six Cannon variants (all chained from v4 3-Sersic Stage 1)
- bpl_3sersic, bpl_smbh_3sersic — tied-centre BPL ± SMBH
- bpl_free_centre, bpl_smbh_free_centre — free-centre BPL ± SMBH (§4.3 reproduction)
- decomp_3sersic, decomp_smbh_3sersic — decomposed Sersic+NFW+shear ± SMBH

## Verification criteria
- All TIED-CENTRE classes should prefer +SMBH at >3σ (ΔlnZ > +4.5)
- FREE-CENTRE BPL should approach +SMBH case BUT at ≥100 pc offset (N+23 §4.3 / App D)

## Implementation status
[ ] Task 0 verify autolens 2026.4 class names
[ ] Task 1-4 implement build funcs + tests
[ ] Task 5-6 push + submit 6 jobs
[ ] Task 7 compute_bayes_matrix.py
[ ] Task 8 §4.3 reproduction analysis
[ ] Task 9 headline notebook
[ ] Task 10 PROGRESS_LOG + commit + memory

Related: [[project-a1201-stage1-v2v3-broken]] (parametric source bottleneck),
[[project-a1201-paper-repro-state]] (overall Spec 05 status).
```

Then add to MEMORY.md:

```markdown
- [A1201 mass-model variants plan](project_a1201_mass_model_variants.md) — Spec 05 Phase 4.5: reproduce N+23 PL/BPL/decomp ± SMBH + §4.3 BPL coaxiality
```

---

## Verification (full plan ship)

End-to-end strict-PASS requires:

1. All 6 Cannon variants COMPLETED with positive logZ (no rails, no posterior pathology).
2. **PL +SMBH vs PL −SMBH ΔlnZ > +4.5** (3σ — N+23's detection threshold).
3. **BPL tied-centre +SMBH vs −SMBH ΔlnZ > +4.5** (the key cross-class consistency check).
4. **Decomposed +SMBH vs −SMBH ΔlnZ > +4.5** (validates against physically-motivated alternative).
5. **BPL free-centre −SMBH lnZ ≈ tied-centre +SMBH lnZ** (within ΔlnZ ≤ 2), AND **mass-light centre separation ≥100 pc** (the §4.3 / Appendix D reproduction).
6. `compute_bayes_matrix.py` output captured in `results/bayes_matrix_3sersic.txt`.
7. `04_a1201_mass_model_matrix.ipynb` executed cleanly.

**Soft-PASS variants**:
- PL/BPL/decomposed detections all positive but ΔlnZ < +4.5 — likely indicates F814W-only is too marginal; suggest re-running on F390W.
- §4.3 free-centre offset < 100 pc — could indicate our BPL parameterization differs from N+23's, OR our cutout/PSF systematics differ.

**FAIL**: any tied-centre +SMBH variant landing ΔlnZ ≤ 0 — that would contradict N+23's central result and would require deeper investigation (potentially indicates our reduction differs substantively).

---

## Out-of-scope

- F390W-band reproduction of these mass-model variants — defer to v0.98 once F390W cutout is wired to driver (Spec 05 plan §Phase 5).
- Pixelised AdaptiveBrightness source variant of the BPL/decomposed tests — that's Spec 05 Phase 4 (`build_adapt_fit` already shipped) + this plan combined; defer to v0.99 after parametric variant lands.
- Full M/L radial gradient (Γ) for decomposed model components — `al.lmp.SersicGradient` exists but requires careful prior-choice; v0.99.
- Independent reproduction of N+23 §4.3 Appendix D Figure (if present) showing the free-centre BPL posterior — would require constructing the equivalent figure with our results.

---

## Sequencing

- Week 1 (after this plan): Tasks 0-4 (driver implementation + tests, ~2-4 hours laptop)
- Week 1: Task 5-6 (push + submit; 6 Cannon jobs queued, ~24-48h wall total assuming concurrent execution)
- Week 2: Tasks 7-9 (matrix + §4.3 reproduction + notebook; ~half-day laptop)
- Week 2: Task 10 (PROGRESS_LOG + commit + memory; ~30 min)

Total: ~1-2 weeks elapsed, ~2-3 Cannon-days of compute (siag_lab; depends on queue priority — currently sharing siag with a1201_with_smbh_3s already queued).
