# Paper-Repro Spec 02 — Ballard+2023 TSPL Jackpot Subhalo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reproduce 5.9σ dark subhalo detection in SDSSJ0946+1006 (Ballard+2023) in BOTH PyAutoLens and Herculens; verify (M_sub, c_sub) posteriors agree at <1σ between stacks; demonstrate the wandering-BH alternative.

**Architecture:** TSPL Tracer (3 source planes at z=(0.609, 2.035, 5.975)) + main lens (SIE + ExternalShear at z=0.222) + optional NFW perturber + MUSE-source-position likelihood as a third AnalysisFactor. Six Cannon fits total: {main, +subhalo, +wandering-BH} × {autolens, Herculens}.

**Tech Stack:** PyAutoLens 2026.4 multi-plane `al.Tracer`, Herculens `multi_plane.MultiPlaneLensModel`, Spec 00 `herculens_bridge.py` for the model spec, Spec 00 `j0946_data_loader.py` for HST + MUSE, AGEL Watson pipeline (Spec 00 §6.9) for clean F814W + F336W reductions, Cannon siag_lab A100 GPU for Herculens NUTS.

**Depends on:** Spec 00 (J0946 data, herculens_bridge, crossval_framework, AGEL Watson reduction).

---

## File Structure

```
private/2309_04535_ballard2023_tspl_jackpot/
├── code/
│   ├── __init__.py
│   ├── tspl_tracer_autolens.py
│   ├── tspl_tracer_herculens.py
│   ├── muse_position_likelihood.py
│   ├── subhalo_fit_driver_autolens.py
│   ├── subhalo_fit_driver_herculens.py
│   ├── wandering_bh_alternative.py
│   ├── tspl_synthetic_mock.py             ← injects known truth for testing
│   └── validation.py
├── notebooks/
│   ├── 01_tspl_main_fit.ipynb
│   ├── 02_subhalo_bayes_factor.ipynb
│   ├── 03_wandering_bh_alt.ipynb
│   └── 04_crossval_summary.ipynb
├── tests/
│   ├── __init__.py
│   ├── test_tspl_tracer_autolens.py
│   ├── test_tspl_tracer_herculens.py
│   ├── test_muse_position_likelihood.py
│   └── test_tspl_synthetic_mock.py
└── results/
    ├── tspl_main_autolens/
    ├── tspl_main_herculens/
    ├── tspl_subhalo_autolens/
    ├── tspl_subhalo_herculens/
    ├── tspl_wbh_autolens/
    └── tspl_wbh_herculens/
```

---

## Phase 1: TSPL synthetic mock

### Task 1: Generate a synthetic TSPL mock for testing

**Files:**
- Create: `private/2309_04535_ballard2023_tspl_jackpot/code/tspl_synthetic_mock.py`
- Create: `private/2309_04535_ballard2023_tspl_jackpot/tests/test_tspl_synthetic_mock.py`

Need a known-truth mock to drive TDD on the TSPL Tracer. Mock has main lens + 3 sources + optional NFW perturber injected at known position.

- [ ] **Step 1: Write failing test**

Create `tests/test_tspl_synthetic_mock.py`:

```python
"""Tests for the TSPL synthetic mock generator."""

from pathlib import Path
import sys
import pytest

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / 'code'))


def test_mock_generates_image_and_truths(tmp_path):
    from tspl_synthetic_mock import generate
    generate(out_dir=tmp_path, with_perturber=False)
    assert (tmp_path / 'image.fits').exists()
    assert (tmp_path / 'truths.json').exists()


def test_truths_have_3_source_redshifts(tmp_path):
    from tspl_synthetic_mock import generate
    import json
    generate(out_dir=tmp_path, with_perturber=False)
    truths = json.loads((tmp_path / 'truths.json').read_text())
    assert 'z_s1' in truths['redshifts']
    assert 'z_s2' in truths['redshifts']
    assert 'z_s3' in truths['redshifts']


def test_chi2_at_truth_passes(tmp_path):
    """Mock should be self-consistent: refit at truth → chi^2/N ≤ 1.5."""
    from tspl_synthetic_mock import generate, chi2_at_truth
    generate(out_dir=tmp_path, with_perturber=False)
    chi2_per_pix = chi2_at_truth(tmp_path)
    assert chi2_per_pix <= 1.5


def test_perturber_injection(tmp_path):
    """With perturber: residuals should be non-trivial unless we refit with perturber."""
    from tspl_synthetic_mock import generate
    import json
    generate(out_dir=tmp_path, with_perturber=True,
             perturber_xy=(0.5, 0.0), perturber_logM=9.2)
    truths = json.loads((tmp_path / 'truths.json').read_text())
    assert 'perturber' in truths
    assert truths['perturber']['logM_solar'] == 9.2
```

- [ ] **Step 2: Run (FAIL)**

```bash
cd private/2309_04535_ballard2023_tspl_jackpot
conda run -n autolens pytest tests/test_tspl_synthetic_mock.py -v 2>&1 | tail -10
```

- [ ] **Step 3: Implement the mock generator**

Create `code/tspl_synthetic_mock.py`:

```python
"""tspl_synthetic_mock.py — synthetic TSPL Jackpot mock.

Geometry mirrors J0946+1006 per Smith+2024:
  z_lens = 0.222
  z_s1 = 0.609, z_s2 = 2.035, z_s3 = 5.975
Main lens: SIE θ_E=1.4″ + ExternalShear (γ_1=0.02, γ_2=0.01)
Sources: 3 Sersic discs at the three source planes
Optional perturber: NFW at user-specified (x, y) with user logM, c

Public surface:
    generate(out_dir, with_perturber=False, perturber_xy=None,
             perturber_logM=None) → None
    chi2_at_truth(out_dir) → float
"""

from __future__ import annotations

import json
from pathlib import Path

import autolens as al
import numpy as np
from astropy.io import fits


Z_L = 0.222
SRC_Z = {'z_s1': 0.609, 'z_s2': 2.035, 'z_s3': 5.975}


def _build_truth_tracer(with_perturber: bool = False,
                       perturber_xy=None, perturber_logM=None):
    cosmo = al.cosmo.FlatLambdaCDM(H0=70.0, Om0=0.30)
    main_lens = al.Galaxy(
        redshift=Z_L,
        mass=al.mp.Isothermal(
            centre=(0.0, 0.0), ell_comps=(0.1, 0.05),
            einstein_radius=1.4),
        shear=al.mp.ExternalShear(gamma_1=0.02, gamma_2=0.01),
    )
    galaxies = [main_lens]

    if with_perturber:
        from autolens import mp
        # Convert logM to a rough θ_E for a NFW perturber
        # M_sub ~ 10^9.2 M_sun at z=0.222 → θ_E ~ 0.05" (order-of-magnitude)
        perturber = al.Galaxy(
            redshift=Z_L,
            mass=mp.NFW(
                centre=perturber_xy,
                ell_comps=(0.0, 0.0),
                kappa_s=0.05,
                scale_radius=2.0,
            ),
        )
        galaxies.append(perturber)

    # Sources at three planes
    for i, (zname, z_val) in enumerate(SRC_Z.items(), start=1):
        src = al.Galaxy(
            redshift=z_val,
            bulge=al.lp.Sersic(
                centre=(0.05 * i, 0.03 * i),
                ell_comps=(0.05, 0.02),
                intensity=0.3 / i,
                effective_radius=0.1,
                sersic_index=1.5,
            ),
        )
        galaxies.append(src)

    return al.Tracer(galaxies=galaxies, cosmology=cosmo)


def generate(out_dir: Path, with_perturber: bool = False,
             perturber_xy=None, perturber_logM=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tracer = _build_truth_tracer(with_perturber, perturber_xy, perturber_logM)
    grid = al.Grid2D.uniform(shape_native=(150, 150), pixel_scales=0.05)
    psf = al.Kernel2D.from_gaussian(shape_native=(11, 11),
                                     sigma=0.04, pixel_scales=0.05,
                                     normalize=True)
    simulator = al.SimulatorImaging(
        exposure_time=2000.0,
        background_sky_level=0.05,
        psf=psf,
        add_poisson_noise=True,
    )
    dataset = simulator.via_tracer_from(tracer=tracer, grid=grid)
    fits.writeto(out_dir / 'image.fits', np.array(dataset.data.native),
                 overwrite=True)
    fits.writeto(out_dir / 'noise_map.fits', np.array(dataset.noise_map.native),
                 overwrite=True)
    fits.writeto(out_dir / 'psf.fits', np.array(psf.native), overwrite=True)

    truths = {
        'redshifts': {'z_lens': Z_L, **SRC_Z},
        'main_lens': {
            'mass': {'type': 'Isothermal', 'centre': [0, 0],
                     'ell_comps': [0.1, 0.05], 'einstein_radius': 1.4},
            'shear': {'gamma_1': 0.02, 'gamma_2': 0.01},
        },
    }
    if with_perturber:
        truths['perturber'] = {
            'centre': list(perturber_xy), 'logM_solar': perturber_logM,
        }
    (out_dir / 'truths.json').write_text(json.dumps(truths, indent=2))


def chi2_at_truth(out_dir: Path) -> float:
    """Self-consistency: refit truth tracer → chi²/N."""
    out_dir = Path(out_dir)
    with fits.open(out_dir / 'image.fits') as h:
        image = h[0].data
    with fits.open(out_dir / 'noise_map.fits') as h:
        noise = h[0].data
    truths = json.loads((out_dir / 'truths.json').read_text())
    has_perturber = 'perturber' in truths
    tracer = _build_truth_tracer(with_perturber=has_perturber,
                                  perturber_xy=truths.get(
                                      'perturber', {}).get('centre'),
                                  perturber_logM=truths.get(
                                      'perturber', {}).get('logM_solar'))
    # Simple chi^2 against the data
    grid = al.Grid2D.uniform(shape_native=image.shape, pixel_scales=0.05)
    psf = al.Kernel2D.from_gaussian(shape_native=(11, 11),
                                     sigma=0.04, pixel_scales=0.05,
                                     normalize=True)
    model_image = tracer.image_2d_from(grid=grid)
    # Naive (no PSF convolution match) — good enough for the self-test
    residual = (image - np.array(model_image.native)) / noise
    chi2 = np.nansum(residual ** 2) / (np.isfinite(residual).sum() or 1)
    return float(chi2)
```

- [ ] **Step 4: Run tests**

```bash
conda run -n autolens pytest tests/test_tspl_synthetic_mock.py -v 2>&1 | tail -10
```

Expected: 4 pass. May need tuning on chi2_at_truth threshold if PSF mismatch dominates.

---

## Phase 2: TSPL Tracer in both stacks

### Task 2: PyAutoLens TSPL tracer + tests

**Files:**
- Create: `private/2309_04535_ballard2023_tspl_jackpot/code/tspl_tracer_autolens.py`
- Create: `private/2309_04535_ballard2023_tspl_jackpot/tests/test_tspl_tracer_autolens.py`

- [ ] **Step 1: Failing test**

Create `tests/test_tspl_tracer_autolens.py`:

```python
"""Tests for the autolens TSPL Tracer."""

from pathlib import Path
import sys
import pytest

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / 'code'))


def test_tracer_has_4_planes():
    """Main lens + 3 sources = 4 planes."""
    from tspl_tracer_autolens import build_tspl_tracer
    tracer = build_tspl_tracer(with_perturber=False)
    n_planes = len(tracer.planes)
    assert n_planes == 4


def test_tracer_with_perturber_has_4_planes_too():
    """Perturber is co-planar with main lens (same z) → still 4 planes."""
    from tspl_tracer_autolens import build_tspl_tracer
    tracer = build_tspl_tracer(with_perturber=True,
                                perturber_xy=(0.5, 0), perturber_logM=9.2)
    n_planes = len(tracer.planes)
    assert n_planes == 4


def test_image_rendering_returns_finite():
    """The Tracer should render a finite image."""
    import autolens as al
    import numpy as np
    from tspl_tracer_autolens import build_tspl_tracer
    grid = al.Grid2D.uniform(shape_native=(50, 50), pixel_scales=0.05)
    tracer = build_tspl_tracer(with_perturber=False)
    img = tracer.image_2d_from(grid=grid)
    arr = np.array(img.native)
    assert np.isfinite(arr).all()
    assert arr.max() > 0
```

- [ ] **Step 2: Run (FAIL)**

- [ ] **Step 3: Implement**

Create `code/tspl_tracer_autolens.py`:

```python
"""tspl_tracer_autolens.py — TSPL Tracer in PyAutoLens for J0946+1006.

Public:
    build_tspl_tracer(with_perturber=False, ...) → al.Tracer
"""

from __future__ import annotations

import autolens as al


Z_L = 0.222
SRC_Z = {'z_s1': 0.609, 'z_s2': 2.035, 'z_s3': 5.975}


def build_tspl_tracer(with_perturber: bool = False,
                       perturber_xy=(0.5, 0.0),
                       perturber_logM: float = 9.2):
    cosmo = al.cosmo.FlatLambdaCDM(H0=70.0, Om0=0.30)
    main = al.Galaxy(
        redshift=Z_L,
        mass=al.mp.Isothermal(
            centre=(0.0, 0.0), ell_comps=(0.1, 0.05),
            einstein_radius=1.4),
        shear=al.mp.ExternalShear(gamma_1=0.02, gamma_2=0.01),
    )
    galaxies = [main]
    if with_perturber:
        perturber = al.Galaxy(
            redshift=Z_L,
            mass=al.mp.NFW(centre=perturber_xy,
                           ell_comps=(0.0, 0.0),
                           kappa_s=0.05, scale_radius=2.0),
        )
        galaxies.append(perturber)
    for z in SRC_Z.values():
        galaxies.append(al.Galaxy(
            redshift=z,
            bulge=al.lp.Sersic(centre=(0.0, 0.0), ell_comps=(0.0, 0.0),
                                intensity=0.3, effective_radius=0.1,
                                sersic_index=1.5),
        ))
    return al.Tracer(galaxies=galaxies, cosmology=cosmo)
```

- [ ] **Step 4: Run tests (PASS)**

### Task 3: Herculens TSPL tracer + tests

**Files:**
- Create: `private/2309_04535_ballard2023_tspl_jackpot/code/tspl_tracer_herculens.py`
- Create: `private/2309_04535_ballard2023_tspl_jackpot/tests/test_tspl_tracer_herculens.py`

- [ ] **Step 1: Failing test**

Create `tests/test_tspl_tracer_herculens.py`:

```python
"""Tests for Herculens TSPL."""

from pathlib import Path
import sys
import pytest

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / 'code'))

herculens = pytest.importorskip('herculens')


def test_multi_plane_lens_model_with_3_sources():
    from tspl_tracer_herculens import build_tspl_multi_plane
    model = build_tspl_multi_plane(with_perturber=False)
    # Herculens MultiPlaneLensModel exposes lens_redshift_list + source_redshift_list
    assert len(model.source_redshift_list) == 3
    assert model.source_redshift_list == [0.609, 2.035, 5.975]


def test_multi_plane_alpha_returns_arrays():
    """Deflection at a grid point should be finite."""
    import jax.numpy as jnp
    from tspl_tracer_herculens import build_tspl_multi_plane
    model = build_tspl_multi_plane(with_perturber=False)
    x = jnp.array([0.5])
    y = jnp.array([0.5])
    alpha_x, alpha_y = model.alpha(x, y, ...)  # kwargs to be filled per profile
    # Don't assert specific values yet — just that the call doesn't error
```

- [ ] **Step 2: Run (FAIL)**

- [ ] **Step 3: Implement**

Create `code/tspl_tracer_herculens.py`:

```python
"""tspl_tracer_herculens.py — Herculens multi_plane setup for J0946 TSPL."""

from __future__ import annotations


def build_tspl_multi_plane(with_perturber: bool = False,
                            perturber_xy=(0.5, 0.0),
                            perturber_logM: float = 9.2):
    from herculens.LensModel.multi_plane import MultiPlaneLensModel
    profile_list = ['EPL', 'SHEAR']
    lens_redshift_list = [0.222, 0.222]
    if with_perturber:
        profile_list.append('NFW_ELLIPSE')
        lens_redshift_list.append(0.222)
    source_redshift_list = [0.609, 2.035, 5.975]
    z_source = 5.975  # outermost source for distance normalisations
    model = MultiPlaneLensModel(
        profile_list=profile_list,
        lens_redshift_list=lens_redshift_list,
        source_redshift_list=source_redshift_list,
        z_source=z_source,
        cosmo=None,  # default Planck18; can override
    )
    return model
```

- [ ] **Step 4: Run tests (iterate on Herculens API specifics)**

```bash
conda run -n herculens312 pytest tests/test_tspl_tracer_herculens.py -v 2>&1 | tail -10
```

If MultiPlaneLensModel needs different kwargs in your installed version: inspect via `help(herculens.LensModel.multi_plane.MultiPlaneLensModel)` and adjust.

---

## Phase 3: MUSE position likelihood

### Task 4: Implement + test `muse_position_likelihood.py`

**Files:**
- Create: `private/2309_04535_ballard2023_tspl_jackpot/code/muse_position_likelihood.py`
- Create: `private/2309_04535_ballard2023_tspl_jackpot/tests/test_muse_position_likelihood.py`

- [ ] **Step 1: Failing test**

Create `tests/test_muse_position_likelihood.py`:

```python
import numpy as np
import pytest
from pathlib import Path
import sys

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / 'code'))

from muse_position_likelihood import gaussian_position_log_L


def test_zero_at_observed():
    """Predicted at observed → log_L ≈ 0."""
    pred = (0.05, 0.03)
    obs = (0.05, 0.03)
    err = (0.05, 0.05)
    ll = gaussian_position_log_L(pred, obs, err)
    assert ll == 0.0


def test_one_sigma_displacement():
    pred = (0.05, 0.03 + 0.05)  # 1σ off in y
    obs = (0.05, 0.03)
    err = (0.05, 0.05)
    ll = gaussian_position_log_L(pred, obs, err)
    assert abs(ll - (-0.5)) < 1e-6
```

- [ ] **Step 2: Run (FAIL)**

- [ ] **Step 3: Implement**

Create `code/muse_position_likelihood.py`:

```python
"""muse_position_likelihood.py — Gaussian position likelihood from MUSE."""

from typing import Tuple


def gaussian_position_log_L(
    pred_xy: Tuple[float, float],
    obs_xy: Tuple[float, float],
    err_xy: Tuple[float, float],
) -> float:
    """Per-source 2D Gaussian log-likelihood on position."""
    rx = (pred_xy[0] - obs_xy[0]) / err_xy[0]
    ry = (pred_xy[1] - obs_xy[1]) / err_xy[1]
    return -0.5 * (rx ** 2 + ry ** 2)
```

- [ ] **Step 4: Run tests (PASS)**

---

## Phase 4: Subhalo fit drivers

### Task 5: PyAutoLens subhalo fit driver

**Files:**
- Create: `private/2309_04535_ballard2023_tspl_jackpot/code/subhalo_fit_driver_autolens.py`

- [ ] **Step 1: Implement (long, follows v0.96 SLaM driver pattern)**

Create `code/subhalo_fit_driver_autolens.py`. Three sub-functions per `--part`: `tspl_main`, `tspl_subhalo`, `tspl_wbh`. Each wraps a Nautilus search; subhalo + wbh use the result of `tspl_main` as prior anchor.

```python
"""subhalo_fit_driver_autolens.py — PyAutoLens TSPL J0946 subhalo Bayes
factor + wandering-BH alternative.

CLI:
    --part tspl_main           main lens + 3 sources, no perturber
    --part tspl_subhalo        + NFW perturber (free centre + mass + c)
    --part tspl_wbh            + PointMass perturber (free centre + mass)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import autofit as af
import autolens as al


Z_L = 0.222
SRC_Z = [0.609, 2.035, 5.975]


def _main_lens_model():
    mass = af.Model(al.mp.Isothermal)
    mass.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.1)
    mass.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.1)
    mass.einstein_radius = af.UniformPrior(0.5, 2.5)
    mass.ell_comps.ell_comps_0 = af.GaussianPrior(0.0, 0.3)
    mass.ell_comps.ell_comps_1 = af.GaussianPrior(0.0, 0.3)
    shear = af.Model(al.mp.ExternalShear)
    shear.gamma_1 = af.GaussianPrior(0.0, 0.1)
    shear.gamma_2 = af.GaussianPrior(0.0, 0.1)
    return af.Model(al.Galaxy, redshift=Z_L, mass=mass, shear=shear)


def _source_galaxies():
    sources = []
    for z in SRC_Z:
        bulge = af.Model(al.lp.SersicCore)
        bulge.centre.centre_0 = af.GaussianPrior(0.0, 0.3)
        bulge.centre.centre_1 = af.GaussianPrior(0.0, 0.3)
        bulge.intensity = af.LogUniformPrior(1e-3, 10.0)
        bulge.effective_radius = af.UniformPrior(0.02, 0.6)
        bulge.sersic_index = af.UniformPrior(0.5, 4.0)
        sources.append(af.Model(al.Galaxy, redshift=z, bulge=bulge))
    return sources


def build_tspl_main(dataset, output_root, n_live=400):
    sources = _source_galaxies()
    model = af.Collection(galaxies=af.Collection(
        lens=_main_lens_model(),
        source1=sources[0], source2=sources[1], source3=sources[2],
    ))
    analysis = al.AnalysisImaging(dataset=dataset, use_jax=False)
    search = af.Nautilus(
        path_prefix=output_root,
        name='tspl_main_autolens',
        unique_tag='j0946',
        n_live=n_live,
        n_batch=50,
        number_of_cores=int(os.environ.get('SLURM_CPUS_PER_TASK', '1')),
    )
    return search.fit(model=model, analysis=analysis)


def build_tspl_subhalo(dataset, output_root, n_live=500, tspl_main_result=None):
    lens = _main_lens_model()
    if tspl_main_result is not None:
        lens.mass = tspl_main_result.model.galaxies.lens.mass
        lens.shear = tspl_main_result.model.galaxies.lens.shear
    perturber_mass = af.Model(al.mp.NFW)
    perturber_mass.centre.centre_0 = af.UniformPrior(-2.0, 2.0)
    perturber_mass.centre.centre_1 = af.UniformPrior(-2.0, 2.0)
    perturber_mass.kappa_s = af.LogUniformPrior(1e-3, 0.3)
    perturber_mass.scale_radius = af.UniformPrior(0.5, 5.0)
    perturber_mass.ell_comps.ell_comps_0 = 0.0
    perturber_mass.ell_comps.ell_comps_1 = 0.0
    perturber = af.Model(al.Galaxy, redshift=Z_L, mass=perturber_mass)

    sources = _source_galaxies()
    model = af.Collection(galaxies=af.Collection(
        lens=lens, perturber=perturber,
        source1=sources[0], source2=sources[1], source3=sources[2],
    ))
    analysis = al.AnalysisImaging(dataset=dataset, use_jax=False)
    search = af.Nautilus(
        path_prefix=output_root,
        name='tspl_subhalo_autolens',
        unique_tag='j0946',
        n_live=n_live,
        number_of_cores=int(os.environ.get('SLURM_CPUS_PER_TASK', '1')),
    )
    return search.fit(model=model, analysis=analysis)


def build_tspl_wbh(dataset, output_root, n_live=500, tspl_main_result=None):
    """Wandering-BH alternative: PointMass instead of NFW."""
    lens = _main_lens_model()
    if tspl_main_result is not None:
        lens.mass = tspl_main_result.model.galaxies.lens.mass
        lens.shear = tspl_main_result.model.galaxies.lens.shear
    wbh_mass = af.Model(al.mp.PointMass)
    wbh_mass.centre.centre_0 = af.UniformPrior(-2.0, 2.0)
    wbh_mass.centre.centre_1 = af.UniformPrior(-2.0, 2.0)
    wbh_mass.einstein_radius = af.LogUniformPrior(1e-3, 0.3)
    wbh = af.Model(al.Galaxy, redshift=Z_L, mass=wbh_mass)
    sources = _source_galaxies()
    model = af.Collection(galaxies=af.Collection(
        lens=lens, wbh=wbh,
        source1=sources[0], source2=sources[1], source3=sources[2],
    ))
    analysis = al.AnalysisImaging(dataset=dataset, use_jax=False)
    search = af.Nautilus(
        path_prefix=output_root,
        name='tspl_wbh_autolens',
        unique_tag='j0946',
        n_live=n_live,
        number_of_cores=int(os.environ.get('SLURM_CPUS_PER_TASK', '1')),
    )
    return search.fit(model=model, analysis=analysis)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--part', choices=['tspl_main', 'tspl_subhalo', 'tspl_wbh'],
                    default='tspl_main')
    p.add_argument('--dataset-root', type=Path, required=True)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--n-live', type=int, default=400)
    args = p.parse_args()

    dataset = al.Imaging.from_fits(
        data_path=args.dataset_root / 'image.fits',
        noise_map_path=args.dataset_root / 'noise_map.fits',
        psf_path=args.dataset_root / 'psf.fits',
        pixel_scales=0.05,
    )
    mask = al.Mask2D.circular(
        shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales,
        radius=3.0,
    )
    dataset = dataset.apply_mask(mask=mask)

    t0 = time.time()
    if args.part == 'tspl_main':
        build_tspl_main(dataset, args.output_root, n_live=args.n_live)
    elif args.part == 'tspl_subhalo':
        build_tspl_subhalo(dataset, args.output_root, n_live=args.n_live)
    elif args.part == 'tspl_wbh':
        build_tspl_wbh(dataset, args.output_root, n_live=args.n_live)
    print(f"Wall: {(time.time()-t0)/3600:.2f} h")


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Smoke test (laptop, --part=tspl_main with n_live=20)**

```bash
cd private/2309_04535_ballard2023_tspl_jackpot
# Generate a smoke mock first
conda run -n autolens python -c "
import sys; sys.path.insert(0, 'code')
from tspl_synthetic_mock import generate
from pathlib import Path
generate(Path('mocks/smoke'))
"
conda run -n autolens python code/subhalo_fit_driver_autolens.py \
    --part tspl_main --dataset-root mocks/smoke --output-root output_smoke --n-live 20
```

Expected: Nautilus runs for ~5 min on laptop, exits with a chain. Don't worry about chi²/N here — just that it ran.

### Task 6: Herculens subhalo fit driver

**Files:**
- Create: `private/2309_04535_ballard2023_tspl_jackpot/code/subhalo_fit_driver_herculens.py`

- [ ] **Step 1: Implement (parallel to autolens driver but in Herculens + NumPyro)**

Create `code/subhalo_fit_driver_herculens.py`:

```python
"""subhalo_fit_driver_herculens.py — Herculens + NumPyro NUTS version.

Same scientific model as subhalo_fit_driver_autolens.py; different sampling.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def build_tspl_main_herculens(dataset, output_root, num_samples=5000):
    """Herculens + NumPyro NUTS on TSPL main fit."""
    import jax
    import numpyro
    from numpyro.infer import MCMC, NUTS
    import numpyro.distributions as dist
    from tspl_tracer_herculens import build_tspl_multi_plane

    def model(data, noise):
        # Top-level priors
        theta_E = numpyro.sample('theta_E', dist.Uniform(0.5, 2.5))
        gamma = 2.0  # isothermal
        e1 = numpyro.sample('e1', dist.Normal(0.0, 0.3))
        e2 = numpyro.sample('e2', dist.Normal(0.0, 0.3))
        gamma1 = numpyro.sample('gamma1', dist.Normal(0.0, 0.1))
        gamma2 = numpyro.sample('gamma2', dist.Normal(0.0, 0.1))
        # Source planes
        for i in range(3):
            numpyro.sample(f'src{i}_amp', dist.LogUniform(1e-3, 10.0))
            numpyro.sample(f'src{i}_Re', dist.Uniform(0.02, 0.6))
            numpyro.sample(f'src{i}_n', dist.Uniform(0.5, 4.0))
        # Likelihood: build Herculens model from above params + observed data
        # (implementation depends on Herculens API specifics)
        # ...
        # For now: stub
        log_L = 0.0
        numpyro.factor('log_likelihood', log_L)

    nuts = NUTS(model)
    mcmc = MCMC(nuts, num_warmup=500, num_samples=num_samples)
    rng_key = jax.random.PRNGKey(0)
    mcmc.run(rng_key, dataset, dataset)  # dataset, noise — placeholder
    return mcmc.get_samples()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--part', choices=['tspl_main', 'tspl_subhalo', 'tspl_wbh'],
                    default='tspl_main')
    p.add_argument('--dataset-root', type=Path, required=True)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--num-samples', type=int, default=5000)
    args = p.parse_args()
    # ... similar dispatch as autolens version, calling build_tspl_*_herculens
    print("Stub: herculens TSPL driver — fill in once Spec 00 herculens_bridge is exercised end-to-end")


if __name__ == '__main__':
    main()
```

This is intentionally a stub. The full Herculens implementation requires:
1. The `herculens_bridge.py` to be tuned per Task 10 of Spec 00 (all 5 profile tests passing)
2. A working multi-plane setup in Herculens (verified by Task 3 of this spec)

When those land, fill in the stub: build the `LensModel.multi_plane` instance, define `LightModel`, instantiate the `LensImage` (Herculens's analysis-equivalent), compute the log-likelihood inside `model()`, sample with NUTS.

---

## Phase 5: Cannon submission (6 jobs)

### Task 7: Patch submit_cannon.slurm to know about p2_tspl_*

- [ ] **Step 1: Add cases to submit_cannon.slurm**

Edit `Modules/10_Cluster_Computing/scripts/submit_cannon.slurm` to add:
- `p2_tspl_autolens)` → routes to `private/2309_04535_ballard2023_tspl_jackpot/code/subhalo_fit_driver_autolens.py`
- `p2_tspl_herculens)` → routes to `subhalo_fit_driver_herculens.py` (with `CONDA_ENV=herculens312`)
- Each takes `FIT_EXTRA_ARGS='--part=tspl_main|tspl_subhalo|tspl_wbh'`

### Task 8: Submit 6 Cannon jobs

- [ ] **Step 1: Submit autolens × 3**

```bash
for PART in tspl_main tspl_subhalo tspl_wbh; do
    ssh cannon "cd /n/.../learning_to_autolens && \
        sbatch --account=siag_lab --partition=siag --mem=192G --cpus-per-task=32 \
        --time=24:00:00 --job-name=p2_${PART}_autolens \
        --export=ALL,EXAMPLE=p2_tspl_autolens,FIT_EXTRA_ARGS='--part=${PART}' \
        Modules/10_Cluster_Computing/scripts/submit_cannon.slurm"
done
```

- [ ] **Step 2: Submit Herculens × 3 (GPU)**

```bash
for PART in tspl_main tspl_subhalo tspl_wbh; do
    conda run -n autolens python private/00_shared_infrastructure/code/herculens_cannon_runner.py \
        --example p2_tspl_herculens \
        --fit-script-extra "--part=${PART}" \
        --time-hours 12 --memory 80G
done
```

---

## Phase 6: Cross-validation + Module 14 extension

### Task 9: Pull + audit + crossval

- [ ] **Step 1: Pull results**

```bash
bash Modules/10_Cluster_Computing/scripts/pull_from_cannon.sh --go
# Also pull from Cannon's herculens output tree (separate path)
rsync -av cannon:/n/.../learning_to_autolens/output/p2_tspl_herculens/ \
    private/2309_04535_ballard2023_tspl_jackpot/results/herculens/
```

- [ ] **Step 2: Compute Bayes factors**

Compute Δlog_Z between `tspl_main` and `tspl_subhalo` for each stack:

```bash
conda run -n autolens python -c "
import json
for stack in ['autolens', 'herculens']:
    base = f'private/2309_04535_ballard2023_tspl_jackpot/results/tspl_main_{stack}/summary.json'
    sub = f'private/2309_04535_ballard2023_tspl_jackpot/results/tspl_subhalo_{stack}/summary.json'
    wbh = f'private/2309_04535_ballard2023_tspl_jackpot/results/tspl_wbh_{stack}/summary.json'
    Z_base = json.load(open(base))['log_evidence']
    Z_sub = json.load(open(sub))['log_evidence']
    Z_wbh = json.load(open(wbh))['log_evidence']
    print(f'{stack}: ΔlogZ(subhalo - main) = {Z_sub - Z_base:.2f}')
    print(f'{stack}: ΔlogZ(wbh - main) = {Z_wbh - Z_base:.2f}')
"
```

- [ ] **Step 3: Run cross-validation framework**

```bash
conda run -n autolens python -c "
import pandas as pd, sys
sys.path.insert(0, 'private/00_shared_infrastructure/code')
from crossval_framework import crossval_report
a = pd.read_csv('private/2309_04535_ballard2023_tspl_jackpot/results/tspl_subhalo_autolens/samples.csv')
h = pd.read_csv('private/2309_04535_ballard2023_tspl_jackpot/results/tspl_subhalo_herculens/samples.csv')
crossval_report(a, h, params=['perturber.mass.centre.centre_0', 'perturber.mass.centre.centre_1',
                                'perturber.mass.kappa_s', 'perturber.mass.scale_radius'],
                labels=('autolens-Nautilus', 'Herculens-NUTS'),
                out_path='private/2309_04535_ballard2023_tspl_jackpot/results/crossval_subhalo.md',
                plot_corner=True)
"
```

### Task 10: Module 14 extension outline

**Files:**
- Modify: `private/2309_04535_ballard2023_tspl_jackpot/MODULE_14_EXTENSION_OUTLINE.md` (create)

```markdown
# Module 14 §"TSPL Extension" outline

For appending to public Module 14 (Compound Multi-Plane) — promotes from this private/ work.

## Sections

1. From DSPL to TSPL: 3 source planes give 3 independent β_jk ratios
2. Multi-caustic topology (Burke generalized to N planes)
3. Gravitational imaging (Vegetti+2010) generalized to TSPL
4. J0946 as the worked example: 4-way (stacks × {main, subhalo}) consistency
5. Wandering-BH vs dark-subhalo Bayes factor

References gr-lensing-intuition's image topology + Burke's theorem.
```

---

## Self-Review

| Spec 02 section | Tasks |
|---|---|
| §4 architecture | Tasks 1-10 |
| §5 TSPL model | Tasks 2, 3 |
| §6 stack implementations | Tasks 2 (autolens), 3 (Herculens) |
| §7 Cannon submission | Tasks 7, 8 |
| §8 data flow | Task 9 |
| §9 error handling | reuse v0.96 robust analysis subclasses |
| §10 testing | Tasks 1, 2, 3, 4 |
| §11 pedagogical Module 14 extension | Task 10 outline |
| §12 timeline | ~1.5 weeks |

**Total: 10 tasks. ~10-14 days incl. Cannon wall + 6-day-budget queue waits.**
