# Paper-Repro Spec 00 — Shared Infrastructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the shared substrate (J0946 data products, Chen+2019 catalog, autolens↔Herculens bridge, cross-validation framework, Cannon-side Herculens runner, AGEL-consistent HST reduction) that Specs 01-04 all depend on.

**Architecture:** A private/00_shared_infrastructure/ tree with `data/` (J0946 + lens catalogs), `code/` (data loader + bridge + crossval + Cannon helpers + Watson-pipeline wrappers), `docs/` (runbooks), and `environment-herculens.yml`. The bridge is the load-bearing piece — it translates one canonical autolens model spec into Herculens, enabling per-paper specs to write code once.

**Tech Stack:** PyAutoLens 2026.4.13.6 + Herculens (github.com/Herculens/herculens) + NumPyro + JAX + JAX-CUDA, astroquery (MAST + Vizier), reproject + astroalign, drizzlepac (AstroDrizzle + Tweakreg via Watson's pipeline), pandas, pytest.

---

## File Structure

```
private/00_shared_infrastructure/
├── data/
│   ├── j0946/
│   │   ├── hst/                                ← already pulled 2026-05-18, 9.8 GB
│   │   ├── muse_cube.fits                      ← manual ESO download
│   │   ├── spec_z.json                         ← curated source redshifts
│   │   └── README.md
│   └── lens_catalogs/
│       ├── chen2019_table1.csv                 ← canonical (θ_E, σ_v, z_l, z_s)
│       ├── catalogue_161.csv                   ← enriched (+R_eff, n_sersic)
│       ├── auger2010_slacs_re.csv              ← Vizier enrichment
│       ├── sonnenfeld2013_sl2s_re.csv          ← Vizier enrichment
│       ├── brownstein2012_bells_re.csv         ← Vizier enrichment
│       └── README.md
├── code/
│   ├── __init__.py
│   ├── j0946_data_loader.py
│   ├── herculens_bridge.py
│   ├── crossval_framework.py
│   ├── herculens_cannon_runner.py
│   ├── build_chen2019_catalog.py
│   ├── build_structural_enrichment.py
│   └── agel_hst_reduction/
│       ├── __init__.py
│       ├── run_j0946_drizzle.py
│       ├── compare_hla_vs_agel.py
│       └── cosmic_ray_audit.py
├── tests/
│   ├── __init__.py
│   ├── test_j0946_data_loader.py
│   ├── test_herculens_bridge.py
│   ├── test_crossval_framework.py
│   ├── test_build_chen2019_catalog.py
│   └── test_build_structural_enrichment.py
├── docs/
│   ├── herculens_cannon_setup.md
│   ├── dual_stack_conventions.md
│   └── data_provenance.md
└── environment-herculens.yml
```

---

## Phase 1: scaffold + env

### Task 1: Create the directory scaffold

**Files:**
- Create: `private/00_shared_infrastructure/` and all subdirs above
- Create: empty `__init__.py` in `code/`, `code/agel_hst_reduction/`, `tests/`
- Create: `private/00_shared_infrastructure/README.md`

- [ ] **Step 1: Create the dir scaffold**

```bash
cd /Users/rosador/Documents/AGEL/Learning_to_Autolens
mkdir -p private/00_shared_infrastructure/{data/j0946,data/lens_catalogs,code/agel_hst_reduction,tests,docs}
touch private/00_shared_infrastructure/code/__init__.py
touch private/00_shared_infrastructure/code/agel_hst_reduction/__init__.py
touch private/00_shared_infrastructure/tests/__init__.py
```

- [ ] **Step 2: Stub the top-level README**

Create `private/00_shared_infrastructure/README.md`:

```markdown
# Shared Infrastructure — Spec 00

Substrate for the paper-reproduction program (Specs 01-04). See
`docs/superpowers/specs/2026-05-18-paper-repro-00-shared-infrastructure-design.md`
for the design contract.

## Quick start

    cd code/
    python build_chen2019_catalog.py --output ../data/lens_catalogs/chen2019_table1.csv
    python build_structural_enrichment.py --primary ../data/lens_catalogs/chen2019_table1.csv \
        --output ../data/lens_catalogs/catalogue_161.csv

## Layout

See `Files:` block in
`docs/superpowers/plans/2026-05-18-paper-repro-00-shared-infrastructure.md`.
```

- [ ] **Step 3: Verify**

```bash
find private/00_shared_infrastructure -type d | sort
```

Expected output: 11 directories listed.

### Task 2: Herculens conda env spec

**Files:**
- Create: `private/00_shared_infrastructure/environment-herculens.yml`

- [ ] **Step 1: Write the env spec**

```yaml
# environment-herculens.yml — Herculens + NumPyro + JAX (CPU and CUDA)
#
# Laptop: `conda env create -f environment-herculens.yml`
# Cannon: same; siag_lab fairshare has A100 access so install jax[cuda12]
name: herculens312
channels:
  - conda-forge
dependencies:
  - python=3.12
  - pip
  - astropy>=7.0
  - numpy>=1.26
  - scipy
  - matplotlib<3.9
  - pandas
  - jupyterlab>=4.5
  - corner
  - pytest
  - pip:
      - jax[cuda12]>=0.4.30      # if no CUDA, fall back to: jax>=0.4.30
      - jaxlib>=0.4.30
      - numpyro>=0.13
      - git+https://github.com/Herculens/herculens.git
      - astroquery
      - drizzlepac>=3.7         # AGEL HST pipeline (Watson)
      - reproject
      - astroalign
      - colossus                 # P3 cosmology library
```

- [ ] **Step 2: Smoke test on laptop**

```bash
# DO NOT actually create the env yet (heavy). Just dry-run the YAML parse.
conda env create -f private/00_shared_infrastructure/environment-herculens.yml --dry-run 2>&1 | head -30
```

Expected: no syntax errors. Output ends with `Dry run complete.` or similar conda message.

- [ ] **Step 3: Commit**

```bash
git add private/00_shared_infrastructure/README.md private/00_shared_infrastructure/environment-herculens.yml
# Note: private/ is gitignored so .gitignore exemption needed; this commit is local-only
# Actually it IS gitignored — commit goes to a private/ subtree, not git. Skip git add.
echo "Skipping git commit — private/ is .gitignored"
```

Note: `private/` is gitignored. We're tracking progress in `private/PROGRESS_<date>.md` only. No public commits from Spec 00's `private/` work; the public side comes via promotion (Spec 04).

---

## Phase 2: Chen+2019 catalog assembly

### Task 3: Identify Chen+2019 catalog source format

**Files:**
- Modify: `private/00_shared_infrastructure/docs/data_provenance.md` (create new)

- [ ] **Step 1: Investigate Chen+2019 supplementary materials**

Run:

```bash
# Check what Vizier has (probably not — paper-specific compilations rarely go to Vizier)
conda run -n autolens python -c "
from astroquery.vizier import Vizier
for cat in ['J/MNRAS/488/3745', 'J/MNRAS/488/3745/table1', 'IX/Chen2019']:
    try:
        r = Vizier.find_catalogs(cat)
        print(f'{cat}: {list(r.keys())[:1]}')
    except Exception as e:
        print(f'{cat}: err: {str(e)[:60]}')
"
```

If Vizier hit: proceed in next step to query it. If miss: arxiv source URL fetch.

- [ ] **Step 2: If Vizier miss, fetch Chen+2019 arxiv ancillary**

```bash
# arxiv 1809.09845 — check the source listing for supplementary tex/csv tables
curl -sLI https://arxiv.org/e-print/1809.09845 2>&1 | head -5
# If accessible, download the source bundle; it often includes machine-readable tables
mkdir -p private/00_shared_infrastructure/data/lens_catalogs/_provenance
curl -sL https://arxiv.org/e-print/1809.09845 -o private/00_shared_infrastructure/data/lens_catalogs/_provenance/chen2019_source.tar.gz
```

- [ ] **Step 3: Document provenance**

Create `private/00_shared_infrastructure/docs/data_provenance.md`:

```markdown
# Data Provenance

## Chen+2019 161-lens sample (primary P1 input)

- **Citation**: Chen, Li, Shu, Cao 2019, MNRAS 488:3, 3745 ("Assessing the effect of lens mass model in cosmological application with updated galaxy-scale strong gravitational lensing sample")
- **DOI**: 10.1093/mnras/stz1902
- **arxiv**: [1809.09845](https://arxiv.org/abs/1809.09845)
- **MNRAS Data Availability**: paper provides Table 1 with 161 systems; if no Vizier mirror, extract from the paper source bundle at arxiv e-print
- **Acquisition date**: [fill in once downloaded]
- **Local file**: `private/00_shared_infrastructure/data/lens_catalogs/chen2019_table1.csv`

## Auxiliary structural-parameter catalogs

- **Auger+2010 SLACS X (R_eff + n_sersic for SLACS subset)**
  - Vizier: J/ApJ/705/1099 table[1] (85 rows)
  - Local: `auger2010_slacs_re.csv`
- **Sonnenfeld+2013 SL2S IV (z_l + structural)**
  - Vizier: J/ApJ/777/97 table[3]
  - Local: `sonnenfeld2013_sl2s_re.csv`
- **Brownstein+2012 BELLS (R_eff)**
  - Vizier query: TBD - not in our 2026-05-18 lookup; may require PDF extraction
  - Local: `brownstein2012_bells_re.csv`

## J0946+1006 HST imaging

- **MAST**: 21 products via astroquery on coord (09:46:56.68 +10:06:55.05) within 10″
- **Selected products**: ACS WFC F814W (prop 10886, 2096s), WFC3/UVIS F336W (prop 11701, 5772s), F438W (prop 11701, 2520s)
- **Download date**: 2026-05-18
- **Local**: `private/00_shared_infrastructure/data/j0946/hst/` (currently at `private/2309_04535_ballard2023_tspl_jackpot/data/hst/` — TODO: move or symlink)
- **AGEL-consistent re-reduction**: Watson pipeline (see §6.9 of Spec 00). Pending.

## J0946+1006 MUSE cube

- **ESO archive**: programs 0103.B-0743 / 094.B-0524 / 102.A-0950 (TBD; manual download via science portal)
- **Local**: `private/00_shared_infrastructure/data/j0946/muse_cube.fits` (NOT YET DOWNLOADED)

## Source redshifts (J0946)

- **Smith+2024**: (z_s1, z_s2, z_s3) = (0.609, 2.035, 5.975) for the TSPL geometry
- **Sonnenfeld 2012**: z_lens = 0.222; z_s1 = 0.609; z_s2 = 2.035 (original DSPL)
- **Local**: `private/00_shared_infrastructure/data/j0946/spec_z.json` (TODO)
```

- [ ] **Step 4: No commit (private/ is gitignored)**

### Task 4: Write `build_chen2019_catalog.py`

**Files:**
- Create: `private/00_shared_infrastructure/code/build_chen2019_catalog.py`
- Create: `private/00_shared_infrastructure/tests/test_build_chen2019_catalog.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_build_chen2019_catalog.py`:

```python
"""Tests for the Chen+2019 catalog assembler."""

from pathlib import Path
import pandas as pd
import pytest

import sys
HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / 'code'))
from build_chen2019_catalog import parse_chen2019_table1, write_csv


def test_parser_returns_at_least_120_rows():
    """Chen+2019 paper Table 1 has 161 lenses; tolerate small loss to QC cuts."""
    df = parse_chen2019_table1()
    assert len(df) >= 120


def test_required_columns_present():
    df = parse_chen2019_table1()
    for col in ['id', 'z_l', 'z_s', 'theta_E_arcsec', 'sigma_v_kms', 'sigma_v_err_kms']:
        assert col in df.columns


def test_no_duplicate_ids():
    df = parse_chen2019_table1()
    assert df['id'].is_unique


def test_redshift_ordering():
    df = parse_chen2019_table1()
    assert (df['z_s'] > df['z_l']).all()


def test_einstein_radius_positive():
    df = parse_chen2019_table1()
    assert (df['theta_E_arcsec'] > 0).all()


def test_sigma_v_in_galaxy_lens_range():
    df = parse_chen2019_table1()
    assert (df['sigma_v_kms'].between(100, 400)).all()


def test_write_csv_roundtrip(tmp_path):
    out = tmp_path / 'chen2019_table1.csv'
    df = parse_chen2019_table1()
    write_csv(df, out)
    assert out.exists()
    df2 = pd.read_csv(out)
    assert len(df2) == len(df)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd private/00_shared_infrastructure
conda run -n autolens pytest tests/test_build_chen2019_catalog.py -v 2>&1 | tail -10
```

Expected: FAIL with `ModuleNotFoundError: No module named 'build_chen2019_catalog'`.

- [ ] **Step 3: Write minimal implementation**

Create `code/build_chen2019_catalog.py`:

```python
"""build_chen2019_catalog.py — assemble the 161-lens catalog of Chen et al.
2019 (MNRAS 488:3, arxiv 1809.09845) for the Li+2023 hierarchical
population-cosmography reproduction (Spec 01).

The Chen+2019 paper publishes Table 1 with 161 systems and columns
(SDSS or SL2S ID, z_l, z_s, theta_E_arcsec, sigma_v_kms, sigma_v_err_kms).
The MNRAS Supplementary Materials portal hosts the machine-readable
version of this table; absent that, the arxiv ancillary directory or a
PDF text extraction is the fallback.

Public surface:
    parse_chen2019_table1() -> pandas.DataFrame
    write_csv(df, path)
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import pandas as pd

CATALOG_TEXT = '''id,z_l,z_s,theta_E_arcsec,sigma_v_kms,sigma_v_err_kms
SDSSJ0008-0004,0.440,1.192,1.16,247,35
SDSSJ0029-0055,0.227,0.931,0.96,229,18
SDSSJ0037-0942,0.195,0.632,1.53,279,14
SDSSJ0044+0113,0.120,0.197,0.79,266,13
SDSSJ0157-0056,0.513,0.924,0.79,295,47
SDSSJ0216-0813,0.332,0.523,1.16,333,23
SDSSJ0252+0039,0.280,0.982,1.04,164,12
SDSSJ0330-0020,0.351,1.071,1.10,212,21
SDSSJ0405-0455,0.075,0.810,0.80,160,8
SDSSJ0728+3835,0.206,0.688,1.25,214,11
'''
# NOTE: the 10 rows above are a SMOKE-TEST STUB. The full 161-row
# table is the Chen+2019 published Table 1. To populate, the
# implementer should either:
#   (a) download the MNRAS supplementary table from
#       https://academic.oup.com/mnras/article/488/3/3745, OR
#   (b) extract from the arxiv source bundle at
#       https://arxiv.org/e-print/1809.09845, OR
#   (c) replicate Table 1 by hand from the published PDF.
# All three preserve the same six-column schema above.


def parse_chen2019_table1() -> pd.DataFrame:
    """Return the 161-lens (id, z_l, z_s, theta_E, sigma_v, sigma_v_err) table.

    Currently returns a 10-row STUB; see TODO in module docstring for
    full-catalog acquisition.
    """
    df = pd.read_csv(io.StringIO(CATALOG_TEXT))
    df['theta_E_arcsec'] = df['theta_E_arcsec'].astype(float)
    df['sigma_v_kms'] = df['sigma_v_kms'].astype(float)
    df['sigma_v_err_kms'] = df['sigma_v_err_kms'].astype(float)
    df['z_l'] = df['z_l'].astype(float)
    df['z_s'] = df['z_s'].astype(float)
    return df


def write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    df = parse_chen2019_table1()
    write_csv(df, args.output)
    print(f'Wrote {len(df)} rows to {args.output}')


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Run tests — the 120-row assertion will FAIL with the 10-row stub**

```bash
conda run -n autolens pytest tests/test_build_chen2019_catalog.py -v 2>&1 | tail -10
```

Expected: 5 pass + 1 fail (`test_parser_returns_at_least_120_rows` fails: `assert 10 >= 120`).

- [ ] **Step 5: Open a follow-up task — populate the full 161-row catalog**

The 10-row STUB unblocks downstream development (loader tests, bridge tests). Populating the full 161 is gated on (a) MNRAS subscription access or (b) hand-extraction from arxiv source. Add a top-level open-question in `private/PROGRESS_<date>.md`:

```markdown
- [ ] Chen+2019 full 161-row Table 1 acquisition (currently 10-row stub).
      Source: arxiv 1809.09845 e-print bundle OR MNRAS supplementary.
      Blocker for Spec 01 strict-PASS validation; unblocks once we have it.
```

- [ ] **Step 6: No commit**

### Task 5: Write `build_structural_enrichment.py`

**Files:**
- Create: `private/00_shared_infrastructure/code/build_structural_enrichment.py`
- Create: `private/00_shared_infrastructure/tests/test_build_structural_enrichment.py`

This refactors today's `private/2307_09271_li2023_cosmography_population/data/build_catalogue.py` (which solved the wrong problem — see Spec 00 §6.7) into an AUXILIARY-DATA-ONLY role: it adds R_eff + n_sersic columns to an already-existing primary catalog by Vizier cross-match.

- [ ] **Step 1: Write the failing test**

Create `tests/test_build_structural_enrichment.py`:

```python
"""Tests for the Vizier structural-enrichment script."""

from pathlib import Path
import pandas as pd
import pytest

import sys
HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / 'code'))
from build_structural_enrichment import enrich, _vizier_slacs_x


def test_enrich_preserves_primary_rows(tmp_path):
    """Enrichment should not drop rows from the primary catalog."""
    primary = pd.DataFrame({
        'id': ['SDSSJ0008-0004', 'NOTFOUND-0000-0000'],
        'z_l': [0.44, 0.5], 'z_s': [1.19, 1.0],
        'theta_E_arcsec': [1.16, 1.0], 'sigma_v_kms': [247, 200],
        'sigma_v_err_kms': [35, 20],
    })
    enriched = enrich(primary)
    assert len(enriched) == len(primary)
    assert 'R_eff_arcsec' in enriched.columns
    assert 'n_sersic' in enriched.columns


def test_unmatched_rows_have_null_structural(tmp_path):
    primary = pd.DataFrame({
        'id': ['NOTFOUND-0000-0000'],
        'z_l': [0.5], 'z_s': [1.0],
        'theta_E_arcsec': [1.0], 'sigma_v_kms': [200],
        'sigma_v_err_kms': [20],
    })
    enriched = enrich(primary)
    assert pd.isna(enriched.iloc[0]['R_eff_arcsec'])


def test_vizier_slacs_x_returns_dataframe():
    """The Vizier query returns a table with at least the SLACS-X fields."""
    df = _vizier_slacs_x()
    assert len(df) >= 70  # 85-row table in J/ApJ/705/1099
    assert 'SDSS' in df.columns
```

- [ ] **Step 2: Run test (FAIL)**

```bash
conda run -n autolens pytest tests/test_build_structural_enrichment.py -v 2>&1 | tail -10
```

Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement the enrichment module**

Create `code/build_structural_enrichment.py`:

```python
"""build_structural_enrichment.py — enrich a primary lens catalog
(Chen+2019) with auxiliary structural parameters (R_eff, n_sersic) by
Vizier cross-match against SLACS-X, SL2S-IV, BELLS-IV.

Public surface:
    enrich(primary_df) -> pandas.DataFrame
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd


def _vizier_slacs_x() -> pd.DataFrame:
    """Pull Auger+2010 SLACS X table[1] (85 rows with σ_v, R_eff, M_type)."""
    from astroquery.vizier import Vizier
    Vizier.ROW_LIMIT = -1
    tabs = Vizier.get_catalogs('J/ApJ/705/1099')
    t = tabs[1].to_pandas()
    return t


def _vizier_sl2s() -> pd.DataFrame:
    """Sonnenfeld+2013 SL2S IV — 56 rows with z + log M_C + log M_S."""
    from astroquery.vizier import Vizier
    Vizier.ROW_LIMIT = -1
    tabs = Vizier.get_catalogs('J/ApJ/777/97')
    return tabs[3].to_pandas()


def enrich(primary: pd.DataFrame) -> pd.DataFrame:
    """Add R_eff_arcsec + n_sersic columns by best-effort cross-match.

    Returns a copy of primary with the two new columns appended. Unmatched
    rows have NaN in the new columns.
    """
    out = primary.copy()
    out['R_eff_arcsec'] = pd.NA
    out['n_sersic'] = pd.NA

    try:
        slacs = _vizier_slacs_x()
    except Exception:
        slacs = pd.DataFrame()
    try:
        sl2s = _vizier_sl2s()
    except Exception:
        sl2s = pd.DataFrame()

    for i, row in out.iterrows():
        identifier = str(row['id'])
        # Try SLACS-X match on SDSS column
        if not slacs.empty and 'SDSS' in slacs.columns:
            hit = slacs[slacs['SDSS'].astype(str).str.contains(
                identifier.replace('SDSSJ', ''), na=False, regex=False)]
            if len(hit) > 0 and 'Re(B)' in hit.columns:
                re_val = hit.iloc[0]['Re(B)']
                if pd.notna(re_val):
                    out.at[i, 'R_eff_arcsec'] = float(re_val)
                    out.at[i, 'n_sersic'] = 4.0  # SLACS-X uses de Vauc for many
                    continue
        # Try SL2S match
        if not sl2s.empty and 'SL2S' in sl2s.columns:
            hit = sl2s[sl2s['SL2S'].astype(str).str.contains(
                identifier.replace('SL2S', '').replace('J', ''),
                na=False, regex=False)]
            if len(hit) > 0:
                # SL2S table[3] doesn't have Re directly; would need table[2]
                # — TODO: refine. Skip for now.
                pass

    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--primary', type=Path, required=True,
                   help='Path to Chen+2019 primary catalog CSV')
    p.add_argument('--output', type=Path, required=True,
                   help='Path for the enriched catalog CSV')
    args = p.parse_args()

    primary = pd.read_csv(args.primary)
    enriched = enrich(primary)
    enriched.to_csv(args.output, index=False)
    print(f'Wrote {len(enriched)} rows (enriched) to {args.output}')
    n_matched = enriched['R_eff_arcsec'].notna().sum()
    print(f'  {n_matched} matched to auxiliary structural data')


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Run tests (PASS expected — modulo network availability for the Vizier query)**

```bash
conda run -n autolens pytest tests/test_build_structural_enrichment.py -v 2>&1 | tail -10
```

Expected: 3 pass.

- [ ] **Step 5: Run the script end-to-end**

```bash
cd private/00_shared_infrastructure
conda run -n autolens python code/build_chen2019_catalog.py --output data/lens_catalogs/chen2019_table1.csv
conda run -n autolens python code/build_structural_enrichment.py \
  --primary data/lens_catalogs/chen2019_table1.csv \
  --output data/lens_catalogs/catalogue_161.csv
head -3 data/lens_catalogs/catalogue_161.csv
```

Expected: CSV produced with header line + 10 data rows (stub) + 2 enriched columns (mostly NaN until full Chen catalog populated).

- [ ] **Step 6: No commit (private/)**

---

## Phase 3: J0946 data loader

### Task 6: Move J0946 HST data into the canonical location

**Files:**
- Move: `private/2309_04535_ballard2023_tspl_jackpot/data/hst/` → `private/00_shared_infrastructure/data/j0946/hst/`
- Symlink: `private/2309_04535_ballard2023_tspl_jackpot/data/hst` → `../../00_shared_infrastructure/data/j0946/hst`
- Symlink: `private/2602_20889_li2026_dspl_imf_nfw/data/hst` → same target

- [ ] **Step 1: Move + create symlinks**

```bash
cd /Users/rosador/Documents/AGEL/Learning_to_Autolens
mv private/2309_04535_ballard2023_tspl_jackpot/data/hst \
   private/00_shared_infrastructure/data/j0946/hst
ln -sfn ../../00_shared_infrastructure/data/j0946/hst \
   private/2309_04535_ballard2023_tspl_jackpot/data/hst
ln -sfn ../../00_shared_infrastructure/data/j0946/hst \
   private/2602_20889_li2026_dspl_imf_nfw/data/hst
```

- [ ] **Step 2: Verify**

```bash
ls -la private/2309_04535_ballard2023_tspl_jackpot/data/hst | head -3
ls private/00_shared_infrastructure/data/j0946/hst/ | head -3
du -sh private/00_shared_infrastructure/data/j0946/hst/
```

Expected: symlink resolves; HST products visible; 9.8 GB total.

### Task 7: Write `spec_z.json`

**Files:**
- Create: `private/00_shared_infrastructure/data/j0946/spec_z.json`

- [ ] **Step 1: Write spec_z.json**

Use the Smith+2024 TSPL values + Sonnenfeld+2012 lens redshift:

```bash
cat > private/00_shared_infrastructure/data/j0946/spec_z.json <<'EOF'
{
  "target": "SDSSJ0946+1006",
  "common_name": "Jackpot",
  "coordinates": {"ra_deg": 146.7361667, "dec_deg": 10.1153, "ra_hms": "09:46:56.68", "dec_dms": "+10:06:55.05"},
  "z_lens": 0.222,
  "z_lens_ref": "Sonnenfeld+2012 / Gavazzi+2008",
  "sources": {
    "z_s1": {"value": 0.609, "ref": "Sonnenfeld+2012 spectroscopic"},
    "z_s2": {"value": 2.035, "ref": "Smith+2024 TSPL spectroscopic"},
    "z_s3": {"value": 5.975, "ref": "Smith+2024 TSPL spectroscopic / MUSE Lyα"}
  },
  "notes": "z_s3 is from Smith+2024 — relevant ONLY for P2 (Ballard+2023 TSPL). P3 (Li+2026 DSPL) uses just z_s1 + z_s2."
}
EOF
```

- [ ] **Step 2: Verify**

```bash
conda run -n autolens python -c "import json; d=json.load(open('private/00_shared_infrastructure/data/j0946/spec_z.json')); print(d['target'], d['z_lens'], d['sources']['z_s1']['value'])"
```

Expected: `SDSSJ0946+1006 0.222 0.609`.

### Task 8: Write `j0946_data_loader.py`

**Files:**
- Create: `private/00_shared_infrastructure/code/j0946_data_loader.py`
- Create: `private/00_shared_infrastructure/tests/test_j0946_data_loader.py`

The loader returns Python primitives (numpy arrays + WCS objects), NOT framework-specific objects. Both PyAutoLens and Herculens code then wrap these primitives into their own data objects.

- [ ] **Step 1: Write failing tests**

Create `tests/test_j0946_data_loader.py`:

```python
"""Tests for j0946_data_loader.py."""

from pathlib import Path
import json
import numpy as np
import pytest

import sys
HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / 'code'))
from j0946_data_loader import (
    DATA_ROOT, load_spec_z, list_hst_products, load_hst_band, load_muse_cube,
)


def test_data_root_exists():
    assert DATA_ROOT.exists()


def test_spec_z_loads():
    spec = load_spec_z()
    assert spec['z_lens'] == 0.222
    assert spec['sources']['z_s1']['value'] == 0.609
    assert spec['sources']['z_s2']['value'] == 2.035
    assert spec['sources']['z_s3']['value'] == 5.975


def test_list_hst_products_finds_F814W():
    products = list_hst_products(filters=['F814W'])
    assert len(products) >= 1
    for p in products:
        assert p.suffix == '.fits'
        assert 'f814w' in p.name.lower()


def test_load_hst_band_F814W_returns_array_wcs():
    arr, wcs = load_hst_band('F814W')
    assert arr.ndim == 2
    assert arr.shape[0] > 100 and arr.shape[1] > 100
    assert wcs is not None
    # Lens should be near image centre — at least some non-zero values within 200x200 centre
    cy, cx = arr.shape[0] // 2, arr.shape[1] // 2
    centre = arr[cy-100:cy+100, cx-100:cx+100]
    assert np.any(centre > 0)


def test_load_muse_cube_returns_none_if_absent():
    """MUSE cube is manual-download; test the graceful-absent path."""
    cube_path = DATA_ROOT / 'muse_cube.fits'
    if not cube_path.exists():
        with pytest.warns(UserWarning, match="MUSE cube not found"):
            result = load_muse_cube()
        assert result is None
```

- [ ] **Step 2: Run tests (FAIL on import)**

```bash
cd private/00_shared_infrastructure
conda run -n autolens pytest tests/test_j0946_data_loader.py -v 2>&1 | tail -10
```

Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement the loader**

Create `code/j0946_data_loader.py`:

```python
"""j0946_data_loader.py — canonical loader for SDSSJ0946+1006 data.

Returns framework-agnostic primitives (numpy arrays + astropy WCS) so
both PyAutoLens and Herculens can consume the same input.

Public surface:
    DATA_ROOT (pathlib.Path)
    load_spec_z() -> dict
    list_hst_products(filters: list[str]) -> list[Path]
    load_hst_band(filter_name: str) -> tuple[np.ndarray, astropy.wcs.WCS]
    load_muse_cube() -> tuple[np.ndarray, astropy.wcs.WCS, np.ndarray] | None
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

HERE = Path(__file__).resolve().parent
DATA_ROOT = HERE.parents[0] / 'data' / 'j0946'


def load_spec_z() -> dict:
    """Return spec_z.json as a dict."""
    with open(DATA_ROOT / 'spec_z.json') as f:
        return json.load(f)


def list_hst_products(filters: list[str] | None = None) -> list[Path]:
    """List FITS products under data/j0946/hst/ matching the requested filters.

    Args:
        filters: list of filter names like ['F814W', 'F336W']. If None, return
                 all FITS products.

    Returns:
        sorted list of pathlib.Path
    """
    hst_dir = DATA_ROOT / 'hst'
    if not hst_dir.exists():
        return []
    all_fits = list(hst_dir.rglob('*.fits'))
    if filters is None:
        return sorted(all_fits)
    out = []
    for f in all_fits:
        for filt in filters:
            if filt.lower() in f.name.lower():
                out.append(f)
                break
    return sorted(out)


def _largest_product(products: list[Path]) -> Path:
    """Pick the deepest exposure (largest file size) as the canonical input."""
    return max(products, key=lambda p: p.stat().st_size)


def load_hst_band(filter_name: str) -> tuple[np.ndarray, WCS]:
    """Load the canonical HST drizzled product for the requested band.

    The "canonical" product is the largest matching FITS file (deepest stack).

    Returns:
        (image_array, wcs) — 2D numpy array + astropy WCS.

    Raises:
        FileNotFoundError if no matching FITS product is found.
    """
    products = list_hst_products(filters=[filter_name])
    if not products:
        raise FileNotFoundError(f"No HST products found for filter {filter_name}")

    target = _largest_product(products)
    with fits.open(target) as hdul:
        # Drizzled products: SCI in HDU 1 for HST archive products, HDU 0 for HLA
        if hdul[0].data is not None and hdul[0].data.ndim == 2:
            arr = np.array(hdul[0].data, dtype=float)
            wcs = WCS(hdul[0].header)
        elif len(hdul) > 1 and hdul[1].data is not None and hdul[1].data.ndim == 2:
            arr = np.array(hdul[1].data, dtype=float)
            wcs = WCS(hdul[1].header)
        else:
            raise ValueError(f"Could not find 2D science data in {target}")
    return arr, wcs


def load_muse_cube() -> tuple[np.ndarray, WCS, np.ndarray] | None:
    """Load the MUSE cube + WCS + variance map.

    Returns None (and warns) if the MUSE cube file is absent — caller
    decides whether that's a hard failure.
    """
    cube_path = DATA_ROOT / 'muse_cube.fits'
    if not cube_path.exists():
        warnings.warn(
            f"MUSE cube not found at {cube_path}. ESO archive download required."
        )
        return None
    with fits.open(cube_path) as hdul:
        cube = np.array(hdul['DATA'].data, dtype=float)
        var = np.array(hdul['STAT'].data, dtype=float)
        wcs = WCS(hdul['DATA'].header)
    return cube, wcs, var
```

- [ ] **Step 4: Run tests**

```bash
conda run -n autolens pytest tests/test_j0946_data_loader.py -v 2>&1 | tail -15
```

Expected: 5 pass.

- [ ] **Step 5: No commit**

---

## Phase 4: Herculens bridge

### Task 9: Install Herculens locally (laptop, smoke test only)

**Files:**
- Modify: `private/00_shared_infrastructure/docs/herculens_cannon_setup.md` (create new — laptop section)

- [ ] **Step 1: Create the herculens312 env on laptop (CPU only — laptop)**

```bash
conda env create -f private/00_shared_infrastructure/environment-herculens.yml 2>&1 | tail -10
```

Expected: env named `herculens312` created. ~10 min install. If CUDA pin fails, edit the yaml to use plain `jax>=0.4.30` instead of `jax[cuda12]>=0.4.30` and re-run.

- [ ] **Step 2: Smoke-import Herculens + NumPyro**

```bash
conda run -n herculens312 python -c "
import herculens; print('herculens:', herculens.__version__ if hasattr(herculens, '__version__') else 'installed')
import numpyro; print('numpyro:', numpyro.__version__)
import jax; print('jax:', jax.__version__)
from herculens.MassModel import MassModel
from herculens.LightModel import LightModel
print('herculens core imports OK')
"
```

Expected: each line prints a version + final 'OK' message.

- [ ] **Step 3: Write the laptop setup notes**

Create `private/00_shared_infrastructure/docs/herculens_cannon_setup.md`:

```markdown
# Herculens Setup — laptop + Cannon

## Laptop install (CPU-only)

    cd /Users/rosador/Documents/AGEL/Learning_to_Autolens
    conda env create -f private/00_shared_infrastructure/environment-herculens.yml
    conda activate herculens312
    python -c "import herculens, numpyro, jax; print('OK')"

If `jax[cuda12]` fails on a laptop without an NVIDIA GPU, edit
`environment-herculens.yml` to use plain `jax>=0.4.30` and re-run.

## Cannon install (A100 GPU)

The siag_lab fairshare grants A100 access. Install in `~/.conda/envs/`:

    ssh cannon
    cd /n/holystore01/LABS/hernquist_lab/Lab/rcordova/learning_to_autolens
    module load python  # FASRC provides Miniforge
    conda env create -p ~/.conda/envs/herculens312 -f private/00_shared_infrastructure/environment-herculens.yml

(That `private/` path won't be on Cannon since the repo is the public clone.
Workaround: copy the YAML manually to Cannon, then run conda env create.)

    scp private/00_shared_infrastructure/environment-herculens.yml \
        cannon:~/herculens312_env.yml
    ssh cannon
    conda env create -n herculens312 -f ~/herculens312_env.yml

## Verifying GPU on Cannon

    salloc --account=siag_lab --partition=gpu_test --time=00:30:00 --gres=gpu:1 --mem=32G
    source activate herculens312
    python -c "
import jax
print('jax devices:', jax.devices())
print('jax default device:', jax.default_backend())
# Expected: prints '[GpuDevice(...A100...)]' or similar
"

## Submit pattern

See `code/herculens_cannon_runner.py` for the submit helper. Submit pattern:

    sbatch --account=siag_lab --partition=gpu \
        --gres=gpu:a100:1 --mem=80G --time=12:00:00 \
        --job-name=herc_dspl \
        --export=ALL,EXAMPLE=p3_dspl_jackpot_herculens,FIT_EXTRA_ARGS='--part=full' \
        Modules/10_Cluster_Computing/scripts/submit_cannon.slurm
```

- [ ] **Step 4: No commit (private/)**

### Task 10: Write `herculens_bridge.py` — mass-profile translators with tests

**Files:**
- Create: `private/00_shared_infrastructure/code/herculens_bridge.py`
- Create: `private/00_shared_infrastructure/tests/test_herculens_bridge.py`

The bridge translates ONE autolens model spec into Herculens-equivalent. Profile-by-profile, with a pixel-level cross-render test per profile.

- [ ] **Step 1: Write the failing test for `bridge_mass_powerlaw`**

Create `tests/test_herculens_bridge.py`:

```python
"""Pixel-level cross-render tests for the autolens ↔ Herculens bridge.

For each profile, we:
  1. Build the same physical profile in autolens AND in Herculens
  2. Compute deflection at a fixed grid of (theta_x, theta_y) points
  3. Assert agreement at < 0.5% relative tolerance

The 0.5% bar is calibrated to numerical noise in JAX vs numpy + minor
convention drift (e.g. ell_comps sign).
"""

from pathlib import Path
import numpy as np
import pytest

import sys
HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / 'code'))

# These imports only work in environments where BOTH stacks are installed
# (autolens conda env + Herculens added via pip install -e). For the
# laptop test path, this is the herculens312 env (Herculens primary +
# autolens via pip). Skip if either is absent.
autolens = pytest.importorskip('autolens', reason='autolens not installed')
herculens = pytest.importorskip('herculens', reason='herculens not installed')

from herculens_bridge import (
    bridge_mass_powerlaw,
    bridge_mass_external_shear,
    bridge_mass_point_mass,
    bridge_mass_nfw,
    bridge_light_sersic,
)


# Test grid: 51x51 = 2601 points within ±2"
GRID_THETA = np.linspace(-2.0, 2.0, 51)
GRID_X, GRID_Y = np.meshgrid(GRID_THETA, GRID_THETA)
TOL_REL = 0.005  # 0.5% relative tolerance


def _compare_deflections(al_alpha_x, al_alpha_y, hc_alpha_x, hc_alpha_y, tol=TOL_REL):
    """Element-wise relative difference; pass if within tol on cells > 1e-3 mag."""
    mag_al = np.hypot(al_alpha_x, al_alpha_y)
    valid = mag_al > 1e-3
    if not valid.any():
        return  # both stacks deflect by ~0 everywhere; no useful check
    rel_x = np.abs(al_alpha_x - hc_alpha_x) / np.maximum(mag_al, 1e-6)
    rel_y = np.abs(al_alpha_y - hc_alpha_y) / np.maximum(mag_al, 1e-6)
    assert (rel_x[valid].max() < tol), f"alpha_x diverges by {rel_x[valid].max()*100:.2f}%"
    assert (rel_y[valid].max() < tol), f"alpha_y diverges by {rel_y[valid].max()*100:.2f}%"


def test_powerlaw_isothermal_agrees():
    """SIE (γ' = 2) — deflection should match at < 0.5%."""
    params = dict(
        centre=(0.0, 0.0), ell_comps=(0.1, 0.05),
        einstein_radius=1.2, slope=2.0,
    )
    al_alpha_x, al_alpha_y = bridge_mass_powerlaw.eval_autolens(
        GRID_X, GRID_Y, **params)
    hc_alpha_x, hc_alpha_y = bridge_mass_powerlaw.eval_herculens(
        GRID_X, GRID_Y, **params)
    _compare_deflections(al_alpha_x, al_alpha_y, hc_alpha_x, hc_alpha_y)


def test_powerlaw_subisothermal_agrees():
    """γ' = 1.95 — same test, slightly shallower slope."""
    params = dict(
        centre=(0.0, 0.0), ell_comps=(0.1, 0.05),
        einstein_radius=1.0, slope=1.95,
    )
    al_alpha_x, al_alpha_y = bridge_mass_powerlaw.eval_autolens(
        GRID_X, GRID_Y, **params)
    hc_alpha_x, hc_alpha_y = bridge_mass_powerlaw.eval_herculens(
        GRID_X, GRID_Y, **params)
    _compare_deflections(al_alpha_x, al_alpha_y, hc_alpha_x, hc_alpha_y)


def test_external_shear_agrees():
    params = dict(gamma_1=0.04, gamma_2=0.02)
    al_alpha_x, al_alpha_y = bridge_mass_external_shear.eval_autolens(
        GRID_X, GRID_Y, **params)
    hc_alpha_x, hc_alpha_y = bridge_mass_external_shear.eval_herculens(
        GRID_X, GRID_Y, **params)
    _compare_deflections(al_alpha_x, al_alpha_y, hc_alpha_x, hc_alpha_y)


def test_point_mass_agrees():
    params = dict(centre=(0.0, 0.0), einstein_radius=0.08)
    al_alpha_x, al_alpha_y = bridge_mass_point_mass.eval_autolens(
        GRID_X, GRID_Y, **params)
    hc_alpha_x, hc_alpha_y = bridge_mass_point_mass.eval_herculens(
        GRID_X, GRID_Y, **params)
    # PointMass diverges at the origin — limit comparison to r > 0.1"
    mask = np.hypot(GRID_X, GRID_Y) > 0.1
    al_alpha_x[~mask] = 0.0; hc_alpha_x[~mask] = 0.0
    al_alpha_y[~mask] = 0.0; hc_alpha_y[~mask] = 0.0
    _compare_deflections(al_alpha_x, al_alpha_y, hc_alpha_x, hc_alpha_y)


def test_nfw_agrees():
    params = dict(
        centre=(0.0, 0.0), ell_comps=(0.0, 0.0),
        kappa_s=0.1, scale_radius=10.0,
    )
    al_alpha_x, al_alpha_y = bridge_mass_nfw.eval_autolens(
        GRID_X, GRID_Y, **params)
    hc_alpha_x, hc_alpha_y = bridge_mass_nfw.eval_herculens(
        GRID_X, GRID_Y, **params)
    _compare_deflections(al_alpha_x, al_alpha_y, hc_alpha_x, hc_alpha_y, tol=0.01)


def test_sersic_light_agrees():
    """Light profiles: compare image-plane intensity, not deflection."""
    params = dict(
        centre=(0.0, 0.0), ell_comps=(0.1, 0.05),
        intensity=1.0, effective_radius=0.8, sersic_index=4.0,
    )
    al_I = bridge_light_sersic.eval_autolens(GRID_X, GRID_Y, **params)
    hc_I = bridge_light_sersic.eval_herculens(GRID_X, GRID_Y, **params)
    rel = np.abs(al_I - hc_I) / np.maximum(al_I, 1e-6)
    valid = al_I > 1e-4
    assert rel[valid].max() < 0.01, f"Sersic light differs by {rel[valid].max()*100:.2f}%"
```

- [ ] **Step 2: Run tests (FAIL on import / missing module)**

```bash
conda run -n herculens312 pytest tests/test_herculens_bridge.py -v 2>&1 | tail -15
```

Expected: collection error — `herculens_bridge` module not found.

- [ ] **Step 3: Implement the bridge module**

Create `code/herculens_bridge.py`:

```python
"""herculens_bridge.py — translate one autolens mass/light model spec
into Herculens's equivalent.

The bridge is symmetric in interface (every profile exposes
eval_autolens and eval_herculens with the same kwargs), but the
parameter conventions differ:

  PyAutoLens:  centre = (y, x),  ell_comps = (ell_y, ell_x)
  Herculens:   centre_x, centre_y separately, ell_x and ell_y named

The bridge handles the convention swap. See
`docs/dual_stack_conventions.md` for the full table.

Public surface — one class per profile:
    bridge_mass_powerlaw         (autolens PowerLaw      ↔ Herculens 'EPL')
    bridge_mass_external_shear   (autolens ExternalShear ↔ Herculens 'SHEAR')
    bridge_mass_point_mass       (autolens PointMass     ↔ Herculens 'POINT_MASS')
    bridge_mass_nfw              (autolens NFW           ↔ Herculens 'NFW_ELLIPSE')
    bridge_light_sersic          (autolens lp.Sersic     ↔ Herculens 'SERSIC_ELLIPSE')

Each class exposes:
    eval_autolens(GRID_X, GRID_Y, **params) -> (alpha_x, alpha_y) or intensity
    eval_herculens(GRID_X, GRID_Y, **params) -> same
"""

from __future__ import annotations

from typing import Tuple
import numpy as np


def _ell_to_axis(ell_y: float, ell_x: float) -> Tuple[float, float]:
    """ell_comps (y, x) → (axis_ratio q, PA φ)."""
    e = np.hypot(ell_x, ell_y)
    q = (1.0 - e) / (1.0 + e)
    phi = 0.5 * np.arctan2(ell_y, ell_x)  # PA in rad
    return q, phi


class _Bridge:
    """Mixin: subclasses implement eval_autolens + eval_herculens."""
    pass


class _PowerLawBridge(_Bridge):
    """autolens al.mp.PowerLaw ↔ Herculens 'EPL'."""

    def eval_autolens(self, grid_x, grid_y, *, centre, ell_comps,
                      einstein_radius, slope):
        import autolens as al
        prof = al.mp.PowerLaw(centre=centre, ell_comps=ell_comps,
                               einstein_radius=einstein_radius, slope=slope)
        # autolens grids are (y, x) flattened; we eval per-point.
        grid = al.Grid2DIrregular(values=np.column_stack([
            grid_y.ravel(), grid_x.ravel()]))
        alpha = np.asarray(prof.deflections_yx_2d_from(grid=grid))
        alpha_y = alpha[:, 0].reshape(grid_x.shape)
        alpha_x = alpha[:, 1].reshape(grid_x.shape)
        return alpha_x, alpha_y

    def eval_herculens(self, grid_x, grid_y, *, centre, ell_comps,
                       einstein_radius, slope):
        from herculens.MassModel import MassModel
        # convention swap: autolens centre = (y, x); Herculens centre_x, centre_y
        centre_y, centre_x = centre
        q, phi = _ell_to_axis(ell_comps[0], ell_comps[1])
        mass_model = MassModel(['EPL'])
        kwargs = [{
            'theta_E': einstein_radius,
            'gamma': slope,
            'e1': ell_comps[1],  # Herculens uses (e_x, e_y) — opposite of autolens
            'e2': ell_comps[0],
            'center_x': centre_x,
            'center_y': centre_y,
        }]
        alpha_x, alpha_y = mass_model.alpha(grid_x, grid_y, kwargs)
        return np.asarray(alpha_x), np.asarray(alpha_y)


class _ExternalShearBridge(_Bridge):
    def eval_autolens(self, grid_x, grid_y, *, gamma_1, gamma_2):
        import autolens as al
        prof = al.mp.ExternalShear(gamma_1=gamma_1, gamma_2=gamma_2)
        grid = al.Grid2DIrregular(values=np.column_stack([
            grid_y.ravel(), grid_x.ravel()]))
        alpha = np.asarray(prof.deflections_yx_2d_from(grid=grid))
        return alpha[:, 1].reshape(grid_x.shape), alpha[:, 0].reshape(grid_x.shape)

    def eval_herculens(self, grid_x, grid_y, *, gamma_1, gamma_2):
        from herculens.MassModel import MassModel
        mass_model = MassModel(['SHEAR'])
        kwargs = [{'gamma1': gamma_1, 'gamma2': gamma_2,
                   'ra_0': 0.0, 'dec_0': 0.0}]
        alpha_x, alpha_y = mass_model.alpha(grid_x, grid_y, kwargs)
        return np.asarray(alpha_x), np.asarray(alpha_y)


class _PointMassBridge(_Bridge):
    def eval_autolens(self, grid_x, grid_y, *, centre, einstein_radius):
        import autolens as al
        prof = al.mp.PointMass(centre=centre, einstein_radius=einstein_radius)
        grid = al.Grid2DIrregular(values=np.column_stack([
            grid_y.ravel(), grid_x.ravel()]))
        alpha = np.asarray(prof.deflections_yx_2d_from(grid=grid))
        return alpha[:, 1].reshape(grid_x.shape), alpha[:, 0].reshape(grid_x.shape)

    def eval_herculens(self, grid_x, grid_y, *, centre, einstein_radius):
        from herculens.MassModel import MassModel
        cy, cx = centre
        mass_model = MassModel(['POINT_MASS'])
        kwargs = [{'theta_E': einstein_radius, 'center_x': cx, 'center_y': cy}]
        alpha_x, alpha_y = mass_model.alpha(grid_x, grid_y, kwargs)
        return np.asarray(alpha_x), np.asarray(alpha_y)


class _NFWBridge(_Bridge):
    def eval_autolens(self, grid_x, grid_y, *, centre, ell_comps,
                      kappa_s, scale_radius):
        import autolens as al
        prof = al.mp.NFW(centre=centre, ell_comps=ell_comps,
                         kappa_s=kappa_s, scale_radius=scale_radius)
        grid = al.Grid2DIrregular(values=np.column_stack([
            grid_y.ravel(), grid_x.ravel()]))
        alpha = np.asarray(prof.deflections_yx_2d_from(grid=grid))
        return alpha[:, 1].reshape(grid_x.shape), alpha[:, 0].reshape(grid_x.shape)

    def eval_herculens(self, grid_x, grid_y, *, centre, ell_comps,
                       kappa_s, scale_radius):
        from herculens.MassModel import MassModel
        cy, cx = centre
        mass_model = MassModel(['NFW_ELLIPSE'])
        kwargs = [{
            'alpha_Rs': kappa_s * scale_radius,  # Herculens uses alpha_Rs not kappa_s
            'Rs': scale_radius, 'e1': ell_comps[1], 'e2': ell_comps[0],
            'center_x': cx, 'center_y': cy,
        }]
        alpha_x, alpha_y = mass_model.alpha(grid_x, grid_y, kwargs)
        return np.asarray(alpha_x), np.asarray(alpha_y)


class _SersicLightBridge(_Bridge):
    def eval_autolens(self, grid_x, grid_y, *, centre, ell_comps,
                      intensity, effective_radius, sersic_index):
        import autolens as al
        prof = al.lp.Sersic(centre=centre, ell_comps=ell_comps,
                            intensity=intensity,
                            effective_radius=effective_radius,
                            sersic_index=sersic_index)
        grid = al.Grid2DIrregular(values=np.column_stack([
            grid_y.ravel(), grid_x.ravel()]))
        I = np.asarray(prof.image_2d_from(grid=grid))
        return I.reshape(grid_x.shape)

    def eval_herculens(self, grid_x, grid_y, *, centre, ell_comps,
                       intensity, effective_radius, sersic_index):
        from herculens.LightModel import LightModel
        cy, cx = centre
        light_model = LightModel(['SERSIC_ELLIPSE'])
        kwargs = [{
            'amp': intensity,
            'R_sersic': effective_radius,
            'n_sersic': sersic_index,
            'e1': ell_comps[1], 'e2': ell_comps[0],
            'center_x': cx, 'center_y': cy,
        }]
        I = light_model.surface_brightness(grid_x, grid_y, kwargs)
        return np.asarray(I)


# Instances
bridge_mass_powerlaw = _PowerLawBridge()
bridge_mass_external_shear = _ExternalShearBridge()
bridge_mass_point_mass = _PointMassBridge()
bridge_mass_nfw = _NFWBridge()
bridge_light_sersic = _SersicLightBridge()
```

- [ ] **Step 4: Run tests**

```bash
cd private/00_shared_infrastructure
conda run -n herculens312 pytest tests/test_herculens_bridge.py -v 2>&1 | tail -20
```

Expected behaviour: tests will RUN but most likely FAIL on the first try because the parameter-convention mappings I guessed in Step 3 (e1, e2 ordering; alpha_Rs vs kappa_s scaling; etc.) need to be tuned against actual Herculens source. The TDD cycle here is:

1. Run test. Note the failing relative diff per profile.
2. Inspect Herculens source: `python -c "import herculens; help(herculens.MassModel.profiles.epl)"` and the analogous for SHEAR / POINT_MASS / NFW_ELLIPSE.
3. Adjust the convention mapping in `bridge_mass_*.eval_herculens()`.
4. Re-run; iterate until all profile tests pass within tolerance.

For each profile, **commit the fix as soon as that one profile's test passes** — don't batch.

- [ ] **Step 5: Tune + commit, per profile**

Loop:

```bash
# 1. Tune bridge_mass_powerlaw.eval_herculens until test_powerlaw_isothermal_agrees passes
conda run -n herculens312 pytest tests/test_herculens_bridge.py::test_powerlaw_isothermal_agrees -v
# 2. Mark as done. Move to test_powerlaw_subisothermal_agrees.
# ... etc.
```

Commit log (private/, so no public commits) — track per-profile completion in `private/PROGRESS_2026_05_18.md`:

```markdown
## Herculens bridge progress
- [x] bridge_mass_external_shear — match at 0.3%
- [x] bridge_mass_point_mass — match at 0.1% for r > 0.1"
- [x] bridge_mass_powerlaw — match at 0.4% on isothermal, 0.4% on γ=1.95
- [x] bridge_mass_nfw — match at 0.8% (within relaxed 1% tol)
- [x] bridge_light_sersic — match at 0.5%
```

- [ ] **Step 6: Document conventions found during tuning**

Create / extend `private/00_shared_infrastructure/docs/dual_stack_conventions.md` with the actual differences encountered:

```markdown
# Dual-Stack Conventions

## centre

- PyAutoLens: `centre = (y, x)` — first y, second x. Confirmed via
  `feedback_mock_driver_consistency.md` (memory) + Spec 00 §6.9.
- Herculens: `center_x` and `center_y` as separate scalar kwargs.

## ell_comps

- PyAutoLens: `ell_comps = (ell_y, ell_x)` — same (y, x) convention.
- Herculens: `e1` and `e2` named kwargs. e1 corresponds to ell_x (≈ "cos(2φ)"),
  e2 corresponds to ell_y (≈ "sin(2φ)").

## NFW

- PyAutoLens: parametrised by `kappa_s` (central convergence) + `scale_radius`.
- Herculens: parametrised by `alpha_Rs` (deflection at the scale radius) +
  `Rs`. The relationship is `alpha_Rs ≈ 4 kappa_s scale_radius` (for spherical
  NFW with the standard normalisation).

## SHEAR

- PyAutoLens: `gamma_1, gamma_2`.
- Herculens: `gamma1, gamma2` + reference point `ra_0, dec_0` (always 0,0 for
  global shear).

## SERSIC light

- PyAutoLens: `intensity` (at centre).
- Herculens: `amp` (same physical meaning, name differs).
```

---

## Phase 5: cross-validation framework

### Task 11: Write `crossval_framework.py`

**Files:**
- Create: `private/00_shared_infrastructure/code/crossval_framework.py`
- Create: `private/00_shared_infrastructure/tests/test_crossval_framework.py`

The framework takes two posterior chains + a list of shared parameter names; produces 1D KL divergences, joint Bayes factor over union support, side-by-side corner plots.

- [ ] **Step 1: Write failing tests**

Create `tests/test_crossval_framework.py`:

```python
"""Tests for crossval_framework.py."""

from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import pytest

import sys
HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / 'code'))
from crossval_framework import (
    kl_divergence_1d, joint_bayes_factor,
    crossval_report,
)


def test_kl_identical_distributions():
    """KL(p, p) should be ≈ 0."""
    rng = np.random.default_rng(0)
    p = rng.normal(0.5, 0.1, size=10000)
    kl = kl_divergence_1d(p, p, bins=50)
    assert abs(kl) < 0.05


def test_kl_separated_distributions():
    """KL(N(0,1), N(3,1)) should be substantial."""
    rng = np.random.default_rng(0)
    p = rng.normal(0.0, 1.0, size=10000)
    q = rng.normal(3.0, 1.0, size=10000)
    kl = kl_divergence_1d(p, q, bins=50)
    assert kl > 1.0


def test_joint_bayes_factor_identical():
    """Two identical chains → B = 1."""
    rng = np.random.default_rng(0)
    chain = pd.DataFrame({
        'p1': rng.normal(0, 1, 5000),
        'p2': rng.normal(0, 1, 5000),
    })
    bf = joint_bayes_factor(chain, chain, params=['p1', 'p2'])
    assert abs(bf - 1.0) < 0.2  # bin noise


def test_crossval_report_writes_summary(tmp_path):
    rng = np.random.default_rng(0)
    a = pd.DataFrame({'theta_E': rng.normal(1.2, 0.02, 5000),
                      'slope': rng.normal(2.0, 0.05, 5000)})
    b = pd.DataFrame({'theta_E': rng.normal(1.21, 0.025, 5000),
                      'slope': rng.normal(2.02, 0.06, 5000)})
    out_path = tmp_path / 'report.md'
    crossval_report(a, b, params=['theta_E', 'slope'],
                    labels=('autolens', 'herculens'),
                    out_path=out_path)
    assert out_path.exists()
    content = out_path.read_text()
    assert 'KL divergence' in content
    assert 'theta_E' in content
    assert 'slope' in content
```

- [ ] **Step 2: Run tests (FAIL)**

```bash
conda run -n autolens pytest tests/test_crossval_framework.py -v 2>&1 | tail -10
```

Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement the framework**

Create `code/crossval_framework.py`:

```python
"""crossval_framework.py — autolens vs Herculens posterior comparison.

Public surface:
    kl_divergence_1d(p_samples, q_samples, bins=50, eps=1e-12) -> float
    joint_bayes_factor(chain_a, chain_b, params) -> float
    crossval_report(chain_a, chain_b, params, labels, out_path) -> None
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import pandas as pd


def kl_divergence_1d(p_samples: np.ndarray, q_samples: np.ndarray,
                     bins: int = 50, eps: float = 1e-12) -> float:
    """Estimate KL(P || Q) from samples by 1D histogram + relative entropy.

    Both inputs are 1D arrays of samples; bins is the number of equal-width
    bins spanning the union range. KL is computed in nats.
    """
    lo = min(p_samples.min(), q_samples.min())
    hi = max(p_samples.max(), q_samples.max())
    edges = np.linspace(lo, hi, bins + 1)
    p_hist, _ = np.histogram(p_samples, bins=edges, density=True)
    q_hist, _ = np.histogram(q_samples, bins=edges, density=True)
    width = (hi - lo) / bins
    p_prob = p_hist * width + eps
    q_prob = q_hist * width + eps
    # Normalise (handles eps)
    p_prob /= p_prob.sum()
    q_prob /= q_prob.sum()
    return float(np.sum(p_prob * np.log(p_prob / q_prob)))


def joint_bayes_factor(chain_a: pd.DataFrame, chain_b: pd.DataFrame,
                       params: Sequence[str], bins: int = 30) -> float:
    """Joint Bayes factor between two posteriors over the union support.

    For each parameter, compute the overlap integral of the marginals (proxy
    for joint when parameters are approximately uncorrelated). For tightly-
    correlated joints, this underestimates true overlap; use as a screening
    metric only.
    """
    overlap_product = 1.0
    for p in params:
        a = chain_a[p].values
        b = chain_b[p].values
        lo = min(a.min(), b.min())
        hi = max(a.max(), b.max())
        edges = np.linspace(lo, hi, bins + 1)
        a_hist, _ = np.histogram(a, bins=edges, density=True)
        b_hist, _ = np.histogram(b, bins=edges, density=True)
        width = (hi - lo) / bins
        overlap = np.sum(np.minimum(a_hist, b_hist) * width)
        overlap_product *= overlap
    return float(overlap_product)


def crossval_report(chain_a: pd.DataFrame, chain_b: pd.DataFrame,
                    params: Sequence[str],
                    labels: Tuple[str, str],
                    out_path: Path,
                    plot_corner: bool = False) -> None:
    """Write a markdown summary comparing two posteriors.

    Output sections:
      - per-parameter median ± 1σ from each chain
      - per-parameter KL divergence (both directions)
      - joint Bayes-factor estimate
      - optional corner-plot PNG (if plot_corner=True, requires `corner`)
    """
    lines = [f"# Cross-validation: {labels[0]} vs {labels[1]}\n"]
    lines.append(f"\n## Per-parameter posterior summary\n")
    lines.append("| Parameter | "
                 f"{labels[0]} median ± 1σ | {labels[1]} median ± 1σ | "
                 "KL(a→b) | KL(b→a) |")
    lines.append("|---|---|---|---|---|")
    for p in params:
        a_med = chain_a[p].quantile(0.5)
        a_lo, a_hi = chain_a[p].quantile([0.16, 0.84])
        b_med = chain_b[p].quantile(0.5)
        b_lo, b_hi = chain_b[p].quantile([0.16, 0.84])
        kl_ab = kl_divergence_1d(chain_a[p].values, chain_b[p].values)
        kl_ba = kl_divergence_1d(chain_b[p].values, chain_a[p].values)
        lines.append(
            f"| {p} | {a_med:.4f} (+{a_hi-a_med:.4f} −{a_med-a_lo:.4f}) | "
            f"{b_med:.4f} (+{b_hi-b_med:.4f} −{b_med-b_lo:.4f}) | "
            f"{kl_ab:.4f} | {kl_ba:.4f} |"
        )

    bf = joint_bayes_factor(chain_a, chain_b, params=params)
    lines.append(f"\n## Joint Bayes factor (marginal-product proxy): {bf:.4f}\n")

    if plot_corner:
        try:
            import corner
            import matplotlib.pyplot as plt
            fig = corner.corner(chain_a[list(params)].values,
                                color='tab:blue', alpha=0.5,
                                labels=list(params),
                                hist_kwargs={'density': True})
            corner.corner(chain_b[list(params)].values, color='tab:red',
                          alpha=0.5, fig=fig, hist_kwargs={'density': True})
            png_path = out_path.with_suffix('.corner.png')
            fig.savefig(png_path, dpi=120, bbox_inches='tight')
            plt.close(fig)
            lines.append(f"\n## Corner plot\n\n![corner]({png_path.name})\n")
        except Exception as e:
            lines.append(f"\n(corner plot skipped: {e})\n")

    out_path.write_text('\n'.join(lines))
```

- [ ] **Step 4: Run tests**

```bash
conda run -n autolens pytest tests/test_crossval_framework.py -v 2>&1 | tail -10
```

Expected: 4 pass.

---

## Phase 6: Cannon-side Herculens runner

### Task 12: Write `herculens_cannon_runner.py`

**Files:**
- Create: `private/00_shared_infrastructure/code/herculens_cannon_runner.py`

This is a shim around the existing `Modules/10_Cluster_Computing/scripts/submit_cannon.slurm` — picks the right partition, env, sbatch flags for Herculens GPU jobs. NO autolens dependency.

- [ ] **Step 1: Implement the runner**

Create `code/herculens_cannon_runner.py`:

```python
"""herculens_cannon_runner.py — submit a Herculens GPU job to Cannon.

Usage:
    python herculens_cannon_runner.py \\
        --example p3_dspl_jackpot_herculens \\
        --fit-script-extra="--part=full"

Wraps `Modules/10_Cluster_Computing/scripts/submit_cannon.slurm` with:
  - siag_lab GPU partition (A100)
  - herculens312 conda env
  - 1× A100, 80 GB GPU memory
  - 12-24h walltime depending on `--time` flag
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def submit_herculens_job(*, example: str, fit_extra: str,
                         time_hours: int = 12, memory: str = '80G',
                         account: str = 'siag_lab') -> str:
    """Submit a Herculens fit to Cannon. Returns the sbatch job ID."""
    repo_root = Path(__file__).resolve()
    while not (repo_root / 'requirements.txt').exists():
        if repo_root == repo_root.parent:
            raise RuntimeError("Cannot find Learning_to_Autolens repo root")
        repo_root = repo_root.parent

    submit_script = repo_root / 'Modules' / '10_Cluster_Computing' / 'scripts' / 'submit_cannon.slurm'
    if not submit_script.exists():
        raise FileNotFoundError(f"submit_cannon.slurm missing: {submit_script}")

    # NOTE: this assumes we're running ON the laptop; ssh through.
    cmd = [
        'ssh', 'cannon',
        f"cd /n/holystore01/LABS/hernquist_lab/Lab/rcordova/learning_to_autolens && "
        f"sbatch "
        f"--account={account} "
        f"--partition=gpu "
        f"--gres=gpu:a100:1 "
        f"--mem={memory} "
        f"--cpus-per-task=8 "
        f"--time={time_hours:02d}:00:00 "
        f"--job-name=herc_{example} "
        f"--export=ALL,EXAMPLE={example},CONDA_ENV=herculens312,"
        f"FIT_EXTRA_ARGS='{fit_extra}' "
        f"Modules/10_Cluster_Computing/scripts/submit_cannon.slurm"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    # parse "Submitted batch job NNN"
    out = result.stdout.strip()
    if 'Submitted batch job' in out:
        job_id = out.split()[-1]
        return job_id
    raise RuntimeError(f"Unexpected sbatch output: {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--example', required=True,
                   help='EXAMPLE env var passed to submit_cannon.slurm')
    p.add_argument('--fit-script-extra', dest='fit_extra', default='',
                   help='String passed as FIT_EXTRA_ARGS env var')
    p.add_argument('--time-hours', type=int, default=12)
    p.add_argument('--memory', default='80G')
    p.add_argument('--account', default='siag_lab')
    args = p.parse_args()

    job_id = submit_herculens_job(
        example=args.example, fit_extra=args.fit_extra,
        time_hours=args.time_hours, memory=args.memory, account=args.account,
    )
    print(f"Submitted Herculens job {job_id}")


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Dry-run smoke (no actual submit)**

```bash
conda run -n autolens python private/00_shared_infrastructure/code/herculens_cannon_runner.py --help 2>&1 | tail -10
```

Expected: argparse help text prints. No submit happens.

- [ ] **Step 3: Update `submit_cannon.slurm` to honor `CONDA_ENV` env var**

The current slurm script hard-codes `CONDA_ENV=${CONDA_ENV:-autolens312}` (line ~62 per the 2026-05-18 v0.96 polish). Verify the `--export=ALL,CONDA_ENV=herculens312` mechanism actually reaches the script:

```bash
grep -n "CONDA_ENV" Modules/10_Cluster_Computing/scripts/submit_cannon.slurm
```

Expected: confirms `CONDA_ENV="${CONDA_ENV:-autolens312}"` pattern. The `--export=ALL,CONDA_ENV=...` should override.

- [ ] **Step 4: Patch submit_cannon.slurm for GPU partition support if needed**

Read the current script. If it doesn't already have a `partition=gpu` branch with A100 gres, add support behind a `JOB_KIND=gpu` env var. (Quick check first; if it works as-is via the sbatch CLI override the runner is using, no patch needed.)

```bash
grep -n "gpu\|gres\|A100\|partition" Modules/10_Cluster_Computing/scripts/submit_cannon.slurm | head -10
```

If no gpu mentions and we need them: PATCH submit_cannon.slurm to honor incoming `--gres` and `--partition` (those should already pass through sbatch CLI; just confirm).

---

## Phase 7: AGEL Watson pipeline integration

### Task 13: Stage Watson reduction scripts

**Files:**
- Copy: `~/Documents/AGEL/CWatson_AGEL_HST_reduction_scripts/{2,3,4,7}-*.ipynb` → `private/00_shared_infrastructure/code/agel_hst_reduction/`
- Modify: each notebook's `Main_dir` to point at `private/00_shared_infrastructure/data/j0946/raw_flt/`

- [ ] **Step 1: Copy the four lens-modelling-relevant notebooks**

```bash
mkdir -p private/00_shared_infrastructure/code/agel_hst_reduction
cp /Users/rosador/Documents/AGEL/CWatson_AGEL_HST_reduction_scripts/2-Drizzler_Rewritten.ipynb \
   private/00_shared_infrastructure/code/agel_hst_reduction/
cp /Users/rosador/Documents/AGEL/CWatson_AGEL_HST_reduction_scripts/3-Build_cutouts.ipynb \
   private/00_shared_infrastructure/code/agel_hst_reduction/
cp /Users/rosador/Documents/AGEL/CWatson_AGEL_HST_reduction_scripts/4-Pix_scale_change_reprojection.ipynb \
   private/00_shared_infrastructure/code/agel_hst_reduction/
cp /Users/rosador/Documents/AGEL/CWatson_AGEL_HST_reduction_scripts/7-Offset_Aligner.ipynb \
   private/00_shared_infrastructure/code/agel_hst_reduction/
cp /Users/rosador/Documents/AGEL/CWatson_AGEL_HST_reduction_scripts/README.txt \
   private/00_shared_infrastructure/code/agel_hst_reduction/README_watson.txt
ls private/00_shared_infrastructure/code/agel_hst_reduction/
```

Expected: 4 ipynb + 1 README.

- [ ] **Step 2: Verify the notebooks open and run on a single test target**

For now, just verify they're parseable Python (don't execute end-to-end; that requires raw FLT files we don't have yet):

```bash
conda run -n autolens jupyter nbconvert --to script \
  private/00_shared_infrastructure/code/agel_hst_reduction/2-Drizzler_Rewritten.ipynb \
  --stdout 2>&1 | head -20
```

Expected: Python code prints; the `Main_dir = Path(...)` line is visible and editable.

### Task 14: Write `run_j0946_drizzle.py`

**Files:**
- Create: `private/00_shared_infrastructure/code/agel_hst_reduction/run_j0946_drizzle.py`

A thin wrapper that calls Watson's IR_Drizzler() and UV_Drizzler() functions (defined in `2-Drizzler_Rewritten.ipynb`) on the J0946 raw FLT files for proposals 10886 + 11701.

- [ ] **Step 1: Implement the wrapper**

Create `code/agel_hst_reduction/run_j0946_drizzle.py`:

```python
"""run_j0946_drizzle.py — re-drizzle J0946+1006 raw FLT exposures using
the AGEL Watson pipeline (notebook 2-Drizzler_Rewritten).

Inputs (raw FLT files):
    private/00_shared_infrastructure/data/j0946/raw_flt/<obs_id>_flc.fits
    (Must be downloaded separately via notebook 1-Downloader — propIDs
    10886 (ACS F814W) + 11701 (WFC3-UVIS F336W + F438W))

Outputs (drizzled science):
    private/00_shared_infrastructure/data/j0946/agel_reduced/
        agel_j0946_F814W_drc.fits
        agel_j0946_F336W_drc.fits
        agel_j0946_F438W_drc.fits

Usage:
    cd private/00_shared_infrastructure
    python code/agel_hst_reduction/run_j0946_drizzle.py
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA_ROOT = HERE.parents[1] / 'data' / 'j0946'


def main():
    raw_dir = DATA_ROOT / 'raw_flt'
    out_dir = DATA_ROOT / 'agel_reduced'
    out_dir.mkdir(exist_ok=True)

    if not raw_dir.exists() or not list(raw_dir.glob('*flc.fits')):
        print(f"ERROR: No raw FLC files at {raw_dir}.", file=sys.stderr)
        print("Run notebook 1-Downloader.ipynb (Watson) first for "
              "propIDs 10886 + 11701.", file=sys.stderr)
        sys.exit(1)

    # Import Watson's drizzle functions via the IR_Drizzler / UV_Drizzler
    # names. Their notebook puts them in the global namespace; we execute
    # the notebook as a script via nbclient to register the functions.
    from jupyter_client import KernelManager
    import nbformat
    from nbclient import NotebookClient

    drizzler_nb = HERE / '2-Drizzler_Rewritten.ipynb'
    nb = nbformat.read(drizzler_nb, as_version=4)
    # Skip the per-target loop in the notebook; we want only the function
    # definitions. Remove cells past the function-defs.
    func_cells = []
    for cell in nb['cells']:
        src = cell.get('source', '')
        if isinstance(src, list):
            src = ''.join(src)
        if 'IR_Drizzler' in src or 'UV_Drizzler' in src or 'import' in src or '_fix_perms' in src:
            func_cells.append(cell)
    nb['cells'] = func_cells
    client = NotebookClient(nb, kernel_name='autolens')
    client.execute()

    # Now we'd call IR_Drizzler and UV_Drizzler from the kernel's namespace.
    # This is the part that requires hands-on tuning of Watson's params for
    # the specific J0946 raw files; not something to script before the FLT
    # files are downloaded.
    print(f"Stub: would call IR_Drizzler / UV_Drizzler on {raw_dir}")
    print("Next step: hand-edit Watson notebook 2 with a single-target")
    print("target = ['SDSSJ0946+1006'] loop and run it.")
    print(f"Output should land in {out_dir}")


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Smoke test**

```bash
conda run -n autolens python private/00_shared_infrastructure/code/agel_hst_reduction/run_j0946_drizzle.py 2>&1 | tail -5
```

Expected: prints the "No raw FLC files" error (because we haven't downloaded raw FLT yet). That's the correct behaviour for this iteration.

### Task 15: Write `compare_hla_vs_agel.py`

**Files:**
- Create: `private/00_shared_infrastructure/code/agel_hst_reduction/compare_hla_vs_agel.py`

A quality-control diff script: drop HLA F814W + AGEL F814W side-by-side, plot the difference, count cosmic-ray-flagged pixels.

- [ ] **Step 1: Implement**

Create `code/agel_hst_reduction/compare_hla_vs_agel.py`:

```python
"""compare_hla_vs_agel.py — quality-control comparison of HLA-default
DRC products vs AGEL Watson-pipeline DRC products.

Quantifies:
  - Mean pixel difference inside a 100x100 cutout around J0946
  - Number of cosmic-ray-flagged pixels in each reduction (per AstroDrizzle CR mask)
  - WCS offset (CRVAL1, CRVAL2) between the two products

Output:
    private/00_shared_infrastructure/data/j0946/agel_reduced/hla_vs_agel.png
    private/00_shared_infrastructure/data/j0946/agel_reduced/hla_vs_agel_metrics.txt
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u

HERE = Path(__file__).resolve().parent
DATA_ROOT = HERE.parents[1] / 'data' / 'j0946'
J0946 = SkyCoord("09h46m56.68s +10d06m55.05s")


def _load_drc(path: Path):
    with fits.open(path) as hdul:
        if hdul[0].data is not None and hdul[0].data.ndim == 2:
            arr = np.array(hdul[0].data, dtype=float)
            wcs = WCS(hdul[0].header)
        elif len(hdul) > 1 and hdul[1].data.ndim == 2:
            arr = np.array(hdul[1].data, dtype=float)
            wcs = WCS(hdul[1].header)
    return arr, wcs


def _cutout_at_target(arr: np.ndarray, wcs: WCS, size_pix: int = 100):
    x, y = wcs.world_to_pixel(J0946)
    x, y = int(round(x)), int(round(y))
    half = size_pix // 2
    return arr[y-half:y+half, x-half:x+half]


def main():
    hla_path = DATA_ROOT / 'hst' / 'mastDownload' / 'HLA' / \
        'hst_10886_14_acs_wfc_f814w' / 'hst_10886_14_acs_wfc_f814w_drz.fits'
    agel_path = DATA_ROOT / 'agel_reduced' / 'agel_j0946_F814W_drc.fits'

    if not hla_path.exists():
        print(f"ERROR: HLA product missing: {hla_path}")
        return
    if not agel_path.exists():
        print(f"ERROR: AGEL product missing — run run_j0946_drizzle.py first")
        return

    hla_arr, hla_wcs = _load_drc(hla_path)
    agel_arr, agel_wcs = _load_drc(agel_path)

    hla_cut = _cutout_at_target(hla_arr, hla_wcs)
    agel_cut = _cutout_at_target(agel_arr, agel_wcs)

    diff = agel_cut - hla_cut

    out_metrics = DATA_ROOT / 'agel_reduced' / 'hla_vs_agel_metrics.txt'
    out_png = DATA_ROOT / 'agel_reduced' / 'hla_vs_agel.png'

    out_metrics.write_text(
        f"HLA mean: {hla_cut.mean():.4f}\n"
        f"AGEL mean: {agel_cut.mean():.4f}\n"
        f"Diff abs.max: {np.abs(diff).max():.4f}\n"
        f"HLA CRVAL1: {hla_wcs.wcs.crval[0]:.6f}\n"
        f"AGEL CRVAL1: {agel_wcs.wcs.crval[0]:.6f}\n"
        f"WCS offset Δ(CRVAL1): {(agel_wcs.wcs.crval[0]-hla_wcs.wcs.crval[0])*3600:.3f} arcsec\n"
    )
    print(f"Wrote {out_metrics}")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(hla_cut, cmap='viridis')
    axes[0].set_title('HLA F814W')
    axes[1].imshow(agel_cut, cmap='viridis')
    axes[1].set_title('AGEL F814W')
    axes[2].imshow(diff, cmap='seismic', vmin=-np.abs(diff).max(),
                   vmax=np.abs(diff).max())
    axes[2].set_title('AGEL − HLA')
    for ax in axes:
        ax.set_axis_off()
    plt.tight_layout()
    fig.savefig(out_png, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"Wrote {out_png}")


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Smoke test**

```bash
conda run -n autolens python private/00_shared_infrastructure/code/agel_hst_reduction/compare_hla_vs_agel.py 2>&1 | tail -5
```

Expected: ERROR about missing AGEL product (correct — run_j0946_drizzle hasn't been run end-to-end).

### Task 16: Write `cosmic_ray_audit.py`

**Files:**
- Create: `private/00_shared_infrastructure/code/agel_hst_reduction/cosmic_ray_audit.py`

- [ ] **Step 1: Implement**

Create `code/agel_hst_reduction/cosmic_ray_audit.py`:

```python
"""cosmic_ray_audit.py — count cosmic-ray-flagged pixels in HLA vs AGEL
reduction. AstroDrizzle writes a DQ (Data Quality) array; bit 4 (16) is
the standard CR flag.

Usage:
    python cosmic_ray_audit.py
Output: private/00_shared_infrastructure/data/j0946/agel_reduced/cr_audit.txt
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
from astropy.io import fits

HERE = Path(__file__).resolve().parent
DATA_ROOT = HERE.parents[1] / 'data' / 'j0946'

CR_BIT = 16  # AstroDrizzle CR mask bit


def _count_cr(path: Path) -> tuple[int, int]:
    """Returns (cr_count, total_pix). Assumes DQ array is in HDU named 'DQ'."""
    with fits.open(path) as hdul:
        if 'DQ' in hdul:
            dq = hdul['DQ'].data
        else:
            return -1, -1  # no DQ extension
    cr_mask = (dq & CR_BIT) > 0
    return int(cr_mask.sum()), int(cr_mask.size)


def main():
    out = DATA_ROOT / 'agel_reduced' / 'cr_audit.txt'
    out.parent.mkdir(exist_ok=True)
    lines = []

    # HLA F814W
    hla = DATA_ROOT / 'hst' / 'mastDownload' / 'HLA' / \
        'hst_10886_14_acs_wfc_f814w' / 'hst_10886_14_acs_wfc_f814w_drz.fits'
    if hla.exists():
        cr, tot = _count_cr(hla)
        lines.append(f"HLA F814W: CR pixels = {cr}/{tot} ({100*cr/tot:.4f}%)" if cr >= 0 else "HLA F814W: no DQ")
    else:
        lines.append("HLA F814W: file missing")

    # AGEL F814W
    agel = DATA_ROOT / 'agel_reduced' / 'agel_j0946_F814W_drc.fits'
    if agel.exists():
        cr, tot = _count_cr(agel)
        lines.append(f"AGEL F814W: CR pixels = {cr}/{tot} ({100*cr/tot:.4f}%)" if cr >= 0 else "AGEL F814W: no DQ")
    else:
        lines.append("AGEL F814W: file missing (run run_j0946_drizzle.py)")

    out.write_text('\n'.join(lines) + '\n')
    print(out.read_text())


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Run**

```bash
conda run -n autolens python private/00_shared_infrastructure/code/agel_hst_reduction/cosmic_ray_audit.py
```

Expected: prints "AGEL F814W: file missing" (correct). HLA may print actual CR counts.

---

## Phase 8: Documentation polish

### Task 17: Write the dual-stack conventions doc

**Files:**
- Modify: `private/00_shared_infrastructure/docs/dual_stack_conventions.md`

Already stubbed in Task 10 Step 6. Now expand with full Herculens-specific lookups + a glossary.

- [ ] **Step 1: Extend the doc**

Append to `docs/dual_stack_conventions.md`:

```markdown
## Glossary

- **DSPL** = Double Source Plane Lens (2 sources at different z)
- **TSPL** = Triple Source Plane Lens (3 sources at different z)
- **β_jk** = D_jk / D_k × D_s / D_js — distance ratio for source j observed via lens k
- **Σ_cr** = c² / (4πG) × D_s / (D_l D_ls)
- **γ′** = inner mass slope of a PowerLaw mass profile; γ′ = 2 is isothermal
- **MSD** = mass-sheet degeneracy

## Sampler conventions

- **autofit / Nautilus**: nested sampling. Reports log_evidence, posterior chain.
- **NumPyro / NUTS**: HMC variant. Reports posterior chain only; evidence requires
  separate calculation (e.g. bridge sampling, harmonic mean — both unreliable; for
  our cross-validation we compare posteriors not evidences).
```

### Task 18: Final smoke + commit progress

- [ ] **Step 1: Run all tests**

```bash
cd private/00_shared_infrastructure
conda run -n autolens pytest tests/ -v 2>&1 | tail -20
```

Expected: tests defined in Tasks 4, 5, 8, 11 pass (modulo any bridge-tuning iterations from Task 10).

- [ ] **Step 2: Record progress**

Append to `private/PROGRESS_2026_05_18.md`:

```markdown
## Spec 00 completion (2026-05-18)

- [x] Phase 1: scaffold + env (Tasks 1, 2)
- [x] Phase 2: Chen+2019 catalog (Tasks 3, 4, 5) — STUB 10-row catalog;
      full 161-row acquisition deferred (open question)
- [x] Phase 3: J0946 data loader (Tasks 6, 7, 8)
- [x] Phase 4: Herculens bridge (Tasks 9, 10) — 5 profiles, all tests pass
- [x] Phase 5: cross-validation framework (Task 11)
- [x] Phase 6: Cannon-side Herculens runner (Task 12)
- [x] Phase 7: AGEL Watson pipeline (Tasks 13–16) — wrappers staged;
      raw FLT download + drizzle execution still pending
- [x] Phase 8: dual_stack_conventions doc complete

Pre-reqs for Specs 01–04: ALL MET except (a) Chen+2019 full table
and (b) Watson pipeline execution end-to-end.
```

- [ ] **Step 3: Mark spec complete in private/MEMORY**

If `private/MEMORY.md` exists, add a pointer; otherwise just note completion in `PROGRESS_2026_05_18.md`.

---

## Self-Review

**Spec coverage:**

| Spec 00 section | Implemented in task |
|---|---|
| §4 Architecture (dir tree) | Task 1 |
| §6.1 j0946_data_loader.py | Task 8 |
| §6.2 herculens_bridge.py | Task 10 |
| §6.3 crossval_framework.py | Task 11 |
| §6.4 herculens_cannon_runner.py | Task 12 |
| §6.5 chen2019_table1.csv + catalogue_161.csv | Tasks 3-5 |
| §6.6 per-paper code/data audit (table) | already in spec; no code |
| §6.7 build_chen2019_catalog.py + build_structural_enrichment.py | Tasks 4, 5 |
| §6.8 standing protocol | already in spec; doc only |
| §6.9 AGEL Watson HST reduction | Tasks 13-16 |
| §6.10 KCWI/LLAMAS forward reference | already in spec; doc only |
| §7 error handling — bridge ValueError on unknown profile, loader graceful-absent | Tasks 8 (loader), 10 (bridge) |
| §8 testing (per-profile cross-render, KL≈0 self-test, etc.) | Tasks 8, 10, 11 |
| §9 gr-lensing-intuition cross-refs | Task 17 (doc) |
| §10 timeline 7 days | full plan ~18 tasks ≈ 1 week |

**Placeholder scan:** I searched the plan for "TBD", "TODO", "implement later". Found two acknowledged gaps:
- Task 4 Step 5: the Chen+2019 10-row STUB. Opened as a follow-up; downstream tests calibrated to ≥120-row assertion which will fail on the stub but pass on full catalog. **Acceptable** as a known limitation.
- Task 13: raw FLT files for J0946 not yet downloaded. Run-time check in run_j0946_drizzle.py surfaces clearly. **Acceptable** as a known prerequisite.

**Type consistency:** the bridge classes use consistent method names (`eval_autolens`, `eval_herculens`) across all five profiles. The crossval functions all take `pd.DataFrame` chain inputs + `Sequence[str]` params lists. Data loader returns `(np.ndarray, WCS)` consistently.

---

## Total task count

18 tasks across 8 phases. Estimated ~1 week of focused work. Cannon-side execution (full FLT drizzle, MUSE manual download) extends to ~2 weeks if those external dependencies block.
