# Example: Galaxy–Galaxy Single Arc (SLACS-style)

## Status

◐ **In progress** — mock + driver + Cannon dispatcher wired (2026-05-11); audited fit pending Cannon submit. This is the **canonical first audited fit** in the curriculum — a new lens modeler's starting point before any compound, DSPL, or multi-plane example.

## Why this example

Every other example in this collection starts with a compound system (`compound_lens`), a DSPL (`double_source_plane`), a multi-plane TDCOSMO setup (`quad_time_delay`), or real-data complexity (`agel_real_target`). A new modeler going straight from Module 03's toy mocks to those skips the most-cited geometry in the strong-lensing literature: **one deflector, one source, one Einstein ring/arc** — the Bolton+08 / Auger+10 / Sonnenfeld+13 SLACS setup.

This example is that setup. One Isothermal+shear deflector at z=0.3, one Sersic source at z=1.0, ~1.2″ Einstein radius — exactly the geometry of every SLACS lens paper plus most of AGEL DR1/2.

## Mock geometry

`mocks/generate_mock.py` renders the mock natively in autolens (no lenstronomy round-trip, so no framework-Sersic mismatch as in v0.92 `mge_to_physical`). Truth values in `mocks/truths.json`. Self-consistency check at the end of the generator: **chi²/N at truth = 0.962, max|res| = 3.84σ** — well within the noise floor.

| Quantity | Value |
|---|---|
| z_lens | 0.3 |
| z_source | 1.0 |
| Lens mass: `Isothermal` | `theta_E=1.2"`, `ell_comps=(0.10, 0.05)`, centre=(0, 0) |
| Lens shear: `ExternalShear` | `gamma_1=0.025`, `gamma_2=-0.015` |
| Lens light: `Sersic` (de Vaucouleurs) | n=4, R_e=1.0″, intensity=1.0, ell_comps=(0.10, 0.05) |
| Source: `Sersic` (Sb disc) | n=1, R_e=0.15″, intensity=0.35, centre=(0.12, -0.08) |
| Cosmology | FlatLambdaCDM(70, 0.30) |
| Imaging | 60×60 px @ 0.05″/px, HST WFC3-IR-like PSF (FWHM 0.12″) |
| Exposure / sky RMS | 1000 s / 0.01 e⁻/s |

## Method

Single Nautilus search, no chaining (`--part=direct`):

```python
lens   = Galaxy(redshift=0.3,
                bulge=lp.Sersic(...),       # 5 free
                mass =mp.Isothermal(...),    # 3 free (centre tied to bulge)
                shear=mp.ExternalShear(...)) # 2 free
source = Galaxy(redshift=1.0,
                bulge=lp.Sersic(...))        # 7 free
```

12 free parameters, n_live=150, wide uninformative priors. Wall ≈ 1–2 h on 32 cores.

A second variant `--part=truth_anchored` uses tight Gaussian priors centred on the mock truth values — used to establish the chi²-at-truth diagnostic baseline (per Module 11 §3 methodology).

## Notebook

`01_galaxy_galaxy_single_arc.ipynb` — pedagogical walkthrough planned: data load → mask → model setup → priors → analysis → search → result audit using Module 11's 6-panel residual walk.

## Running on Cannon

```bash
sbatch --account=siag_lab --partition=siag \
       --time=4:00:00 --mem=64G --cpus-per-task=32 \
       --job-name=ggsa_direct \
       --export=ALL,EXAMPLE=galaxy_galaxy_single_arc,FIT_EXTRA_ARGS='--part=direct' \
       Modules/10_Cluster_Computing/scripts/submit_cannon.slurm
```

(Drop the `--account` / `--partition` flags to fall back to the slurm script's defaults.)

## Exercises

1. **Recover Einstein radius.** Compare the posterior on `mass.einstein_radius` to truth (1.2″). Should bracket within 1σ.
2. **Effect of shear marginalisation.** Run with `shear` fixed at (0, 0). What happens to the lens mass posterior? Demonstrates shear's role as a marginalisation tool.
3. **Mass-follows-light coupling.** Free the mass centre vs. tied to bulge centre. The fit should pick mass-follows-light from the data alone — but with what σ?
4. **Source size / Einstein radius degeneracy.** Plot the joint posterior on `(source.effective_radius, lens.einstein_radius)` — there's a classic degeneracy here.

## References

- **Bolton et al. (2008)**, ApJ 682, 964 — SLACS sample and the canonical single-Sersic-source fit.
- **Auger et al. (2010)**, ApJ 724, 511 — SLACS structural decomposition; basis for Module 11's f_DM and γ′ recovery.
- **Sonnenfeld et al. (2013)**, ApJ 777, 98 — SL2S sample, mass-to-light evolution.
- `autolens_workspace_latest/scripts/imaging/modeling/start_here.py` — canonical autolens example for the same geometry.

## What this example DOESN'T cover

- **Multiple sources / multiple deflectors** — see `compound_lens`, `compound_lens_zoo`, `double_source_plane`.
- **Time delays** — see `quad_time_delay`.
- **Pixelized source reconstruction** — see Module 05 + `compound_lens/02_compound_slam.ipynb`.
- **MGE lens light** — see Module 09 + `mge_to_physical`.
- **Physical mass decomposition (stars + dark)** — see `mge_to_physical` + Module 11.
- **Real data complications** — see `agel_real_target` (HST cutout, empirical PSF, neighbor masking).
