# Cluster scripts — where the SLaM wrappers live

If you're adapting the SLaM workflow (`04_search_chaining_slam.ipynb` /
`fit_module04.py`) and looking for the wrapper functions —
`source_lp.run`, `source_pix.run_1`, `source_pix.run_2`, `light_lp.run`,
`mass_total.run` — they are **NOT in this `scripts/` directory**.

They live in a single file at the **repository root**:

```
Learning_to_Autolens/slam_v2026.py
```

(The `slam_v2026.cpython-3xx.pyc` you may find under `__pycache__/` is just the
compiled cache — gitignored, don't read it. `git pull` if you don't see the
`.py`.)

Each wrapper is a `types.SimpleNamespace` exposing `.run`, backed by a private
function in `slam_v2026.py`:

| Call | Function | What it does (fixed vs free, settings) |
|---|---|---|
| `source_lp.run` | `_source_lp_run` | MGE lens light + Isothermal mass + ExternalShear + MGE source; `AnalysisImaging(use_jax=False)`; Nautilus `n_live=100` |
| `source_pix.run_1` | `_source_pix_run_1` | mesh `RectangularAdaptDensity(28,28)` + `reg.Adapt`; mass *priors* chained from source_lp (`mass_from`, centre unfixed); adapt image + positions auto-derived from the source_lp result |
| `source_pix.run_2` | `_source_pix_run_2` | mesh `RectangularAdaptImage(28,28)` + `reg.Adapt`; **mass FIXED** to run_1's instance |
| `light_lp.run` | `_light_lp_run` | refits lens light; mass + shear + source **fixed** |
| `mass_total.run` | `_mass_total_run` | `PowerLaw` mass free (priors chained from source_pix); light + source **fixed** |

The adapt image flows stage→stage via
`al.galaxy_name_image_dict_via_result_from(...)` → `al.AdaptImages(...)`, and the
positions likelihood via `result.positions_likelihood_from(factor=3.0,
minimum_threshold=0.2)`.

Note: `mass_total.run` has no SMBH/point-mass argument — add a central
`al.mp.PointMass` to the mass model yourself (tie its centre to the galaxy
centre) if your system needs one.
