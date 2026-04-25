"""
Build the noise_map FITS and parse the lenstronomy truth-params text file
into a tidy JSON for use by the autolens fit.

The lenstronomy mocks at:

    /Users/rosador/Documents/AGEL/lenstronomy_AGEL_modules/tutorials_DB_2025_09/

ship with image FITS + a separate `mock_true_params/mock_N_params.txt`
file that uses Python-literal repr (dict/list/tuples). They do NOT ship
with a noise-map FITS. The truth params include `background_rms` (Gaussian
sky noise per pixel) and `exp_time` (seconds, electrons/sec convention).

We build a Gaussian-equivalent noise map per autolens's standard:

    sigma_pix = sqrt( max(image_e_per_s, 0)/exp_time + background_rms**2 )

This matches autolens's `al.PreloadOverSampleSubgrid` expectations and is
the same recipe used by autolens's own `simulator.py` examples. For
production AGEL targets you would replace this with the real noise map
computed by `astropy.nddata` / `drizzlepac` from the observed flat-fielded
exposure, but for a synthetic mock the analytic recipe is exact.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
from astropy.io import fits

OUT = Path(__file__).parent
IMG = fits.getdata(OUT / "lenstronomy_mock_1_image.fits")
PARAMS_TXT = (OUT / "lenstronomy_mock_1_params.txt").read_text()


def _parse_lenstronomy_params(text: str) -> dict:
    """Parse the lenstronomy `key: <python literal>` text format."""
    out: dict = {}
    for line in text.splitlines():
        if not line.strip() or ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip()
        val = val.strip()
        if val.startswith(("[", "{", "(")) or val.replace(".", "", 1).replace("-", "", 1).isdigit():
            try:
                out[key] = ast.literal_eval(val)
            except (ValueError, SyntaxError):
                out[key] = val
        else:
            out[key] = val
    return out


truths = _parse_lenstronomy_params(PARAMS_TXT)
exp_time = float(truths["exp_time"])
bg_rms = float(truths["background_rms"])
pixel_scale = float(truths["pixel_scale"])

noise_map = np.sqrt(np.maximum(IMG, 0.0) / exp_time + bg_rms**2).astype(np.float32)
fits.writeto(OUT / "lenstronomy_mock_1_noise.fits", noise_map, overwrite=True)
print(f"Wrote noise_map: shape={noise_map.shape}, "
      f"min={noise_map.min():.5f}, max={noise_map.max():.5f}")
print(f"  bg_rms={bg_rms}, exp_time={exp_time}s, pixel_scale={pixel_scale}\"")

clean_truths = {
    "provenance": {
        "source_repo": "lenstronomy_AGEL_modules/tutorials_DB_2025_09",
        "lenstronomy_mock": "mock_1",
        "framework_of_origin": "lenstronomy 1.10.4",
        "adapted_for": "PyAutoLens 2026.4.13.6",
    },
    "imaging": {
        "shape_native": list(IMG.shape),
        "pixel_scales_arcsec": pixel_scale,
        "exp_time_s": exp_time,
        "background_rms": bg_rms,
        "ra_at_xy_0": float(truths["ra_at_xy_0"]),
        "dec_at_xy_0": float(truths["dec_at_xy_0"]),
    },
    "cosmology": truths["cosmo"],
    "redshifts": {
        "lens_primary": truths["redshift_list"][0],
        "lens_secondary": truths["redshift_list"][1],
        "source": truths["redshift_list"][2],
    },
    "lens_model_list": truths["lens_model_list"],
    "kwargs_lens": truths["kwargs_lens"],
    "source_model_list": truths["source_model_list"],
    "kwargs_source": truths["kwargs_source"],
    "lens_light_model_list": truths["lens_light_model_list"],
    "kwargs_lens_light": truths["kwargs_lens_light"],
}

(OUT / "truths.json").write_text(json.dumps(clean_truths, indent=2))
print(f"Wrote truths.json: keys={list(clean_truths.keys())}")
print()
print("Mock summary:")
print(f"  Primary lens (z={clean_truths['redshifts']['lens_primary']}): "
      f"EPL theta_E={truths['kwargs_lens'][0]['theta_E']:.3f}\", "
      f"gamma={truths['kwargs_lens'][0]['gamma']:.2f}")
print(f"  Secondary lens (z={clean_truths['redshifts']['lens_secondary']}): "
      f"EPL theta_E={truths['kwargs_lens'][2]['theta_E']:.3f}\"")
print(f"  Source (z={clean_truths['redshifts']['source']}): "
      f"{len(truths['kwargs_source'])} Sersic components")
print(f"  Lens light: Sersic R_e={truths['kwargs_lens_light'][0]['R_sersic']}\"  "
      f"n={truths['kwargs_lens_light'][0]['n_sersic']}")
