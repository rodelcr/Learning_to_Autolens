"""
Build noise-map FITS + parsed truths.json for each lenstronomy mock 2-6.

Same recipe as Examples/mge_to_physical/mocks/build_mock_artifacts.py — see
the PROVENANCE.md there for the full lenstronomy → autolens parameter
convention reference.

This script handles the *batch* version: loops over mocks 2 through 6 and
emits per-mock noise FITS + per-mock truths JSON. The image FITS and the
shared PSF are already copied verbatim by the README setup step.

Run with the autolens conda env:

    python build_mock_artifacts.py
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
from astropy.io import fits

OUT = Path(__file__).parent
MOCK_INDICES = [2, 3, 4, 5, 6]


def _parse_lenstronomy_params(text: str) -> dict:
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


for n in MOCK_INDICES:
    img_path = OUT / f"lenstronomy_mock_{n}_image.fits"
    params_path = OUT / f"lenstronomy_mock_{n}_params.txt"
    img = fits.getdata(img_path)
    truths = _parse_lenstronomy_params(params_path.read_text())
    exp_time = float(truths["exp_time"])
    bg_rms = float(truths["background_rms"])

    noise = np.sqrt(np.maximum(img, 0.0) / exp_time + bg_rms**2).astype(np.float32)
    fits.writeto(OUT / f"lenstronomy_mock_{n}_noise.fits", noise, overwrite=True)

    clean = {
        "provenance": {
            "source_repo": "lenstronomy_AGEL_modules/tutorials_DB_2025_09",
            "lenstronomy_mock": f"mock_{n}",
            "framework_of_origin": "lenstronomy 1.10.4",
            "adapted_for": "PyAutoLens 2026.4.13.6",
        },
        "imaging": {
            "shape_native": list(img.shape),
            "pixel_scales_arcsec": float(truths["pixel_scale"]),
            "exp_time_s": exp_time,
            "background_rms": bg_rms,
            "ra_at_xy_0": float(truths["ra_at_xy_0"]),
            "dec_at_xy_0": float(truths["dec_at_xy_0"]),
        },
        "cosmology": truths["cosmo"],
        "redshifts": {
            "lens_primary":   truths["redshift_list"][0],
            "lens_secondary": truths["redshift_list"][1],
            "source":         truths["redshift_list"][2],
        },
        "lens_model_list":       truths["lens_model_list"],
        "kwargs_lens":           truths["kwargs_lens"],
        "source_model_list":     truths["source_model_list"],
        "kwargs_source":         truths["kwargs_source"],
        "lens_light_model_list": truths["lens_light_model_list"],
        "kwargs_lens_light":     truths["kwargs_lens_light"],
    }
    (OUT / f"truths_mock_{n}.json").write_text(json.dumps(clean, indent=2))

    primary = truths["kwargs_lens"][0]
    second = truths["kwargs_lens"][2] if len(truths["kwargs_lens"]) > 2 else {}
    print(f"mock_{n}:  z_l={truths['redshift_list'][0]}, "
          f"z_s={truths['redshift_list'][2]}, "
          f"theta_E_primary={primary.get('theta_E', '?'):.3f}\", "
          f"gamma_primary={primary.get('gamma', '?')}, "
          f"theta_E_secondary={second.get('theta_E', '?')}, "
          f"cosmo Om={truths['cosmo'].get('Om')}, w={truths['cosmo'].get('w')}")
