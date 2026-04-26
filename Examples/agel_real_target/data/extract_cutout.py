"""
Cutout extraction for AGEL013322-125201A from full HST/ACS WFC F606W frame.

Source data lives outside this repository:
    ~/Documents/AGEL/AGEL013322-125201A_HST_ACS_606/hst_17307_4b_acs_wfc_f606w_jf544b_drc.fits

The full frame is 5848 × 5855 pixels. The lens AGEL013322-125201A sits at
HST pixel coordinate (3622, 882) (located via segment-cat.ecsv lookup at
RA = 23.34°, Dec = -12.87°). We extract a 200 × 200 px cutout (10" × 10"
at 0.05"/px) centered on the lens, build a noise map from the WHT
extension (inverse-variance map drizzled by HST pipeline), and write
out FITS files ready for `al.Imaging.from_fits`.

Run with the autolens conda env (path absolutism is intentional — this
script needs the original full-frame FITS at the documented path):

    python extract_cutout.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from astropy.io import fits

OUT = Path(__file__).parent
SRC = Path(
    "/Users/rosador/Documents/AGEL/AGEL013322-125201A_HST_ACS_606/"
    "hst_17307_4b_acs_wfc_f606w_jf544b_drc.fits"
)

# Lens pixel coordinate in the full HST frame, found via:
#   segment-cat.ecsv lookup: closest object to RA=23.34, Dec=-12.87 is ID=97
LENS_X_PIX = 3622
LENS_Y_PIX = 882
HALF_SIZE = 100  # extract 200x200 stamp = 10" x 10"


def main():
    if not SRC.exists():
        raise SystemExit(
            f"Source HST frame not found at {SRC} — adjust the SRC constant "
            "in this script if the file lives elsewhere."
        )

    with fits.open(SRC) as hdul:
        sci = hdul["SCI"].data
        wht = hdul["WHT"].data
        sci_header = hdul["SCI"].header

    # Cutout
    y, x = LENS_Y_PIX, LENS_X_PIX
    sci_cut = sci[y - HALF_SIZE:y + HALF_SIZE,
                  x - HALF_SIZE:x + HALF_SIZE].astype(np.float32)
    wht_cut = wht[y - HALF_SIZE:y + HALF_SIZE,
                  x - HALF_SIZE:x + HALF_SIZE].astype(np.float32)

    # Noise map from WHT (inverse-variance map). HST drizzle WHT units are
    # 1 / variance in the same units as SCI (electrons/s here). So:
    #
    #     sigma_pixel = 1 / sqrt(WHT)
    #
    # Where WHT==0 (masked / zero-coverage), set sigma to a large
    # placeholder so autolens won't fit those pixels.
    with np.errstate(divide="ignore", invalid="ignore"):
        noise_map = np.where(wht_cut > 0, 1.0 / np.sqrt(wht_cut), 1e6).astype(np.float32)

    # PSF: for the scaffold, we use a simple Gaussian model. ACS/WFC F606W
    # has ~0.085" FWHM (sigma ≈ 0.036") — below pixel scale (0.05"). For
    # publication-grade work, replace with a TinyTim-modelled PSF or an
    # empirical PSF from a star in the same frame; see README §Caveats.
    psf_size = 21
    psf = np.zeros((psf_size, psf_size), dtype=np.float32)
    cy, cx = psf_size // 2, psf_size // 2
    sigma_pix = 0.036 / 0.05
    for j in range(psf_size):
        for i in range(psf_size):
            r2 = (j - cy)**2 + (i - cx)**2
            psf[j, i] = np.exp(-r2 / (2 * sigma_pix**2))
    psf /= psf.sum()

    fits.writeto(OUT / "image.fits", sci_cut, overwrite=True)
    fits.writeto(OUT / "noise_map.fits", noise_map, overwrite=True)
    fits.writeto(OUT / "psf.fits", psf, overwrite=True)

    # Catalog metadata for traceability
    metadata = {
        "target_name": "AGEL013322-125201A",
        "alt_name": "DCLS0133-1252",
        "source_fits": str(SRC),
        "instrument": "HST/ACS/WFC",
        "filter": "F606W",
        "exptime_s": float(sci_header.get("EXPTIME", 674.0)),
        "pixel_scale_arcsec": 0.05,
        "lens_pixel_in_full_frame": [LENS_X_PIX, LENS_Y_PIX],
        "cutout_half_size_pix": HALF_SIZE,
        "cutout_shape": list(sci_cut.shape),
        "redshifts": {
            "lens": 0.30,
            "source": 1.6,
            "comment": ("Approximate values from AGEL DR2 catalog; verify "
                        "against the canonical Keck spectroscopic redshifts "
                        "before publication."),
        },
        "psf_model": "Gaussian sigma=0.036 arcsec (placeholder; see README Caveats)",
        "noise_map_recipe": "1/sqrt(WHT) from drizzle inverse-variance map",
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2))

    print(f"Wrote: image.fits, noise_map.fits, psf.fits, metadata.json in {OUT}")
    print(f"Cutout: {sci_cut.shape} px @ 0.05\"/px = "
          f"{sci_cut.shape[0]*0.05:.1f}\" x {sci_cut.shape[1]*0.05:.1f}\"")
    print(f"Image stats: min={sci_cut.min():.4f}, max={sci_cut.max():.4f}, "
          f"median={np.median(sci_cut):.4f}")
    print(f"Noise stats: min={noise_map[noise_map<1e3].min():.4f}, "
          f"median={np.median(noise_map[noise_map<1e3]):.4f}")


if __name__ == "__main__":
    main()
