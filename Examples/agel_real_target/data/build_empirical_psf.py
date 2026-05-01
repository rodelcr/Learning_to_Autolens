"""
build_empirical_psf.py — Empirical PSF from bright stars in the AGEL013322
HST/ACS F606W drizzled frame.

Replaces the placeholder Gaussian PSF (sigma=0.036") with a stack of bright
isolated point sources from the same image. This is the standard empirical
PSF approach: better than a Gaussian, simpler to deploy than TinyTim, and
captures the actual instrumental + drizzle resampling effects in the data.

Methodology:
  1. Read the HST point-source catalog (sub-arcsec astrometric positions)
  2. Filter for clean point sources: Flags=0, MagErr<0.05, S/N>50, CI in [1.05, 1.25]
  3. Reject any star with another catalog source within stamp radius (avoid
     blends + contaminants in the stack)
  4. Extract 51x51 px (2.55") stamps centered on each surviving star
  5. Local-background subtract each stamp
  6. Recenter via flux-weighted centroid (sub-pixel)
  7. Normalize each to unit total flux, then sigma-clip stack the median
  8. Re-normalize and write `data/psf.fits` (overwrites the placeholder)

Run from the repo root:
    conda run -n autolens python Examples/agel_real_target/data/build_empirical_psf.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.stats import sigma_clipped_stats, SigmaClip
from scipy.ndimage import shift as ndi_shift


HERE = Path(__file__).parent.resolve()
META = json.loads((HERE / "metadata.json").read_text())
SRC_FITS = Path(META["source_fits"])
SRC_CAT = SRC_FITS.parent / SRC_FITS.name.replace("drc", "point-cat").replace(".fits", ".ecsv")

STAMP_HALF = 25  # 51x51 stamps -> 2.55" at 0.05"/px
ISOLATION_RADIUS = 60  # px; reject star if any catalog source within this radius
LENS_X, LENS_Y = META["lens_pixel_in_full_frame"]


def load_image_and_catalog():
    print(f"Reading {SRC_FITS.name} ...", flush=True)
    with fits.open(SRC_FITS) as hdul:
        # ACS drizzled has SCI in extension 1
        data = hdul[1].data.astype(float)
    print(f"  shape: {data.shape}", flush=True)
    cat = Table.read(SRC_CAT)
    print(f"  catalog: {len(cat)} sources", flush=True)
    return data, cat


def filter_clean_point_sources(cat: Table) -> Table:
    sn = cat["FluxAp2"] / cat["FluxErrAp2"]
    clean = (
        (cat["Flags"] == 0)
        & (cat["MagErrAp2"] < 0.05)
        & (sn > 50)
        & (cat["CI"] > 1.05)
        & (cat["CI"] < 1.25)
    )
    sub = cat[clean].copy()
    sub.sort("FluxAp2", reverse=True)
    print(f"Clean PSF candidates after quality cuts: {len(sub)}", flush=True)
    return sub


def reject_blended(candidates: Table, full_cat: Table, isolation_r: int = ISOLATION_RADIUS) -> Table:
    """Drop any candidate that has a catalog neighbor (any source) within
    isolation_r pixels. This keeps the PSF stack uncontaminated."""
    keep = []
    cx_full = np.asarray(full_cat["X-Center"])
    cy_full = np.asarray(full_cat["Y-Center"])
    for row in candidates:
        cx, cy = row["X-Center"], row["Y-Center"]
        dist = np.sqrt((cx_full - cx) ** 2 + (cy_full - cy) ** 2)
        # The candidate itself is in dist=0; any other source within isolation_r is bad.
        n_neighbors = (dist > 0) & (dist < isolation_r)
        if not n_neighbors.any():
            keep.append(row)
    out = Table(rows=keep, names=candidates.colnames)
    print(f"After isolation filter ({isolation_r} px = {isolation_r*0.05:.1f}\"): {len(out)}", flush=True)
    return out


def reject_near_lens(candidates: Table, lens_x: float, lens_y: float,
                     exclusion_r: int = 200) -> Table:
    """Don't use stars within exclusion_r px of the lens system itself
    (avoid systematic spatial bias from drizzle resampling at one location)."""
    dist = np.sqrt((candidates["X-Center"] - lens_x) ** 2 +
                   (candidates["Y-Center"] - lens_y) ** 2)
    keep = candidates[dist > exclusion_r]
    print(f"After lens-vicinity exclusion ({exclusion_r} px): {len(keep)}", flush=True)
    return keep


def extract_stamp(data: np.ndarray, x: float, y: float, half: int = STAMP_HALF):
    """Integer-shift extract; returns None if stamp falls off image."""
    ix, iy = int(round(x)), int(round(y))
    y0, y1 = iy - half, iy + half + 1
    x0, x1 = ix - half, ix + half + 1
    if y0 < 0 or x0 < 0 or y1 > data.shape[0] or x1 > data.shape[1]:
        return None
    return data[y0:y1, x0:x1].copy()


def recenter_subpixel(stamp: np.ndarray) -> np.ndarray:
    """Sub-pixel re-centre to put the flux-weighted centroid exactly at the
    central pixel. Uses bilinear-spline shift."""
    ny, nx = stamp.shape
    cy_target, cx_target = ny // 2, nx // 2
    yy, xx = np.indices(stamp.shape)
    # Use only the central 21x21 region for centroid (defeats wing biases)
    cw = 10
    ymask = (yy >= cy_target - cw) & (yy <= cy_target + cw)
    xmask = (xx >= cx_target - cw) & (xx <= cx_target + cw)
    mask = ymask & xmask
    flux = np.maximum(stamp - np.median(stamp), 0)
    flux = np.where(mask, flux, 0)
    if flux.sum() <= 0:
        return stamp
    cy_now = (flux * yy).sum() / flux.sum()
    cx_now = (flux * xx).sum() / flux.sum()
    return ndi_shift(stamp, (cy_target - cy_now, cx_target - cx_now), order=3, mode="constant", cval=0.0)


def stack_psf(stamps: list[np.ndarray]) -> np.ndarray:
    """Sigma-clip median stack across normalized stamps."""
    arr = np.stack(stamps, axis=0)
    sc = SigmaClip(sigma=3.0, maxiters=3)
    masked = sc(arr, axis=0)
    return np.ma.median(masked, axis=0).filled(0.0)


def main():
    data, cat = load_image_and_catalog()

    candidates = filter_clean_point_sources(cat)
    candidates = reject_blended(candidates, cat)
    candidates = reject_near_lens(candidates, LENS_X, LENS_Y)
    if len(candidates) < 3:
        raise RuntimeError(f"Only {len(candidates)} stars after filters — need 3+")

    print(f"Extracting + recentering {min(len(candidates), 20)} stamps ...", flush=True)
    stamps = []
    used_rows = []
    for row in candidates[:20]:  # cap at 20 best stars
        stamp = extract_stamp(data, row["X-Center"] - 1.0, row["Y-Center"] - 1.0)
        if stamp is None:
            continue
        # Local-background subtract via outer-annulus sigma-clipped mean
        edge = np.concatenate([stamp[:5, :].ravel(), stamp[-5:, :].ravel(),
                                stamp[:, :5].ravel(), stamp[:, -5:].ravel()])
        _, bg, _ = sigma_clipped_stats(edge, sigma=3.0)
        stamp = stamp - bg
        stamp = recenter_subpixel(stamp)
        flux = stamp.sum()
        if flux <= 0:
            continue
        stamps.append(stamp / flux)
        used_rows.append(row)
    print(f"  {len(stamps)} usable stamps", flush=True)

    psf = stack_psf(stamps)
    psf = psf / psf.sum()

    # Sanity check: FWHM
    profile = np.median([psf.max()], axis=None)
    half = profile / 2.0
    yy, xx = np.indices(psf.shape)
    cy, cx = psf.shape[0] // 2, psf.shape[1] // 2
    r2 = (yy - cy) ** 2 + (xx - cx) ** 2
    above_half = (psf >= half).sum()
    fwhm_px_estimate = 2 * np.sqrt(above_half / np.pi)
    print(f"  Empirical PSF FWHM ~ {fwhm_px_estimate:.2f} px = {fwhm_px_estimate*0.05:.3f}\"")

    # Backup the placeholder, write empirical
    placeholder_backup = HERE / "psf_gaussian_placeholder.fits"
    psf_path = HERE / "psf.fits"
    if psf_path.exists() and not placeholder_backup.exists():
        import shutil
        shutil.copy(psf_path, placeholder_backup)
        print(f"  Backed up old placeholder to {placeholder_backup.name}", flush=True)

    fits.PrimaryHDU(data=psf.astype(np.float32)).writeto(psf_path, overwrite=True)
    print(f"  Wrote empirical PSF: {psf_path.name} (shape={psf.shape}, sum={psf.sum():.4f})",
          flush=True)

    # Update metadata
    META["psf_model"] = (
        f"Empirical: median-stacked sigma-clipped from {len(stamps)} bright "
        f"isolated point sources in same drizzled frame; built by "
        f"`data/build_empirical_psf.py`. FWHM ~ {fwhm_px_estimate*0.05:.3f}\"."
    )
    META["psf_n_stars_stacked"] = len(stamps)
    META["psf_fwhm_arcsec_empirical"] = float(fwhm_px_estimate * 0.05)
    (HERE / "metadata.json").write_text(json.dumps(META, indent=2))
    print("Updated metadata.json psf_model field.", flush=True)


if __name__ == "__main__":
    main()
