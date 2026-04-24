"""compound_multiplane_plot.py — Compact multi-plane view for compound fits.

PyAutoLens 2026.4's default visualisation emits `fit_1.png`, `fit_2.png`, …,
one per source plane in a multi-plane fit. Each is a full 12-panel grid
(4 columns × 3 rows) with identical Data / Model / Residual / Chi²
panels and ONLY the "Source Plane (Zoomed)" panel changes between them.

That's wasteful to flip through. This script tiles the changing panels
into a single figure:

    Top row:     Data, Model Image, Normalized Residual, Chi² Map  (from fit_1)
    Bottom row:  N "Source Plane (Zoomed)" panels, one per fit_N

Pure image-level: no fit reconstruction, no aggregator session. Reads the
existing `fit_N.png` files from a results dir and writes
`compound_multiplane.png` alongside them.

Usage:
    python compound_multiplane_plot.py \
        --results-dir Examples/compound_lens/results/compound_direct_fit/
"""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


def _crop_panel(fit_img: Image.Image, row: int, col: int,
                n_rows: int = 3, n_cols: int = 4) -> Image.Image:
    """Crop one panel from a PyAutoLens fit_subplot image grid."""
    W, H = fit_img.size
    # PyAutoLens subplots have non-zero inter-panel padding we ignore — the
    # per-panel crop includes the panel's colour-bar strip on its right.
    panel_w = W // n_cols
    panel_h = H // n_rows
    left   = col * panel_w
    upper  = row * panel_h
    right  = left + panel_w
    lower  = upper + panel_h
    return fit_img.crop((left, upper, right, lower))


def _tile_images(top_panels: list[Image.Image],
                 bottom_panels: list[Image.Image],
                 bottom_labels: list[str] | None = None) -> Image.Image:
    """Stack a top row and a bottom row with matching per-panel widths."""
    # Top row: fixed panel width W0, height H0
    W0 = max(p.width  for p in top_panels)
    H0 = max(p.height for p in top_panels)
    top_canvas = Image.new("RGBA", (W0 * len(top_panels), H0), (255, 255, 255, 255))
    for i, p in enumerate(top_panels):
        top_canvas.paste(p, (i * W0, 0))

    # Bottom row: same panel height H0, width W0 (force scale)
    bot_panels_resized = [p.resize((W0, H0), Image.LANCZOS) for p in bottom_panels]
    total_bot_w = W0 * len(bottom_panels)
    bottom_canvas = Image.new("RGBA", (total_bot_w, H0), (255, 255, 255, 255))
    for i, p in enumerate(bot_panels_resized):
        bottom_canvas.paste(p, (i * W0, 0))

    # If the bottom is narrower/wider than the top, centre the smaller
    if total_bot_w < top_canvas.width:
        # centre the bottom under the top
        bot_full = Image.new("RGBA", (top_canvas.width, H0), (255, 255, 255, 255))
        offset = (top_canvas.width - total_bot_w) // 2
        bot_full.paste(bottom_canvas, (offset, 0))
        bottom_canvas = bot_full
    elif total_bot_w > top_canvas.width:
        # widen the top to match (shouldn't normally happen with 4 top panels)
        top_full = Image.new("RGBA", (total_bot_w, H0), (255, 255, 255, 255))
        offset = (total_bot_w - top_canvas.width) // 2
        top_full.paste(top_canvas, (offset, 0))
        top_canvas = top_full

    # Optional labels — render them as a header STRIP above the bottom row
    # (not overlaid on top of the existing autolens panel titles).
    header_strip = None
    if bottom_labels is not None:
        from PIL import ImageDraw, ImageFont
        HEADER_H = 110
        header_strip = Image.new("RGBA", (bottom_canvas.width, HEADER_H),
                                 (245, 245, 245, 255))
        draw = ImageDraw.Draw(header_strip)
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 48)
        except Exception:
            font = ImageFont.load_default()
        offset_x = (bottom_canvas.width - total_bot_w) // 2
        for i, label in enumerate(bottom_labels):
            x = offset_x + i * W0 + 30
            # Center vertically in header_strip
            draw.text((x, HEADER_H // 2 - 24), label, fill=(40, 40, 40, 255), font=font)

    # Stack vertically
    total_h = top_canvas.height + (header_strip.height if header_strip else 0) + bottom_canvas.height
    out = Image.new("RGBA", (top_canvas.width, total_h), (255, 255, 255, 255))
    y = 0
    out.paste(top_canvas, (0, y)); y += top_canvas.height
    if header_strip is not None:
        out.paste(header_strip, (0, y)); y += header_strip.height
    out.paste(bottom_canvas, (0, y))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", type=Path, required=True,
                   help="Directory containing fit_1.png, fit_2.png, ...")
    p.add_argument("--plane-labels", nargs="*", default=None,
                   help="Optional per-plane labels (e.g. 'z=0.5' 'z=0.8' 'z=1.7')")
    p.add_argument("--output", type=Path, default=None,
                   help="Output PNG path (default: <results-dir>/compound_multiplane.png)")
    args = p.parse_args()

    # Find all source-plane subplots in redshift order.
    #   - fit_subplot.png  ← export_results.py renamed fit_1.png to this
    #   - fit_2.png, fit_3.png, … for additional source planes
    # So the ordered list is: fit_subplot.png (= plane 1), then fit_2, fit_3, …
    fit_pngs: list[Path] = []
    if (args.results_dir / "fit_subplot.png").exists():
        fit_pngs.append(args.results_dir / "fit_subplot.png")
    fit_pngs.extend(sorted(args.results_dir.glob("fit_[0-9]*.png")))
    if not fit_pngs:
        raise SystemExit(f"No fit_*.png or fit_subplot.png found in {args.results_dir}")

    print(f"[multiplane] Using {len(fit_pngs)} source-plane subplots:")
    for p in fit_pngs:
        print(f"             {p.name}")

    # Load all
    fit_imgs = [Image.open(str(p)).convert("RGBA") for p in fit_pngs]

    # Top row comes from fit_1 (or whatever is first)
    first = fit_imgs[0]
    # Standard PyAutoLens 12-panel fit grid: 4 cols × 3 rows
    # Row 0: Data, Data (Source Scale), S/N, Model
    # Row 1: Lens Light Model, Lens Light Subtracted, Source Model, Source Plane (Zoomed)
    # Row 2: Normalized Residual, Normalized Residual 1σ, Chi², Source Plane (No Zoom)
    top_panels = [
        _crop_panel(first, row=0, col=0),  # Data
        _crop_panel(first, row=0, col=3),  # Model Image
        _crop_panel(first, row=2, col=0),  # Normalized Residual
        _crop_panel(first, row=2, col=2),  # Chi-Squared Map
    ]

    # Bottom row: the Source Plane (Zoomed) panel (row 1, col 3) from each fit_N
    bottom_panels = [_crop_panel(img, row=1, col=3) for img in fit_imgs]
    bottom_labels = args.plane_labels if args.plane_labels else [
        f"plane {i+1}" for i in range(len(fit_imgs))
    ]
    if len(bottom_labels) != len(fit_imgs):
        raise SystemExit(f"--plane-labels ({len(bottom_labels)}) doesn't match "
                         f"fit_*.png count ({len(fit_imgs)})")

    out_img = _tile_images(top_panels, bottom_panels, bottom_labels)
    out_path = args.output or (args.results_dir / "compound_multiplane.png")
    out_img.save(out_path)
    print(f"[multiplane] wrote {out_path} ({out_path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
