"""
export_results.py — Extract lightweight, git-trackable artifacts from a
Nautilus output tree.

Motivation:
    A full Nautilus output directory is hundreds of MB (samples, checkpoint,
    image grids, per-iteration state) — too large to commit. But a new user
    cloning the repo should still see the *finished* results without running
    the pipeline. This script bridges the gap: for each completed search, it
    writes a handful of small artifacts that each module's notebook can load
    without needing the raw search output.

Artifacts written per search:
    results/<search_name>/
        fit_subplot.png       (max-LL residual / normalized-residual subplot)
        corner.pdf            (posterior corner plot)
        info.txt              (model.info — human-readable model tree)
        model_results.txt     (model.results — best-fit values + uncertainties)
        samples.csv           (copy of Nautilus samples.csv)
        samples_summary.json  (copy of Nautilus samples_summary.json)
        summary.json          (max_log_likelihood + log_evidence + source_dir)

Usage:
    python export_results.py \\
        --output-root /path/to/output/module_04 \\
        --repo-root   /path/to/Learning_to_Autolens \\
        --module      04

The script is idempotent — re-running overwrites stale artifacts but skips
corner-plot rendering if the PDF already exists (pass --force to rebuild).

Discovery:
    A search is "completed" if its `files/samples_summary.json` exists. This
    marker survives cleanup of Nautilus internal state (unlike `search_internal/`,
    which the original version of this script relied on and which only exists
    mid-run). The hash directory is `<samples_summary>.parent.parent`.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import traceback
from pathlib import Path


def find_search_dirs(output_root: Path):
    """Yield the NEWEST completed-search hash directory per search name.

    A hash dir has `files/samples_summary.json` once Nautilus finishes and
    the search is finalized. We skip directories where that file is missing
    (partial / crashed runs) so the export doesn't emit empty summaries.

    When multiple hashes exist for the same search name (e.g. a prior run
    with stale priors + a fresh run with the corrected priors both
    finished), we keep only the newest. Otherwise both get exported to
    the same `Modules/XX/results/<search_name>/` destination and iteration
    order determines which one wins — which is exactly how the 2026-04-20
    Mod 04 re-run silently exported the stale numbers over the fresh ones.
    """
    # Map search_name → (newest_mtime, hash_dir)
    newest: dict[str, tuple[float, Path]] = {}
    stale: list[Path] = []
    for p in output_root.rglob("files/samples_summary.json"):
        hash_dir = p.parent.parent
        search_name = hash_dir.parent.name
        mtime = p.stat().st_mtime
        if search_name not in newest or mtime > newest[search_name][0]:
            if search_name in newest:
                stale.append(newest[search_name][1])
            newest[search_name] = (mtime, hash_dir)
        else:
            stale.append(hash_dir)

    if stale:
        print(f"[export] WARNING: ignored {len(stale)} stale hash dir(s) "
              "(older siblings of a newer run for the same search):",
              flush=True)
        for s in stale:
            print(f"  {s}", flush=True)
        print("[export] Delete them on the cluster once you've confirmed "
              "the newer run is correct.", flush=True)

    for _mtime, hash_dir in newest.values():
        yield hash_dir


def export_one(search_dir: Path, dest: Path, repo_root: Path | None = None, force: bool = False):
    """Export artifacts for one completed search.

    If `repo_root` is given, `summary.json`'s `source_dir` is written as a
    path relative to repo_root so the committed artifact is portable across
    users (otherwise Cannon's `/n/holystore01/LABS/<lab>/Lab/<user>/...`
    leaks into git). Falls back to the absolute string when the search dir
    isn't under repo_root.
    """
    import autofit as af
    from autofit.plot import corner_cornerpy
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dest.mkdir(parents=True, exist_ok=True)
    search_name = search_dir.parent.name
    print(f"[export] {search_name} → {dest}", flush=True)

    # 1. Copy static text/data files that don't need autolens loaded
    copies = [
        (search_dir / "model.info",              dest / "info.txt"),
        (search_dir / "model.results",           dest / "model_results.txt"),
        (search_dir / "files" / "samples.csv",   dest / "samples.csv"),
        (search_dir / "files" / "samples_summary.json",
                                                 dest / "samples_summary.json"),
    ]
    for src, dst in copies:
        if src.exists():
            shutil.copy2(src, dst)

    # 2. Load SearchOutput — gives us samples + max_log_likelihood cheaply
    try:
        so = af.SearchOutput(directory=search_dir)
    except Exception as e:
        print(f"[export] SearchOutput load failed: {e}", flush=True)
        traceback.print_exc()
        so = None

    # 3. summary.json — small scalar dict for quick programmatic access
    if repo_root is not None:
        try:
            source_dir_str = str(search_dir.resolve().relative_to(repo_root.resolve()))
        except ValueError:
            source_dir_str = str(search_dir)
    else:
        source_dir_str = str(search_dir)
    summary = {
        "search_name": search_name,
        "source_dir": source_dir_str,
        "max_log_likelihood": None,
        "log_evidence": None,
        "chi_squared_total": None,
        "chi_squared_per_pixel": None,
        "n_unmasked_pixels": None,
        "max_abs_normalized_residual": None,
    }
    if so is not None:
        try:
            summary["max_log_likelihood"] = float(so.max_log_likelihood)
        except Exception:
            pass
        try:
            if so.samples_summary is not None:
                le = getattr(so.samples_summary, "log_evidence", None)
                summary["log_evidence"] = float(le) if le is not None else None
        except Exception:
            pass

    # Chi-squared diagnostics. PyAutoLens already wrote the best-fit
    # residual/chi-squared maps to <search>/image/fit.fits during the run —
    # we just read them to avoid having to reconstruct the fit object.
    # Pixels outside the mask have chi_squared = 0 exactly, so counting
    # non-zero cells gives N_unmasked without needing to interpret the
    # MASK HDU's boolean convention.
    fit_fits = search_dir / "image" / "fit.fits"
    if fit_fits.exists():
        try:
            from astropy.io import fits as _fits
            import numpy as _np
            with _fits.open(fit_fits) as hdul:
                chi2_map = hdul["CHI_SQUARED_MAP"].data
                normres_map = hdul["NORMALIZED_RESIDUAL_MAP"].data
            n_unmasked = int((chi2_map != 0).sum())
            if n_unmasked > 0:
                chi2_total = float(chi2_map.sum())
                summary["chi_squared_total"] = chi2_total
                summary["chi_squared_per_pixel"] = chi2_total / n_unmasked
                summary["n_unmasked_pixels"] = n_unmasked
                summary["max_abs_normalized_residual"] = float(
                    _np.nanmax(_np.abs(normres_map))
                )
        except Exception as e:
            print(f"[export] chi2 diagnostics failed: {e}", flush=True)
            summary["chi_squared_status"] = f"read error: {e}"
    else:
        # Missing fit.fits typically means the search was resumed from a
        # checkpoint and autolens skipped the visualization regeneration
        # step (`force_visualize_overwrite: false` in general.yaml). The
        # quality metrics can't be recovered from the lightweight files
        # alone. Flag explicitly in the summary so the consumer knows
        # *why* the fields are null rather than assuming the fit failed.
        summary["chi_squared_status"] = (
            "image/fit.fits not found — likely a resumed search skipped "
            "visualization. Fix: re-run with force_visualize_overwrite=True "
            "in the config, or call analysis.visualize(paths, instance, "
            "during_analysis=False) post-fit in the fit script."
        )
        print(f"[export]   WARNING: {fit_fits} missing → residual metrics null",
              flush=True)

    with open(dest / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # 4. Fit subplot — autolens has already rendered image/fit.png for every
    # search during the run, so we just copy it rather than rebuilding a Fit
    # object (which would require the dataset + tracer reconstruction).
    fit_png_src = search_dir / "image" / "fit.png"
    fit_png_dst = dest / "fit_subplot.png"
    if fit_png_src.exists():
        shutil.copy2(fit_png_src, fit_png_dst)
    else:
        print(f"[export] no image/fit.png under {search_dir} — skipping fit subplot",
              flush=True)

    # 5. Corner plot — render to PDF unless already there
    corner_pdf = dest / "corner.pdf"
    if corner_pdf.exists() and not force:
        print("[export] corner.pdf already present, skipping (use --force to rebuild)",
              flush=True)
        return
    if so is None or so.samples is None:
        print("[export] no samples available — skipping corner plot", flush=True)
        return
    try:
        corner_cornerpy(
            samples=so.samples,
            path=str(dest),
            filename="corner",
            format="pdf",
        )
        plt.close("all")
    except Exception as e:
        print(f"[export] corner plot failed: {e}", flush=True)
        traceback.print_exc()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", type=Path,
                   help="Root of Nautilus outputs (walks recursively)")
    p.add_argument("--module", type=str,
                   help="Module number (04, 05, 09) to export into "
                        "Modules/<module>_.../results/. Requires --repo-root.")
    p.add_argument("--repo-root", type=Path,
                   help="Learning_to_Autolens repo root (for destination path)")
    p.add_argument("--search-dir", type=Path,
                   help="Single search output dir to export (overrides --output-root)")
    p.add_argument("--dest", type=Path,
                   help="Destination dir (required with --search-dir)")
    p.add_argument("--force", action="store_true",
                   help="Regenerate plots even if they already exist")
    args = p.parse_args()

    if args.search_dir:
        if not args.dest:
            p.error("--dest is required with --search-dir")
        export_one(args.search_dir, args.dest, repo_root=args.repo_root, force=args.force)
        return

    if not args.output_root:
        p.error("need either --output-root or --search-dir")
    if not args.module or not args.repo_root:
        p.error("--module and --repo-root are required when using --output-root")

    candidates = list((args.repo_root / "Modules").glob(f"{args.module}_*"))
    if not candidates:
        sys.exit(f"No module dir matching {args.module}_* under Modules/")
    module_dir = candidates[0]
    results_root = module_dir / "results"
    results_root.mkdir(exist_ok=True)

    n = 0
    for search_dir in find_search_dirs(args.output_root):
        dest = results_root / search_dir.parent.name
        export_one(search_dir, dest, repo_root=args.repo_root, force=args.force)
        n += 1
    print(f"[export] exported {n} search(es) → {results_root}", flush=True)


if __name__ == "__main__":
    main()
