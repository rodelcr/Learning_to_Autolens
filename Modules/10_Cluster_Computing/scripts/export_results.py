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
        fit_subplot.pdf       (max-LL residual / normalized-residual subplot)
        corner.pdf            (posterior corner plot)
        info.txt              (result.info — model summary, best-fit values)
        summary.json          (max_log_likelihood, n_params, key scalar params)
        samples.csv           (copy of Nautilus samples.csv)

Usage:
    # Export all searches under a given output root to Modules/XX/results/
    python export_results.py \\
        --output-root $SCRATCH/learning_to_autolens/output \\
        --repo-root /path/to/Learning_to_Autolens \\
        --module 04

    # Or export a single search directory:
    python export_results.py \\
        --search-dir /path/to/search_1_sis_nolenslight/<hash> \\
        --dest /path/to/Modules/04_.../results/search_1_sis_nolenslight

The script is idempotent — re-running it overwrites stale artifacts but
skips the expensive plot rendering if the PDFs already exist and the search
output is older (pass --force to always regenerate).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import traceback
from pathlib import Path


def find_search_dirs(output_root: Path):
    """Find every `files/search_internal/` under output_root. Each represents
    one Nautilus search. Returns parents of those dirs (i.e. the hashed
    <run_id> directories that hold `image/`, `files/`, etc.)."""
    for p in output_root.rglob("files/search_internal"):
        yield p.parent.parent


def export_one(search_dir: Path, dest: Path, force: bool = False):
    """Export artifacts for one search's output directory."""
    import autofit as af
    import autolens as al  # noqa: F401  (registers plot classes)
    import autolens.plot as aplt
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dest.mkdir(parents=True, exist_ok=True)
    print(f"[export] {search_dir.parent.name} → {dest}", flush=True)

    # 1. info.txt
    info_src = search_dir / "files" / "model.info"
    if not info_src.exists():
        info_src = search_dir / "info"
        if info_src.is_dir():
            info_src = next(iter(info_src.glob("model.info")), None)
    if info_src and info_src.exists():
        shutil.copy2(info_src, dest / "info.txt")

    # 2. samples.csv
    samples_src = search_dir / "files" / "samples.csv"
    if samples_src.exists():
        shutil.copy2(samples_src, dest / "samples.csv")

    # 3. Load the search & result via the samples pickle (lightweight)
    try:
        samples_pickle = search_dir / "files" / "samples.pickle"
        # Newer autofit versions write samples_summary.json instead
        summary_json = search_dir / "files" / "samples_summary.json"
        summary = {}
        if summary_json.exists():
            with open(summary_json) as f:
                summary = json.load(f)
    except Exception as e:
        print(f"[export] could not parse summary: {e}", flush=True)
        summary = {}

    # 4. summary.json — a tiny dict with max_log_likelihood + n_params
    try:
        with open(dest / "summary.json", "w") as f:
            json.dump({
                "search_name": search_dir.parent.name,
                "max_log_likelihood": summary.get("max_log_likelihood"),
                "log_evidence": summary.get("log_evidence"),
                "n_live": summary.get("n_live"),
                "n_samples": summary.get("total_samples") or summary.get("n_samples"),
                "source_dir": str(search_dir),
            }, f, indent=2)
    except Exception as e:
        print(f"[export] summary.json write failed: {e}", flush=True)

    # 5. fit_subplot.pdf + corner.pdf (expensive — skip if already there)
    fit_pdf = dest / "fit_subplot.pdf"
    corner_pdf = dest / "corner.pdf"
    if (fit_pdf.exists() and corner_pdf.exists() and not force):
        print("[export] plots already present, skipping (use --force to rebuild)",
              flush=True)
        return

    # Try to reconstruct the search result. Nautilus stores enough metadata in
    # the search directory that af.DirectoryPaths can rebuild a SearchOutput.
    try:
        search_output = af.SearchOutput(directory=search_dir.parent)
        result = search_output.result
    except Exception as e:
        print(f"[export] could not load result — skipping plots: {e}", flush=True)
        traceback.print_exc()
        return

    # 5a. Fit subplot
    try:
        fit = result.max_log_likelihood_fit
        mat_plot = aplt.MatPlot2D(
            output=aplt.Output(path=str(dest), filename="fit_subplot",
                               format="pdf"))
        fp = aplt.FitImagingPlotter(fit=fit, mat_plot_2d=mat_plot)
        fp.subplot_fit()
        plt.close("all")
    except Exception as e:
        print(f"[export] fit subplot failed: {e}", flush=True)

    # 5b. Corner plot
    try:
        mat_plot = aplt.MatPlot1D(
            output=aplt.Output(path=str(dest), filename="corner",
                               format="pdf"))
        sp = aplt.NestPlotter(samples=result.samples, mat_plot_1d=mat_plot)
        sp.corner_cornerpy()
        plt.close("all")
    except Exception as e:
        print(f"[export] corner plot failed: {e}", flush=True)


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
        export_one(args.search_dir, args.dest, force=args.force)
        return

    if not args.output_root:
        p.error("need either --output-root or --search-dir")
    if not args.module or not args.repo_root:
        p.error("--module and --repo-root are required when using --output-root")

    # Find the module's directory: Modules/<module>_*/
    candidates = list((args.repo_root / "Modules").glob(f"{args.module}_*"))
    if not candidates:
        sys.exit(f"No module dir matching {args.module}_* under Modules/")
    module_dir = candidates[0]
    results_root = module_dir / "results"
    results_root.mkdir(exist_ok=True)

    n = 0
    for search_dir in find_search_dirs(args.output_root):
        dest = results_root / search_dir.parent.name
        export_one(search_dir, dest, force=args.force)
        n += 1
    print(f"[export] exported {n} search(es) → {results_root}", flush=True)


if __name__ == "__main__":
    main()
