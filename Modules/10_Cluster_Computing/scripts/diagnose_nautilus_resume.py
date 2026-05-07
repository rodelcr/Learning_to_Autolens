"""diagnose_nautilus_resume.py — inspect a Nautilus checkpoint.hdf5 to
discriminate between the three deadlock hypotheses (task #111):

  B1  model-hash mismatch — checkpoint was written under one model spec,
      resume tries to load it into a different model spec. Behavior
      undefined. Symptom: workers spin at 80-95% CPU but no log output
      and no checkpoint mtime advance. Fix: use a fresh unique_tag.

  B2  stale Nautilus checkpoint format — Nautilus version that wrote the
      checkpoint differs from the one trying to resume. Symptom: same as
      B1, distinguishable by HDF5 structure or version-string field.
      Fix: bump Nautilus pin and document.

  B3  bad sample point in resume queue — one live point causes the
      likelihood to hang (e.g. cosmology pushed to unphysical region).
      Symptom: same, distinguishable by the recorded live points
      including extreme cosmology / mass-profile values. Fix: prior
      bounds (TruncatedGaussian).

Usage:
    python diagnose_nautilus_resume.py <path_to_checkpoint.hdf5>

    or, with no argument, scans the four known stuck checkpoints from
    the v3 truth_fc deadlock (task #111) on the laptop side:
        Examples/compound_lens_zoo/results/.../checkpoint.hdf5

Reports for each: file size, last mtime, Nautilus version field if
present, dimensionality, n_live, top of the live-point parameter array.
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

try:
    import h5py
except ImportError:
    sys.exit("h5py not available — `pip install h5py` first")
import numpy as np


# Cannon-side default checkpoint locations (relative to repo root). These
# are the four checkpoints from the May 1-4 truth_fc runs that the v3
# resumes failed to advance.
DEFAULT_CHECKPOINTS_RELATIVE = [
    "output/mock_2_R5_truth_freecosmo/mock_2_R5_truth_freecosmo/91e4886aa364bb3a8feda53821fec358/files/search_internal/checkpoint.hdf5",
    "output/mock_3_R5_truth_freecosmo/mock_3_R5_truth_freecosmo/4d1f5261b5aebf9ebac5fe47d2158b24/files/search_internal/checkpoint.hdf5",
    "output/mock_5_R5_truth_freecosmo/mock_5_R5_truth_freecosmo/4bc4710b5c2de376f7c1ce08e3f8369c/files/search_internal/checkpoint.hdf5",
]


def walk_h5(g: h5py.Group, prefix: str = "/") -> list[tuple[str, str]]:
    """Recursively walk an HDF5 group, return list of (path, summary)."""
    out: list[tuple[str, str]] = []
    for name, item in g.items():
        path = f"{prefix}{name}"
        if isinstance(item, h5py.Group):
            attrs = ", ".join(f"{k}={item.attrs[k]}" for k in item.attrs)
            out.append((path, f"<group> {attrs}"))
            out.extend(walk_h5(item, prefix=path + "/"))
        elif isinstance(item, h5py.Dataset):
            shape = item.shape
            dtype = item.dtype
            attrs = ", ".join(f"{k}={item.attrs[k]}" for k in item.attrs)
            summary = f"<dataset> shape={shape}, dtype={dtype}"
            if attrs:
                summary += f"   attrs: {attrs}"
            out.append((path, summary))
    return out


def diagnose(path: Path) -> dict:
    """Return a dict of diagnostic findings for one checkpoint.

    Focused on the three hypothesis discriminators:
      - File-level version attrs and bound-count (B2 stale format)
      - n_dim / n_live (B1 model-hash mismatch — should match the
        current model's dimensionality)
      - Live-point parameter min/max per dimension (B3 unphysical
        cosmology — Om0 < 0 or w0 < -1.5 = smoking gun)
    """
    findings = {"path": str(path)}
    if not path.exists():
        findings["exists"] = False
        return findings
    findings["exists"] = True
    findings["size_bytes"] = path.stat().st_size
    findings["mtime_iso"] = dt.datetime.fromtimestamp(
        path.stat().st_mtime).isoformat()

    with h5py.File(path, "r") as f:
        # File-level Nautilus version attr (B2 discriminator)
        findings["file_attrs"] = {k: _to_py(f.attrs[k]) for k in f.attrs}

        # Bound count — proxy for how far the chain advanced
        bounds = sorted([k for k in f.keys() if k.startswith("bound_")],
                        key=lambda b: int(b.split("_", 1)[1]))
        findings["n_bounds"] = len(bounds)
        findings["last_bound"] = bounds[-1] if bounds else None

        # Pull n_dim and any version-related fields off bound_1 (the
        # first NautilusBound — bound_0 is just the unit cube).
        if "bound_1" in f:
            b1 = f["bound_1"]
            findings["bound_1_attrs"] = {k: _to_py(b1.attrs[k]) for k in b1.attrs}

        # Sampler / live points (B1 + B3)
        if "sampler" in f:
            samp = f["sampler"]
            findings["sampler_attrs"] = {k: _to_py(samp.attrs[k]) for k in samp.attrs}
            findings["sampler_keys"] = list(samp.keys())

            # Look for live-point datasets — likely 'live_points',
            # 'live_log_l', 'live_x' etc.
            for key in samp.keys():
                obj = samp[key]
                if isinstance(obj, h5py.Dataset):
                    sample_summary = {
                        "shape": list(obj.shape),
                        "dtype": str(obj.dtype),
                    }
                    # If 2D + first dim looks like n_live, also dump per-column
                    # min/max to flag B3 (extreme parameter values).
                    if obj.ndim == 2 and obj.shape[0] > 10:
                        try:
                            data = obj[:]
                            sample_summary["per_col_min"] = (
                                data.min(axis=0).tolist())
                            sample_summary["per_col_max"] = (
                                data.max(axis=0).tolist())
                        except Exception as e:
                            sample_summary["error"] = f"{type(e).__name__}: {e}"
                    elif obj.ndim == 1:
                        try:
                            data = obj[:]
                            sample_summary["min"] = float(data.min())
                            sample_summary["max"] = float(data.max())
                            sample_summary["mean"] = float(data.mean())
                        except Exception as e:
                            sample_summary["error"] = f"{type(e).__name__}: {e}"
                    findings.setdefault("sampler_datasets", {})[key] = sample_summary

    return findings


def _to_py(v):
    """Convert numpy scalars / arrays to Python primitives for JSON-friendly print."""
    if hasattr(v, "tolist"):
        return v.tolist()
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    return v


def print_report(findings: dict) -> None:
    print("=" * 78)
    print(f"checkpoint: {findings['path']}")
    if not findings.get("exists"):
        print("  *** FILE DOES NOT EXIST ***")
        return
    size_mb = findings["size_bytes"] / 2**20
    print(f"  size:        {size_mb:.1f} MB")
    print(f"  last mtime:  {findings['mtime_iso']}")
    if findings["file_attrs"]:
        print(f"  file attrs:  {findings['file_attrs']}")
    print(f"  bound count: {findings.get('n_bounds')}  (last: {findings.get('last_bound')})")
    if "bound_1_attrs" in findings:
        b1 = findings["bound_1_attrs"]
        print(f"  bound_1:     n_dim={b1.get('n_dim')}, "
              f"n_neural_bounds={b1.get('n_neural_bounds')}, "
              f"n_reject={b1.get('n_reject')}, n_sample={b1.get('n_sample')}")
    if "sampler_attrs" in findings:
        sa = findings["sampler_attrs"]
        # Print only the small/interesting ones
        small = {k: v for k, v in sa.items()
                 if not (isinstance(v, list) and len(v) > 10)}
        print(f"  sampler attrs (small): {small}")
        if "sampler_keys" in findings:
            print(f"  sampler keys: {findings['sampler_keys']}")
    if "sampler_datasets" in findings:
        for key, info in findings["sampler_datasets"].items():
            print(f"  sampler/{key}: shape={info['shape']}, dtype={info['dtype']}")
            if "per_col_min" in info:
                # Just print first 8 dims to keep output manageable
                mn = info["per_col_min"][:8]
                mx = info["per_col_max"][:8]
                print(f"     per-col min (first 8): {[f'{x:.3g}' for x in mn]}")
                print(f"     per-col max (first 8): {[f'{x:.3g}' for x in mx]}")
                if len(info["per_col_min"]) > 8:
                    # Last few dims are most likely cosmology
                    mn_last = info["per_col_min"][-3:]
                    mx_last = info["per_col_max"][-3:]
                    print(f"     per-col min (last 3):  {[f'{x:.3g}' for x in mn_last]}")
                    print(f"     per-col max (last 3):  {[f'{x:.3g}' for x in mx_last]}")
            elif "min" in info:
                print(f"     min={info['min']:.4g}  max={info['max']:.4g}  mean={info['mean']:.4g}")
    print()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", nargs="?", type=Path,
                   help="Path to checkpoint.hdf5. If omitted, scans the "
                        "known truth_fc stuck checkpoints relative to "
                        "REPO_ROOT (defaults to current dir).")
    p.add_argument("--repo-root", type=Path, default=Path.cwd(),
                   help="Repo root for resolving default checkpoint paths "
                        "(default: cwd)")
    args = p.parse_args()

    if args.checkpoint:
        targets = [args.checkpoint]
    else:
        targets = [args.repo_root / rel for rel in DEFAULT_CHECKPOINTS_RELATIVE]

    print(f"Scanning {len(targets)} checkpoint(s)...")
    print()
    for t in targets:
        try:
            findings = diagnose(t)
        except Exception as e:
            findings = {"path": str(t), "exists": True,
                        "error": f"{type(e).__name__}: {e}"}
            print(f"\n*** ERROR opening {t}: {findings['error']} ***\n")
            continue
        print_report(findings)

    print()
    print("Hypothesis discrimination guide:")
    print("  - If file_attrs / top_level_keys vary across checkpoints from")
    print("    the same fit family: B1 (model-hash mismatch).")
    print("  - If a 'version' or 'nautilus_version' attr disagrees with the")
    print("    currently-installed nautilus.__version__: B2 (stale format).")
    print("  - If live-point parameter ranges include extreme values (Om0<0,")
    print("    w0<-1.5, etc.): B3 (bad sample point — fix is prior bounds).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
