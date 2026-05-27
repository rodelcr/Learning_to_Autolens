# Module 15 — Pedagogical Audit & Split

*Audit date: 2026-05-27. Author: split task.*

## Scope

Module 15 (`15_radial_arcs.ipynb`) grew organically from a 5-section
axisymmetric primer (§1–§5) to a 10-section / 39-cell notebook spanning
analytic numpy, PyAutoLens cross-class comparisons, an idealized spiral-lens
demo with the SED_inferred_color_images base_class parameters, an NSC-vs-SMBH
critical test, a "fake DESJ0206" with two source blobs, and a revised §4 +
§5. The notebook itself flags the need for a split *twice* (§3.9 + §3.10
both say "deferred to follow-on spec").

This file is that audit. It is paired with the actual split into
`Modules/15a_Radial_Arcs_Analytic_Foundation/` and
`Modules/15b_Radial_Arcs_PyAutoLens_Realistic/`.

## Per-section claims and how they flow

| § | One-line claim | Stack | Status |
|---|---|---|---|
| 1 | Caustic / critical-curve definitions; $\theta_r$ varies with γ′ | numpy | Foundational. Flows from CLAUDE.md "Module 03 prereq" cleanly. |
| 2 | Magnification asymptotics: $\mu\propto 1/d$ tangentially, $1/\sqrt d$ radially | numpy | Standalone analytic; the SIS-has-no-radial-curve remark is the critical setup for §3 sweeps. |
| 3 | $\beta_r(\gamma')$ slope $\Rightarrow$ radial arcs constrain γ′ | numpy | The output `d(beta_r)/d(gamma) = +nan` because the central-difference straddles γ=2 (where $r_r$ vanishes). Minor cosmetic bug; doesn't affect the figure. |
| 3.5 | 2D ray-traced demo: source-position + γ′ sweeps | numpy | Excellent bridge — first 2D images, still no PyAutoLens. The "AGEL spiral lenses are sub-isothermal" callout is the lead-in to §3.6. |
| 3.6 | "Same total mass, different shape" via composite PL+SMBH | numpy | The first M_BH content. Topology change (3 images → 2) is the headline. Mass-shape callout box is load-bearing. |
| 3.7 | Spiral-source-style structured-source extension of §3.6 | numpy | Logical extension, still no autolens. Three matched-tangential models. Contains the first "TODO: full Bayesian degeneracy break" marker — appropriate. |
| 3.8 | 4-class PyAutoLens comparison (PL/BPL/NSC/SMBH) | PyAutoLens | **The first PyAutoLens cell.** Transition is *abrupt* — no "we now switch from analytic to autolens" prose. This is the largest pedagogical gap in the notebook. The figure itself is excellent (5×3 grid). |
| 3.9 | Student base_class 1–4 morphologies + ±SMBH | PyAutoLens | Pivots from "generic cross-class" to "exact student parameters." Cite-back to memory `reference_sed_inferred_color_images` is appropriate. |
| 3.9c | NSC vs SMBH at matched total mass | PyAutoLens | Cleanly answers the "is NSC distinguishable from PointMass?" question §3.9 implicitly raises. |
| 3.10 | "Fake DESJ0206" with EPL + 2 source blobs | PyAutoLens | Pivots again from idealized to a target-realistic demo. Necessary epilogue but feels tacked-on inside §3 numbering. |
| 4 | "Spiral lenses break degeneracy from lensing alone" + cumulative-κ̄ plot | numpy + text | Revised 2026-05-26. Strong narrative tying §3.6–§3.9 to N+23/Shajib+/Ferrami+. |
| 5 | Handoff to `Examples/radial_arc_smbh`, Mod 13, AGEL papers | text | Standard. |

## Where the flow breaks

**Gap 1 — Theory-first violated at §3.8.** Per repo CLAUDE.md, every code
cell is preceded by a markdown cell explaining the physics. §3.8's markdown
header sells the *physical* claim (4 mass classes, matched outer ring,
visual distinguishability) but does NOT explain *why we are now using
PyAutoLens*. A student reading top-to-bottom hits a wall: 7 analytic-numpy
sections, then suddenly `import autolens as al` with no transition.

**Gap 2 — Sub-section numbering inconsistency.** §3.5 → §3.6 → §3.7 → §3.8
→ §3.9 → §3.9c → §3.10 has no consistent depth pattern. §3.9 is treated as
a major section in headings (`## 3.9`) but §3.9c is a `### 3.9c`. The
numbering implies §3.9c is a sub-result of §3.9 but its critical claim
(NSC vs SMBH distinguishability) is independent.

**Gap 3 — §4 sits awkwardly.** The revised §4 framing ("spiral lenses break
the degeneracy from lensing alone") is the *capstone* of §3.6–§3.9 + §3.10.
But the cumulative-κ̄ code in §4 demonstrates the *original* (point-source,
SLACS-era) degeneracy — i.e. it sets up §4's thesis by showing the
degeneracy is real *in the single-radius slice*. The reader has to track
both framings simultaneously. Splitting helps: 15a keeps the
"degeneracy is real" framing as a closing remark; 15b's §4 emphasises the
"spiral lenses transcend it" framing as the connection to imaging-fit
posteriors.

**Gap 4 — Two "deferred-split" notices inside the notebook.** §3.9 and
§3.10 each contain a markdown block titled "Pedagogical audit + module split
— deferred to follow-on spec." These break the reading flow and are now
obsolete after this split.

**Gap 5 — §3 cell #8 prints `nan` for $d\beta_r/d\gamma'$.** Cosmetic; the
central-difference window straddles γ=2 where the radial caustic disappears.
The figure annotation also shows `nan`. Easy fix: use a one-sided difference
or restrict to γ<2. *Not* fixing as part of this split (out of scope), but
flagging for a follow-up touch in 15a.

## Sections that don't fight each other

§3.6 and §3.7 are complementary (point-source vs extended source treatment
of the same composite mass model). §3.9 and §3.9c are complementary
(±SMBH vs NSC-vs-SMBH on the same lens base). §3.8 vs §3.9: §3.8 is
generic + comparative, §3.9 is target-specific + isolating; they
reinforce.

The one near-overlap is §3.6 (numpy) and §3.7 (numpy) both demonstrating
"radial arc breaks degeneracy" with composite mass — but §3.6 uses a point
source (topology-change demo) and §3.7 uses a structured source
(quantitative RMS-by-region demo). The framing is distinct enough; keep
both.

## Recommendation — split at the §3.5/§3.6 boundary

| Notebook | Sections retained | Renumbering |
|---|---|---|
| **15a — Analytic Foundation** | original §1, §2, §3, §3.5; truncated §4 ("degeneracy is real in the single-radius slice"); §5 handoff to 15b | Heads remain §1, §2, §3, §3.5 (carrying the in-text "3.5" for continuity), then renamed §4 + §5. |
| **15b — PyAutoLens Realistic** | original §3.6, §3.7, §3.8, §3.9, §3.9c, §3.10; revised §4 ("spiral lenses transcend the degeneracy"); §5 handoff back / forward | Renumber §3.6 → §1, §3.7 → §2, §3.8 → §3, §3.9 → §4 (with §3.9c → §4c, §3.10 → §5), revised §4 → §6, §5 → §7. |

**Renumbering choice rationale.** Inside 15a, the inner-section numbering
(§1–§3.5) is *already* a clean enumeration of the original module's
analytic core, and the original "3.5" remains motivated by being a half-step
into the full simulation. We keep it. Inside 15b, the §3.6–§3.9c labels
are no longer meaningful once §1–§3.5 are absent, so we renumber from §1.

**Helper module placement.** `_spiral_lens_helpers.py` is PyAutoLens-only
(imports `autolens as al`). 15a never touches it. Copy the helpers file
into `15b/` and leave 15a clean. The original `15_radial_arcs.ipynb` keeps
the file at its original path.

**Original notebook fate.** Add a top-of-notebook markdown banner pointing
to 15a + 15b. Leave the rest intact — the original is referenced from
`HANDOFF_2026_05_27.md`, `PROGRESS_LOG.md`, and the v0.96 release notes;
deleting it would break those provenance links.

## What this audit deliberately did NOT do

- Did not fix the §3 `nan` print (out of scope; cosmetic).
- Did not regenerate the original notebook's figures (the original is
  retained as-is).
- Did not split `Examples/radial_arc_smbh/` — that's a separate iteration.
- Did not introduce cross-imports between 15a and 15b — 15a is
  pure numpy; 15b is self-contained PyAutoLens.

## Verification

- Both child notebooks executed via `jupyter execute` from the `autolens`
  env exit 0. Runtimes recorded in the task report.
- `_spiral_lens_helpers.py` accessible from 15b directory (copied).
- This audit file is at the path
  `Modules/15_Radial_Arcs_Caustic_Topology/PEDAGOGICAL_AUDIT_2026_05_27.md`.
- Curriculum table in repo-root `CLAUDE.md` updated: the single row
  `| 15 | Radial Arcs ... | ✓ shipped (v0.96) ...|` is replaced by two
  rows for 15a + 15b.
