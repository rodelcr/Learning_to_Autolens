# Paper-Repro Spec 04 — Cross-Validation + Promotion Plan

**Date:** 2026-05-18
**Scope:** Sec 01-03 cross-paper validation, tool-development synthesis, promotion of `private/` derivatives into public `Modules/` + `Examples/`.
**Depends on:** Specs 00-03 (all four), Phase 3 v0.97 deliverables.

---

## 1. Context

When P1, P2, and P3 land in `private/`, we have:
- Three published-number reproductions (each in BOTH stacks where applicable)
- Six fits on J0946+1006 (P2 main+subhalo+wbh × 2 stacks)
- Two fits on J0946 main lens from P3 × 2 stacks
- One 161-lens population fit × 2 samplers (autofit + NumPyro)

This spec wires those individual artefacts into:
- **One internal-consistency check** — P2 main and P3 main fits should agree on the J0946 main-lens parameters
- **One tool-development synthesis** — what did we learn about each stack from doing the same scientific work in both?
- **One promotion plan** — which `private/` outputs become public curriculum?

## 2. Goals

- Publish-quality "did we reproduce the papers?" summary
- Quantitative comparison: stack vs stack, sampler vs sampler, single-system vs population
- Tool-development feedback: surface any bugs, missing features, convention drifts in either PyAutoLens or Herculens — submit upstream as PRs if warranted
- 3 new public Examples, 2 new public Modules, 2 module extensions, 1 Learning_to_Lens module, 1 new repo-level integration doc

## 3. Non-goals

- New scientific results beyond what the papers establish
- Generic Herculens/autolens comparison benchmark suite — focus on the three specific use cases at hand
- Public-facing PR on Herculens/autolens repos — surface findings in our `docs/tool_development_report.md`; PRs are a follow-up at the user's discretion

## 4. Architecture

```
private/04_crossval_promotion/
├── validation/
│   ├── j0946_main_lens_consistency.ipynb       ← 4-way: {P2-autolens, P2-herculens, P3-autolens, P3-herculens}
│   ├── autolens_herculens_crossval.ipynb       ← per-paper summary
│   ├── p1_sampler_crossval.ipynb               ← Nautilus vs NUTS for P1
│   └── tool_development_findings.md
├── promotion/
│   ├── promotion_plan.md                       ← module/example additions, with timing
│   ├── module_16_outline.md                    ← Module 16 (Hierarchical Bayesian Cosmography) full outline
│   ├── module_17_outline.md                    ← Module 17 (Dynamical Mass Decomposition) full outline
│   ├── module_14_tspl_extension_outline.md     ← Module 14 §"TSPL Extension"
│   ├── ltl_09.5_outline.md                     ← Learning_to_Lens 09.5_Hierarchical_Cosmography
│   ├── ltl_11_outline.md                       ← Learning_to_Lens 11_Stellar_Dynamics_Jeans
│   └── HERCULENS_INTEGRATION_draft.md          ← repo-level doc (draft for public commit)
└── README.md
```

## 5. The four-way J0946 consistency check

P2 (TSPL fit) and P3 (DSPL fit) both fit the J0946+1006 imaging. Where the models overlap (main-lens mass profile, lens light decomposition), both papers' posteriors should agree at 1-2σ. With both stacks per paper, we have 4 posteriors on the same parameters:

| Parameter | P2-autolens | P2-herculens | P3-autolens | P3-herculens | Cross-paper σ |
|---|---|---|---|---|---|
| θ_E (main lens) | … | … | … | … | … |
| γ′ (slope) | … | … | … | … | … |
| ell_comps_0 | … | … | … | … | … |
| ell_comps_1 | … | … | … | … | … |

**Pass criterion:** All 4 posteriors agree at <2σ on the main-lens parameters. Failure modes that surface here:
- TSPL model is mis-specified in one stack → cross-paper disagreement
- DSPL model in P3 is biased by ignoring the third source → P2/P3 disagreement
- One stack has a numerical bug → in-paper disagreement between stacks

## 6. Tool-development synthesis (`tool_development_findings.md`)

Sections:
1. **Performance** — wall-clock per fit per stack on identical hardware. Expected: Herculens-NUTS on A100 ~3-10× faster than autolens-Nautilus on 32-core CPU for likelihood-bottlenecked DSPL fits; less advantage for the point-source-dominated TSPL fits where the bottleneck is the multi-image solver, not the likelihood evaluation.
2. **Convention drift** — every per-profile sign/parameter-name disagreement found by `herculens_bridge.py`. Catalogue them.
3. **API gaps** — features one stack has but the other doesn't: e.g., autolens's `lp_basis.Basis` MGE pattern, Herculens's `multi_plane` flexibility, autolens's `af.FactorGraphModel`, Herculens's automatic differentiation for hyperparameter gradient.
4. **Sampler stability** — where Nautilus and NUTS disagree (P1 case), document the cause: prior boundary effects, mode-coverage differences, NUTS step-size adaptation issues. Don't assume either sampler is "right" — both can be wrong.
5. **Documentation gaps** — which features in each stack are under-documented (where we had to read source code to figure them out). PR-targets for upstream improvement.

## 7. Promotion plan

### 7.1 New public Examples

| Example dir | Status criterion | Promotion sources |
|---|---|---|
| `Examples/hierarchical_population_cosmography/` | Both samplers strict-PASS on 30-lens synthetic + real 161-lens within 1σ of Li+2023 | private/2307_… code + Module 16 |
| `Examples/tspl_jackpot/` | Both stacks recover 5.9σ subhalo + ΛCDM-consistent (M_sub, c_sub) | private/2309_… code + Module 14 extension |
| `Examples/dspl_jackpot_imf_nfw/` | Both stacks recover Li+2026 (M_*, M_h, α, γ_inner) at <1σ | private/2602_… code + Module 17 |

### 7.2 New public Learning_to_Autolens modules

| Module | Sources |
|---|---|
| **Module 16** — Hierarchical Bayesian Cosmography | Spec 01 §10.1 (P1 work) |
| **Module 17** — Dynamical Mass Decomposition via Jeans | Spec 03 §10.1 (P3 work) |

### 7.3 Module extensions

| Module | Extension |
|---|---|
| **Module 14** — Compound Multi-Plane | §"TSPL Extension" — recursive multi-plane lens equation for N sources, β_jk cross-terms; J0946 worked example. From Spec 02 §11.1. |

### 7.4 Learning_to_Lens additions

| Module | Status |
|---|---|
| `09.5_Hierarchical_Cosmography.wl` | NEW — derives the σ_v ↔ θ_E ↔ cosmology mapping symbolically. Spec 01 §10.2 |
| `11_Stellar_Dynamics_Jeans/` | NEW (full module) — anisotropic Jeans, σ_v aperture projection, Mamon & Łokas 2005 kernel. Spec 03 §10.2 |

### 7.5 New repo-level doc

`HERCULENS_INTEGRATION.md` — public summary of the dual-stack work: rationale, what's gained, what to install, where to learn more (without revealing private/ details). Becomes the entry point for any future user wanting to use both stacks in this curriculum.

## 8. Cross-references (gr-lensing-intuition)

The tool-development synthesis explicitly references gr-lensing-intuition's:
- **"Software packages"** section in `references/domain-knowledge.md` (PyAutoLens, lenstronomy, Herculens conventions table) — extended with our findings
- **"Common pitfalls"** section — each new pitfall found during the dual-stack work goes here
- **"Order-of-magnitude estimates"** — extended with computational order-of-magnitude (CPU-h per fit) for the four reproduction targets

## 9. Sequencing

1. Specs 01, 02, 03 complete (independently); their cross-paper artefacts land
2. `validation/j0946_main_lens_consistency.ipynb` — laptop-only
3. `validation/autolens_herculens_crossval.ipynb` — laptop-only, ~2h to render
4. `tool_development_findings.md` — write-up; ~1 day
5. Module 16 + Module 17 drafts — ~2 days each
6. Promotion to public — refactoring of `private/code/` → `Examples/<name>/code/`, tested for executability outside `private/`
7. Final commit batch tags `v0.98-alpha` once the promotion lands

## 10. Risks + mitigations

- **Risk:** Herculens fits disagree with autolens by > 1σ. **Mitigation:** This IS the methodology contribution; document it. Investigate (a) bridge bugs (b) sampler issues (c) prior-bound effects.
- **Risk:** ESO MUSE data restricted-access / proprietary period. **Mitigation:** Use only public products; if proprietary, document limitation and use Sonnenfeld 2012's published source-position table.
- **Risk:** A100 partition unavailable when needed. **Mitigation:** Herculens runs on CPU too (no NUTS speedup); plan fallback to autolens Nautilus stack.
- **Risk:** BELLS catalogue not on Vizier → P1 stays at 141 lenses, not 161. **Mitigation:** Document the discrepancy; Li+2023 result depends weakly on the BELLS sample.

## 11. Timeline

- Day 1-2: J0946 4-way consistency notebook (assumes Specs 01-03 results in hand)
- Day 2-4: `autolens_herculens_crossval.ipynb` per paper
- Day 4-5: P1 sampler crossval
- Day 5-7: tool_development_findings.md synthesis
- Day 7-10: Module 16, Module 17 outlines → drafts → notebooks
- Day 10-12: Module 14 + LtL extensions
- Day 12-14: public promotion (Example refactoring, README cross-links)
- Day 14: `HERCULENS_INTEGRATION.md` + final commit + v0.98-alpha tag
- Total: ~2 weeks after Specs 01-03 land
