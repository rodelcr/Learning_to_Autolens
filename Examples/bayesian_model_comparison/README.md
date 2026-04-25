# Example: Bayesian Model Comparison via Log-Evidence

## Status

◐ **In progress** — pedagogy notebook + worked-examples placeholders shipped, awaiting P3+P4 Cannon results to fill the empirical table.

## What this example is for

Every other Example in this collection has, at some point, casually said *"the Bayes factor is e^{151}, this strongly favours the model with shear"* or similar. This example **stops to explain what that actually means**, double-checking the references, walking through Occam's razor, and showing what makes log-evidence comparison trustworthy versus when it can mislead.

It is a **pedagogy notebook**, not a fitting notebook — there's no new lens to fit. The worked examples reuse Cannon-pulled results from `compound_lens/`, `disky_spiral_lens/`, `mge_to_physical/`, and `compound_lens_zoo/`.

## What it covers

1. **Bayes' theorem and the marginal likelihood** — why the integral $Z = \int \mathcal{L}(\theta) \pi(\theta) \, d\theta$ encodes both *fit quality* and *prior volume*.
2. **The Bayes factor** $B_{12} = Z_1 / Z_2$ and the log form $\Delta \ln Z$.
3. **The Jeffreys (1961) and Kass & Raftery (1995) scales** with proper citations and the *factor-of-2 trap* between $\ln B$ and $2\ln B$.
4. **Occam's razor** — why a more flexible nested model can lose despite achieving higher $\mathcal{L}_\mathrm{max}$. The intuition (extra prior volume = automatic penalty) plus a 1-D analytic worked example.
5. **Worked examples from this repo** —
   - compound_lens v3 (no shear, Iso) vs v4 (Iso + shear): Δlog Z = +151. *strong* on Trotta's scale.
   - compound_lens v4 vs EPL (PowerLaw + shear): Δlog Z = +144 in favour of EPL. *strong*.
   - compound_lens EPL vs PIX (pixelised source, fixed mass): Δlog Z = +107 in favour of PIX. *strong*.
   - disky_spiral_lens single-Sersic vs two-Sersic light: Δlog Z = +61,969. *decisive* (Jeffreys); off the chart on every scale.
   - (pending) mge_to_physical stars-only vs stars+DM (P3 result).
   - (pending) compound_lens_zoo single-source vs two-source on a steep-γ′ mock (P4 exercise 4).
6. **Confounders** — numerical accuracy of nested-sampling Z (Lange 2023's Nautilus achieves ~0.01–0.03 on benchmark problems, but real-world deflectionless degeneracies degrade this), prior dependence (the most common confounder for students), model misspecification (when neither model is the truth).
7. **When NOT to use Bayes factors** — improper priors, prior-dominated regimes, model averaging instead, p-values for hypothesis testing where appropriate.

## Notebook

`01_bayesian_model_comparison.ipynb` — the pedagogy notebook with the worked-example numbers slotted from the other Examples.

## References (all verified)

```bibtex
@book{Jeffreys1961,
  author = {Jeffreys, Harold},
  title = {Theory of Probability},
  edition = {3},
  publisher = {Oxford University Press},
  year = {1961},
  note = {Bayes-factor scale in Appendix B}
}

@article{KassRaftery1995,
  author = {Kass, Robert E. and Raftery, Adrian E.},
  title = {Bayes Factors},
  journal = {Journal of the American Statistical Association},
  volume = {90}, number = {430}, pages = {773--795}, year = {1995},
  doi = {10.1080/01621459.1995.10476572}
}

@article{Trotta2008,
  author = {Trotta, Roberto},
  title = {Bayes in the sky: Bayesian inference and model selection in cosmology},
  journal = {Contemporary Physics},
  volume = {49}, number = {2}, pages = {71--104}, year = {2008},
  doi = {10.1080/00107510802066753},
  eprint = {0803.4089}, archivePrefix = {arXiv}
}

@article{Suyu2006,
  author = {Suyu, S. H. and Marshall, P. J. and Hobson, M. P. and Blandford, R. D.},
  title = {A Bayesian analysis of regularized source inversions in gravitational lensing},
  journal = {MNRAS}, volume = {371}, number = {2}, pages = {983--998}, year = {2006},
  doi = {10.1111/j.1365-2966.2006.10733.x},
  eprint = {astro-ph/0601493}
}

@article{VegettiKoopmans2009,
  author = {Vegetti, S. and Koopmans, L. V. E.},
  title = {Bayesian strong gravitational-lens modelling on adaptive grids},
  journal = {MNRAS}, volume = {392}, number = {3}, pages = {945--963}, year = {2009},
  doi = {10.1111/j.1365-2966.2008.14005.x}
}

@article{Vegetti2010,
  author = {Vegetti, S. and Koopmans, L. V. E. and Bolton, A. and Treu, T. and Gavazzi, R.},
  title = {Detection of a dark substructure through gravitational imaging},
  journal = {MNRAS}, volume = {408}, number = {4}, pages = {1969--1981}, year = {2010},
  doi = {10.1111/j.1365-2966.2010.16865.x}
}

@article{Lange2023,
  author = {Lange, Johannes U.},
  title = {nautilus: boosting Bayesian importance nested sampling with deep learning},
  journal = {MNRAS}, volume = {525}, number = {2}, pages = {3181--3194}, year = {2023},
  doi = {10.1093/mnras/stad2441}
}
```

### Note on the *factor-of-2 trap*

The literature uses two conflicting conventions for the same scale:

- **Jeffreys (1961), Trotta (2008)**: thresholds in $|\ln B|$ (natural log Bayes factor, in nats). Trotta Table 1: weak 1.0, moderate 2.5, strong 5.0.
- **Kass & Raftery (1995)**: thresholds in $2\ln B$ (the *deviance scale*, motivated by analogy with the chi-squared distribution). Their Table 1: 0–2 "not worth more than a bare mention", 2–6 "positive", 6–10 "strong", >10 "very strong".

Both are used. The factor-of-2 means a Kass-Raftery "strong" (6 in $2\ln B$) is the same as a Trotta "moderate" (3 in $\ln B$). The notebook tabulates both side-by-side to avoid the confusion.

## Why this example exists

Every PyAutoLens fit produces a `log_evidence`. Almost every workflow involves comparing two or more fits via that number. Yet it is the most commonly *misinterpreted* output of nested sampling, and a notebook that doesn't carefully distinguish the conventions, address Occam's razor, and call out when log-evidence isn't trustworthy will leave students with confidently wrong intuitions. The Examples that *use* Bayes factors (compound_lens, disky_spiral_lens, etc.) point readers here for the explanation, freeing them to be more concise about the conclusion (`"e^151 favours v4 — see bayesian_model_comparison/ for the scale"`).
