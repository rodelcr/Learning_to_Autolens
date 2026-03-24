(* ============================================================ *)
(* 04_search_chaining_and_priors.wl                             *)
(* ============================================================ *)
(*                                                              *)
(* Purpose:                                                     *)
(*   Symbolic exploration of the concepts behind search         *)
(*   chaining and the SLaM pipeline:                            *)
(*     - Curse of dimensionality: prior volume scaling          *)
(*     - Gaussian prior passing from posteriors                 *)
(*     - Information gain from chaining                         *)
(*     - Power-law mass profile (SLaM MASS TOTAL upgrade)       *)
(*     - SIE → PowerLaw parameter mapping                       *)
(*                                                              *)
(* References:                                                  *)
(*   - Nightingale, Dye & Massey (2018), Sec. 6                 *)
(*   - Skilling (2004), Nested Sampling                         *)
(*   - Congdon & Keeton (2018), Sec. 4.4                        *)
(*                                                              *)
(* Run: wolframscript -file 04_search_chaining_and_priors.wl    *)
(*                                                              *)
(* Author: Rodrigo Córdova Rosado (Harvard CfA)                 *)
(* Built with: Claude Code                                      *)
(* ============================================================ *)

Print[""];
Print["============================================================"];
Print["Module 04: Search Chaining & Priors — Symbolic Verification"];
Print["============================================================"];
Print[""];

(* ============================================================ *)
(* 1. CURSE OF DIMENSIONALITY                                   *)
(* ============================================================ *)
(* The fraction of prior volume containing the posterior shrinks *)
(* exponentially with dimension N:                               *)
(*   V_post / V_prior ~ (σ_post / Δ_prior)^N                   *)
(* ============================================================ *)

Print["--- 1. Curse of Dimensionality ---"];
Print[""];

(* Fraction of prior volume occupied by posterior *)
fractionVolume[sigmaOverDelta_, nDim_] := sigmaOverDelta^nDim;

Print["Posterior volume fraction (σ/Δ = 0.01):"];
dims = {5, 10, 15, 20, 25, 30};
Do[
  frac = fractionVolume[0.01, n];
  Print["  N = ", n, " params: V_post/V_prior = ",
    ScientificForm[N[frac], 2]],
  {n, dims}
];
Print[""];
Print["At N=25, the posterior occupies ~10^{-50} of the prior!"];
Print["This is why uninformed searching in high dimensions fails."];
Print[""];

(* ============================================================ *)
(* 2. GAUSSIAN PRIOR PASSING                                    *)
(* ============================================================ *)
(* When Search 1 finds θ_E = 1.62 ± 0.05, Search 2 uses        *)
(* π(θ_E) = N(1.62, σ_width) as its prior.                     *)
(*                                                              *)
(* The effective search volume shrinks dramatically:             *)
(*   V_informed / V_uninformed = (σ_width / Δ_prior)^N_passed   *)
(* ============================================================ *)

Print["--- 2. Prior Passing: Information Gain ---"];
Print[""];

(* Example: SIE model with 7 mass parameters *)
(* Uninformed: each parameter spans Δ ~ 5 units *)
(* Informed: each parameter has σ ~ 0.1 from Search 1 *)
nMassParams = 7;
deltaUninformed = 5.0;
sigmaInformed = 0.1;

volumeRatio = (sigmaInformed / deltaUninformed)^nMassParams;
Print["Mass model (7 params):"];
Print["  Uninformed prior volume ∝ Δ^7 = ", N[deltaUninformed^nMassParams]];
Print["  Informed prior volume ∝ σ^7 = ", ScientificForm[N[sigmaInformed^nMassParams], 2]];
Print["  Volume reduction: ", ScientificForm[N[volumeRatio], 2]];
Print["  → Search 2 explores ", ScientificForm[N[1/volumeRatio], 2],
  "× smaller volume"];
Print[""];

(* Information gain in bits *)
(* KL divergence: Gaussian posterior → Gaussian prior *)
(* D_KL = (1/2)[ln(σ_prior²/σ_post²) + σ_post²/σ_prior² - 1 + (μ_post-μ_prior)²/σ_prior²] *)
(* For centered priors (μ_prior = μ_post), simplifies to: *)
klGaussian[sigmaPrior_, sigmaPost_] :=
  (1/2) (Log[sigmaPrior^2/sigmaPost^2] + sigmaPost^2/sigmaPrior^2 - 1);

Print["Information gain per parameter (KL divergence):"];
Print["  Uninformed (σ=5) → Posterior (σ=0.05): ",
  NumberForm[N[klGaussian[5.0, 0.05]], 4], " nats = ",
  NumberForm[N[klGaussian[5.0, 0.05] / Log[2]], 4], " bits"];
Print[""];

(* ============================================================ *)
(* 3. POWER-LAW MASS PROFILE                                    *)
(* ============================================================ *)
(* The PowerLaw generalizes the SIE:                             *)
(*   κ(θ) = (3-γ')/2 × (θ_E/θ)^(γ'-1)                         *)
(* where γ' is the density slope.                                *)
(*   γ' = 2: isothermal (SIE)                                   *)
(*   γ' > 2: steeper than isothermal                             *)
(*   γ' < 2: shallower than isothermal                           *)
(* (cf. C&K Sec. 4.4)                                           *)
(* ============================================================ *)

Print["--- 3. Power-Law Mass Profile ---"];
Print[""];

(* Convergence *)
kappaPL[\[Theta]_, \[Theta]E_, \[Gamma]_] :=
  (3 - \[Gamma])/2 * (\[Theta]E/\[Theta])^(\[Gamma] - 1);

(* Verify: γ' = 2 recovers SIS convergence κ = θ_E/(2θ) *)
kappaCheck = Simplify[kappaPL[\[Theta], \[Theta]E, 2]];
Print["Power-law convergence: κ(θ) = (3-γ')/2 × (θ_E/θ)^(γ'-1)"];
Print["  At γ' = 2: κ = ", kappaCheck, "  (SIS ✓)"];
Print[""];

(* Mean convergence inside θ_E *)
Print["Mean convergence inside θ_E:"];
Do[
  meanK = (1/(Pi tE^2)) * Integrate[
    kappaPL[\[Theta], tE, gam] * 2 Pi \[Theta],
    {\[Theta], 0, tE},
    Assumptions -> tE > 0
  ];
  Print["  γ' = ", gam, ": <κ>(θ_E) = ", Simplify[meanK]],
  {gam, {1.5, 2.0, 2.5}}
];
Print["  (All equal 1 — θ_E is defined by <κ> = 1 for any slope.)"];
Print[""];

(* Deflection angle *)
(* For a circular power-law: α(θ) = θ_E × (θ_E/θ)^(γ'-2) *)
alphaPL[\[Theta]_, \[Theta]E_, \[Gamma]_] :=
  \[Theta]E * (\[Theta]E/\[Theta])^(\[Gamma] - 2);

Print["Deflection angle: α(θ) = θ_E × (θ_E/θ)^(γ'-2)"];
Print["  At γ' = 2: α = ", Simplify[alphaPL[\[Theta], \[Theta]E, 2]],
  "  (constant, SIS ✓)"];
Print["  At γ' = 1: α = ", Simplify[alphaPL[\[Theta], \[Theta]E, 1]],
  "  (point mass)"];
Print[""];

(* ============================================================ *)
(* 4. SLaM PARAMETER COUNT PER STAGE                            *)
(* ============================================================ *)

Print["--- 4. SLaM Parameter Budget ---"];
Print[""];

stages = {
  {"SOURCE LP", "Lens light (7) + SIE (5) + shear (2) + source Sérsic (7)", 21},
  {"SOURCE PIX run_1", "Mass (5) + shear (2) + reg (1)", 8},
  {"SOURCE PIX run_2", "Mesh (1-2) + reg (2-3)", 4},
  {"LIGHT LP", "Lens light (7)", 7},
  {"MASS TOTAL", "Mass (6) + shear (2) + source reg (1-3)", 10}
};

Print["  Stage                  Free params   Strategy"];
Print["  ─────────────────────  ───────────   ────────────────────"];
Do[
  Print["  ", PaddedForm[st[[1]], {22, 0}], "  ~",
    PaddedForm[st[[3]], {2, 0}], "          ",
    StringTake[st[[2]], Min[40, StringLength[st[[2]]]]]],
  {st, stages}
];
Print[""];
Print["  Compare: fitting everything at once → ~26 params"];
Print["  SLaM max per stage: ~21 (SOURCE LP), but with broad priors only once"];
Print[""];

(* ============================================================ *)
(* 5. NESTED SAMPLING SCALING                                   *)
(* ============================================================ *)
(* Nested sampling efficiency scales roughly as:                 *)
(*   N_evals ~ n_live × N_dim × ln(V_prior / V_posterior)       *)
(* Chaining reduces both N_dim and V_prior for later searches.   *)
(* ============================================================ *)

Print["--- 5. Nested Sampling Scaling ---"];
Print[""];

nLive = 100;
nEvalsEstimate[nDim_, volumeRatio_] := nLive * nDim * Log[1/volumeRatio];

(* Uninformed: 26 params, σ/Δ = 0.01 *)
nEvalFull = nEvalsEstimate[26, 0.01^26];
Print["Full model (26 params, uninformed):"];
Print["  ~", Round[nEvalFull], " likelihood evaluations"];

(* Chained: 8 params mass search, σ/Δ = 0.1 (informed from SOURCE LP) *)
nEvalChained = nEvalsEstimate[8, 0.1^8];
Print["Chained mass search (8 params, informed):"];
Print["  ~", Round[nEvalChained], " likelihood evaluations"];
Print["  Speedup: ~", Round[nEvalFull / nEvalChained], "×"];
Print[""];

Print["============================================================"];
Print["All verifications complete."];
Print["============================================================"];
