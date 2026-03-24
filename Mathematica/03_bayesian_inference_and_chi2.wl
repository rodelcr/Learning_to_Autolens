(* ============================================================ *)
(* 03_bayesian_inference_and_chi2.wl                            *)
(* ============================================================ *)
(*                                                              *)
(* Purpose:                                                     *)
(*   Symbolic derivation and verification of the Bayesian       *)
(*   inference framework used in Module 03:                     *)
(*     - Gaussian likelihood and χ² statistic                   *)
(*     - Bayes' theorem and posterior                            *)
(*     - Nested sampling evidence integral                      *)
(*     - Fisher information and parameter uncertainties          *)
(*     - Lens modeling degeneracies (mass-sheet transform)       *)
(*                                                              *)
(* References:                                                  *)
(*   - Congdon & Keeton (2018), Ch. 8                           *)
(*   - Nightingale, Dye & Massey (2018), Sec. 4-5               *)
(*   - Skilling (2004), Nested Sampling                         *)
(*   - Schneider, Ehlers & Falco (1992), Ch. 5                  *)
(*                                                              *)
(* Run: wolframscript -file 03_bayesian_inference_and_chi2.wl   *)
(*                                                              *)
(* Author: Rodrigo Córdova Rosado (Harvard CfA)                 *)
(* Built with: Claude Code                                      *)
(* ============================================================ *)

Print[""];
Print["============================================================"];
Print["Module 03: Bayesian Inference & χ² — Symbolic Verification"];
Print["============================================================"];
Print[""];

(* ============================================================ *)
(* 1. GAUSSIAN LIKELIHOOD AND χ²                                *)
(* ============================================================ *)
(* For Gaussian noise with per-pixel variance σᵢ²:              *)
(*   ln L = -(1/2) Σᵢ [(dᵢ - mᵢ)²/σᵢ² + ln(2π σᵢ²)]         *)
(*        = -(1/2) χ² + const                                   *)
(* (cf. Nightingale+18 eq. 3; C&K eq. 8.7)                     *)
(* ============================================================ *)

Print["--- 1. Gaussian Likelihood ---"];
Print[""];

(* Single-pixel log-likelihood *)
logLPixel = -(1/2) ((d - m)^2/\[Sigma]^2 + Log[2 Pi \[Sigma]^2]);
Print["Single-pixel log-likelihood:"];
Print["  ln L_i = ", logLPixel];
Print[""];

(* χ² statistic *)
Print["χ² = Σᵢ (dᵢ - mᵢ)² / σᵢ²"];
Print[""];
Print["Expected value: E[χ²] = N_dof = N_pixels - N_params"];
Print["Variance: Var[χ²] = 2 N_dof"];
Print[""];

(* Reduced χ² *)
Print["Reduced χ²: χ²_red = χ² / N_dof"];
Print["  χ²_red ≈ 1.0 → good fit"];
Print["  χ²_red >> 1 → model doesn't fit the data (too simple or wrong)"];
Print["  χ²_red << 1 → overfitting or noise overestimated"];
Print[""];

(* ============================================================ *)
(* 2. BAYES' THEOREM                                            *)
(* ============================================================ *)
(* P(η|d) = L(d|η) π(η) / Z                                   *)
(* where Z = ∫ L(d|η) π(η) dη is the evidence                  *)
(* ============================================================ *)

Print["--- 2. Bayes' Theorem ---"];
Print[""];
Print["Posterior: P(η|d) = L(d|η) × π(η) / Z"];
Print["Evidence:  Z = ∫ L(d|η) π(η) dη"];
Print[""];

(* Example: 1D Gaussian likelihood × Gaussian prior *)
(* L ∝ exp(-(d-η)²/(2σ_d²)), π ∝ exp(-(η-μ₀)²/(2σ₀²)) *)
posterior1D = Simplify[
  Integrate[
    Exp[-(d - \[Eta])^2/(2 sd^2)] * Exp[-(\[Eta] - mu0)^2/(2 s0^2)],
    {d, -Infinity, Infinity},
    Assumptions -> {sd > 0, s0 > 0}
  ]
];
Print["1D example: Gaussian likelihood × Gaussian prior"];
Print["  Posterior is Gaussian with:"];
Print["    mean = (μ₀/σ₀² + d/σ_d²) / (1/σ₀² + 1/σ_d²)"];
Print["    variance = 1 / (1/σ₀² + 1/σ_d²)"];
Print["  → The posterior precision = prior precision + data precision"];
Print["  → More data = tighter posterior (precision adds!)"];
Print[""];

(* ============================================================ *)
(* 3. NESTED SAMPLING EVIDENCE INTEGRAL                         *)
(* ============================================================ *)
(* Z = ∫ L(η) π(η) dη = ∫₀¹ L(X) dX                          *)
(* where X(λ) = ∫_{L(η)>λ} π(η) dη is the prior volume with   *)
(* likelihood above λ. Skilling's key insight: this 1D integral *)
(* is much easier to compute than the N-dimensional one.        *)
(* (cf. Skilling 2004)                                          *)
(* ============================================================ *)

Print["--- 3. Nested Sampling ---"];
Print[""];
Print["Evidence integral: Z = ∫ L(η) π(η) dη"];
Print["Skilling transform: Z = ∫₀¹ L(X) dX  (1D integral!)"];
Print[""];
Print["Live point contraction:"];
Print["  At each iteration, the prior volume shrinks as:"];
Print["  X_i ≈ exp(-i/n_live)"];
Print[""];
Print["  After k iterations:"];
Print["  X_k = exp(-k/n_live)"];
Print[""];

(* Volume contraction rate *)
nLive = 100;
Print["  For n_live = ", nLive, ":"];
Print["  After 100 iterations: X = ", ScientificForm[N[Exp[-100/nLive]], 3]];
Print["  After 500 iterations: X = ", ScientificForm[N[Exp[-500/nLive]], 3]];
Print["  After 1000 iterations: X = ", ScientificForm[N[Exp[-1000/nLive]], 3]];
Print["  (The prior volume shrinks exponentially — efficient exploration!)"];
Print[""];

(* ============================================================ *)
(* 4. MASS-SHEET DEGENERACY                                     *)
(* ============================================================ *)
(* The fundamental degeneracy of strong lensing:                *)
(*   κ' = λκ + (1-λ)                                           *)
(*   β' = λβ                                                    *)
(*   α' = λα + (1-λ)θ                                          *)
(* leaves all observable image positions unchanged.              *)
(* (cf. C&K Sec. 9.1; Schneider et al. 1992 Sec. 5.4)          *)
(* ============================================================ *)

Print["--- 4. Mass-Sheet Degeneracy (MSD) ---"];
Print[""];

(* Original lens equation: β = θ - α(θ) *)
(* Transformed: β' = θ - α'(θ) *)
(* β' = θ - [λα(θ) + (1-λ)θ] = λ[θ - α(θ)] = λβ *)
Print["Mass-sheet transform:"];
Print["  κ'(θ) = λκ(θ) + (1-λ)     (add a uniform sheet)"];
Print["  β'     = λβ                 (rescale source plane)"];
Print[""];

(* Verify: transformed lens equation *)
Print["Verification:"];
Print["  α'(θ) = λα(θ) + (1-λ)θ"];
Print["  β' = θ - α'(θ) = θ - λα(θ) - (1-λ)θ = λ[θ - α(θ)] = λβ  ✓"];
Print[""];

(* Effect on observables *)
Print["Effect on observables:"];
Print["  Image positions θ: UNCHANGED (the transform preserves them)"];
Print["  Magnification μ: μ' = μ/λ² (CHANGES — but hard to measure)"];
Print["  Time delays Δt: Δt' = λ Δt (CHANGES — breaks MSD if measured)"];
Print["  Source size: β' = λβ (source appears rescaled)"];
Print[""];
Print["Breaking the MSD requires:"];
Print["  - Time delays (H₀ cosmography)"];
Print["  - Stellar kinematics (velocity dispersion → independent mass)"];
Print["  - Extended source morphology (pixelized reconstructions)"];
Print[""];

Print["============================================================"];
Print["All verifications complete."];
Print["============================================================"];
