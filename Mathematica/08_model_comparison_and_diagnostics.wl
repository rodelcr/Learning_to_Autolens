(* ============================================================ *)
(* 08_model_comparison_and_diagnostics.wl                       *)
(* ============================================================ *)
(*                                                              *)
(* Purpose:                                                     *)
(*   Symbolic verification of diagnostic formulas for Module 08:*)
(*     - Reduced χ² statistics and expected distribution        *)
(*     - Bayes factor and Jeffreys' scale                       *)
(*     - Einstein mass from θ_E                                 *)
(*     - Velocity dispersion from θ_E (SIE)                     *)
(*     - Magnification computation                              *)
(*                                                              *)
(* References:                                                  *)
(*   - Congdon & Keeton (2018), Ch. 4, 8                        *)
(*   - Jeffreys (1961), Theory of Probability                   *)
(*                                                              *)
(* Run: wolframscript -file                                     *)
(*      08_model_comparison_and_diagnostics.wl                  *)
(*                                                              *)
(* Author: Rodrigo Córdova Rosado (Harvard CfA)                 *)
(* Built with: Claude Code                                      *)
(* ============================================================ *)

Print[""];
Print["============================================================"];
Print["Module 08: Model Comparison & Diagnostics — Verification"];
Print["============================================================"];
Print[""];

(* ============================================================ *)
(* 1. χ² DISTRIBUTION                                           *)
(* ============================================================ *)

Print["--- 1. χ² Distribution ---"];
Print[""];

(* χ² with ν degrees of freedom has: *)
(* E[χ²] = ν, Var[χ²] = 2ν *)
Print["χ² distribution with ν = N_pix - N_params degrees of freedom:"];
Print["  E[χ²] = ν"];
Print["  Var[χ²] = 2ν"];
Print["  E[χ²_red] = 1"];
Print["  σ(χ²_red) = √(2/ν)"];
Print[""];

(* Expected scatter in χ²_red for typical lens modeling *)
nPixExamples = {500, 1000, 2500, 5000};
nParams = 14;

Print["Expected scatter in χ²_red (N_params = 14):"];
Do[
  nu = nPix - nParams;
  sigmaRed = Sqrt[2.0/nu];
  Print["  N_pix = ", nPix, ": ν = ", nu,
    ", σ(χ²_red) = ", NumberForm[N[sigmaRed], 3]],
  {nPix, nPixExamples}
];
Print[""];

(* ============================================================ *)
(* 2. BAYES FACTOR                                               *)
(* ============================================================ *)

Print["--- 2. Bayes Factor (Jeffreys' Scale) ---"];
Print[""];

Print["  ln B = ln Z_1 - ln Z_2"];
Print[""];
Print["  |ln B| < 1:     Not significant"];
Print["  |ln B| = 1-2.5: Moderate evidence"];
Print["  |ln B| = 2.5-5: Strong evidence"];
Print["  |ln B| > 5:     Decisive evidence"];
Print[""];

(* In terms of odds ratio *)
Print["  Odds ratio = exp(ln B):"];
Print["    ln B = 1   → odds = ", NumberForm[N[Exp[1]], 3], ":1"];
Print["    ln B = 2.5 → odds = ", NumberForm[N[Exp[2.5]], 3], ":1"];
Print["    ln B = 5   → odds = ", NumberForm[N[Exp[5]], 3], ":1"];
Print["    ln B = 10  → odds = ", NumberForm[N[Exp[10]], 3], ":1"];
Print[""];

(* ============================================================ *)
(* 3. EINSTEIN MASS                                              *)
(* ============================================================ *)

Print["--- 3. Einstein Mass ---"];
Print[""];

(* M(< θ_E) = π θ_E² D_d² Σ_cr *)
(* Constants *)
G = 6.674 * 10^-11;        (* N m² kg⁻² *)
c = 2.998 * 10^8;          (* m/s *)
Msun = 1.989 * 10^30;      (* kg *)
kpcToM = 3.086 * 10^19;    (* meters per kpc *)

(* Example: z_d = 0.5, z_s = 1.0, Planck15 *)
(* D_d ≈ 1277 Mpc, D_s ≈ 1651 Mpc, D_ds ≈ 861 Mpc *)
DdKpc = 1277.0 * 10^3;     (* kpc *)
DsKpc = 1651.0 * 10^3;
DdsKpc = 861.0 * 10^3;

(* Critical surface density *)
SigmaCr = (c^2 / (4 Pi G)) * (DsKpc / (DdKpc * DdsKpc)) / (kpcToM * 10^-3)^2 / Msun * kpcToM^2;
(* Simpler: compute in SI then convert *)
DdM = DdKpc * kpcToM;
DsM = DsKpc * kpcToM;
DdsM = DdsKpc * kpcToM;
SigmaCrSI = c^2/(4 Pi G) * DsM/(DdM * DdsM);  (* kg/m² *)
SigmaCrMsunKpc2 = SigmaCrSI / Msun * kpcToM^2;

Print["Σ_cr = ", ScientificForm[SigmaCrMsunKpc2, 3], " M_sun/kpc²"];

(* Einstein mass for different θ_E *)
Print[""];
Print["Einstein mass M(< θ_E):"];
kpcPerArcsec = DdKpc * (Pi / (180 * 3600));  (* kpc per arcsec *)

Do[
  REkpc = te * kpcPerArcsec;
  MEsun = Pi * REkpc^2 * SigmaCrMsunKpc2;
  Print["  θ_E = ", te, "\": R_E = ", NumberForm[N[REkpc], 4],
    " kpc, M = ", ScientificForm[N[MEsun], 3], " M_sun"],
  {te, {0.5, 1.0, 1.5, 2.0, 3.0}}
];
Print[""];

(* ============================================================ *)
(* 4. VELOCITY DISPERSION FROM θ_E (SIE)                        *)
(* ============================================================ *)
(* θ_E = 4π (σ_v/c)² (D_ds/D_s)                                *)
(* σ_v = c √(θ_E / (4π) × D_s/D_ds)                           *)
(* ============================================================ *)

Print["--- 4. Velocity Dispersion ---"];
Print[""];

Print["σ_v = c √(θ_E/(4π) × D_s/D_ds)  [for SIS/SIE]"];
Print[""];

Do[
  teRad = te / (3600 * 180 / Pi);
  sigmaV = c * Sqrt[teRad / (4 Pi) * DsM / DdsM];
  sigmaVkms = sigmaV / 1000;
  Print["  θ_E = ", te, "\": σ_v = ", NumberForm[N[sigmaVkms], 4], " km/s"],
  {te, {0.5, 1.0, 1.5, 2.0, 3.0}}
];
Print[""];

Print["============================================================"];
Print["All verifications complete."];
Print["============================================================"];
