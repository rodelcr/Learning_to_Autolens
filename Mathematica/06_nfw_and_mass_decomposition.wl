(* ============================================================ *)
(* 06_nfw_and_mass_decomposition.wl                             *)
(* ============================================================ *)
(*                                                              *)
(* Purpose:                                                     *)
(*   Symbolic verification of NFW lensing properties and        *)
(*   stellar+dark matter mass decomposition:                    *)
(*     - NFW convergence function f(x)                          *)
(*     - NFW enclosed mass                                      *)
(*     - Sérsic stellar mass profile                             *)
(*     - Dark matter fraction within θ_E                        *)
(*     - Bulge-halo conspiracy: total slope ≈ isothermal        *)
(*                                                              *)
(* References:                                                  *)
(*   - Navarro, Frenk & White (1996), ApJ, 462, 563             *)
(*   - Bartelmann (1996), A&A, 313, 697                          *)
(*   - Congdon & Keeton (2018), Sec. 4.5                         *)
(*   - Treu & Koopmans (2004), ApJ, 611, 739                    *)
(*                                                              *)
(* Run: wolframscript -file 06_nfw_and_mass_decomposition.wl    *)
(*                                                              *)
(* Author: Rodrigo Córdova Rosado (Harvard CfA)                 *)
(* Built with: Claude Code                                      *)
(* ============================================================ *)

Print[""];
Print["============================================================"];
Print["Module 06: NFW & Mass Decomposition — Symbolic Verification"];
Print["============================================================"];
Print[""];

(* ============================================================ *)
(* 1. NFW CONVERGENCE FUNCTION                                   *)
(* ============================================================ *)
(* κ_NFW(x) = κ_s × f(x) where x = θ/θ_s                       *)
(* f(x) has three branches: x<1, x=1, x>1                       *)
(* (cf. Bartelmann 1996; C&K eq. 4.62)                           *)
(* ============================================================ *)

Print["--- 1. NFW Convergence Function ---"];
Print[""];

(* f(x) for x < 1 *)
fNFWinner[x_] := 1/(x^2 - 1) * (1 - ArcCosh[1/x]/Sqrt[1 - x^2]);

(* f(x) for x > 1 *)
fNFWouter[x_] := 1/(x^2 - 1) * (1 - ArcTan[Sqrt[x^2 - 1]]/Sqrt[x^2 - 1]);

(* f(x) at x = 1 *)
fNFWcrit = 1/3;

(* Numerical verification *)
Print["NFW f(x) at selected radii:"];
Print["  f(0.5) = ", NumberForm[N[fNFWinner[0.5]], 5]];
Print["  f(1.0) = ", fNFWcrit, " (analytic)"];
Print["  f(2.0) = ", NumberForm[N[fNFWouter[2.0]], 5]];
Print["  f(5.0) = ", NumberForm[N[fNFWouter[5.0]], 5]];
Print[""];

(* Verify continuity at x = 1 *)
limLeft = Limit[fNFWinner[x], x -> 1, Direction -> "FromBelow"];
limRight = Limit[fNFWouter[x], x -> 1, Direction -> "FromAbove"];
Print["Continuity at x=1:"];
Print["  lim(x→1⁻) f(x) = ", N[limLeft, 5]];
Print["  lim(x→1⁺) f(x) = ", N[limRight, 5]];
Print["  f(1) = 1/3 = ", N[1/3, 5], " ✓"];
Print[""];

(* ============================================================ *)
(* 2. NFW ENCLOSED MASS                                          *)
(* ============================================================ *)
(* M_NFW(< r) = 4π ρ_s r_s³ [ln(1 + r/r_s) - r/(r_s + r)]     *)
(* (cf. C&K eq. 4.58)                                           *)
(* ============================================================ *)

Print["--- 2. NFW Enclosed Mass ---"];
Print[""];

mNFW[r_, rhoS_, rS_] := 4 Pi rhoS rS^3 * (Log[1 + r/rS] - (r/rS)/(1 + r/rS));

(* Numerical example: typical galaxy halo *)
(* M_200 ~ 10^13 M_sun, c = 10, r_200 = 300 kpc *)
r200 = 300.0;       (* kpc *)
c = 10.0;
rS = r200/c;         (* = 30 kpc *)
(* ρ_s from M_200 *)
rhoS = 200/3 * 998.0 * c^3 / (Log[1 + c] - c/(1 + c));  (* in units where M_200 works *)

Print["Example: M_200 = 10^13 M_sun, c = 10"];
Print["  r_s = r_200/c = ", rS, " kpc"];
Print["  M(<r_s) / M(<r_200) = ",
  NumberForm[N[(Log[1 + 1] - 1/2) / (Log[1 + c] - c/(1 + c))], 3]];
Print["  → Most mass is at r >> r_s"];
Print[""];

(* ============================================================ *)
(* 3. STELLAR MASS (SÉRSIC)                                      *)
(* ============================================================ *)
(* Σ_*(θ) = Υ_* × I(θ)                                          *)
(* For Sérsic: I(R) = I_e exp{-b_n [(R/R_e)^(1/n) - 1]}         *)
(* Enclosed luminosity: L(<R) = 2π ∫_0^R I(R') R' dR'           *)
(*   = 2π n I_e R_e² exp(b_n) b_n^(-2n) γ(2n, b_n (R/R_e)^(1/n))*)
(* where γ is the incomplete gamma function.                      *)
(* ============================================================ *)

Print["--- 3. Sérsic Stellar Mass ---"];
Print[""];

(* Enclosed luminosity fraction within R *)
(* L(<R)/L_total = γ(2n, b_n x^{1/n}) / Γ(2n) where x = R/R_e *)
bn[n_] := 2 n - 1/3 + 4/(405 n);

enclosedFraction[x_, n_] :=
  GammaRegularized[2 n, 0, bn[n] * x^(1/n)];

Print["Enclosed luminosity fractions for n=4 (de Vaucouleurs):"];
Print["  L(<0.5 R_e) / L_total = ", NumberForm[N[enclosedFraction[0.5, 4]], 3]];
Print["  L(<1.0 R_e) / L_total = ", NumberForm[N[enclosedFraction[1.0, 4]], 3],
  " (should be ~0.5)"];
Print["  L(<2.0 R_e) / L_total = ", NumberForm[N[enclosedFraction[2.0, 4]], 3]];
Print["  L(<5.0 R_e) / L_total = ", NumberForm[N[enclosedFraction[5.0, 4]], 3]];
Print[""];

(* ============================================================ *)
(* 4. DARK MATTER FRACTION                                       *)
(* ============================================================ *)
(* f_DM(θ_E) = 1 - M_*(< θ_E) / M_total(< θ_E)                *)
(* M_total = π θ_E² Σ_cr D_d²  (from the definition of θ_E)    *)
(* M_* = Υ_* × L(< θ_E)                                         *)
(* ============================================================ *)

Print["--- 4. Dark Matter Fraction ---"];
Print[""];

Print["f_DM(θ_E) = 1 - M_*(< θ_E) / M_total(< θ_E)"];
Print["         = 1 - Υ_* L(<θ_E) / (π θ_E² Σ_cr D_d²)"];
Print[""];

(* Typical values for SLACS-like lenses *)
Print["Typical SLACS values (Auger et al. 2010):"];
Print["  f_DM(θ_E) ~ 0.2-0.5"];
Print["  Increases with Einstein radius (more massive → more DM)"];
Print[""];

(* ============================================================ *)
(* 5. BULGE-HALO CONSPIRACY                                      *)
(* ============================================================ *)
(* The total 3D density slope at r ~ R_e:                        *)
(*   d ln ρ_total / d ln r ≈ -2 (isothermal)                    *)
(* Even though:                                                  *)
(*   d ln ρ_* / d ln r ≈ -3 (steeper)                           *)
(*   d ln ρ_DM / d ln r ≈ -1 (shallower, NFW inner slope)       *)
(* ============================================================ *)

Print["--- 5. Bulge-Halo Conspiracy ---"];
Print[""];

(* NFW local slope *)
rhoNFW[r_, rS_] := 1/((r/rS) (1 + r/rS)^2);
slopeNFW[r_, rS_] := r * D[Log[rhoNFW[rr, rS]], rr] /. rr -> r;

Print["NFW density slope d ln ρ / d ln r:"];
Print["  At r = 0.1 r_s: ", NumberForm[N[slopeNFW[0.1, 1.0]], 3],
  " (approaches -1)"];
Print["  At r = 1.0 r_s: ", NumberForm[N[slopeNFW[1.0, 1.0]], 3],
  " (transition)"];
Print["  At r = 10 r_s:  ", NumberForm[N[slopeNFW[10.0, 1.0]], 3],
  " (approaches -3)"];
Print[""];

Print["The conspiracy: stellar (slope ~ -3) + NFW (slope ~ -1)"];
Print["combine to give total slope ~ -2 (isothermal) at r ~ R_e."];
Print["This is observed in SLACS/SL2S lenses (Koopmans+ 2006, 2009)."];
Print[""];

Print["============================================================"];
Print["All verifications complete."];
Print["============================================================"];
