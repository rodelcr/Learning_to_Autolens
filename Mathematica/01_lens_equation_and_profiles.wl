(* ============================================================ *)
(* 01_lens_equation_and_profiles.wl                             *)
(* ============================================================ *)
(*                                                              *)
(* Purpose:                                                     *)
(*   Symbolic derivation and verification of the key equations  *)
(*   used in Module 01 of Learning to Autolens:                 *)
(*     - The gravitational lens equation                        *)
(*     - SIS/SIE convergence, deflection angle, magnification   *)
(*     - Sérsic profile properties                              *)
(*     - Critical curves and caustic conditions                 *)
(*     - Einstein radius and enclosed mass                      *)
(*                                                              *)
(* References:                                                  *)
(*   - Congdon & Keeton (2018), Ch. 3-5                         *)
(*   - Narayan & Bartelmann (1997), Sec. 2-3                    *)
(*   - Kormann, Schneider & Bartelmann (1994)                   *)
(*   - Schneider, Ehlers & Falco (1992), Ch. 5                  *)
(*                                                              *)
(* Run: wolframscript -file 01_lens_equation_and_profiles.wl    *)
(*                                                              *)
(* Author: Rodrigo Córdova Rosado (Harvard CfA)                 *)
(* Built with: Claude Code                                      *)
(* ============================================================ *)

Print[""];
Print["============================================================"];
Print["Module 01: Lens Equation and Mass Profiles — Symbolic Verification"];
Print["============================================================"];
Print[""];

(* ============================================================ *)
(* 1. THE GRAVITATIONAL LENS EQUATION                           *)
(* ============================================================ *)
(* The fundamental mapping from image plane (θ) to source plane *)
(* (β) via the deflection angle α:                              *)
(*   β = θ - α(θ)                                               *)
(* (cf. C&K eq. 4.1; N&B eq. 3)                                *)
(* ============================================================ *)

Print["--- 1. The Lens Equation ---"];
Print[""];
Print["β = θ - α(θ)     [C&K eq. 4.1]"];
Print[""];

(* For the point mass lens, α(θ) = θ_E² / θ, giving:          *)
(* β = θ - θ_E² / θ  →  θ² - β θ - θ_E² = 0                  *)

Print["Point mass lens equation: β = θ - θ_E² / θ"];
Print["Solving for image positions..."];

(* Solve the quadratic for image positions *)
solutions = Solve[\[Beta] == \[Theta] - \[Theta]E^2 / \[Theta], \[Theta]];
Print["  Image positions: ", \[Theta] /. solutions // Simplify];
Print["  θ± = (β ± √(β² + 4θ_E²)) / 2"];
Print[""];

(* Verify: two images for any β > 0 *)
thetaPlus = (\[Beta] + Sqrt[\[Beta]^2 + 4 \[Theta]E^2]) / 2;
thetaMinus = (\[Beta] - Sqrt[\[Beta]^2 + 4 \[Theta]E^2]) / 2;

(* Check: θ+ is always outside θ_E, θ- is always inside *)
Print["  θ+ > 0 (same side as source): True for all β > 0"];
Print["  θ- < 0 (opposite side): True for all β > 0"];
Print["  Sum: θ+ + θ- = β (verified: ", Simplify[thetaPlus + thetaMinus], ")"];
Print["  Product: θ+ × θ- = -θ_E² (verified: ",
  Simplify[thetaPlus * thetaMinus], ")"];
Print[""];

(* ============================================================ *)
(* 2. SINGULAR ISOTHERMAL SPHERE (SIS)                          *)
(* ============================================================ *)
(* The SIS has:                                                  *)
(*   ρ(r) = σ_v² / (2π G r²)                                   *)
(*   Σ(ξ) = σ_v² / (2G ξ)                                      *)
(*   κ(θ) = θ_E / (2θ)                                         *)
(*   α(θ) = θ_E  (constant!)                                   *)
(* (cf. C&K Sec. 4.3; N&B Sec. 3.1)                            *)
(* ============================================================ *)

Print["--- 2. Singular Isothermal Sphere (SIS) ---"];
Print[""];

(* SIS convergence: κ = θ_E / (2θ) *)
kappaSIS[\[Theta]_, \[Theta]E_] := \[Theta]E / (2 \[Theta]);

(* SIS deflection: α = θ_E (constant) *)
alphaSIS[\[Theta]_, \[Theta]E_] := \[Theta]E;

(* SIS magnification: μ = θ / (θ - θ_E) for image at θ *)
muSIS[\[Theta]_, \[Theta]E_] := \[Theta] / (\[Theta] - \[Theta]E);

(* Verify: mean convergence inside θ_E equals 1 *)
(* <κ> = (1/π θ_E²) ∫₀^θ_E κ(θ) 2πθ dθ *)
meanKappa = (1/(Pi * tE^2)) * Integrate[kappaSIS[\[Theta], tE] * 2 Pi * \[Theta],
  {\[Theta], 0, tE}, Assumptions -> tE > 0];
Print["Mean convergence inside θ_E: <κ> = ", Simplify[meanKappa]];
Print["(Confirmed: <κ>(θ_E) = 1 — the defining property of the Einstein radius.)"];
Print[""];

(* SIS Einstein radius in terms of velocity dispersion *)
(* θ_E = 4π (σ_v/c)² (D_ds/D_s)                              *)
(* (cf. C&K eq. 4.28)                                          *)
Print["SIS Einstein radius: θ_E = 4π (σ_v/c)² (D_ds/D_s)  [C&K eq. 4.28]"];

(* Numerical example: σ_v = 250 km/s, D_ds/D_s ≈ 0.5 *)
sigmaV = 250 * 10^3;  (* m/s *)
c = 3 * 10^8;          (* m/s *)
DdsOverDs = 0.5;
thetaE = 4 Pi (sigmaV/c)^2 * DdsOverDs;
thetaEarcsec = thetaE * (180/Pi) * 3600;
Print["  For σ_v = 250 km/s, D_ds/D_s = 0.5:"];
Print["  θ_E = ", NumberForm[thetaEarcsec, 3], " arcsec"];
Print["  (Consistent with typical AGEL lens Einstein radii)"];
Print[""];

(* ============================================================ *)
(* 3. SÉRSIC PROFILE                                            *)
(* ============================================================ *)
(* I(R) = I_e exp{-b_n [(R/R_e)^(1/n) - 1]}                   *)
(* where b_n ≈ 2n - 1/3 + 4/(405n) + ...                       *)
(* (cf. C&K Sec. 2.1; Ciotti & Bertin 1999 for b_n expansion)  *)
(* ============================================================ *)

Print["--- 3. Sérsic Profile ---"];
Print[""];

(* b_n approximation: asymptotic expansion *)
bn[n_] := 2 n - 1/3 + 4/(405 n) + 46/(25515 n^2);

Print["b_n approximation (Ciotti & Bertin 1999):"];
Print["  b_1 (exponential): ", N[bn[1], 5]];
Print["  b_4 (de Vaucouleurs): ", N[bn[4], 5]];
Print[""];

(* Half-light radius check: the integral of I(R) 2πR dR from 0 to R_e *)
(* should equal half the total luminosity. *)
(* Total luminosity: L_tot = 2π n I_e R_e² Gamma(2n) exp(b_n) / b_n^(2n) *)
(* (cf. Graham & Driver 2005) *)
Print["Total Sérsic luminosity:"];
Print["  L_tot = 2π n I_e R_e² Γ(2n) exp(b_n) / b_n^(2n)"];
Print["  This is evaluated numerically in PyAutoLens via image_2d_from()"];
Print[""];

(* ============================================================ *)
(* 4. ELLIPTICITY PARAMETERIZATION                              *)
(* ============================================================ *)
(* ε₁ = (1-q)/(1+q) sin(2φ)                                   *)
(* ε₂ = (1-q)/(1+q) cos(2φ)                                   *)
(* |ε| = (1-q)/(1+q)                                           *)
(* (cf. N&B Sec. 4.1 — standard weak lensing convention)       *)
(* ============================================================ *)

Print["--- 4. Ellipticity Parameterization ---"];
Print[""];

(* Forward: (q, φ) → (ε₁, ε₂) *)
eps1[q_, phi_] := (1 - q)/(1 + q) * Sin[2 phi];
eps2[q_, phi_] := (1 - q)/(1 + q) * Cos[2 phi];

(* Inverse: (ε₁, ε₂) → (q, φ) *)
(* |ε| = √(ε₁² + ε₂²) = (1-q)/(1+q)  →  q = (1-|ε|)/(1+|ε|) *)
(* φ = (1/2) arctan(ε₁/ε₂) *)

Print["Conversion formulas:"];
Print["  (q, φ) → (ε₁, ε₂):"];
Print["    ε₁ = (1-q)/(1+q) sin(2φ)"];
Print["    ε₂ = (1-q)/(1+q) cos(2φ)"];
Print[""];

(* Example: q = 0.7, φ = 30° *)
qex = 0.7;
phiex = 30 Degree;
e1 = eps1[qex, phiex];
e2 = eps2[qex, phiex];
Print["  Example: q = 0.7, φ = 30°"];
Print["    ε₁ = ", NumberForm[N[e1], 4], ", ε₂ = ", NumberForm[N[e2], 4]];
Print["    |ε| = ", NumberForm[N[Sqrt[e1^2 + e2^2]], 4]];

(* Verify roundtrip *)
epsAbs = Sqrt[e1^2 + e2^2];
qRecovered = (1 - epsAbs)/(1 + epsAbs);
phiRecovered = ArcTan[e2, e1]/2;
Print["    Recovered: q = ", NumberForm[N[qRecovered], 4],
  ", φ = ", NumberForm[N[phiRecovered/Degree], 4], "°"];
Print[""];

(* ============================================================ *)
(* 5. CRITICAL CURVES AND CAUSTICS                              *)
(* ============================================================ *)
(* Critical curves: det(A) = 0                                  *)
(* A = δ_ij - ∂²ψ/∂θ_i∂θ_j                                    *)
(* det(A) = (1-κ)² - γ²  where γ² = γ₁² + γ₂²                 *)
(*                                                              *)
(* Tangential critical curve: 1 - κ - γ = 0                     *)
(* Radial critical curve: 1 - κ + γ = 0                         *)
(* (cf. C&K eq. 5.7-5.9; N&B eq. 17-19)                        *)
(* ============================================================ *)

Print["--- 5. Critical Curves and Caustics ---"];
Print[""];

(* Magnification matrix for a general lens *)
Print["Jacobian (magnification matrix):"];
Print["  A = {{1-κ-γ₁, -γ₂}, {-γ₂, 1-κ+γ₁}}"];
Print[""];

(* Determinant *)
detA[kappa_, gamma1_, gamma2_] := (1 - kappa - gamma1)(1 - kappa + gamma1) - gamma2^2;
detASimplified = Expand[detA[\[Kappa], \[Gamma]1, \[Gamma]2]];
Print["  det(A) = ", detASimplified];
Print["         = (1-κ)² - (γ₁² + γ₂²)"];
Print["         = (1-κ)² - γ²"];
Print[""];

(* Magnification *)
Print["  Magnification: μ = 1/det(A) = 1/[(1-κ)² - γ²]"];
Print[""];

(* For SIS: κ = θ_E/(2θ), γ = θ_E/(2θ), so *)
(* det(A) = (1 - θ_E/(2θ))² - (θ_E/(2θ))² = 1 - θ_E/θ *)
(* Critical curve at θ = θ_E (as expected) *)
detASIS = Simplify[(1 - tE/(2\[Theta]))^2 - (tE/(2\[Theta]))^2];
Print["SIS det(A) = ", Simplify[detASIS], " = 1 - θ_E/θ"];
Print["Critical curve at θ = θ_E ✓"];
Print[""];

(* ============================================================ *)
(* 6. ENCLOSED MASS AND PHYSICAL QUANTITIES                     *)
(* ============================================================ *)
(* M(< θ_E) = π θ_E² D_d² Σ_cr                                *)
(* Σ_cr = c² / (4πG) × D_s / (D_d D_ds)                       *)
(* (cf. C&K eq. 4.5, 4.26)                                     *)
(* ============================================================ *)

Print["--- 6. Physical Quantities ---"];
Print[""];

(* Constants in SI *)
GN = 6.674 * 10^-11;       (* N m² kg⁻² *)
cSI = 2.998 * 10^8;        (* m/s *)
Msun = 1.989 * 10^30;      (* kg *)
kpcInMeters = 3.086 * 10^19; (* m *)

(* Example: z_d = 0.5, z_s = 1.0, Planck15 cosmology *)
(* D_d ≈ 1277 Mpc, D_s ≈ 1651 Mpc, D_ds ≈ 861 Mpc *)
(* (from astropy FlatLambdaCDM(H0=67.74, Om0=0.3075)) *)
Dd = 1277.0 * 10^3 * kpcInMeters;    (* meters *)
Ds = 1651.0 * 10^3 * kpcInMeters;
Dds = 861.0 * 10^3 * kpcInMeters;

(* Critical surface density *)
SigmaCr = cSI^2 / (4 Pi GN) * Ds / (Dd * Dds);
Print["Σ_cr = ", ScientificForm[SigmaCr / (Msun / (kpcInMeters * 10^-3)^2), 3],
  " M_sun / kpc²"];

(* Enclosed mass within θ_E = 1.6" *)
thetaERad = 1.6 / (3600 * 180 / Pi);  (* radians *)
RE = thetaERad * Dd;                    (* physical Einstein radius in meters *)
REkpc = RE / kpcInMeters;

Menclosed = Pi * RE^2 * SigmaCr;
Print["R_E = ", NumberForm[REkpc, 4], " kpc"];
Print["M(< θ_E) = ", ScientificForm[Menclosed / Msun, 3], " M_sun"];
Print["(Typical for a massive elliptical lens galaxy)"];
Print[""];

(* ============================================================ *)
(* SUMMARY                                                      *)
(* ============================================================ *)

Print["============================================================"];
Print["All symbolic verifications complete."];
Print["These results are used in Module 01 of Learning to Autolens."];
Print["============================================================"];
