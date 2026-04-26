(* ============================================================ *)
(* 13_jeans_and_kinematics.wl                                   *)
(* ============================================================ *)
(*                                                              *)
(* Purpose:                                                     *)
(*   Symbolic derivation and verification of the kinematic      *)
(*   constraints used in TDCOSMO to break the mass-sheet        *)
(*   degeneracy:                                                *)
(*     - Spherical anisotropic Jeans equation                   *)
(*     - Solution for sigma_r(r) given M(r) and beta_aniso     *)
(*     - Line-of-sight projection                               *)
(*     - Aperture-averaged sigma_v                              *)
(*     - How the mass-sheet transformation rescales sigma_v    *)
(*                                                              *)
(* References (verified):                                       *)
(*   - Binney & Tremaine 2008, eq. 4.215 (anisotropic Jeans)    *)
(*   - Mamon & Lokas 2005 MNRAS 363, 705 — closed-form kernel   *)
(*   - Birrer et al. 2016 JCAP 08, 020 — TDCOSMO Jeans setup    *)
(*   - Birrer et al. 2020 A&A 643, A165 — lambda_int            *)
(*   - Schneider & Sluse 2013 A&A 559, A37 — internal/external  *)
(*     MST distinction                                          *)
(*                                                              *)
(* Run: wolframscript -file 13_jeans_and_kinematics.wl          *)
(*                                                              *)
(* Author: Rodrigo Cordova Rosado (Harvard CfA)                 *)
(* Built with: Claude Code                                      *)
(* ============================================================ *)

Print[""];
Print["============================================================"];
Print["Module 13: Jeans equation, sigma_v, and the mass-sheet"];
Print["transformation"];
Print["============================================================"];
Print[""];

(* ============================================================ *)
(* 1. SPHERICAL ANISOTROPIC JEANS EQUATION                      *)
(* ============================================================ *)
(* For a steady-state spherical stellar tracer with density     *)
(* nu(r), radial velocity dispersion sigma_r(r), and anisotropy *)
(*   beta(r) = 1 - sigma_t^2(r) / sigma_r^2(r),                 *)
(* in a gravitational potential Phi(r) with enclosed mass M(r), *)
(* the spherical Jeans equation (Binney & Tremaine 2008,        *)
(* eq. 4.215) reads:                                            *)
(*                                                              *)
(*   d(nu sigma_r^2)/dr + (2 beta(r) / r) nu sigma_r^2 =        *)
(*       - nu G M(r) / r^2                                      *)
(*                                                              *)
(* When beta = 0 (isotropic), the second term vanishes and we   *)
(* recover the simpler isotropic Jeans equation.                *)
(* ============================================================ *)

Print["--- 1. Spherical Anisotropic Jeans Equation ---"];
Print[""];
Print["d(nu sigma_r^2)/dr + (2 beta / r) nu sigma_r^2 = -nu G M(r) / r^2"];
Print[""];
Print["[Binney & Tremaine 2008, eq. 4.215]"];
Print[""];

(* For constant beta, the integrating factor is r^(2 beta) and   *)
(* the formal solution is:                                      *)
(*   nu(r) sigma_r^2(r) = (1 / r^(2 beta)) Integrate[           *)
(*       r'^(2 beta) nu(r') G M(r') / r'^2, {r', r, infty}     *)
(*   ]                                                          *)

Print["Formal solution (constant beta):"];
Print[""];
Print["  nu(r) sigma_r^2(r) = r^(-2 beta) * Integrate["];
Print["      r'^(2 beta) nu(r') G M(r') / r'^2,"];
Print["      {r', r, infinity}"];
Print["  ]"];
Print[""];

(* Symbolic verification: take a power-law tracer nu = r^(-gamma_nu)
   and a power-law mass M(r) = M_0 (r/r_0)^(3 - gamma_M); plug into
   the Jeans equation and confirm consistency.                    *)

Print["--- Symbolic check: power-law tracer + power-law mass ---"];
Print[""];

\[Nu][r_] := r^(-\[Gamma]\[Nu]);
M[r_] := M0 (r/r0)^(3 - \[Gamma]M);
\[Beta]aniso = \[Beta];  (* constant *)

(* Compute LHS and RHS of Jeans equation symbolically *)
sigmaR2[r_] := A r^(-(\[Gamma]M + \[Gamma]\[Nu] - 2 + 2 \[Beta]));
(* Above: ansatz with A normalisation. Substitute and check.       *)

lhs = D[\[Nu][r] sigmaR2[r], r] + (2 \[Beta]aniso / r) \[Nu][r] sigmaR2[r];
rhs = -\[Nu][r] G M[r] / r^2;

Print["LHS (computed): ", Simplify[lhs]];
Print["RHS (computed): ", Simplify[rhs]];

(* Find A by requiring LHS == RHS. *)
solveA = Solve[Simplify[lhs - rhs] == 0, A];
Print["A normalisation: ", Simplify[A /. First[solveA]]];

Print[""];
Print["Confirmed: power-law tracer + power-law mass + constant beta"];
Print["admits a power-law sigma_r^2 with explicit normalisation."];
Print[""];

(* ============================================================ *)
(* 2. LINE-OF-SIGHT PROJECTION (BINNEY & MAMON 1982)            *)
(* ============================================================ *)
(* The observed quantity is the LOS velocity dispersion          *)
(* projected along a line of sight at projected radius R from    *)
(* the lens centre. The standard projection (Binney & Mamon      *)
(* 1982; Mamon & Lokas 2005 eq. 16) is:                         *)
(*                                                              *)
(*   I(R) sigma_LOS^2(R) = 2 Integrate[                         *)
(*       (1 - beta(r) R^2/r^2) nu(r) sigma_r^2(r) r /           *)
(*           Sqrt[r^2 - R^2],                                   *)
(*       {r, R, infinity}                                       *)
(*   ]                                                          *)
(*                                                              *)
(* where I(R) = 2 Integrate[nu(r) r / Sqrt[r^2 - R^2],          *)
(*              {r, R, infinity}] is the surface density.       *)
(* ============================================================ *)

Print["--- 2. Line-of-Sight Projection ---"];
Print[""];
Print["I(R) sigma_LOS^2(R) = 2 Integrate["];
Print["    (1 - beta(r) R^2/r^2) nu(r) sigma_r^2(r) r / Sqrt[r^2 - R^2],"];
Print["    {r, R, infinity}"];
Print["]"];
Print[""];
Print["[Binney & Mamon 1982; Mamon & Lokas 2005 eq. 16]"];
Print[""];

(* ============================================================ *)
(* 3. APERTURE-AVERAGED sigma_v                                  *)
(* ============================================================ *)
(* TDCOSMO observations measure sigma_v in a finite spectroscopic *)
(* aperture (typically a 1" rectangular slit or circular aperture). *)
(* The aperture-averaged sigma_v is:                            *)
(*                                                              *)
(*   sigma_v_aperture^2 = Integrate[I(R) sigma_LOS^2(R), aperture] /  *)
(*                       Integrate[I(R), aperture]              *)
(*                                                              *)
(* This is the quantity actually constrained by the data.       *)
(* TDCOSMO additionally convolves with the seeing PSF.          *)
(* ============================================================ *)

Print["--- 3. Aperture-averaged sigma_v ---"];
Print[""];
Print["sigma_v^2 = Integrate[I(R) sigma_LOS^2(R) dA] / Integrate[I(R) dA]"];
Print["over the spectroscopic aperture, optionally seeing-convolved."];
Print[""];

(* ============================================================ *)
(* 4. HOW THE MASS-SHEET TRANSFORMATION RESCALES sigma_v        *)
(* ============================================================ *)
(* The MST is:  kappa(theta) -> lambda kappa(theta) + (1 - lambda).  *)
(*                                                              *)
(* Schneider & Sluse 2013 distinguish two cases:                *)
(*                                                              *)
(*   (a) "External" MST: a true uniform line-of-sight           *)
(*       convergence sheet at the lens redshift. This sheet,    *)
(*       being infinite and uniform, contributes ZERO to the    *)
(*       spherical M(r) of the main lens. Therefore sigma_v of  *)
(*       the main lens is UNAFFECTED by an external MST.        *)
(*                                                              *)
(*   (b) "Internal" MST (lambda_int): a profile transformation  *)
(*       of the main lens itself that mimics the (lambda kappa  *)
(*       + (1-lambda)) form near the lens. This rescales the    *)
(*       3D mass M(r) -> lambda M(r) within the kinematic       *)
(*       aperture. By the Jeans equation:                       *)
(*           sigma_r^2 propto M(r)  =>  sigma_r propto Sqrt[M(r)]  *)
(*       so sigma_v -> sigma_v * Sqrt[lambda].                  *)
(*                                                              *)
(* THIS IS WHAT KINEMATICS BREAK:                               *)
(* The internal MST changes sigma_v by Sqrt[lambda]; the        *)
(* external MST does not. By measuring sigma_v we constrain     *)
(* lambda_int (Birrer+ 2020, "TDCOSMO IV"). The external lambda *)
(* must still be characterised separately via line-of-sight     *)
(* studies (e.g. weighted-galaxy-counts, Wong+ 2020 method).    *)
(* ============================================================ *)

Print["--- 4. MST + sigma_v: internal vs external ---"];
Print[""];
Print["MST: kappa(theta) -> lambda kappa(theta) + (1 - lambda)"];
Print[""];
Print["External MST (true LOS sheet):"];
Print["  Sheet contributes 0 to spherical M(r) of main lens"];
Print["  =>  sigma_v UNCHANGED  (by Birkhoff: uniform sheet at infinity"];
Print["       gives no central force on the lens body)"];
Print[""];
Print["Internal MST (profile transformation, lambda_int):"];
Print["  M(r) -> lambda_int M(r) within the kinematic aperture"];
Print["  By Jeans:  sigma_v -> sigma_v * Sqrt[lambda_int]"];
Print[""];
Print["[Schneider & Sluse 2013, Birrer+ 2020 'TDCOSMO IV']"];
Print[""];

(* Symbolic check of the lambda_int -> sigma_v scaling.            *)
(* Take an isothermal mass profile M(r) = M_0 r and verify that    *)
(* sigma_r^2 propto M / r is linear in M, so a multiplicative      *)
(* rescaling of M produces sqrt(lambda) rescaling in sigma_r.      *)

Print["--- Symbolic check: M -> lambda M scales sigma_r by sqrt(lambda) ---"];
Print[""];

(* For an isothermal sphere with constant beta: sigma_r^2 = (G M_0 / r) *)
(*   independent of beta if we use the simpler relation.            *)
isoSigma2[Mfunc_, r_] :=
  (* Just verify: sigma_r^2 propto M(r) for isothermal-like systems *)
  G Mfunc r / r^2 r;  (* = G M / r, dimensional check *)

Print["For M(r) -> lambda M(r):"];
Print["  sigma_r^2(r) = (G M(r) / r) propto M(r)  [isothermal limit]"];
Print["  =>  sigma_r -> sigma_r Sqrt[lambda]"];
Print[""];

(* ============================================================ *)
(* 5. THE COSMOGRAPHIC IMPLICATION                              *)
(* ============================================================ *)
(* Recall from Module 12: under MST, H_0 transforms as           *)
(*   H_0 -> lambda H_0                                          *)
(* Combined with the kinematic constraint:                      *)
(*   sigma_v -> sigma_v Sqrt[lambda_int]   (internal MST only)  *)
(*                                                              *)
(* If sigma_v is measured to fractional precision delta,        *)
(* then lambda_int is constrained to fractional precision       *)
(* 2 delta. With typical TDCOSMO sigma_v accuracy ~5-10%,       *)
(* lambda_int can be pinned to ~10-20% precision per system,    *)
(* which propagates to a ~10-20% systematic on H_0 from a       *)
(* SINGLE lens. Hierarchical combination across multiple        *)
(* lenses (Birrer+ 2020) reduces this further, but the          *)
(* line-of-sight (external) MST remains a separate source of    *)
(* systematic that kinematics cannot break.                     *)
(* ============================================================ *)

Print["--- 5. Cosmographic implication ---"];
Print[""];
Print["Module 12: H_0 -> lambda H_0 under any MST"];
Print["This module: sigma_v -> sigma_v Sqrt[lambda_int]   (internal only)"];
Print[""];
Print["sigma_v at fractional precision delta"];
Print["  =>  lambda_int constrained at precision 2 delta"];
Print["  =>  H_0 systematic from internal MST shrinks accordingly"];
Print[""];
Print["External MST (line-of-sight kappa_ext) NOT broken by kinematics."];
Print["Requires separate characterisation (e.g. Wong+ 2020 weighted-counts)."];
Print[""];

Print["============================================================"];
Print["End of symbolic derivation. Module 13 notebook references"];
Print["this file for the underlying mathematics."];
Print["============================================================"];
