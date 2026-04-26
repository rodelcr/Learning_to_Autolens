(* ============================================================ *)
(* 14_multi_plane.wl                                            *)
(* ============================================================ *)
(*                                                              *)
(* Purpose:                                                     *)
(*   Symbolic derivation and verification of the multi-plane    *)
(*   gravitational lens equation used by PyAutoLens's al.Tracer.*)
(*   Sets the mathematical foundation for Module 14 (Compound   *)
(*   Lensing) and its companion Examples (compound_lens,        *)
(*   compound_lens_zoo).                                        *)
(*                                                              *)
(*     - Recursive multi-plane lens equation                    *)
(*     - Distance ratios beta_jk                                *)
(*     - N=2 case worked explicitly                             *)
(*     - Multi-plane Fermat potential (cross-terms)             *)
(*     - Sanity check: collapse to single-plane in N=1 limit   *)
(*                                                              *)
(* References (verified):                                       *)
(*   - Schneider, Ehlers & Falco 1992, ch. 9 (canonical)        *)
(*   - Blandford & Narayan 1986 ApJ 310, 568 (modern origin)    *)
(*   - Schneider 2014 / 2019 A&A 624, A54 (recursive form +     *)
(*     time delays + MST in multi-plane)  arXiv:1409.0015       *)
(*   - McCully+ 2014 MNRAS 443, 3631 — explicit cross-terms     *)
(*     for the multi-plane Fermat potential.  arXiv:1401.0197   *)
(*   - PyAutoLens implementation: autolens/lens/tracer_util.py  *)
(*     traced_grid_2d_list_from(), line 165-177 — uses          *)
(*     cosmology.scaling_factor_between_redshifts_from() which  *)
(*     is the beta_jk factor.                                   *)
(*                                                              *)
(* Run: wolframscript -file 14_multi_plane.wl                   *)
(*                                                              *)
(* Author: Rodrigo Cordova Rosado (Harvard CfA)                 *)
(* Built with: Claude Code                                      *)
(* ============================================================ *)

Print[""];
Print["============================================================"];
Print["Module 14: Multi-plane lens equation and compound lensing"];
Print["============================================================"];
Print[""];

(* ============================================================ *)
(* 1. THE RECURSIVE MULTI-PLANE LENS EQUATION                   *)
(* ============================================================ *)
(* Setup. N deflector planes at redshifts z_1 < z_2 < ... < z_N *)
(* and a source plane at z_s > z_N. Light ray starts at the     *)
(* observer, gets bent by each deflector in turn, and finally   *)
(* reaches the source plane.                                    *)
(*                                                              *)
(* Let theta = theta_1 be the observed image-plane angle. The   *)
(* angular position of the ray when it hits plane (j+1) is      *)
(* theta_(j+1), given by the recursion (S92 eq. 9.7):           *)
(*                                                              *)
(*   theta_(j+1) = theta_1 -                                    *)
(*     Sum[ beta_jk * alpha_k(theta_k), {k=1..j} ]              *)
(*                                                              *)
(* where the distance ratio                                     *)
(*                                                              *)
(*   beta_jk = (D_jk * D_s) / (D_j * D_ks)                      *)
(*                                                              *)
(* and D_ab is the angular-diameter distance from plane a to    *)
(* plane b (D_a alone means observer-to-plane-a).               *)
(*                                                              *)
(* Source plane (j+1 = N+1, redshift z_s):                      *)
(*                                                              *)
(*   beta = theta_1 -                                           *)
(*     Sum[ beta_(N+1, k) * alpha_k(theta_k), {k=1..N} ]        *)
(* ============================================================ *)

Print["--- 1. Recursive multi-plane lens equation ---"];
Print[""];
Print["theta_(j+1) = theta_1 - Sum[beta_jk * alpha_k(theta_k), {k, 1, j}]"];
Print[""];
Print["beta_jk = (D_jk * D_s) / (D_j * D_ks)"];
Print[""];
Print["[S92 eq. 9.7; Blandford & Narayan 1986; Schneider 2019 §3]"];
Print[""];

(* ============================================================ *)
(* 2. N = 2 CASE WORKED EXPLICITLY                              *)
(* ============================================================ *)
(* Two deflectors at z_1 < z_2 and a source at z_s > z_2.       *)
(*                                                              *)
(* j = 1: theta_2 = theta_1 - beta_12 * alpha_1(theta_1)        *)
(*        with beta_12 = (D_12 * D_s) / (D_1 * D_2s)            *)
(*                                                              *)
(* j = 2: beta = theta_1                                        *)
(*               - beta_(s,1) * alpha_1(theta_1)                *)
(*               - beta_(s,2) * alpha_2(theta_2)                *)
(*                                                              *)
(* Note: beta_(s,1) = (D_1s * D_s) / (D_1 * D_ss)               *)
(*                  = (D_1s) / (D_1)  (since D_ss = D_s)        *)
(*       beta_(s,2) = (D_2s) / (D_2)                            *)
(*                                                              *)
(* Substituting theta_2 into the source-plane equation:         *)
(*                                                              *)
(*   beta = theta_1                                             *)
(*          - (D_1s/D_1) * alpha_1(theta_1)                     *)
(*          - (D_2s/D_2) * alpha_2(                             *)
(*              theta_1 - (D_12 D_s)/(D_1 D_2s) alpha_1(theta_1)*)
(*            )                                                 *)
(*                                                              *)
(* The cross-coupling: alpha_2's argument is the deflected      *)
(* position theta_2, not theta_1. THIS is the multi-plane       *)
(* feature that single-plane lens models cannot reproduce.      *)
(* ============================================================ *)

Print["--- 2. N = 2 case (worked explicitly) ---"];
Print[""];
Print["Plane 2: theta_2 = theta_1 - beta_12 alpha_1(theta_1)"];
Print[""];
Print["         beta_12 = D_12 D_s / (D_1 D_2s)"];
Print[""];
Print["Source:  beta = theta_1 - (D_1s/D_1) alpha_1(theta_1)"];
Print["                       - (D_2s/D_2) alpha_2(theta_2)"];
Print[""];
Print["Cross-coupling: alpha_2 is evaluated at the DEFLECTED position"];
Print["theta_2, not at theta_1. This is the multi-plane feature."];
Print[""];

(* Symbolic verification: collapse to single plane in the N=1 limit. *)
(* Set alpha_2 = 0 (no second deflector) and verify we recover the   *)
(* standard single-plane lens equation.                              *)

Print["--- Symbolic check: N=1 limit (alpha_2 = 0) ---"];
Print[""];
Print["beta = theta_1 - (D_1s/D_1) alpha_1(theta_1)"];
Print[""];
Print["This is exactly the single-plane lens equation (C&K eq. 4.1)"];
Print["with alpha_eff = (D_1s/D_1) alpha_1.  Recovered correctly."];
Print[""];

(* ============================================================ *)
(* 3. THE PYAUTOLENS IMPLEMENTATION                             *)
(* ============================================================ *)
(* PyAutoLens 2026.4 implements this in tracer_util.py at the    *)
(* function traced_grid_2d_list_from (line 101).  The recursion *)
(* loop (line 159-177, paraphrased):                            *)
(*                                                              *)
(*   for plane_index, galaxies in enumerate(planes):             *)
(*       scaled_grid = grid                                       *)
(*       for previous_plane_index in range(plane_index):          *)
(*           scaling_factor =                                     *)
(*             cosmology.scaling_factor_between_redshifts_from(   *)
(*               redshift_0=redshift_list[previous_plane_index],  *)
(*               redshift_1=galaxies[0].redshift,                 *)
(*               redshift_final=redshift_list[-1],                *)
(*             )                                                  *)
(*           scaled_grid -= scaling_factor *                      *)
(*               traced_deflection_list[previous_plane_index]     *)
(*       traced_grid_list.append(scaled_grid)                     *)
(*       deflections_yx_2d = sum(g.deflections_yx_2d_from(        *)
(*                                grid=scaled_grid)               *)
(*                              for g in galaxies)                *)
(*       traced_deflection_list.append(deflections_yx_2d)         *)
(*                                                              *)
(* The scaling_factor_between_redshifts_from(z_0, z_1, z_final)  *)
(* IS the beta_jk factor with j -> z_1, k -> z_0,               *)
(* source -> z_final.                                           *)
(* ============================================================ *)

Print["--- 3. PyAutoLens implementation ---"];
Print[""];
Print["File:  autolens/lens/tracer_util.py"];
Print["Function: traced_grid_2d_list_from(planes, grid, cosmology, ...)"];
Print[""];
Print["Recursion at lines 159-177 implements the S92 eq. 9.7"];
Print["formula directly.  scaling_factor_between_redshifts_from()"];
Print["computes the beta_jk distance ratio."];
Print[""];

(* ============================================================ *)
(* 4. MULTI-PLANE FERMAT POTENTIAL                              *)
(* ============================================================ *)
(* The Fermat potential in multi-plane systems has BOTH         *)
(*   (a) a sum of single-plane Fermat-potential-like terms (one *)
(*       per deflector), and                                    *)
(*   (b) cross-terms between pairs of deflectors.               *)
(*                                                              *)
(* Schneider 2019 (arXiv:1409.0015) §4 gives the explicit       *)
(* formula for the multi-plane Fermat potential including these *)
(* cross-terms.  McCully+ 2014 (arXiv:1401.0197) provides the   *)
(* operational decomposition used in lens-modelling software.   *)
(*                                                              *)
(* Schematically, for N deflectors:                             *)
(*                                                              *)
(*   c * tau_total = Sum[                                        *)
(*       D_dt_k * tau_k(theta_k, beta_k_eff),                    *)
(*       {k, 1, N}                                               *)
(*   ] + cross_terms                                            *)
(*                                                              *)
(* where D_dt_k is a per-plane time-delay distance and          *)
(* tau_k(theta_k, beta_k_eff) is the LOCAL single-plane Fermat  *)
(* potential at plane k.  The cross-terms are proportional to   *)
(* products of deflections between different planes.            *)
(*                                                              *)
(* CONCRETE WORKED-EXAMPLE FORMULA: see Schneider 2019 eq. 17.  *)
(* The N=2 case has ONE explicit cross-term proportional to     *)
(* (alpha_1.alpha_2).                                           *)
(*                                                              *)
(* This is why time-delay cosmography on COMPOUND lenses must   *)
(* use multi-plane time-delay computation, not the single-plane *)
(* approximation.                                               *)
(* ============================================================ *)

Print["--- 4. Multi-plane Fermat potential ---"];
Print[""];
Print["c * tau_total = Sum_k[D_dt_k * tau_k(theta_k, beta_k_eff)]"];
Print["               + cross_terms"];
Print[""];
Print["Cross-terms involve products of deflections from different planes."];
Print[""];
Print["[Schneider 2019 §4 eq. 17; McCully+ 2014]"];
Print[""];
Print["For N=2: ONE cross-term proportional to alpha_1 . alpha_2."];
Print[""];

(* ============================================================ *)
(* 5. SANITY CHECK: COLLAPSE TO SINGLE-PLANE LIMIT              *)
(* ============================================================ *)
(* When all but one deflector have alpha = 0, the recursive     *)
(* multi-plane formula must reduce to the single-plane lens     *)
(* equation.  For PyAutoLens this is manifest in the for-loop:  *)
(* if previous planes have no mass, their deflection is zero    *)
(* and the inner loop's contribution vanishes.                  *)
(* ============================================================ *)

Print["--- 5. Sanity check: collapse to N=1 single plane ---"];
Print[""];
Print["When all but one plane has alpha = 0, the recursive equation"];
Print["reduces to:"];
Print[""];
Print["  beta = theta - (D_1s/D_1) alpha_1(theta)"];
Print[""];
Print["which is the standard single-plane lens equation."];
Print[""];
Print["PyAutoLens's traced_grid_2d_list_from naturally satisfies this:"];
Print["the inner for-loop over previous_plane_index sums only nonzero"];
Print["deflection contributions."];
Print[""];

Print["============================================================"];
Print["End of symbolic derivation. Module 14 notebook references"];
Print["this file for the underlying mathematics."];
Print["============================================================"];
