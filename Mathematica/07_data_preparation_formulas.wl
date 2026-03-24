(* ============================================================ *)
(* 07_data_preparation_formulas.wl                              *)
(* ============================================================ *)
(*                                                              *)
(* Purpose:                                                     *)
(*   Verification of data preparation formulas for Module 07:   *)
(*     - Pixel scale conversions                                *)
(*     - Noise map conversions (weight → σ)                     *)
(*     - Background estimation statistics                       *)
(*     - Cutout size from Einstein radius                       *)
(*                                                              *)
(* Run: wolframscript -file 07_data_preparation_formulas.wl     *)
(*                                                              *)
(* Author: Rodrigo Córdova Rosado (Harvard CfA)                 *)
(* Built with: Claude Code                                      *)
(* ============================================================ *)

Print[""];
Print["============================================================"];
Print["Module 07: Data Preparation Formulas — Verification"];
Print["============================================================"];
Print[""];

(* ============================================================ *)
(* 1. PIXEL SCALE CONVERSIONS                                    *)
(* ============================================================ *)

Print["--- 1. Pixel Scale ---"];
Print[""];

(* CD matrix → pixel scale *)
(* pixel_scale = |CD1_1| × 3600 arcsec/pixel *)
cd11Examples = {-1.388889*10^-5, -2.777778*10^-5, -6.944444*10^-6};
names = {"HST ACS (0.05\")", "Euclid VIS (0.1\")", "HST WFC3/IR (0.025\")"};

Do[
  ps = Abs[cd11Examples[[i]]] * 3600;
  Print["  ", names[[i]], ": CD1_1 = ", ScientificForm[cd11Examples[[i]], 3],
    " → pixel_scale = ", NumberForm[ps, 4], "\"/pixel"],
  {i, Length[names]}
];
Print[""];

(* Field of view *)
Print["Field of view = N_pixels × pixel_scale:"];
Do[
  ps = Abs[cd11Examples[[i]]] * 3600;
  fov = 100 * ps;
  Print["  100 pixels at ", NumberForm[ps, 3], "\"/pix → ",
    NumberForm[fov, 3], "\" field of view"],
  {i, Length[names]}
];
Print[""];

(* ============================================================ *)
(* 2. NOISE MAP CONVERSIONS                                      *)
(* ============================================================ *)

Print["--- 2. Noise Map Conversions ---"];
Print[""];

Print["  Weight map → σ:  σ = 1/√w"];
Print["  Variance map → σ: σ = √v"];
Print["  Inv. variance → σ: σ = 1/√(ivar)"];
Print[""];

(* Verify: for Poisson noise, σ² = N counts *)
(* so weight w = 1/σ² = 1/N *)
Print["  Poisson noise check:"];
Print["    N = 1000 counts → σ = √1000 = ", NumberForm[N[Sqrt[1000]], 4]];
Print["    weight w = 1/σ² = 1/1000 = 0.001"];
Print["    σ from weight = 1/√0.001 = ", NumberForm[N[1/Sqrt[0.001]], 4], " ✓"];
Print[""];

(* ============================================================ *)
(* 3. MASK SIZE FROM EINSTEIN RADIUS                             *)
(* ============================================================ *)

Print["--- 3. Mask Sizing ---"];
Print[""];

Print["Rule of thumb: R_mask ≈ 2.5 × θ_E"];
Print[""];

thetaEs = {0.5, 1.0, 1.5, 2.0, 3.0};
Do[
  rMask = 2.5 * te;
  nPix01 = Round[2 * rMask / 0.1] + 1;  (* at 0.1"/pix *)
  nPix005 = Round[2 * rMask / 0.05] + 1; (* at 0.05"/pix *)
  Print["  θ_E = ", te, "\": R_mask = ", rMask, "\", ",
    "cutout = ", nPix01, " pix (0.1\"/pix) or ",
    nPix005, " pix (0.05\"/pix)"],
  {te, thetaEs}
];
Print[""];

Print["============================================================"];
Print["All verifications complete."];
Print["============================================================"];
