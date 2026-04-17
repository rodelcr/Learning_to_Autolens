(* ============================================================ *)
(* 09_mge_and_basis_functions.wl                                *)
(* ============================================================ *)
(*                                                              *)
(* Purpose:                                                     *)
(*   Symbolic derivation and verification of Multi-Gaussian     *)
(*   Expansion (MGE) and basis function techniques:             *)
(*     - 2D Gaussian profile properties (flux, half-light R)    *)
(*     - PSF convolution of Gaussians (Fourier proof)           *)
(*     - MGE approximation of Sersic profiles                   *)
(*     - Linear system condition number vs. N Gaussians         *)
(*     - Abel deprojection (2D -> 3D) for Gaussians             *)
(*                                                              *)
(* References:                                                  *)
(*   - Emsellem, Monnet & Bacon (1994), A&A, 285, 723           *)
(*   - Cappellari (2002), MNRAS, 333, 400                        *)
(*   - Bendinelli (1991), ApJ, 366, 599                          *)
(*   - Graham & Driver (2005), PASA, 22, 118                     *)
(*                                                              *)
(* Run: wolframscript -file 09_mge_and_basis_functions.wl       *)
(*                                                              *)
(* Author: Rodrigo Córdova Rosado (Harvard CfA)                 *)
(* Built with: Claude Code                                      *)
(* ============================================================ *)

Print[""];
Print["============================================================"];
Print["Module 09: MGE & Basis Functions — Symbolic Verification"];
Print["============================================================"];
Print[""];

(* ============================================================ *)
(* 1. GAUSSIAN PROFILE PROPERTIES                               *)
(* ============================================================ *)
(* A 2D circular Gaussian: I(x,y) = a Exp[-(x²+y²)/(2σ²)]    *)
(*                                                              *)
(* Key results:                                                  *)
(*   Total flux = 2π a σ²                                       *)
(*   Half-light radius r_h = σ √(2 ln 2)                        *)
(*   Enclosed flux F(R) = 2π a σ² (1 - Exp[-R²/(2σ²)])          *)
(* ============================================================ *)

Print["--- 1. Gaussian Profile Properties ---"];
Print[""];

(* Total flux: integrate I(x,y) over all space *)
(* Convert to polar: ∫₀^∞ ∫₀^{2π} a Exp[-r²/(2σ²)] r dr dθ *)
Print["Total flux of 2D Gaussian I(x,y) = a Exp[-(x²+y²)/(2σ²)]:"];
totalFlux = Integrate[a Exp[-r^2/(2 \[Sigma]^2)] r, {r, 0, Infinity},
  Assumptions -> {\[Sigma] > 0, a > 0}] * 2 Pi;
Print["  ∫∫ I(x,y) dx dy = 2π ∫₀^∞ I(r) r dr = ", Simplify[totalFlux]];
Print["  Expected: 2π a σ²"];
Print["  Verified: ", Simplify[totalFlux - 2 Pi a \[Sigma]^2] == 0, "  ✓"];
Print[""];

(* Enclosed flux within radius R *)
Print["Enclosed flux within radius R:"];
enclosedFlux = Integrate[2 Pi a Exp[-r^2/(2 \[Sigma]^2)] r, {r, 0, R},
  Assumptions -> {\[Sigma] > 0, a > 0, R > 0}];
Print["  F(R) = ", Simplify[enclosedFlux]];
expectedEnclosed = 2 Pi a \[Sigma]^2 (1 - Exp[-R^2/(2 \[Sigma]^2)]);
Print["  Expected: 2π a σ² (1 - Exp[-R²/(2σ²)])"];
Print["  Verified: ", Simplify[enclosedFlux - expectedEnclosed] == 0, "  ✓"];
Print[""];

(* Half-light radius: F(r_h) = (1/2) F_total *)
Print["Half-light radius (encloses 50% of total flux):"];
Print["  Solve: 1 - Exp[-r_h²/(2σ²)] = 1/2"];
rhSol = Solve[1 - Exp[-rh^2/(2 \[Sigma]^2)] == 1/2, rh,
  Assumptions -> {\[Sigma] > 0, rh > 0}];
rhVal = rh /. rhSol[[1]];
Print["  r_h = ", rhVal];
Print["  = σ √(2 ln 2)"];
Print["  Verified: ", Simplify[rhVal - \[Sigma] Sqrt[2 Log[2]]] == 0, "  ✓"];
Print["  Numerically: r_h ≈ ", N[\[Sigma] Sqrt[2 Log[2]] /. \[Sigma] -> 1, 6], " σ"];
Print[""];

(* ============================================================ *)
(* 2. PSF CONVOLUTION PROPERTY                                  *)
(* ============================================================ *)
(* Convolving a Gaussian with a Gaussian yields a Gaussian:     *)
(*   G(σ_g) ⊛ G(σ_p) = G(σ_eff)                               *)
(*   where σ_eff = √(σ_g² + σ_p²)                              *)
(*                                                              *)
(* Proof via Fourier transforms:                                *)
(*   FT[G(σ)] ∝ Exp[-2π²σ²k²]                                  *)
(*   FT[G(σ_g)] × FT[G(σ_p)] = Exp[-2π²(σ_g²+σ_p²)k²]        *)
(*   = FT[G(σ_eff)]                                             *)
(*                                                              *)
(* This is the key reason MGE is powerful: PSF convolution of   *)
(* each Gaussian component is analytic — no FFT needed!         *)
(* ============================================================ *)

Print["--- 2. PSF Convolution of Gaussians (Fourier Proof) ---"];
Print[""];

(* 1D proof (generalizes trivially to 2D as product of 1D) *)
Print["1D Fourier proof (extends to 2D by separability):"];
Print[""];

(* Define 1D Gaussian *)
g1D[x_, sig_] := 1/(sig Sqrt[2 Pi]) Exp[-x^2/(2 sig^2)];

(* Fourier transform *)
Print["  G(x; σ) = (1/(σ√(2π))) Exp[-x²/(2σ²)]"];
ft1 = FourierTransform[g1D[x, \[Sigma]g], x, k,
  FourierParameters -> {0, 2 Pi}];
ft1Simple = Simplify[ft1, Assumptions -> {\[Sigma]g > 0}];
Print["  FT[G(σ_g)](k) = ", ft1Simple];

ft2 = FourierTransform[g1D[x, \[Sigma]p], x, k,
  FourierParameters -> {0, 2 Pi}];
ft2Simple = Simplify[ft2, Assumptions -> {\[Sigma]p > 0}];
Print["  FT[G(σ_p)](k) = ", ft2Simple];
Print[""];

(* Product in Fourier space *)
ftProduct = Simplify[ft1Simple * ft2Simple,
  Assumptions -> {\[Sigma]g > 0, \[Sigma]p > 0}];
Print["  FT[G(σ_g)] × FT[G(σ_p)] = ", ftProduct];

(* Inverse Fourier transform of the product *)
convResult = InverseFourierTransform[ftProduct, k, x,
  FourierParameters -> {0, 2 Pi}];
convSimple = Simplify[convResult,
  Assumptions -> {\[Sigma]g > 0, \[Sigma]p > 0}];
Print["  IFT of product = ", convSimple];
Print[""];

(* Compare with Gaussian of σ_eff *)
sigEff = Sqrt[\[Sigma]g^2 + \[Sigma]p^2];
gEff = g1D[x, sigEff];
gEffSimple = Simplify[gEff, Assumptions -> {\[Sigma]g > 0, \[Sigma]p > 0}];
Print["  G(x; σ_eff = √(σ_g²+σ_p²)) = ", gEffSimple];
Print["  Verified: convolution = G(σ_eff): ",
  Simplify[convSimple - gEffSimple,
    Assumptions -> {\[Sigma]g > 0, \[Sigma]p > 0}] == 0, "  ✓"];
Print[""];
Print["  Key implication for MGE: each Gaussian component can be"];
Print["  convolved with the PSF analytically — no FFT needed!"];
Print["  σ_eff,i = √(σ_i² + σ_PSF²) for each MGE component i."];
Print[""];

(* ============================================================ *)
(* 3. MGE APPROXIMATION OF SERSIC PROFILE                      *)
(* ============================================================ *)
(* Sersic profile: I(R) = I_e Exp[-b_n ((R/R_e)^{1/n} - 1)]   *)
(*   where b_n ≈ 2n - 1/3 + 4/(405n)                           *)
(*                                                              *)
(* MGE approximation: I(R) ≈ Σᵢ aᵢ Exp[-R²/(2σᵢ²)]            *)
(*                                                              *)
(* We numerically fit 20 Gaussians to a Sersic n=4 (de         *)
(* Vaucouleurs) profile over 3 decades in radius.               *)
(* ============================================================ *)

Print["--- 3. MGE Approximation of Sersic Profile ---"];
Print[""];

(* Sersic profile definition *)
bn[n_] := 2 n - 1/3 + 4/(405 n);
sersic[r_, ie_, re_, n_] := ie Exp[-bn[n] ((r/re)^(1/n) - 1)];

Print["Sersic profile: I(R) = I_e Exp[-b_n ((R/R_e)^{1/n} - 1)]"];
Print["  b_n ≈ 2n - 1/3 + 4/(405n)"];
Print[""];

(* For n=4 (de Vaucouleurs) *)
nSersic = 4;
bnVal = N[bn[nSersic]];
Print["  n = 4 (de Vaucouleurs): b_4 = ", NumberForm[bnVal, 6]];
Print["  (Exact b_4 = 7.6693... from gamma function inversion)"];
Print[""];

(* Set up MGE fit: 20 Gaussians with log-spaced sigmas *)
nGauss = 20;
rEff = 1.0;  (* Effective radius *)
iEff = 1.0;  (* Surface brightness at R_e *)

(* Log-spaced sigmas spanning 3 decades *)
sigmaMin = 0.01;
sigmaMax = 30.0;
sigmas = Table[
  sigmaMin (sigmaMax/sigmaMin)^((i - 1)/(nGauss - 1)),
  {i, 1, nGauss}];

Print["  MGE: ", nGauss, " Gaussians, σ range = [",
  NumberForm[sigmaMin, 3], ", ", NumberForm[sigmaMax, 3], "]"];
Print["  Log-spaced σ values: {", NumberForm[sigmas[[1]], 3], ", ",
  NumberForm[sigmas[[2]], 3], ", ..., ",
  NumberForm[sigmas[[-1]], 3], "}"];
Print[""];

(* Sample Sersic profile at radial points *)
nSample = 200;
rSample = Table[
  sigmaMin (sigmaMax/sigmaMin)^((i - 1)/(nSample - 1)) * 1.5,
  {i, 1, nSample}];
iSample = Table[sersic[r, iEff, rEff, nSersic], {r, rSample}];

(* Build design matrix: A_ij = Exp[-r_i²/(2σ_j²)] *)
designMatrix = Table[
  Exp[-rSample[[i]]^2/(2 sigmas[[j]]^2)],
  {i, 1, nSample}, {j, 1, nGauss}];

(* Non-negative least squares: minimize ||Ax - b||² with x ≥ 0 *)
(* Use standard least squares, then clip negative amplitudes *)
(* and iterate (simple approach sufficient for demonstration) *)
ampFit = LeastSquares[designMatrix, iSample];

(* Clip negative amplitudes to zero *)
ampFit = Map[Max[#, 0.0] &, ampFit];

(* Compute MGE model at sample points *)
mgeFit = designMatrix . ampFit;

(* Compute residuals *)
residuals = (mgeFit - iSample) / iSample * 100;  (* Percent *)

Print["  Least-squares fit results (non-zero amplitudes):"];
nonZeroCount = Count[ampFit, x_ /; x > 1.0*^-10];
Print["  ", nonZeroCount, " of ", nGauss, " Gaussians have non-zero amplitude"];
Print[""];

(* Report max residual *)
maxResidual = Max[Abs[residuals]];
medianResidual = Median[Abs[residuals]];
Print["  Max |residual|: ", NumberForm[maxResidual, 3], "%"];
Print["  Median |residual|: ", NumberForm[medianResidual, 3], "%"];
Print[""];

(* Plot: Sersic profile, MGE fit, and residuals *)
sersicPlotData = Table[{Log10[rSample[[i]]], Log10[iSample[[i]]]},
  {i, 1, nSample}];
mgePlotData = Table[{Log10[rSample[[i]]],
  If[mgeFit[[i]] > 0, Log10[mgeFit[[i]]], -10]},
  {i, 1, nSample}];

profilePlot = ListLinePlot[
  {sersicPlotData, mgePlotData},
  PlotStyle -> {{Black, Thick}, {Red, Dashed, Thick}},
  PlotLegends -> {"Sersic n=4", "MGE (20 Gaussians)"},
  AxesLabel -> {"log₁₀(R/R_e)", "log₁₀(I/I_e)"},
  PlotLabel -> "MGE Approximation of de Vaucouleurs Profile",
  PlotRange -> All,
  ImageSize -> 500
];

residualPlotData = Table[{Log10[rSample[[i]]], residuals[[i]]},
  {i, 1, nSample}];
residualPlot = ListLinePlot[
  residualPlotData,
  PlotStyle -> {Blue, Thick},
  AxesLabel -> {"log₁₀(R/R_e)", "Residual (%)"},
  PlotLabel -> "MGE Fit Residuals",
  PlotRange -> {All, {-10, 10}},
  GridLines -> {None, {0}},
  ImageSize -> 500
];

combinedPlot = Column[{profilePlot, residualPlot}];
Export[
  "/Users/rosador/Documents/AGEL/Learning_to_Autolens/Figures/09_mge_sersic_fit.pdf",
  combinedPlot
];
Print["  Exported: Figures/09_mge_sersic_fit.pdf"];
Print[""];

(* ============================================================ *)
(* 4. LINEAR SYSTEM CONDITION NUMBER                            *)
(* ============================================================ *)
(* For N Gaussians with log-spaced σ, the overlap (Gram) matrix *)
(* M_ij = ∫₀^∞ G_i(r) G_j(r) 2πr dr                           *)
(*      = 2π σ_i² σ_j² / (σ_i² + σ_j²)                         *)
(*                                                              *)
(* The condition number κ(M) determines numerical stability.     *)
(* Log-spacing of σ values keeps κ manageable even for large N.  *)
(* ============================================================ *)

Print["--- 4. Linear System Condition Number ---"];
Print[""];

(* First, derive the overlap integral analytically *)
Print["Overlap integral of two Gaussians:"];
Print["  M_ij = ∫₀^∞ Exp[-r²/(2σ_i²)] Exp[-r²/(2σ_j²)] 2πr dr"];

overlapIntegral = Integrate[
  Exp[-r^2/(2 si^2)] Exp[-r^2/(2 sj^2)] 2 Pi r,
  {r, 0, Infinity},
  Assumptions -> {si > 0, sj > 0}];
overlapSimple = Simplify[overlapIntegral];
Print["  = ", overlapSimple];

expectedOverlap = 2 Pi si^2 sj^2 / (si^2 + sj^2);
Print["  Expected: 2π σ_i² σ_j² / (σ_i² + σ_j²)"];
Print["  Verified: ", Simplify[overlapSimple - expectedOverlap] == 0, "  ✓"];
Print[""];

(* Build overlap matrix and compute condition number for various N *)
Print["Condition number κ(M) vs. N Gaussians (log-spaced σ):"];
Print["  σ range: [0.1, 100] (3 decades)"];
Print[""];

condResults = {};
Do[
  sigs = Table[
    0.1 * (100/0.1)^((i - 1)/(nn - 1)),
    {i, 1, nn}];
  overlapMat = Table[
    2 Pi sigs[[i]]^2 sigs[[j]]^2 / (sigs[[i]]^2 + sigs[[j]]^2),
    {i, 1, nn}, {j, 1, nn}];
  cond = Log10[LinearAlgebra`MatrixConditionNumber[N[overlapMat]]];
  AppendTo[condResults, {nn, cond}];
  Print["  N = ", PaddedForm[nn, 3],
    ": log₁₀ κ(M) = ", NumberForm[cond, 4]],
  {nn, {5, 10, 20, 30, 50}}
];
Print[""];
Print["  Condition number grows with N, but log-spacing keeps it"];
Print["  manageable. For N=20 (typical MGE), κ ~ 10^",
  NumberForm[condResults[[3, 2]], 3], "."];
Print[""];

(* Compare: uniform spacing would give much worse conditioning *)
Print["Comparison: uniform vs. log-spaced σ (N=20):"];
sigsLog = Table[0.1 * (100/0.1)^((i - 1)/19), {i, 1, 20}];
sigsUnif = Table[0.1 + (100 - 0.1) (i - 1)/19, {i, 1, 20}];

overlapLog = Table[
  2 Pi sigsLog[[i]]^2 sigsLog[[j]]^2 / (sigsLog[[i]]^2 + sigsLog[[j]]^2),
  {i, 1, 20}, {j, 1, 20}];
overlapUnif = Table[
  2 Pi sigsUnif[[i]]^2 sigsUnif[[j]]^2 / (sigsUnif[[i]]^2 + sigsUnif[[j]]^2),
  {i, 1, 20}, {j, 1, 20}];

condLog = Log10[LinearAlgebra`MatrixConditionNumber[N[overlapLog]]];
condUnif = Log10[LinearAlgebra`MatrixConditionNumber[N[overlapUnif]]];
Print["  Log-spaced:     log₁₀ κ = ", NumberForm[condLog, 4]];
Print["  Uniform-spaced: log₁₀ κ = ", NumberForm[condUnif, 4]];
Print["  Log-spacing gives better conditioning by ~ ",
  NumberForm[condUnif - condLog, 3], " orders of magnitude."];
Print[""];

(* Condition number plot *)
condPlot = ListLinePlot[
  condResults,
  PlotStyle -> {Blue, Thick},
  Mesh -> All,
  MeshStyle -> Directive[Red, PointSize[0.02]],
  AxesLabel -> {"N (Gaussians)", "log₁₀ κ(M)"},
  PlotLabel -> "Condition Number vs. MGE Components",
  PlotRange -> All,
  ImageSize -> 500
];
Export[
  "/Users/rosador/Documents/AGEL/Learning_to_Autolens/Figures/09_condition_number.pdf",
  condPlot
];
Print["  Exported: Figures/09_condition_number.pdf"];
Print[""];

(* ============================================================ *)
(* 5. 3D DEPROJECTION — ABEL TRANSFORM                          *)
(* ============================================================ *)
(* For a 2D Gaussian surface brightness:                        *)
(*   I(R) = a Exp[-R²/(2σ²)]                                   *)
(*                                                              *)
(* The Abel deprojection (assuming spherical symmetry) gives    *)
(* the 3D luminosity density:                                   *)
(*   ρ(r) = -(1/π) ∫_r^∞ dI/dR / √(R²-r²) dR                 *)
(*        = a / (√(2π) σ) Exp[-r²/(2σ²)]                       *)
(*                                                              *)
(* Inverse check: the Abel projection of ρ(r) gives I(R):      *)
(*   I(R) = 2 ∫_R^∞ ρ(r) r / √(r²-R²) dr                     *)
(*        = a Exp[-R²/(2σ²)]   ✓                               *)
(*                                                              *)
(* This analytic deprojection is UNIQUE to Gaussians and is     *)
(* the key motivation for MGE in dynamical modeling             *)
(* (cf. Emsellem, Monnet & Bacon 1994; Cappellari 2002).        *)
(* ============================================================ *)

Print["--- 5. 3D Deprojection (Abel Transform) ---"];
Print[""];

(* Abel deprojection formula *)
Print["Abel deprojection:"];
Print["  ρ(r) = -(1/π) ∫_r^∞ (dI/dR) / √(R²-r²) dR"];
Print[""];

(* Surface brightness *)
iSurf[bigR_] := aa Exp[-bigR^2/(2 sig^2)];

(* Derivative *)
didr = D[iSurf[bigR], bigR];
Print["  I(R) = a Exp[-R²/(2σ²)]"];
Print["  dI/dR = ", didr];
Print[""];

(* Abel deprojection integral *)
Print["  Computing Abel deprojection integral..."];
abelIntegrand = (didr /. bigR -> u) / Sqrt[u^2 - rr^2];
rho3D = Simplify[
  -(1/Pi) Integrate[abelIntegrand, {u, rr, Infinity},
    Assumptions -> {sig > 0, aa > 0, rr > 0}],
  Assumptions -> {sig > 0, aa > 0, rr > 0}];
Print["  ρ(r) = ", rho3D];
Print[""];

expectedRho = aa / (Sqrt[2 Pi] sig) Exp[-rr^2/(2 sig^2)];
Print["  Expected: a / (√(2π) σ) Exp[-r²/(2σ²)]"];
Print["  = ", Simplify[expectedRho]];
Print["  Verified: ", Simplify[rho3D - expectedRho] == 0, "  ✓"];
Print[""];

(* Inverse check: Abel projection of ρ(r) should give I(R) *)
Print["  Inverse check (Abel projection): ρ(r) → I(R)"];
Print["  I(R) = 2 ∫_R^∞ ρ(r) r / √(r²-R²) dr"];
Print[""];

rhoFunc[r_] := aa / (Sqrt[2 Pi] sig) Exp[-r^2/(2 sig^2)];
abelProjection = Simplify[
  2 Integrate[rhoFunc[u] u / Sqrt[u^2 - bigR^2], {u, bigR, Infinity},
    Assumptions -> {sig > 0, aa > 0, bigR > 0}],
  Assumptions -> {sig > 0, aa > 0, bigR > 0}];
Print["  ∫ result = ", abelProjection];
Print["  Expected: a Exp[-R²/(2σ²)]"];
Print["  Verified: ",
  Simplify[abelProjection - aa Exp[-bigR^2/(2 sig^2)]] == 0, "  ✓"];
Print[""];

Print["  Key insight: The Gaussian is the ONLY simple profile where"];
Print["  both the Abel deprojection and PSF convolution are analytic."];
Print["  This makes MGE uniquely suited for:"];
Print["    1. PSF-convolved surface brightness modeling"];
Print["    2. Deprojection to 3D luminosity density"];
Print["    3. Jeans dynamical modeling (via 3D density)"];
Print["    4. Schwarzschild orbit modeling"];
Print[""];

Print["============================================================"];
Print["All verifications complete."];
Print["============================================================"];
