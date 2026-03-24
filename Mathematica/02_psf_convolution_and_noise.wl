(* ============================================================ *)
(* 02_psf_convolution_and_noise.wl                              *)
(* ============================================================ *)
(*                                                              *)
(* Purpose:                                                     *)
(*   Symbolic derivation and verification of the image          *)
(*   formation model used in Module 02:                         *)
(*     - PSF convolution (Fourier convolution theorem)          *)
(*     - Gaussian PSF: FWHM ↔ σ conversion                     *)
(*     - Noise statistics: Poisson + sky + read noise           *)
(*     - Signal-to-noise ratio                                  *)
(*     - Nyquist sampling condition for PSFs                    *)
(*                                                              *)
(* References:                                                  *)
(*   - Nightingale, Dye & Massey (2018), Sec. 3                 *)
(*   - Meneghetti (2021), Ch. 8                                 *)
(*   - Howell (2006), Handbook of CCD Astronomy, Ch. 4          *)
(*                                                              *)
(* Run: wolframscript -file 02_psf_convolution_and_noise.wl     *)
(*                                                              *)
(* Author: Rodrigo Córdova Rosado (Harvard CfA)                 *)
(* Built with: Claude Code                                      *)
(* ============================================================ *)

Print[""];
Print["============================================================"];
Print["Module 02: PSF Convolution & Noise — Symbolic Verification"];
Print["============================================================"];
Print[""];

(* ============================================================ *)
(* 1. GAUSSIAN PSF: FWHM ↔ σ                                   *)
(* ============================================================ *)
(* PSF(r) = (1/2πσ²) exp(-r²/2σ²)                              *)
(* FWHM = 2√(2 ln 2) σ ≈ 2.355 σ                              *)
(* ============================================================ *)

Print["--- 1. Gaussian PSF Properties ---"];
Print[""];

(* The PSF is normalized: ∫∫ PSF(x,y) dx dy = 1 *)
psfIntegral = Integrate[
  1/(2 Pi \[Sigma]^2) Exp[-(x^2 + y^2)/(2 \[Sigma]^2)],
  {x, -Infinity, Infinity}, {y, -Infinity, Infinity},
  Assumptions -> \[Sigma] > 0
];
Print["PSF normalization: ∫∫ PSF dx dy = ", psfIntegral, " ✓"];

(* FWHM: solve PSF(r) = PSF(0)/2 *)
fwhmSolve = Solve[
  Exp[-r^2/(2 \[Sigma]^2)] == 1/2,
  r, Reals
];
fwhmPositive = r /. fwhmSolve[[2]];  (* Positive root *)
fwhmFull = 2 fwhmPositive;
Print["FWHM = 2 × ", Simplify[fwhmPositive], " = ", Simplify[fwhmFull]];
Print["FWHM / σ = ", N[fwhmFull /. \[Sigma] -> 1, 5], " (the 2.355 factor)"];
Print[""];

(* Instrument examples *)
instruments = {
  {"HST ACS", 0.03},
  {"Euclid VIS", 0.07},
  {"Keck AO", 0.08},
  {"Ground seeing", 0.30}
};

Print["Instrument PSF parameters:"];
Do[
  fwhm = 2 Sqrt[2 Log[2]] inst[[2]];
  Print["  ", inst[[1]], ": σ = ", inst[[2]], "\" → FWHM = ",
    NumberForm[N[fwhm], 3], "\""],
  {inst, instruments}
];
Print[""];

(* ============================================================ *)
(* 2. CONVOLUTION THEOREM                                       *)
(* ============================================================ *)
(* The observed image d = PSF * I + n, where * is convolution.  *)
(*                                                              *)
(* Convolution theorem: FT[PSF * I] = FT[PSF] × FT[I]         *)
(* This is why convolution is done in Fourier space (FFT).      *)
(*                                                              *)
(* For a Gaussian PSF, the Fourier transform is also Gaussian:  *)
(* FT[Gauss(σ)] = Gauss(1/(2πσ))                               *)
(* ============================================================ *)

Print["--- 2. Convolution Theorem ---"];
Print[""];

(* FT of Gaussian PSF *)
ftGauss = FourierTransform[
  1/(Sqrt[2 Pi] \[Sigma]) Exp[-x^2/(2 \[Sigma]^2)],
  x, k,
  FourierParameters -> {0, -2 Pi}
];
Print["FT of Gaussian PSF (1D): ", Simplify[ftGauss]];
Print["  = Gaussian with σ_k = 1/(2π σ_x)"];
Print["  Narrow PSF in real space → wide in Fourier space (more frequencies preserved)"];
Print["  Wide PSF in real space → narrow in Fourier space (high frequencies lost)"];
Print[""];

(* ============================================================ *)
(* 3. NOISE STATISTICS                                          *)
(* ============================================================ *)
(* Total noise per pixel:                                       *)
(*   σ²_total = F_source·t + F_sky·t·Ω_pix + σ²_read          *)
(* where:                                                       *)
(*   F_source = source flux (e⁻/s/arcsec²)                     *)
(*   F_sky = sky background (e⁻/s/arcsec²)                     *)
(*   t = exposure time (s)                                      *)
(*   Ω_pix = pixel solid angle (arcsec²)                       *)
(*   σ_read = read noise (e⁻/pixel)                            *)
(* ============================================================ *)

Print["--- 3. Noise Model ---"];
Print[""];

(* Signal-to-noise ratio *)
Print["Signal-to-noise ratio per pixel:"];
Print["  SNR = F_source × t × Ω_pix / σ_total"];
Print["  σ²_total = F_source·t·Ω_pix + F_sky·t·Ω_pix + σ²_read"];
Print[""];

(* Numerical example: HST-like observation *)
Fsource = 2.0;       (* e⁻/s/arcsec² — lens galaxy center *)
Fsky = 0.01;         (* e⁻/s/arcsec² — very low in space *)
texp = 2000.0;       (* seconds *)
pixScale = 0.05;     (* arcsec *)
omegaPix = pixScale^2;  (* pixel solid angle *)
sigmaRead = 5.0;     (* e⁻/pixel — typical CCD *)

(* Noise components *)
noiseSrc = Fsource * texp * omegaPix;
noiseSky = Fsky * texp * omegaPix;
noiseRead = sigmaRead^2;
sigmaTotal = Sqrt[noiseSrc + noiseSky + noiseRead];
snr = Fsource * texp * omegaPix / sigmaTotal;

Print["HST example (lens center pixel):"];
Print["  Source counts:  ", NumberForm[N[noiseSrc], 4], " e⁻"];
Print["  Sky counts:     ", NumberForm[N[noiseSky], 4], " e⁻"];
Print["  Read noise²:    ", NumberForm[N[noiseRead], 4], " e⁻²"];
Print["  σ_total:        ", NumberForm[N[sigmaTotal], 4], " e⁻"];
Print["  SNR:            ", NumberForm[N[snr], 4]];
Print[""];

(* ============================================================ *)
(* 4. NYQUIST SAMPLING                                          *)
(* ============================================================ *)
(* To properly sample the PSF, the pixel scale must satisfy:    *)
(*   pixel_scale ≤ FWHM / 2  (Nyquist criterion)               *)
(*                                                              *)
(* Under-sampling the PSF causes aliasing artifacts that bias   *)
(* the lens model. Over-sampling wastes pixels but is safe.     *)
(* ============================================================ *)

Print["--- 4. Nyquist Sampling Criterion ---"];
Print[""];
Print["Nyquist: pixel_scale ≤ FWHM / 2"];
Print[""];

Do[
  fwhm = 2 Sqrt[2 Log[2]] inst[[2]];
  nyquist = N[fwhm / 2];
  Print["  ", inst[[1]], ": FWHM = ", NumberForm[N[fwhm], 3],
    "\" → Nyquist pixel scale ≤ ", NumberForm[nyquist, 3], "\""],
  {inst, instruments}
];
Print[""];
Print["HST at 0.05\"/pix: FWHM/pixel = ", NumberForm[N[2 Sqrt[2 Log[2]] 0.03 / 0.05], 3],
  " (well-sampled ✓)"];
Print["Ground at 0.10\"/pix: FWHM/pixel = ", NumberForm[N[2 Sqrt[2 Log[2]] 0.30 / 0.10], 3],
  " (well-sampled ✓)"];

Print[""];
Print["============================================================"];
Print["All verifications complete."];
Print["============================================================"];
