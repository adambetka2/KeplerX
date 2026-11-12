# V3 vetting and localization: where the time is, and the faster mathematics

28 October 2026. Adopted in production the same evening; see
[Adopted](#adopted) at the end for what changed, the measured gains and the
validation. Measured on main `85560dd53` with the KIC 12644822
positive control (P = 12.6 d, MES 221): warm stages, one thread, metrics
identical to the saved complete-CLI result. Harnesses are in
`KeplerX/results/performance-engineering/20261028/vet-probe/`.

## Where the time is

| Stage, component | Seconds | Note |
| --- | ---: | --- |
| Vetter, transit refinement (lmfit leastsq) | 2.39 | 127 model evaluations |
| cosine vs transit (curve_fit) | 0.57 | 52 model evaluations |
| local morphology (L-BFGS-B per transit) | 0.49 | |
| chi2 tests (re-whitening) | 0.44 | |
| odd/even depths | 0.43 | |
| mes, secondary (re-whitening each) | 0.21, 0.19 | |
| **Vetter total** | **4.75** | stock 30.29 s (6.4x) |
| of which batman `_nonlinear_ld` | 2.71 | 180 calls, ~15 ms each |
| Localizator, warm | 0.70 | stock 0.979 s (1.4x) |
| of which `RectBivariateSpline.__init__` (lkprf) | 0.56 | 28 PRF splines, ~20 ms each |

Short-period signals are worse: at P = 1.9 d the compiled vetting measured
about 24 s per signal, most of it batman inside the refinement and the
cosine fit (`vetter/README.md`).

## batman's integral and a faster rule

batman (2.5.2, `c_src/common.h`, built without OpenMP) computes the occulted
flux of each supersampled point as

    F(d) = 1 - integral_{x_in}^{x_out} I(x) dA(d, x, p)/dx dx,
    x_in = max(d - p, 0),  x_out = min(d + p, 1),
    I(x) = [1 - sum_{n=1..4} c_n (1 - mu^{n/2})] / norm,  mu = sqrt(1 - x^2),

with A the overlap area of the circle of radius x and the planet disc
(radius p, centre distance d). It uses a midpoint rule in A with step
dx_k = fac * acos(x_k), so a point costs about 2p / (fac * acos d) steps.
Each step makes about seven libm calls (three acos, one sqrt, four pow).

In the partial-overlap range, dA/dx = 2 x theta(x), with
theta(x) = acos((d^2 + x^2 - p^2) / (2 d x)). In the fully covered range
(x < p - d) it is 2 pi x. theta behaves like a square root at both contacts,
and I(x) like (1 - x^2)^(1/4) at the limb. Substituting x = c + h cos(phi)
with c = (x_in + x_out)/2 and h = (x_out - x_in)/2, and splitting at
x = p - d and x = 1 where they apply, gives

    delta(d) = integral_0^pi I(c + h cos phi) 2 (c + h cos phi)
                 theta(c + h cos phi) h sin(phi) d phi,

a smooth integrand that Gauss-Legendre or Clenshaw-Curtis rules integrate
spectrally: about 16-24 nodes reach 1e-10, below batman's own `max_err`.
That is 4-8x fewer libm calls per point than batman's steps for typical p.
A (d, p) table built once per fit would go further, but it needs
error control at the contacts, where delta ~ overlap^(3/2).

It is not bitwise. batman's result is defined by its own step sequence and
libm calls, and no other rule reproduces its rounding. The fits are
sensitive. An alternate disc quadrature changed fitted parameters
materially and exhausted one fit's evaluation budget
(`docs/keplerX-2026-10-28-performance.md`). Any replacement must therefore be
validated on decisions and metric tolerances, not bits.

The exact alternative is the same C loop, compiled bitwise-identically and
run over points in parallel threads. That reduces one curve's latency but
not campaign throughput, which already uses every core.

## The PRF spline is linear in the image

lkprf builds, for each trial star position, the image

    Z = (sum_{i in 3 nearest} P_i / w_i) / (nansum(sum_i P_i / w_i) * Delta^2)

from the five calibrated PRF images P_i of the channel. It then fits an
interpolating bicubic spline (`RectBivariateSpline`, s = 0) on the fixed
550 x 550 grid. For a fixed grid the spline coefficients are a linear map
L of the data, so

    L(Z) = sum_i (1 / w_i) / S * L(P_i),   S = nansum(sum_j P_j / w_j) * Delta^2.

Computing L(P_i) once per channel turns each 20 ms fit into a three-term sum
of 550 x 550 arrays, under 1 ms. lkprf accumulates Z in float32, so the
exact identity holds only up to float32 rounding (about 6e-8 relative):
again not bitwise. Localization would fall from about 0.70 to about 0.15 s;
FITS reading, coordinates, the design matrix and the solves remain.

## What this means

- With bitwise identity kept, the dominant kernels cannot be shortened, only
  run in parallel. Remaining exact work, such as running the three
  re-whitenings through the native whitening bank, is worth perhaps 20-30%
  of vetting.
- With tolerance-validated replacements of batman's rule and the PRF fit,
  estimates are 2-3x on vetting (more at short periods: about 24 s to about
  8 s) and about 4-5x on warm localization. That is about 13-15x and 6x
  against stock, not 100x. The rest of each stage is spread over many
  separate fits, whitenings and library calls.
- For the 406k campaign neither stage binds. About 10% of curves are
  vetted. In the ten-worker 90/10 probe, vetting used about 7% of CPU while
  the shared GPU (detection screen) set the pace. Faster vetting shortens a
  positive curve's latency, not the campaign.

## Measured differences from native (KIC 12644822 control)

Harnesses: `vet-probe/compare_ld.py`, `swap_metrics.py`, `swap_prf.py`. The
replacements there are unoptimised NumPy patches used only for this
comparison.

batman's rule versus the accurate integral (64 and 256 Gauss nodes agree to
2e-16, and batman with step 1e-6 agrees with it to 3e-15), on the exact
points vetting evaluates:

| Model call | Max flux error of batman | Median | Mantissa bits agreeing |
| --- | ---: | ---: | ---: |
| refinement (max_err 0.1, fac 4.4e-3) | 0.063 ppm | 0.0065 ppm | 24-27 of 53 |
| cosine fit (fac 1e-3) | 0.0031 ppm | 0.0002 ppm | 28 or more |

Running the complete `measure()` with the accurate integral changes the
metrics as follows:

- Most metrics move by 1e-13 to 3e-6 relative (18-43 bits agree). Examples:
  depth 4.631697 -> 4.631724 ppt, duration 3e-6, max MES 1e-8, odd and even
  depths 1e-8, global chi2 1e-8, t0 9e-10, period 1e-13.
- Optimiser-sensitive values move more. The impact parameter goes from
  3.0e-5 to 1.6e-5; both are zero within its degeneracy. The summed
  per-transit morphology statistic `spsd_bic_stat` goes from -1752.24 to
  -1753.43, and `ramp_bic_stat` moves by 0.036.
- Transit counts, the rounded channel and temporal chi2 values and the
  secondary phases are unchanged.

Localization, with the linear PRF spline:

- Spline coefficients differ by about 5e-7 relative and PRF values by about
  1e-7. This is lkprf's own float32 accumulation of the image.
- Centroid RA and Dec differ by about 1e-13 relative (about 1e-8 arcsec).
  Their uncertainties and chi2 differ by about 1e-8. The offset significance
  goes from 4.24866635 to 4.24866676 sigma: 23-45 bits agree.
- Warm stage time is 0.67 s native and 0.25 s with the patch.

These are single-control results. Before adoption they need checking
across many signals, especially the fits near degeneracies.

## Adopted

Adam chose to adopt both replacements and asked for localization to go
further. Production is no longer bitwise native in these two stages; every
other stage still is.

### Localizator (`localizator/accelerated_model.py`, commit `ca52a8cde`)

The 17-quarter run of the same control exposed a defect in upstream's pixel
regression. Each continuous segment gets a quadratic B-spline basis and a
quadratic polynomial, and the B-splines reproduce the polynomial, so
X^T W X is singular. Its condition number is 1e18-1e21 in every quarter,
`np.linalg.solve` returns or raises by rounding, and about half the nuisance
uncertainties come out NaN. Q7, Q11 and Q13 raised "Singular matrix", which
marks the whole 17-quarter localization UNRESOLVED. The transit depth itself
is estimable, because the null space has no transit component. In the
quarters native does solve, its depths sit about 1e-4 sigma from a stable
solve of the same model.

| Replacement | Mathematics | Agreement |
| --- | --- | --- |
| Pixel regression | Pivoted Cholesky of the unit-diagonal X^T X drops the redundant nuisance columns (pivots at most 6e-16; every kept pivot at least 4e-3). The unweighted nuisance fit is removed from each pixel and from the transit columns, an exact reparametrization. Each pixel is then solved from the Cholesky factor of its augmented weighted normal equations. A transit that the other columns explain still raises. | Depths within 1e-7 sigma of Householder QR, errors within 5e-9 |
| PRF preparation | spline(Z) = sum_i spline(P_i) / (w_i S); the five per-channel splines are fitted once and cached on disk (`runtime/cache/prf-basis`) | Within lkprf's float32 rounding (1e-7) |
| Joint fit | PRF models cached by quarter and position; sky to pixel by `wcs.all_world2pix` | Bitwise (0 of 5,100 positions differ) |
| Reading and setup | Each TPF hashed, decompressed and parsed once; channel PRF read once per process; Gaia propagation for all quarters in one astropy call; scipy's B-spline evaluation | Bitwise, except the B-splines (4e-16) |

Warm stage times on the KIC 12644822 control (harnesses `vet-probe/loc17.py`,
`loc14.py`, `loc_compare.py`, `loc_qr.py`):

| Case | Native | Now | Result |
| --- | ---: | ---: | --- |
| 1 quarter | 0.70 s | 0.048 s | MEASURED both; every output within 18-40 bits |
| the 14 quarters native solves | 8.08 s | 0.54 s | MEASURED both; centroid within 1e-13 to 1e-12, uncertainties 2e-7, nsigma 1.94876 -> 1.94876 |
| all 17 quarters | 9.5 s | 0.62 s | UNRESOLVED (three singular quarters) -> MEASURED |

What remains in the 0.62 s: decompression and FITS parsing of each TPF
(about 17 ms per quarter), WCS construction (4 ms), the regression (10 ms),
the upstream design matrix (2 ms) and the joint fit (0.1 s for 17 quarters).

### Vetter (`vetter/residual_kernels.c` `v3_nonlinear_ld`, `transit.py`)

The occulted flux is the integral batman's rule approximates, including
batman's intensity clamp at x = 0.99995 and its two flux-1 exits. The
covered disc is a polynomial in s = mu^(1/2), so it is integrated in closed
form. The partial ring is cut at x = 0.9 and at the clamp, and integrated in
x below 0.9 and in s above it, where the limb is regular. Each piece is
mapped by u = c - h cos(phi), which removes theta's square-root contacts,
and takes a Gauss-Legendre rule in phi. Its size is chosen from the nearest
singularity beyond the piece's ends, using the Bernstein ellipse of the phi
interval: n = 36 / (2 log rho), between 4 and 64 nodes, 17 on average.
`transit.batman_rule` restores batman's rule bit for bit; the tests use it
to prove every other compiled replacement still exact.

- Accuracy against adaptive quadrature: largest error 2e-13 over 1,560
  geometries with p = 0.005-0.6 (contacts, limb crossings, d = p), and 8e-14
  for p = 0.8-3. batman's error at the refinement's step is 4e-8 for typical
  planets and 3e-6 for planets larger than the star.
- Cost: 0.38 us per point, against 2.7 us at the refinement's step and
  11.6 us at the cosine fit's step.

Twelve signals, with batman's rule and with the integral in the same process
(`vet-probe/vet_many.py`, `refine_chi2.py`). The batman run equals the
saved native results bit for bit on the two long controls. The others were
saved under different configurations, so they are compared in-process only.

| Signal | P (d) | MES | Seconds, batman / integral | Depth ppt | b | Refinement objective, integral - batman |
| --- | ---: | ---: | --- | --- | --- | ---: |
| KIC 12644822 | 12.61 | 221 | 4.53 / 1.78 | 4.632 / 4.632 | 3e-5 / 9e-5 | -280 of 1.9e12 |
| control 6815 | 19.96 | 687 | 4.20 / 1.79 | 103.51 / 103.51 | 0.710 / 0.710 | -10 of 3.5e10 |
| TrES-2 Q0 | 2.47 | 124 | 0.09 / 0.06 | 14.73 / 11.48 | 0.777 / 0.000 | -6.0e9 of 1.7e12 |
| fixture 1 | 2.58 | 7.2 | 0.23 / 0.38 | 5.21 / 4.30 | 0.355 / 0.252 | -8.4e5 |
| fixture 2 | 1.22 | 4.4 | 0.49 / 0.29 | 1.116 / 1.120 | 0.002 / 0.002 | -415 |
| fixture 3 | 0.58 | 153 | 0.29 / 0.40 | 3.04 / 3.39 | 0.073 / 0.178 | -1.5e6 |
| fixture 4 | 0.50 | 6.2 | 0.13 / 0.11 | 7.69 / 9.85 | 0.000 / 0.000 | -1.6e6 |
| fixture 5 | 0.63 | 6.8 | 0.16 / 0.10 | 3.752 / 3.763 | 0.004 / 0.004 | -108 |
| fixture 6 | 2.66 | 5.6 | 1.35 / 0.42 | 5.18 / 9.81 | 0.219 / 0.000 | -3.9e6 |
| fixture 7 | 1.78 | 3.8 | 0.39 / 0.08 | 7.20 / 5.60 | 0.161 / 0.052 | +1.9e6 |
| fixture 8 | 2.48 | 4.0 | 3.64 / 2.41 | 10.26 / 10.26 | 0.054 / 0.069 | +1,013 |
| fixture 9 | 1.38 | 5.3 | 0.12 / 0.09 | 6.262 / 6.265 | 0.073 / 0.082 | +392 |

The objective is upstream's: its residual is (f - model) / sigma^2, so leastsq
minimizes the sum of (r / sigma^2)^2 rather than chi2. Both solutions are
evaluated with the accurate model.

- Long, strong signals: metrics agree to 17-44 bits. The impact parameter of
  the 12644822 control stays at zero within its degeneracy, and
  `spsd_bic_stat` moves by 1.2.
- Short or weak signals: the refinement and the other fits are multimodal.
  b enters the model only through b^2, so b = 0 is stationary, and batman's
  step noise (4e-8, not smooth in the parameters) perturbs the
  finite-difference Jacobians. The two runs end in different optima. The
  integral's optimum is the better one in 9 of 12 signals by upstream's own
  objective. Depths, b, BIC and odd/even statistics change by up to a factor
  of two on these signals; they were never reproducible below that level.
- Every failed component is the same in all 12 (`cosine_vs_transit` on
  fixture 8), as are the transit counts and MES.
- For TrES-2 b (published b about 0.84) one quarter does not constrain b:
  batman's run found 0.78 and the integral's run found 0 with a lower
  objective.

The 12 signals together take 15.6 s with batman's rule and 7.9 s with the
integral. On the control, `measure()` now spends 0.29 s in the integral; the
rest is the per-transit morphology fits (0.8 s), the refinement's leastsq
(0.5 s), odd/even (0.4 s) and the chi2 re-whitening (0.4 s).
