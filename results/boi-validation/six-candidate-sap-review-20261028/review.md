# Six queued main-Rust signals: SAP fit audit

All six were held at native SAP fitting before the V3 probability stage. Their FPP and NFPP from those attempts were not computed. The October 12 ledger also records UNESTIMATED FPP for each exact signal.

Native radius-ratio precision gate: k / ((q84(k)-q16(k))/2) >= 3. All six fail it. These are geometry/size constraints; this statistic differs from the detection or depth S/N.

| KIC | Supplied P (d) | Local PSF depth (%) | Rough PSF radius (Earth) | SAP k: 16%, 50%, 84% | SAP impact b median | SAP k S/N | SAP duration (h) | Live / effective target |
|---|---:|---:|---:|---|---:|---:|---:|---|
| 6184073 | 105.366148 | 4.778 | 64.9 | 0.357, 0.462, 0.902 | 1.128 | 1.70 | 12.34 | 500 / 5000 |
| 3935011 | 3.141265 | 1.485 | 11.2 | 0.133, 0.441, 0.816 | 1.313 | 1.29 | 1.51 | 500 / 5000 |
| 5698054 | 80.721628 | 3.505 | 12.3 | 0.213, 0.350, 0.668 | 0.997 | 1.54 | 4.04 | 500 / 5000 |
| 893184 | 39.810773 | 3.109 | 23.6 | 0.148, 0.363, 0.883 | 1.218 | 0.99 | 13.40 | 500 / 5000 |
| 6209806 | 14.226579 | 2.309 | 10.8 | 0.234, 0.548, 0.785 | 1.464 | 1.99 | 2.23 | 300 / 2000 |
| 8234517 | 18.670731 | 1.352 | 10.5 | 0.188, 0.593, 0.835 | 1.455 | 1.83 | 6.13 | 500 / 5000 |

KIC 5698054 additionally failed transit protection: one cadence reached by the fitted model was outside the original protected transit region. Every other listed native check passed for the six cases.

The prior SAP fits used five worker processes, dlogz_init=0.1 and 20 exposure subsamples. Five used 500 live points / 5000 effective-sample target. KIC 6209806 used 300 / 2000. The current PSF candidates all use 500 / 5000.

KIC 6184073: adopted host R=2.72 solar, Teff=4892 K; the local PSF depth implies ~65 Earth radii, or ~0.60 solar radius, at that host. A large eclipse with this host is not a strong ordinary-planet prospect. The held SAP geometry has b~1.13 and k 0.36-0.90 (16-84%), not a measured planetary radius.

Relative follow-up interest among these six: KIC 3935011, 5698054 and 8234517 remain plausible giant-companion candidates worth the already queued PSF inference. KIC 6209806 has a much brighter neighbour at 8.93 arcsec and variable local depths; KIC 8234517 also has a brighter neighbour at 8.56 arcsec. KIC 893184 has a larger rough size, nominal odd/even difference 3.43 sigma, only seven measured events and earlier stellar-radius disagreement. These concerns do not reproduce an EB mechanism and do not establish false-positive dispositions.

Rough radius caveat: Rough sizes use the earlier local PSF event-depth diagnostic and the declared current host radius: 109.076*R_star*sqrt(depth). They are not fitted radii. Grazing geometry, delivered PSF response, local-baseline choices, blends and stellar uncertainties remain to be assessed.

Full priors, parameter quantiles, checks, local diagnostics and evidence hashes: [review.json](review.json).
