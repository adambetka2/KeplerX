# Final three KIC checks — 28 October 2026

All three reached terminal outcomes by **2026-10-28T13:09:22+01:00**. No jobs remain, the queue coordinator and case workers have exited, and the completion monitor has been deleted. No new planet was validated.

| Signal | Native fit | Probability result | Scientific outcome |
|---|---|---|---|
| KIC 6113801 / BOI-2102266442767690112-1 | Completed; held only on transit protection | Not run; FPP and NFPP unavailable | REMAINS_PC; preparation/fit acceptance unresolved |
| KIC 4948140, signal 3 / BOI-2073666564790289664-3 | Passed all six acceptance checks | FPP 62.54%, NFPP 61.79%; 2 repeats, early stop | REMAINS_PC; not statistically validated |
| KIC 3935011 / BOI-2100203999471640064-1 | Skipped as requested; direct PSF_FLAT input | FPP 97.30%, NFPP 97.30%; 2 repeats, early stop | REMAINS_PC; nearby false-positive model preference, no reproduced EB denial |

## KIC 6113801

The sampler completed 631,706 likelihood calls. The native review reports weight ESS 4,291 and radius-ratio S/N 19.59. The declared settings remained 500 live points and a 5,000 effective-sample target. Every native acceptance check except protection passed.

The supplied duration was 2.40 hours; the fitted duration was 5.04 hours (4.74 hours for the representative exact model). The model reaches **855 cadences outside the original detrending protection**. The fitted period is 0.9337559343 days. The representative intended-host planet-model radius is 1.95 Earth radii, but this is provisional and is not an adopted planetary radius. No probability study was allocated because the fit was held.

## KIC 4948140, signal 3

The fitted period is 0.5773190449 days and duration is 1.55 hours. All native acceptance checks passed; the review reports weight ESS 5,163 and radius-ratio S/N 17.82. The representative native radius is 2.08 Earth radii. Across the two probability repeats, best intended-host TP draws have radii 1.875–1.915 Earth radii; these best draws are not a radius posterior.

The two FPP values were 62.88% and 62.20%; NFPP values were 62.13% and 61.45%. Multiple nearby eclipsing-binary scenarios account for most of the false-positive weight. The predeclared futility screen stopped the campaign after two repeats. The seasonal-depth and listed physical checks passed, but probability thresholds did not.

## KIC 3935011

This study supplied the original 3.1412645577-day clock and delivered PSF_FLAT curve directly to the exact fast_TRICERATOPS V3 signed-response likelihood. It used quarter median normalization, reviewed per-quarter noise and 80 exposure subsamples. No additional juliet/dynesty fit was performed. The measured positive Gaia parallax was retained; no Bailer-Jones replacement was needed for this source.

The two FPP values were 94.6041% and 99.9996%, with essentially the same NFPP. Nearby-source NEB scenario **Gaia DR3 2100203995173423872** dominates both repeats. The best intended-host TP draws have radii 18.74–19.33 Earth radii; they are conditional best draws, not an adopted radius or credible interval. The two-repeat futility screen stopped this study.

## Timing and scope

Probability calculations took approximately **9.8 minutes per repeat** for KIC 4948140 and **7.1 minutes per repeat** for KIC 3935011, with each first pair calculated in parallel. Both stopped after the first pair, so a full 20-repeat runtime was unnecessary. KIC 6113801 finished its native output at 13:00 Warsaw; the last direct calculation finished at 13:08:45 Warsaw, followed by closeout at 13:09:22 Warsaw.

All three remain unvalidated candidates. A failed native gate or statistical preference for a nearby source is not a reproduced astrophysical false-positive mechanism. No DENIED_FP disposition was assigned. All earlier removed signals remain outside scheduling and their records are retained.

Full parameter values, exact identities, native settings, repeat timings, scenario rankings and links to hashed evidence are in [results.json](results.json).
