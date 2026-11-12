# TLS proposal experiment

The isolated `TLS/` copy tests stock `transitleastsquares==2.0.1` on the
existing PSF Flat 1,000-case M/K calibration sample. It leaves the sealed
survey, original M/K detector and running statistical validation unchanged.

## Comparison

Keep the original 600 synthetic planets, 100 known planets and 300 controls,
case IDs, prepared arrays, masks, injection support and scoring clocks.
The worker receives only the bound photometry, never case labels or truth.
Score after the complete blind search, using the copied original clock
matcher, zero-support exclusions and parent-cluster paired statistic.
Every eligibility failure stays in its original denominator.

The candidate replaces the BLS/coherence proposal search with stock TLS
`fused-pl`, the upstream approximate default backend. It retains up to 64
local TLS-periodogram maxima per original period band (0.5–30 and 30–100 d).
It applies the existing depth and coarse-S/N limits before the unchanged
physical-template refinement, consistency gates, proposal deduplication and
global-fundamental conditioning. The fixed-period inverse-box diagnostic
remains; it is not a BLS search for planet periods.

TLS receives `1 + filtered_flux` and the inherited local-point errors. Its
input cleaner must retain every inherited cadence. Computational mass/radius
bounds encode the existing physical-density envelope of 1–150 g/cm³; they
are not measured host properties. TLS's period-grid argument is two because
its grid caps periods at span/n, while three observed transit centres can
span two periods. The existing final minimum remains three observed events.
Density fallback is an explicit ineligibility, never a silent solar default.
The duration grid and seed ranking are TLS-specific, so this experiment
compares proposal algorithms rather than claiming identical trial grids.

Use the saved BLS results only after checking the original source, design,
prepared-input identities and numerical-runtime differences. Preserve those
differences in the report; reusing historical results does not establish
an equal-load speed benchmark. Report recovery, paired wins/losses, control
proposal burden, strata, errors and CPU/elapsed costs. No detection, runtime
or false-alarm superiority is assumed.

## Run

Install `TLS/requirements.txt` through `uv` into a separate environment.
Use one worker, one numerical thread and reduced priority while sharing the
Mac. Put run outputs, logs and caches under an admitted working directory.

```bash
python -m TLS.calibration prepare --sample ORIGINAL_USABLE1000_V2 \
  --clocks ORIGINAL_FINAL10000_CLOCKS_CSV \
  --baseline ORIGINAL_USABLE1000_V2/pilot/ORIGINAL_DESIGN --out NEW_RUN
python -m TLS.calibration run --out NEW_RUN
```

Preparation checks all 1,000 identities and binds the original inputs before
search. The original sample paths may be relocated only to their matching
`prepared/` filenames with unchanged hashes. The scorer's clock file must
match the original plan. A first unexpected search error stops allocation
and preserves the exception, diagnostics and arrays; it is not a recovery
miss silently substituted for a result.

The recorded sample is on volume UUID
`5617B8CF-DE4E-45F6-AA95-0AA9D902D992`, which was not mounted when this
experiment was prepared. The two mounted corpus discs and local 2,501-star
real Flat study do not contain this calibration sample. No replacement
sample or full 1,000-case search has been launched.

## Limits

The synthetic injections were added after delivered flattening. Recovery
does not measure upstream flattening attenuation. This is a development
comparison, not a planet catalogue or statistical validation. The separate
20%-recovery gate on at least 10,000 cases remains open.
