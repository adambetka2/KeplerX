# PSF validation queue ETA

Measured 2026-10-27T20:54:35+01:00

Central estimate: about 59 hours, finishing Friday 30 October in the morning (Warsaw time).

| Scenario | Remaining time | Completion, Warsaw |
|---|---:|---|
| fast | 37.8 h | 2026-10-29T10:40+01:00 |
| central | 58.4 h | 2026-10-30T07:21+01:00 |
| slow | 101.0 h | 2026-11-01T01:56+01:00 |

44 runnable unfinished BOIs: 3 currently fitting and 41 waiting for earlier steps or a fit slot. Two acquisition failures remain stopped and are excluded.

Estimate uses 36 measured native windows and 8 estimated windows, the completed fit history, and recent live throughput. Three fits run concurrently, each with three workers and 20 subexposures. The central model uses 450,000 calls per fit and 75 ns wall time per model point per call. Probability and final checks run automatically and overlap the fits.

The range is an engineering estimate, not a confidence interval. It assumes the Mac stays awake and current CPU availability and settings continue.

Prototype audit: only inline terminal Python was used; no skip module was saved or integrated, so there was no source file to delete.
