# Derived catalogue annotations

`annotate.py` annotates a completed M/K vetting run without changing it.
Run `uv run python pipeline/catalogue/annotate.py --run RUN --output OUTPUT`.
The output directory must be new; incomplete vetting is rejected.

BOIs receive same-Gaia-source period-ratio flags and saved catalogue matches.
HELD and DISMISSED proposals receive catalogue checks only, in a second stage.
Every original proposal and disposition is retained.
Period flags do not establish shared transit timings or signal independence.
Complete reports are reused; missing comparisons run the frozen matcher.
Host-compatible confirmed-planet associations are separated from nearby matches.
Counts describe the detected proposal cohort, not overall detection completeness.

The frozen survey detector and vetting implementations remain separate.
Method, thresholds and execution records: `../../docs/mk-catalogue-flags.md`.
