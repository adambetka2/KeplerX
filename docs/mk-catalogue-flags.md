# Annotate the completed M/K vetting catalogue

Adam authorized this pass on 19 October 2026. Start only after the full
9,182-source, 17,940-proposal vetting run finishes. Preserve its measurements,
periods, hypotheses, first failures and dispositions.

1. For final BOI proposals only, flag commensurate periods from the same Gaia
   source and export the saved known-signal comparisons.
2. For all remaining proposals, including HELD and DISMISSED, export catalogue
   comparisons only. Do not perform harmonic or alias tests on these proposals.

## Period flags

Compare every BOI pair from the same Gaia source. Flag the closest reduced
ratio `m/n`, with numerator and denominator at most 15, when
`abs((P_long / P_short) / (m/n) - 1) <= 0.001`. The 0.1% tolerance matches the
frozen catalogue matcher's period screening threshold. Preserve the measured
ratio, mismatch and both signal identities in the output.

These are possible aliases or harmonics. A resonance can also satisfy this
screen. No transit-timing agreement, signal subtraction, residual search or
physical independence is established. Equal periods remain possible duplicates.
Never merge proposals or infer a number of independent planets from these flags.
An extraction aperture's KIC is not a Gaia source identity.

## Known signals and pipeline performance

Reuse complete original complete-catalogue-common-recurrence-v1 comparisons, with
their frozen 12 October catalogue snapshot, positions, original proposal
timing hypotheses and observed transit centres. These cover cumulative KOIs
(including false positives), DR25 TCEs, confirmed planets, Villanova Kepler EBs
and Bienias–Szabó 2025. Verify the snapshot and every completed batch binding.
Read each source evidence file once per stage. For missing or incomplete comparisons,
run the same frozen matcher on the original signal hypotheses and observed
centres. This includes signals that left vetting before catalogue measurement.
Record reused and newly computed comparisons separately. No FITS reads, network
requests or transit fits are needed.

Keep catalogue classifications and host relations in separate columns. Count
host-compatible CONFIRMED associations by original disposition; nearby confirmed
planets do not count as host recoveries. Export incomplete ephemeris coverage
explicitly. Associations remain candidate recovery evidence, without a new
physical attribution or confirmation claim.

The performance denominator is the detected proposal cohort. Overall detection
recall would require checking every comparable known planet in the searched
stellar cohort, including stars with no proposal; this pass does not measure it.

## Run and outputs

Run `uv run python pipeline/catalogue/annotate.py --run RUN --output OUTPUT`.
Use a new output directory outside the active vetting run. Four source readers
operate after vetting finishes. The BOI stage completes before the remaining
catalogue-only stage starts. Sources shared by both stages are read twice;
the second read verifies that their evidence did not change.

Each stage writes `signals.csv`, `catalogue-matches.csv`,
`possible-harmonic-alias-pairs.csv`, `catalogue-checks.jsonl` and `summary.json`.
The final summary records input hashes, output hashes, scope and counts.
Missing scientific inputs or changed bound inputs stop execution with a clear
failure. An existing output directory is never overwritten.

## Queued execution

The server job was queued at 15:03 UTC on 19 October in
`/srv/kbonus/dump/mk-catalogue-flags-20261019-v1/`. Its `wait_and_run.py`
waits for the full vetting run's completion and verifies the prepared code
hashes before starting the two stages. It writes only in this new directory.
`queue.json` records waiting, execution, completion or the first failure.
Results go in `final/`; execution logs and readiness evidence stay alongside it.

Six tests cover the supplied rounded P, P/3 and P/7 example, source identity,
BOI-only flagging, full outcome preservation, changed-input rejection,
incomplete coverage and catalogue computation for an early dismissed signal.
Ruff and the complexity limit of 10 pass. The queued receipt is
`../results/mk-catalogue-flags-20261019.json`.
