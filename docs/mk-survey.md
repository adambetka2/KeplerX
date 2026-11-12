# M/K SAP survey

The first run of the programme's detector over the whole M- and K-dwarf host
sample (`dwarf-cohort.md`): 57,889 sources in SAP source flux, searched with
the one design of `../dwarfs/shallow/current-designs.json`. Adam set the
settings below on 14 October 2026 and deferred the 10,000-case gate: the server
goes back after this run, so the survey runs now with the best paired-round
design. Runner: `../dwarfs/harness/survey.py` and `survey.sh`.

## Decisions

| Item | Setting |
| --- | --- |
| Design | `physical_local_floor05_coherence_duration_dedup_sap_source_flux_global_fundamental_knot075_drop002`: knot spacing 0.75 d; the 0.02 d duration removed from both coarse grids (0.80 of the full-grid CPU, identical proposals on three development cases) |
| Input | SAP source flux prepared by the worker from the read-only corpus: SAP minus the `(1 - CROWDSAP)` share of the quarter median, CROWDSAP from the `LIGHTCURVE_Qn` header, then the unchanged survey preparation; bit-identical to the development cohort inputs on three crowded sources |
| Targets | `../results/mk-survey-targets-20261014.csv`: 57,889 hosts (27,454 M, 30,435 K); 6,239 skipped as hopeless; 51,650 queued |
| Hopeless | the white-noise matched-filter SNR of a 1,000 ppm transit stays below 7.1 at 3 h and at 1 h over every quarter, with the SAP noise median of the source's G bin (`../results/mk-survey-targets-20261014.json`) |
| Order | bright first: `phot_g_mean_mag` ascending, then `source_id` |
| Evidence | array policy `proposals`: one compact JSON per source (decisions and scalars, 1–2 MB) and an npz only for sources with proposals (prepared records, both physical bands' time, prepared flux, nuisance residual, noise and quarter, and the period, SNR and depth periodograms of the bands that seeded a proposal; 2–4 MB); full arrays remain a calibration-round setting |
| Workers | 32 at nice 5, unpinned, since 06:10 UTC on 14 October (ramp on the idle host: 24 workers searched 16.9 M cadences per hour, 32 workers 17.4 M); 8 on E-cores 24–31 before that, beside the BOI validation |
| Interruption | a result is written atomically after its npz; `STOP` lets workers finish their source; a relaunch continues the queue |

## Run

On the server, from the `server` branch checkout:

```bash
cd /home/adam/spacelab
date -u; pgrep -fa 'statistical_validation|split_round|survey.py'; df -h /srv/kbonus; free -g
dwarfs/harness/survey.sh prepare mk-sap-20261014-v1 results/mk-survey-targets-20261014.csv
dwarfs/harness/survey.sh launch mk-sap-20261014-v1 0 32
dwarfs/harness/survey.sh status mk-sap-20261014-v1
```

`prepare` refuses an existing folder, checks the free space, snapshots the
source and binds the preparation files; `launch NAME FIRST COUNT` starts
workers `FIRST..FIRST+COUNT-1`, each with its own numba cache and `TMPDIR`; it
does not refuse names that already run, so read `status` (`live_workers`) first.
Everything lives in `/srv/kbonus/dwarfs/surveys/mk-sap-20261014-v1/`:
`identity.json`, `queue.json`, `designs.json`, `source/`,
`results/<source_id>.json|.npz`, `summaries/`, `claims/`, `workers/`, `logs/`.
A worker prints one JSON line per source into `logs/worker-NN.log`.

Isolation. The survey writes only inside its folder and reads the corpus. It
never touches `/srv/kbonus/survey` (sealed), `/srv/kbonus/dwarfs/rounds` and
`lanes/` (removal pending Adam's authorization) or `/srv/kbonus/dump/boi-validation-*`
(the BOI statistical validation of 13–14 October, stopped by Adam at 05:31 UTC
on 14 October after 13 of 160 runs). The host's package power is capped at
65 W (RAPL long-term limit), so idle threads are not free capacity, and nice
cannot protect a job on another logical CPU. Beside that validation, 16
unpinned workers slowed its scenarios 1.9–2.3x (00:13–00:48 UTC), so 8 workers
ran pinned to the E-cores it left free (`taskset -c 24-31 dwarfs/harness/survey.sh
launch mk-sap-20261014-v1 0 8`, 00:53–05:32 UTC): its scenarios took
0.86–1.21x their idle time and the survey completed 78–97 sources per hour
(3.3 M cadences per hour at 04:30–05:30 UTC).

Workers. Measured on 14 October on the otherwise idle host by cadence rows
searched per hour, each source weighted by the overlap of its search span with
the window (`/srv/kbonus/dump/mk-survey-ramp-20261014/measure.py`; windows read
20 minutes after they closed, no source still running): 24 unpinned workers
(05:50–06:10 UTC) searched 16.9 M rows and 501 sources per hour at a median
4.7 CPU-s per 1,000 rows; 32 workers (06:15–06:40 UTC) 17.4 M rows and 502
sources per hour at 6.4 CPU-s. The hyperthreads add 3% under the cap, and the
rule set before the data (keep 32 unless 24 wins on both rows per hour and
per-worker speed) keeps 32. 16 workers were not measured on the idle host: the
16-worker figure of 8.2 M rows per hour was taken beside the validation, and
under the cap more cores at lower clocks beat fewer cores at higher ones for
this compute-bound search.

Stopping and resuming. `stop` writes `STOP`; workers exit after their current
source, and `launch` continues the queue. After a kill or a crash, `reclaim`
removes the claims of dead workers (`reclaim --failed` also clears
`UNEXPECTED_ERROR` results for a retry). `catalogue` writes `sources.csv`,
`proposals.csv` and `catalogue.json` at any time; `status` reports the
completed count, the reached G, the statuses, proposals, the 20- and 60-minute
throughput, the projected hours, live workers, stale claims, sizes and load.

## Budget

| Quantity | Value |
| --- | --- |
| Search cost | 2.5 CPU-s per 1,000 cadences for one worker on an idle host; 4.7 with 24 workers and 6.4 with 32 under the 65 W cap (14 October) |
| Wall time | about 4 days from 07:00 UTC on 14 October at 32 workers: 50,291 sources and 517,000 quarters left (1.65 G usable cadences at the measured median of 3,200 per quarter) at 17.4 M cadences per hour, 95 h by cadences and 100 h by sources, if the per-cadence cost holds for the fainter remainder (G 15.7–19.0, median 17.3) |
| Storage | 2.0 MB per source measured over the first 1,341 (1.27 GiB of results, 1.46 GiB of evidence npz for 314 sources with proposals), about 105 GB for the queue; a failed source keeps its partial arrays, about 133 MB; 814 GiB free at 07:00 UTC on 14 October; keep 20 GiB free |

## Record and limits

- Proposals are detector dispositions, not planets. Vetting is the survey's,
  outside the programme.
- The 10,000-case gate (20% recovery) was not run: deferred by Adam on
  14 October. The design's evidence is the development-200 round: 33 of 200
  planets, 21 active controls and 89 control proposals against 24, 27 and 116
  for the selected design; one protected loss (2105767081631335552) against it
  and three cases that turn detector-ineligible against knot 1.5 d
  (`dwarf-programme.md`).
- The physical-stage odd/even, harmonic and inverse gates were not fixed. Knot
  0.75 d reduces their losses (13 ceiling planets in the 13 October analysis,
  5 at knot 1.0 d); demoting them to report-only or recalibrating the inverse
  test stays untested.
- The band drop was checked for identical proposals on three cases only.
- The hopeless skip is a white-noise bound at 1,000 ppm, recorded with its
  noise medians in the targets JSON.
- `identity.json` binds the design, the source snapshot, the preparation files
  (`lc2.py`, `run.py`, `signal_contracts.py`, the imported `evolution.py` and
  `survey.py`) and the numerical runtime by hash; a worker refuses a changed
  file. `mk-sap-20261014-v1` started from f7e78fd, which bound `evolution.py`
  only as its snapshot copy while workers import the checkout copy; the run
  stays valid while `/home/adam/spacelab` stays at f7e78fd, so the checkout is
  not updated before the run has ended.
- Sources with a segment of fewer distinct cadences than spline columns are
  lost: 3–4 isolated cadences spanning 0.76–1.12 d, because `spline_basis` caps
  the degree by the cadence count but not the 0.75 d interior knots. When the
  search's sparse LU detects the singular block the source is
  `DETECTOR_INELIGIBLE` (`SINGULAR_FIXED_NUISANCE_BASIS`); when it does not, a
  proposal-bearing source fails in the global-fundamental conditioning
  (`UNEXPECTED_ERROR`, `Full rank fixed nuisance basis required`). Of the first
  287 sources: 4 and 1, plus 1 searched with an undetected singular block.
  Capping the columns at the cadence count recovered 4 of the 5, the lost
  proposal unchanged; the fifth (457 segments over 5,738 cadences) then failed
  the profiled-energy check. Report:
  `/srv/kbonus/dump/mk-survey-rank-20261014/report.json`. The affected sources
  are rerun in a derived generation once Adam decides on the fix.
- A second sparse-data rule drops whole sources: `whiten()` in
  `../pipeline/shallow/run.py` takes each quarter's scatter as 1.4826 times the
  median absolute residual after the 0.5 d biweight, and a quarter whose median
  residual is exactly zero (the trend passes through a handful of cadences)
  makes the source `DETECTOR_INELIGIBLE` (`NONPOSITIVE_QUARTER_SCATTER`). A
  designed rule, not a defect; dropping only the degenerate quarter would keep
  the source. At 11:09 UTC on 14 October, 3,747 sources were done to G 16.10
  with 1,675 proposals: 49 lost (1.3%: 32 singular basis, 8 full-rank and 5
  profiled-energy errors, 4 quarter scatter at G 16.05–16.10 with 1,100–4,300
  usable cadences) and 18 input-ineligible for too few cadences. The loss is
  0.6–1.4% in every G bin with more than 300 sources.
- `launch` does not refuse names that already run. On 14 October a separate
  capacity controller launched workers 8..15 at 05:32:43 UTC and this runbook's
  operator launched the same names 24 seconds later; the duplicate set was
  terminated after 9 minutes, its 8 claims reclaimed and the group relaunched
  as 16..23 (`logs/launch.log`). Read `status` before every `launch`.
- At 07:00 UTC on 14 October, 1,341 sources were done to G 15.72: 1,316
  searched, 9 input-ineligible, 11 detector-ineligible, 5 unexpected errors;
  314 sources carry 711 proposals.
- Verified on the Mac on 14 October under Python 3.11: five bright K dwarfs end
  to end, two with proposals (the known BOI of 2079714428492352768 recovered at
  3.447 d), with the compact results, the evidence npz, `status`, `catalogue`
  and `reclaim`; the shallow tests pass.

## Dismissed-source follow-up

Adam authorized `mk-sap-dismissed-20261014-v1` on 14 October: search exactly
the 6,239 sources excluded by the 1,000 ppm pre-search cut (6,169 M and 70 K).
Their recorded white-noise SNR of 4.8–6.9 becomes 24–34.5 at 5,000 ppm under
the same period and duration assumptions; this estimate does not establish
detector recovery. The follow-up uses the parent design, including its
6,000 ppm depth ceiling, and the same SAP preparation and numerical runtime.

Inputs live in `/srv/kbonus/dwarfs/inputs/mk-sap-dismissed-20261014-v1/`.
`targets.csv` preserves every original field except the explicit queue override
`hopeless=0`, retaining `original_hopeless=1`. `selection.json` records the
authorization, hashes and exact-subset checks. The prepared queue contains all
6,239 sources, skips zero and has no overlap with the parent's 51,650 sources.

The enabled user service `mk-sap-dismissed-followup-20261014.service` polls
once per minute and launches 32 workers only after every parent source has a
result and all parent workers have exited. It respects either run's `STOP`
file and records a launch before starting workers to prevent duplicates.
The controller, bound `plan.json`, `state.json`, `launch.json` after launch
and passing `verification.json` live in
`/srv/kbonus/dump/mk-sap-dismissed-followup-20261014/`.
Keep the bound checkout files and runtime unchanged until both runs finish.
Use `dwarfs/harness/survey.sh status mk-sap-dismissed-20261014-v1` for results.

## Vetting

Both searches completed on 18 October: the parent at 03:12 UTC and the
dismissed-source follow-up at 03:49 UTC. All 57,889 sources have an outcome;
56,786 were searched, 161 were input-ineligible, 740 detector-ineligible and
202 raised numerical errors. The parent supplied 17,747 proposals from 9,051
sources; the follow-up supplied 193 proposals from 131 sources.

Adam authorized vetting all 17,940 proposals on 19 October. The run is
`/srv/kbonus/dump/mk-sap-vetting-20261019-v1/`. Its `adapter.py` bridges the
two completed searches into a private snapshot of the existing pipeline at
`f7e78fd`, with a separate Python 3.11 runtime. The original survey evidence,
detector outputs and server checkout stay unchanged.

The two-source preflight completed at 05:22:39 UTC, preserving all 465 supplied
events. The full run started immediately afterwards with batch 1 of 144.
`../results/mk-sap-vetting-20261019.json` records that launch and its bindings;
its status is a dated snapshot, not a live monitor.

Before supplying events, preparation reproduces every proposal source's
saved SAP source-flux job and prepared records, including byte equality with
the saved detector npz. Each original proposal keeps its period, epoch,
duration and observed event windows under a unique run-scoped identity.
Vetting measures native delivered SAP and diagnostic arrays; only `sap_qdiag`
is declared searched. As in the existing SAP pipeline, FLATTEN_MASK applies
only to PSF Flat and PSF Nova. The five gates, fits, thresholds and first-failure
policy are unchanged.

After this full run completes, a separately queued catalogue pass will flag
possible period aliases/harmonics among final BOIs only, then complete catalogue
checks for every other proposal, including HELD and DISMISSED. It preserves all
original dispositions. See `mk-catalogue-flags.md` for scope and execution.

The complete 406,547-source corpus supplies 60 arcsec neighbour cones.
One frozen shared-event index covers all 4,645,851 supplied event windows
from both searches across all batches.
The controller requires an end-to-end preflight on the known 3.447 d BOI
source 2079714428492352768 and a follow-up M dwarf before starting 144 batches
of at most 64 sources, with 24 source workers and at most eight network workers.
Pixel and PRF caches are private copies, and each batch requires 20 GiB free.
An execution failure preserves its traceback and stops the controller.

Read `progress.json` and `execution.log` for the current stage and batch.
`inventory.json` and `source-ledger.jsonl` account for every original source;
`exports.json` binds every proposal-source handoff. `verification.json` records
the completed preflight. Batch plans, logs, evidence and ledgers live under
`work/batches/`; final `catalogue.json` references every completed batch.
BOI remains a pipeline designation, and the deferred recovery gate remains open.

The unchanged vetting test folder has 317 passing tests and two failing alias
fixtures. Both omit the catalogue evidence required before contamination
admission; adding a complete, unmatched catalogue result to those fixtures
reproduces their expected behavior. The original failures and the fixture-only
diagnostic remain under `setup/`; no scientific gate was changed.
