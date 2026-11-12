# KeplerX: the study's time on the M1 Pro (3 November 2026)

3 November 2026. Task (Adam, 3 November, TODO item 6): shorten the
full study, "even by 20-30%", without losing a real planet. Changes:
`KeplerX/detector/compiled_search.py` (dispatch), `gpu_screening.py`,
`gpu_screening.metal`, `gpu_screening_host.m` and `config.json`
(`vetting.dismissed_metrics`). Evidence:
`KeplerX/results/performance-engineering/2026-11-03-study-time/`.

## Result

On a 300-curve mix of the study's three kinds of search, through the
production batch runner on 8 workers, the study's work took **375 s
instead of 451 s (-17%, 1.20 times the throughput)**. Every detection,
search mask, vetting disposition, fired rule and binary-evidence record is
identical to the old code's on all 300 curves. The planned study (PSF Flat
everywhere, SAP where PSF Flat covers under 90% and on SAP-only sources,
10.8 days by TODO item 6's estimate) falls to about **9.0 days** on this
Mac as it is. The 4-day target is not reached. The Mac is swapping while
it runs (below), so a restart before the study may be worth more again.

| Leg (8 workers unless noted) | Wall (s) | Detect worker-s | Detect CPU-s | Vet worker-s | Vet CPU-s | GPU busy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline, 2dd55f6cd | 452 | 1,965 | 558 | 1,464 | 1,289 | |
| baseline again (drift check) | 450 | 1,956 | 553 | 1,445 | 1,281 | 91% |
| baseline on 10 workers | 467 | 2,494 | 616 | 1,886 | 1,421 | 89% |
| policy-first vetting | 405 | 2,239 | 539 | 825 | 733 | 93% |
| + GPU screen changes | 382 | 2,033 | 544 | 853 | 765 | 91% |
| + three GPU slots (adopted) | **375** | 1,964 | 544 | 862 | 770 | 93% |
| + CPU folding of the longest periods, unlimited | 702 | 4,164 | 3,395 | 1,244 | 1,003 | 85% |
| + CPU folding, at most two workers at once | 446 | 2,303 | 1,366 | 1,066 | 925 | 90% |
| adopted, 10 workers | 372 | 2,568 | 570 | 933 | 810 | 92% |

GPU busy is the mean of `ioreg`'s device utilization, sampled every 2 s.

## The benchmark

`mix-300.jsonl`: 150 PSF Flat curves drawn by weight from the survey
benchmark's diverse set, 75 SAP curves of PSF Flat sources and 75 SAP-only
curves (both prepared from LC2 in the batch), run as
`run_leg.sh NAME CONFIG WORKERS [ENV...]`, which records the commit, the
uncommitted Detector and Vetter diff, the wall time, GPU utilization and
free memory. In the baseline, workers spent 3,475 s: detection 1,965 s
(558 CPU-s, the rest waiting on the GPU), vetting 1,464 s (1,289 CPU-s,
42% of worker time), preparation 46 s.

## What changed

- **Vetting: policy first** (`vetting.dismissed_metrics: "policy"`, an
  option since 2 November). A signal the policy already dismisses skips
  upstream's odd/even trapezoid fits and local morphology; the binary gate
  reads its own odd/even box and binary evidence, which still run.
  Re-vetting 40 of the baseline's cases (158 signals, 152 dismissed) from
  their stored detections: every disposition, fired rule, binary evidence
  and shared metric identical, vetting CPU 314.7 against 188.8 s (-40%).
  In the batch, vetting worker time fell 43%. Dismissed signals no longer
  carry those two metrics, so a later policy that reads them cannot be
  replayed on the study's dismissed signals.
- **Window fold, densely.** Each SIMD group now looks up 32 epochs'
  cadence ranges at once, and the tile's threads take its cadences back to
  back (a SIMD prefix sum and a shuffle search). Before, each SIMD group
  took one epoch slot after three dependent table loads and often filled
  half its lanes. The cadences visited, the integers added and the flags
  set are the same.
- **Window histograms sized to their tiles.** Threadgroup memory is set per
  class instead of a fixed 15 KB, and tiles from 3,072 bins hold 128 bins
  (was 64): half the tiles and partials at long periods.
- **The longest periods in blocks.** The longest-period chunk (16,384
  periods) goes to the GPU as eight dispatches of 2,048 periods, whose
  partial buffers are tens of MB instead of up to 595 MB per dispatch.
- **Three GPU slots** (`KEPLERX_GPU_SLOTS`, default now 3): a third
  command buffer fills the gaps while the other holders copy buffers.

## Exactness

- GPU screen: the first search pass of 12 survey curves (83 chunks),
  screened three times by each variant in alternating rounds. Cells, flags
  and resolved folds bitwise identical. Device time 7.17 against
  6.55-6.57 s (-8.5%, two rounds), dispatch wall 8.05 against 7.10-7.16 s;
  the dense loop alone -4.6%, per-class histograms alone -0.8%.
- Dispatch in blocks: the first pass of 8 curves, fold results bitwise
  identical (also with 24,576 of the longest periods folded on the CPU
  instead); device time 4.74
  against 4.48 s, pass wall 16.1 against 10.8 s.
- Batch: all 300 curves identical in every leg above (detections without
  their clocks, search masks, dispositions, fired rules, binary evidence).
- The golden and end-to-end tests pass (35, the frozen-runtime validator
  test skipped as usual), among them the GPU-screened search against the
  CPU search bit for bit on four quarters.

## Rejected

- **Folding the longest periods on the CPU while every GPU slot is held**
  (exact: bitwise identical with folding forced). Unlimited, it moved 1.25
  million periods to the CPU: GPU seconds -36%, but detection CPU rose from
  558 to 3,395 s, the Mac saturated and the mix took 702 s. Limited to two
  workers at once: 446 s. A CPU fold costs about 14-18 CPU-s per GPU-s
  saved, and this Mac has no such spare CPU.
- **Ten workers:** 467 s on the old code and 372 s on the adopted one
  (8 workers: 375 s); GPU busy stays near 92% and the Mac swaps more
  (7.4 GB), so the study stays on 8.
- Timing-only variants on the same inputs: removing every duration's
  atomic adds saves 10% of the long-period chunk, the whole per-tile
  reduction 28%, the merge 1% and threadgroup launches 5%; 96-bin tiles
  nothing. The long-period tiles are latency-bound
  ([window fold](keplerX-2026-11-01-gpu-window-fold.md)), and no remaining
  piece of the screen offers more than a few per cent exactly.

## The Mac

During every leg the Mac swapped: fseventsd holds a 10 GB footprint and a
full core (the growth seen when millions of files were renamed on the
archive disc on 1 November), kernel_task took 170-180% CPU, system time
20-40% of the CPU, swap 3-9 GB and free memory fell to 16-33 MB. An
open desktop app's renderer took about one core while it was active, and
with no KeplerX job running WindowServer and the app's GPU process kept
the GPU partly busy; the same fixed screen ran 20% slower within half an
hour. Restarting fseventsd
(`sudo killall fseventsd`, launchd restarts it) or rebooting before the
study frees that memory and lowers the risk of the 3 November
memory-reclaim kill (TODO item 5), which needs critical pressure; keeping
other apps idle while it runs leaves the GPU and a core to the study.
