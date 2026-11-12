# A Detector worker killed by macOS memory reclamation (3 November 2026)

The task (Adam, 3 November): find why the kernel killed a Detector worker
(`vm_reclaim: Killing due to virtual-memory guard`, 02:09 UTC) in the PSF
Flat side of the lane-pair calibration run, 4 workers beside SAP's 8. With
`stop_on_error` one death stops a batch, and the survey runs for days.
Evidence, probes and re-runs:
`KeplerX/results/software-checks/2026-11-03-worker-killed-by-memory-reclaim/`.
Machine: M1 Pro, macOS 26.5.1, kernel xnu-12377.121.6, whose source Apple
publishes (`osfmk/vm/vm_reclaim.c`, `bsd/kern/kern_memorystatus.c`,
libsyscall `vm_reclaim.c`, libmalloc `xzone_malloc`).

## Findings

- **Not out of memory, and not our code.** memorystatus never chose the
  worker. A consistency check in macOS's deferred memory reclamation killed
  it, after a kernel fault left that check's bookkeeping wrong. KeplerX
  unmaps nothing itself, and the GPU screen copies its inputs into Metal
  buffers; opening the Metal device adds no reclamation state.
- **The mechanism.** macOS 26's allocator (xzone malloc) caches freed large
  blocks and queues every freed span in a ring shared with the kernel; the
  kernel unmaps queued ranges when memory runs short. Under critical
  pressure memorystatus drains every process's ring with page faults
  disabled (`vm_deferred_reclamation_gc(RECLAIM_GC_DRAIN, …,
  RECLAIM_NO_FAULT | RECLAIM_NO_WAIT)`). `reclaim_chunk` marks a chunk of
  entries busy, unmaps them, then writes the ring's new head; with faults
  disabled a failed write is ignored. If the head write fails after the
  unmaps, the ring still starts at a range the kernel has already unmapped.
  Every later pass begins there, finds the range gone, logs the guard line
  (a deallocation gap, not fatal for a third-party process) and stops
  without advancing. When the process later maps new memory into that hole,
  the next pass unmaps it silently and counts its bytes a second time, and
  the ring's accounting check then kills the process.
- **What the log shows.** The batch started at 01:42 UTC. At 02:01:06 the
  first critical-pressure purge since then (2.7 GB from all processes)
  logged nothing for worker 4874. From 02:01:16 the kernel logged
  `[4874] Killing due to virtual-memory guard at (0x72e400000,
  0x72eff8000)` 250 times, about once a second, the 02:05:37 purge
  included, while the worker kept finishing cases. At 02:09:24 it started a
  new case; the lines stopped; at 02:09:29: `more bytes have been reclaimed
  (681295872) than are supposedly in buffer (636190720)`, then
  `Force-exiting python3.11 [4874]`, EXC_GUARD of type virtual memory,
  flavour 9 (`kGUARD_EXC_RECLAIM_ACCOUNTING_FAILURE`). The broken pool
  failed all 4 cases in flight and the batch stopped after 126 of 924.
- **The stuck range was one of the worker's freed arrays.** In the
  pipeline's interpreter a freed 12,550,144-byte NumPy array becomes a ring
  entry (start, start + 0xbf8000) on a 4 MB boundary, the shape and size of
  the stuck range.
- **Reproduced downstream of the failed write** (`probes/stale_entry.py`):
  with an unmapped range at the head, each drain logs the same guard line,
  leaves the head where it was, and the process lives; once the process
  maps new memory into the hole, the next drain unmaps it without a line. The failed head write itself cannot be observed on a
  release kernel. It is inferred from the code, from the purge that
  preceded the first line by 10 s, and from the double count the accounting
  kill needs; no other path leaves an unmapped range at the head.
- **No result was harmed.** From the production commit (source digest
  0c03e692) the 14 cases worker 4874 finished, several while its ring was
  stuck, and its 4 in-flight cases reproduce exactly: candidates, vetting
  metrics and dispositions, BOIs, search masks and per-signal records.
  Only run records, logs, the GPU clock and file receipts differ.
- **The rest of the night's log is benign.** 336 `Failed to free(reusable)`
  lines came from short-lived zsh, rsync and DockHelper processes exiting
  during a pass. Worker 4874 is the only process this code tried to kill in
  the 11.5 hours the unified log kept.

## Protections and their cost

1. **`MallocLargeCache=0`** (read by release libmalloc): no large-block
   cache, no queued spans, no ring, so the kernel's path cannot reach the
   process. A ProcessPoolExecutor worker inherits it, and the Detector
   process has no ring after opening Metal. Outputs are identical with and
   without it (18 cases three ways, 8 cases in two more pairs). The cost,
   from two interleaved pairs on the shared machine under load: user CPU
   unchanged; system CPU ×2.1 in detection and ×3.0 in vetting (+1.5 and
   +1.7 s per case); wall time +30% in detection (+7.3 s per case) and +18%
   in vetting (+3.2 s). Every large array is then fresh zero-filled memory,
   which waits for pages when memory is short.
2. **Pinning the ring's first page** (`mlock` of 16 KB, found with the
   exported debugging call `mach_vm_reclaim_get_rings_for_task`): the head,
   busy and tail indices stay resident and writable, so the purge's head
   write cannot fail. The kernel reserves the ring's full size at creation
   and resizes it in place, so the pin lasts. No runtime cost. Untested
   against the trigger itself, which needs critical pressure on the shared
   machine, and it relies on a private call.

Neither is applied. Recommendation: pin the page in every KeplerX process
(the batch, its workers, single stages), and switch to `MallocLargeCache=0`
if the guard line ever appears again. A worker death still stops a batch;
running the in-flight cases again once in a fresh pool would keep a survey
going through any worker death, this one or memorystatus's.

Launches outside `run.sh` (`uv run python -m calibration.…`) get neither
protection unless they take it themselves.

## For Apple (Feedback Assistant, not filed)

xnu-12377.121.6, `osfmk/vm/vm_reclaim.c`, `reclaim_chunk`: when
`reclaim_copyout_head` fails with EFAULT under `RECLAIM_NO_FAULT` (the
memorystatus drain), the entries just removed stay between head and busy.
The next pass removes them again: a gap gives `kGUARD_EXC_DEALLOC_GAP`
(non-fatal for most processes) and the ring never advances; a range the
process has mapped again in the meantime is removed silently, and the double
count ends in `kGUARD_EXC_RECLAIM_ACCOUNTING_FAILURE`. Observed on macOS
26.5.1 (25F80) with python3.11 under critical memory pressure; the stale
state reproduces with an entry the process unmaps itself.
