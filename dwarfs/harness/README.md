# Round harness and survey runner

Scripts that run one paired development round of the research detector on the
kbonus server, and the runner of the M/K SAP survey. The cohort checks, source
snapshot, job receipts, runtime identity, array policy and scorer all come from
`../shallow/evolution.py`; the harness only changes how jobs execute and how a
finished round is reviewed.

- `round.sh prepare|launch|wait|score NAME ...` drives a round end to end.
  `DWARFS`, `COHORT`, `PY` and `RESERVE_GIB` override the server defaults.
- `split_round.py` prepares a round, runs single-thread workers that claim jobs
  from a shared queue, finishes the round and replays single cases.
- `paired_review.py` compares a candidate round's scored cases with an
  incumbent's on the frozen cohort; `review_summary.py` prints the headline.
- `loss_ledger.py` and `physical_stage_audit.py` explain lost planets and
  physical-stage decisions with the scorer's truth. Never feed them into a design.
- `make_design.py` and `merge_design.py` derive one-design files from
  `../shallow/current-designs.json`.
- `survey.py prepare|worker|status|catalogue|reclaim` and `survey.sh` run the
  M/K survey: a bright-first queue, SAP source-flux preparation inside the
  worker from the read-only corpus, atomic per-source results under the
  `proposals` array policy, a `STOP` file and dead-claim reclaim.
  Runbook: `../../docs/mk-survey.md`.
