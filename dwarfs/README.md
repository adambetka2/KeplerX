# M/K-dwarf programme

Research code for recovering 40–1,000 ppm transits in the SAP light curves of
faint (G 16–18) KBonus M and K dwarfs. The programme develops data preparation,
processing and detection; vetting belongs to the survey system in `../pipeline/`.
Status, cohorts, round results and the gate: `../docs/dwarf-programme.md`.

- `shallow/`: the research detector lineage. It extends the survey's shallow
  BLS detector (`../pipeline/shallow/run.py`) with local-noise, coherence,
  physical-template and seed designs, and scores a round blind against the
  frozen cohort clocks.
- `harness/`: scripts that run one paired development round on the kbonus
  server and review it against an incumbent, and the runner of the M/K SAP
  survey (`../docs/mk-survey.md`).
- `cohort/`: the programs that froze the development and gate cohorts:
  physical injections into native M/K-dwarf curves and parent-disjoint controls.
- `requirements.txt`: the pinned runtime shared by the three folders.

Every detector change is a paired round on the frozen development cohort before
it runs at scale. Cohorts, rounds and surveys live on the server under `/srv/kbonus/dwarfs`.
