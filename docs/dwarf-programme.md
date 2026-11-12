# M/K-dwarf programme

Goal: through research, develop methods of data preparation, processing and
detection that recover 40–1,000 ppm transits in the SAP light curves of faint
(G 16–18) KBonus M and K dwarfs. Vetting is out of scope; the survey's vetting
software exists. The gate before the detector runs on the M/K subset is at
least 20% recovery on a cohort of 10,000 or more sampled cases, reported with
its control burden. Code: `../dwarfs/`. Hosts: `dwarf-cohort.md`.

## Cohorts

| Cohort | Cases | Composition | Use |
| --- | --- | --- | --- |
| `development200-sap-source-flux-paired-20261012-v1` | 302 | 172 synthetic planets, 28 real Kepler planets, 102 controls (14 each of catalogue FP, native EB, instrumental artifact, original and inverted; 32 year-scrambled) | every paired round |
| `final10000-v7` | 10,000 | 6,000 synthetic planets, 1,000 real planets, 3,000 controls (200 catalogue FP, 250 native EB, 600 instrumental, 650 original, 650 inverted, 650 year-scrambled) | the gate; prepared; the search is deferred (14 October) |
| `golden1000`, `promotion1000`, `mixed1000` | 1,000 | earlier promotion sets | history |

Development inputs are SAP source flux: SAP minus the `(1 - CROWDSAP)` share of each
quarter's median, then the unchanged survey preparation. 287 of the 302
development cases prepare (all 200 planets, 87 controls). Of the 200 planets,
46 pass the nominal white-noise screen (matched-filter SNR >= 7.1 with at least
3 events) using the current input and noise estimate. This is a conditional
diagnostic, not a physical ceiling or a bound on improved extraction. On
`mixed1000` (11 October) the design of that day recovered 65
of 200 catalogue-ranked planets and 24 of 600 faint synthetic injections:
faint recall, not catalogue recall, is the programme's problem.
Builders: `../dwarfs/cohort/`.

`final10000-v7` retains the earlier mixed preparation: 9,415 full-PSF, 385 SAP
and 200 archived PSF Flat cases. It has not been searched. Adam requested a
separate local PSF Flat pilot on the same frozen host/clock population on
14 October; see `psf-flat-retrieval.md`. Its 1,000 cases do not pass the
10,000-case programme gate. The server SAP survey continues unchanged.

## Scoring

A round runs one design on every case blind. `evolution.py report` scores
recoveries against the frozen clocks and `paired_review.py` compares the round
with an incumbent. Adoption needs a net planet gain of at least one, no
protected identity lost, no searched-eligibility regression, and no increase in
active controls, total control proposals or catalogue-control proposals.

## Development-200 results, 13 October 2026

| Design | Planets | Active controls / proposals | CPU-h | Note |
| --- | --- | --- | --- | --- |
| selected: `physical_local_floor05_coherence_duration_dedup_sap_source_flux_global_fundamental` | 24 | 27 / 116 | 13.3 | incumbent |
| knot 1.5 d | 27 | 26 / 103 | 14.3 | one protected loss at SNR 7.07 |
| knot 1.0 d | 31 | 21 / 103 | 38.4 | one protected loss; sparse LU cost grows with knots |
| knot 0.75 d | 33 | 21 / 89 | | against knot 1.5 d: +6, no protected loss, no control increase, but 3 cases turn detector-ineligible; against selected: protected loss 2105767081631335552 |
| event rescue and veto v3 (`c3d5afe`) | 32 | 25 / 85 | | passes every adoption clause; owner panel pending |
| long-band periodogram null gate v4 | 24 | 23 / 75 | | |
| knot 1.5 d with consistency recalibration | 34 | 41 / 295 | | fails every control clause |
| knot 1.5 d with alias-aware seed dedup | 28 | 24 / 140 | | +1 against knot 1.5 d; total control proposals increase |
| knot 1.5 d with local windows doubled (0.5 -> 1.0 d, 1.0 -> 2.0 d) | 31 | 23 / 96 | | passes all six clauses against selected; +4 against knot 1.5 d with protected loss K01835.03 |
| knot 1.5 d with 128 refinements per duration (from 64) | 27 | 26 / 111 | | net 0 against knot 1.5 d; catalogue-control proposals increase |
| knot 1.5 d with periods to 300 d (from 100 d) | 27 | 24 / 92 | | net 0 against knot 1.5 d; control clauses pass |
| comboB: knot 1.0 d, alias dedup, rescue and veto; comboA adds consistency recalibration | unscored | | | 286 of 287 jobs searched; one case raised "Full rank fixed nuisance basis required" in the global-fundamental conditioner, so `finish` refused both rounds |
| retain5 (five QUALITY bits cleared) with knot 1.5 d | unscored | | | 287 of 287 jobs searched; `finish` and scoring not run; the identity binds pre-reorganisation paths |

Loss mechanisms (`detector-intelligence.md`): 13 ceiling planets fail the
physical-stage odd/even, harmonic and inverse tests, which are miscalibrated on
faint variable dwarfs; alias seeds of strong planets take 45% of the physical
seed quota on real multi-planet hosts; 37 of 200 scorer clocks are inconsistent
with the data span. The 13 October chain ended at 16:37 UTC: knot 0.75 d and
the doubled windows lead on planets, and each fails one clause against knot
1.5 d. Rounds live in `/srv/kbonus/dwarfs/lanes/detector-loss-20261013/rounds/`.

## 14 October 2026

Adam chose knot 0.75 d with the 0.02 d duration dropped as the survey design
(`../dwarfs/shallow/current-designs.json`) and deferred the 10,000-case gate:
the server is returned after one run over the whole M/K host sample
(`mk-survey.md`). The one-design files and source patches of the 13 October
rounds left the tree (last in commit 9fb7652). The physical-gate losses, the
full-rank failure of comboA and comboB and retain5 stay open.

## Workflow

On the server, `dwarfs/harness/round.sh prepare|launch|wait|score NAME ...`
runs a paired round from a one-design JSON (`make_design.py` and
`merge_design.py` derive one from `../dwarfs/shallow/current-designs.json`),
with the cohort, runtime and rounds under `/srv/kbonus/dwarfs`; source
variants run from a separate checkout. `dwarfs/harness/survey.sh` runs the M/K
survey (`mk-survey.md`). New cohorts and target lists come from
`dwarfs/cohort/` on the Mac against the mounted corpus.

## Next

1. Run the M/K survey (`mk-survey.md`) and catalogue its proposals.
2. Fix the physical-gate losses and the full-rank nuisance-basis failure (it
   also stops the conditioning of survey sources with sparse segments,
   `mk-survey.md`) before rerunning comboA and comboB; rebind and finish retain5.
3. Run the adopted design on `final10000-v7` when compute returns; the gate is
   20% recovery.
