# Research tracks

Research outside the main line, kept because it may still yield fruit.
Nothing here is adopted. Lanes live on the server under
`/srv/kbonus/dwarfs/lanes/<lane>`; rounds under `/srv/kbonus/dwarfs/rounds`.

## Detector and input research (M/K programme)

| Lane | Question | State |
| --- | --- | --- |
| `pixel-comparison-20-20261014` | does target-protected pixel correction improve recovery against SAP, PSF and PSF Flat? | completed 20 new sources; faint recovery 9/480 versus SAP 12, PSF 15 and conditional Flat 47; retention and held alarms pass, superiority fails; not adopted; see `pixel-comparison-20.md` and `kbonus-extraction-comparison.md` |
| `detector-loss-20261013` | why detectable planets are lost; knot spacing, consistency recalibration, alias dedup, combinations, window, budget, grid | chain finished 13 October 16:37 UTC; scored in `dwarf-programme.md`; knot 0.75 d and window 2x lead, both fail one clause against knot 1.5 d; comboA and comboB unscored after one job error; the chain's design files and patches left the tree on 14 October (last in 9fb7652) |
| `lightcurve-information-a-20261012`, `lightcurve-information-b` | event-level rescue and veto of the physical gates; time-slide nulls; information in LC2 and pixels | rescue v3 passes the adoption clauses; LC2 levers small (+3–6% SNR); the 5.6–5.9x pixel-information gain on two stamps is formal only: the time-slide validation found the full-pixel formal advantage does not survive calibration (1.33x and 1.51x became 0.21–0.84x) and the PRF-weighted scalar is conservative; its empirical comparison against delivered SAP was proposed, not run |
| `retention-20261013` | cadences discarded by the QUALITY == 0 rule (12% of in-transit cadences) | retain5 round: 287 of 287 jobs done, `finish` and scoring not run; its identity binds pre-reorganisation paths |
| `representation-audit-20261012` | coverage, extraction, joint views, multiscale, response sensitivity, sampling of the SAP input | no qualified extraction gain; a declared-response joint view was an artifact |
| `sap-noise-upside-20261013`, `sap-physical-noise-20261013` | paired SAP covariance and physical-noise audit on the 172 injections | reports in the lane and branch records |
| `seed-coherence-research-20261013`, `seed-diversity-research-20261013` | physical seed budgets; alias-aware seed allocation | audits in the lanes |
| `pixel-false-alarm-research-20261013` | stationary pixel weights against false alarms | diagnostics only |
| `clock-support-contribution-20261013` | how scorer clocks support recoveries | audit v1 |
| `overnight-20261011`, `restart-20261011`, `g4-saved-peer-20261011`, generation gold lanes | Design generations 1–3 (baseline BLS, duration and local-noise search, OU precision, wavelet shared Q), their peer reviews and serialization gold | superseded by the physical and coherence line |
| `rounds/faint-*` | seed priority, global and profiled box seeds, OU covariance, SAP crowding | not adopted; SAP source flux and duration dedup were |

## Branches bundled on 13 October 2026

`/srv/kbonus/archive/git-branch-archives/20261013T094956Z/` holds a bundle of
every branch with `refs.tsv`; the branches were deleted afterwards.

- `shallow-calibration-20261010` (cbbd387): the calibrate and
  evolve_controls lineage with wavelet and covariance engines and a DR25 review.
  The two cohort builders moved into `dwarfs/cohort/`.
- `catalogue-vetting-20261012` (cb86e06): require complete catalogue vetting.
- `sap-physical-noise-20261013` (8ec018e), `seed-coherence-research-20261013`
  (3cb3dc4), `seed-diversity-research-20261013` (9cd82ba),
  `research/sap-noise-upside-20261013` (f60749e): the lane reports above.
- `research/lightcurve-information-a` (9063a52, merged) and
  `research/lightcurve-information-b` (d412be0): event rescue, time-slide
  nulls, pixel information.
- `calibration/review-publication-20261013` and `server` (38876b4, cd9d682):
  the server lineage before the merge. The Mac `main` (46bd10c, 4e3f363)
  carried a multi-process layer for the shallow detector (`--workers`), checked
  for execution equivalence in `/srv/kbonus/dwarfs/lanes/performance-20261010`;
  the survey and the research lineage bind the single-process `run.py`, so the
  layer was not kept.
- Mac work saved in `dump/reorganisation-20261013/mac-uncommitted-20261013/`:
  The shallow `run.py` with biweight-slider, detrend and search-grid
  additions, and the vetting statistical validation (`statistical_validation.py`,
  `scenario_bounds.py`, in progress on 13 October).

## Retired campaigns

Removed from git on 13 October 2026; copies in
`/srv/kbonus/archive/records/repo-docs-20261013/` and `/srv/kbonus/archive/tracks/`.

- Outburst, pulsar-neighbour, rare-object and visual-transit reviews of
  5–8 October (`tracks/brightening-subsets-20261007`, `neighbour-validation-20261007`,
  `pulsar-neighbours`, `pulsar-neighbour-interest`, `pulsar-screen`).
- `tweets/` figures, `pipeline/visual/` rendering, `video/` film workspace note.
- `GPU/` Metal port, `mcp/` server, `jepa-metal/`, `pipeline/rare/`, `up/`
  working notes: history in git only.
