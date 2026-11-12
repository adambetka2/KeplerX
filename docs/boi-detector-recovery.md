# BOI detector recovery

The 14 October 2026 local paired run searched the current 125-entry BOI
delivery on 94 hosts. Both configurations used native SAP_QDIAG source flux
with the unchanged quarter CROWDSAP correction and preparation. The source
snapshot is the running server detector at `f7e78fd`; no server job or original
survey outcome was changed.

The copied configuration changes only the maximum depth from 6,000 to
60,000 ppm and the long-band maximum period from 100 to 300 days. Both keep
the 0.5-day lower limit, physical templates, ranking, masks and gates.

| Reference group | Entries | Original recovered | Copy recovered |
| --- | ---: | ---: | ---: |
| All BOIs | 125 | 37 (29.6%) | 58 (46.4%) |
| Depth at most 1,000 ppm | 22 | 5 | 5 |
| Depth above 1,000 through 6,000 ppm | 53 | 21 | 18 |
| Depth above 6,000 through 60,000 ppm | 50 | 11 (22%) | 35 (70%) |
| Period above 100 days | 5 | 0 | 3 |

Depth groups use the original catalogue measurements, mostly PSF photometry;
every search used SAP. The median host G is 17.60 mag. Nineteen hosts are in
the current M/K survey cohort (5 M, 14 K); 75 are outside it.

Strict recovery uses the unchanged programme matcher over the full prepared
baseline: period drift, epoch offset and every reference centre must agree
within half the recorded duration plus one Kepler exposure. The scorer accepts
the displayed clock or a preserved passing proposal variant. Workers receive
no reference clocks or depths. Harmonic matches are diagnostic only.

The copy gains 28 strict recoveries and loses 7; all seven losses retain
accepted harmonic matches. Its other outcomes are 33 harmonic-only matches,
33 misses and one detector-ineligible entry. All three previously statistically
validated signals and Kepler-565 b are recovered by both configurations.
Seven reference periods below 0.5 days remain outside both searches.
BOI-2077240046296834304-1 fails `SINGULAR_FIXED_NUISANCE_BASIS` in both.

Inputs came from the read-only local corpus (57 hosts) and a read-only server
transfer (37). All seven SAP arrays equal the preserved delivery FITS arrays.
Independent replay verifies the recovered clocks, input and source hashes,
runtime identity and the two configuration changes. The snapshot's 18 tests
and four scorer checks pass; Ruff and the complexity check were run.

The [CSV catalogue](../results/boi-recovery-20261014/catalogue.csv),
[summary](../results/boi-recovery-20261014/summary.json),
[configurations](../results/boi-recovery-20261014/designs.json) and
[additional diagnostics](../results/boi-recovery-20261014/additional-diagnostics.json)
are lean records. Full inputs, masks, proposals, decisions, first failures,
source snapshot, runtime and logs remain in `dump/boi-recovery-20261014/`.

This selected-signal experiment does not measure survey completeness or false
alarm rates. Recovery does not change an earlier scientific disposition or
establish an independent planet. The widened copy is not promoted to the
running M/K survey.
