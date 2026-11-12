# Cheap shallow-BOI follow-up screen

The October 26, 2026 screen covers exactly 2,164 historically unreviewed shallow
periodic groups on 1,900 sources. It joins the frozen catalogue to the saved
October 10 shallow-search and Gaia exports. Every included source, signal, view,
period, epoch and duration matches its frozen proposal. No detector, vetting,
native-fit or probability calculation was rerun.

## Results

| Priority slice | Groups |
| --- | ---: |
| Original search S/N at least 10 | 1,157 |
| Original search S/N at least 15 | 256 |
| S/N at least 15, positive reported paired FLAME mass/radius, RUWE at most 1.4 | 77 |
| Same slice without a matching Rust or M/K clock on that source | 70 |
| Detailed saved native/source evidence available in that last slice | 8 |
| Local SAP S/N at least 10 and target-compatible saved localization in that last slice | 7 |

The seven candidates are on six hosts. All seven have no nearby low-order
DR25 TCE/KOI clock warning within 120 arcsec in the dated October 25 snapshots.
The remaining detailed case has local SAP S/N -0.42 and needs investigation of
its photometry before probability allocation. This is not a physical denial.
The other 62 members of the 70-group slice need detailed evidence recovery;
seven is not an upper limit on useful signals in this lane.

| Host | Period, days | Search S/N | Local SAP S/N |
| --- | ---: | ---: | ---: |
| KIC8573780 | 3.446824 | 56.28 | 64.48 |
| KIC10528996 | 13.744702 | 45.29 | 40.16 |
| KIC10528996 | 8.555780 | 28.03 | 27.54 |
| KIC6113801 | 0.933771 | 18.36 | 20.39 |
| KIC6368753 | 9.548870 | 16.49 | 15.40 |
| KIC9328806 | 8.193725 | 15.53 | 13.05 |
| KIC11086244 | 3.544578 | 15.24 | 21.30 |

Four of these seven have a completed historical SAP shape diagnostic. The two
KIC10528996 signals and KIC11086244 have an incomplete-event-membership shape
status; review the complete native events and masks before fitting. A historical
shape diagnostic is not an accepted V3 posterior. Two signals on one star do
not establish the independence required for multiplicity validation.

## Interpretation and limits

S/N and host properties select a cheap priority slice; they are not new
catalogue, detector or physical-validation gates. Search S/N and local SAP S/N
are different statistics. Missing host properties remain missing. All groups
retain original catalogue membership and dispositions.

The full set has at least three recorded event windows and two quarters per
group. Event windows are not individually significant transit detections.
The supplied duration is at the 9.6-hour search-grid maximum for 1,701 groups;
this warrants native shape and preparation review, not a universal duration veto.

The saved Gaia query is dated October 10. The ephemeris screen covers the 77
strong host-property cases with the October 25 NASA snapshots and reports chance
expectations. Its single nearby warning is BOI-2129405688275182336-1, associated
in clock with DR25 TCE KIC10794509 about 73.8 arcsec away. No contamination
mechanism or physical false positive is established by an association alone.
No same-star catalogue match was found in this scoped screen; this is not
literature closure or a novelty claim.

The seven candidates merit authenticated native preparation under the adopted
V3 protocol. Complete actual aperture/source authentication, full stellar scene,
exact-signal literature, joint native timing/shape/noise fits and measured
detrending transfer before admitting probability integrations. This cheap screen
establishes no fully prepared V3 case and launches no expensive calculation.

After this screen, the PI authorized internal-disc evidence storage. New original
light-curve and Gaia-scene input authentication for the seven candidates on six
hosts runs under `other/boi-v3-20261025/shallow-20261026/`; the
[authorization](../results/boi-validation/internal-evidence-storage-20261026.json)
records the location. This input preflight does not admit a probability run.
The [completed input preflight](../results/boi-validation/shallow-native-input-preflight-20261026.json)
authenticates all six original MAST products against their frozen bytes and
FITS source/checksum records, and preserves native SAP arrays and fresh Gaia
scene-coordinate queries internally. Actual apertures, physical scene properties,
literature and accepted native fits remain open.

## Records

- [Summary](../results/boi-validation/shallow-cheap-screen-20261026/summary.json)
- [Seven-candidate shortlist](../results/boi-validation/shallow-cheap-screen-20261026/shortlist.csv)
- [All 2,164 groups](../results/boi-validation/shallow-cheap-screen-20261026/all-2164.csv)
- [77-group priority slice](../results/boi-validation/shallow-cheap-screen-20261026/strong-host-properties-77.csv)
- [Measurements and provenance](../results/boi-validation/shallow-cheap-screen-20261026/screen.json)
- [Catalogue screen](../results/boi-validation/shallow-cheap-screen-20261026/catalogue-screen.json)
- [Native-preparation handoff](../results/boi-validation/shallow-cheap-screen-20261026/followup-readiness.json)
