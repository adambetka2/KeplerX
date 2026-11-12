# KBonus survey results

The 406,547-source survey completed on 10 October 2026 at 21:50:31 UTC,
4.9978 days after launch. The final catalogue contains 147 grouped main-transit
BOIs (156 proposals, 116 sources), and 3,427 grouped BOIs across all retained
lanes (3,447 proposals, 2,907 sources). BOI records the pipeline disposition;
physical confirmation and novelty require follow-up.

## Catalogue membership and manual review

The current main Rust catalogue includes all 147 original groups, with 156
proposals on 116 sources. On October 26, 2026, the PI withdrew manual exclusions
as a catalogue-membership criterion. The [current selection](../results/boi-validation/boi-current-selection-20261026.json)
restores 33 initially excluded groups and one later delivery removal. It also
retains the 12 historical delivery entries from other lanes, for 159 entries;
159 is not the main Rust completeness denominator.

The [membership policy](../results/boi-validation/current-catalogue.json) points
to the full frozen catalogue. Manual YES/NO choices and the 125-entry delivery
remain historical review and follow-up records. They must not filter the
reported detector results or completeness assessment. No frozen proposal,
vetting outcome, measurement or validation outcome changed. Follow-up readiness
is assessed separately; a restored BOI is not thereby validated.

Original PSF_FLAT is frozen at `87bd04403228f3cd3a67e25e17dc9ade650662b7`.
PSF/SAP used `a3647447180953dcbe2f8262f263fc7c20a63006` with
`rust-transit-only-v1`, admitting main Rust TRANSIT_LIKE proposals with at least
three detected transits. Archive, native replay, eligible-signal evidence gates,
frozen bindings and final grouping checks passed.

## Files in Git

[results/kbonus-20261010/](../results/kbonus-20261010/) contains exact compressed
final catalogues, a browsable BOI proposal CSV and the complete compact outcome
ledger for all 99,505 proposals: 60,858 HELD, 35,200 DISMISSED and 3,447 BOI.
The ledger retains lane, view, source and signal identifiers, period, first
failure stage, reason and reason codes. Catalogues retain original BOI grouping,
ephemerides, recorded event units, detector classifications and evidence paths.
The lean backup index records compressed and original SHA256 hashes. Frozen
configuration, code/input bindings, source evidence hashes and verification
receipts are included. All fields are derived from completed frozen outputs;
scientific calculations and classifications are unchanged.

Decompress the catalogue `.json.gz` files and verify their original SHA256
against `lean-backup-index.json` to recover the exact original catalogue bytes.
Read `compact-signals.csv.gz` to recover all signal outcomes. Original absolute
paths remain unchanged; `path-map.json` maps the verified local export.

## Full evidence

The verified local export at `/Volumes/discPCIe/kbonus` contains 457,159 original
files and 181,491,898,238 bytes. Source evidence, fits, gate results, localization
arrays, original detailed disposition tables, logs and provenance remain there
and on the server. Original files are below `server/` with their server paths.
`preservation/files.jsonl.gz` provides per-file hashes; local verification also
checked 10,113 original gzip source hashes. Raw light curves and disposable
caches are intentionally omitted and their input identities are retained.

Git restores the catalogue and compact outcomes. Detailed numerical evidence
requires the local export or preserved server. Full evidence and archive chunks
are not uploaded to GitHub; the partial draft release was removed at Adam's
request.
