# M- and K-dwarf hosts

The programme's host selections are photometric preselections from the
406,547 KBonus background sources. They do not establish spectral types,
completeness, planet occurrence or habitable-zone planets.

## M dwarfs

`pipeline/photometry/select_m_dwarfs.py` indexes the published KBonus M-dwarf
catalogue (`hlsp_kbonus-bkg_kepler_kepler_mstars-catalog_kepler_v1.0_cat.fits`,
hash-verified) against the LC2 corpus. The snapshot
`dump/m-dwarfs-20261005/selection/m-dwarfs.sqlite` holds every match with its
shard path; its `selection` view contains 27,454 sources.

## K dwarfs

`pipeline/photometry/select_k_dwarfs.py` builds the 6 October 2026 snapshot in
`dump/k-dwarfs-20261006/selection/`: `k-dwarfs.parquet` (selected candidates),
`k-dwarfs.sqlite` (all background sources with Gaia parameters, views and the
first failed criterion), `lc2/` links to the original shards, `manifest.json`
and `verification.json`. The `selection` view contains 30,435 sources; the
populations do not overlap.

`candidates` requires `3900 <= teff_gspphot < 5300 K` and
`logg_gspphot >= 4.2 dex`. `selection` also requires no membership in the
M-dwarf list, `parallax / parallax_error >= 10`, `RUWE <= 1.4`, and an
extinction-corrected offset from the
[Mamajek dwarf sequence](https://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt)
between -1.0 and +1.5 mag. The offset is
`G + 5*log10(parallax_mas) - 10 - A_G` minus the sequence's absolute G at
`BP-RP-E(BP-RP)`, with Gaia DR2 colours and raw DR3 parallaxes.

| View | Additional condition |
| --- | --- |
| `near_4300` | `4100 <= Teff < 4500 K` |
| `late_k` | `3900 <= Teff < 4500 K` |
| `bright` | `G < 16` |
| `priority` | `near_4300` and `G < 16` |
| `needs_review` | membership, astrometry or CMD issues |
| `classified` | every background source with its first failed criterion |

Gaia matched every cohort identity; 47,198 sources lack GSP-Phot temperatures.
The [4300 K study](https://arxiv.org/abs/2501.03214) motivates the priority
group. The CMD uses Gaia's own extinction, so it is a consistency check. The
Gaia DR3 documentation describes the temperature-extinction degeneracy and
faint-star gravity systematics; interval flags are retained without extra cuts.

Source IDs are decimal text; convert to `int` only when reading LC2 rows.
Rebuild from the frozen TAP inputs (exact-ID upload, ADQL query, job record,
VOTable, dwarf sequence) with the command in `../pipeline/photometry/README.md`;
`prepare --out dump/<new-snapshot>` writes the upload and query for a new snapshot.

## The 6,000-source sample (8 October 2026)

3,000 M and 3,000 K dwarfs, drawn uniformly without replacement from each
`selection` view in numeric source-ID order with recorded seeds, without any
detector or photometry filter. The survey pipeline ran both detectors in
PSF flat, full PSF and SAP on the Mac. Outputs, inputs, logs and checks are in
`dump/dwarfs-6000-20261008/` on discPCIe: `boi-catalogue.json`,
`signal-ledger.json` and per-view disposition CSVs.

Result: 99 BOIs on 76 stars, 3,659 held, 1,615 dismissed. A review found that
74 of the 92 periodic BOIs have a fitted duration of exactly 9.60 h, that
several far-apart stars share a 2.96-day ephemeris, that the nine ultra-short
periods last 17–42% of the orbit, and that the single-event BOIs sit at the
BKJD 373 focus transient. These BOIs are pipeline dispositions and not planets;
a duration-density and cross-star ephemeris pass precedes any claim.
