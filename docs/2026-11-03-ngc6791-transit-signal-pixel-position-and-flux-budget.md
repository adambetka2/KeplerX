# 2026-11-03 NGC 6791 transit signal: pixel position and flux budget

Written on 3 November 2026 while the Mac ran the SAP calibration batches.
Read-only on the
data; nothing in the pipelines changed. The lead comes from the 17 October
exploration: Gaia DR3 2051287341327402624 (KIC 2436588, G 18.49, NGC 6791
member), period 9.28698 d, epoch BKJD 138.123.

## What was measured

Light curves: the KBonus LC2 corpus on discSSD/discPCIe (both the delivered
PSF Flat and the stitched SAP), folded on the 17 October ephemeris with a
per-event linear baseline (±0.5 d window, ±0.12 d excluded, ±0.06 d in
transit). Pixels: the Kepler superstamp tile kplr100000940 for Q11, Q12 and
Q16 on discSSD (100 x 20 pixels), as per-event in-minus-out-of-transit
difference images averaged over 11, 9 and 8 events.

## Findings

1. **Depth over all events is 5.7 ppt, not 8–9.** SAP: 5.66 ± 0.16 ppt over
   116 events; PSF Flat: 5.51 ± 0.17 over 101 events. The 8–9 ppt quoted on
   17 October is Q12 alone (8.9 ± 1.1). Odd 5.75 ± 0.23, even 5.52 ± 0.23.
   Every one of the 116 events shows a dip. No secondary at phase 0.5:
   0.05 ± 0.12 ppt (SAP), 0.16 ± 0.12 (PSF Flat), as observed.
2. **The depth changes by quarter far beyond its errors:** 1.9 ppt in Q1
   (3 events), 3.6 in Q10, 8.4 in Q2, 9.6 in Q5, 10.3 in Q4. By season:
   6.3, 5.1, 5.4, 6.3 ppt. This is dilution changing with the aperture, not
   the signal.
3. **The KBonus photometry of this cluster core is contaminated and the PSF
   lane does not decrowd it.** The target's median flux is 1,505 e-/s (PSF)
   and 1,383 e-/s (SAP) against about 500 e-/s expected for G 18.49
   (Kp about 18.3). PSF and SAP agree to 10% for every star in the
   neighbourhood. The G 14.26 star 15.7 arcsec away (Gaia DR3
   2051288067182953344, proper motion (−11.4, −7.8) mas/yr, a foreground
   star) is absent from the LC2 corpus; its roughly 25,000 e-/s are carried
   by two faint catalogued neighbours (15,673 e-/s on a G 18.3 star and
   8,622 e-/s on a G 18.7 star). Depths here are diluted by factors of 3 to
   30 and the factor changes with season.
4. **The neighbours show the same dip, apportioned by season.** The G 16.76
   star 9.3 arcsec away: 1.5 ppt (0.8 to 2.4 by season, moving opposite to
   the target's). The G 18.7 star at 6.8 arcsec: 0.7 ppt (5.0 in season 0).
   Four others at 11–15 arcsec: 0.2–0.6 ppt. Flux budget: neither the G 16.76
   star nor the G 14.26 star can host the signal, because the light they
   would need to put into the target's photometry exceeds the target's whole
   measured flux. The summed absolute light-loss over seven stars is 63–67
   e-/s in season 0, about 31 in seasons 1 and 2 and 22–29 in season 3, so
   part of the light-loss lands in sources the corpus does not carry; the
   budget alone cannot fix the true depth.
5. **The difference images put the dip at the target's column on the tile's
   bottom edge** at S/N 17 (Q11), 11 (Q12) and 8 (Q16). In Q11 the target
   sits at pixel (53.40, +0.07), the G 19.61 member at (52.72, −0.68) and the
   G 16.76 star at (53.64, −2.25); the superstamp WCS checked against 12–13
   isolated G < 17.5 stars is good to +0.1–0.2 px with 0.3 px scatter. A
   Gaussian fit to the Q11 dip gives x 52.95 ± 0.06, y 0.14 ± 0.07, sigma
   0.49 px, total 5.6 ± 0.6 e-/s above the edge. The y excludes the G 16.76
   star and every neighbour two or more pixels below the edge. It matches the
   target, but the fitted y depends on the assumed PRF width (−0.39 at sigma
   0.9 px), and with the position fixed the G 19.61 star fits better than the
   target in Q11 (chi-square 76 against 107) and Q16 (45 against 59) while
   the target fits better in Q12 (32 against 50). These tiles cut the PSF at
   the edge; they cannot separate two stars one pixel apart.
6. **Both candidates are cluster members.** Target proper motion (−0.22,
   −1.96) mas/yr against the cluster's (−0.43, −2.30) with 0.14 mas/yr scatter
   among 565 G < 18 stars within 3 arcmin, consistent at the target's
   precision; the G 19.61 star is at (−0.64, −2.41), the G 16.76 star at
   (−0.53, −2.32).
7. **True depth.** If on the target: 5.6–7.7 e-/s of dip above the edge
   against roughly 525 e-/s expected with 0.65–0.77 of the PRF inside the
   fitted box gives 1.4–2.3%, and the KBonus route agrees (5.66 ppt times
   1,505/525 = 1.6%), so Rp/R* 0.12–0.15 and 1.1–1.4 R_J on a 0.96 R_sun
   member. If on the G 19.61 star (about 190 e-/s, less of it on the tile):
   roughly 7%, a 0.2 R_sun companion, an eclipsing M dwarf. The secondary
   limit (about 0.3 ppt observed at 2 sigma) does not exclude that.

8. **Shape.** A batman fit to all 116 events (SAP, local baselines, a
   free depth scale per quarter to absorb the dilution) gives a transit
   3.0–3.7 h long and a chi-square nearly flat in impact parameter: b 0
   implies a stellar density of 3.0 solar, b 0.7 gives 1.14 solar (the
   0.94 M_sun / 0.96 R_sun member has 1.06), b 0.85 gives 0.5. A grazing
   geometry is mildly preferred (delta chi-square 13 over 6,632 points).
   The duration fits either candidate host and does not separate them.

## How the signal was found

In an open-ended cluster census on 17 October 2026:
Gaia DR3 mixture-model membership for NGC 6791 and NGC 6819 (2,483 and
1,327 KBonus members), both LC2 views, an FFT harmonic-sum spectrum followed
by a box search of each peak's fundamental with masking, PSF flux excess and
CROWDSAP as dilution estimates, batman injections on six stars, and vetting
flags for odd/even, SAP-only, dominant events, neighbour signals, leaks,
secondaries and sizes. Over 3,810 members it kept 13 clean detections on 11
sources; this star ranked second (box signal-to-noise 44.2 SAP, 42.0 PSF,
116 events, observed 4,397 ppm, dilution-corrected 10,923 to 16,923 ppm,
1.0 to 1.25 R_J). Its neighbour table held only KBonus sources, so the
G 19.61 star one pixel away never entered its flux budget or pixel test.
Adam stopped that study the same evening; the candidate stayed as a lead.

## Why the v1 survey has no signal here

The v1 catalogue (results/kbonus-20261010, every signal outcome) holds no
row for this source. The frozen runner (commit 87bd044) keeps only cadences
with SAP_QUALITY equal to zero. KBonus copies the superstamp TPF's QUALITY
column to every source cut from that stamp, and in a 100 x 20 pixel custom
aperture a cosmic ray anywhere sets bit 128 (and 8192), so this star keeps
2,113 of 52,062 cadences, all in Q12. One quarter cannot meet the
two-quarter coverage rule, and eight events would not reach MES 8 either.

This is a population effect, not one star. Across the LC2 corpus:

| Sources | Count | Median share of cadences at quality 0 | With any v1 signal | v1 BOI signals |
|---|---|---|---|---|
| Superstamp origin (TPF id 100000925 and up) | 10,663 | 7.9% | 2.8% | 0 |
| All others | 395,884 | 90.9% | 14.6% | 3,447 |

The deficit holds in every magnitude bin from G 16 to 19. The NGC 6791 and
NGC 6819 fields are therefore a coverage hole of the v1 survey, not a null
result, and no v1 upper limit should be quoted for them. KeplerX already
retains bits 16, 128, 8192, 131072 and 262144 (photometry
retained_quality_bits), so its survey will search these sources; their
photometry is still contaminated as described above, so any detection there
needs the pixel test before a depth is quoted.

## Resolution with the full point spread function (3 November, afternoon)

Adam approved fetching the missing data from MAST. Three files of tile
kplr100000939 (CCD rows 803 to 822, directly below tile 940, same 100
columns) for Q11, Q12 and Q16, and the Kepler PRF models for channels 53
and 81 (kplr16.1 and kplr24.1), now live in discSSD/data/pixels/kepler and
kepler-prf. Each quarter's two tiles were mosaicked on their common cadences
into a 40-row by 100-column frame, so the target's point spread function is
whole. The direct image around the target was fitted as a sum of Kepler
PRFs at the Gaia positions of every star brighter than G 21 within nine
pixels, with free fluxes and one global shift; the shift came out at +0.05
to +0.10 px in x and +0.08 to +0.14 px in y, and the bright stars' fitted
fluxes match their magnitudes (the G 14.26 star 24,600 against 24,800 e-/s
expected; the target 517 to 520 e-/s). The per-event difference image was
then fitted with one PRF of free position and amplitude, with the position
fixed on each neighbour in turn, and with two PRFs fixed on the target and
the G 19.61 star.

| Quarter | Channel | Events | Dip, e-/s | Free position, px from G 19.61 | px from target | chi-square fixed on G 19.61 | fixed on target | Two-source split, e-/s (target / G 19.61) |
|---|---|---|---|---|---|---|---|---|
| Q11 | 53 | 11 | 13.6 ± 0.5 | 0.09 | 0.97 | 204 | 783 | 0.9 ± 0.6 / 12.9 ± 0.5 |
| Q12 | 81 | 9 | 13.7 ± 0.5 | 0.08 | 0.96 | 223 | 832 | 1.6 ± 0.4 / 12.9 ± 0.5 |
| Q16 | 81 | 8 | Q16_PLACEHOLDER |

The free fits have chi-square 197 and 220 for 165 degrees of freedom. The
dip is on the G 19.61 star, not on the KBonus target, in two independent
channels, seasons and PRF models. The light-loss of 13.6 e-/s against the
roughly 175 e-/s that a Kp 19.4 star gives is an eclipse of about 8 percent.

**The star is a known eclipsing binary.** Gaia EDR3 2051287311268711168 is
KIC 2436579, listed by Sanjayan et al. (2022, Acta Astronomica 72, 77;
arXiv 2303.07422; VizieR J/AcA/72/77) in their Table 5 of non-member
binaries of NGC 6791 as "eclipsing" with P = 9.287153(22) d and T0 = BJD
2454971.1134, which is BKJD 138.113, fourteen minutes from the census epoch
of 138.123. Their survey worked from Fourier spectra of every superstamp
pixel, which is why it found a G 19.6 binary that KBonus does not carry as a
source. They classify it as a non-member on Gaia EDR3 astrometry; its
proper motion (-0.64, -2.41) mas/yr sits 0.2 mas/yr from the cluster mean,
so the call rests on parallax or their threshold and does not matter here.

## Verdict

Not a planet candidate. The 9.287-day signal on the NGC 6791 member Gaia
DR3 2051287341327402624 is the primary eclipse of the known binary KIC
2436579, 4 arcseconds (one Kepler pixel) away, leaking into the member's
KBonus photometry at 5.7 ppt because the superstamp photometry is not
decrowded. The 17 October lead is closed. The bright neighbours were never
the source either.

## Lesson for the surveys

In the NGC 6791 and NGC 6819 superstamps the KBonus PSF lane is not
decrowded, a star missing from the corpus pushes its light onto catalogued
neighbours, and depths are season-dependent lower limits. Any candidate
there needs the pixel test before a depth or radius is quoted.

A second lesson: a KBonus source in a superstamp can carry a neighbour's
eclipse at the ppt level with a clean odd/even and no secondary, and the
pixel test on a single tile cannot separate stars one pixel apart when the
target sits on the tile edge. The KeplerX Localizator should mosaic the
adjacent superstamp tiles and fit the Kepler PRF with all Gaia neighbours
before any depth is quoted for a superstamp source, and should match every
cluster signal against Sanjayan et al. (2022) for NGC 6791 and the
corresponding catalogue for NGC 6819.

Scripts (scratch, not kept in the repo): fold.py, fold2.py,
budget.py, diffimg.py, dipfit.py, wcscheck.py, shape.py, prffit.py.
