# Third-party notices

KeplerX is under the MIT License ([LICENSE](LICENSE)).

## Upstream copies

These folders hold other authors' code, kept for reference and comparison (renames are recorded in `KeplerX/upstream.json`). They keep their original licences.

| Path | Upstream | Licence |
| --- | --- | --- |
| `KeplerX/detector/upstream/`, `KeplerX/vetter/upstream/` | [TRASHDUMP and RECYCLEBin](https://github.com/robertfwilson/trashdump) by Robert F. Wilson, which include [FFA](https://github.com/petigura/FFA) by Erik Petigura (MIT) and parts of [Lightkurve](https://github.com/lightkurve/lightkurve) (MIT) | No licence published by the author; all rights remain with the author |
| `KeplerX/localizator/upstream/` | [INCInERATOr](https://github.com/juliemoqs/incinerator) by Julie Moquin, Robert F. Wilson, Jorge Martínez-Palomera and Christina Hedges | MIT ([LICENSE](KeplerX/localizator/upstream/LICENSE)) |
| `KeplerX/validator/upstream/dynesty/` | [dynesty](https://github.com/joshspeagle/dynesty) by Josh Speagle, Sergey Koposov and contributors | MIT ([LICENSE](KeplerX/validator/upstream/dynesty/LICENSE)) |
| `KeplerX/tests/fixtures/dwarf-sequence.txt` | [Mean dwarf stellar color and effective temperature sequence](https://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt) by Eric Mamajek | The author's terms; cite Pecaut & Mamajek (2013) |

## Adapted from

KeplerX's own code, adapted from:

- `fast_triceratops/engine/`, `KeplerX/validator/probability/engine/` — [TRICERATOPS](https://github.com/stevengiacalone/triceratops) by Steven Giacalone
- `fast_triceratops/native_orbits.c`, `KeplerX/validator/transit_likelihood.c`, `KeplerX/vetter/residual_kernels.c` — [batman](https://github.com/lkreidberg/batman) by Laura Kreidberg and [juliet](https://github.com/nespinoza/juliet) by Néstor Espinoza
- `pipeline/shallow/bls.c` — [Astropy](https://www.astropy.org)'s BLS
- `pipeline/vetting/kepler_prf.py` — [lkprf](https://github.com/lightkurve/lkprf) by Christina Hedges and the TESS Science Support Center
