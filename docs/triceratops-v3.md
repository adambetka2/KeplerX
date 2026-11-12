# TRICERATOPS Kepler V3

`TRICERATOPS_KEPLER_V3` changes one part of upstream TRICERATOPS
([Giacalone et al. 2021](https://arxiv.org/abs/2002.00691), pinned at
`d20875c`): the likelihood of each scenario. Six further changes prepare its
inputs for KBonus photometry. The scenarios, prior functions and validation
criteria of [post-boi-protocol.md](post-boi-protocol.md) are unchanged.

**Joint quarter-response likelihood.** Each photon is scored with its own
quarter's flux shares, because Kepler's quarterly rotation changes each star's
share of a KBonus aperture, by a median factor of three and up to 21 across 12
BOI scenes. A single averaged dilution can then credit a neighbour's signal to
the target, or the reverse.

1. **Calibrated Kepler PRF responses.** Those shares come from the calibrated
   Kepler PRF at the authenticated apertures and Gaia positions, because stock
   TRICERATOPS's fixed 0.75-pixel Gaussian is a TESS approximation that
   misplaces light in crowded Kepler apertures.
2. **Flux-closure gate.** A quarter is used only if its modelled response
   reproduces the observed SAP flux within 20%, because a response that doesn't
   account for the measured light biases every dilution-dependent probability.
   The delivered KBonus crowding keywords match the data in only 29% of
   quarters.
3. **Reviewed per-quarter noise.** Formal errors are scaled by each quarter's
   robust out-of-transit scatter and red-noise factor, because formal KBonus SAP
   errors understate the measured scatter, by a median of 8% and up to 6.7
   times over 169 quarters of 18 BOIs, and would make the likelihood
   overconfident.
4. **Per-case exposure integration.** Each model is integrated with the fewest
   of 20, 40 or 80 sub-samples that is accurate to a tenth of the bin noise,
   because 20 sub-samples misrepresent sharp ingresses at 30-minute cadence. For
   KIC 7373485 the error was 84 ppm against a 65 ppm limit.
5. **Depth-based field-population limit.** Background stars are drawn down to
   the faintest magnitude that could still produce the observed depth when fully
   eclipsed, because upstream's fixed T = 21 limit leaves out capable
   contaminants of faint BOIs.
6. **Scene adapter.** The exact Gaia sources and per-quarter geometry are bound
   directly, because upstream takes the parent TPF's owner as the host, and for
   a KBonus background source that is usually a different star.

## Status

Adam adopted V3 for statistical validation on October 25, 2026. A complete
20-repeat pass with a clean physical closeout is a statistical validation; stock
TRICERATOPS reruns are diagnostics. At 8 R⊕ or more, a pass is a Possible Planet
(Gomez Barrientos et al. 2025).

The likelihood and the scene adapter are local `post_boi` code of October 21,
2026 (`quarters.py`, `scene.py`, `patches/`). The PRF responses and changes 2–5
are applied by the October 25, 2026 overnight tools and are not yet in
`post_boi`. [quarter-likelihood.md](quarter-likelihood.md) has the evidence and
the known-object calibration.
