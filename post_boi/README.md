# Post-BOI validation

One preparation path for survey and M/K BOIs; joint quarter-response likelihood.
Read `docs/post-boi-protocol.md`; copy `candidate-template.json` into a new case.
Install the reviewed repairs in `docs/validation-repairs-20261021.md` in a new runtime.
Apply the quarter callback patch there; read `docs/quarter-likelihood.md` for controls.
Fill fields from evidence (null is no default); use authenticated native photons.
`psf_flat.py` reads original Flat; `psf_response.py`/`psf_scene.py` require reviewed response calibration.
`kbonus.py` (flux closure, quarter noise), `ephemeris.py` and `occultation.py` close out V3.
`python -m post_boi.prepare CANDIDATE PREPARED` protects all declared transits.
`python -m post_boi.native CANDIDATE PREPARED FIT` delegates to juliet.
Review the fit and source/spacecraft/stellar readiness before freezing.
Judge protection by observed-photon signal transfer, not an all-tail contact veto.
`python -m post_boi.campaign freeze CANDIDATE NEW_OUTPUT` declares all seeds.
Run `verify` and `resume` from `NEW_OUTPUT/source`, through the pinned runtime.
Twenty final-clock repeats use upstream models and one million attempts per scenario.
The first-two futility screen controls allocation only; no partial admission.
Recovery diagnostics default to zero; exactly three require a timing reason.
Resume skips hash-verified repeats; retain failures. Upstream has no within-scenario checkpoint/full weight audit.
Legacy campaigns keep their own frozen source, outputs and original protocol.
