# Vetting

Downstream evidence and dispositions for detector proposals. Vetting never
searches a light curve: it measures the events a detector supplied and gates
them in a declared order (shape, orbital, instrument, localization,
contamination). A signal that passes every gate in one photometric view is a
BOI; every other signal keeps its first failure stage and reason.
Protocol: `../../docs/survey-protocol.md`.

- `vetting.py` runs the stages for a batch. `vetting_triage.py` applies the
  gates in order; `vetting_disposition.py` with `disposition-policy.json`
  writes HELD, DISMISSED or BOI; `boi.py` catalogues the BOI proposals.
- `vetting_signals.py`, `vetting_flux.py`, `vetting_models.py` and
  `vetting_periodic.py` associate, characterize, fit and time the supplied
  events; `vetting_systematics.py` collects instrument evidence and
  `vetting_alias.py` tests shorter-period eclipse trains.
- `vetting_pixels.py`, `vetting_scene_prf.py` and `kepler_prf.py` (adapted
  from lkprf) retrieve parent target pixels, localize the
  event in difference images and compare the scene with Kepler PRFs;
  `parent_index.py` builds the verified parent-TPF index.
- `vetting_evidence.py`, `vetting_host.py`, `catalogue_search.py` and
  `catalogue_snapshot.py` supply catalogue, shared-time and host context from
  frozen snapshots.
- `calibrate.py` sets the disposition policy by replaying the opened
  development controls; `vetting_checkpoint.py` transfers completed evidence
  into a repair generation.
- `statistical_validation.py` runs the fixed 20-seed TIC census declared in
  `statistical-validation.json`; `validation_results.py` admits only the exact
  40-result two-target census and can replay every retained weight array.
- `config.json` holds the gate settings; `cloud-*.json` bind the replayable
  catalogue, parent and pixel inputs; `requirements.txt` pins the survey
  environment; `test_*.py` are the contract tests.

Stages run through `python pipeline/run.py <stage>`; see `../README.md`.
