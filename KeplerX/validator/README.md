# Validator

The Validator takes a localized candidate to statistical validation: a
duration-aware preparation, the native transit fit, and the TRICERATOPS V3
false-positive probability over twenty frozen repeats, with its closeout. Run
`run.sh prepare-native` and `run.sh fit-native`; the probability campaign
still needs the frozen V3 runtime.

- `preparation.py`: duration-aware preparation
- `native_fit.py`, `native_likelihood.py`: native transit fit
- `transit_likelihood.c`, `kernel_bindings.py`: compiled likelihood
- `protocol.py`, `campaign.py`, `quarters.py`: V3 protocol, repeats
- `scene.py`, `aperture.py`, `psf_flat.py`, `psf_response.py`, `psf_scene.py`: scene, responses
- `protection.py`, `kbonus.py`, `ephemeris.py`, `occultation.py`: closeout
- `probability/`: TRICERATOPS V3 engine
- `records.py`, `benchmark.py`, `juliet_reference.py`: receipts, benchmark, golden
- `dependencies.py`, `patches/`, `frozen-runtime-requirements.txt`: frozen runtime
- `candidate-template.json`, `upstream/`: candidate record, vendored dynesty

More: ../../docs/keplerX-subsystem-details.md (Validator).
