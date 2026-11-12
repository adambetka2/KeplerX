# Tests

KeplerX keeps golden tests (a whole stage on fixed real inputs against a
reference that does not move: vendored upstream, a closed form or a stored
record) and end-to-end tests (the real pipeline on fixture photometry, judged
on scientific outcomes), and no unit tests. Run them all with `run.sh check`.

- `test_golden_detector.py`, `test_golden_vetter.py`: stages against upstream
- `test_golden_localizator.py`: accelerated against native
- `test_golden_binary_gate.py`: binary gate decisions
- `test_golden_preparation.py`, `test_golden_calibration.py`: preparation, physics
- `test_golden_validator.py`, `test_golden_validator_probability.py`: Validator
- `test_e2e_known_planet.py`, `test_e2e_localization.py`: TrES-2 b, KOI K00791.01
- `test_e2e_planet_ephemerides.py`, `test_e2e_long_transit_planet.py`: Detector finds
- `test_e2e_sap_lane.py`, `test_e2e_calibration_study.py`: SAP lane, study
- `test_e2e_failures.py`, `test_e2e_validator.py`: broken inputs, Validator fit
- `conftest.py`, `fixtures/`, `golden/`: shared inputs, stored records

What each test must hold: ../../docs/keplerX-subsystem-details.md (Tests).
