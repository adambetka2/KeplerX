# Vetter

The Vetter measures each detected signal (refined fit, MES, chi-square,
secondary, odd/even, morphology, binary evidence) with a faster copy of
upstream's vetting that gives the same numbers. A calibrated policy may then
dismiss a signal but never accept one; a numerical failure leaves it
UNRESOLVED. Run it with `run.sh vet PREPARED DETECTIONS OUTPUT`.

- `program.py`: stage entry, dispositions
- `metrics.py`: metric components, failures
- `policy.py`: scores, binary gate
- `binary_evidence.py`, `odd_even_box.py`: binary tests
- `cross_lane.py`: other lane's evidence
- `accelerated_vetter.py`, `gap_filling.py`: faster upstream Vetter
- `exact_fit.py`: lmfit-identical fits
- `residual_kernels.c`, `kernel_bindings.py`: compiled models
- `transit_model.py`: accurate batman integral
- `upstream/`: vendored original, unchanged

More: ../../docs/keplerX-subsystem-details.md (Vetter).
