# Detector

The Detector searches a prepared light curve for periodic transits. It whitens
the flux with upstream's template bank, folds trial periods of 0.5-500 days
(bounded on the GPU, resolved exactly on the CPU) and repeats masked passes
until nothing new appears. Run it with `run.sh detect PREPARED OUTPUT`.

- `program.py`: stage entry, outputs
- `upstream_adapter.py`: arrays into upstream's Dump
- `compiled_search.py`: whitening, search, GPU dispatch
- `search_kernels.c`, `kernel_bindings.py`: exact CPU kernels
- `gpu_screening.py`, `gpu_screening.metal`, `gpu_screening_host.m`: GPU screen
- `gap_padding.py`: segments, padding
- `flattened_templates.py`: flattened long templates
- `single_event_veto.py`, `candidate_ephemerides.py`: veto, ephemerides
- `repeat_rule.py`: passes, repeats
- `stable_ranking.cpp`: tie-exact sorting
- `upstream/`: vendored original, unchanged

More: ../../docs/keplerX-subsystem-details.md (Detector).
