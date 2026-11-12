# Probability engine

The Validator's TRICERATOPS V3 engine computes a candidate's scenario
probabilities (lnZ, FPP, NFPP) with the same outputs as pinned TRICERATOPS
d20875c, faster and in float64 only. It is a copy of `spacelab/fast_triceratops`
(1c04e9be2) and runs in the frozen V3 runtime.

- `engine/`: static V3 scenarios, campaigns' route
- `joint.py`: fused joint quarter likelihood
- `kernels.py`: fast scenario likelihoods
- `scenarios.py`: exact scenario repairs
- `workflow.py`: preserved experiment runner
- `bench.py`: kernel replay benchmark
- `gold.py`: offline Kepler-10 b case

More: ../../../docs/keplerX-subsystem-details.md (Validator probability engine).
