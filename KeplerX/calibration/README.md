# Calibration

Calibration builds the KBonus-realistic sample: real lane hosts with injected
planets, binaries and artifacts, diluted and flattened as KBonus delivers
them. It runs the pipeline on that sample as one study and scores efficiency
and false alarms against each signal's information S/N.

- `sample_builder.py`: builds the sample
- `population.py`: stellar inputs, weights
- `signal_injection.py`: injections, flattening emulator
- `artifacts.py`: artifact models
- `lane_census.py`: lane census, realism
- `lane_twin.py`: SAP lane's twin
- `study.py`: one calibration study
- `scoring.py`, `vetting_policy.py`: scores, policy fits
- `survey_benchmark.py`, `benchmark_scoring.py`: survey benchmark
- `localization_benchmark.py`: localization benchmark
- `registered-sample.json`: registered sample

More: ../../docs/keplerX-subsystem-details.md (Calibration).
