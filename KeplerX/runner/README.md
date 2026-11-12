# Runner

The Runner executes single stages and batches over an explicit manifest:
long-lived workers run prepare, detect, vet and localize for each case and
record wall time, CPU and the source they ran. A run's outputs live in one
SQLite database. Run a batch with `run.sh batch MANIFEST OUTPUT --workers N`.

- `program.py`: stage execution, records
- `batch.py`: manifest workers, dispatch
- `outputs.py`: `results.sqlite` storage
- `bois.py`: joined stage evidence

More: ../../docs/keplerX-subsystem-details.md (Runner).
