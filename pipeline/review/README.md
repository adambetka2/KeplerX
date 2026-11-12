# BOI review

Offline review application for the survey's BOI catalogue. It reads the
preserved archive and verified display extracts, never reruns detection,
fitting or vetting, and stores human decisions as a separate dataset.

```bash
uv run --no-project python pipeline/review/app.py
```

Open <http://127.0.0.1:8877>; `Launch.command` does the same from Finder.
Defaults: archive `/Volumes/discPCIe/kbonus`; review data
`dump/kbonus-review-20261011` with `reviews.sqlite` (decisions and an
append-only audit history), `datasets/<id>/catalogue.{json,csv}` exports and
the checksummed `plots/` and `curves/` extracts. Override with `--archive`,
`--work` and `--port`. The database is bound to the frozen catalogue hash and
rejects any other catalogue instead of migrating decisions.

Three detector queues hold the grouped transit, shallow and sparse-event BOIs;
a fourth keeps legacy survivor-review objects. Each BOI appears once with all
its detector and view variants. Keys: Y include, N exclude, Z undo, arrows move.
The signal monitor shows retained event windows and odd/even cycles; the
stellar monitor shows SAP, PSF and PSF FLAT cadences by quarter or orbital
phase with the frozen masks in amber. Median bins are visual summaries, not fits.

`prepare.py` builds the read-only display extracts from the server corpus
(`--full` for complete curves) and writes a manifest after completion. Never
commit the extracts.
