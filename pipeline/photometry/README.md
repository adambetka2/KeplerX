# Photometry

Readers and builders for the KBonus LC2 corpus and the M- and K-dwarf host
selections. Original data stay read-only.

- `lc2.py` reads complete LC2 sources and their extraction views (PSF, PSF
  flat, SAP and native-quarter diagnostics) from the mounted corpus through
  each disc's `data/index.sqlite`.
- `export_cohort.py` freezes the 406,547-source cohort's identities and
  schemas for a fresh MAST retrieval; `ingest.py` rebuilds the corpus from
  verified MAST originals into Parquet shards with hash manifests.
- `select_m_dwarfs.py` indexes the published KBonus M-dwarf catalogue against
  the corpus; `select_k_dwarfs.py prepare|build` builds the Gaia-selected
  K-dwarf catalogue. Criteria and outputs: `../../docs/dwarf-cohort.md`.
- `requirements.txt` pins the reader environment.

```bash
uv run --no-project --with-requirements pipeline/photometry/requirements.txt \
  python pipeline/photometry/select_k_dwarfs.py build --inputs dump/k-dwarfs-20261006 \
  --m-dwarfs dump/m-dwarfs-20261005/selection/m-dwarfs.sqlite --out dump/k-dwarfs-20261006/rebuilt-selection
```
