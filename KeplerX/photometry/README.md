# Photometry

Photometry reads one KBonus lane of a source from LC2 or native arrays and
prepares it for the Detector: PSF Flat as delivered, or the stitched SAP
decrowded and flattened as KBonus flattened PSF (the `sap_flat` view). Run it
with `run.sh prepare REQUEST OUTPUT`.

- `program.py`: stage entry, lane reading
- `flattening.py`: KBonus flattening emulated
- `star_share.py`: star's share of SAP
- `field-artifact-epochs.json`: field-wide artifact dates

More: ../../docs/keplerX-subsystem-details.md (Photometry).
