# Localizator

The Localizator finds where a kept signal's light comes from: it fits the
transit depth in each pixel of the parent target pixel files and locates the
source with a PRF model among the Gaia DR3 stars around it. Signals the Vetter
dismissed are not localized. Run it with
`run.sh localize PREPARED DETECTIONS VETTING REQUEST.json OUTPUT`.

- `program.py`: stage entry, native adapter
- `pixel_requests.py`: which pixels to read
- `mast_pixels.py`: TPFs streamed from MAST
- `accelerated_model.py`: faster pixel and PRF maths
- `gaia_field.py`: local Kepler-field Gaia table
- `gaia_cone_search.py`: ESA archive cone search
- `upstream/`: vendored original, unchanged

More: ../../docs/keplerX-subsystem-details.md (Localizator).
