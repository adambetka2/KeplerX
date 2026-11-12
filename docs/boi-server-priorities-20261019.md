# BOI server preparation priorities — October 19, 2026

This queue reviews all 125 delivered BOIs and proposes the next preparation work.
It launches no additional validation campaign. The current KIC 4829494 study is
saved separately in the [study result](../results/boi-validation/boi-further-study-20261019.json).

Use the [JSON evidence and queue](../results/boi-validation/boi-server-priorities-20261019.json)
or [sortable CSV](../results/boi-validation/boi-server-priorities-20261019.csv).
There are 116 other unresolved signals after setting aside the current study.

## First preparation batch

These are priorities for coherent input/native-model preparation, followed by
a readiness review before any complete statistical calculation. Their order is
not a calibrated probability that they are planets.

| Order | KIC | Period, days | Verified scene sources | Later native cases passing shape and numerics |
|---:|---:|---:|---:|---|
| 1 | 8690308 | 39.527119 | 39 | None yet |
| 2 | 8540098 | 6.950816 | 27 | psf-flat, sap |
| 3 | 10987243 | 18.593866 | 48 | None yet |
| 4 | 10196617 | 32.954419 | 28 | tic-sap-all |

**KIC 8690308.** A fresh target with no retained catalogue/harmonic flag, matching Gaia/TIC apparent radii, precise parallax and target-preferred scene fits in 7 of 8 quarters. No neighbor is more than 3 mag brighter within the retained 60 arcsec scene.
Next: Bind the own-SAP/PSF response and a defensible TIC/Gaia stellar prior; fit the fixed original signal with native baseline/noise inference before a complete scene census.
Retained limitation: No native posterior or FPP is admitted yet. The TIC mass/radius catalogue errors are missing; uncertainty must be declared and justified rather than treating catalogue points as exact.

**KIC 8540098.** Both retained PSF and SAP conditional native fits pass the declared shape and numerical checks; the field has 27 catalogued sources and consistent Gaia/TIC apparent radii.
Next: Authenticate quarter extraction response, constrain the brighter neighbor and unresolved companions, and declare complete stellar/body/scenario priors before the full calculation.
Retained limitation: A brighter source lies 16.6 arcsec away. The existing giant-body/phase-curve scope must be retained; no mass or FPP has been admitted. Apparent size is not an exclusion rule.

**KIC 10987243.** A largely untried 18.59 day signal with target-preferred scene fits in 11 of 13 quarters, a catalogued Gaia mass/radius and no retained catalogue/harmonic flag.
Next: Authenticate source response and the coherent stellar prior, then obtain native transit/noise inference. Keep the 40.5 arcsec brighter neighbor in the scene.
Retained limitation: No admitted native posterior or FPP. The delivered metadata has no TIC identity; a future exact-host catalogue/stellar-prior step must not assume a missing TIC entry means no such identity exists.

**KIC 10196617.** The later reviewed TIC-SAP endpoint passes the declared native and numerical checks. The 28-source scene is relatively small and all 3 measured localization quarters prefer the target.
Next: Recover the exact later TIC-prior/input bindings and authenticate SAP response to the 12.5 arcsec brighter star; preserve the failed Gaia sensitivity and earlier schedule error before deciding the coherent census model.
Retained limitation: The older feasibility state predates the passed TIC endpoint. A neighbor is about 5.4 mag brighter; scene size alone does not make this a clean blend problem.

## Conditional next batch

- **KIC 7300018:** Resolve the two unsupported TIC-neighbor metallicity models with authenticated coherent priors, retain every affected scenario, then assess sparse-weight precision for the complete same-photon calculation. Crowded 86-source retained scene. More identical draws cannot repair an unsupported neighbor model; do not drop the neighbors or substitute an unreviewed metallicity.

- **KIC 3330404:** Resolve the factor of 1.88 catalogue stellar-radius discrepancy with parallax/photometry and an appropriate stellar model; retain the TIC-native failure while defining the coherent route. Low-significance parallax and discrepant stellar radii make the apparent 2.64/4.96 Earth radius descriptors conditional. The ExoFOP refresh returned an unusable empty body, not an absence result.

- **KIC 7729457:** Establish a coherent photometry/parallax stellar prior and response to the two bright 12 arcsec neighbors, then obtain native inference. Faint G=18.25 host, low-significance parallax and discrepant catalogue radii. The low family-count proxy does not predict easy validation.

- **KIC 8805426:** Review a justified stellar-density-prior model and source response before any new census. Preserve the failed free-density experiment and its convergence/SNR values. Earlier native Rhat 2.01 and radius-ratio SNR 1.66 failed the declared fit criteria; only 4 of 7 scene quarters prefer the target and the nearest resolved source is 3.9 arcsec away. This is not an immediate full-census target.

## Coverage and interpretation

| Next-work group | BOIs |
|---|---:|
| Untried native preparation pool | 38 |
| Targeted native response review | 2 |
| Signal identity first | 38 |
| Host characterization first | 12 |
| Model or source work first | 6 |
| Outside unresolved queue | 8 |
| Delivered view authentication | 9 |
| Conditional preparation batch | 4 |
| Recent bounded feasibility review | 2 |
| First preparation batch | 4 |
| Contamination residual first | 1 |
| Current study completed | 1 |

Each signal has one primary work group; all other recorded issues remain in
its evidence. Shared hosts are grouped for efficient preparation. Possible
harmonics need independent event support and receive no multiplicity boost.

The older source scan covered PSF_FLAT. Nine QDIAG entries therefore need their
delivered-view source copy checked; this does not establish absent data or a
scientific integrity failure. Their unverified scene counts and cost estimates
are unset. A missing scene is never treated as an isolated one-source field.

Later reviewed native endpoints are retained alongside the earlier feasibility
cutoff. A passed conditional shape fit is not a full scenario posterior. Existing
failed endpoints, unfavorable probabilities and specific next steps remain explicit.

KIC 9220619 and KIC 9786351 also retain today's authenticated fixed-clock
feasibility measurements. Neither has a new native posterior or FPP from those
screens. Their next work is parent exposure/extraction-response authentication
and coherent stellar/scene priors; they are not labelled wholly untried.

October 19 NASA and SIMBAD requests for the eight shortlisted hosts returned no
rows in the declared queries. Three ExoFOP responses had exact TIC/Gaia DR3 identity
matches and no listed planets or follow-up files. The KIC 3330404 response was empty
and unusable; four delivered records lacked TIC IDs for that request. These are
query coverage statements, not universal absence or novelty claims. Original
nearby catalogue associations remain in the queue.

Apparent radii assume unit dilution. Nominal secondary, parity and modulation SNR
values are diagnostics. Neither size nor a missing test is used as a denial.
The 15+3N scenario count is a workload proxy, not an adopted future support or
wall-clock estimate. Future runs need exact-source readiness, complete priors,
a reviewed freeze and the unchanged declared scientific thresholds.
