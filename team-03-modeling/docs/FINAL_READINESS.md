# Final repository readiness

The canonical combined project is `AsvM45/ADSC_AA_Team3`. The official ADSC repository is outside this release and has not been modified by the integration.

Both Team 3 remotes were refreshed during the final review. The forecasting source snapshot and original team files are accounted for in [the integration manifest](INTEGRATION_MANIFEST.md), with [named contributor credits](../CONTRIBUTORS.md). Existing analysis scripts remain runnable; the release pipeline builds on the original forecasting implementation.

Implemented release scope: ingestion/validation, airport reference retrieval, descriptive exploration, seasonal baselines, pooled gradient-boosted adjustment, retrospective matched-case evaluation, matured calibration, uncertainty, controlled feature ablations, experiment/failure manifests, local figures and six-section rubric evidence. See [the contract](RELEASE_CONTRACT.md) for assumptions and limitations and the README for reproduction commands. Historical outputs are not copied into the release as fresh measurements.

Final hardening removes executable pickle-cache deserialization from passenger loading; bounds public downloads and validates each HTTPS redirect against approved source hosts; and checks privacy-sensitive directory/environment names without case sensitivity. Regression tests exercise the request boundaries without network access. These changes do not alter the forecasting equations or feature definitions. New runs still require fresh compatible source fingerprints; old calibration manifests must not be relabeled.

A scoped offline security scan of commit `316c004756f311b9cf2c46dabb28394100cf0819`, including an independent baseline review, found no confirmed exploitable vulnerability under the local-operator CLI threat model. That finding is scoped to the reviewed code and assumptions. It is not a dependency advisory audit, a guarantee against every threat, or scientific approval. The subsequent hardening is covered by targeted synthetic regression tests.

Remaining human decisions: independent scientific review, source-owner confirmation of release timing/missing rows/duplicates, sponsor-approved publication, and the official submission handoff. Retrospective results do not establish an untouched holdout. External historical vintages, a validated 36-month model, alternative Ridge/XGBoost challengers, a dashboard and cloud experiment infrastructure are outside this release; open research/backlog items are not marked complete merely because the core is runnable.

AA inputs, derived metrics, plots, fitted models and run fingerprints remain in ignored local directories. Only source, synthetic tests and public-safe documentation are included in Git or CI.
