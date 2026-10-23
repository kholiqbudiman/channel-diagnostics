# Evaluating Channel Diagnostics for Capacity Reallocation in Ultra-Compact CNNs

Research artifact for the manuscript by Kholiq Budiman, Aufaclav Zatu Kusuma Frisky, and Jazi Eko Istiyanto. Publication status: manuscript prepared for submission; no publication DOI assigned here.

## Research question
Can diagnostics measured on a trained compact CNN guide channel reallocation when candidate architectures are trained from fresh initializations under matched parameter and MAC budgets? The study evaluates prediction fidelity, architecture-selection utility, and the improvement available among candidates. It does not introduce a validated universal allocation algorithm.

## Contents and scope
- `research_artifacts/pilot_audit`: corrected pilot records (36 runs).
- `research_artifacts/stability_audit` and `stability_sources`: stability records and historical source (72 runs).
- `research_artifacts/directed_20261017`: directed intervention records (126 records, including 18 reused Base runs; 108 new runs).
- `research_artifacts/proxy_transfer_20261017`: prospective DTD protocol, data manifests, commitments, and completed results (63 runs), plus development-context diagnostic comparisons.
- `research_artifacts/channel_retraining_20261017`: explanatory masking, bidirectional-response, and post-outcome projection analyses.
- `tables`: retained LaTeX table rows checked against result summaries.

There are 279 distinct training runs, not 297 independent runs. Historical folder names containing `old_comparison` refer to development-context analyses; incorrect legacy semantic-label results are not the primary evidence.

## Verify retained results (CPU; no dataset download required)
Use Python 3 and NumPy. From this repository root:

```bash
python -m pip install -r requirements-audit.txt
python research_artifacts/verify_corrected_evidence.py
python research_artifacts/verify_directed_tables.py
python research_artifacts/verify_empirical_evidence.py
python research_artifacts/channel_retraining_20261017/verify_channel_retraining.py
```

See `AUDIT_RESULTS.json` for the actual local execution results. These checks reconcile retained metrics, confusion matrices, selection histories, decisions, table values, source locks, and projection identities. They do not rerun image inference or independently authenticate remote checkpoints. Retained server audit receipts are historical evidence, not a fresh server audit.

## Datasets
See `DATASETS.md`. Images and trained checkpoint binaries are not bundled. Exact split manifests and recorded hashes are retained where available. Downloading the current upstream dataset does not itself establish byte-level identity with the recorded experiment; compare against retained manifests.

## Training versus result verification
Training and analysis sources are retained for provenance. Historical training scripts contain environment-specific paths and scheduling assumptions. They require data acquisition, original dependencies, and path adaptation; the audit requirements file is not a complete GPU training environment. Do not run historical schedulers as a quick start. The four CPU verification commands above are the tested entry points.

DTD uses internal partitions of official train1+val1 images for the prospective transfer study, not the official DTD benchmark test protocol. Post-outcome fitted slopes are explanatory analyses, not prospective predictors.

## Integrity and reuse
`ARTIFACT_SHA256.json` records distributed file hashes. Original evidence files are preserved without rewriting their hashes or provenance paths. No dataset redistribution rights are granted here. A code license has not yet been selected by the authors; this draft does not grant an open-source license. Please cite the final article when available and retain original dataset attribution.
