# Train-only EEG preprocessing and validation threshold calibration — real results

Date: 2026-10-15 (UTC+8). Workflow: https://github.com/ishchy/ishchy/actions/runs/38736861952
Status: **both matrix jobs SUCCESS**. Artifacts:
- UCI: \`eeg_feature_calibration_uci\` (11973278615)
- PhysioNet: \`eeg_feature_calibration_physionet\` (11974587889)

## Protocol

- Raw labels, participant splits, and future-time partitions unchanged.
- Same-script reference: plain StandardScaler fitted on training only, no class weights, 0.5 decision threshold.
- Candidates: StandardScaler or training 1%-99% winsorization + train-fitted StandardScaler, each with/without train-only class weighting.
- Model checkpoints, SVM C and candidate choice made on the validation subset only.
- Threshold from validation ROC/Youden, restricted to [0.05,0.95] and shrunk halfway toward 0.5 to mitigate calibration overfit. Evaluation saves identical test probabilities scored at fixed 0.5 and at the validation-selected threshold.
- Fixed models: FFSEEG-Residual Top-K8, FullGraphConv, MLP, MI-SVM-8, RBF-SVM-14. 65 max epochs for neural models; fixed seed 24701.
- PhysioNet: 109 subject-disjoint original 10 folds, 6538 test windows per model.
- UCI: chronological 3 forward-time folds with purge 256 samples and no overlapping test rows, 7488 test samples per model.
- These comparisons are exploratory; previously inspected test results informed the design.

## Results: 10-fold PhysioNet averages

| Model | Reference BA | Reference F1 | Reference AUC | Val-selected + calibrated BA | Val-selected F1 | Val-selected AUC |
|---|---:|---:|---:|---:|---:|---:|
| FFSEEG-Residual | 75.12% | 73.32% | 82.33% | **74.88%** | 73.07% | 82.34% |
| FullGraphConv | **75.22%** | **73.67%** | **82.66%** | 74.46% | 72.64% | 82.38% |
| MLP | 74.08% | 72.23% | 82.18% | 74.19% | 72.26% | 82.13% |
| MI-SVM-8 | 75.11% | 73.14% | 80.43% | 74.68% | 72.63% | 80.41% |
| RBF-SVM-14 | 75.29% | 73.44% | 80.71% | 74.71% | 72.87% | 80.83% |

FFSEEG-Residual reference -> calibrated validation-selected: BA **−0.24 pp**. Val threshold only on the baseline: BA **74.78%**, a **−0.34 pp** change from fixed 0.5 on identical predictions. The selected candidate evaluated at fixed 0.5 gives 75.05% BA, confirming the final 74.88% is not a gain. No new superiority claim.

## Results: 3-fold chronological UCI averages

| Model | Reference BA | Reference F1 | Reference AUC | Val-selected + calibrated BA | Val-selected F1 | Val-selected AUC |
|---|---:|---:|---:|---:|---:|---:|
| FFSEEG-Residual | **52.62%** | 44.74% | 57.48% | **54.15%** | 45.22% | **53.91%** |
| FullGraphConv | 49.57% | 41.46% | 50.54% | 50.08% | 42.55% | 50.41% |
| MLP | 52.39% | 40.98% | 54.16% | 50.61% | 42.23% | 51.59% |
| MI-SVM-8 | **55.57%** | 43.06% | 56.90% | 54.89% | 41.84% | 56.65% |
| RBF-SVM-14 | 52.90% | 44.21% | 50.98% | 51.43% | 44.39% | 51.41% |

FFSEEG-Residual reference -> selected + calibrated improves BA **+1.53 pp** but AUC **−3.57 pp**; it still underperforms the standard MI-SVM-8 BA (55.57%). Threshold only on the baseline improves BA just **+0.05 pp**, whereas preprocessing/weights candidate at fixed 0.5 gives **52.52%** BA, below reference.

## Interpretation

1. Train-only winsorization / class weighting and validation threshold adjustment do **not** yield a reproducible overall gain across both datasets. Do not discard negative tests.
2. PhysioNet: the 10 subject-held-out folds have run-confounded R01/R02 class labels, making performance **unfit as unconfounded eye-state recognition evidence**.
3. UCI: single participant and one recording; 3 future-time blocks are not 3 independent study participants; accuracy near chance and minority class shift limit generalization claims.
4. FFSEEG residual Top-K keeps 14-node processing with masked inputs, so no claim of actual graph computation reduction or electrode-count saving without separate measurement.
5. This is an exploratory pipeline comparison on previously inspected splits, not independent confirmatory testing; the user should NOT rework numerical results to reach target metrics.

## Files

Each workflow Artifact contains:
- \`outputs/results/all_candidate_results_exploratory.csv\`
- \`outputs/results/validation_chosen_fold_results.csv\`
- \`outputs/results/validation_chosen_oof_predictions.csv\`
- \`outputs/results/comparison_summary.csv\`
- \`outputs/results/partition_audit.csv\`
- \`outputs/results/metadata.json\`
- \`training.log\`, \`input_sha256.txt\`, \`SHA256SUMS.txt\`

No manuscript performance-superiority claim is currently justified. Recommended next objective: independently validated lower-electrode or lower-latency design, or a new genuinely unconfounded multi-subject task.
