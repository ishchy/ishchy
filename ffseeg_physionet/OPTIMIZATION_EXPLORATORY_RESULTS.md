# Exploratory FFSEEG sparse Top-K optimization (2026-10-15)

**Status: exploratory; not an independent confirmatory validation.**

## Data and protocol

- Immutable real EEG source: GitHub Actions run 38539853059 attempt 2, artifact ID 11911025625.
- 109 PhysioNet EEGMMIDB subjects, 6,538 windows, original 10 subject-held-out folds, disjoint train/validation/test subjects.
- 14 relative-alpha channel features; R01 open-eyes / R02 closed-eyes task is confounded with recording run. Do not report this result as unconfounded eye-state detection.
- Top-K mask and MI initialization use training-only signals, validation-only early stopping, no outer-test labels in fitting.

## New model

A straight-through global Top-K gate (K=8) and two message-passing layers are optimized jointly; the residual variant also preserves masked original channel values as a direct classifier input. This replaces the previous two-stage soft-gate-to-hard-head training protocol. Both variations remain investigational.

## Full GitHub Actions follow-up: run 38683554755

| Variant | Seed | Mean 10-fold BA | F1 | AUC |
|---|---:|---:|---:|---:|
| Residual Top-K GNN | 24701 | 75.2118% | 73.4703% | 82.2498% |
| Residual Top-K GNN | 24702 | 75.1954% | 73.3421% | 82.5134% |
| Residual Top-K GNN | 24703 | 75.4061% | 73.6675% | 82.6029% |
| No-residual Top-K GNN | 24701 | 75.1301% | 73.4577% | 82.4218% |

Residual mean BA across three seeds: **75.2711%** (seed-to-seed SD 0.1172 percentage points).
The subject-wise mean of three residual predictions' BA compared with the fixed original FFSEEG-GNN shows +1.9725 pp (109-subject paired-bootstrap CI +1.0753 to +2.9001 pp; Wilcoxon p=0.000255). Against the original FullGraphConv, delta +0.4292 pp (CI -0.4952 to +1.3609 pp, p=0.5094).

Matched seed 24701 residual-minus-noresidual: +0.0612 pp (CI -0.7034 to +0.7951 pp; p=0.3946), **not significant**. The residual shortcut cannot be credited independently for the gain.

Every one of the 4 variants has 6,538 out-of-fold predictions over 109 subjects and 10 test folds; all output SHA-256 hashes verified. Earlier GitHub run 38682903145 reproduced a seed 24701 result of 75.1197% BA; repeat runs on different CPU runner environments are not bitwise identical, so tighten determinism before a paper-ready release.

## Interpretation

- The new end-to-end sparse selector achieves a modest, reproducible advantage over the *old FFSEEG-GNN* in these exploratory held-out folds.
- It has **not** demonstrated superiority over FullGraphConv. A nonlinear MI+SVM control is competitive.
- The architecture was developed with awareness of the previous held-out scores. Reusing the outer test partitions makes the reported p values descriptive/exploratory, not confirmatory.
- For an academic claim: pre-register architecture/hyperparameters, use a genuinely new held-out dataset with no run/label confound, compare all models with identical inputs, measure latency and channel reduction, and repeat multiple seeds before manuscript revision.

Links: [Latest full workflow run](https://github.com/ishchy/ishchy/actions/runs/38683554755) · [Draft PR](https://github.com/ishchy/ishchy/pull/1)
