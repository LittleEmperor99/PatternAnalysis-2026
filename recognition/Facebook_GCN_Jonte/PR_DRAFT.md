# Local PR draft — not submitted

Target: `shakes76/PatternAnalysis-2026:topic-recognition` from
`LittleEmperor99/PatternAnalysis-2026:topic-recognition`.

**Proposed title:** Classify Facebook pages with a two-layer GCN and calibrated referral policy

## Proposed body

This Normal-difficulty project implements transductive Facebook page
classification with a 128–64–4 PyG GCN and a feature-only MLP baseline. It asks
whether graph context improves confidence reliability and automation coverage
enough to justify the additional computation. The modular implementation includes
pinned data loading/auditing, disjoint label roles, training/checkpointing,
label-free inference, explicit final-test release and reproducible evaluation.

On the same frozen 4,494-node test partition, GCN achieves 92.48% accuracy and
0.920009 macro-F1 versus MLP's 78.95% and 0.772748. Unchanged calibration and
referral rules accept 92.68% of test nodes at 4.80% observed accepted error for
GCN, versus 47.86% coverage at 4.18% for MLP. Confident incorrect predictions
remain; the report recommends supervised analyst assistance, not guaranteed
autonomous safety. It includes per-class metrics, OvR AUROC, t-SNE, resource
profiling, calibration comparisons and five genuine test failure autopsies.

The dataset is the existing 128-feature file linked by Appendix A item 6,
pinned by source revision and checksum after original PyG hosting failed.
Its precomputed SVD representation differs from the original PyG artifact;
this limitation is explicit. No datasets or weights are included in the current
tree. Reproduction commands recreate both models and their paired policy.

Validation includes analytic metric fixtures, real-data integrity/normalization
tests, label-role guards, checkpoint reloads, full frozen evaluation and
fresh-process inference. Final verification details are in the README and
submission checklist. AI assistance is disclosed in the README.

Methods/data references: [Kipf & Welling](https://arxiv.org/abs/1609.02907),
[Guo et al.](https://proceedings.mlr.press/v70/guo17a.html),
[t-SNE](https://www.jmlr.org/papers/v9/vandermaaten08a.html),
[UCI dataset](https://archive.ics.uci.edu/dataset/527/facebook+large+page+page+network)
and the [specification-linked artifact](https://github.com/gayanku/facebook_page-page_dataset).

## Before posting

- Confirm tutor feasibility check-off status; it is currently pending.
- Resolve/review the three small generated reports retained in historical
  commit `514067a`, although absent from the current tree.
- Stage 8 commit is approved; obtain separate explicit approval to push and create this PR.
