# Facebook Page Classification with a Two-layer GCN

COMP3710 individual Pattern Analysis project, Normal difficulty.

## Current status

Stage 1 is committed as `419b7db` (not pushed). Stage 2 loading, auditing and
splitting have passed synthetic and real-data tests and await student review.
The student explicitly approved the specification-linked, precomputed 128-feature
file after the original PyG download failed. Its different representation and
provenance are documented below. This is not a completed submission. No task
models, training, classification results or final feasibility review exist yet.

The selected project compares a feature-only MLP with a two-layer graph
convolutional network for transductive Facebook page classification. The planned
investigation asks whether graph context improves classification and confidence
reliability, and how referring uncertain predictions changes error and coverage.
These are planned methods, not findings.

Internal Phase 1 completion target: **20 October 2026**. The student-provided
assessment due date is 30 October 2026; verify the exact official deadline and
the separate compulsory midpoint consultation with teaching staff.

## Repository layout and development workflow

Develop inside `recognition/Facebook_GCN_Jonte/` on the personal fork's
`topic-recognition` branch. `upstream` refers to
<https://github.com/shakes76/PatternAnalysis-2026>; `origin` must refer to the
student's fork. The final PR targets upstream `topic-recognition`.

The assessment requires `modules.py`, `dataset.py`, `train.py`, `predict.py`
and this README. Add each implementation file in its approved stage, rather
than adding empty placeholders. Review and test each milestone before an
explicitly authorised commit. There are no automatic commits.

Datasets, processed caches, model weights, environments and generated experiment
files are excluded by the root `.gitignore`. Curated report figures can later
live in `figures/` and be committed. Never add excluded files with `git add -f`.

## Windows environment setup

Run the following from the repository root in PowerShell. Use the explicit
interpreter path so activation and PowerShell execution-policy changes are not
required. The CUDA build is intended for the verified local NVIDIA GPU.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r recognition/Facebook_GCN_Jonte/requirements-torch.txt
.\.venv\Scripts\python.exe -m pip install -r recognition/Facebook_GCN_Jonte/requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe recognition/Facebook_GCN_Jonte/verify_environment.py --require-cuda
```

The direct dependencies are pinned in `requirements.txt`.
`requirements-lock.txt` records exact installed transitive versions
for this Windows/Python environment. To reproduce that environment, install
`requirements-torch.txt` first, then `requirements-lock.txt` instead of the direct
dependency file. Other operating systems may need different PyTorch wheels.

In VS Code, use **Python: Select Interpreter** and choose
`.venv/Scripts/python.exe`. A default workspace setting is provided; an existing
VS Code interpreter selection may override that default.

## Stage 1 verification

The following automated checks passed on 17 October 2026:

| Check | Actual outcome |
| --- | --- |
| GitHub repository metadata | `LittleEmperor99/PatternAnalysis-2026` is a fork of `shakes76/PatternAnalysis-2026` |
| Branch and tracking | Local `topic-recognition` tracks `origin/topic-recognition` |
| Starting revision | Local, origin and upstream topic branches all at `552b9029a3fbee3e12d0b4e95f38e73d8d1d51a5` |
| Isolated interpreter | `.venv/Scripts/python.exe`, Python 3.13.14, Windows x64 |
| PyTorch / PyG | `2.8.0+cu126` / `2.6.1` |
| Other direct packages | NumPy 2.2.6; scikit-learn 1.7.2; Matplotlib 3.10.7 |
| Dependency consistency | `python -m pip check`: no broken requirements |
| CPU operations | Tensor multiplication, GCNConv forward and finite gradients passed |
| GPU operations | The same checks passed on RTX 2060 SUPER with CUDA runtime 12.6 |
| Interoperability and plotting | scikit-learn metric, NumPy/PyTorch conversion and in-memory PNG rendering passed |
| Git exclusions | Representative datasets, caches, weights and output paths ignored; report figures allowed |

The environment script uses tiny artificial tensors exclusively for software
checks. It does not train a task model, use the Facebook dataset, or produce
experimental evidence. GPU success here is not a measured full-graph memory or
training-time estimate.

GitHub CLI (`gh`) was unavailable, so repository setup used Git and GitHub's
public repository metadata after the student supplied the fork URL. No commit,
push or PR was created; push authentication has not been tested. An initial
inspection of the existing global Python packages encountered malformed package
metadata; the isolated environment passed its own package checks.

The student must still select the VS Code interpreter, review the setup and
code, verify their Git author identity, and confirm the official feasibility
check-off arrangements. The student subsequently approved Stage 1 and explicitly
authorised its commit and Stage 2 development. No push has been made.

## Stage 2: loading, audits and fixed splits

The setting is **transductive node classification**. Every node's descriptive
features and every graph edge remain available, including edges across splits.
This does not test generalisation to unseen graphs or future pages.

The original `FacebookPagePage` download host fails DNS resolution. Appendix A,
item 6 of the supplied PDF links to
<https://github.com/gayanku/facebook_page-page_dataset>; on 17 October 2026 the
student explicitly approved that existing file with its provenance disclosed.
It contains **precomputed 128-dimensional TruncatedSVD features**, not a verified
copy of GraphMining's original embeddings. The source's feature-generation code
uses all node descriptions without labels. Using existing unlabelled features
of all nodes is consistent with the declared transductive setting; this does not
establish an inductive evaluation or equivalence to published PyG benchmarks.

`dataset.py` downloads the immutable source revision
`ec01d02c39da4385052023ed28e36f0794f61dc8`, verifying SHA256
`7a1b8a0687515e9859d1ca649dc276825f5db44568894dfbb6b55ebc1cf6a9c1`
on **every load**. An offline copy must match the same pinned bytes. There is
no automatic source fallback. The file's numeric keys `X`, `y`, `edges_x` and
`edges_y` are adapted to a native `torch_geometric.data.Data` object. The built-in
`FacebookPagePage` loader cannot directly read this schema. No new embedding is
fitted and feature values are preserved. NumPy pickle is disabled; the optional
`page_name` object array is never read. No processed pickle cache is needed.

The loader checks dimensions, finite features, integer targets, edge bounds and
reciprocity, and writes a source/checksum manifest. Supplied edges and self-loops
are preserved. The audit reports **342,004 edge entries**, including 358 self-loop
entries and 179 duplicate ordered entries: reversing a self-loop duplicates it.
There are 171,002 unique unordered pairs including loops and zero isolated nodes.
These counts describe the source; no graph repairs are applied silently.

Final Stage 2 verification also checks PyG 2.6.1's unweighted COO `gcn_norm`
operator with its default self-loop addition and normalization. It replaces
the supplied loops with one self-loop per node: 22,470 loops and 364,116
normalized edge entries. The resulting normalized sparse matrix matches the
matrix obtained from deduplicated input. Thus the documented duplicate loops
do not double self-messages under these specific defaults. Other operators,
explicit edge weights, or disabling self-loop handling require a fresh check.
This is an operator verification, not a GCN implementation or experiment.

Observed features are float32 with shape `(22470, 128)`, all finite, ranging
approximately from -2.037593 to 3.002028. Independent comparison with the raw UCI
release confirms every unique directed edge and every label by node ID. The
actual class mapping is alphabetical and corrects the source README's mapping:

| ID | Class | Nodes |
| --- | --- | ---: |
| 0 | company | 6,495 |
| 1 | government | 6,880 |
| 2 | politician | 5,768 |
| 3 | tvshow | 3,327 |

Split seed **42** is fixed independently of later model training seeds.
Stratified splitting uses scikit-learn 1.7.2 with local `random_state` values;
it does not change global NumPy/PyTorch random states. Actual allocations for
22,470 nodes are:

| Role | Nodes | Purpose |
| --- | ---: | --- |
| Training | 13,482 | Fit task-model weights |
| Validation, total | 4,494 | Union of the three roles below |
| Selection | 2,247 | Early stopping and model selection |
| Calibration | 1,123 | Later confidence calibration |
| Policy | 1,124 | Later rejection-threshold selection |
| Test | 4,494 | Final evaluation after the protocol is frozen |

Per-class split counts (company, government, politician, tvshow) are:

| Partition | Company | Government | Politician | TV show |
| --- | ---: | ---: | ---: | ---: |
| Train | 3,897 | 4,128 | 3,461 | 1,996 |
| Validation, total | 1,299 | 1,376 | 1,153 | 666 |
| Selection | 649 | 688 | 577 | 333 |
| Calibration | 325 | 344 | 288 | 166 |
| Policy | 325 | 344 | 288 | 167 |
| Test | 1,299 | 1,376 | 1,154 | 665 |

The five distinct partitions are exhaustive and disjoint; the combined validation
mask deliberately overlaps its own three subpartitions. Stratification and
descriptive auditing use labels to allocate and count classes. No task model is
trained by the dataset code.

The returned `graph` contains only `x`, `edge_index` and `num_nodes`, with **no
labels**. Labels are separate vectors in boolean-mask order, obtained through
`targets("train")`, `targets("selection")`, etc. `targets("test")` raises unless
`allow_test=True` is explicitly supplied. Combined `val` targets are rejected to
avoid accidentally using calibration/policy labels for checkpoint selection.
This API reduces accidental leakage; it is not a security boundary. Later
training code must still be inspected and tested for correct label use.

Download/load and audit the approved file, then run tests from the repository
root. The dataset download uses a 30-second socket timeout. An existing local
copy of the pinned file can be imported with `--raw-file <path>` instead.

```powershell
.\.venv\Scripts\python.exe recognition/Facebook_GCN_Jonte/dataset.py --audit-output recognition/Facebook_GCN_Jonte/outputs/dataset_audit.json
.\.venv\Scripts\python.exe recognition/Facebook_GCN_Jonte/test_dataset.py -v
```

Latest result on 17 October 2026: **all 28 tests passed: 22 synthetic tests and
6 real-data integration/operator tests, with no skips** in the tested local environment.
Real tests verify exact source bytes and unchanged numeric inputs, graph
integrity, repeatable stratified splits, target separation/mask order and an
independent raw-UCI edge/label comparison. The optional UCI comparison is skipped
on a clean machine unless its reference archive is downloaded; the other four
real tests run after the approved file is downloaded. Synthetic
coverage includes class balance, disjoint/exhaustive masks, seed repeatability,
unchanged global random states, label separation, explicit test-target access,
malformed input rejection, checksum/corruption detection, failed-download cleanup
and offline loading/reloading with artificial fixtures. All checks concern the
data pipeline; they are not task-model performance results.

A separate fresh-root download from the pinned URL also succeeded and matched
the same SHA256. The default-root audit command succeeded and wrote the actual
dataset statistics and mask hashes to the ignored output JSON.

To reproduce the independent comparison, download the pinned raw reference:

```powershell
New-Item -ItemType Directory -Force recognition/Facebook_GCN_Jonte/data/reference | Out-Null
Invoke-WebRequest -Uri 'https://archive.ics.uci.edu/static/public/527/facebook+large+page+page+network.zip' -OutFile recognition/Facebook_GCN_Jonte/data/reference/uci-facebook.zip
.\.venv\Scripts\python.exe recognition/Facebook_GCN_Jonte/test_dataset.py -v
```

The comparison test verifies the reference archive's checksum before using it.
Generated JSON, downloaded files and Python caches remain excluded from Git;
audit tables and provenance are included here as documentation. See
[DATA_PROVENANCE.md](DATA_PROVENANCE.md) for exact source details and the original
download investigation. No additional dependency was needed in Stage 2.

## Sources consulted for setup

- Supplied `COMP3710_Report_2026_draft_v2.3.pdf`, especially Sections 2.1 and 3–6.
  Its cover says Version 2.2 and DRAFT; confirm the operative version with staff.
- PyTorch installation: <https://pytorch.org/get-started/locally/>.
- PyG installation: <https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html>.
- PyG dataset: <https://pytorch-geometric.readthedocs.io/en/2.6.1/generated/torch_geometric.datasets.FacebookPagePage.html>.
