# Facebook Page Classification with a Two-layer GCN

COMP3710 individual Pattern Analysis project, Normal difficulty.

## Current status

Stage 1: repository and environment setup. This is an incremental report, not
a completed submission. Dataset loading, models, experiments, evaluation,
inference and feasibility evidence have not been implemented yet. No measured
classification results are available.

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
check-off arrangements. Stage 2 remains subject to approval.

## Sources consulted for setup

- Supplied `COMP3710_Report_2026_draft_v2.3.pdf`, especially Sections 2.1 and 3–6.
  Its cover says Version 2.2 and DRAFT; confirm the operative version with staff.
- PyTorch installation: <https://pytorch.org/get-started/locally/>.
- PyG installation: <https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html>.
- PyG dataset: <https://pytorch-geometric.readthedocs.io/en/2.6.1/generated/torch_geometric.datasets.FacebookPagePage.html>.
