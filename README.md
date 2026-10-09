# DAppSCAN_Training — the same models, retrained on DAppSCAN

A separate training study from `code/Training_Code` (the Slither-labelled runs).
Nothing here reads or writes the old tree. Code, data, outputs and logs all live
in this folder.

- **Dataset:** [InPlusLab/DAppSCAN](https://github.com/InPlusLab/DAppSCAN)
  (Zheng et al., IEEE TSE 50(6), 2024). SWC weaknesses were annotated by hand
  from 608 audit reports of real DApp projects, pinned at commit `66a56619`.
- **Models:** ModernBERT-large, SecureBERT 2.0, CodeLlama-7B, Qwen3-1.7B,
  OpenCoder-1.5B, OpenMythos-770M, the CodeBERT and TextCNN baselines, and the
  MoE over the experts.
- **Task:** file-level multi-label classification over 13 SWC classes.
  Every model sees whole files through sliding windows.

How to run it on the cluster: **RUNBOOK.md**. What is in the data: **data/v3/DATA_CARD.md**.

### Getting the data

The built tables are not in this repository. They contain the DAppSCAN contracts' source code,
and DAppSCAN has no license file. Rebuild them (CPU, a few minutes; on the cluster run the
`build` and `balance` steps through `sbatch SC_cpu.sh`):

```bash
bash fetch_dappscan.sh     # DAppSCAN at the pinned commit 66a56619 -> data/raw/DAppSCAN
python -m scvd_dapp build --raw data/raw/DAppSCAN --out data/v3/dappscan_v3_all.parquet \
       --min-class-files 1 --keep-project-level --eval-min-files 20      # 9,492 files, 34 types
python -m scvd_dapp balance --in data/v3/dappscan_v3_all.parquet \
       --out data/v3/dappscan_v3.parquet --safe-ratio 1.0                  # 1,836 files
python -m scvd_dapp build --raw data/raw/DAppSCAN --out data/dappscan_v1.parquet   # v1/v2 table
```

Each table's fingerprint is recorded in its `.meta.json` (v3: `2dac42463f25…`, balanced v3:
`c827a997aef1…`, v1: `48a83c433189…`); a rebuild should reproduce it.

---

## Why the data needed work before training

Numbers are for the pinned commit. The cleaning invariants (no label text left, spans, folds) are
re-checked by `tests/test_core.py`.

| found in raw DAppSCAN | what the builder does |
|---|---|
| **Labels are written into the source.** All 948 annotated files have comments like `// SWC-107-Reentrancy: L399-422` (1,646 annotations, 1,647 comments). A model trained on the raw text learns to read the answer. | Strips the comments. Canonicalises the layout of *every* file: tabs → 4 spaces, trailing whitespace and blank lines removed. Removing a comment line alone would still leave a gap exactly at the flagged code, because annotators sometimes inserted the comment on a new line and sometimes typed it over a blank one. Annotated line numbers are remapped. Check: after cleaning, all 70 annotated files that also exist unannotated in DAppSCAN are byte-identical to that copy, and 18 of 19 randomly sampled files are identical to upstream GitHub at the audited commit. The 19th differs by two code lines, a version difference rather than a label trace. |
| **30.8% of files are duplicates.** For example, Synthetix was audited about 23 times. In 65 cases the same code is flagged in one audit and unflagged in another. | Merges exact duplicates (after normalisation) and unions their labels. Splits by **codebase** (the repository in the xlsx), so re-audits never straddle train and test. Codebases that share near-duplicate files are merged too, up to a size cap. Remaining near-duplicate exposure: 3.9% of test files. It is reported as a separate diagnostic. |
| **SWC-102 and SWC-103 are project-level.** In projects where the floating pragma is flagged, only 7.5% of the files that also use a floating pragma carry the label. | Leaves both classes out (`--keep-project-level` puts them back). |
| **19 classes are too rare to evaluate** (fewer than 20 files). | Leaves them out. Files annotated *only* with those classes are excluded, because they are known-vulnerable and can't serve as negatives. |
| **Tests and mocks are never annotated** (0 of 3,726 test/mock files); interface-only files have no code bodies. | Drops unannotated test, mock and interface-only files (the 4 annotated interfaces are kept). |
| **Unannotated doesn't mean secure.** The authors only mapped SWC-classifiable findings. | The label is "no SWC weakness annotated". The file-length baseline gives the floor. |

Result: **9,424 files** (731 with ≥1 class), **13 classes**, **5 grouped and
stratified folds**. Every class has 4 to 42 positives in every fold. The
default split is test = fold 0, val = fold 1, train = folds 2 to 4 (60/20/20).

## What changed vs the Slither pipeline (`scvd`), and why

| | Slither study (`scvd`) | this study (`scvd_dapp`) |
|---|---|---|
| labels | Slither detector output, 39 classes | auditors' SWC annotations, 13 classes (the class list is read from the dataset build, never hardcoded) |
| long files | first `max_length` tokens only | **sliding windows** over the whole file. A window is positive for a class only if it overlaps an annotated line range. File score = max over windows. With head truncation at 1,024 tokens, only 39% of annotated line ranges would start inside what the model sees (80% at 4,096) |
| split | random 70/15/15 | fixed 5 folds, grouped by codebase and stratified per class; any fold can be the test fold (`--test-fold`) |
| resampling | `balance_secure` before the split (thinned val/test too) | train split only: all positive windows, all negative windows of annotated files, 30% of windows from unannotated files, rare classes oversampled at most 5× |
| model selection | val F1-micro @0.5 | val file-level macro average precision (threshold-free) |
| epochs | 5 | up to 10, early stopping (patience 3) |
| context | 1,024 (ModernBERT, SecureBERT 2.0, CodeLlama) vs 512 (Qwen3, OpenCoder), a confound | 1,024-token windows for every model except CodeBERT (512 is its positional limit) |

Model hyperparameters (learning rate, batch size, LoRA setup, ASL loss) are
carried over from `Training_Code/configs/` unchanged. OpenMythos has no scvd
config; its settings are in `configs/openmythos.yaml` (see the caveat below).

**F1** (the only F1 reported) = macro-averaged F1 over all 13 classes at file
level, with per-class thresholds tuned on validation. Every class has test
support, so "all classes" and "supported classes" give the same number.

Before writing any result, every run re-scores validation with the model it
kept and stops unless that reproduces the score the model was selected with.
This guards the window → file mapping, which a reordered prediction batch would
silently scramble.

**Expect much lower F1 than the Slither table.** Per-class base rates in test
are 0.2% to 2.3% of files, and the labels are human findings, not a detector's
own output. The file-length-only baseline scores **F1 0.021** (macro-AP 0.024)
and is the floor every model should be read against.

**OpenMythos caveat.** OpenMythos has no public pretrained weights. Its
backbone starts from the project's Slither-trained checkpoint (`swc_770m_v1`,
step 12,500) with a fresh 13-class head. The HF models start from their public
pretrained weights instead. Report this next to its number. Its learning rate
(2e-5) and loop count (4) are placeholders until checked against
`Open-Mythos/train_classifier.py`.

## v2 (October 2026): file-level training

v1 trained on windows and scored each file by its best window. The transformers over-fit early, and a
plain TF-IDF model beat them all. v2 trains on whole files, pooling each file's windows with per-class
attention, and regularises harder: frozen lower layers or top-only LoRA, early stopping on validation
loss. Each transformer is blended with the TF-IDF model, and all five folds are run. Same data and same
F1 as v1. See RUNBOOK "v2" and `submit_v2.sh`.

## v3 (October 2026): balanced table, all SWC types

At the advisor's request, v3 trains for the best scores the labels allow:
- **Safe files cut:** every fold keeps all labelled files plus one unlabelled file per labelled file, test sets included.
- **All 34 SWC types trained:** F1 is reported over the 15 types with at least 20 files.
- **More oversampling** of rare types.
- **Two starting points:** each model is tried from its public weights and from its Slither-trained weights. The variant that does better on validation is kept.

Scores are higher than v2's for every method, the length-only floor included, so quote the floor from the same table. See RUNBOOK "v3".

v3 runs from its own folder on the cluster, `code/DAppSCAN_Training_v3`. `code/DAppSCAN_Training` is used by the mixed-dataset work.

## Layout

```
DAppSCAN_Training/
├── scvd_dapp/              the package  (python -m scvd_dapp build|train|baseline|moe|collect)
│   ├── build_dataset.py    raw DAppSCAN -> data/dappscan_v1.parquet (+ meta, card, near-dup pairs)
│   ├── taxonomy.py         class list, read from the dataset meta
│   ├── windows.py          tokenize once, cut windows, label windows from annotated lines
│   ├── data.py             folds, train-only resampling, collator
│   ├── models.py           HF encoder / decoder+LoRA (from scvd)
│   ├── mythos.py           OpenMythos classifier (mirrors run_openmythos.py)
│   ├── textcnn.py          TextCNN baseline
│   ├── train.py            orchestrator (HF Trainer; native loop for TextCNN/OpenMythos)
│   ├── evaluate.py         file-level report: F1, per-class table, thresholds, logits
│   ├── baselines.py        file-length-only floor
│   ├── moe.py              gate over finished runs (scvd's gating network)
│   └── collect.py          outputs/RESULTS.md
├── configs/                one YAML per model + moe.yaml
├── data/                   dappscan_v1.parquet, .meta.json, .neardup.csv, .excluded.csv, DATA_CARD.md
├── tests/test_core.py      invariants (no label text, window labels, folds)
├── SC.sh                   GPU job (picks SCenv or the openmythos env from the config)
├── SC_cpu.sh               CPU jobs on batch-impulse (build, baseline, dryrun, moe, collect)
├── fetch_dappscan.sh       sparse clone of DAppSCAN at the pinned commit (~220 MB)
└── RUNBOOK.md
```

Each run writes `outputs/<run>/`:
- `config.json`
- `taxonomy.json`
- `final_model/` (safetensors)
- `test_results.json`
- `thresholds.json`
- `logits_{train,val,test}.npz` (file-level; the MoE is fit on these)
