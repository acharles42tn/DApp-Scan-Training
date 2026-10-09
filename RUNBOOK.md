# Runbook — DAppSCAN training on the TTU cluster

Everything below runs from **`$ROOT/code/DAppSCAN_Training`**, where
`ROOT=/work/projects/phd-acharles42/acharles42`. This folder is a sibling of
`Training_Code` and `comparison_study`, and it touches neither.

Same rules as the main runbook:
- GPU jobs go to `gpu-warp`.
- CPU jobs go to `batch-impulse`.
- Nothing that loads data runs on the login node.
- Check `squeue` after every submit. The web terminal has re-run commands on reconnect before.

---

## 1. Put the folder on the cluster (once)

1. In OnDemand **Files**, go to `$ROOT/code/` and upload `DAppSCAN_Training.zip`.
2. In the OnDemand shell:

```bash
ROOT=/work/projects/phd-acharles42/acharles42
cd $ROOT/code
unzip -q DAppSCAN_Training.zip          # creates DAppSCAN_Training/ (the zip can go afterwards)
cd DAppSCAN_Training
sed -i 's/\r$//' *.sh configs/*.yaml    # no-op unless a file was saved on Windows (CRLF trap)
chmod +x *.sh
mkdir -p logs outputs
```

The built dataset is already in `data/`, so training can start right away.

Models download into `$ROOT/.cache/huggingface` unless `DAPPSCAN_CACHE` says otherwise. To see which
cache already holds them (e.g. from the Slither runs):

```bash
ls -d $ROOT/.cache/huggingface/hub/models--* $ROOT/code/Training_Code/.cache/huggingface/hub/models--* 2>/dev/null
# if they are under Training_Code:   export DAPPSCAN_CACHE=$ROOT/code/Training_Code/.cache
```

## 2. Check the data (recommended, ~5 min, CPU)

```bash
# invariants: no label text left in the source, window labels, folds (needs the parquet)
srun --account=phd-acharles42 --partition=batch-impulse --mem=16G --time=00:15:00 \
     bash -c "spack load py-attrs 2>/dev/null; source $ROOT/SCenv/bin/activate && python tests/test_core.py"

# optional: rebuild from DAppSCAN itself and confirm the identical build
# (needs git >= 2.25 for sparse-checkout; if git doesn't know `sparse-checkout`, skip this --
#  the shipped parquet is the one to train on)
bash fetch_dappscan.sh                                   # login node: sparse clone, ~220 MB
sbatch SC_cpu.sh build --out data/rebuild/dappscan_v1.parquet
grep '"fingerprint"' data/dappscan_v1.meta.json data/rebuild/dappscan_v1.meta.json
# both must read 48a83c433189d0bd...
```

## 3. Floor and dry runs (CPU, minutes)

```bash
sbatch SC_cpu.sh baseline               # file-length-only floor -> outputs/length_baseline/
for m in modernbert securebert2 codebert codellama qwen3 opencoder textcnn openmythos; do
    sbatch SC_cpu.sh dryrun $m          # downloads the tokenizer, windows every split, no model, no GPU
done
squeue -u $USER
```

Each dry run writes `outputs/dryrun_<model>/windows_report.json`. Look for
`label_visibility: 1.0` in every split. For ModernBERT at 1,024 expect about
17,029 train, 5,696 val and 5,411 test windows. Any tokenizer or environment
problem surfaces here without spending a GPU slot.

## 4. GPU smoke tests (~10–30 min each)

These catch out-of-memory and loading problems before a multi-hour job.
Priorities: CodeLlama (memory), the Qwen3 and OpenCoder heads, and OpenMythos
(checkpoint loading + memory).

```bash
for m in modernbert securebert2 codebert codellama qwen3 opencoder textcnn openmythos; do
    sbatch SC.sh $m --max-samples 200 --epochs 1 --run-name smoke_$m
done
squeue -u $USER
```

The smoke test passed if the log shows `Consistency check passed` and ends with
`F1 (macro over 13 classes …)` and `Saved outputs/smoke_<m>/test_results.json`.
The number itself means nothing.
`collect` marks these runs `smoke test? yes`. If OpenMythos runs out of memory,
change its config to `batch_size: 2` and `gradient_accumulation_steps: 8`, which
keeps the effective batch at 16.

## 5. Full runs

```bash
for m in modernbert securebert2 codebert codellama qwen3 opencoder textcnn openmythos; do
    sbatch SC.sh $m
done
squeue -u $USER          # exactly 8 jobs
```

Rough wall time on one A100-40GB, with at most 10 epochs and early stopping:

| model | estimate |
|---|---|
| TextCNN | < 1 h |
| CodeBERT | 1–2 h |
| SecureBERT 2.0 | 1–2 h |
| ModernBERT-large | 2–3 h |
| Qwen3-1.7B | 3–6 h |
| OpenCoder-1.5B | 3–6 h |
| OpenMythos-770M | 4–8 h |
| CodeLlama-7B | 10–16 h |

Each job ends by predicting the train, val and test windows (for the MoE).

## 6. MoE and the results table (CPU)

```bash
# after modernbert, securebert2, codellama and openmythos have finished:
sbatch SC_cpu.sh moe                     # experts listed in configs/moe.yaml
sbatch SC_cpu.sh collect                 # -> outputs/RESULTS.md
```

`collect` warns if runs come from different dataset builds. It also keeps the
leakage diagnostic (F1 on test files with no near-duplicate in train) in its
own table.

## 7. Optional: other folds (cross-validation)

```bash
sbatch SC.sh modernbert --test-fold 1 --val-fold 2 --run-name modernbert_f1
sbatch SC.sh modernbert --test-fold 2 --val-fold 3 --run-name modernbert_f2
# ... one run per fold; each is a complete, independent train/val/test
```

---

## Checklist before trusting a number

1. **Right environment.** The first lines of `logs/slurm-<job>.out` show the python path and the torch/transformers versions: SCenv (torch 2.5.1 / transformers 5.5.4) for everything except OpenMythos, which uses the openmythos env (torch 2.11 / transformers 5.7.0).
2. **Same dataset build.** `dataset.fingerprint` in `test_results.json` matches `data/dappscan_v1.meta.json` for every run in the table.
3. **Not a smoke run.** `dataset.max_samples` is `null`.
4. **Consistency check passed.** Each run re-scores validation after training and stops if the kept model doesn't reproduce the score it was selected with (which is what misaligned predictions look like). The log line is `Consistency check passed`.
5. **The F1.** Use the `f1` field: macro over the 13 classes, file level, thresholds tuned on val. Everything under `diagnostics` and `metrics` is secondary.
6. **Compare against the floor.** Read every model against the length baseline (F1 about 0.021), not against 0.
7. **`squeue` shows exactly the jobs you meant to submit.**

## Where things are

| what | where |
|---|---|
| dataset + card | `data/dappscan_v1.parquet`, `data/DATA_CARD.md`, `data/dappscan_v1.meta.json` |
| near-duplicate pairs / excluded files | `data/dappscan_v1.neardup.csv`, `data/dappscan_v1.excluded.csv` |
| per-run outputs | `outputs/<run>/` (config, final_model, test_results, thresholds, logits) |
| logs | `logs/slurm-<job>.out`, `logs/<run>_<timestamp>.log` |
| HF model cache | `$ROOT/.cache/huggingface` by default. Models not found there download once (~25 GB in total). To reuse another cache that already holds them, `export DAPPSCAN_CACHE=<dir containing huggingface/hub>` before `sbatch` |
| OpenMythos init checkpoint | `$ROOT/experiments/2026-05_swc_pretrain/checkpoints/swc_770m_v1/step_00012500_final.pt` (`mythos_init_ckpt`) |

---

## v2 — file-level retrain with TF-IDF blend and 5-fold CV (added 2026-10-05)

**Why.**
- In v1, every transformer peaked in its first epochs and then over-fit.
- A plain TF-IDF + logistic regression on whole files beat all of them: fold 0 F1 0.058 vs 0.031 for the best transformer, and 0.057 ± 0.006 over five folds against a floor of 0.036.
- Rebalancing the training data made no difference in quick tests.

So v2 changes how the models are trained. The data, the folds and the F1 stay the same.

| | v1 | v2 |
|---|---|---|
| what a training example is | one window, labelled from the annotated lines it overlaps | one **file**, labelled with the file's classes |
| file score | max over windows (favours long files) | per-class **attention pooling** over the file's windows |
| over-fitting control | full fine-tune (encoders) / LoRA on all layers | bottom half frozen (encoders) / LoRA on the top half only; lower LR; separate head LR; dropout |
| epoch kept / early stopping | best val macro-AP (noisy) | lowest **val loss** (smooth), patience 2, ≤ 8 epochs |
| balancing (train only) | windows: all positive, hard negatives, 30% of easy | files: all labelled, a fresh 30% of unlabelled each epoch, rare classes duplicated (≤ 5×); BCE with capped per-class `pos_weight` |
| partner model | — | TF-IDF + logistic regression (`scvd_dapp tfidf`), blended per fold (`scvd_dapp blend`, weight picked on val) |
| folds | fold 0 only | all five (test = k, val = k+1) |

Code: `scvd_dapp/filelevel.py`, `tfidf.py`, `blend.py`, `collect.py:collect_cv`;
configs `configs/*_v2.yaml`; tests `tests/test_v2.py`. v1 code paths are unchanged
(`training_unit: window` is still the default).

```bash
cd $ROOT/code/DAppSCAN_Training
python tests/test_v2.py               # 7 tests, seconds (login node is fine: no dataset load)
bash submit_v2.sh pilot               # fold 0: ModernBERT + SecureBERT 2.0, TF-IDF, floor, blends (~1 h)
# check the pilot: outputs/v2/{modernbert,securebert2,tfidf,length,modernbert_tfidf,securebert2_tfidf}_f0
bash submit_v2.sh full                # all five models x five folds (+ TF-IDF, floor, blends per fold)
sbatch SC_cpu.sh cvsummary            # -> outputs/v2/RESULTS_v2.md (mean ± sd over folds)
```

`submit_v2.sh` is safe to re-run. It skips queued job names and finished runs. It requests memory sized to the model (32/40/64 GB) instead of SC.sh's 120 GB, which kept the v1 jobs waiting. Blends wait for their model and the TF-IDF run (`--dependency=afterok`).

Each v2 log prints one line per epoch:
`EPOCH n | train loss | val loss | val file macro-AP | val F1-macro@0.5`.
After training, the run re-scores validation and stops unless the restored weights reproduce the selected epoch's val loss.

## v3 — balanced table, all SWC types, Slither-initialised option (added 2026-10-08)

**Why.** The advisor asked for the best scores the data allows, with three changes on top of v2:
cut the files with no annotated weakness drastically, bring back the SWC types that v1/v2 left
out, and oversample the labelled files more. The Slither-trained weights are tried as a starting
point too.

| | v2 | v3 |
|---|---|---|
| table | `data/dappscan_v1.parquet`: 9,424 files, 731 labelled, 13 types | `data/v3/dappscan_v3.parquet`: 1,836 files, 918 labelled + 918 unlabelled |
| unlabelled ("safe") files | all kept in val/test; 30% sampled per epoch in train | **1 per labelled file in every fold, test included** (`scvd_dapp balance`); like Slither's `balance_secure`, which kept 30% |
| SWC types | 13 (≥ 20 files; SWC-102/103 and 19 rare types left out) | **all 34 trained**; F1 over the **15** with ≥ 20 files (the 13 + SWC-102, SWC-103) |
| oversampling (train) | types with < 50 files → toward 150 (≤ 5×) | types with < 100 files → toward 200 (≤ 5×) |
| starting weights | public base model | both: base (`<model>_v3`) and the Slither-trained `Training_Code/outputs/<model>/final_model` (`<model>_v3s`; decoders: the LoRA adapter merged first); the better one on fold-0 **validation** goes on |

Everything else (file-level training, frozen bottom half / top-half LoRA, early stopping on
val loss, TF-IDF blend, F1 procedure) is v2. Scores on the balanced table are higher than on the
natural test sets for every method, the length-only floor included, so always quote the floor
from the same table. CPU preview on all five folds: TF-IDF F1 0.130 ± 0.013, length floor 0.100 ± 0.018.

The two tables were built by:

```bash
python -m scvd_dapp build --raw data/raw/DAppSCAN --out data/v3/dappscan_v3_all.parquet \
       --min-class-files 1 --keep-project-level --eval-min-files 20      # 9,492 files, 34 types
python -m scvd_dapp balance --in data/v3/dappscan_v3_all.parquet \
       --out data/v3/dappscan_v3.parquet --safe-ratio 1.0                  # 1,836 files
```

**Where.** v3 lives in its own folder, `$ROOT/code/DAppSCAN_Training_v3`, a complete copy of the
package. `$ROOT/code/DAppSCAN_Training` belongs to the mixed-dataset (Slither + DAppSCAN) work, which
changes some of the same files; on 2026-10-08 an `unzip -o` of the v3 update into that folder
overwrote 9 of them (the fix re-extracts them from `DAppSCAN_Training_v3_mix_update.zip`). Never
unzip one experiment's package over another's folder. The model cache (`$ROOT/.cache`) and the Slither
checkpoints are shared and only read.

Set up (once): upload `DAppSCAN_Training_v3.zip` to `$ROOT/code/` in OnDemand Files, then

```bash
cd $ROOT/code && unzip -n -q DAppSCAN_Training_v3.zip     # -n: never overwrite; creates DAppSCAN_Training_v3/
```

Run (from `$ROOT/code/DAppSCAN_Training_v3`):

```bash
python tests/test_v3.py && python tests/test_v2.py     # seconds; backbone tests skip without the tiny models
bash submit_v3.sh pilot      # fold 0: 5 models x {base, Slither init} + TF-IDF + floor + blends
# pick, per model, the variant with the higher fold-0 VALIDATION macro-AP (file_level.history in
# outputs/v3/<cfg>_f0/test_results.json), then:
bash submit_v3.sh full modernbert_v3s securebert2_v3 ...   # folds 1-4 for the chosen configs
sbatch SC_cpu.sh cvsummary --outputs ./outputs/v3          # -> outputs/v3/RESULTS_v3.md (F1, AP, P, R)
```
