# DAppSCAN_Training

A separate training study from `code/Training_Code` (the Slither-labelled runs).
Nothing here reads or writes the old tree. Code, data, outputs and logs all live
in this folder.

- **Dataset:** [InPlusLab/DAppSCAN](https://github.com/InPlusLab/DAppSCAN)
  (Zheng et al., IEEE TSE 50(6), 2024). SWC weaknesses were annotated by hand
  from 608 audit reports of real DApp projects, pinned at commit `66a56619`.
- **Models:** ModernBERT-large, SecureBERT 2.0, CodeLlama-7B, Qwen3-1.7B,
  OpenCoder-1.5B, OpenMythos-770M.
- **Task:** file-level multi-label classification over DAppSCAN dataset.
  Every model sees whole files through sliding windows.
