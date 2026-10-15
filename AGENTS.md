# AGENTS.md

Instructions for any AI coding agent (Claude Code, Codex, or similar) working in this
repository. Read this file fully before writing or editing any code.

## 1. What this project is

**Title:** Reliability-Aware Audio-Visual Fusion for Generalizable Deepfake Detection
**Course context:** SRM Report Writing (21CSC402P), undergraduate research project
**Team:** Aditya Sahu, Akash Cherian, Prachurya Subhasmita

This is an audio-visual deepfake **detector**: given a short video clip with audio,
predict whether it has been manipulated. The specific research contribution is a
**reliability estimator** — a small module that looks at per-window modality-quality
signals, unimodal-head confidence, and audio-visual synchronization, and outputs a
softmax weighting `(w_v, w_a)` over the visual and audio branches before fusion,
instead of fusing them with fixed/equal weights.

**The full technical specification lives in `docs/PROJECT_BRIEF.md`. That document is
the canonical source of truth for architecture, math, datasets, splits, training
stages, evaluation protocol, and the 12-week timeline. This file (AGENTS.md) tells you
*how to work in the repo*; `docs/PROJECT_BRIEF.md` tells you *what to build*. If the
two ever conflict, `docs/PROJECT_BRIEF.md` wins — flag the conflict instead of
silently picking one.**

Every file in this repository currently contains only a one-line docstring/comment
stating its purpose. Nothing is implemented yet. Your job is to fill these in, in the
order given in Section 4 below, without changing the module boundaries the skeleton
already encodes (each file's docstring tells you exactly what belongs in it and
nothing else).

## 2. Non-negotiable constraints

These come directly from `docs/PROJECT_BRIEF.md` and must not be relaxed for
convenience, even under a tight deadline:

- **No dataset redistribution.** Code must read licensed datasets (FakeAVCeleb, DFDC)
  from a local path the user provides; never download, scrape, or bundle actual video
  files into the repo, and never write code that would upload/share them.
- **No claims beyond what is tested.** Do not write comments, docstrings, log
  messages, or report text asserting state-of-the-art performance, "first" status, or
  that a hypothesis (H1–H4 in the brief) is supported before it has actually been
  evaluated. If a result is negative or a gate collapses, that is a reportable finding,
  not a bug to hide.
- **Leakage-resistant splits are mandatory, not optional.** Every training script must
  consume manifests from `manifests/splits/`, never re-split data ad hoc inside a
  training loop. Source-identity groups must never cross train/val/test. Test-only
  degradation recipes must never appear in training.
- **Synchronization is a feature, not a verdict.** Never implement logic that labels a
  clip fake purely because audio-visual offset is nonzero, or real because it is
  zero — `sync_features.py` output feeds the reliability estimator and classifier as
  one input among several, per `docs/PROJECT_BRIEF.md` Section 7.3.
- **Validation set is for thresholds/temperature only.** Never select a final model or
  report a headline number based on the test set; test is touched once, at the end.
- **Reproducibility checklist (Section 13 of the brief)** must be kept satisfiable at
  every stage: seeds logged, configs saved with every checkpoint, split manifests
  versioned, raw/calibrated/optimized metrics kept distinct.

## 3. Repository map

```
docs/PROJECT_BRIEF.md   Canonical spec — architecture, math, datasets, protocol, timeline
data/                    Local, gitignored dataset root (raw/, processed/, cache/)
manifests/               Versioned split + source-table CSVs (small, always committed)
configs/                 YAML configs for data/model/train/eval — fill in fields, keep code config-driven
src/decode/              Raw video/audio decoding
src/preprocess/          Face/audio feature extraction, quality features, alignment, augmentation
src/models/              Encoders, projector, sync features, reliability estimator, fusion, full model
src/train/               Losses + staged training scripts (unimodal -> fixed fusion -> reliability fusion -> distillation)
src/evaluate/            Metrics, calibration, degradation suite, cross-manipulation/cross-dataset eval, efficiency
src/utils/               Manifest building, splitting, hashing, seeding, logging, config loading
tests/                   Unit tests — see Section 5, write these alongside the code they test
notebooks/               Manual audit notebook(s)
reports/                 Generated output (gitignored except final figures)
artifacts/               Trained checkpoints (gitignored)
scripts/                 CLI entry points that orchestrate the above into runnable stages
Makefile                 Stub targets mapping to the scripts/ CLI entry points
```

Every existing file's docstring is authoritative for that file's scope. If you find
you need functionality that doesn't obviously belong in any existing file, prefer
adding a new, clearly-named file over overloading an existing one — keep modules
small and single-purpose, matching the granularity already established.

## 4. Build order

Follow this order; do not jump ahead to model code before the data foundation is
solid, and do not skip tests. This mirrors the staged plan in
`docs/PROJECT_BRIEF.md` Sections 8 and 14.

1. **`src/utils/config_loader.py`, `src/utils/seed_utils.py`, `src/utils/logging_utils.py`**
   Small, foundational utilities everything else depends on.
2. **`src/decode/`, `src/preprocess/`** — decoding and feature extraction. Implement
   against the input contract in `docs/PROJECT_BRIEF.md` Section 6 exactly (fixed
   25 fps decode, 2s/16-frame windows, 160x160 face crops, 80-bin log-Mel, recorded
   quality features). Write `tests/test_alignment.py` and
   `tests/test_deterministic_decode.py` alongside this step.
3. **`src/utils/manifest_builder.py`, `src/utils/splitting.py`, `src/utils/hashing.py`**
   Build `manifests/source_table.csv` and the split manifests before any model code
   runs. Write `tests/test_split_exclusivity.py` and `tests/test_duplicate_hash.py`
   here — these tests must pass before proceeding.
4. **`src/models/visual_encoder.py`, `audio_encoder.py`, `alignment_projector.py`,
   `sync_features.py`, `modality_heads.py`** — the per-modality building blocks.
5. **`src/train/train_unimodal.py`** — get visual-only and audio-only baselines
   training and reporting metrics before touching fusion at all.
6. **Fixed fusion**: extend `src/models/fusion_classifier.py` with an equal-weight
   mode and get `src/train/train_fixed_fusion.py` working. This baseline must exist
   and be measured before the reliability gate is introduced (Section 8, staged plan
   step 3) — do not build the gate first "because it's the interesting part."
7. **`src/models/reliability_estimator.py`, `src/train/losses.py`** — implement the
   gate and `L_rel` exactly as specified in Section 7.2 (detached soft target,
   temperature `tau_r`, KL loss). Write `tests/test_gate_weights_sum.py` here.
8. **`src/models/full_model.py`, `src/train/train_reliability_fusion.py`,
   `src/train/trainer.py`** — wire the full system together with AMP, gradient
   accumulation, and checkpointing.
9. **`src/evaluate/`** — metrics, calibration, degradation suite, cross-manipulation
   and cross-dataset evaluation, efficiency profiling, reliability diagnostics. Write
   `tests/test_metrics.py` and `tests/test_missing_modality_mask.py` here.
10. **`src/train/distillation.py`** — only after step 9 produces a frozen, evaluated
    float model to distill from.
11. **`scripts/`** — CLI wiring last, once the underlying functions they call exist.

## 5. Testing expectations

- Every file under `tests/` already has a docstring naming exactly what it must
  verify — implement that test, not a different one.
- Tests must be able to run against small fixture clips in `tests/fixtures/`
  (permitted, non-dataset samples you or the user provide), never against the full
  licensed datasets.
- `make test` (→ `pytest tests/`) should be runnable at any point after step 2 above
  without requiring GPU access or the full dataset to be present.

## 6. Coding conventions

- Python 3.10+, type hints on public functions, docstrings on every module/class/
  function (extend the existing one-liners, don't delete them).
- No hardcoded paths, hyperparameters, or magic numbers in `src/` — everything
  configurable lives in `configs/*.yaml` and is loaded via
  `src/utils/config_loader.py`.
- Keep `docs/PROJECT_BRIEF.md`'s math notation consistent in code: name variables
  `z_v`, `z_a`, `w_v`, `w_a`, `p_v`, `p_a`, `p_f` to match the brief so anyone
  cross-referencing code and spec isn't guessing at correspondences.
- When a design decision in the brief is marked **[Proposed]** rather than
  **[Established]**, implement it as configurable/toggleable where reasonable (e.g.
  loss weights, ablation flags for sync/quality features) rather than hardcoding it,
  since the team may need to revise it after early experiments.

## 7. What "done" looks like for this scaffold

A file is done when it satisfies its own docstring's stated purpose, passes the
relevant test(s) in `tests/`, is driven entirely by `configs/`, and does not silently
violate any constraint in Section 2. The project as a whole is not "done" until the
reproducibility checklist in `docs/PROJECT_BRIEF.md` Section 13 is fully checked off.
