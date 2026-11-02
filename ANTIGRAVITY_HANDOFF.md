# Handoff for Google Antigravity

**Workspace:** `/home/tsuki/Work/report-writing-project`  
**Snapshot date:** 27 October 2026 (Asia/Kolkata)  
**Current user goal:** continue this project as a working audio-visual deepfake detector and finish the scheduled PBL-3/report/viva materials.

## Read first

1. Read `AGENTS.md` fully before changing code.
2. Read `docs/PROJECT_BRIEF.md`; it is the canonical technical specification and overrides conflicting summaries here.
3. Check `git status --short` before editing. The workspace contains intentional, uncommitted data-pipeline, evaluation, frontend, and report changes. Do not reset/revert or overwrite existing assessment files without reviewing them.

## Instructions for continuing

- Treat this as active work. Recheck the files and current date, then make the next useful change; do not stop after returning another plan or status summary.
- Follow the staged build order in `AGENTS.md`. Finish and verify prerequisites before advancing to later stages. Keep changes within the existing module boundaries and configuration-driven design.
- Preserve all existing user/project changes, local checkpoints, raw data, generated results, and dated PBL-2 materials. Never clean or reset the workspace as a shortcut.
- Continue toward a runnable project and the 10 November report/viva deadline. Prioritize a reviewable implementation or deliverable over speculative polish. Update the PBL-3 report and presentation only when supported by current evidence.
- Use `.venv/bin/python` for project commands. Run the relevant tests alongside code changes as required by `AGENTS.md`; clearly report what was and was not run.
- Before any evaluation, identify its split and scope. Use validation for model selection, thresholds, and temperature. Keep the primary test locked until the final model/configuration is frozen, then evaluate it once. Never tune on the primary test or MLAAD target results.
- Keep paired FakeAVCeleb results separate from audio-only MLAAD transfer. Preserve label definitions, class counts, confidence intervals, checkpoint/config hashes, and limitations when adding result claims.
- Do not redistribute licensed media, include raw clips in artifacts, or use restricted clips in slides/demo without permission. Use only permitted local media for the demo.
- Avoid unnecessary clarification pauses. Make reversible implementation and document decisions using the existing specification; ask only when missing information materially blocks the next step. Give concise progress updates and finish with changed files, evidence, and remaining blockers.

## Project

**Title:** Reliability-Aware Audio-Visual Fusion for Generalizable Deepfake Detection (the project brief title is “Cross-Manipulation Generalizable Audio-Visual Deepfake Detection via Reliability-Aware Lightweight Fusion”).

**Team:** Aditya Sahu (RA2311026010814), Akash Cherian (RA2311026010852), Prachurya Subhasmita (RA2311026010855). Course: Report Writing, 21CSC402P, SRMIST.

The detector processes a short clip and predicts whether it is manipulated. Its proposed contribution is a per-window reliability gate that weights visual and audio representations using modality-quality features, unimodal confidence/uncertainty, and audio-visual synchronization features. Synchronization is only an auxiliary input, never an authenticity verdict.

## Non-negotiable project constraints

- Never redistribute, upload, or bundle licensed media. Raw data stays under local, gitignored `data/raw/` paths.
- Every training run consumes committed manifests under `manifests/splits/`; do not split inside a training loop. Source identities/derivatives must not cross partitions.
- Fit thresholds and temperatures on validation only. Keep the primary test split untouched until the final evaluation, then evaluate it once.
- Keep raw, calibrated, and optimized-runtime metrics distinct. Do not claim a hypothesis, superiority, generalization, or robustness before the matching experiment supports it.
- `docs/PROJECT_BRIEF.md` defines the architecture, 25 fps decode, 2-second/16-frame windows, 160×160 face crops, 16 kHz audio, 80-bin log-Mel input, staged training, metrics, and reproducibility requirements.

## Data present

- **FakeAVCeleb v1.2:** local at `data/raw/FakeAVCeleb_v1.2` (about 6.3 GB; 21,544 clips). It is the paired audio-visual training/evaluation set. The clips are licensed and must not be copied into the repository or presentation.
- Main identity-grouped split counts:

  | Split | Total | Real | Fake |
  |---|---:|---:|---:|
  | Train | 15,018 | 350 | 14,668 |
  | Validation | 2,136 | 50 | 2,086 |
  | Test | 4,390 | 100 | 4,290 |

  A direct manifest check found zero source-identity overlap among train, validation, and test. The test manifest has not been run through the model. Each split is about 97.7% fake; training uses balanced sampling.
- **MLAAD-tiny German slice:** local at `data/raw/MLAAD-tiny` (about 641 MB), pinned revision `9143e5ea709575ebab6bec52840a1043aada7bb1`. Use only as an audio-only external transfer probe. Generated audio is CC BY-NC 4.0; bona-fide audio has its own original license. Do not redistribute either.
- **DFDC:** not present locally; optional and not required for current paired AV experiments. If obtained later, treat as visual-only unless audio-label provenance is established.
- `manifests/splits/cross_manipulation_faceswap/` contains a prepared held-out faceswap split. No model evaluation has been run on it. Its “seen” test subset has only five identities; keep that scope exploratory.

## Current implementation

- Model/preprocessing/evaluation code is under `src/`; configs are in `configs/`; scripts are under `scripts/`.
- Visual branch: MobileNetV3-Small plus temporal processing. Audio branch: depthwise-separable CNN. Fusion includes unimodal heads, a reliability MLP, and quality/synchronization features.
- Important implementation gap: face extraction currently uses OpenCV Haar detection. Landmark alignment and the proposed stronger detector backend are not implemented.
- Local browser demo: `web/index.html`, `scripts/serve_web.py`, `src/inference/clip_service.py`, `configs/web_app_config.yaml`. It binds to `127.0.0.1:7860`, processes uploads locally, and requires a full AV checkpoint. Start with:

  ```bash
  make web CHECKPOINT=artifacts/checkpoints/reliability_fusion/best_model.pt
  ```

  The available checkpoint is a one-epoch development artifact. The app chooses its decision threshold using the FakeAVCeleb validation split; a demo score is not forensic proof.
- Use `.venv/bin/python` for project commands. The system `/usr/bin/python` does not have the project dependencies. Last observed runtime was Python 3.11, PyTorch `2.14.1+cpu`; evaluation ran on CPU.

## Results available (do not overstate)

### Primary paired AV validation

Saved histories at `artifacts/checkpoints/*/history.json` record one epoch, seed 42, on the FakeAVCeleb validation split:

| Model | Validation AUROC |
|---|---:|
| Visual-only | 0.7974 |
| Audio-only | 0.8625 |
| Equal-weight fixed fusion | **0.9426** |
| Reliability-aware fusion | 0.7957 |

This is a single early validation snapshot, not a final estimate. Fixed fusion leads in this snapshot; the reliability gate has not shown a benefit. There are no primary test, degradation, missing-modality, calibration, multi-seed, or latency results yet.

### Full German MLAAD-tiny external audio probe

Full report: `reports/external_audio/mlaad_tiny_de_audio_eval.json` (under gitignored `reports/`). Full-slice evaluation completed with the audio checkpoint; threshold came from FakeAVCeleb validation and was **not** retuned on the target.

- 2,820 windows: 1,500 spoof / 1,320 bona-fide; 1,491 source-utterance groups.
- AUROC **0.3777**, source-group bootstrap 95% CI **[0.3588, 0.3964]** (1,000 replicates).
- PR-AUC 0.5116; balanced accuracy 0.4827; F1 0.6561; recall 0.8820; specificity 0.0833 at threshold 0.2950.
- This is a negative exploratory transfer result. FakeAVCeleb training label `clip_fake` and MLAAD-tiny bona-fide/spoof-audio labels differ. Keep it separate from paired AV results; do not retune threshold or describe it as evidence about all languages/detectors.
- The earlier smoke report under `reports/external_audio_smoke/` used only 16 paired source groups (32 windows); it is not a benchmark result.

## Assessment artifacts already prepared

- PBL-2 (16 Oct 2026):
  - IEEE rough draft: `docs/pbl2_ieee_draft.tex`, `docs/pbl2_ieee_draft.pdf` (8 pages, letter).
  - PBL-2 full report: `FP2_Reliability_Aware_AV_Deepfake_Detection_FINAL.docx`, `FP2_Reliability_Aware_AV_Deepfake_Detection_FINAL.pdf` (18 pages, A4), and `docs/report-writing-paper.docx`.
- PBL-3 (27 Oct 2026):
  - Result analysis & final report draft: `docs/PBL3_Results_Analysis_and_Final_Report_Draft.md`, `.docx`, `.pdf` (8 pages, A4).
  - Presentation: `docs/PBL3_Presentation.html` (interactive widescreen), `docs/PBL3_Presentation.pdf` (7 widescreen pages), and `docs/PBL3_Presentation_Outline.md`.
- Report & Viva with Demo (10 Nov 2026):
  - Complete project report & Turnitin plagiarism compliance guide: `docs/FINAL_PROJECT_REPORT_AND_PLAGIARISM_GUIDE.md`, `.docx`, `.pdf` (15 pages).
  - Final viva defense & live demo execution guide: `docs/FINAL_VIVA_AND_DEMO_GUIDE.md`, `.docx`, `.pdf` (6 pages).
- `README.md` links all three milestone materials.

## Deadlines and next work

- PBL-2 assessment was 16 October 2026; its report and IEEE rough draft are already present.
- PBL-3 assessment is 27 October 2026 (today in this snapshot): results, conclusion, future enhancements, final-report draft, presentation.
- Final report, plagiarism report, viva, and demo are due 10 November 2026.

Recommended next steps, following `AGENTS.md` and the brief:

1. Review/finish the PBL-3 report and presentation with the team; keep preliminary results clearly marked.
2. Run planned longer unimodal/fixed-fusion baselines and multiple seeds if compute permits. The existing checkpoints are one-epoch development artifacts.
3. Train reliability fusion with the staged schedule; record gate-weight distributions, entropy, modality losses, and collapse diagnostics. Tune only on validation.
4. Run the planned degradation/missing-modality and faceswap-held-out experiments. Maintain the distinct in-domain and held-out protocols.
5. Select/freeze the final model on validation, fit threshold/temperature there, then evaluate the locked primary test once. Update the report with actual results, uncertainty intervals, and named-hardware latency.
6. Rehearse the local web demo with permitted media; never place licensed clips in slides or source control.

## Verification state

- Last known repository unit-test run before the frontend changes: 72 passed. Tests were **not rerun** during the report/document turn; do not claim new frontend test coverage.
- Frontend was manually smoke-tested previously: local server/status/page worked, a synthetic fixture-derived clip received an inference response, invalid upload handling returned an error, and desktop/mobile layouts were inspected. Do not cite the synthetic prediction as a metric.
- During report preparation, the main split identities were checked for overlap (zero), the full external MLAAD probe completed, `git diff --check` passed, and the PBL-3 PDFs were inspected.
- Workspace has many intentional uncommitted changes. Do not clean the tree or remove local reports/checkpoints/data.
