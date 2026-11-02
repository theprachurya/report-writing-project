# Reliability-Aware Audio-Visual Fusion for Generalizable Deepfake Detection

## PBL-3 Assessment Report and Final Report Draft

**Assessment date:** 27 October 2026  
**Course:** Report Writing (21CSC402P)  
**Department:** Department of Computing Technologies, SRM Institute of Science and Technology, Kattankulathur, Chennai  
**Team:** Aditya Sahu (RA2311026010814), Akash Cherian (RA2311026010852), Prachurya Subhasmita (RA2311026010855)  
**Project status:** Working implementation with preliminary validation results. Final primary-test evaluation is still pending.

---

## Abstract

This project implements a compact audio-visual deepfake detector that estimates a per-window reliability weight for visual and audio evidence before fusion. It uses short aligned video and audio windows, a MobileNetV3-Small visual branch, a depthwise-separable audio CNN, modality-specific prediction heads, synchronization and quality features, and a small reliability estimator. The repository also contains leakage-aware split manifests, evaluation utilities, and a local browser interface for demonstrating clip inference.

The current FakeAVCeleb v1.2 inventory contains 21,544 clips. The identity-grouped split contains 15,018 training, 2,136 validation, and 4,390 reserved test clips. After one training epoch, validation AUROC was 0.7974 for visual-only, 0.8625 for audio-only, 0.9426 for equal-weight fusion, and 0.7957 for reliability-aware fusion. Thus, equal-weight fusion leads in this early clean-validation snapshot; the gate has not yet demonstrated a benefit. A separate audio-only transfer probe evaluated the full German MLAAD-tiny slice using the threshold selected on FakeAVCeleb validation. It reached AUROC 0.3777 (source-utterance bootstrap 95% CI 0.3588–0.3964) on 2,820 windows. This negative transfer result is kept separate because the target labels describe bona-fide versus spoof audio, while the training label describes whether the full FakeAVCeleb clip is fake. The primary test split has not been evaluated. The evidence supports reporting the implementation and early results, but does not support a claim that reliability weighting improves generalization or robustness.

**Keywords:** audio-visual deepfake detection, reliability-aware fusion, modality quality, identity-disjoint evaluation, external transfer, reproducibility.

## 1. Introduction

Deepfake clips can manipulate the face, the voice, or both. A detector that uses only video can miss speech synthesis, and an audio-only detector can miss a face swap. Combining the modalities can help when their evidence is complementary, but fixed fusion can also give too much influence to a stream that is missing or degraded. Prior work studies audio-visual inconsistency and multimodal fusion, so this project does not claim to be the first audio-visual detector or the first dynamic fusion method [3]–[5].

The project asks whether a lightweight, interpretable reliability estimator can use modality-quality indicators, unimodal confidence, and audio-visual synchronization features to choose per-window fusion weights. Synchronization is treated as one feature among several. A nonzero audio-video offset is not by itself evidence of manipulation, and a zero offset does not prove authenticity.

The central assessment question is therefore empirical: does reliability-aware fusion improve discrimination, calibration, or performance under degradation compared with unimodal and fixed-fusion baselines? The available runs are preliminary and do not yet answer that question.

## 2. Problem statement and objectives

Given a short video clip with audio, predict whether the clip is manipulated. The model produces visual and audio embeddings (z_v) and (z_a), unimodal probabilities (p_v) and (p_a), gate weights (w_v) and (w_a), and a fused probability (p_f). The gate weights are normalized so that (w_v+w_a=1).

The project objectives are:

1. Build an aligned video/audio preprocessing pipeline with recorded quality and missingness features.
2. Establish visual-only, audio-only, and fixed-fusion baselines before evaluating reliability fusion.
3. Keep source identities and derivative clips within one split; fit thresholds and calibration on validation data only.
4. Evaluate generalization, degradation robustness, calibration, and inference cost without making claims unsupported by measurements.
5. Provide a local browser interface that can demonstrate the implemented model on a permitted short clip.

## 3. Technology choices and implementation status

| Component | Technology in the repository | Status and rationale |
|---|---|---|
| Language and model framework | Python 3.11 environment; PyTorch and torchvision | Implemented for configurable model construction and training. PyTorch is an established research framework [1]. |
| Video decoding and face crops | OpenCV-based decoding and Haar face detection; 25 fps decode, 16 sampled frames, 160×160 RGB crops | Implemented baseline. The project brief proposed a more accurate detector and landmark alignment; these are not implemented yet. |
| Audio processing | torchaudio and NumPy; mono 16 kHz audio and 80-bin log-Mel features | Implemented for the specified short-window pipeline. |
| Visual model | MobileNetV3-Small with temporal processing | Implemented. MobileNetV3 was designed for constrained compute settings [2]. |
| Audio model | Depthwise-separable convolutional neural network | Implemented as the compact audio branch. |
| Fusion and reliability | Modality heads, temporal alignment projector, synchronization features, quality/missingness features, and reliability MLP | Implemented. The performance of the gate remains under evaluation. |
| Data and reproducibility | CSV source/split manifests, YAML configuration, deterministic seed utilities, checkpoint configuration files | Implemented. Main train/validation/test manifests are identity-grouped; the test manifest remains reserved. |
| Metrics and analysis | scikit-learn/SciPy-based discrimination metrics, calibration and evaluation utilities | Implemented. Full primary-test and robustness reports remain outstanding. |
| Browser demonstration | Local HTML/CSS/JavaScript frontend served by Python's standard-library HTTP server | Implemented at `http://127.0.0.1:7860`; inference stays on the local machine. The current checkpoint is a one-epoch development artifact. |

The configuration currently uses an OpenCV Haar detector, even though landmark alignment remains in the proposed specification. Reports and demonstrations must identify that implementation gap rather than describe the proposed detector as completed. Dataset media are local and excluded from version control; the repository does not redistribute the licensed clips.

## 4. Dataset and evaluation protocol

### 4.1 Main paired audio-visual data

The local FakeAVCeleb v1.2 inventory contains 21,544 clips. Its source table and main split manifests are committed as metadata; the media themselves remain local. Splits were created by source identity, with duplicate checks, so a source identity does not cross the main train, validation, and test partitions.

| Split | Clips | Real (`clip_fake=false`) | Fake (`clip_fake=true`) | Purpose |
|---|---:|---:|---:|---|
| Train | 15,018 | 350 | 14,668 | Model fitting |
| Validation | 2,136 | 50 | 2,086 | Model selection and threshold/calibration fitting |
| Test | 4,390 | 100 | 4,290 | Locked final evaluation; not yet used |
| **Total** | **21,544** | **500** | **21,044** | — |

The local split is highly imbalanced: approximately 97.7% of clips in each partition are labelled fake. The training configuration uses balanced sampling. AUROC is reported for the preliminary comparison because it does not depend on a decision threshold. Future threshold-dependent results must include the threshold source and class counts; PR-AUC must be interpreted against the observed positive-class prevalence.

The saved data audit sampled 20 local clips; all 20 passed the recorded decode, log-Mel, and alignment checks. This is a pipeline smoke check, not an audit of every clip. A separate fixture-only check uses non-dataset sample media.

### 4.2 External audio-only transfer probe

The full German MLAAD-tiny slice was used only as an external audio-only probe. It contains 2,820 windows: 1,500 spoof and 1,320 bona-fide. There are 1,491 source-utterance groups; 171 spoof windows have no local bona-fide source audio. The model was trained on FakeAVCeleb's `clip_fake` label, which does not have identical semantics to the target task's bona-fide-versus-spoof-audio labels. The result is therefore exploratory and must not be combined with paired audio-visual metrics.

The threshold was selected using the FakeAVCeleb validation audio head with Youden's J. It was not retuned on MLAAD-tiny. Confidence intervals were calculated with 1,000 bootstrap resamples grouped by source utterance.

### 4.3 Cross-manipulation protocol status

A separate faceswap-held-out split has been prepared. Its held-out partition contains 1,194 clips from 464 identities. No model has yet been trained and evaluated on this protocol, so there is no cross-manipulation result to report. The small seen-family test partition has only five identities and is treated as exploratory in the split metadata.

## 5. Model and training method

Each input uses a two-second window. Video is decoded at 25 fps and represented by 16 sampled face frames. Audio is converted to mono 16 kHz and an 80-bin log-Mel spectrogram. The encoder outputs are pooled into eight temporal bins. Visual face-detection, blur, face-area and luminance statistics are retained with audio activity, loudness, clipping, SNR-proxy and duration features.

The visual encoder uses MobileNetV3-Small; the audio encoder uses a depthwise-separable CNN. Separate heads estimate (p_v) and (p_a). Synchronization features summarize similarity across a small range of temporal offsets. The reliability MLP receives per-modality features and outputs (w_v,w_a). A fused classifier estimates (p_f). Training includes fused and unimodal classification losses and a detached soft-target KL term for the reliability gate.

The preliminary checkpoints were trained for one epoch with seed 42. The configured target schedule is longer and uses three seeds, but the checkpoints presently in `artifacts/checkpoints/` are development artifacts, not final models. The current training run does not establish the proposed robustness benefits.

## 6. Result analysis

### 6.1 FakeAVCeleb identity-disjoint validation snapshot

The following AUROC values are from the saved one-epoch histories and the 2,136-clip validation split. They are single-run validation observations, not final test estimates.

| Model | Validation AUROC | Difference from audio-only | Difference from fixed fusion |
|---|---:|---:|---:|
| Visual-only | 0.7974 | −0.0651 | −0.1452 |
| Audio-only | 0.8625 | — | −0.0801 |
| Equal-weight fixed fusion | **0.9426** | **+0.0801** | — |
| Reliability-aware fusion | 0.7957 | −0.0668 | −0.1469 |

In this snapshot, equal-weight fusion has the highest validation AUROC. It is 0.0801 above the stronger unimodal branch, audio-only. The reliability-aware model is 0.1469 below fixed fusion and 0.0668 below audio-only. Its 0.0017 difference from visual-only is too small to interpret without repeated runs and uncertainty estimates.

These values do not establish that fixed fusion is the best final model or that reliability gating is ineffective. The experiment has one epoch and one seed, and the gate is not evaluated under degradation or missing-modality conditions. The proposed reason for reliability fusion is improved resilience when modality usefulness differs; clean validation alone does not test that claim. Possible optimization explanations must remain hypotheses until gate-weight distributions, branch losses, and controlled experiments are examined.

### 6.2 Full German MLAAD-tiny audio-only transfer

| Metric | Result |
|---|---:|
| AUROC | **0.3777** |
| Source-utterance bootstrap 95% CI | **[0.3588, 0.3964]** |
| PR-AUC | 0.5116 |
| Balanced accuracy at the fixed source threshold | 0.4827 |
| F1 at the fixed source threshold | 0.6561 |
| Recall / specificity at the fixed source threshold | 0.8820 / 0.0833 |
| Windows (spoof / bona-fide) | 2,820 (1,500 / 1,320) |
| Threshold | 0.2950, selected on FakeAVCeleb validation; not retuned |

At this fixed threshold the confusion matrix, with rows denoting actual bona-fide/spoof and columns denoting predicted bona-fide/spoof, is:

|  | Predicted bona-fide | Predicted spoof |
|---|---:|---:|
| Actual bona-fide | 110 | 1,210 |
| Actual spoof | 177 | 1,323 |

The transfer result is poor: AUROC is below 0.5, and specificity is low at the source-selected threshold. The PR-AUC of 0.5116 is also below the target spoof prevalence of 0.5319. The operating point was deliberately not adjusted on the target set. This result indicates that this one-epoch FakeAVCeleb audio checkpoint does not transfer usefully to this German audio-only probe under the stated label mapping. It is not evidence about every audio detector, every language, or paired audio-visual detection.

### 6.3 What the current results support

- The data and model pipeline can produce checkpoints and validation histories for unimodal and fusion variants.
- Equal-weight fusion is the strongest variant in the current clean-validation snapshot.
- The reliability-aware model has not demonstrated a clean-validation improvement after one epoch.
- The separate external audio-only probe shows weak transfer and demonstrates why cross-dataset results must preserve task and label semantics.
- There is no primary test score, calibration result, degradation result, missing-modality result, latency profile, or cross-manipulation result yet.

The project's hypotheses H1–H4 remain unconfirmed. In particular, the current evidence neither establishes nor refutes H1–H3, and H4 cannot be evaluated without a measured latency and resource profile on named hardware.

## 7. Conclusion

The project now has a working end-to-end scaffold: aligned audio/video preprocessing, compact unimodal encoders, fixed and reliability-aware fusion models, source-identity split manifests, evaluation code, saved development checkpoints, and a local web interface for clip-level demonstration. The dataset pipeline sampled 20 clips successfully, and the primary split preserves the required source-identity separation.

The preliminary performance results are mixed. Equal-weight fusion reached validation AUROC 0.9426 after one epoch, while reliability-aware fusion reached 0.7957. On the separate full German MLAAD-tiny audio probe, the audio branch reached AUROC 0.3777 with a 95% source-group bootstrap interval of 0.3588–0.3964. The latter is an exploratory transfer failure under a different target-label definition. These results do not support a claim that reliability-aware fusion generalizes better. The primary test split remains untouched, and the hypotheses remain open pending the planned experiments.

## 8. Future enhancements and completion plan

| Period | Work | Evidence to save |
|---|---|---|
| 27–30 Oct | Run full-length unimodal and fixed-fusion training; log the actual epoch count, seed, configuration, validation loss and AUROC. | Checkpoint and config per seed; comparable validation table |
| 31 Oct–2 Nov | Train reliability fusion with staged encoder freezing; inspect (w_v,w_a), gate entropy, modality losses, and reliability loss; tune proposed loss settings on validation only. | Gate diagnostics; selected validation configuration; no test-based selection |
| 3–5 Nov | Run validation-side degradation/missing-modality analyses and finalize the held-out manipulation protocol. | Robustness curves and gate-weight summaries; protocol manifest |
| 6–7 Nov | Choose the final model using validation only; fit temperature and operating threshold on validation; freeze checkpoint/config/hash. | Final model card and reproducibility record |
| 8 Nov | Traverse the locked primary test once; calculate raw and calibrated metrics separately, group-bootstrap intervals, and the planned separate transfer results. | Final evaluation JSON and tables |
| 9 Nov | Update the report, prepare the viva presentation, rehearse the local browser demo, and verify licensing-safe demo media. | Final report, presentation, demo checklist |
| 10 Nov | Report and viva assessment. | Final submission |

If compute limits prevent the planned number of epochs or seeds, report the runs actually completed and retain the limitation. Do not relax the identity-grouped split, select on test, tune the external threshold, or describe an unmeasured hypothesis as supported.

## 9. Final report draft structure

The following structure is ready to extend after the final evaluation. Sections 1–8 above contain the current evidence and can be used as the draft text.

1. Title page and team details.
2. Abstract and keywords.
3. Introduction, problem statement, objectives, and research questions.
4. Related work and the bounded research contribution.
5. Dataset access, label semantics, preprocessing, and leakage controls.
6. Proposed architecture, loss functions, implementation technologies, and web demonstration.
7. Experimental setup, baseline hierarchy, metrics, calibration, and uncertainty procedure.
8. Results: validation development, final untouched test, cross-manipulation, degradation, missing modality, external transfer, and efficiency as separate subsections.
9. Discussion, limitations, responsible use, and hypothesis status.
10. Conclusion and future work.
11. References and reproducibility appendix: seeds, config/checkpoint hashes, manifest checksums, package/hardware versions, and metric files.

**Final-report update rule:** replace the current preliminary statements only when the matching final evidence file exists. Keep preliminary validation, raw test, calibrated test, cross-dataset transfer, and optimized-runtime results visibly distinct.

## 10. Viva demonstration notes

Start the local interface with:

```bash
make web CHECKPOINT=artifacts/checkpoints/reliability_fusion/best_model.pt
```

Then open `http://127.0.0.1:7860`. The browser page returns a fused score, modality scores, reliability weights, and quality flags. Explain that the current checkpoint is a one-epoch development model and the displayed score is a model output, not a forensic proof. Use only a short clip permitted for the demonstration; do not copy licensed dataset media into the repository or presentation.

## References

[1] A. Paszke et al., “PyTorch: An Imperative Style, High-Performance Deep Learning Library,” *Advances in Neural Information Processing Systems*, vol. 32, 2019. [Proceedings](https://proceedings.neurips.cc/paper/2019/hash/bdbca288fee7f92f2bfa9f7012727740-Abstract.html).

[2] A. Howard et al., “Searching for MobileNetV3,” in *Proceedings of the IEEE/CVF International Conference on Computer Vision*, 2019, pp. 1314–1324. [CVF Open Access](https://openaccess.thecvf.com/content_ICCV_2019/html/Howard_Searching_for_MobileNetV3_ICCV_2019_paper.html).

[3] H. Khalid, S. Tariq, M. Kim, and S. S. Woo, “FakeAVCeleb: A Novel Audio-Video Multimodal Deepfake Dataset,” in *NeurIPS Datasets and Benchmarks*, 2021. [Proceedings](https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/hash/d9d4f495e875a2e075a1a4a6e1b9770f-Abstract-round2.html).

[4] W. Yang et al., “AVoiD-DF: Audio-Visual Joint Learning for Detecting Deepfake,” *IEEE Transactions on Information Forensics and Security*, vol. 18, pp. 2015–2029, 2023, doi: 10.1109/TIFS.2023.3262148.

[5] R. Wang, D. Ye, L. Tang, Y. Zhang, and J. Deng, “AVT²-DWF: Improving Deepfake Detection With Audio-Visual Fusion and Dynamic Weighting Strategies,” *IEEE Signal Processing Letters*, vol. 31, pp. 1960–1964, 2024, doi: 10.1109/LSP.2024.3433596.

## Evidence and provenance

- Main validation AUROCs: `artifacts/checkpoints/*/history.json` and their saved `config.yaml` files; seed 42, one epoch each.
- Main split counts and labels: `manifests/splits/{train,val,test}_manifest.csv`.
- Data audit sample: `manifests/data_audit_report.json` (20 sampled clips; no primary test inference).
- External probe metrics: `reports/external_audio/mlaad_tiny_de_audio_eval.json`; checkpoint SHA-256 `66a6045482023150190ddb6cd502d1eae2a319c773ce194e3a4543834ec2fbd1`; seed 42; PyTorch 2.14.1+cpu; full German slice, no smoke limit.
- External dataset revision: `9143e5ea709575ebab6bec52840a1043aada7bb1`.
- Main test status: untouched; no final paired audio-visual test metrics are included in this draft.
