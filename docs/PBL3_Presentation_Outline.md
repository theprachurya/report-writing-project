# PBL-3 Presentation Outline

**Project:** Reliability-Aware Audio-Visual Fusion for Generalizable Deepfake Detection  
**Assessment:** 27 October 2026  
**Team:** Aditya Sahu · Akash Cherian · Prachurya Subhasmita

## Slide 1 — Problem and goal

- A video may manipulate the face, voice, or both.
- Audio and video can have different quality in the same clip.
- Goal: estimate whether adaptive modality weighting helps over unimodal and fixed-fusion baselines.

**Speaker note:** Synchronization is an auxiliary signal, not a fake/real rule.

## Slide 2 — Proposed system

- 2-second aligned audio/video window.
- 16 face frames and 80-bin log-Mel audio features.
- MobileNetV3-Small visual branch and depthwise-separable audio CNN.
- Modality heads, quality/synchronization features, reliability MLP, fused classifier.

**Speaker note:** Explain (p_v,p_a,p_f) and weights (w_v,w_a); emphasize (w_v+w_a=1).

## Slide 3 — Technology and working demo

- Python, PyTorch/torchvision, OpenCV, torchaudio, scikit-learn/SciPy, YAML configs.
- CSV identity-grouped manifests and saved checkpoint configurations.
- Local browser frontend: video upload → preprocessing → fused score and reliability display.

**Speaker note:** The current frontend uses a one-epoch development checkpoint. A displayed score demonstrates the pipeline, not validated forensic certainty.

## Slide 4 — Dataset and protocol

- FakeAVCeleb v1.2 local inventory: 21,544 clips.
- Train 15,018; validation 2,136; locked test 4,390.
- Main splits are source-identity grouped; the dataset is highly imbalanced.
- Primary test is not evaluated yet.

**Speaker note:** Do not show or redistribute licensed media in the slides.

## Slide 5 — Preliminary paired AV results

| Model | One-epoch validation AUROC |
|---|---:|
| Visual only | 0.7974 |
| Audio only | 0.8625 |
| Equal-weight fusion | **0.9426** |
| Reliability-aware fusion | 0.7957 |

**Speaker note:** One seed and one epoch only. Fixed fusion is ahead in this snapshot; the gate has not shown an advantage. No final-test claim.

## Slide 6 — External audio-only transfer

- Full German MLAAD-tiny slice: 2,820 windows, 1,491 source-utterance groups.
- AUROC 0.3777; 95% grouped-bootstrap interval 0.3588–0.3964.
- Threshold was selected on FakeAVCeleb validation and was not retuned.
- Target label semantics differ from FakeAVCeleb `clip_fake`.

**Speaker note:** Treat this as a negative exploratory transfer finding, separate from paired AV results.

## Slide 7 — Conclusion and next work

- The implementation and local inference demo work.
- Current evidence favors equal-weight fusion on early clean validation.
- Reliability benefit, degradation robustness, calibration, cross-manipulation, and efficiency remain unmeasured.
- Next: full training, repeated seeds, gate diagnostics, robustness tests, validation-only selection, one final test pass.

**Speaker note:** Keep the conclusion tied to the current evidence and use the November 10 deadline to frame the remaining work.

## Demo checklist

1. Start the server with `make web CHECKPOINT=artifacts/checkpoints/reliability_fusion/best_model.pt`.
2. Open `http://127.0.0.1:7860` and confirm the model status is ready.
3. Use only an approved short clip; verify audio/video are processed locally.
4. Explain the fused score, validation threshold, modality scores, and gate weights.
5. State that the current checkpoint is a development artifact and a detector score is not proof of manipulation.
