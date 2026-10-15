# Cross-Manipulation Generalizable Audio-Visual Deepfake Detection via Reliability-Aware Lightweight Fusion

**Master technical document — project proposal and execution plan**  
**Team:** Aditya Sahu · Akash Cherian · Prachurya Subhasmita  
**Status:** proposed undergraduate research system; no performance claims are made before experiments.

> **Reading convention.** Statements marked **[Established]** summarize the uploaded `deep-research-report.md` and the cited literature. Statements marked **[Proposed]** are the team’s design decisions and experimental commitments. A proposed module is not presented as a published result or as a claim of priority.

## Executive abstract

Audio-visual deepfakes may alter a face, a voice, or both. A detector that simply concatenates audio and visual features assumes that both streams deserve comparable trust; that assumption can be poor when video is compressed, the face crop fails, audio is noisy, or a manipulation leaves one stream comparatively informative. **[Established]** Earlier work has shown that audio-visual inconsistency, affective mismatch, and cross-modal representations can be useful, while newer work also reports dynamic weighting strategies [1–6]. The uploaded report correctly identifies generalization to unseen manipulations, degradation robustness, and efficient inference as important practical concerns.

This project proposes a small, reproducible detector that extracts aligned face-motion and speech-spectrogram features, estimates **per-window modality reliability**, and fuses the features using learned weights. Synchronization is retained as an *auxiliary reliability signal*, not the headline claim: a well-synchronized fake can still be fake, and a real video can be imperfectly synchronized. The system will be judged against unimodal and conventional-fusion baselines under identity-disjoint, manipulation-held-out, and controlled degradation protocols. The initial scope uses short clips, frozen/precomputed face crops where possible, compact encoders, mixed precision, and a single modest GPU. The intended contribution is evidence about whether reliability-aware lightweight fusion improves the accuracy–robustness–latency trade-off under a clearly specified protocol—not a claim that it universally solves deepfake detection.

## 1. Background, problem, and motivation

### 1.1 Background

**[Established]** Deepfake detection is a binary media-authenticity task: given a video clip with audio, estimate whether the clip has been manipulated. Visual evidence can include face texture, blending artifacts, and motion; audio evidence can include synthesis/voice-conversion artifacts and temporal speech structure. Cross-modal evidence can include correspondence between mouth motion, speech timing, and broader affective/content cues. Chugh et al. used audio-visual dissonance; Mittal et al. used affective agreement; AVoiD-DF and AVFF model joint audio-visual evidence [1–4].

### 1.2 Problem statement

Given an audio-video sample \(x=(V,A)\), predict \(y\in\{0,1\}\), where 0 is real and 1 is fake. Standard fusion uses \(\hat y=f(E_v(V),E_a(A))\). The central problem is that the usefulness of \(V\) and \(A\) is *instance dependent*:

- video may be low resolution, blurred, re-encoded, occluded, or have failed face detection;
- audio may be silent, noisy, clipped, off-screen, or not actually manipulated;
- a synthetic clip may be temporally consistent, so a synchronization-only detector can fail;
- a model can memorize generator- or identity-specific artifacts instead of learning transferable evidence.

**[Proposed problem formulation]** Design a compact detector that estimates relative reliability of visual and audio evidence for each short window, uses it in fusion, and tests whether this improves performance against fixed/naïve fusion when manipulation family, identity, codec, and quality change.

### 1.3 Motivation and research gaps

| Gap | Evidence from prior work | Project response |
|---|---|---|
| Fusion is not automatically superior to unimodal detection | Muppalla, Jia, and Lyu explicitly motivate integration because existing audio-visual methods do not always exceed unimodal analysis [5]. | Compare visual-only, audio-only, fixed fusion, and reliability-aware fusion; report when fusion loses. |
| Cross-manipulation transfer is difficult | AVoiD-DF evaluates joint learning and audio-visual inconsistencies, motivating manipulation-aware testing [3]. | Hold out manipulation categories/generators before training. |
| Cross-modal correspondence is useful but insufficient alone | Dissonance, affective cues, and AVFF exploit correspondence [1,2,4]. | Use synchronization as a feature supplied to the gate and classifier, while retaining independent forensic features. |
| Dynamic weighting already exists in the literature | AVT²-DWF proposes dynamic weight fusion [6]. | Do not claim first use of dynamic weighting. Narrow novelty to a transparent, reliability-targeted lightweight design and its evaluation protocol. |
| Clean-benchmark scores do not establish deployment robustness | Compression and domain shift can change forensic cues. | Use controlled codec/noise tests and report latency, calibration, and failure cases. |

## 2. Research questions, hypotheses, objectives, and novelty

### Research questions

1. Does learned per-window reliability weighting improve AUROC, balanced accuracy, and calibration over unimodal and fixed-fusion baselines?
2. Does it retain a relative advantage when one modality is degraded or absent?
3. Does the advantage persist when the manipulation family is held out and source identities are disjoint?
4. What accuracy, calibration, model-size, and latency trade-off is achievable on one modest GPU and a CPU?

### Testable hypotheses

- **H1:** Reliability-aware fusion has higher mean AUROC than equal-weight and concatenation fusion on the pre-registered degradation suite.
- **H2:** Under isolated audio or video degradation, the gate shifts *in the expected direction* and reduces the performance drop relative to fixed fusion.
- **H3:** On unseen manipulation categories, it matches or exceeds the strongest unimodal baseline; if it does not, the hypothesis is not supported.
- **H4:** A compact model can be measured, not assumed, to achieve practical short-window latency with a documented accuracy cost versus a larger reference.

### Objectives and success criteria

1. Build a reproducible aligned audio-visual pipeline with provenance manifests and source-identity grouping.
2. Implement compact audio, visual, fixed-fusion, synchronization-augmented, and reliability-aware models.
3. Evaluate in-domain, cross-manipulation, cross-dataset where valid, and degradation settings.
4. Report discrimination, calibration, uncertainty, efficiency, and subgroup/quality breakdowns with confidence intervals.
5. Release code/configuration/split manifests only to the extent allowed by dataset terms; never redistribute restricted videos.

### Novelty assessment

**Defensible proposed novelty:** an implementable reliability estimator whose inputs explicitly combine modality-quality indicators, unimodal confidence/uncertainty, and cross-modal consistency; a reliability-target loss tied to unimodal correctness under modality-specific augmentation; and a lightweight evaluation protocol that treats synchronization as secondary evidence. **Not claimed:** first audio-visual detector, first dynamic fusion, first cross-modal consistency method, or state-of-the-art performance. AVFF and AVT²-DWF make broad prior-art claims impossible to support [4,6].

## 3. Scope and non-scope

**In scope:** clip-level real/fake classification; face-centred talking-head clips; audio and video degradation; identity-disjoint experimental splits; ablations; post-hoc calibration; modest-GPU feasibility.

**Out of scope:** legal proof of authorship, universal media provenance, localization as a promised deliverable, live streaming at arbitrary resolution, multilingual/general-domain guarantees, adversarial-attack defense, training a generative model, or deployment against people without appropriate authorization. A detector score is forensic decision support, not conclusive proof of manipulation.

## 4. Literature survey

| Work | Established contribution | Relation / limitation for this project |
|---|---|---|
| Chugh et al., ACM MM 2020 [1] | Learns modality dissonance and reports temporal localization. | Establishes that mismatch is useful; mismatch alone can miss synchronized fakes. |
| Mittal et al., ACM MM 2020 [2] | Uses audio-visual affective cues and a Siamese/triplet formulation. | Supports complementary cues; affect prediction adds cost and may be domain-sensitive. |
| Yang et al., IEEE TIFS 2023, AVoiD-DF [3] | Audio-visual joint learning based on inconsistency; includes DefakeAVMiT evaluation. | Strong reference for cross-manipulation evaluation; not a reason to omit leakage controls. |
| Oorloff et al., CVPR 2024, AVFF [4] | Two-stage self-supervised cross-modal representation learning and supervised fine-tuning. | Strong, likely heavier reference; its published FakeAVCeleb result is not directly comparable without the same split/protocol. |
| Muppalla et al., 2023 [5] | Integrates fine-grained unimodal labels and binary classification; notes that multimodal methods do not always beat unimodal methods. | Justifies rigorous unimodal baselines. |
| Wang et al., 2024, AVT²-DWF [6] | Dual-transformer dynamic weight fusion. | Direct prior art for dynamic weights; treat as an arXiv preprint unless a peer-reviewed version is verified. |
| Ilyas et al., Applied Soft Computing 2023, AVFakeNet [7] | Dense Swin Transformer end-to-end audio-visual detection. | Useful reference model but not suitable as the first single-GPU baseline. |
| Khalid et al., FakeAVCeleb [8] | Audio-video benchmark with synthesized audio/video combinations. | Primary multimodal dataset candidate; access and label structure require audit. |
| Dolhansky et al., DFDC [9] | Large face-swap dataset built with consented actors and multiple facial modification algorithms. | Useful visual-training/robustness resource; do not casually label it an audio-deepfake benchmark. |

## 5. Data plan, access, licensing, and ethical caveats

### 5.1 Dataset roles

| Dataset | Role | Important caveat |
|---|---|---|
| **FakeAVCeleb** [8] | Main multimodal benchmark for reliability experiments. | The official repository describes request-form access and a separate data license; code is MIT but that does **not** license the videos. The reported 20,000-video table is highly class-imbalanced (500 real/19,500 fake). Audit actual version, labels, and permissions after access. |
| **DFDC full** [9] | Optional visual pretraining/visual robustness set; possibly a separate visual-only transfer evaluation. | Meta describes a full set of roughly 124k clips and eight facial modification algorithms, with access prerequisites. It is fundamentally a face-swap benchmark. Preserve audio only where its provenance and label semantics are verified; do not claim all DFDC fake audio is synthetic. |
| Small internal *derived* stress set | Codec/noise/blur variants generated from already permitted test clips. | Derived files inherit source restrictions and must not be shared publicly unless permitted. |

No web scraping, celebrity-voice cloning, or collection of non-consented media is needed. Cite datasets exactly as required; archive the accepted terms, version date, and manifest checksum privately. Confirm institutional policy before downloading or processing personal data.

### 5.2 Labels and imbalance

**[Proposed]** Keep two label fields: `video_manipulated` and `audio_manipulated`, plus `clip_fake` derived by a documented rule. Never infer an audio-manipulation label from a video-fake label. For binary experiments, balance class sampling in each training epoch and report natural-prevalence metrics separately. If a dataset lacks reliable manipulation-family metadata, it cannot support a generator-held-out claim; use it only for the protocol it actually supports.

### 5.3 Leakage-resistant splits

1. Build a source table before any augmentation: source identity, source video, dataset, manipulation family, audio condition, and clip ID.
2. Group all real/fake derivatives of the same source video and all clips of one identity into one partition. Also use perceptual/video hashes as a duplicate check.
3. Make a 70/10/20 train/validation/test split at source-identity group level where identity metadata exists. Select thresholds only on validation.
4. For the **cross-manipulation test**, choose the held-out manipulation family *before* training. Training and validation use only seen families; test uses held-out identities and the held-out family. Report a separate in-domain test; never mix its result with held-out results.
5. Generate degradations after partitioning and keep every derivative in its parent partition. Test augmentations never appear in training at the exact same encoding recipe.
6. Keep all splits as versioned CSV/JSON manifests; seed alone is insufficient.

## 6. Preprocessing and input contract

**[Proposed initial contract]** Decode a clip at a fixed 25 fps, sample a 2-second window with 16 uniformly spaced face frames (effective 8 fps), and extract matching 2 seconds of mono 16 kHz audio. A window is valid only if face detection succeeds on at least 12/16 frames and voice activity exceeds a documented threshold; invalid windows are retained with missingness flags rather than silently discarded.

- **Visual:** RetinaFace/MediaPipe-style face detection; expand the face box by 20%; align by landmarks; resize RGB crops to 160×160. Record face size, detector confidence, blur (variance of Laplacian), saturation/brightness, and failed-frame fraction.
- **Audio:** resample to 16 kHz mono; peak-normalize only after recording clipping; compute 80-bin log-Mel spectrograms (25 ms window, 10 ms hop). Record RMS, voice-activity fraction, clipped-sample ratio, estimated SNR proxy, and duration mismatch.
- **Alignment:** divide each window into \(T=8\) temporal bins. Pool visual and audio encoder outputs to one 256-D vector per bin. Store the timestamp transform and any dropped frames.
- **Training augmentation:** randomly apply *one modality at a time* in many batches: video resize/re-encode/blur/frame drop, or audio noise/reverberation/low-pass/packet-like dropout. Apply jointly degraded cases later. Augmentation labels describe quality, never fake/real status.

## 7. Proposed realistic architecture

```text
2 s clip
 ├─ 16 aligned face crops ─ Visual MobileNetV3-small + temporal depthwise Conv ─ z_v[1..8]
 ├─ 16 kHz log-Mel audio ─ Audio DS-CNN / MobileNetV3-small ──────────────── z_a[1..8]
 ├─ quality features q_v, q_a; missingness flags
 └─ sync features c (cosine/cross-correlation across small offsets)

[z_v, z_a, q_v, q_a, c, uncertainty] ─ Reliability MLP ─ softmax(w_v,w_a)
                  │                                         │
                  ├─ modality heads p_v,p_a ────────────────┤
                  └─ projected embeddings ─ weighted fusion ┴─ classifier p_f
```

### 7.1 Module-by-module design

| Module | Proposed minimal implementation | Output / reason |
|---|---|---|
| Visual encoder | ImageNet-pretrained MobileNetV3-Small (or EfficientNet-B0 if memory permits), 160×160, followed by depthwise temporal Conv1D over 8 bins. | \(z_v\in\mathbb R^{8\times256}\); uses appearance and short motion without a large 3D CNN. |
| Audio encoder | 2-D depthwise-separable CNN on log-Mel patches, width multiplier 0.75–1.0. | \(z_a\in\mathbb R^{8\times256}\); substantially cheaper than a transformer. |
| Alignment projector | Separate linear layers map \(z_v,z_a\) to 128-D shared space. | Comparable features without pretending raw modalities share a coordinate system. |
| Synchronization feature | For offsets \(\delta\in\{-2,-1,0,1,2\}\), compute mean cosine similarity \(s_\delta\). Use \(s_0\), \(\max_\delta s_\delta\), best offset, and dispersion. | Secondary evidence: timing/correspondence and possible decoding delay. |
| Quality features | Visual: detection success, blur, face area, luminance statistics. Audio: VAD, RMS, clipping, SNR proxy. | Transparent input signals; quality does not equal authenticity. |
| Modality heads | Small linear classifiers on mean-pooled \(z_v,z_a\). | \(p_v,p_a\), losses and uncertainty estimates. |
| Reliability estimator | 2-layer MLP (input ≈ 30–50 features; 128 hidden; dropout 0.1) producing gate logits. | \(w_v,w_a\); tiny parameter cost and inspectable inputs. |
| Fusion classifier | \(f=[w_v\bar z_v,w_a\bar z_a,c,w_v,w_a]\) → 256-D MLP → logit. | Keeps the gate and consistency evidence visible to classifier. |

### 7.2 Reliability estimator definition

For modality \(m\in\{v,a\}\), let \(h_m\) contain pooled feature statistics, quality features, missingness, and uncertainty from its modality head. Let \(c\) be synchronization/consistency features. The proposed estimator is

\[
g = \mathrm{MLP}([h_v,h_a,c]), \qquad
[w_v,w_a] = \mathrm{softmax}(g/T_g), \qquad w_v+w_a=1.
\]

Here *reliability* means estimated usefulness of a modality for the label on this window under the training distribution; it is **not** a ground-truth measure of media quality or truthfulness. A gate should be expected to lower a weight when the modality is missing/degraded or its head is uncertain, but this behavior must be measured rather than assumed.

To give the gate an observable training signal, define per-window unimodal binary cross-entropy losses \(\ell_v\) and \(\ell_a\), using logits from the modality heads. A detached soft target is

\[
t_m=\frac{\exp(-\operatorname{stopgrad}(\ell_m)/\tau_r)}
{\sum_{j\in\{v,a\}}\exp(-\operatorname{stopgrad}(\ell_j)/\tau_r)}.
\]

The reliability loss is \(\mathcal L_{rel}=\mathrm{KL}(t\|w)\). It does not assert that the easier modality is intrinsically more authentic; it regularizes the gate toward the modality currently more predictive. Add modality-specific degradations and missing-modality training so this target does not collapse to a fixed data-set shortcut. If it still collapses (e.g., \(w_v\approx1\) always), report that outcome and adjust the claim, not the result.

### 7.3 Synchronization is secondary

**[Established]** Cross-modal correspondence has precedent [1–4]. **[Proposed constraint]** Synchronization features affect the gate and final classifier but do not independently determine the label. This avoids the false premise “out of sync = fake”: dubbed real clips, speaker turns, music, and transmission delay may be misaligned, while advanced fakes may be well aligned.

## 8. Mathematical training objective and stages

Let \(p_f\), \(p_v\), and \(p_a\) be fused and unimodal probabilities. The initial objective is

\[
\mathcal L = \underbrace{\mathrm{BCE}(y,p_f)}_{\mathcal L_f}
+ \alpha[\mathrm{BCE}(y,p_v)+\mathrm{BCE}(y,p_a)]
+ \beta\mathcal L_{rel}
+ \gamma\mathcal L_{con}
+ \eta\mathcal L_{bal}.
\]

- \(\mathcal L_{con}\) is optional, low-cost InfoNCE on *real, aligned* pairs within a batch: matching audio/video temporal bins are positives and other clips/bins are negatives. It is omitted if it harms the small-data setting.
- \(\mathcal L_{bal}=\sum_m(w_m-0.5)^2\) is used only for a brief warm-up or diagnosed collapse; it must not force equal weights after training.
- Start with \(\alpha=0.25,\beta=0.1,\gamma=0.05,\eta=0\) and tune on validation only. These are starting values, not published constants.

### Staged training plan

1. **Data audit:** decode, validate labels, create group split, create a 200-clip visual/audio inspection sheet.
2. **Unimodal baselines:** train visual and audio branches separately; establish feasibility and error profiles.
3. **Fixed fusion:** concatenate or equal-weight pooled features; verify that fusion helps before introducing a gate.
4. **Reliability-aware fusion:** add modality heads, gate target, quality features, and modality-specific augmentation. Freeze encoders for initial 3–5 epochs if unstable, then unfreeze the last blocks.
5. **Optional self-supervised warm-up:** 5–10 epochs of real-only alignment contrastive learning, only if there are enough permitted real clips and an ablation is affordable.
6. **Distillation/quantization:** after the best reproducible float model is frozen; never compare a distilled model with a differently trained teacher without stating it.
7. **Calibration:** choose temperature on validation only; report raw and temperature-scaled test results separately.

## 9. Evaluation plan

### 9.1 Baselines

1. Visual-only compact encoder.
2. Audio-only compact encoder.
3. Late equal-logit average.
4. Feature concatenation + MLP (naïve fusion).
5. Fixed learned global weights (one \(w_v,w_a\) for all samples).
6. Reliability gate without synchronization features.
7. Reliability gate without quality features.
8. Full proposed system.

AVFF/AVoiD-DF are literature comparators, not required reimplementations. If an external published number is mentioned, preserve its original split/dataset and label it **not directly comparable**.

### 9.2 Tests

| Test | Design | Primary report |
|---|---|---|
| In-domain | Identity/source-disjoint clean test, same seen manipulation families. | AUROC, balanced accuracy, F1, ECE. |
| Cross-manipulation | Hold out a documented family/generator and its source identities before model selection. Repeat if ≥3 groups exist. | Per-held-out-family mean ± bootstrap 95% CI. |
| Cross-dataset | Train/evaluate only where modality labels and licenses are compatible. DFDC result is visual-transfer unless audio semantics are audited. | AUROC with no threshold retuning. |
| Video degradation | H.264 CRF 18/28/38, resize 720/480/360, mild blur, frame drop. | Change from clean metric and gate shift. |
| Audio degradation | AWGN at 20/10/0 dB proxy, codec/re-sampling, low-pass, dropout. | Change from clean metric and gate shift. |
| Joint/missing modality | Both degraded; zeroed audio or no-face flag with architecture-supported masking. | Safe degradation, not fabricated confidence. |

### 9.3 Metrics and reporting

- **Discrimination:** ROC-AUC (primary), PR-AUC for imbalanced settings, balanced accuracy, F1, precision, recall, confusion matrix at validation-chosen threshold.
- **Calibration:** Brier score, Expected Calibration Error (15 equal-mass bins), reliability diagram, negative log-likelihood; post-hoc temperature scaling only from validation.
- **Reliability behavior:** mean \(w_v,w_a\) by degradation level; correlation with degradation severity and unimodal loss; gate entropy; missing-modality failure rate. Weight shifts are diagnostic, not standalone proof.
- **Efficiency:** parameters, checkpoint size, peak GPU memory, p50/p95 per-window latency and windows/s on the named GPU and named CPU; include preprocessing separately and end-to-end.
- **Uncertainty:** stratified bootstrap 95% CIs over source identities, three seeds for final contenders, and per-manipulation breakdowns. Do not select the best seed as the headline score.

## 10. Lightweight optimization and computational feasibility

### Initial budget

**[Proposed target environment]** one 6–8 GB consumer GPU, 16 GB system RAM, PyTorch mixed precision. Begin with batch size 8 (2-second windows), gradient accumulation to effective batch 32, and cached 160×160 face crops/log-Mels. Actual batch size, latency, and memory are measurements to report, not promises.

| Choice | Why it is feasible | Trade-off |
|---|---|---|
| 16 frames / 2 s | Limits decoding and vision cost. | May miss long-range temporal clues. |
| MobileNet/DS-CNN encoders | Few million parameters instead of full video transformers. | Lower capacity; must not be compared as equal-compute to AVFF. |
| Precompute deterministic features | Enables iteration on a modest GPU. | Preserve provenance and avoid preprocessing leakage. |
| AMP + gradient accumulation | Reduces memory without changing effective batch target. | Requires numerical checks. |
| Freeze then partial fine-tune | Stabilizes small data and shortens first experiments. | May cap accuracy. |
| Distillation | Teacher (larger encoder or best full model) teaches compact student logits/features. | Extra training; report teacher and student separately. |
| Dynamic quantization / ONNX | Can reduce CPU inference cost after validation. | Recalibrate and remeasure after conversion. |

## 11. Implementation stack and project structure

- Python 3.10+, PyTorch/torchvision, torchaudio, OpenCV/FFmpeg, numpy/pandas/scikit-learn, albumentations, matplotlib/seaborn, MLflow or TensorBoard, and `uv`/conda lockfile.
- Pin package versions, GPU driver/CUDA versions, model weights/checksums, ffmpeg build, dataset release, and split manifest hash.
- Suggested structure: `data/` (ignored), `manifests/`, `src/{decode,preprocess,models,train,evaluate}`, `configs/`, `tests/`, `reports/`, and `artifacts/`.
- Unit tests: timestamp alignment, group split exclusivity, no duplicate hash across partitions, missing-modality mask, gate weights sum to one, metric implementation, and deterministic decode of a fixture clip.

## 12. Expected outcomes, failure cases, limitations, and risks

### Expected outcomes

The concrete deliverables are: cleaned manifests; trained baselines; full ablation table; degradation curves; calibration plots; latency profile; a short model card; reproducibility package; and a poster figure. The desired scientific result is either support for H1–H4 or a clear finding that reliability-aware fusion does not help under one or more conditions.

### Anticipated failure cases and mitigation

| Risk/failure | Why it matters | Mitigation |
|---|---|---|
| Gate always favors video/audio | Dataset shortcut or weak opposite branch. | Inspect loss/quality relationship, use modality dropout, retain ablation, report collapse. |
| FakeAVCeleb real/fake imbalance | Accuracy can be misleading. | Balanced sampling; AUROC/PR-AUC; class-weight sensitivity. |
| Source/identity leakage | Artificially high scores. | Group split before augmentation; hashes and source audit. |
| Synchronization misfires on real dubbing/music | False positives. | Treat as feature only; include non-speech and delay cases in error analysis. |
| DFDC audio label ambiguity | Invalid multimodal conclusion. | Restrict DFDC to visual role unless audited metadata supports audio claims. |
| Dataset access delay | Blocks main benchmark. | Start pipeline using official samples/allowed small subset; do not substitute scraped data. |
| Low-memory training instability | Delays experiments. | Smaller crops, cache, freeze encoders, reduce temporal bins, use one seed for debugging. |
| Domain/demographic bias | Harmful uneven performance. | Report available subgroup proxies cautiously; do not infer sensitive attributes from faces. |

### Limitations

This project detects patterns in benchmark distributions, not the intent behind media. It may fail on high-quality, synchronized multimodal fakes; real clips with dubbing/poor capture; unseen languages; unseen codec chains; multiple speakers; non-frontal faces; and deliberate adversarial perturbations. Reliability weights are learned model behavior, not calibrated physical quality scores. Dataset licensing and availability may limit reproducibility of raw data.

## 13. Reproducibility checklist

- [ ] Dataset version, accepted terms date, source manifest, and file hashes recorded.
- [ ] Group/identity/manipulation split manifest committed; all augmentations derive after split.
- [ ] All random seeds and determinism settings logged.
- [ ] Config file saved with every checkpoint and metric artifact.
- [ ] Validation-only threshold and temperature; untouched test set.
- [ ] Three final seeds and identity-bootstrap confidence intervals.
- [ ] Raw, calibrated, and optimized-runtime metrics kept distinct.
- [ ] Data not redistributed; scripts expect the user’s licensed local data path.
- [ ] Model card states intended use, non-use, dataset caveats, and known failures.

## 14. Twelve-week execution timeline

| Week | Milestone | Exit criterion |
|---|---|---|
| 1 | Read terms, request data, write data dictionary and ethics note. | Access status and non-redistribution plan documented. |
| 2 | Decoder, face/audio extraction, audit notebook. | 200 clips inspected; invalid cases quantified. |
| 3 | Group manifests, duplicate checks, degradation generator. | Zero known cross-split source overlap. |
| 4 | Visual and audio baselines. | Reproducible validation runs. |
| 5 | Fixed fusion and initial error analysis. | Fusion comparison table. |
| 6 | Reliability gate + quality features. | Gate diagnostics and first ablation. |
| 7 | Synchronization auxiliary and ablations. | Full vs no-sync/no-quality results. |
| 8 | Cross-manipulation protocol. | Held-out test locked before final training. |
| 9 | Compression/noise/missing-modality suite. | Robustness curves. |
| 10 | Calibration, latency, optional distillation. | Named-hardware profile. |
| 11 | Reruns, confidence intervals, failure review. | Final tables/figures. |
| 12 | Report, poster, code cleanup, model card. | Reproducibility checklist complete. |

## 15. Poster-ready summary

### Problem
Audio and visual streams in a deepfake clip are not always equally useful. Compression can damage faces; noise can damage speech; and simple feature concatenation may over-trust poor evidence.

### Proposed method
Two compact encoders extract aligned face-motion and log-Mel speech features. A small reliability estimator reads modality quality, unimodal uncertainty, and audio-visual consistency. It produces normalized weights \(w_v,w_a\) for reliability-aware fusion. Synchronization is an auxiliary cue, so the detector does not equate lip-sync mismatch with fakery.

### Evaluation
Compare visual-only, audio-only, equal-weight, concatenation, and gated fusion. Prevent leakage by splitting by source identity/video and holding out manipulation families. Stress test H.264 compression, resizing, blur, noise, codec changes, and missing modalities. Report AUROC, PR-AUC, F1, calibration, reliability diagrams, parameter count, and end-to-end latency.

### Contribution claim
An undergraduate-feasible, transparent reliability-aware fusion design and rigorous robustness protocol. The project tests—not assumes—that adaptive trust helps on unseen manipulations and degraded media.

## References

1. K. Chugh, P. Gupta, A. Dhall, and R. Subramanian, “[Not made for each other—Audio-Visual Dissonance-based Deepfake Detection and Localization](https://doi.org/10.1145/3394171.3413700),” *ACM Multimedia*, 2020.
2. T. Mittal et al., “[Emotions Don’t Lie: An Audio-Visual Deepfake Detection Method Using Affective Cues](https://arxiv.org/abs/2003.06711),” *ACM Multimedia*, 2020.
3. W. Yang et al., “[AVoiD-DF: Audio-Visual Joint Learning for Detecting Deepfake](https://doi.org/10.1109/TIFS.2023.3262148),” *IEEE Transactions on Information Forensics and Security*, 18, 2015–2029, 2023.
4. T. Oorloff et al., “[AVFF: Audio-Visual Feature Fusion for Video Deepfake Detection](https://openaccess.thecvf.com/content/CVPR2024/html/Oorloff_AVFF_Audio-Visual_Feature_Fusion_for_Video_Deepfake_Detection_CVPR_2024_paper.html),” *CVPR*, 2024, pp. 27102–27112.
5. S. Muppalla, S. Jia, and S. Lyu, “[Integrating Audio-Visual Features for Multimodal Deepfake Detection](https://arxiv.org/abs/2310.03827),” 2023.
6. J. Wang et al., “[AVT²-DWF: Improving Deepfake Detection with Audio-Visual Fusion and Dynamic Weighting Strategies](https://arxiv.org/abs/2403.14974),” arXiv:2403.14974, 2024. *Preprint; cite and characterize accordingly.*
7. H. Ilyas, A. Javed, and K. M. Malik, “AVFakeNet: A unified end-to-end Dense Swin Transformer deep learning model for audio–visual deepfakes detection,” *Applied Soft Computing*, 2023. Verify volume/pages via the publisher library before final submission.
8. H. Khalid, S. Tariq, M. Kim, and S. S. Woo, “[FakeAVCeleb: A Novel Audio-Video Multimodal Deepfake Dataset](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/d9d4f495e875a2e075a1a4a6e1b9770f-Abstract-round2.html),” *NeurIPS Datasets and Benchmarks Track*, 2021. Access: [official project repository](https://github.com/DASH-Lab/FakeAVCeleb).
9. B. Dolhansky et al., “[The DeepFake Detection Challenge (DFDC) Dataset](https://arxiv.org/abs/2006.07397),” arXiv:2006.07397, 2020; dataset access: [Meta DFDC page](https://ai.meta.com/datasets/dfdc/).

---

### Source note

This document was developed from the uploaded `deep-research-report.md` as the primary project brief, then corrected/expanded to distinguish literature-supported statements from implementable proposals. Dataset access, licenses, release versions, and bibliographic fields must be rechecked immediately before formal submission because they can change.
