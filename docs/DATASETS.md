# Dataset access and storage plan

**Reviewed:** 2026-10-15

**Project:** Reliability-Aware Audio-Visual Fusion for Generalizable Deepfake Detection

This note records the project’s current dataset roles, possible additions, access cautions, and storage priorities for the planned Linux environment. It does not change the experiment protocol in [`PROJECT_BRIEF.md`](PROJECT_BRIEF.md). A candidate becomes part of the actual protocol only after its labels, terms, identities, and split compatibility have been audited and the brief is updated.

## Current project datasets

| Dataset | Project role | Current notes |
|---|---|---|
| [FakeAVCeleb](https://github.com/DASH-Lab/FakeAVCeleb) | Main multimodal benchmark | A project member reports that access has been granted. The exact release, package layout, license terms, labels, and disk footprint still need to be recorded from the authorized copy. Access for one member does not grant access to other users. |
| [DFDC](https://ai.meta.com/datasets/dfdc/) | Optional visual pretraining or visual-transfer evaluation | Keep it visual-only unless the audio provenance and labels are separately verified. Meta describes a 124k-video full version and requires an AWS account and IAM access setup. |

Request-form access, a public code repository, or a no-cost download should not be treated as an unrestricted media license. Read the dataset’s own terms before downloading, processing, or sharing anything. A repository’s code license does not automatically apply to its dataset.

## Candidate alternatives

These are candidates to investigate, not datasets already approved for training. “AV” means the source documents describe manipulations in audio, video, or both. Audio-only and visual-only results must remain distinct from the project’s paired audio-visual evaluation.

| Dataset | Modalities and possible role | Access and storage notes | Priority for this project |
|---|---|---|---|
| [LAV-DF](https://github.com/ControlNet/LAV-DF) | AV manipulation and temporal-localization benchmark; the closest additional match to the project | The official repository requires agreement to dataset terms. Measure the actual download and extracted sizes before choosing it. Rebuild identity-disjoint manifests; do not assume the supplied split meets this project’s requirements. | **First AV alternative to assess** |
| [AV-Deepfake1M / AV-Deepfake1M++](https://github.com/ControlNet/AV-Deepfake1M) | Large-scale audio-only, video-only, and joint manipulation data with segment metadata | Challenge-hosted access and dataset terms apply. The repository describes over 1M videos, and announces a 2M-video ++ release. Full storage requirements are not established here; do not download to the 130 GB system until measured. | Later, preferably with external storage |
| [FaceForensics++](https://github.com/ondyari/faceforensics) | Visual manipulation training or transfer evaluation | Access is controlled by the project’s request process and terms. Treat as visual-only for this study. | Optional visual benchmark |
| [Celeb-DF v2](https://cse.buffalo.edu/~siweilyu/celeb-deepfakeforensics) | Visual face-swap transfer evaluation | Follow the project’s access instructions and terms. It is not a paired fake-audio dataset. | Optional visual benchmark |
| [KoDF](https://github.com/deepbrainai-research/kodf) | Visual-transfer evaluation with Korean video content | Access and the currently distributed release are governed by the official project instructions. Treat as visual-only unless audio manipulation labels are independently established. | Optional domain-diversity benchmark |
| [DeeperForensics-1.0](https://github.com/EndlessSora/DeeperForensics-1.0) | Visual robustness under real-world perturbations | Non-commercial research terms and request approval apply. Its official download guide lists about **284 GB**, so the full release does not fit this machine. | Defer unless external storage is available |
| [MLAAD-tiny](https://huggingface.co/datasets/mueller91/MLAAD-tiny) | Audio-only branch development or out-of-domain audio evaluation | The dataset card lists **3.54 GB** and CC BY-NC 4.0. Check the card and included notices for source-specific conditions. It cannot provide a paired AV result. | Best small audio candidate |
| [WaveFake](https://zenodo.org/records/5642694) | Audio-only cross-generator evaluation | The record lists **28.9 GB** for the generated-audio archive and a CC BY-SA 4.0 license. Review the datasheet and source-data conditions; keep its result separate from AV metrics. | Optional if space remains |
| [ASVspoof 5](https://zenodo.org/records/14498691) | Audio-only spoof/deepfake research | Zenodo lists **142.3 GB** and asks users to read the README and license. The full corpus exceeds the Linux disk capacity. | Do not place the full set on this machine |

The catalog favors datasets with official project, paper, or archive pages. Access rules and releases can change; recheck the linked terms when acquisition is planned. “Free” may mean no purchase price while still requiring approval, attribution, non-commercial use, or other restrictions.

## Disk plan for the Linux environment

The available disk is approximately **130 GB total**, not 130 GB of dataset space. Until the actual free-space report is known, cap dataset media and derived caches at roughly **70–80 GB combined**, leaving room for Linux, environments, temporary decode files, checkpoints, and reports.

1. Check free space with `df -h` and measure each downloaded and extracted dataset with `du -sh`.
2. Check both archive size and extracted size before downloading. Extraction may temporarily require both copies at once.
3. Start with FakeAVCeleb. Add at most one additional AV dataset after measuring it; add MLAAD-tiny only if the remaining budget allows. Prefer an external drive for larger collections.
4. Do not retain duplicate raw, extracted, and uncompressed frame/audio copies without a measured need. Treat feature caches as storage costs too; make them reproducible and disposable.
5. If a subset is necessary, select it deterministically, record the selection seed and source IDs, and create versioned manifests. Never make ad hoc splits inside training code.

More datasets do not automatically make the study stronger. Keep in-domain, cross-dataset, audio-only, visual-only, and degradation results separate, and check for identity overlap and label compatibility before interpreting a transfer result.

## Linux handoff and recordkeeping

The README currently documents `FAKEAVCELEB_ROOT` and `DFDC_ROOT`, with dataset roots kept outside tracked source files. Set local paths in the Linux environment or supported config; do not commit machine-specific paths. Additional candidate datasets are not necessarily wired into the current loader.

For each authorized local copy, record the following in a local/private acquisition record and commit only the portions allowed by the dataset terms:

- official dataset name, release/version, citation, and source URL;
- date access was confirmed and the applicable terms/license version;
- archive and extracted sizes, plus checksums where permitted;
- actual label mapping and metadata fields used by the project;
- identity/source grouping and the resulting split-manifest version/hash.

Do **not** commit videos, audio, frames, derived media/features, credentials, private download links, signed access forms, or personal access-request information. Public manifests or hashes should be committed only when the dataset terms permit them.

### Source pages

- FakeAVCeleb: [official repository](https://github.com/DASH-Lab/FakeAVCeleb) and [official access page](https://sites.google.com/view/fakeavcelebdash-lab/download).
- DFDC: [Meta dataset page](https://ai.meta.com/datasets/dfdc/).
- LAV-DF: [official repository and terms](https://github.com/ControlNet/LAV-DF); [paper](https://doi.org/10.1016/j.cviu.2023.103818).
- AV-Deepfake1M: [official repository](https://github.com/ControlNet/AV-Deepfake1M); [AV-Deepfake1M++ paper](https://arxiv.org/abs/2507.20579).
- FaceForensics++: [official repository](https://github.com/ondyari/faceforensics).
- Celeb-DF v2: [official project page](https://cse.buffalo.edu/~siweilyu/celeb-deepfakeforensics).
- KoDF: [official repository](https://github.com/deepbrainai-research/kodf).
- DeeperForensics-1.0: [official repository](https://github.com/EndlessSora/DeeperForensics-1.0) and [dataset size/access guide](https://github.com/189569400/DeeperForensics-1.0/blob/master/dataset/README.md).
- MLAAD-tiny: [dataset card and license](https://huggingface.co/datasets/mueller91/MLAAD-tiny).
- WaveFake: [Zenodo record](https://zenodo.org/records/5642694) and [dataset datasheet](https://zenodo.org/records/5642694/files/datasheet.pdf).
- ASVspoof 5: [Zenodo record, README, and license](https://zenodo.org/records/14498691).
