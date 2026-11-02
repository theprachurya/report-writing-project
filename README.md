# Reliability-aware audio-visual deepfake detection

This repository implements a compact audio-visual detector with a per-window reliability gate. It follows the architecture, input contract, split protocol, and evaluation rules in [`docs/PROJECT_BRIEF.md`](docs/PROJECT_BRIEF.md), which matches the attached master plan. No benchmark results are claimed by the code.

## Setup

Use Python 3.10 or newer, install the packages in `requirements.txt`, and make `ffmpeg` and `ffprobe` available on `PATH` for audio tracks in video containers. Obtain access under each dataset's terms, download it yourself to a local volume, and point the config at that copy. Dataset media must stay outside this repository.

### Linux setup

Install FFmpeg with your distribution's package manager, then create an environment and configure the licensed dataset roots:

```bash
sudo apt-get install ffmpeg
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
export FAKEAVCELEB_ROOT="/path/to/FakeAVCeleb"
export DFDC_ROOT="/path/to/DFDC"
```

Set either dataset root only if you have access to that dataset. The code uses the FakeAVCeleb root for the main multimodal benchmark; DFDC is reserved for visual-transfer experiments unless its audio labels are audited.

### Obtain licensed datasets

- **FakeAVCeleb:** request access through the [official dataset page](https://sites.google.com/view/fakeavcelebdash-lab/download) and its linked request form. The form requires researcher and institution details and explicit agreement to the terms. Access is not automated by this project.
- **DFDC:** follow Meta's [official access page](https://ai.meta.com/datasets/dfdc/). It currently requires an AWS account, an IAM user with access keys, and the AWS account number. DFDC is used here for visual-transfer experiments; its audio labels are not treated as verified.

See [`docs/DATASETS.md`](docs/DATASETS.md) for alternative dataset candidates, modality caveats, access notes, and storage planning for the Linux environment.

The repository includes a local browser interface alongside its research CLI and notebook. FakeAVCeleb is already present in the local data area on the current machine. MLAAD-tiny is an optional audio-only external test set. Its generated files are non-commercial, and its bona-fide audio keeps the separate M-AILABS license. The downloader is an explicit command and defaults to ignored `data/raw/` (the root is configurable):

```bash
python -m pip install -r requirements-data.txt
python scripts/download_mlaad_tiny.py  # downloads the pinned German slice by default
```

The German slice is reserved for an exploratory external audio-only transfer probe. FakeAVCeleb's audio branch is trained against the clip-level fake label, while MLAAD-tiny labels bona-fide versus spoof audio; interpret these as different label semantics. MLAAD-tiny is not added to AV training manifests and must not be combined with paired AV results.

Do not put dataset files under tracked source directories or commit them. After access is granted, set roots outside the repository, for example:

```powershell
$env:FAKEAVCELEB_ROOT = 'E:\licensed-data\FakeAVCeleb'
$env:DFDC_ROOT = 'E:\licensed-data\DFDC'
```

The official forms require account, affiliation, or agreement details. Complete them yourself; this project does not submit forms, accept terms, or handle account credentials.

PowerShell example:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:FAKEAVCELEB_ROOT = 'D:\licensed-data\FakeAVCeleb'
```

You can also set `datasets.fakeavceleb.root` in `configs/data_config.yaml`. Keep local media under `data/raw/` or another untracked location.

## Run the project

### Open the local review interface

Start the browser app with a full audio-visual checkpoint:

```bash
python scripts/serve_web.py --checkpoint artifacts/checkpoints/reliability_fusion/best_model.pt
```

Then open `http://127.0.0.1:7860`. The server binds to the local machine by default. It scores the first configured 2-second window, chooses its decision threshold from the FakeAVCeleb validation split, and caches that threshold for the checkpoint and manifest. The first prediction can take longer while this threshold is fit. Upload limits are in `configs/web_app_config.yaml`; the app preprocesses each upload in a temporary directory and removes it after scoring. Use a full AV checkpoint; unimodal audio checkpoints cannot produce a fused prediction.

1. Build the source table and identity-disjoint split manifests, then inspect the audit report:

   ```powershell
   python scripts/run_data_audit.py --config configs/data_config.yaml
   ```

   This audit writes `manifests/source_table.csv`, the standard train/validation/test manifests, and `manifests/data_audit_report.json`. Training reads these manifests and never creates or changes a split.

   For a separate cross-manipulation experiment, freeze the configured `faceswap` family before training and build its manifests. The current split has 27 train, 4 validation, and 5 seen-test identities, with 464 held-out identities. Treat this run as exploratory because the seen-identity pool is too small for strong confirmatory generalization claims; inspect `split_metadata.json` before training:

   ```bash
   python scripts/build_cross_manip_split.py
   python -m src.train.train_reliability_fusion --manifest-dir manifests/splits/cross_manipulation_faceswap
   python scripts/run_full_evaluation.py --manifest-dir manifests/splits/cross_manipulation_faceswap --checkpoint <cross-manipulation-checkpoint>
   ```

   The cross-manipulation command writes a separate directory containing its train/validation/held-out manifests, a seen-family test manifest, and `split_metadata.json`; it leaves the standard in-domain split intact.

2. Train the unimodal baselines, followed by fixed fusion and reliability fusion:

   ```powershell
   python -m src.train.train_unimodal --modality visual
   python -m src.train.train_unimodal --modality audio
   python -m src.train.train_fixed_fusion --mode equal_weight
   python -m src.train.train_reliability_fusion
   ```

   Use `--epochs 1 --seed 42` for a short pipeline run. The first epoch extracts and caches features under ignored `data/cache/features/`; subsequent epochs reuse them. Checkpoints and their YAML configs are written under `artifacts/checkpoints/`.

3. Evaluate the selected reliability-fusion checkpoint:

   ```powershell
   python scripts/run_full_evaluation.py --checkpoint artifacts/checkpoints/reliability_fusion/best_model.pt
   ```

   Thresholds and temperature are fit on validation predictions; the test loader is traversed once. Raw and temperature-scaled metrics are saved separately in `reports/evaluation/evaluation_results.json`.

4. Run the CPU unit suite and fixture-only data smoke check:

   ```powershell
   python -m pytest tests -q
   python scripts/run_data_audit.py --fixture-only --output-dir reports/fixture-audit
   ```

The fixture audit checks decode and feature contracts only; it does not generate dataset manifests or train a model. Cross-manipulation, cross-dataset, and controlled degradation experiments remain separate study runs and are not included in the clean in-domain evaluation command.

5. Run the optional MLAAD-tiny audio-only transfer evaluation with an explicit trained checkpoint:

   ```bash
   python scripts/run_mlaad_audio_eval.py --checkpoint artifacts/checkpoints/unimodal_audio/best_model.pt
   ```

   The script chooses its threshold from the FakeAVCeleb validation audio head, does not use FakeAVCeleb's locked test split, and does not retune on MLAAD-tiny. Add `--limit-groups 16` only for an inference smoke run; limited output is not a benchmark result. Checkpoints currently present in this workspace are one-epoch development artifacts, not final reported models.

## Data and modeling notes

- The default manifest builder supports the documented FakeAVCeleb directory layout and DFDC metadata layout. DFDC is filtered out of multimodal training and is reserved for visual-transfer use because its audio manipulation labels are not verified.
- Face crops use the configured OpenCV Haar detector as a local, no-weight-download baseline. Landmark alignment and a RetinaFace/MediaPipe backend are not currently implemented; set expectations accordingly when reporting experiments.
- PyTorch/torchvision pretrained weights are optional. If torchvision or its weights are unavailable, the visual encoder logs and uses its compact convolutional fallback; record that configuration with results.
- Feature caches/checkpoints/reports are local derived artifacts and are gitignored. Split CSVs are small metadata manifests and should be reviewed/versioned without including licensed media.

## Repository guidance

- [`AGENTS.md`](AGENTS.md) describes module boundaries and build constraints.
- [`docs/PROJECT_BRIEF.md`](docs/PROJECT_BRIEF.md) is the canonical technical protocol.
- `configs/*.yaml` holds data, model, training, and evaluation settings.
