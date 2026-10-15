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

1. Build the source table and identity-disjoint split manifests, then inspect the audit report:

   ```powershell
   python scripts/run_data_audit.py --config configs/data_config.yaml
   ```

   The command writes `manifests/source_table.csv`, `manifests/splits/{train,val,test}_manifest.csv`, and `manifests/data_audit_report.json`. Training reads these manifests and never creates or changes a split.

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
   python scripts/run_data_audit.py --output-dir reports/fixture-audit
   ```

The fixture audit checks decode and feature contracts only; it does not generate dataset manifests or train a model. Cross-manipulation, cross-dataset, and controlled degradation experiments remain separate study runs and are not included in the clean in-domain evaluation command.

## Data and modeling notes

- The default manifest builder supports the documented FakeAVCeleb directory layout and DFDC metadata layout. DFDC is filtered out of multimodal training and is reserved for visual-transfer use because its audio manipulation labels are not verified.
- Face crops use the configured OpenCV Haar detector as a local, no-weight-download baseline. Landmark alignment and a RetinaFace/MediaPipe backend are not currently implemented; set expectations accordingly when reporting experiments.
- PyTorch/torchvision pretrained weights are optional. If torchvision or its weights are unavailable, the visual encoder logs and uses its compact convolutional fallback; record that configuration with results.
- Feature caches/checkpoints/reports are local derived artifacts and are gitignored. Split CSVs are small metadata manifests and should be reviewed/versioned without including licensed media.

## Repository guidance

- [`AGENTS.md`](AGENTS.md) describes module boundaries and build constraints.
- [`docs/PROJECT_BRIEF.md`](docs/PROJECT_BRIEF.md) is the canonical technical protocol.
- `configs/*.yaml` holds data, model, training, and evaluation settings.
