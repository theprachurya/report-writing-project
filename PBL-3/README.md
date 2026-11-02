# PBL-3 Submission Package

**Course:** Report Writing (21CSC402P)  
**Assessment Date:** 27 October 2026  
**Marks:** 20  
**Project:** Reliability-Aware Audio-Visual Fusion for Generalizable Deepfake Detection

## Team
| Name | Register Number |
|------|----------------|
| Aditya Sahu | RA2311026010814 |
| Akash Cherian | RA2311026010852 |
| Prachurya Subhasmita | RA2311026010855 |

## Contents Required
1. Result Analysis (Performance Metrics)
2. Conclusion, Future Enhancement & Final Report Draft

## Files in this Package

### Reports
| File | Description |
|------|-------------|
| `PBL3_Report.pdf` | **Primary PDF Report (19 pages, all diagrams embedded, print-ready)** |
| `PBL3_Report.docx` | **Primary Word Document (formatted with styled tables and embedded figures)** |
| `PBL3_Report.md` | Complete PBL-3 report source (Markdown) |
| `PBL3_Results_Analysis_and_Final_Report_Draft.pdf` | Brief draft PDF (8 pages) |
| `PBL3_Results_Analysis_and_Final_Report_Draft.docx` | Brief draft Word format |

### Presentation
| File | Description |
|------|-------------|
| `presentation/PBL3_Presentation.pdf` | Slide deck (7 pages) |
| `presentation/PBL3_Presentation.html` | Interactive HTML presentation |

### Figures
| File | Description |
|------|-------------|
| `figures/auroc_with_deltas.png` | AUROC comparison + pairwise difference heatmap |
| `figures/mlaad_per_generator.png` | Per-generator AUROC for MLAAD transfer probe |
| `figures/loss_components.png` | Loss component breakdown (all 4 models) |
| `figures/hypothesis_status.png` | H1-H4 evaluation status dashboard |
| `figures/completion_timeline.png` | Gantt-style completion plan (27 Oct → 10 Nov) |
| `figures/mlaad_confusion.png` | MLAAD confusion matrix with metrics |
| `figures/system_architecture_diagram.png` | System architecture |
| `figures/validation_auroc_comparison.png` | Validation AUROC comparison |
| `figures/data_splits_distribution.png` | Data splits distribution |
| `figures/mlaad_audio_transfer.png` | MLAAD audio transfer analysis |

### Scripts
| File | Description |
|------|-------------|
| `generate_diagrams.py` | Python script to regenerate all PBL-3 diagrams |

## Key Results Summary

### FakeAVCeleb Validation Snapshot (One Epoch, Seed 42)

| Model | Val AUROC | Δ from Fixed Fusion |
|-------|-----------|-------------------|
| Visual-only | 0.7974 | −0.1452 |
| Audio-only | 0.8625 | −0.0801 |
| Equal-weight fixed fusion | **0.9426** | — |
| Reliability-aware fusion | 0.7957 | −0.1469 |

### MLAAD-tiny German Audio Transfer Probe

| Metric | Value |
|--------|-------|
| AUROC | 0.3777 |
| Bootstrap 95% CI | [0.3588, 0.3964] |
| Balanced Accuracy | 0.4827 |
| F1 | 0.6561 |
| Recall / Specificity | 0.8820 / 0.0833 |

### Hypothesis Status
All four hypotheses (H1–H4) remain **NOT EVALUATED**. The primary test split (4,390 clips) has not been touched.

## Completion Plan (27 Oct → 10 Nov 2026)

| Period | Work |
|--------|------|
| 27–30 Oct | Full-length training runs |
| 31 Oct–2 Nov | Reliability fusion + gate diagnostics |
| 3–5 Nov | Degradation + cross-manipulation evaluation |
| 6–7 Nov | Final model selection + calibration |
| 8 Nov | Primary test (once only) |
| 9 Nov | Report + viva preparation |
| **10 Nov** | **VIVA ASSESSMENT** |
