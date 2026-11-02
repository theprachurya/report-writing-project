# PBL-2 Submission Package

**Course:** Report Writing (21CSC402P)  
**Assessment Date:** 16 October 2026  
**Marks:** 20  
**Project:** Reliability-Aware Audio-Visual Fusion for Generalizable Deepfake Detection

## Team
| Name | Register Number |
|------|----------------|
| Aditya Sahu | RA2311026010814 |
| Akash Cherian | RA2311026010852 |
| Prachurya Subhasmita | RA2311026010855 |

## Contents Required
- Technologies to implement the problem defined
- Result Analysis (preliminary)
- IEEE paper rough draft (Abstract through Result Analysis)

## Files in this Package

### Reports
| File | Description |
|------|-------------|
| `PBL2_Report.pdf` | **Primary PDF Report (16 pages, all diagrams embedded, print-ready)** |
| `PBL2_Report.docx` | **Primary Word Document (formatted with styled tables and embedded figures)** |
| `PBL2_Report.md` | Complete PBL-2 report source (Markdown) |
| `FP2_Reliability_Aware_AV_Deepfake_Detection_FINAL.pdf` | Faculty presentation report (18 pages, A4) |
| `FP2_Reliability_Aware_AV_Deepfake_Detection_FINAL.docx` | Same in Word format |

### IEEE Draft
| File | Description |
|------|-------------|
| `ieee_draft/pbl2_ieee_draft.tex` | IEEE conference paper LaTeX source |
| `ieee_draft/pbl2_ieee_draft.pdf` | Compiled IEEE paper (8 pages) |

### Figures
| File | Description |
|------|-------------|
| `figures/auroc_comparison.png` | Validation AUROC bar chart (4 models) |
| `figures/data_splits_detailed.png` | Dataset split sizes + class distribution |
| `figures/training_losses.png` | Training vs validation loss + loss components |
| `figures/technology_stack.png` | Technology stack layer diagram |
| `figures/architecture_flow.png` | System architecture flow diagram |
| `figures/staged_training.png` | 6-stage training pipeline |
| `figures/eval_protocol.png` | Evaluation protocol overview |
| `figures/system_architecture_diagram.png` | Detailed architecture (from generation script) |
| `figures/validation_auroc_comparison.png` | AUROC comparison (from generation script) |
| `figures/data_splits_distribution.png` | Data splits (from generation script) |

### Scripts
| File | Description |
|------|-------------|
| `generate_diagrams.py` | Python script to regenerate all diagrams |

## Key Preliminary Results (One-Epoch Validation)

| Model | Validation AUROC |
|-------|-----------------|
| Visual-only baseline | 0.7974 |
| Audio-only baseline | 0.8625 |
| Equal-weight fixed fusion | **0.9426** |
| Reliability-aware fusion | 0.7957 |

> **Note:** These are preliminary validation-only observations from one epoch with seed 42. No final test metrics, multi-seed estimates, or robustness results are included. The snapshot neither supports nor refutes the project hypotheses.
