# Manuscript Figure Index

This directory contains the publication figures for the PeerJ Computer Science manuscript. SVG files are the editable vector masters; PDF files are the manuscript-ready vector exports; PNG files are high-resolution review copies.

| Manuscript figure | Figure title | Source experiment or artifact | Intended manuscript section | SVG | PDF | PNG | Status |
|---|---|---|---|---|---|---|---|
| Figure 1 | Synthetic Dataset Generation Pipeline | `configs/first_real_qr_10k_dataset.yaml`; dataset implementation; dataset manifest | Section 3.2 | [`figure1_dataset_pipeline.svg`](figure1_dataset_pipeline.svg) | [`figure1_dataset_pipeline.pdf`](figure1_dataset_pipeline.pdf) | [`figure1_dataset_pipeline.png`](figure1_dataset_pipeline.png) | Complete and validated |
| Figure 2 | Overall Mamba-TransQR Architecture | `configs/phase3_pilot.yaml`; model implementation; completed Phase 3 configuration | Section 3.4 | [`figure2_mambatransqr_architecture.svg`](figure2_mambatransqr_architecture.svg) | [`figure2_mambatransqr_architecture.pdf`](figure2_mambatransqr_architecture.pdf) | [`figure2_mambatransqr_architecture.png`](figure2_mambatransqr_architecture.png) | Complete and validated |
| Figure 3 | Training and Optimization Workflow | `configs/phase3_pilot.yaml`; loss and training implementation | Section 3.6 / Training strategy | [`figure3_training_workflow.svg`](figure3_training_workflow.svg) | [`figure3_training_workflow.pdf`](figure3_training_workflow.pdf) | [`figure3_training_workflow.png`](figure3_training_workflow.png) | Complete and validated |
| Figure 4 | Optimization Dynamics: (a) Training Loss, (b) Validation PSNR, and (c) Validation SSIM | `results/phase3_pilot/training_history.csv`; `results/phase3_pilot/validation_history.csv` | Section 4.1 | [`figure4_training_curves.svg`](figure4_training_curves.svg) | [`figure4_training_curves.pdf`](figure4_training_curves.pdf) | [`figure4_training_curves.png`](figure4_training_curves.png) | Complete and validated; 30 logged epochs, without smoothing or interpolation |
| Figure 5 | Combined Experimental Results: (a) Controlled Ablation Study and (b) Qualitative Restoration Results | `results/ablation/final/ablation_table.csv`; fixed-test-split samples `qr_008724_moderate`, `qr_001505_moderate`, `qr_000310_moderate`, and `qr_003747_moderate`; Phase 3 EMA best checkpoint | Section 4.2 | [`figure5_experimental_results.svg`](figure5_experimental_results.svg) | [`figure5_experimental_results.pdf`](figure5_experimental_results.pdf) | [`figure5_experimental_results.png`](figure5_experimental_results.png) | Complete; quantitative values verified; standalone restored sample files must be recovered for the external archive |

## Numbering note

The original validated dataset-pipeline and architecture assets were created with their figure numbers reversed. They were renamed, without regenerating or altering their contents, to follow manuscript order: the dataset pipeline is Figure 1 and the architecture is Figure 2.

## Figure 5 verification

The ablation panel uses only the completed controlled 30-epoch cohort. Its
displayed PSNR/SSIM values match `results/ablation/final/ablation_table.csv`
after rounding. The four qualitative inputs are members of the fixed test
split and selection seed 42 is stated in the figure. Their clean, damaged, and
metadata inputs remain in the local dataset and are listed in
`docs/peerj_archival_manifest.md`. The standalone restored PNG outputs used to
compose the figure were not found during the release audit; they must be
recovered or regenerated deterministically from the verified Phase 3
checkpoint before the external artifact deposit. No 15-epoch pilot value is
used.
