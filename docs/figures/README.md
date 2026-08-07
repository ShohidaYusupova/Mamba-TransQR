# Manuscript Figure Index

This directory contains the publication figures for the PeerJ Computer Science manuscript. SVG files are the editable vector masters; PDF files are the manuscript-ready vector exports; PNG files are high-resolution review copies.

| Manuscript figure | Figure title | Source experiment or artifact | Intended manuscript section | SVG | PDF | PNG | Status |
|---|---|---|---|---|---|---|---|
| Figure 1 | Synthetic Dataset Generation Pipeline | `configs/first_real_qr_10k_dataset.yaml`; `src/mambatransqr/data/dataset_builder.py`; `src/mambatransqr/data/qr_generator.py`; `src/mambatransqr/data/qr_degradation.py`; `datasets/first_real_qr_10k/dataset_manifest.csv` | Section 3.2 | [`figure1_dataset_pipeline.svg`](figure1_dataset_pipeline.svg) | [`figure1_dataset_pipeline.pdf`](figure1_dataset_pipeline.pdf) | [`figure1_dataset_pipeline.png`](figure1_dataset_pipeline.png) | Complete and validated |
| Figure 2 | Overall Mamba-TransQR Architecture | `configs/phase3_pilot.yaml`; `src/mambatransqr/models/transqr.py`; `src/mambatransqr/models/transformer_block.py`; `src/mambatransqr/models/mamba_block.py`; `src/mambatransqr/models/fusion_block.py`; `src/mambatransqr/models/decoder.py`; completed Phase 3 model configuration | Section 3.4 | [`figure2_mambatransqr_architecture.svg`](figure2_mambatransqr_architecture.svg) | [`figure2_mambatransqr_architecture.pdf`](figure2_mambatransqr_architecture.pdf) | [`figure2_mambatransqr_architecture.png`](figure2_mambatransqr_architecture.png) | Complete and validated |
| Figure 3 | Training and Optimization Workflow | `configs/phase3_pilot.yaml`; `src/mambatransqr/losses/multiscale.py`; `src/mambatransqr/training/optimizer.py`; `src/mambatransqr/training/scheduler.py`; `src/mambatransqr/training/ema.py`; `src/mambatransqr/training/trainer.py` | Section 3.6 / Training strategy | [`figure3_training_workflow.svg`](figure3_training_workflow.svg) | [`figure3_training_workflow.pdf`](figure3_training_workflow.pdf) | [`figure3_training_workflow.png`](figure3_training_workflow.png) | Complete and validated |
| Figure 4 | Optimization Dynamics: (a) Training Loss, (b) Validation PSNR, and (c) Validation SSIM | `results/ablation/full_model/training_history.csv`; `results/ablation/full_model/validation_history.csv` (byte-identical to the completed `results/phase3_pilot/` histories) | Section 4.1 | [`figure4_training_curves.svg`](figure4_training_curves.svg) | [`figure4_training_curves.pdf`](figure4_training_curves.pdf) | [`figure4_training_curves.png`](figure4_training_curves.png) | Complete and validated; 30 logged epochs, without smoothing or interpolation |
| Figure 5 | Combined Experimental Results: (a) Controlled Ablation Study and (b) Qualitative Restoration Results | Planned sources: completed controlled 30-epoch ablation artifacts under `results/ablation/`; validated qualitative restoration examples are not yet packaged | Section 4.2 | Pending | Pending | Pending | **Pending — not generated; no qualitative examples or plots have been fabricated** |

## Numbering note

The original validated dataset-pipeline and architecture assets were created with their figure numbers reversed. They were renamed, without regenerating or altering their contents, to follow manuscript order: the dataset pipeline is Figure 1 and the architecture is Figure 2.

## Figure 5 constraint

Figure 5 must be generated only after suitable qualitative restoration examples are selected from completed experimental outputs and verified against their source images. Its ablation panel must use only the completed controlled 30-epoch ablation results. Archived 15-epoch pilot values must not be included in the publication panel.
