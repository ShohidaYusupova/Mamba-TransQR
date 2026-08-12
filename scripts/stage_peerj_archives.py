"""Build deterministic local PeerJ/Zenodo staging archives without uploading."""

from __future__ import annotations

import argparse
import os
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXED_TIME = (2026, 8, 12, 0, 0, 0)


def _write_zip(output: Path, entries: list[tuple[Path, str]]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for source, archive_name in sorted(entries, key=lambda item: item[1]):
            info = zipfile.ZipInfo(archive_name.replace("\\", "/"), FIXED_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, source.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def _tree(directory: Path, prefix: str) -> list[tuple[Path, str]]:
    return [
        (path, f"{prefix}/{path.relative_to(directory).as_posix()}")
        for path in directory.rglob("*")
        if path.is_file()
    ]


def _tracked_results() -> list[tuple[Path, str]]:
    names = subprocess.check_output(
        ["git", "ls-files", "results"], cwd=ROOT, text=True
    ).splitlines()
    entries = []
    for name in names:
        path = ROOT / name
        if path.is_file():
            entries.append((path, name))
    for directory in sorted((ROOT / "results" / "ablation").iterdir()):
        if not directory.is_dir() or directory.name in {"final"}:
            continue
        for filename in (
            "benchmark_summary.csv",
            "resolved_config.yaml",
            "run_metadata.json",
            "training_history.csv",
            "validation_history.csv",
        ):
            path = directory / filename
            if path.is_file():
                entries.append((path, path.relative_to(ROOT).as_posix()))
    return entries


def _source_tree() -> list[tuple[Path, str]]:
    names = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        text=True,
    ).splitlines()
    excluded_prefix = "release/peerj_v1.0.0/"
    entries = []
    for name in names:
        if name.startswith(excluded_prefix):
            continue
        path = ROOT / name
        if path.is_file():
            entries.append((path, f"Mamba-TransQR-1.0.0/{name}"))
    return entries


def stage() -> None:
    release = ROOT / "release" / "peerj_v1.0.0"
    dataset = ROOT / "datasets" / "first_real_qr_10k"
    checkpoints = [
        ROOT / "checkpoints" / "phase3_pilot" / "best.pt",
        ROOT / "checkpoints" / "ablation" / "without_mamba" / "best.pt",
        ROOT / "checkpoints" / "ablation" / "without_qr_structure_loss" / "best.pt",
        ROOT / "checkpoints" / "ablation_pre_rerun" / "without_decode_consistency_loss" / "best.pt",
        ROOT / "checkpoints" / "ablation_pre_rerun" / "without_ema" / "best.pt",
        ROOT / "checkpoints" / "ablation_pre_rerun" / "without_refinement_decoder" / "best.pt",
    ]
    checkpoint_names = [
        "full_phase3_best.pt",
        "without_mamba_best.pt",
        "without_qr_structure_loss_best.pt",
        "without_decode_consistency_loss_best.pt",
        "without_ema_best.pt",
        "without_refinement_decoder_best.pt",
    ]
    figure_entries = _tree(ROOT / "docs" / "figures" / "figure5_samples", "figure5_samples")
    for suffix in ("pdf", "png", "svg"):
        path = ROOT / "docs" / "figures" / f"figure5_experimental_results.{suffix}"
        figure_entries.append((path, f"figure5_experimental_results.{suffix}"))

    _write_zip(
        release
        / "source_and_reproducibility"
        / "mambatransqr-peerj-source-and-reproducibility.zip",
        _source_tree(),
    )

    _write_zip(
        release / "dataset_or_dataset_manifest" / "mambatransqr-peerj-dataset.zip",
        _tree(dataset, "first_real_qr_10k"),
    )
    _write_zip(
        release / "checkpoints" / "mambatransqr-peerj-checkpoints.zip",
        list(zip(checkpoints, checkpoint_names, strict=True)),
    )
    _write_zip(
        release / "manuscript_results" / "mambatransqr-peerj-results.zip",
        _tracked_results(),
    )
    _write_zip(
        release / "figure_supporting_data" / "mambatransqr-peerj-figure5-data.zip",
        figure_entries,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    os.chdir(ROOT)
    stage()


if __name__ == "__main__":
    main()
