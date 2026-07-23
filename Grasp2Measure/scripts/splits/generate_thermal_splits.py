from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


LEVEL_PATTERN = re.compile(r"Levelclass(?P<level>[^_]+)", flags=re.IGNORECASE)
CONTAINER_PATTERN = re.compile(r"ContainerModel(?P<container>[^_]+)", flags=re.IGNORECASE)


def level_key(path: Path) -> str:
    match = LEVEL_PATTERN.search(path.stem)
    if match:
        return match.group("level").lower()
    return "__unknown_level__"


def container_key(path: Path) -> str:
    match = CONTAINER_PATTERN.search(path.stem)
    if match:
        return match.group("container").lower()
    return "__unknown_container__"


def joint_stratum_key(path: Path) -> str:
    return f"{container_key(path)}__{level_key(path)}"


def collect_samples(data_root: Path, pattern: str) -> list[Path]:
    samples = sorted(
        path for path in data_root.glob(pattern)
        if path.is_file() and LEVEL_PATTERN.search(path.stem)
    )
    if not samples:
        raise SystemExit(f"No CSV files matched {data_root / pattern}")
    return samples


def stratified_folds(samples: list[Path], folds: int, seed: int) -> list[list[Path]]:
    grouped: dict[str, list[Path]] = defaultdict(list)
    for sample in samples:
        grouped[joint_stratum_key(sample)].append(sample)
    for key, group_samples in sorted(grouped.items()):
        if len(group_samples) < folds:
            raise SystemExit(f"Stratum {key} has only {len(group_samples)} samples; need at least {folds}.")

    rng = np.random.default_rng(seed)
    fold_paths: list[list[Path]] = [[] for _ in range(folds)]
    for key, group_samples in sorted(grouped.items()):
        shuffled = list(group_samples)
        rng.shuffle(shuffled)
        chunks = np.array_split(np.asarray(shuffled, dtype=object), folds)
        sizes = [len(chunk) for chunk in chunks]
        if max(sizes) - min(sizes) > 1:
            raise SystemExit(f"Unexpected imbalanced fold sizes for {key}: {sizes}")
        for fold_index, chunk in enumerate(chunks):
            fold_paths[fold_index].extend(chunk.tolist())
    return fold_paths


def write_cv_splits(split_dir: Path, data_root: Path, fold_paths: list[list[Path]]) -> None:
    all_paths = [path for fold in fold_paths for path in fold]
    for fold_index, test in enumerate(fold_paths):
        test_set = set(test)
        train = [path for path in all_paths if path not in test_set]
        fold_dir = split_dir / f"cv{len(fold_paths)}" / f"fold_{fold_index}"
        write_split(fold_dir / "train.csv", data_root, sorted(train))
        write_split(fold_dir / "val.csv", data_root, sorted(test))
        write_split(fold_dir / "test.csv", data_root, sorted(test))
        print(f"cv{len(fold_paths)} fold={fold_index} train={len(train)} val={len(test)} test={len(test)}")


def split_samples(
    samples: list[Path],
    data_root: Path,
    test_ratio: float,
    seed: int,
) -> tuple[list[Path], list[Path]]:
    grouped: dict[str, list[Path]] = defaultdict(list)
    for sample in samples:
        grouped[level_key(sample)].append(sample)

    rng = np.random.default_rng(seed)
    train: list[Path] = []
    test: list[Path] = []
    for level, level_samples in sorted(grouped.items()):
        shuffled = list(level_samples)
        rng.shuffle(shuffled)
        n_test = max(1, int(round(len(shuffled) * test_ratio)))
        if n_test >= len(shuffled):
            raise SystemExit(
                f"Level {level} has only {len(shuffled)} sample(s); "
                "cannot leave both train and test non-empty."
            )
        test.extend(shuffled[:n_test])
        train.extend(shuffled[n_test:])

    if not train:
        raise SystemExit("Split left no training samples")
    if not test:
        raise SystemExit("Split left no testing samples")
    return sorted(train), sorted(test)


def write_split(path: Path, data_root: Path, samples: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["path"])
        for sample in samples:
            writer.writerow([sample.relative_to(data_root).as_posix()])


def load_json_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as input_file:
        config = json.load(input_file)
    if not isinstance(config, dict):
        raise SystemExit(f"Config must be a JSON object: {path}")
    return config


def resolve_config_path(base_dir: Path, value: object | None) -> Path | None:
    if value is None:
        return None
    path = Path(str(value))
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def main() -> int:
    project_root = Path(__file__).resolve().parents[2]
    repo_root = project_root.parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "thermal_splits_config.example.json",
        help="Thermal dataset config JSON.",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=repo_root / "Workspace" / "C_model_training" / "thermal",
        help="Thermal dataset root.",
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=project_root / "splits" / "thermal",
        help="Output directory for train/val/test split CSV files.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="CSV directory. Defaults to <base-dir>/data/merged.",
    )
    parser.add_argument("--pattern", default="*.csv")
    parser.add_argument("--test-ratio", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--cv-folds", type=int, default=0, help="Also write ContainerModel+LevelClass stratified k-fold splits under <split-dir>/cvK.")
    args = parser.parse_args()

    base_dir = args.base_dir.resolve()
    config = load_json_config(args.config)
    split_config = config.get("split", {})
    if not isinstance(split_config, dict):
        raise SystemExit("Config field split must be an object")

    data_root = (
        args.data_root
        or resolve_config_path(base_dir, split_config.get("data_root"))
        or base_dir / "data" / "merged"
    ).resolve()
    pattern = str(split_config.get("pattern", args.pattern))
    test_ratio = float(args.test_ratio if args.test_ratio is not None else split_config.get("test_ratio", 0.15))
    seed = int(args.seed if args.seed is not None else split_config.get("seed", 42))
    if not 0.0 < test_ratio < 1.0:
        raise SystemExit("--test-ratio must be between 0 and 1")

    samples = collect_samples(data_root, pattern)
    train, test = split_samples(samples, data_root, test_ratio, seed)
    split_dir = args.split_dir.resolve()
    write_split(split_dir / "train.csv", data_root, train)
    write_split(split_dir / "val.csv", data_root, test)
    write_split(split_dir / "test.csv", data_root, test)
    print(f"train={len(train)} val={len(test)} test={len(test)}")
    if args.cv_folds:
        if args.cv_folds < 2:
            raise SystemExit("--cv-folds must be at least 2")
        write_cv_splits(split_dir, data_root, stratified_folds(samples, args.cv_folds, seed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
