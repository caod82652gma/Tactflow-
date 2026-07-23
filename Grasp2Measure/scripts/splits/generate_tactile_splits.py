from __future__ import annotations

import argparse
import csv
import fnmatch
import json
from pathlib import Path
from typing import Any

import numpy as np


LABELS = ("15ml", "50ml", "100ml", "200ml")
SPLITS = ("train", "val", "test")


class Sample:
    def __init__(self, path: Path, label: str, batch: str | None) -> None:
        self.path = path
        self.label = label
        self.batch = batch

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def stem(self) -> str:
        return self.path.stem

    def split_path(self) -> str:
        return f"{self.label}/{self.name}"


def collect_samples(dataset_root: Path, variant: str) -> dict[str, list[Path]]:
    data_root = dataset_root / "data" / variant
    samples: dict[str, list[Path]] = {}
    for label in LABELS:
        label_dir = data_root / label
        if not label_dir.exists():
            raise SystemExit(f"Missing data directory: {label_dir}")
        paths = sorted(label_dir.glob("*.csv"))
        if not paths:
            raise SystemExit(f"No CSV files found in {label_dir}")
        samples[label] = paths
    return samples


def collect_batch_lookup(dataset_root: Path) -> dict[tuple[str, str], str]:
    lookup: dict[tuple[str, str], str] = {}
    rawdata_root = dataset_root / "rawdata"
    if not rawdata_root.exists():
        return lookup
    for batch_dir in sorted(rawdata_root.iterdir()):
        if not batch_dir.is_dir():
            continue
        for label in LABELS:
            label_dir = batch_dir / label
            if not label_dir.exists():
                continue
            for raw_file in label_dir.glob("*.bin"):
                lookup[(label, f"{raw_file.stem}.csv")] = batch_dir.name
    return lookup


def collect_sample_records(dataset_root: Path, variant: str) -> list[Sample]:
    batch_lookup = collect_batch_lookup(dataset_root)
    records: list[Sample] = []
    for label, paths in collect_samples(dataset_root, variant).items():
        for path in paths:
            records.append(Sample(path, label, batch_lookup.get((label, path.name))))
    return records


def split_label_paths(
    paths: list[Path],
    rng: np.random.Generator,
    test_ratio: float,
) -> tuple[list[Path], list[Path], list[Path]]:
    shuffled = list(paths)
    rng.shuffle(shuffled)
    n_total = len(shuffled)
    n_test = max(1, int(round(n_total * test_ratio)))
    test = shuffled[:n_test]
    train = shuffled[n_test:]
    if not train:
        raise SystemExit("Split ratios left no training samples")
    return train, list(test), test


def split_random(
    samples_by_label: dict[str, list[Path]],
    rng: np.random.Generator,
    test_ratio: float,
) -> dict[str, list[Path]]:
    split_samples: dict[str, list[Path]] = {name: [] for name in SPLITS}
    for paths in samples_by_label.values():
        label_train, label_val, label_test = split_label_paths(
            paths,
            rng,
            test_ratio,
        )
        split_samples["train"].extend(label_train)
        split_samples["val"].extend(label_val)
        split_samples["test"].extend(label_test)
    return split_samples


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as input_file:
        config = json.load(input_file)
    if not isinstance(config, dict):
        raise SystemExit(f"Split config must be a JSON object: {path}")
    return config


def get_rule_list(config: dict[str, Any], field_name: str, *aliases: str) -> list[Any]:
    rules: list[Any] = []
    found = False
    for name in (field_name, *aliases):
        value = config.get(name, [])
        if name in config:
            found = True
        if not isinstance(value, list):
            raise SystemExit(f"{name} must be a list")
        rules.extend(value)
    if not found:
        return []
    return rules


def sample_matches_rule(sample: Sample, rule: dict[str, Any]) -> bool:
    label = rule.get("label")
    if label is not None and sample.label != label:
        return False

    batch = rule.get("batch")
    if batch is not None and sample.batch != batch:
        return False

    name_glob = rule.get("name_glob")
    if name_glob is not None and not fnmatch.fnmatch(sample.name, str(name_glob)):
        return False

    stem_glob = rule.get("stem_glob")
    if stem_glob is not None and not fnmatch.fnmatch(sample.stem, str(stem_glob)):
        return False

    path_glob = rule.get("path_glob")
    if path_glob is not None and not fnmatch.fnmatch(sample.split_path(), str(path_glob)):
        return False

    return True


def normalize_required_name(value: str) -> str:
    name = Path(value).name
    if name.lower().endswith(".bin"):
        return f"{Path(name).stem}.csv"
    return name


def select_required_names(
    samples: list[Sample],
    names: list[Any],
    field_name: str,
) -> set[Path]:
    by_name: dict[str, list[Sample]] = {}
    by_split_path: dict[str, Sample] = {}
    for sample in samples:
        by_name.setdefault(sample.name, []).append(sample)
        by_split_path[sample.split_path()] = sample

    selected: set[Path] = set()
    missing: list[str] = []
    ambiguous: list[str] = []
    for value in names:
        raw_name = str(value).replace("\\", "/")
        normalized = normalize_required_name(raw_name)
        sample = by_split_path.get(normalized)
        if sample is not None:
            selected.add(sample.path)
            continue

        matches = by_name.get(normalized, [])
        if len(matches) == 1:
            selected.add(matches[0].path)
        elif len(matches) > 1:
            ambiguous.append(raw_name)
        else:
            missing.append(raw_name)

    if missing:
        raise SystemExit(
            f"{field_name} contains names not found in data: "
            f"{', '.join(missing[:10])}"
        )
    if ambiguous:
        raise SystemExit(
            f"{field_name} contains ambiguous names. "
            f"Use label/name form like 15ml/file.csv: {', '.join(ambiguous[:10])}"
        )
    return selected


def select_rule_samples(
    samples: list[Sample],
    rules: list[Any],
    rng: np.random.Generator,
    field_name: str,
) -> set[Path]:
    selected: set[Path] = set()
    for rule in rules:
        if isinstance(rule, str):
            selected.update(select_required_names(samples, [rule], field_name))
            continue
        if not isinstance(rule, dict):
            raise SystemExit(f"{field_name} entries must be strings or objects")
        names = rule.get("names")
        if names is not None:
            if not isinstance(names, list):
                raise SystemExit(f"{field_name} names must be a list")
            selected.update(select_required_names(samples, names, field_name))
            continue
        matches = [
            sample for sample in samples
            if sample.path not in selected and sample_matches_rule(sample, rule)
        ]
        limit = rule.get("limit")
        if limit is not None:
            limit_int = int(limit)
            if limit_int < 0:
                raise SystemExit(f"{field_name} limit must be non-negative")
            rng.shuffle(matches)
            matches = matches[:limit_int]
        min_count = int(rule.get("min_count", 0))
        if len(matches) < min_count:
            raise SystemExit(
                f"{field_name} rule matched {len(matches)} samples, "
                f"less than min_count={min_count}: {rule}"
            )
        selected.update(sample.path for sample in matches)
    return selected


def move_required_samples(
    train: list[Sample],
    test: list[Sample],
    required_train: set[Path],
    required_test: set[Path],
) -> tuple[list[Sample], list[Sample]]:
    train_by_path = {sample.path: sample for sample in train}
    test_by_path = {sample.path: sample for sample in test}
    all_by_path = train_by_path | test_by_path

    final_train = [
        sample for sample in train
        if sample.path not in required_test
    ]
    final_test = [
        sample for sample in test
        if sample.path not in required_train
    ]

    final_train_paths = {sample.path for sample in final_train}
    final_test_paths = {sample.path for sample in final_test}
    for path in sorted(required_train):
        if path not in final_train_paths:
            final_train.append(all_by_path[path])
    for path in sorted(required_test):
        if path not in final_test_paths:
            final_test.append(all_by_path[path])

    return final_train, final_test


def allocate_by_batch_ratios(
    samples: list[Sample],
    batch_ratios: dict[str, Any],
    rng: np.random.Generator,
    default_train_ratio: float,
    default_test_ratio: float,
) -> tuple[list[Sample], list[Sample]]:
    train: list[Sample] = []
    test: list[Sample] = []
    grouped: dict[str, list[Sample]] = {}
    for sample in samples:
        grouped.setdefault(sample.batch or "__unknown_batch__", []).append(sample)

    for batch, batch_samples in sorted(grouped.items()):
        rng.shuffle(batch_samples)
        ratios = batch_ratios.get(batch, batch_ratios.get("*", {}))
        if ratios:
            train_ratio = float(ratios.get("train", 0.0))
            test_ratio = float(ratios.get("test", 0.0))
        else:
            train_ratio = default_train_ratio
            test_ratio = default_test_ratio
        if train_ratio < 0 or test_ratio < 0:
            raise SystemExit(f"Batch {batch} has negative train/test ratio")
        if train_ratio + test_ratio > 1.0:
            raise SystemExit(f"Batch {batch} train/test ratios sum to more than 1")
        if train_ratio + test_ratio <= 0:
            raise SystemExit(f"Batch {batch} has no positive train/test ratio")
        n_train = int(len(batch_samples) * train_ratio)
        n_test = int(len(batch_samples) * test_ratio)
        train.extend(batch_samples[:n_train])
        test.extend(batch_samples[n_train:n_train + n_test])
    return train, test


def split_constrained(
    samples: list[Sample],
    config: dict[str, Any],
    rng: np.random.Generator,
    default_test_ratio: float,
) -> dict[str, list[Path]]:
    train_required_rules = get_rule_list(config, "train_required")
    test_required_rules = get_rule_list(config, "test_required")
    train_unrequired_rules = get_rule_list(config, "train_unrequired", "train_unrequire")
    test_unrequired_rules = get_rule_list(config, "test_unrequired", "test_unrequire")

    required_train = select_rule_samples(samples, train_required_rules, rng, "train_required")
    required_test = select_rule_samples(samples, test_required_rules, rng, "test_required")
    unrequired_train = select_rule_samples(samples, train_unrequired_rules, rng, "train_unrequired")
    unrequired_test = select_rule_samples(samples, test_unrequired_rules, rng, "test_unrequired")
    overlap = required_train & required_test
    if overlap:
        names = ", ".join(sorted(path.name for path in overlap)[:10])
        raise SystemExit(f"Samples required by both train and test: {names}")
    unrequired = unrequired_train | unrequired_test
    required_unrequired_overlap = (required_train | required_test) & unrequired
    if required_unrequired_overlap:
        names = ", ".join(sorted(path.name for path in required_unrequired_overlap)[:10])
        raise SystemExit(f"Samples listed as both required and unrequired: {names}")

    batch_ratios = config.get("batch_ratios", {})
    if not isinstance(batch_ratios, dict):
        raise SystemExit("batch_ratios must be an object")
    train, test = allocate_by_batch_ratios(
        samples,
        batch_ratios,
        rng,
        default_train_ratio=max(0.0, 1.0 - default_test_ratio),
        default_test_ratio=default_test_ratio,
    )
    train, test = move_required_samples(train, test, required_train, required_test)

    train_paths = [sample.path for sample in train if sample.path not in unrequired]
    test_paths = [sample.path for sample in test if sample.path not in unrequired]
    if not train_paths:
        raise SystemExit("Constrained split left no training samples")
    if not test_paths:
        raise SystemExit("Constrained split left no testing samples")
    return {
        "train": train_paths,
        "val": list(test_paths),
        "test": test_paths,
    }


def write_split(path: Path, dataset_root: Path, samples: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["path"])
        for sample in sorted(samples):
            writer.writerow([sample.relative_to(dataset_root / "data" / sample.parents[1].name).as_posix()])


def stratified_folds(records: list[Sample], folds: int, seed: int) -> list[list[Path]]:
    grouped: dict[str, list[Sample]] = {}
    for record in records:
        grouped.setdefault(record.label, []).append(record)
    for label, samples in sorted(grouped.items()):
        if len(samples) < folds:
            raise SystemExit(f"Stratum {label} has only {len(samples)} samples; need at least {folds}.")

    rng = np.random.default_rng(seed)
    fold_paths: list[list[Path]] = [[] for _ in range(folds)]
    for label, samples in sorted(grouped.items()):
        shuffled = list(samples)
        rng.shuffle(shuffled)
        chunks = np.array_split(np.asarray(shuffled, dtype=object), folds)
        sizes = [len(chunk) for chunk in chunks]
        if max(sizes) - min(sizes) > 1:
            raise SystemExit(f"Unexpected imbalanced fold sizes for {label}: {sizes}")
        for fold_index, chunk in enumerate(chunks):
            fold_paths[fold_index].extend(sample.path for sample in chunk.tolist())
    return fold_paths


def write_cv_splits(split_dir: Path, dataset_root: Path, fold_paths: list[list[Path]]) -> None:
    all_paths = [path for fold in fold_paths for path in fold]
    for fold_index, test in enumerate(fold_paths):
        test_set = set(test)
        train = [path for path in all_paths if path not in test_set]
        fold_dir = split_dir / f"cv{len(fold_paths)}" / f"fold_{fold_index}"
        write_split(fold_dir / "train.csv", dataset_root, train)
        write_split(fold_dir / "val.csv", dataset_root, test)
        write_split(fold_dir / "test.csv", dataset_root, test)
        print(f"cv{len(fold_paths)} fold={fold_index} train={len(train)} val={len(test)} test={len(test)}")


def main() -> int:
    project_root = Path(__file__).resolve().parents[2]
    repo_root = project_root.parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=repo_root / "Workspace" / "C_model_training" / "tactile",
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=project_root / "splits" / "tactile",
        help="Output directory for train/val/test split CSV files.",
    )
    parser.add_argument("--variant", choices=["base", "mean", "idw4"], default="idw4")
    parser.add_argument("--mode", choices=["random", "constrained"], default="constrained")
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "tactile_splits_config.example.json",)
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Retained for old commands; val.csv is now copied from test.csv.",
    )
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cv-folds", type=int, default=0, help="Also write stratified k-fold splits under <split-dir>/cvK.")
    args = parser.parse_args()

    dataset_root = args.base_dir.resolve()
    rng = np.random.default_rng(args.seed)
    if args.mode == "random":
        split_samples = split_random(
            collect_samples(dataset_root, args.variant),
            rng,
            args.test_ratio,
        )
    else:
        if args.config is None:
            raise SystemExit("--config is required when --mode constrained")
        config = load_config(args.config)
        split_samples = split_constrained(
            collect_sample_records(dataset_root, args.variant),
            config,
            rng,
            args.test_ratio,
        )

    split_dir = args.split_dir.resolve()
    write_split(split_dir / "train.csv", dataset_root, split_samples["train"])
    write_split(split_dir / "val.csv", dataset_root, split_samples["val"])
    write_split(split_dir / "test.csv", dataset_root, split_samples["test"])
    print(
        f"mode={args.mode} "
        f"train={len(split_samples['train'])} "
        f"val={len(split_samples['val'])} "
        f"test={len(split_samples['test'])}"
    )
    if args.cv_folds:
        if args.cv_folds < 2:
            raise SystemExit("--cv-folds must be at least 2")
        write_cv_splits(
            split_dir,
            dataset_root,
            stratified_folds(collect_sample_records(dataset_root, args.variant), args.cv_folds, args.seed),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
