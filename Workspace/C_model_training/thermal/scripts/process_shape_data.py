from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any


def find_repo_root(start: Path) -> Path:
    for path in (start, *start.parents):
        if (path / "Vet6USB_pyqt").exists():
            return path
    raise RuntimeError("Could not find repository root containing Vet6USB_pyqt")


ROOT = find_repo_root(Path(__file__).resolve())
VET6_ROOT = ROOT / "Vet6USB_pyqt"
if str(VET6_ROOT) not in sys.path:
    sys.path.insert(0, str(VET6_ROOT))

from app.services.protocol import (  # noqa: E402
    AD_COUNT,
    HEATMAP_ROW_COUNT,
    TACTILE_COL_COUNT,
    HeatmapBatch,
    HeatmapBatchAssembler,
    VET6_8ChipsParser,
    raw_to_mv,
)
from app.services.tactile_interpolation import interpolate_gripper_points  # noqa: E402


VOLUME_DIRS = ("15ml", "50ml", "100ml", "200ml")
GRIPPER_TEMP_K = 0.013  # degC / raw

GRIPPER_ARRAY_LAYOUTS = {
    "left": (
        ("AD7", 0, "P1"),
        ("AD3", 1, "P2"),
        ("AD8", 2, "P3a"),
        ("AD4", 3, "P3b"),
    ),
    "right": (
        ("AD1", 0, "P1"),
        ("AD5", 1, "P2"),
        ("AD6", 2, "P3a"),
        ("AD2", 3, "P3b"),
    ),
}
GRIPPER_GLOBAL_AD_INDEX = {
    "AD1": 0,
    "AD2": 1,
    "AD3": 2,
    "AD4": 3,
    "AD5": 4,
    "AD6": 5,
    "AD7": 6,
    "AD8": 7,
}
GRIPPER_TACTILE_VERTICAL_FLIP_LABELS = {"AD3", "AD4", "AD5", "AD6"}
GRIPPER_TACTILE_HORIZONTAL_FLIP_LABELS = {"AD5", "AD6", "AD7", "AD8"}


Baseline = list[list[list[float]]]
TemperatureBaseline = list[float]


def iter_batches(path: Path) -> tuple[list[HeatmapBatch], int, int]:
    parser = VET6_8ChipsParser()
    assembler = HeatmapBatchAssembler()
    batches: list[HeatmapBatch] = []
    frame_count = 0

    with path.open("rb") as file:
        while True:
            chunk = file.read(1024 * 1024)
            if not chunk:
                break
            for row_frame in parser.feed(chunk):
                frame_count += 1
                batch = assembler.ingest(row_frame)
                if batch is not None:
                    batches.append(batch)

    return batches, frame_count, parser.error_count


def compute_zero_baseline(zero_file: Path) -> tuple[Baseline, int]:
    batches, _, _ = iter_batches(zero_file)
    baseline = [
        [[0.0] * TACTILE_COL_COUNT for _ in range(HEATMAP_ROW_COUNT)]
        for _ in range(AD_COUNT)
    ]
    if not batches:
        return baseline, 0

    for batch in batches:
        for ad in range(AD_COUNT):
            for row in range(HEATMAP_ROW_COUNT):
                for col in range(TACTILE_COL_COUNT):
                    baseline[ad][row][col] += batch.raw_by_ad_row_ch[ad][row][col]

    count = float(len(batches))
    for ad in range(AD_COUNT):
        for row in range(HEATMAP_ROW_COUNT):
            for col in range(TACTILE_COL_COUNT):
                baseline[ad][row][col] /= count
    return baseline, len(batches)


def compute_temperature_baseline(zero_file: Path) -> tuple[TemperatureBaseline, int]:
    batches, _, _ = iter_batches(zero_file)
    baseline = [0.0] * AD_COUNT
    if not batches:
        return baseline, 0

    for batch in batches:
        for ad in range(AD_COUNT):
            baseline[ad] += batch.raw_by_ad_row_ch[ad][0][7]

    count = float(len(batches))
    for ad in range(AD_COUNT):
        baseline[ad] /= count
    return baseline, len(batches)


def raw_to_temperature_c(raw: int, raw0: float, ambient_c: float = 25.0) -> float:
    return ambient_c + GRIPPER_TEMP_K * (raw - raw0)


def corrected_tactile_value(
    batch: HeatmapBatch,
    baseline: Baseline,
    ad: int,
    row: int,
    col: int,
) -> float:
    return batch.raw_by_ad_row_ch[ad][row][col] - baseline[ad][row][col]


def display_cell_value(
    batch: HeatmapBatch,
    baseline: Baseline,
    label: str,
    display_row_from_bottom: int,
    display_col_from_left: int,
) -> float:
    source_row = display_col_from_left
    if label in GRIPPER_TACTILE_HORIZONTAL_FLIP_LABELS:
        source_row = TACTILE_COL_COUNT - 1 - source_row

    source_col = display_row_from_bottom
    if label in GRIPPER_TACTILE_VERTICAL_FLIP_LABELS:
        source_col = TACTILE_COL_COUNT - 1 - source_col

    ad = GRIPPER_GLOBAL_AD_INDEX[label]
    return corrected_tactile_value(batch, baseline, ad, source_row, source_col)


def display_values_by_role(
    batch: HeatmapBatch,
    baseline: Baseline,
    side: str,
) -> dict[str, list[list[float]]]:
    values: dict[str, list[list[float]]] = {}
    for label, _, role in GRIPPER_ARRAY_LAYOUTS[side]:
        values[role] = [
            [
                display_cell_value(
                    batch,
                    baseline,
                    label,
                    TACTILE_COL_COUNT - 1 - row,
                    col,
                )
                for col in range(TACTILE_COL_COUNT)
            ]
            for row in range(HEATMAP_ROW_COUNT)
        ]
    return values


def tactile_columns() -> list[str]:
    return [
        f"R{row + 1}C{col + 1}"
        for row in range(HEATMAP_ROW_COUNT)
        for col in range(TACTILE_COL_COUNT)
    ]


def write_zeroed_csv(
    path: Path,
    source_dir: Path,
    baseline: Baseline,
) -> tuple[int, int, int]:
    rows_written = 0
    batches_written = 0
    parser_errors = 0
    header = [
        "SourceFile",
        "Index",
        "AD",
        *tactile_columns(),
        "TemperatureRaw16",
        "Temperature_mV",
    ]

    with path.open("w", newline="", encoding="utf-8-sig") as output:
        writer = csv.writer(output)
        writer.writerow(header)

        for raw_file in sorted(source_dir.glob("*.bin")):
            batches, _, errors = iter_batches(raw_file)
            parser_errors += errors
            for batch in batches:
                batches_written += 1
                for ad in range(AD_COUNT):
                    row_values: list[object] = [
                        raw_file.name,
                        batches_written,
                        f"AD{ad + 1}",
                    ]
                    for row in range(HEATMAP_ROW_COUNT):
                        for col in range(TACTILE_COL_COUNT):
                            row_values.append(
                                f"{corrected_tactile_value(batch, baseline, ad, row, col):.6f}"
                            )
                    temp_raw = batch.raw_by_ad_row_ch[ad][0][7]
                    row_values.extend([temp_raw, f"{raw_to_mv(temp_raw):.6f}"])
                    writer.writerow(row_values)
                    rows_written += 1

    return rows_written, batches_written, parser_errors


def interpolation_columns() -> list[str]:
    empty_values = {role: [[0.0] * TACTILE_COL_COUNT for _ in range(HEATMAP_ROW_COUNT)]
                    for role in ("P1", "P2", "P3a", "P3b")}
    return [point.name for point in interpolate_gripper_points(empty_values, "mean")]


def side_prefix(side: str) -> str:
    return "L" if side == "left" else "R"


def semantic_base_columns() -> list[str]:
    columns: list[str] = []
    for side in ("left", "right"):
        prefix = side_prefix(side)
        for _, _, role in GRIPPER_ARRAY_LAYOUTS[side]:
            for row in range(HEATMAP_ROW_COUNT):
                for col in range(TACTILE_COL_COUNT):
                    columns.append(f"{prefix}_{role}_R{row + 1}C{col + 1}")
    return columns


def semantic_temperature_columns() -> list[str]:
    return [f"AD{ad + 1}_Temperature_C" for ad in range(AD_COUNT)]


def semantic_interpolation_columns(method: str) -> list[str]:
    columns: list[str] = []
    for side in ("left", "right"):
        prefix = side_prefix(side)
        empty_values = {
            role: [[0.0] * TACTILE_COL_COUNT for _ in range(HEATMAP_ROW_COUNT)]
            for _, _, role in GRIPPER_ARRAY_LAYOUTS[side]
        }
        for point in interpolate_gripper_points(empty_values, method):
            safe_name = (
                point.name
                .replace(" ", "_")
                .replace("-", "_")
            )
            columns.append(f"{prefix}_{safe_name}")
    return columns


def semantic_display_values(
    batch: HeatmapBatch,
    baseline: Baseline,
    side: str,
) -> list[str]:
    values: list[str] = []
    display_values = display_values_by_role(batch, baseline, side)
    for _, _, role in GRIPPER_ARRAY_LAYOUTS[side]:
        for row in range(HEATMAP_ROW_COUNT):
            for col in range(TACTILE_COL_COUNT):
                values.append(f"{display_values[role][row][col]:.6f}")
    return values


def semantic_temperature_values(
    batch: HeatmapBatch,
    temperature_baseline: TemperatureBaseline,
) -> list[str]:
    values: list[str] = []
    for ad in range(AD_COUNT):
        temp_raw = batch.raw_by_ad_row_ch[ad][0][7]
        values.append(
            f"{raw_to_temperature_c(temp_raw, temperature_baseline[ad]):.6f}"
        )
    return values


def semantic_interpolated_values(
    batch: HeatmapBatch,
    baseline: Baseline,
    side: str,
    method: str,
) -> list[str]:
    display_values = display_values_by_role(batch, baseline, side)
    points = interpolate_gripper_points(display_values, method)
    return [f"{point.value:.6f}" for point in points]


def variant_feature_columns(variant: str) -> list[str]:
    if variant == "base":
        return semantic_base_columns()
    if variant == "mean":
        return semantic_interpolation_columns("mean")
    if variant == "idw4":
        return semantic_interpolation_columns("idw4")
    raise ValueError(f"Unknown tactile variant: {variant}")


def variant_feature_values(
    batch: HeatmapBatch,
    baseline: Baseline,
    variant: str,
) -> list[str]:
    values: list[str] = []
    if variant == "base":
        for side in ("left", "right"):
            values.extend(semantic_display_values(batch, baseline, side))
        return values

    method = "mean" if variant == "mean" else "idw4"
    for side in ("left", "right"):
        values.extend(semantic_interpolated_values(batch, baseline, side, method))
    return values


def parse_decimal_token(token: str) -> str:
    return token.replace("p", ".")


def parse_raw_file_metadata(raw_file: Path, fallback_volume: str) -> dict[str, str]:
    stem = raw_file.stem
    metadata = {
        "ContainerModel": fallback_volume,
        "Ambient_C": "",
        "Liquid_C": "",
        "LevelClass": "",
    }

    match = re.search(
        r"ContainerModel(?P<container>.+?)_"
        r"Tamb(?P<ambient>[-\dp]+)C_"
        r"Tliquid(?P<liquid>[-\dp]+)C_"
        r"Levelclass(?P<level>[^_]+)",
        stem,
        flags=re.IGNORECASE,
    )
    if match:
        metadata["ContainerModel"] = match.group("container")
        metadata["Ambient_C"] = parse_decimal_token(match.group("ambient"))
        metadata["Liquid_C"] = parse_decimal_token(match.group("liquid"))
        metadata["LevelClass"] = match.group("level")

    return metadata


def merged_feature_columns() -> list[str]:
    return [
        "ContainerModel",
        "Ambient_C",
        "Liquid_C",
        "LevelClass",
        "Volume",
        "SourceFile",
        "Index",
        *semantic_base_columns(),
        *semantic_interpolation_columns("idw4"),
        *semantic_temperature_columns(),
    ]


def merged_feature_values(
    batch: HeatmapBatch,
    baseline: Baseline,
    temperature_baseline: TemperatureBaseline,
    metadata: dict[str, str],
    volume: str,
    raw_file: Path,
    index: int,
) -> list[object]:
    row_values: list[object] = [
        metadata["ContainerModel"],
        metadata["Ambient_C"],
        metadata["Liquid_C"],
        metadata["LevelClass"],
        volume,
        raw_file.name,
        index,
    ]
    row_values.extend(variant_feature_values(batch, baseline, "base"))
    row_values.extend(variant_feature_values(batch, baseline, "idw4"))
    row_values.extend(semantic_temperature_values(batch, temperature_baseline))
    return row_values


def write_merged_file_csv(
    raw_file: Path,
    output_file: Path,
    baseline: Baseline,
    temperature_baseline: TemperatureBaseline,
    volume: str,
) -> tuple[int, int, int]:
    rows_written = 0
    output_file.parent.mkdir(parents=True, exist_ok=True)
    batches, _, parser_errors = iter_batches(raw_file)
    metadata = parse_raw_file_metadata(raw_file, volume)

    with output_file.open("w", newline="", encoding="utf-8-sig") as output:
        writer = csv.writer(output)
        writer.writerow(merged_feature_columns())
        for index, batch in enumerate(batches, start=1):
            writer.writerow(
                merged_feature_values(
                    batch,
                    baseline,
                    temperature_baseline,
                    metadata,
                    volume,
                    raw_file,
                    index,
                )
            )
            rows_written += 1

    return rows_written, len(batches), parser_errors


def write_tactile_file_csv(
    raw_file: Path,
    output_file: Path,
    baseline: Baseline,
    temperature_baseline: TemperatureBaseline,
    volume: str,
    variant: str,
) -> tuple[int, int, int]:
    rows_written = 0
    output_file.parent.mkdir(parents=True, exist_ok=True)
    batches, _, parser_errors = iter_batches(raw_file)
    header = [
        "Volume",
        "SourceFile",
        "Index",
        *variant_feature_columns(variant),
        *semantic_temperature_columns(),
    ]

    with output_file.open("w", newline="", encoding="utf-8-sig") as output:
        writer = csv.writer(output)
        writer.writerow(header)
        for index, batch in enumerate(batches, start=1):
            writer.writerow([
                volume,
                raw_file.name,
                index,
                *variant_feature_values(batch, baseline, variant),
                *semantic_temperature_values(batch, temperature_baseline),
            ])
            rows_written += 1

    return rows_written, len(batches), parser_errors


def discover_tactile_raw_files(raw_bin_dir: Path) -> list[tuple[str, Path]]:
    if not raw_bin_dir.exists():
        return []

    discovered: list[tuple[str, Path]] = []
    for raw_file in sorted(raw_bin_dir.rglob("*.bin")):
        relative = raw_file.relative_to(raw_bin_dir)
        if len(relative.parts) == 1:
            if not re.match(r"batch_\d+_.+", raw_bin_dir.name, flags=re.IGNORECASE):
                continue
            volume = raw_bin_dir.name
        else:
            volume = relative.parts[0]
        if volume.lower() in {"raw_bin", "zero", "zeros", "baseline", "data", "data_csv"}:
            continue
        batch_match = re.match(r"batch_\d+_(?P<volume>.+)", volume, flags=re.IGNORECASE)
        if batch_match:
            volume = batch_match.group("volume")
        discovered.append((volume, raw_file))
    return discovered


def discover_legacy_raw_files(base_dir: Path) -> list[tuple[str, Path]]:
    discovered: list[tuple[str, Path]] = []
    for volume in VOLUME_DIRS:
        source_dir = base_dir / volume
        for raw_file in sorted(source_dir.glob("*.bin")):
            discovered.append((volume, raw_file))
    return discovered


def default_raw_bin_dir(base_dir: Path) -> Path:
    for structured in (
        base_dir / "raw_bin",
        base_dir / "tactile" / "raw_bin",
        base_dir / "data" / "tactile" / "raw_bin",
    ):
        if structured.exists():
            return structured
    return base_dir


def default_tactile_output_root(base_dir: Path) -> Path:
    return base_dir / "data" / "merged"


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


def write_model_csv(
    path: Path,
    source_dir: Path,
    baseline: Baseline,
    temperature_baseline: TemperatureBaseline,
    volume: str,
    interpolation_method: str | None,
) -> tuple[int, int, int]:
    rows_written = 0
    batches_written = 0
    parser_errors = 0
    header = [
        "Volume",
        "SourceFile",
        "Index",
        *semantic_base_columns(),
    ]
    if interpolation_method is not None:
        header.extend(semantic_interpolation_columns(interpolation_method))
    header.extend(semantic_temperature_columns())

    with path.open("w", newline="", encoding="utf-8-sig") as output:
        writer = csv.writer(output)
        writer.writerow(header)

        for raw_file in sorted(source_dir.glob("*.bin")):
            batches, _, errors = iter_batches(raw_file)
            parser_errors += errors
            for batch in batches:
                batches_written += 1
                row_values: list[object] = [volume, raw_file.name, batches_written]
                for side in ("left", "right"):
                    row_values.extend(semantic_display_values(batch, baseline, side))
                if interpolation_method is not None:
                    for side in ("left", "right"):
                        row_values.extend(
                            semantic_interpolated_values(
                                batch,
                                baseline,
                                side,
                                interpolation_method,
                            )
                        )
                row_values.extend(
                    semantic_temperature_values(batch, temperature_baseline)
                )
                writer.writerow(row_values)
                rows_written += 1

    return rows_written, batches_written, parser_errors


def write_interpolated_csv(
    path: Path,
    source_dir: Path,
    baseline: Baseline,
    method: str,
) -> tuple[int, int, int]:
    rows_written = 0
    batches_written = 0
    parser_errors = 0
    point_names = interpolation_columns()

    with path.open("w", newline="", encoding="utf-8-sig") as output:
        writer = csv.writer(output)
        writer.writerow(["SourceFile", "Index", "Side", *point_names])

        for raw_file in sorted(source_dir.glob("*.bin")):
            batches, _, errors = iter_batches(raw_file)
            parser_errors += errors
            for batch in batches:
                batches_written += 1
                for side in ("left", "right"):
                    display_values = display_values_by_role(batch, baseline, side)
                    points = interpolate_gripper_points(display_values, method)
                    writer.writerow([
                        raw_file.name,
                        batches_written,
                        side,
                        *[f"{point.value:.6f}" for point in points],
                    ])
                    rows_written += 1

    return rows_written, batches_written, parser_errors


def find_default_zero_file(base_dir: Path) -> Path:
    candidates = sorted(base_dir.glob("*.bin"))
    if not candidates:
        for candidate_dir in (
            base_dir / "raw_bin",
            base_dir / "tactile" / "raw_bin",
            base_dir / "data" / "tactile" / "raw_bin",
        ):
            if candidate_dir.exists():
                candidates = sorted(candidate_dir.glob("*.bin"))
                break
    if len(candidates) != 1:
        raise SystemExit(
            "Expected exactly one zero-point .bin in base-dir, "
            "raw_bin, tactile/raw_bin, or data/tactile/raw_bin. "
            "Pass --zero-file explicitly if needed."
        )
    return candidates[0]


def process_all(
    base_dir: Path,
    zero_file: Path,
    raw_bin_dir: Path,
    tactile_output_root: Path,
    append_summary: bool = False,
) -> None:
    tactile_output_root.mkdir(parents=True, exist_ok=True)
    baseline, zero_batches = compute_zero_baseline(zero_file)
    temperature_baseline, temperature_zero_batches = compute_temperature_baseline(
        zero_file
    )
    if zero_batches == 0 or temperature_zero_batches == 0:
        raise SystemExit(f"No complete zero-point batches parsed from {zero_file}")

    raw_files = discover_tactile_raw_files(raw_bin_dir)
    if not raw_files and raw_bin_dir == base_dir:
        raw_files = discover_legacy_raw_files(base_dir)
    raw_files = [
        (volume, raw_file)
        for volume, raw_file in raw_files
        if raw_file.resolve() != zero_file
    ]
    if not raw_files:
        raise SystemExit(f"No tactile .bin files found under {raw_bin_dir}")

    summary_path = tactile_output_root / "processing_summary.csv"
    write_header = not append_summary or not summary_path.exists()
    with summary_path.open("a" if append_summary else "w", newline="", encoding="utf-8-sig") as summary_file:
        summary = csv.writer(summary_file)
        if write_header:
            summary.writerow([
                "Volume",
                "SourceFile",
                "Output",
                "Rows",
                "Batches",
                "ParserErrors",
                "ZeroFile",
                "ZeroBatches",
            ])

        for volume, raw_file in raw_files:
            output_file = tactile_output_root / f"{raw_file.stem}.csv"
            rows, batches, errors = write_merged_file_csv(
                raw_file,
                output_file,
                baseline,
                temperature_baseline,
                volume,
            )
            summary.writerow([
                volume,
                raw_file.name,
                output_file.relative_to(tactile_output_root).as_posix(),
                rows,
                batches,
                errors,
                zero_file.name,
                zero_batches,
            ])


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Convert VET6 8chips raw shape data, subtract tactile zero offsets, "
            "and export mean/IDW-4 tactile interpolation CSVs."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "splits_config.example.json",
        help="Thermal dataset config JSON.",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=None,
        help="Thermal dataset root. Defaults to this script's parent directory.",
    )
    parser.add_argument(
        "--raw-bin-dir",
        type=Path,
        help=(
            "Tactile raw .bin directory. Defaults to "
            "<base-dir>/raw_bin, then <base-dir>/tactile/raw_bin, "
            "then <base-dir>/data/tactile/raw_bin, then <base-dir>."
        ),
    )
    parser.add_argument(
        "--zero-file",
        type=Path,
        help=(
            "Zero-point raw .bin. Defaults to the single .bin in base-dir, "
            "raw_bin, tactile/raw_bin, or data/tactile/raw_bin."
        ),
    )
    parser.add_argument(
        "--output-root",
        "--tactile-output-root",
        dest="output_root",
        type=Path,
        help=(
            "Thermal CSV output root. Defaults to config process.output_root "
            "or <base-dir>/data/merged."
        ),
    )
    args = parser.parse_args()

    default_base_dir = Path(__file__).resolve().parents[1]
    base_dir = (args.base_dir or default_base_dir).resolve()
    config = load_json_config(args.config)
    process_config = config.get("process", {})
    if not isinstance(process_config, dict):
        raise SystemExit("Config field process must be an object")

    tactile_output_root = (
        args.output_root
        or resolve_config_path(base_dir, process_config.get("output_root"))
        or default_tactile_output_root(base_dir)
    ).resolve()

    batch_configs = process_config.get("batches")
    if batch_configs is not None:
        if args.zero_file is not None or args.raw_bin_dir is not None:
            raise SystemExit("--zero-file/--raw-bin-dir cannot be combined with process.batches")
        if not isinstance(batch_configs, list):
            raise SystemExit("Config field process.batches must be a list")
        for index, batch_config in enumerate(batch_configs):
            if not isinstance(batch_config, dict):
                raise SystemExit("Each process.batches entry must be an object")
            zero_file = resolve_config_path(base_dir, batch_config.get("zero_file"))
            raw_bin_dir = resolve_config_path(base_dir, batch_config.get("raw_bin_dir"))
            if zero_file is None or raw_bin_dir is None:
                raise SystemExit("Each process.batches entry requires zero_file and raw_bin_dir")
            print(f"Batch: {batch_config.get('name', raw_bin_dir.name)}")
            process_all(
                base_dir,
                zero_file.resolve(),
                raw_bin_dir.resolve(),
                tactile_output_root,
                append_summary=index > 0,
            )
            print(f"Zero file:  {zero_file.resolve()}")
            print(f"Raw bin dir: {raw_bin_dir.resolve()}")
    else:
        zero_file = (
            args.zero_file
            or resolve_config_path(base_dir, process_config.get("zero_file"))
            or find_default_zero_file(base_dir)
        ).resolve()
        raw_bin_dir = (
            args.raw_bin_dir
            or resolve_config_path(base_dir, process_config.get("raw_bin_dir"))
            or default_raw_bin_dir(base_dir)
        ).resolve()
        process_all(base_dir, zero_file, raw_bin_dir, tactile_output_root)
        print(f"Zero file:  {zero_file}")
        print(f"Raw bin dir: {raw_bin_dir}")
    print(f"Thermal output root: {tactile_output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
