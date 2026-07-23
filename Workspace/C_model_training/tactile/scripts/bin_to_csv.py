from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


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
    columns: list[str] = []
    for ad in range(AD_COUNT):
        columns.extend([
            f"AD{ad + 1}_TemperatureRaw16",
            f"AD{ad + 1}_Temperature_mV",
        ])
    return columns


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


def semantic_temperature_values(batch: HeatmapBatch) -> list[object]:
    values: list[object] = []
    for ad in range(AD_COUNT):
        temp_raw = batch.raw_by_ad_row_ch[ad][0][7]
        values.extend([temp_raw, f"{raw_to_mv(temp_raw):.6f}"])
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


def write_tactile_file_csv(
    raw_file: Path,
    output_file: Path,
    baseline: Baseline,
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
                *semantic_temperature_values(batch),
            ])
            rows_written += 1

    return rows_written, len(batches), parser_errors


def discover_tactile_raw_files(raw_bin_dir: Path) -> list[tuple[str, Path]]:
    if not raw_bin_dir.exists():
        return []

    discovered: list[tuple[str, Path]] = []
    for raw_file in sorted(raw_bin_dir.rglob("*.bin")):
        relative = raw_file.relative_to(raw_bin_dir)
        if len(relative.parts) > 1:
            volume = relative.parts[0]
        else:
            volume = raw_file.parent.name
        if volume.lower() in {"raw_bin", "zero", "zeros", "baseline"}:
            volume = "unlabeled"
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
    if (base_dir / "raw_bin").exists():
        return base_dir
    structured = base_dir / "tactile"
    if structured.exists():
        return structured
    return base_dir / "data" / "tactile"


def write_model_csv(
    path: Path,
    source_dir: Path,
    baseline: Baseline,
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
                row_values.extend(semantic_temperature_values(batch))
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
) -> None:
    tactile_output_root.mkdir(parents=True, exist_ok=True)
    baseline, zero_batches = compute_zero_baseline(zero_file)
    if zero_batches == 0:
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

    for volume, raw_file in raw_files:
        for variant in ("base", "mean", "idw4"):
            output_file = (
                tactile_output_root
                / variant
                / volume
                / f"{raw_file.stem}.csv"
            )
            rows, batches, errors = write_tactile_file_csv(
                raw_file,
                output_file,
                baseline,
                volume,
                variant,
            )
            print(
                f"{variant}/{volume}/{output_file.name}: "
                f"rows={rows} batches={batches} parser_errors={errors}"
            )


def iter_batch_dirs(rawdata_dir: Path) -> list[Path]:
    if not rawdata_dir.exists():
        raise SystemExit(f"Missing rawdata directory: {rawdata_dir}")
    batch_dirs = [
        path for path in sorted(rawdata_dir.iterdir())
        if path.is_dir() and path.name.lower().startswith("batsh")
    ]
    if not batch_dirs:
        raise SystemExit(f"No batsh* directories found under {rawdata_dir}")
    return batch_dirs


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Convert VET6 8chips raw shape data, subtract tactile zero offsets, "
            "and export mean/IDW-4 tactile interpolation CSVs."
        )
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Tactile dataset root containing rawdata/, data/, splits/, and scripts/.",
    )
    args = parser.parse_args()

    base_dir = args.base_dir.resolve()
    output_root = base_dir / "data"
    for batch_dir in iter_batch_dirs(base_dir / "rawdata"):
        zero_file = find_default_zero_file(batch_dir).resolve()
        print(f"Batch: {batch_dir.name}")
        print(f"Zero file: {zero_file}")
        process_all(base_dir, zero_file, batch_dir.resolve(), output_root.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
