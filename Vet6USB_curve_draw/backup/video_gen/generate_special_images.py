"""
Generate only the requested special still images for Parts B experiments.

This script reuses the data-processing functions from the existing experiment
scripts, but it does not call their main() functions and does not generate
videos.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PLOT_ROOT = SCRIPT_DIR.parent
REPO_ROOT = PLOT_ROOT.parent
PARTS_B_DIR = PLOT_ROOT / "07_PartsB_test"
OUTPUT_ROOT = SCRIPT_DIR / "special_image_output"

if str(PLOT_ROOT) not in sys.path:
    sys.path.insert(0, str(PLOT_ROOT))

from plot_style import COLORS, apply_measurement_style  # noqa: E402


SCRIPT_PATHS = {
    "test8": PARTS_B_DIR / "08_tactile_slip_pipeline.py",
    "test9a": PARTS_B_DIR / "09a_Tactile_force_broken_curve.py",
    "test9b": PARTS_B_DIR / "09b_Tactile_Contact_mask_extraction.py",
    "test10": PARTS_B_DIR / "10_Temp_difftemp.py",
}


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_png_pdf(fig: plt.Figure, out_dir: Path, stem: str) -> Path:
    png_dir = ensure_dir(out_dir / "png")
    pdf_dir = ensure_dir(out_dir / "pdf")
    png_path = png_dir / f"{stem}.png"
    pdf_path = pdf_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path


def safe_stem(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in text)


def six_indices(length: int) -> np.ndarray:
    if length <= 0:
        return np.zeros(6, dtype=int)
    return np.linspace(0, length - 1, 6).round().astype(int)


def set_taxel_ticks(ax: plt.Axes) -> None:
    ax.set_xticks(range(6))
    ax.set_yticks(range(6))
    ax.tick_params(length=0, labelsize=6)


def active_row_slice(frames: np.ndarray, keep_rows: int = 4) -> tuple[slice, np.ndarray]:
    rows = int(frames.shape[1])
    keep = max(1, min(int(keep_rows), rows))
    if keep >= rows:
        row_numbers = np.arange(rows)
        return slice(0, rows), row_numbers
    row_energy = np.nansum(np.abs(frames), axis=(0, 2))
    window_energy = np.convolve(row_energy, np.ones(keep, dtype=float), mode="valid")
    start = int(np.nanargmax(window_energy))
    row_numbers = np.arange(start, start + keep)
    return slice(start, start + keep), row_numbers


def negative_lsb_intensity(frames: np.ndarray) -> np.ndarray:
    """Convert negative tactile LSB response to positive intensity."""
    return np.maximum(-np.asarray(frames, dtype=float), 0.0)


def apply_cropped_heatmap_ticks(ax: plt.Axes, row_numbers: np.ndarray) -> None:
    ax.set_xticks(range(6))
    ax.set_yticks(range(len(row_numbers)))
    ax.set_yticklabels([str(int(row)) for row in row_numbers])
    ax.tick_params(length=0, labelsize=6)


def representative_force_trial(trials: list[object]) -> object | None:
    valid = [trial for trial in trials if getattr(trial, "rise_y", np.asarray([])).size >= 2]
    if not valid:
        return None
    scores = np.asarray([getattr(trial, "quality_score", np.nan) for trial in valid], dtype=float)
    if np.any(np.isfinite(scores)):
        return valid[int(np.nanargmin(scores))]
    lengths = np.asarray([trial.rise_y.size for trial in valid], dtype=int)
    return valid[int(np.argmax(lengths))]


def choose_contact_trial(trials: list[object]) -> object | None:
    valid = [trial for trial in trials if np.isfinite(getattr(trial, "coverage_pct", np.nan))]
    if not valid:
        return trials[0] if trials else None
    coverage = np.asarray([trial.coverage_pct for trial in valid], dtype=float)
    center = float(np.nanmedian(coverage))
    return valid[int(np.nanargmin(np.abs(coverage - center)))]


def plot_test8_slip_time_grid(args: argparse.Namespace) -> Path | None:
    mod = load_module("special_test8_slip", SCRIPT_PATHS["test8"])
    trial_paths = mod.discover_trials(args.test8_data_root)
    if not trial_paths:
        print(f"test8: no tactile slip CSV files under {args.test8_data_root}")
        return None

    results, failures = mod.batch_process(
        trial_paths_dict=trial_paths,
        apply_blur=args.gaussian_blur,
        gaussian_sigma=args.gaussian_sigma,
    )
    selected = mod.select_low_variance_trials(
        results=results,
        select_n=args.test8_select_n,
        feature=args.test8_selection_feature,
    )
    speeds = [speed for speed in mod.supplemental_display_speeds(selected) if selected.get(speed)]
    speeds = speeds[:6]
    if not speeds:
        print("test8: no selected slip trials")
        return None

    row_trials = []
    for speed in speeds:
        try:
            trial = mod.representative_forward_cop_trial(selected[speed])
        except Exception:
            trial = mod.representative_trial(selected[speed])
        row_trials.append((speed, trial))

    cropped_rows: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    intensity_samples = []
    for row_idx, (_speed, trial) in enumerate(row_trials):
        row_slice, row_numbers = active_row_slice(trial.X_slip, keep_rows=4)
        cropped_rows[row_idx] = (row_slice, row_numbers)
        intensity_samples.append(negative_lsb_intensity(trial.X_slip[:, row_slice, :]).ravel())

    finite = np.concatenate(
        [
            values[np.isfinite(values)]
            for values in intensity_samples
            if np.any(np.isfinite(values))
        ]
    )
    vmax = max(float(np.percentile(finite, 98)), float(mod.FIXED_ACTIVITY_THRESHOLD_LSB), 1.0)

    fig, axes = plt.subplots(len(row_trials), 6, figsize=(16.4, 1.62 * len(row_trials)), squeeze=False)
    last_im = None
    for row, (speed, trial) in enumerate(row_trials):
        row_slice, row_numbers = cropped_rows[row]
        frames = negative_lsb_intensity(trial.X_slip[:, row_slice, :])
        indices = six_indices(frames.shape[0])
        for col, frame_idx in enumerate(indices):
            ax = axes[row, col]
            last_im = ax.imshow(
                frames[int(frame_idx)],
                cmap="inferno",
                vmin=0.0,
                vmax=vmax,
                origin="upper",
                interpolation="nearest",
                aspect="auto",
            )
            apply_cropped_heatmap_ticks(ax, row_numbers)
            if row == 0:
                ax.set_title(f"t = {frame_idx / mod.FS:.2f} s", fontsize=10, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"{speed * 100:g} cm/s\nrow", fontsize=9)
            else:
                ax.set_yticklabels([])
            if row != len(row_trials) - 1:
                ax.set_xticklabels([])

    fig.suptitle("08 tactile slip evolution: negative LSB intensity, six time samples per speed", y=0.995)
    fig.subplots_adjust(left=0.055, right=0.935, bottom=0.06, top=0.91, wspace=0.10, hspace=0.08)
    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes, fraction=0.018, pad=0.012)
        cbar.set_label("-LSB intensity (larger = stronger force)")

    out = save_png_pdf(fig, args.output_root / "test8_tactile_slip", "08_slip_1_to_6_time_grid_6x6")
    print(f"test8: saved {out}")
    if failures:
        print(f"test8: skipped {len(failures)} failed trials")
    return out


def plot_test9a_force_rise_only(args: argparse.Namespace) -> Path | None:
    mod = load_module("special_test9a_force", SCRIPT_PATHS["test9a"])
    mod.FORCE_DISTANCE_GROUPS = tuple(args.force_groups)
    trial_paths = mod.discover_force_rise_trials(args.force_data_root, tuple(args.force_groups))
    if not any(paths for paths in trial_paths.values()):
        print(f"test9a: no force CSV files under {args.force_data_root}")
        return None

    results, failures = mod.batch_process_force_rise(
        trial_paths_dict=trial_paths,
        apply_blur=args.gaussian_blur,
        gaussian_sigma=args.gaussian_sigma,
    )
    selected = mod.select_low_variance_force_trials(
        results=results,
        select_n=args.force_select_n,
        n_grid_points=mod.FORCE_SELECTION_GRID_POINTS,
    )
    force_levels = mod.load_force_levels(args.force_data_root / "force_sensor.txt")

    fig, ax = plt.subplots(figsize=(9.8, 5.2))
    annotation_levels = np.linspace(0.20, 0.82, len(args.force_groups))
    for distance_label in args.force_groups:
        trials = selected.get(distance_label, [])
        if not trials:
            continue
        grid, curves = mod.align_force_rise_curves_by_time(trials)
        if curves.size == 0:
            continue
        mean_y = np.mean(curves, axis=0)
        std_y = np.std(curves, axis=0, ddof=0)
        color = mod.FORCE_GROUP_COLORS.get(distance_label, COLORS["blue"])
        label = mod.force_legend_label(distance_label, force_levels)
        ax.plot(grid, mean_y, color=color, linewidth=1.9, label=label)
        ax.fill_between(grid, mean_y - std_y, mean_y + std_y, color=color, alpha=0.14, linewidth=0)

        rise_peak_times = np.asarray(
            [(trial.peak_idx - trial.t_start) / mod.FS for trial in trials if trial.peak_idx > trial.t_start],
            dtype=float,
        )
        if np.any(np.isfinite(rise_peak_times)):
            rise_peak_x = float(np.nanmedian(rise_peak_times))
            peak_idx = int(np.argmin(np.abs(grid - rise_peak_x)))
        else:
            peak_idx = int(np.nanargmax(mean_y))
        peak_x = float(grid[peak_idx])
        peak_y = float(mean_y[peak_idx])

        base_y = float(np.nanmin(mean_y[: max(2, min(8, mean_y.size))]))
        amp = max(peak_y - base_y, 1.0)
        y10 = base_y + 0.10 * amp
        y90 = base_y + 0.90 * amp
        before_peak = mean_y[: peak_idx + 1]
        x10_candidates = np.flatnonzero(before_peak >= y10)
        x90_candidates = np.flatnonzero(before_peak >= y90)
        x10 = float(grid[int(x10_candidates[0])]) if x10_candidates.size else float(grid[0])
        x90 = float(grid[int(x90_candidates[0])]) if x90_candidates.size else peak_x
        rise_ms_values = np.asarray([trial.rise_10_90_ms for trial in trials], dtype=float)
        rise_ms = float((x90 - x10) * 1000.0)
        if np.any(np.isfinite(rise_ms_values)):
            rise_ms = float(np.nanmean(rise_ms_values))

        ax.axvline(x10, color=color, linestyle=":", linewidth=1.05, alpha=0.75)
        ax.axvline(x90, color=color, linestyle=":", linewidth=1.05, alpha=0.75)
        y_arrow = base_y + annotation_levels[args.force_groups.index(distance_label)] * amp
        ax.annotate(
            "",
            xy=(x90, y_arrow),
            xytext=(x10, y_arrow),
            arrowprops={"arrowstyle": "<->", "color": color, "linewidth": 1.45, "alpha": 0.92},
        )
        ax.text(
            (x10 + x90) / 2.0,
            y_arrow + 0.025 * amp,
            f"{rise_ms:.0f} ms",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
            color=color,
        )

        ax.scatter([peak_x], [peak_y], s=54, color=color, edgecolors="white", linewidths=1.0, zorder=4)
        annotation = f"rise peak\n{peak_y:.0f} LSB"
        ax.annotate(
            annotation,
            xy=(peak_x, peak_y),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=10.5,
            fontweight="bold",
            color=color,
            arrowprops={"arrowstyle": "-", "color": color, "linewidth": 1.2, "alpha": 0.9},
        )

    ax.set_title("09a tactile force rising curves with rise time and peak point")
    ax.set_xlabel("Time from rise onset [s]")
    ax.set_ylabel("Array envelope [LSB]")
    ax.set_xlim(0.0, mod.FORCE_RISE_PLOT_DURATION_S)
    ax.margins(x=0.02, y=0.10)
    ax.legend(frameon=False, ncol=2, fontsize=8)
    fig.tight_layout()

    out = save_png_pdf(fig, args.output_root / "test9a_tactile_force", "09a_force_rise_only_annotated")
    print(f"test9a: saved {out}")
    if failures:
        print(f"test9a: skipped {len(failures)} failed trials")
    return out


def contact_abs_frames(mod: ModuleType, trial: object, apply_blur: bool, gaussian_sigma: float) -> np.ndarray:
    x = mod.load_trial(trial.path)
    x_bc, _baseline_std = mod.baseline_correct(x)
    x_filtered = mod.maybe_blur_frames(x_bc, apply_blur=apply_blur, sigma_cells=gaussian_sigma)
    stop = max(int(trial.t_start) + 1, int(trial.peak_idx) + 1, int(trial.t_stable_idx) + 1)
    stop = min(stop, x_filtered.shape[0])
    return np.abs(x_filtered[int(trial.t_start) : stop])


def plot_test9b_contact_mask_time_grid(args: argparse.Namespace) -> Path | None:
    mod = load_module("special_test9b_mask", SCRIPT_PATHS["test9b"])
    mod.FORCE_DISTANCE_GROUPS = tuple(args.force_groups)
    trial_paths = mod.discover_force_rise_trials(args.force_data_root, tuple(args.force_groups))
    if not any(paths for paths in trial_paths.values()):
        print(f"test9b: no force CSV files under {args.force_data_root}")
        return None

    results, failures = mod.batch_process_contact_masks(
        trial_paths_dict=trial_paths,
        apply_blur=args.gaussian_blur,
        gaussian_sigma=args.gaussian_sigma,
        threshold_lsb=args.activity_threshold,
    )
    row_trials = []
    for group in args.force_groups:
        trial = choose_contact_trial(results.get(group, []))
        if trial is not None:
            row_trials.append((group, trial))
    if not row_trials:
        print("test9b: no usable contact-mask trials")
        return None

    cropped_rows: dict[int, tuple[slice, np.ndarray, np.ndarray]] = {}
    intensity_samples = []
    for row_idx, (_group, trial) in enumerate(row_trials):
        frames = contact_abs_frames(mod, trial, args.gaussian_blur, args.gaussian_sigma)
        row_slice, row_numbers = active_row_slice(frames, keep_rows=4)
        cropped = frames[:, row_slice, :]
        cropped_rows[row_idx] = (row_slice, row_numbers, cropped)
        intensity_samples.append(cropped.ravel())

    finite = np.concatenate(
        [
            values[np.isfinite(values)]
            for values in intensity_samples
            if np.any(np.isfinite(values))
        ]
    )
    vmax = max(float(np.percentile(finite, 98)), float(args.activity_threshold), 1.0)

    fig, axes = plt.subplots(len(row_trials), 6, figsize=(16.4, 1.62 * len(row_trials)), squeeze=False)
    last_im = None
    for row, (group, trial) in enumerate(row_trials):
        _row_slice, row_numbers, frames = cropped_rows[row]
        indices = six_indices(frames.shape[0])
        cumulative_masks = np.maximum.accumulate(frames, axis=0) >= float(trial.threshold_lsb)
        for col, frame_idx in enumerate(indices):
            ax = axes[row, col]
            last_im = ax.imshow(
                frames[int(frame_idx)],
                cmap="magma",
                vmin=0.0,
                vmax=vmax,
                origin="upper",
                interpolation="nearest",
                aspect="auto",
            )
            if np.any(cumulative_masks[int(frame_idx)]):
                ax.contour(
                    cumulative_masks[int(frame_idx)].astype(float),
                    levels=[0.5],
                    colors="white",
                    linewidths=0.85,
                )
            apply_cropped_heatmap_ticks(ax, row_numbers)
            if row == 0:
                ax.set_title(f"t = {frame_idx / mod.FS * 1000.0:.0f} ms", fontsize=10, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"{group.replace('cm', ' cm')}\nrow", fontsize=9)
            else:
                ax.set_yticklabels([])
            if row != len(row_trials) - 1:
                ax.set_xticklabels([])

    fig.suptitle("09b contact-mask evolution: response intensity with mask contour", y=0.995)
    fig.subplots_adjust(left=0.055, right=0.935, bottom=0.06, top=0.91, wspace=0.10, hspace=0.08)
    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes, fraction=0.018, pad=0.012)
        cbar.set_label("|LSB| intensity (larger = stronger contact)")

    out = save_png_pdf(fig, args.output_root / "test9b_tactile_contact_mask", "09b_contact_mask_1_to_6_time_grid_6x6")
    print(f"test9b: saved {out}")
    if failures:
        print(f"test9b: skipped {len(failures)} failed trials")
    return out


def plot_test10_temperature_combined(args: argparse.Namespace) -> list[Path]:
    overlay_path = args.temp_existing_png_root / "10_P0_P1_P2_signed_normalized_overlay.png"
    tau_path = args.temp_existing_png_root / "10_tau_fast_tau_slow_vs_k.png"
    missing = [path for path in (overlay_path, tau_path) if not path.exists()]
    if missing:
        for path in missing:
            print(f"test10: missing source figure {path}")
        return []

    overlay_img = mpimg.imread(overlay_path)
    tau_img = mpimg.imread(tau_path)

    overlay_h, overlay_w = overlay_img.shape[:2]
    top_blank = overlay_h * 0.30
    fig_width = 10.2
    fig_height = fig_width * (overlay_h + top_blank) / overlay_w
    fig, ax_main = plt.subplots(figsize=(fig_width, fig_height))

    ax_main.imshow(overlay_img, extent=(0, overlay_w, overlay_h, 0), aspect="equal")
    ax_main.set_xlim(0, overlay_w)
    ax_main.set_ylim(overlay_h, -top_blank)
    ax_main.set_axis_off()
    ax_main.set_title("Signed normalized response overlay", fontsize=12, fontweight="bold", pad=8)

    ax_sub = ax_main.inset_axes([0.61, 0.79, 0.34, 0.18])
    ax_sub.imshow(tau_img)
    ax_sub.set_axis_off()
    for spine in ax_sub.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_edgecolor("#4a4a4a")

    fig.suptitle("10 temperature response summary", y=0.995, fontsize=13, fontweight="bold")
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.035, top=0.93)
    out = save_png_pdf(fig, args.output_root / "test10_temperature_curves", "10_signed_overlay_with_tau_subfigure")
    print(f"test10: saved {out}")
    return [out]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate requested VET6USB special still images only.")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--tests", nargs="+", default=["test8", "test9a", "test9b", "test10"])
    parser.add_argument("--gaussian-blur", action="store_true", default=False)
    parser.add_argument("--gaussian-sigma", type=float, default=0.65)

    parser.add_argument("--test8-data-root", type=Path, default=REPO_ROOT / "workspace" / "tact_sliptest")
    parser.add_argument("--test8-select-n", type=int, default=5)
    parser.add_argument("--test8-selection-feature", default="f_centroid")

    parser.add_argument("--force-data-root", type=Path, default=REPO_ROOT / "workspace" / "tact_forcetest")
    parser.add_argument("--force-groups", nargs="+", default=["1cm", "2cm", "3cm", "4cm", "5cm", "6cm"])
    parser.add_argument("--force-select-n", type=int, default=10)
    parser.add_argument("--activity-threshold", type=float, default=35.0)

    parser.add_argument(
        "--temp-existing-png-root",
        type=Path,
        default=PLOT_ROOT / "result_display" / "test10_temperature_curves" / "difftemp" / "png",
    )
    return parser.parse_args()


def main() -> None:
    apply_measurement_style()
    args = parse_args()
    args.output_root = ensure_dir(args.output_root)
    requested = set(args.tests)

    saved: list[Path] = []
    if "test8" in requested:
        path = plot_test8_slip_time_grid(args)
        if path is not None:
            saved.append(path)
    if "test9a" in requested:
        path = plot_test9a_force_rise_only(args)
        if path is not None:
            saved.append(path)
    if "test9b" in requested:
        path = plot_test9b_contact_mask_time_grid(args)
        if path is not None:
            saved.append(path)
    if "test10" in requested:
        saved.extend(plot_test10_temperature_combined(args))

    print(f"Saved special images: {len(saved)}")
    for path in saved:
        print(path)


if __name__ == "__main__":
    main()
