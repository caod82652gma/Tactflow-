"""
Generate time-varying videos for VET6USB experiment figures.

This script only produces visualization videos. It reuses the processing
functions from the existing experiment scripts, but it does not call their
main() functions and does not touch result_display outputs.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import shutil
import sys
from pathlib import Path
from types import ModuleType

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
import numpy as np

try:
    from scipy.ndimage import gaussian_filter, zoom
except ImportError:  # pragma: no cover - scipy is available in the experiment env
    gaussian_filter = None
    zoom = None


SCRIPT_DIR = Path(__file__).resolve().parent
PLOT_ROOT = SCRIPT_DIR.parent
REPO_ROOT = PLOT_ROOT.parent
PARTS_B_DIR = PLOT_ROOT / "07_PartsB_test"
OUTPUT_ROOT = SCRIPT_DIR / "video_output"

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


def safe_stem(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in text)


def frame_indices(length: int, max_frames: int | None) -> np.ndarray:
    if length <= 0:
        return np.asarray([], dtype=int)
    if max_frames is None or max_frames <= 0 or length <= max_frames:
        return np.arange(length, dtype=int)
    return np.unique(np.linspace(0, length - 1, max_frames).round().astype(int))


def animation_writer(path: Path, fps: int):
    if path.suffix.lower() == ".gif":
        return path, PillowWriter(fps=fps)
    if shutil.which("ffmpeg"):
        return path.with_suffix(".mp4"), FFMpegWriter(fps=fps)
    print("ffmpeg not found; falling back to GIF output.")
    return path.with_suffix(".gif"), PillowWriter(fps=fps)


def save_animation(anim: FuncAnimation, path: Path, fps: int, dpi: int) -> Path:
    ensure_dir(path.parent)
    out_path, writer = animation_writer(path, fps)
    anim.save(out_path, writer=writer, dpi=dpi)
    plt.close(anim._fig)
    return out_path


def set_taxel_axes(ax) -> None:
    ax.set_xticks(range(6))
    ax.set_yticks(range(6))
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")


def active_row_window(frames: np.ndarray, n_rows: int) -> tuple[int, int]:
    rows = int(frames.shape[1])
    keep = max(1, min(int(n_rows), rows))
    if keep >= rows:
        return 0, rows
    row_energy = np.nansum(np.abs(frames), axis=(0, 2))
    window_energy = np.convolve(row_energy, np.ones(keep, dtype=float), mode="valid")
    start = int(np.nanargmax(window_energy))
    return start, start + keep


def interpolate_frames_time(frames: np.ndarray, factor: int) -> np.ndarray:
    factor = max(1, int(factor))
    if factor <= 1 or frames.shape[0] < 2:
        return frames
    target_len = (frames.shape[0] - 1) * factor + 1
    target = np.linspace(0.0, frames.shape[0] - 1, target_len)
    left = np.floor(target).astype(int)
    right = np.clip(left + 1, 0, frames.shape[0] - 1)
    frac = (target - left).reshape(-1, 1, 1)
    return (1.0 - frac) * frames[left] + frac * frames[right]


def oversample_spatial_frames(
    frames: np.ndarray,
    row_factor: int,
    col_factor: int,
    smooth_sigma: float,
) -> np.ndarray:
    processed = np.asarray(frames, dtype=float)
    if smooth_sigma > 0 and gaussian_filter is not None:
        processed = gaussian_filter(processed, sigma=(0.0, smooth_sigma, smooth_sigma), mode="nearest")

    row_factor = max(1, int(row_factor))
    col_factor = max(1, int(col_factor))
    if row_factor == 1 and col_factor == 1:
        return processed
    if zoom is not None:
        return zoom(processed, zoom=(1, row_factor, col_factor), order=3, mode="nearest")
    return np.repeat(np.repeat(processed, row_factor, axis=1), col_factor, axis=2)


def prepare_slip_frames(
    frames: np.ndarray,
    active_rows: int,
    time_oversample: int,
    row_oversample: int,
    col_oversample: int,
    smooth_sigma: float,
) -> np.ndarray:
    start, stop = active_row_window(frames, active_rows)
    cropped = frames[:, start:stop, :]
    temporal = interpolate_frames_time(cropped, time_oversample)
    return oversample_spatial_frames(temporal, row_oversample, col_oversample, smooth_sigma)


def remove_contour(contour: object | None) -> None:
    if contour is None:
        return
    if hasattr(contour, "remove"):
        contour.remove()
        return
    for collection in getattr(contour, "collections", []):
        collection.remove()


def make_heatmap_video(
    frames: np.ndarray,
    out_path: Path,
    fps: int,
    dpi: int,
    max_frames: int | None,
    title: str,
    threshold: float | None = None,
    cmap: str = "RdBu_r",
    signed: bool = True,
    time_scale: float = 1.0,
    time_unit: str = "s",
) -> Path | None:
    indices = frame_indices(frames.shape[0], max_frames)
    if indices.size == 0:
        return None

    finite = frames[np.isfinite(frames)]
    if finite.size == 0:
        return None
    if signed:
        vmax = float(np.percentile(np.abs(finite), 98))
        vmax = max(vmax, 1.0)
        vmin = -vmax
    else:
        vmin = 0.0
        vmax = float(np.percentile(finite, 98))
        vmax = max(vmax, 1.0)

    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    im = ax.imshow(frames[indices[0]], cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
    contour_holder: list[object | None] = [None]
    set_taxel_axes(ax)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Response [LSB]")
    time_text = ax.text(
        0.02,
        1.02,
        "",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
    )
    ax.set_title(title)

    def update(pos: int):
        idx = int(indices[pos])
        frame = frames[idx]
        im.set_data(frame)
        remove_contour(contour_holder[0])
        contour_holder[0] = None
        if threshold is not None:
            active = np.abs(frame) >= threshold if signed else frame >= threshold
            if np.any(active):
                contour_holder[0] = ax.contour(
                    active.astype(float),
                    levels=[0.5],
                    colors="black",
                    linewidths=0.8,
                )
        time_text.set_text(f"t = {idx * time_scale:.3g} {time_unit}")
        return [im, time_text]

    anim = FuncAnimation(fig, update, frames=len(indices), interval=1000 / fps, blit=False)
    return save_animation(anim, out_path, fps=fps, dpi=dpi)


def make_slip_process_gif(
    frames: np.ndarray,
    out_path: Path,
    fps: int,
    dpi: int,
    max_frames: int | None,
    threshold: float,
    cmap: str,
    active_rows: int,
    time_oversample: int,
    row_oversample: int,
    col_oversample: int,
    smooth_sigma: float,
) -> Path | None:
    frames = prepare_slip_frames(
        frames=frames,
        active_rows=active_rows,
        time_oversample=time_oversample,
        row_oversample=row_oversample,
        col_oversample=col_oversample,
        smooth_sigma=smooth_sigma,
    )
    indices = frame_indices(frames.shape[0], max_frames)
    if indices.size == 0:
        return None

    finite = frames[np.isfinite(frames)]
    if finite.size == 0:
        return None
    vmax = float(np.percentile(np.abs(finite), 98))
    vmax = max(vmax, float(threshold), 1.0)

    fig, ax = plt.subplots(figsize=(8.0, 4.0), facecolor="white")
    im = ax.imshow(
        frames[indices[0]],
        cmap=cmap,
        vmin=-vmax,
        vmax=vmax,
        origin="upper",
        aspect="auto",
        interpolation="bicubic",
    )
    ax.set_axis_off()
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)

    def update(pos: int):
        im.set_data(frames[int(indices[pos])])
        return [im]

    anim = FuncAnimation(fig, update, frames=len(indices), interval=1000 / fps, blit=True)
    return save_animation(anim, out_path.with_suffix(".gif"), fps=fps, dpi=dpi)


def run_test8(args: argparse.Namespace) -> list[Path]:
    mod = load_module("video_test8_slip", SCRIPT_PATHS["test8"])
    trial_paths = mod.discover_trials(args.test8_data_root)
    if not trial_paths:
        print(f"test8: no tactile slip CSV files under {args.test8_data_root}")
        return []

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
    out_dir = ensure_dir(args.output_root / "test8_tactile_slip")
    saved: list[Path] = []

    for speed in sorted(speed for speed, trials in selected.items() if trials):
        trial = mod.representative_forward_cop_trial(selected[speed])
        speed_cmps = speed * 100.0
        out = out_dir / f"08_slip_process_{speed_cmps:g}cmps_{safe_stem(trial.path.stem)}.gif"
        path = make_slip_process_gif(
            trial.X_slip,
            out,
            fps=args.fps,
            dpi=args.dpi,
            max_frames=args.max_frames,
            threshold=mod.FIXED_ACTIVITY_THRESHOLD_LSB,
            cmap="RdBu_r",
            active_rows=args.test8_active_rows,
            time_oversample=args.test8_time_oversample,
            row_oversample=args.test8_row_oversample,
            col_oversample=args.test8_col_oversample,
            smooth_sigma=args.test8_smooth_sigma,
        )
        if path:
            saved.append(path)
            print(f"test8: saved {path}")

    if failures:
        print(f"test8: skipped {len(failures)} failed trials")
    return saved


def representative_force_trial(trials: list[object]) -> object | None:
    valid = [trial for trial in trials if getattr(trial, "rise_y", np.asarray([])).size >= 2]
    if not valid:
        return None
    scores = np.asarray([getattr(trial, "quality_score", np.nan) for trial in valid], dtype=float)
    if np.any(np.isfinite(scores)):
        return valid[int(np.nanargmin(scores))]
    lengths = np.asarray([trial.rise_y.size for trial in valid], dtype=int)
    return valid[int(np.argmax(lengths))]


def run_test9a(args: argparse.Namespace) -> list[Path]:
    mod = load_module("video_test9a_force", SCRIPT_PATHS["test9a"])
    trial_paths = mod.discover_force_rise_trials(args.force_data_root, tuple(args.force_groups))
    if not any(paths for paths in trial_paths.values()):
        print(f"test9a: no force CSV files under {args.force_data_root}")
        return []

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
    out_dir = ensure_dir(args.output_root / "test9a_tactile_force")
    saved: list[Path] = []

    for group in args.force_groups:
        trial = representative_force_trial(selected.get(group, []))
        if trial is None:
            continue
        out = out_dir / f"09a_force_step_{safe_stem(group)}_{safe_stem(trial.path.stem)}.mp4"
        path = make_curve_video(
            x=trial.rise_x,
            y=trial.rise_y,
            out_path=out,
            fps=args.fps,
            dpi=args.dpi,
            max_frames=args.max_frames,
            title=f"Test9a step response - {group}",
            xlabel="Time [s]",
            ylabel="Tactile envelope [LSB]",
            color=mod.FORCE_GROUP_COLORS.get(group, COLORS["blue"]),
        )
        if path:
            saved.append(path)
            print(f"test9a: saved {path}")

    if failures:
        print(f"test9a: skipped {len(failures)} failed trials")
    return saved


def choose_contact_trial(trials: list[object]) -> object | None:
    valid = [trial for trial in trials if np.isfinite(getattr(trial, "coverage_pct", np.nan))]
    if not valid:
        return trials[0] if trials else None
    coverage = np.asarray([trial.coverage_pct for trial in valid], dtype=float)
    center = float(np.nanmedian(coverage))
    return valid[int(np.nanargmin(np.abs(coverage - center)))]


def contact_abs_frames(mod: ModuleType, trial: object, apply_blur: bool, gaussian_sigma: float) -> np.ndarray:
    X = mod.load_trial(trial.path)
    X_bc, _baseline_std = mod.baseline_correct(X)
    X_filtered = mod.maybe_blur_frames(X_bc, apply_blur=apply_blur, sigma_cells=gaussian_sigma)
    stop = max(int(trial.t_start) + 1, int(trial.peak_idx) + 1, int(trial.t_stable_idx) + 1)
    stop = min(stop, X_filtered.shape[0])
    return np.abs(X_filtered[int(trial.t_start) : stop])


def make_contact_mask_video(
    mod: ModuleType,
    trial: object,
    out_path: Path,
    fps: int,
    dpi: int,
    max_frames: int | None,
    apply_blur: bool,
    gaussian_sigma: float,
) -> Path | None:
    frames = contact_abs_frames(mod, trial, apply_blur, gaussian_sigma)
    indices = frame_indices(frames.shape[0], max_frames)
    if indices.size == 0:
        return None
    vmax = float(np.percentile(frames[np.isfinite(frames)], 98))
    vmax = max(vmax, float(trial.threshold_lsb), 1.0)

    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    im = ax.imshow(frames[indices[0]], cmap="magma", vmin=0, vmax=vmax, origin="upper")
    contour_holder: list[object | None] = [None]
    set_taxel_axes(ax)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("|Response| [LSB]")
    time_text = ax.text(0.02, 1.02, "", transform=ax.transAxes, ha="left", va="bottom", fontsize=9)
    ax.set_title(f"Test9b contact mask - {trial.distance_label}")

    def update(pos: int):
        idx = int(indices[pos])
        frame = frames[idx]
        cumulative = np.max(frames[: idx + 1], axis=0) >= trial.threshold_lsb
        im.set_data(frame)
        remove_contour(contour_holder[0])
        contour_holder[0] = None
        if np.any(cumulative):
            contour_holder[0] = ax.contour(
                cumulative.astype(float),
                levels=[0.5],
                colors="white",
                linewidths=1.0,
            )
        time_text.set_text(f"t = {idx / mod.FS * 1000.0:.1f} ms")
        return [im, time_text]

    anim = FuncAnimation(fig, update, frames=len(indices), interval=1000 / fps, blit=False)
    return save_animation(anim, out_path, fps=fps, dpi=dpi)


def run_test9b(args: argparse.Namespace) -> list[Path]:
    mod = load_module("video_test9b_mask", SCRIPT_PATHS["test9b"])
    trial_paths = mod.discover_force_rise_trials(args.force_data_root, tuple(args.force_groups))
    if not any(paths for paths in trial_paths.values()):
        print(f"test9b: no force CSV files under {args.force_data_root}")
        return []

    results, failures = mod.batch_process_contact_masks(
        trial_paths_dict=trial_paths,
        apply_blur=args.gaussian_blur,
        gaussian_sigma=args.gaussian_sigma,
        threshold_lsb=args.activity_threshold,
    )
    out_dir = ensure_dir(args.output_root / "test9b_contact_mask")
    saved: list[Path] = []

    for group in args.force_groups:
        trial = choose_contact_trial(results.get(group, []))
        if trial is None:
            continue
        out = out_dir / f"09b_contact_mask_{safe_stem(group)}_{safe_stem(trial.path.stem)}.mp4"
        path = make_contact_mask_video(
            mod=mod,
            trial=trial,
            out_path=out,
            fps=args.fps,
            dpi=args.dpi,
            max_frames=args.max_frames,
            apply_blur=args.gaussian_blur,
            gaussian_sigma=args.gaussian_sigma,
        )
        if path:
            saved.append(path)
            print(f"test9b: saved {path}")

    if failures:
        print(f"test9b: skipped {len(failures)} failed trials")
    return saved


def make_curve_video(
    x: np.ndarray,
    y: np.ndarray,
    out_path: Path,
    fps: int,
    dpi: int,
    max_frames: int | None,
    title: str,
    xlabel: str,
    ylabel: str,
    color: str,
) -> Path | None:
    indices = frame_indices(len(x), max_frames)
    if indices.size == 0:
        return None

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(x, y, color=COLORS["gray"], linewidth=0.9, alpha=0.28)
    line, = ax.plot([], [], color=color, linewidth=2.0)
    point, = ax.plot([], [], "o", color=color, markersize=5)
    time_text = ax.text(0.02, 0.96, "", transform=ax.transAxes, ha="left", va="top", fontsize=9)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(float(np.nanmin(x)), float(np.nanmax(x)))
    ymin, ymax = float(np.nanmin(y)), float(np.nanmax(y))
    pad = max((ymax - ymin) * 0.08, 1.0)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.grid(True, color="#d9d9d9", linewidth=0.7, alpha=0.65)
    fig.tight_layout()

    def update(pos: int):
        idx = int(indices[pos])
        line.set_data(x[: idx + 1], y[: idx + 1])
        point.set_data([x[idx]], [y[idx]])
        time_text.set_text(f"t = {x[idx]:.3g} s")
        return line, point, time_text

    anim = FuncAnimation(fig, update, frames=len(indices), interval=1000 / fps, blit=False)
    return save_animation(anim, out_path, fps=fps, dpi=dpi)


def build_temperature_trials(mod: ModuleType, args: argparse.Namespace, point: str, data_dir: Path) -> list[object]:
    files = mod.discover_temperature_files(data_dir, args.temp_pattern)
    trials = []
    failures = 0
    for path in files:
        try:
            trials.append(
                mod.load_and_process_trial(
                    csv_path=path,
                    sampling_rate_hz=args.temp_sampling_rate_hz,
                    cold_junction_c=args.cold_junction_c,
                    offset_mode=args.temp_offset_mode,
                    fixed_offset_raw16=args.temp_offset_raw16,
                    infer_jaw0_from_initial_lsb=bool(args.infer_jaw0_from_initial_lsb),
                    baseline_window_s=args.temp_baseline_window_s,
                    noise_tail_s=args.temp_noise_tail_s,
                    contact_threshold_raw16=args.temp_contact_threshold_raw16,
                    contact_rise_frames=args.temp_contact_rise_frames,
                    contact_fit_frames=args.temp_contact_fit_frames,
                    contact_smooth_window=args.temp_contact_smooth_window,
                )
            )
        except Exception as exc:
            failures += 1
            print(f"test10: skipped {point}/{path.name}: {exc}")
    if not trials:
        return []
    if not args.temp_skip_fit:
        mod.fit_trials(
            trials,
            fit_start_skip_s=args.temp_fit_start_skip_s,
            baseline_window_s=args.temp_baseline_window_s,
            noise_tail_s=args.temp_noise_tail_s,
        )
        mod.fit_normalized_trials(trials)
        trials, r2_filtered = mod.filter_trials_by_r2(trials, args.temp_min_r2)
        failures += len(r2_filtered)
    elif any(trial.normalized_fit is None for trial in trials):
        mod.fit_normalized_trials(trials)
    trials, consistency_filtered = mod.filter_trials_by_consistency(trials)
    failures += len(consistency_filtered)
    if failures:
        print(f"test10: {point} skipped/filtered {failures} trials")
    plot_trials, _dropped = mod.select_trials_for_temperature_plot(
        trials,
        n_heating=mod.PLOT_A_NUM_HEATING_CURVES,
        n_cooling=mod.PLOT_A_NUM_COOLING_CURVES,
        mode=mod.PLOT_A_SELECTION_MODE,
        point_name=point,
    )
    return plot_trials


def make_temperature_video(
    mod: ModuleType,
    trials: list[object],
    point: str,
    direction: str,
    out_path: Path,
    fps: int,
    dpi: int,
    max_frames: int | None,
    cold_junction_c: float,
) -> Path | None:
    if not trials:
        return None
    t_max = max(float(trial.t_s[-1]) for trial in trials if trial.t_s.size)
    indices = frame_indices(1000, max_frames)
    if indices.size == 0:
        return None
    time_grid = np.linspace(0.0, t_max, 1000)
    frame_times = time_grid[indices]

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    if direction == "heating":
        palette = [COLORS["red"], COLORS["orange"], COLORS["purple"], COLORS["green"]]
    elif direction == "cooling":
        palette = [COLORS["blue"], COLORS["teal"], COLORS["purple"], COLORS["green"]]
    else:
        palette = [COLORS["blue"], COLORS["teal"], COLORS["green"], COLORS["orange"], COLORS["red"], COLORS["purple"]]
    lines = []
    markers = []
    smoothed = []
    for idx, trial in enumerate(trials):
        color = palette[idx % len(palette)]
        smooth_window = min(mod.SMOOTH_WINDOW, trial.T_c.size)
        y_smooth = mod.moving_average(trial.T_c, smooth_window)
        smoothed.append(y_smooth)
        ax.plot(trial.t_s, y_smooth, color=color, linewidth=0.75, alpha=0.18)
        line, = ax.plot([], [], color=color, linewidth=1.65, label=trial.label)
        marker, = ax.plot([], [], "o", color=color, markersize=4)
        lines.append(line)
        markers.append(marker)

    ax.axhline(cold_junction_c, color=COLORS["gray"], linewidth=0.9, linestyle="--")
    time_line = ax.axvline(0.0, color="black", linewidth=0.8, alpha=0.7)
    time_text = ax.text(0.02, 0.96, "", transform=ax.transAxes, ha="left", va="top", fontsize=9)
    ax.set_title(f"Test10 {direction} temperature-time curves - {point}")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Temperature [degC]")
    ax.set_xlim(0, t_max)
    all_y = np.concatenate([trial.T_c for trial in trials if trial.T_c.size])
    ymin, ymax = float(np.nanmin(all_y)), float(np.nanmax(all_y))
    pad = max((ymax - ymin) * 0.08, 0.5)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.grid(True, color="#d9d9d9", linewidth=0.7, alpha=0.65)
    ax.legend(frameon=False, fontsize=6.2, loc="best")
    fig.tight_layout()

    def update(pos: int):
        now = float(frame_times[pos])
        for trial, y_smooth, line, marker in zip(trials, smoothed, lines, markers):
            end = int(np.searchsorted(trial.t_s, now, side="right"))
            if end <= 0:
                line.set_data([], [])
                marker.set_data([], [])
            else:
                line.set_data(trial.t_s[:end], y_smooth[:end])
                marker.set_data([trial.t_s[end - 1]], [y_smooth[end - 1]])
        time_line.set_xdata([now, now])
        time_text.set_text(f"t = {now:.3g} s")
        return [*lines, *markers, time_line, time_text]

    anim = FuncAnimation(fig, update, frames=len(frame_times), interval=1000 / fps, blit=False)
    return save_animation(anim, out_path, fps=fps, dpi=dpi)


def split_temperature_trials_by_direction(
    mod: ModuleType,
    trials: list[object],
) -> dict[str, list[object]]:
    grouped: dict[str, list[object]] = {"heating": [], "cooling": []}
    for trial in trials:
        slope = float(mod.compute_response_window_slope_c_per_s(trial))
        if not np.isfinite(slope) or slope == 0.0:
            slope = float(getattr(trial, "steady_c", np.nan) - getattr(trial, "baseline_c", np.nan))
        if slope > 0:
            grouped["heating"].append(trial)
        elif slope < 0:
            grouped["cooling"].append(trial)
    return grouped


def run_test10(args: argparse.Namespace) -> list[Path]:
    mod = load_module("video_test10_temperature", SCRIPT_PATHS["test10"])
    out_dir = ensure_dir(args.output_root / "test10_temperature")
    saved: list[Path] = []
    for point in args.temp_points:
        data_dir = args.temp_data_root / point
        if not data_dir.exists():
            print(f"test10: skipping {point}, missing {data_dir}")
            continue
        trials = build_temperature_trials(mod, args, point, data_dir)
        if not trials:
            print(f"test10: skipping {point}, no usable trials")
            continue
        by_direction = split_temperature_trials_by_direction(mod, trials)
        for direction in ("heating", "cooling"):
            direction_trials = by_direction[direction]
            if not direction_trials:
                print(f"test10: skipping {point}/{direction}, no usable trials")
                continue
            out = out_dir / f"10_temperature_{safe_stem(point)}_{direction}.mp4"
            path = make_temperature_video(
                mod=mod,
                trials=direction_trials,
                point=point,
                direction=direction,
                out_path=out,
                fps=args.fps,
                dpi=args.dpi,
                max_frames=args.max_frames,
                cold_junction_c=args.cold_junction_c,
            )
            if path:
                saved.append(path)
                print(f"test10: saved {path}")
    return saved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate VET6USB experiment animation videos only.")
    parser.add_argument(
        "--tests",
        nargs="+",
        choices=("test8", "test9a", "test9b", "test10", "all"),
        default=["all"],
    )
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--fps", type=int, default=18)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--max-frames", type=int, default=180, help="0 means use every source frame.")
    parser.add_argument("--gaussian-blur", action="store_true", default=False)
    parser.add_argument("--gaussian-sigma", type=float, default=0.65)
    parser.add_argument("--activity-threshold", type=float, default=35.0)

    parser.add_argument("--test8-data-root", type=Path, default=REPO_ROOT / "workspace" / "tact_sliptest")
    parser.add_argument("--test8-select-n", type=int, default=5)
    parser.add_argument("--test8-selection-feature", default="f_centroid")
    parser.add_argument("--test8-active-rows", type=int, default=4)
    parser.add_argument("--test8-time-oversample", type=int, default=4)
    parser.add_argument("--test8-row-oversample", type=int, default=2)
    parser.add_argument("--test8-col-oversample", type=int, default=2)
    parser.add_argument("--test8-smooth-sigma", type=float, default=0.55)

    parser.add_argument("--force-data-root", type=Path, default=REPO_ROOT / "workspace" / "tact_forcetest")
    parser.add_argument("--force-groups", nargs="+", default=["6cm"])
    parser.add_argument("--force-select-n", type=int, default=10)

    parser.add_argument("--temp-data-root", type=Path, default=REPO_ROOT / "workspace" / "temp_test_difftemp")
    parser.add_argument("--temp-points", nargs="+", default=["P0", "P1", "P2", "P3"])
    parser.add_argument("--temp-pattern", default="*.csv")
    parser.add_argument("--temp-sampling-rate-hz", type=float, default=400.0)
    parser.add_argument("--cold-junction-c", type=float, default=25.0)
    parser.add_argument("--temp-offset-mode", choices=("per-file-baseline", "fixed"), default="per-file-baseline")
    parser.add_argument("--temp-offset-raw16", type=float, default=None)
    parser.add_argument("--infer-jaw0-from-initial-lsb", type=int, choices=(0, 1), default=0)
    parser.add_argument("--temp-baseline-window-s", type=float, default=0.5)
    parser.add_argument("--temp-noise-tail-s", type=float, default=5.0)
    parser.add_argument("--temp-fit-start-skip-s", type=float, default=2.0)
    parser.add_argument("--temp-contact-threshold-raw16", type=float, default=35.0)
    parser.add_argument("--temp-contact-rise-frames", type=int, default=2)
    parser.add_argument("--temp-contact-fit-frames", type=int, default=12)
    parser.add_argument("--temp-contact-smooth-window", type=int, default=9)
    parser.add_argument("--temp-min-r2", type=float, default=0.0)
    parser.add_argument("--temp-skip-fit", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    apply_measurement_style()
    args.output_root = ensure_dir(args.output_root)
    if args.max_frames <= 0:
        args.max_frames = None

    tests = ["test8", "test9a", "test9b", "test10"] if "all" in args.tests else args.tests
    runners = {
        "test8": run_test8,
        "test9a": run_test9a,
        "test9b": run_test9b,
        "test10": run_test10,
    }
    saved: list[Path] = []
    for test in tests:
        saved.extend(runners[test](args))

    print("Video generation complete")
    print(f"Saved videos: {len(saved)}")
    for path in saved:
        print(path)


if __name__ == "__main__":
    main()
