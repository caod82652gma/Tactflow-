"""
Time-difference analysis for the 8-AD multi-channel tactile recordings.

The script trims each recording to the centered 1500 samples, splits that
window into 50 repeats of 30 samples, estimates the fundamental phase of each
AD channel in every repeat, and converts pairwise phase differences to time
differences using the supplied excitation frequency.

Sign convention:
    Delta t(row, col) = t_row - t_col.
    Positive values mean the row channel lags the column channel.

conda activate vet6usb_pyqt
python Vet6USB_curve_draw/A_calibration_plots/07_Time_difftest.py

"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PLOT_ROOT = SCRIPT_DIR.parent
REPO_ROOT = PLOT_ROOT.parent
sys.path.insert(0, str(PLOT_ROOT))

from plot_style import COLORS, apply_measurement_style, clear_experiment_outputs, save_chinese_png  # noqa: E402


OUTPUT_DIR = PLOT_ROOT / "result_display" / "A_calibration" / "test7_timediff"
DATA_ROOT = REPO_ROOT / "Workspace" / "A_calibration" / "A07_time_response"

# Experiment parameters. Edit these values when the excitation setting changes.
SIN_VOLTAGE = "V+-2.18"
SIN_FREQUENCY_HZ = 4.63e3
TRIANGLE_VOLTAGE = "V+-2.06"
TRIANGLE_FREQUENCY_HZ = 84.90e3

# Final Delta t correction for the sine group, applied before plotting/tables.
# Each pair involving these channels is divided by the configured divisor.
# If a pair matches more than one rule, the earlier rule takes precedence.
SIN_DEFAULT_PAIR_DIVISOR = 4.0
SIN_PAIR_DIVISOR_PRIORITY = (
    ("AD3", 5.0),
    ("AD4", 4.0),
)

CROP_POINTS = 1500
REPEAT_COUNT = 50
POINTS_PER_REPEAT = CROP_POINTS // REPEAT_COUNT
AD_CHANNELS = [f"AD{i}" for i in range(1, 9)]
TACTILE_COLUMNS = [f"{ad}_TactileRaw(16bit)" for ad in AD_CHANNELS]


def format_frequency_label(freq_hz: float) -> str:
    return f"{freq_hz / 1e3:.2f}k".replace(".", "p")


@dataclass(frozen=True)
class MeasurementConfig:
    label: str
    waveform: str
    voltage: str
    signal_frequency_hz: float
    csv_path: Path


@dataclass
class MeasurementResult:
    config: MeasurementConfig
    crop_start: int
    crop_end: int
    observed_cycles_per_sample: float
    phase_repeats_rad: np.ndarray
    delta_t_repeats_s: np.ndarray
    delta_phase_repeats_rad: np.ndarray

    @property
    def delta_t_mean_s(self) -> np.ndarray:
        return np.mean(self.delta_t_repeats_s, axis=0)

    @property
    def delta_t_std_s(self) -> np.ndarray:
        return np.std(self.delta_t_repeats_s, axis=0, ddof=1)

    @property
    def delta_phase_mean_deg(self) -> np.ndarray:
        return np.rad2deg(circular_mean(self.delta_phase_repeats_rad, axis=0))

    @property
    def delta_phase_std_deg(self) -> np.ndarray:
        return np.rad2deg(circular_std(self.delta_phase_repeats_rad, axis=0))


DEFAULT_MEASUREMENTS = (
    MeasurementConfig(
        label=f"sin_{format_frequency_label(SIN_FREQUENCY_HZ)}",
        waveform="sine",
        voltage=SIN_VOLTAGE,
        signal_frequency_hz=SIN_FREQUENCY_HZ,
        csv_path=DATA_ROOT / "sin.csv",
    ),
    MeasurementConfig(
        label=f"triangle_{format_frequency_label(TRIANGLE_FREQUENCY_HZ)}",
        waveform="triangle",
        voltage=TRIANGLE_VOLTAGE,
        signal_frequency_hz=TRIANGLE_FREQUENCY_HZ,
        csv_path=DATA_ROOT / "triangle.csv",
    ),
)


def wrap_to_pi(angle_rad: np.ndarray) -> np.ndarray:
    return np.angle(np.exp(1j * angle_rad))


def circular_mean(angle_rad: np.ndarray, axis: int = 0) -> np.ndarray:
    return np.angle(np.mean(np.exp(1j * angle_rad), axis=axis))


def circular_std(angle_rad: np.ndarray, axis: int = 0) -> np.ndarray:
    resultant = np.abs(np.mean(np.exp(1j * angle_rad), axis=axis))
    resultant = np.clip(resultant, 1e-12, 1.0)
    return np.sqrt(-2.0 * np.log(resultant))


def centered_crop(values: np.ndarray, n_points: int) -> tuple[np.ndarray, int, int]:
    if values.shape[0] < n_points:
        raise ValueError(f"Need at least {n_points} samples, got {values.shape[0]}")
    start = (values.shape[0] - n_points) // 2
    end = start + n_points
    return values[start:end], start, end


def load_tactile_csv(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    missing = [col for col in TACTILE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    return df[TACTILE_COLUMNS].to_numpy(dtype=float)


def estimate_observed_frequency_cycles_per_sample(values: np.ndarray) -> float:
    """Estimate the dominant observed fundamental from AD1 after center crop."""
    ad1 = values[:, 0] - np.mean(values[:, 0])
    window = np.hanning(ad1.size)
    spectrum = np.abs(np.fft.rfft(ad1 * window))
    spectrum[0] = 0.0

    peak_idx = int(np.argmax(spectrum))
    if peak_idx <= 0:
        raise ValueError("Could not estimate a non-zero observed frequency")

    # Quadratic interpolation around the peak gives a sub-bin frequency estimate.
    if 1 <= peak_idx < spectrum.size - 1:
        alpha = spectrum[peak_idx - 1]
        beta = spectrum[peak_idx]
        gamma = spectrum[peak_idx + 1]
        denom = alpha - 2.0 * beta + gamma
        offset = 0.5 * (alpha - gamma) / denom if abs(denom) > 1e-12 else 0.0
        offset = float(np.clip(offset, -0.5, 0.5))
    else:
        offset = 0.0
    return (peak_idx + offset) / ad1.size


def estimate_segment_phases(segment: np.ndarray, cycles_per_sample: float) -> np.ndarray:
    """Return one fundamental phase per channel for one repeat segment."""
    centered = segment - np.mean(segment, axis=0, keepdims=True)
    n = np.arange(segment.shape[0], dtype=float)[:, None]
    reference = np.exp(-2j * np.pi * cycles_per_sample * n)
    coeff = np.sum(centered * reference, axis=0)
    return np.angle(coeff)


def apply_final_delta_corrections(config: MeasurementConfig, delta_t: np.ndarray) -> np.ndarray:
    corrected = delta_t.copy()
    if config.waveform != "sine":
        return corrected

    for row in range(len(AD_CHANNELS)):
        for col in range(row):
            pair_channels = {AD_CHANNELS[row], AD_CHANNELS[col]}
            pair_divisor = SIN_DEFAULT_PAIR_DIVISOR
            for channel, divisor in SIN_PAIR_DIVISOR_PRIORITY:
                if channel in pair_channels:
                    pair_divisor = divisor
                    break
            corrected[:, row, col] = corrected[:, row, col] / pair_divisor
            corrected[:, col, row] = -corrected[:, row, col]
    return corrected


def analyze_measurement(config: MeasurementConfig) -> MeasurementResult:
    raw = load_tactile_csv(config.csv_path)
    cropped, crop_start, crop_end = centered_crop(raw, CROP_POINTS)
    observed_freq = estimate_observed_frequency_cycles_per_sample(cropped)

    repeats = cropped.reshape(REPEAT_COUNT, POINTS_PER_REPEAT, len(AD_CHANNELS))
    phase_repeats = np.vstack(
        [estimate_segment_phases(segment, observed_freq) for segment in repeats]
    )

    phase_i = phase_repeats[:, :, None]
    phase_j = phase_repeats[:, None, :]
    delta_phase = wrap_to_pi(phase_i - phase_j)

    # Positive Delta t means row channel lags column channel.
    delta_t = -delta_phase / (2.0 * np.pi * config.signal_frequency_hz)
    delta_t = apply_final_delta_corrections(config, delta_t)

    return MeasurementResult(
        config=config,
        crop_start=crop_start,
        crop_end=crop_end,
        observed_cycles_per_sample=observed_freq,
        phase_repeats_rad=phase_repeats,
        delta_t_repeats_s=delta_t,
        delta_phase_repeats_rad=delta_phase,
    )


def save_png_figure(fig: plt.Figure, output_dir: Path, stem: str) -> str:
    png_dir = output_dir / "png"
    png_dir.mkdir(parents=True, exist_ok=True)
    path = png_dir / f"{stem}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    save_chinese_png(fig, path)
    return str(path)


def plot_mean_std_heatmap(result: MeasurementResult, output_dir: Path) -> str:
    mean_ns = result.delta_t_mean_s * 1e9
    std_ns = result.delta_t_std_s * 1e9
    max_abs = float(np.nanmax(np.abs(mean_ns)))
    color_limit = max(max_abs, 1e-9)

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    im = ax.imshow(mean_ns, cmap="coolwarm", vmin=-color_limit, vmax=color_limit)
    ax.set_xticks(np.arange(len(AD_CHANNELS)), labels=AD_CHANNELS)
    ax.set_yticks(np.arange(len(AD_CHANNELS)), labels=AD_CHANNELS)
    ax.set_xlabel("Reference channel")
    ax.set_ylabel("Compared channel")
    for row in range(len(AD_CHANNELS)):
        for col in range(len(AD_CHANNELS)):
            text_color = "white" if abs(mean_ns[row, col]) > 0.55 * color_limit else "black"
            ax.text(
                col,
                row,
                f"{mean_ns[row, col]:.1f}\n+/-{std_ns[row, col]:.1f}",
                ha="center",
                va="center",
                fontsize=7,
                color=text_color,
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mean Delta t (ns)")
    fig.tight_layout()
    path = save_png_figure(fig, output_dir, f"07_{result.config.label}_delta_t_mean_std_heatmap")
    plt.close(fig)
    return path


def plot_pairwise_boxplot(result: MeasurementResult, output_dir: Path) -> str:
    repeats_ns = result.delta_t_repeats_s * 1e9
    labels: list[str] = []
    data: list[np.ndarray] = []
    for row in range(1, len(AD_CHANNELS)):
        labels.append(f"{AD_CHANNELS[row]}-AD1")
        data.append(repeats_ns[:, row, 0])

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    box = ax.boxplot(
        data,
        tick_labels=labels,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color=COLORS["red"], linewidth=1.2),
        boxprops=dict(facecolor=COLORS["teal"], alpha=0.65, linewidth=0.8),
        whiskerprops=dict(color=COLORS["gray"], linewidth=0.8),
        capprops=dict(color=COLORS["gray"], linewidth=0.8),
    )
    for patch in box["boxes"]:
        patch.set_edgecolor(COLORS["gray"])

    x_positions = np.arange(1, len(data) + 1)
    rng = np.random.default_rng(7)
    for x_pos, values in zip(x_positions, data):
        jitter = rng.normal(0.0, 0.035, size=values.size)
        ax.scatter(
            np.full(values.size, x_pos) + jitter,
            values,
            s=10,
            color=COLORS["blue"],
            alpha=0.35,
            edgecolors="none",
        )

    ax.axhline(0.0, color=COLORS["gray"], linewidth=0.9, linestyle="--")
    ax.set_ylabel("Delta t vs AD1 (ns)")
    ax.tick_params(axis="x", labelrotation=25)
    fig.tight_layout()
    path = save_png_figure(fig, output_dir, f"07_{result.config.label}_delta_to_AD1_boxplot")
    plt.close(fig)
    return path


def plot_unique_pair_boxplot(result: MeasurementResult, output_dir: Path) -> str:
    repeats_ns = result.delta_t_repeats_s * 1e9
    labels: list[str] = []
    data: list[np.ndarray] = []
    for row in range(len(AD_CHANNELS)):
        for col in range(row):
            labels.append(f"{AD_CHANNELS[row]}-{AD_CHANNELS[col]}")
            data.append(repeats_ns[:, row, col])

    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    ax.boxplot(
        data,
        tick_labels=labels,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color=COLORS["red"], linewidth=1.0),
        boxprops=dict(facecolor=COLORS["blue"], alpha=0.45, linewidth=0.7),
        whiskerprops=dict(color=COLORS["gray"], linewidth=0.7),
        capprops=dict(color=COLORS["gray"], linewidth=0.7),
    )
    ax.axhline(0.0, color=COLORS["gray"], linewidth=0.9, linestyle="--")
    ax.set_ylabel("Delta t (ns)")
    ax.tick_params(axis="x", labelrotation=75, labelsize=7)
    fig.tight_layout()
    path = save_png_figure(fig, output_dir, f"07_{result.config.label}_all_pair_boxplot")
    plt.close(fig)
    return path


def write_summary_metrics(result: MeasurementResult, output_dir: Path) -> None:
    csv_dir = output_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = csv_dir / f"07_{result.config.label}_delta_to_AD1_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "measurement",
                "delta_channel",
                "mean_delta_t_ns",
                "std_delta_t_ns",
            ]
        )

        ad1_repeats_ns = result.delta_t_repeats_s[:, :, 0] * 1e9
        mean_values: list[float] = []
        std_values: list[float] = []
        for ch_idx, channel in enumerate(AD_CHANNELS[1:], start=1):
            values = ad1_repeats_ns[:, ch_idx]
            mean_delta = float(np.mean(values))
            std_delta = float(np.std(values, ddof=1))
            mean_values.append(mean_delta)
            std_values.append(std_delta)
            writer.writerow(
                [
                    result.config.label,
                    f"{channel}_vs_AD1",
                    f"{mean_delta:.6f}",
                    f"{std_delta:.6f}",
                ]
            )
        writer.writerow(
            [
                result.config.label,
                "board_pair_average",
                f"{np.mean(mean_values):.6f}",
                f"{np.mean(std_values):.6f}",
            ]
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for CSV and figure outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    apply_measurement_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_dir = args.output_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    clear_experiment_outputs((args.output_dir, csv_dir), args.output_dir, "07_")

    results = [analyze_measurement(config) for config in DEFAULT_MEASUREMENTS]
    figure_paths: list[str] = []
    for result in results:
        figure_paths.append(plot_mean_std_heatmap(result, args.output_dir))
        figure_paths.append(plot_pairwise_boxplot(result, args.output_dir))
        figure_paths.append(plot_unique_pair_boxplot(result, args.output_dir))
        write_summary_metrics(result, args.output_dir)

    print(f"Saved time-difference outputs to: {args.output_dir}")
    print("Figures:")
    for path in figure_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
