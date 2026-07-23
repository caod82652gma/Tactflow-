#!/usr/bin/env python
# coding: utf-8

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "cycle_noise" / "Temp_AD2_20260508_142306.csv"
CSV_DIR = SCRIPT_DIR / "csv"
PNG_DIR = SCRIPT_DIR / "png"
CN_PNG_DIR = SCRIPT_DIR / "png_cn"
PDF_DIR = SCRIPT_DIR / "pdf"
for folder in (CSV_DIR, PNG_DIR, CN_PNG_DIR, PDF_DIR):
    folder.mkdir(parents=True, exist_ok=True)

FS = 200.0
V_REF = 5.0
ADC_FULL_SCALE = 2**15
DETREND_WINDOW_SAMPLES = 401

CN_TEXT_RULES = {
    "Time (s)": "时间 (s)",
    "Noise after drift removal (LSB)": "去漂移后噪声 (LSB)",
    "TemperatureRaw16 detrended noise (LSB)": "TemperatureRaw16 去趋势噪声 (LSB)",
    "Count": "计数",
    "Temperature detrended noise": "温度去趋势噪声",
    "VGND detrended noise": "VGND 去趋势噪声",
    "Temp ±1σ": "温度 ±1σ",
}


def cn_label(text):
    if text is None:
        return text
    translated = str(text)
    for english, chinese in sorted(CN_TEXT_RULES.items(), key=lambda item: len(item[0]), reverse=True):
        translated = translated.replace(english, chinese)
    return translated


def save_chinese_png(fig, output_path: Path) -> Path:
    originals = []
    for artist in fig.findobj(match=mpl.text.Text):
        original = artist.get_text()
        localized = cn_label(original)
        if localized != original:
            originals.append((artist, original))
            artist.set_text(localized)
    fig.savefig(output_path, dpi=300)
    for artist, original in originals:
        artist.set_text(original)
    return output_path


def resolve_input_path(text: str | None) -> Path:
    if not text:
        return DEFAULT_INPUT

    path = Path(text)
    candidates = [path]
    if not path.is_absolute():
        candidates = [
            Path.cwd() / path,
            SCRIPT_DIR / path,
            SCRIPT_DIR.parent / path,
        ]

    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.exists():
            return candidate
    return candidates[0].resolve()


def read_csv_columns(path: Path) -> dict[str, np.ndarray]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        columns: dict[str, list[float]] = {name: [] for name in reader.fieldnames or []}
        for row in reader:
            for name in columns:
                text = (row.get(name) or "").strip()
                if text:
                    columns[name].append(float(text))

    if not columns:
        raise ValueError(f"{path} has no readable CSV columns.")
    return {name: np.asarray(values, dtype=float) for name, values in columns.items()}


def raw16_to_visual_mv(raw16: float | np.ndarray) -> float | np.ndarray:
    return np.asarray(raw16) / ADC_FULL_SCALE * V_REF * 1000.0


def residual(values: np.ndarray) -> np.ndarray:
    return values - float(np.mean(values))


def rolling_median_baseline(values: np.ndarray, window_size: int = DETREND_WINDOW_SAMPLES) -> np.ndarray:
    window_size = min(window_size, len(values))
    if window_size % 2 == 0:
        window_size -= 1
    if window_size < 3:
        return np.full_like(values, float(np.mean(values)), dtype=float)

    half = window_size // 2
    padded = np.pad(values.astype(float), (half, half), mode="edge")
    baseline = np.empty(len(values), dtype=float)
    for index in range(len(values)):
        baseline[index] = np.median(padded[index:index + window_size])
    return baseline


def detrended_noise(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    baseline = rolling_median_baseline(values)
    noise = values.astype(float) - baseline
    noise = noise - float(np.mean(noise))
    return noise, baseline


def percentile_pp(values: np.ndarray, low: float = 0.5, high: float = 99.5) -> float:
    lo, hi = np.percentile(values, [low, high])
    return float(hi - lo)


def summarize_signal(name: str, values: np.ndarray) -> dict[str, object]:
    noise, baseline = detrended_noise(values)
    mean_removed_noise = residual(values)
    std_lsb = float(np.std(noise, ddof=1)) if len(noise) > 1 else 0.0
    rms_lsb = float(np.sqrt(np.mean(noise * noise)))
    pp_lsb = float(np.max(noise) - np.min(noise))
    pp_99_lsb = percentile_pp(noise)
    std_mv = float(raw16_to_visual_mv(std_lsb))

    return {
        "Signal": name,
        "Samples": len(values),
        "Mean_Raw16": round(float(np.mean(values)), 6),
        "Min_Raw16": round(float(np.min(values)), 6),
        "Max_Raw16": round(float(np.max(values)), 6),
        "Detrend_Method": f"rolling_median_{DETREND_WINDOW_SAMPLES}_samples",
        "Estimated_Baseline_Min_Raw16": round(float(np.min(baseline)), 6),
        "Estimated_Baseline_Max_Raw16": round(float(np.max(baseline)), 6),
        "Estimated_Baseline_PeakToPeak_LSB": round(float(np.max(baseline) - np.min(baseline)), 6),
        "MeanRemoved_Std_LSB": round(float(np.std(mean_removed_noise, ddof=1)), 6),
        "Noise_Mean_LSB": round(float(np.mean(noise)), 6),
        "Noise_Std_LSB": round(std_lsb, 6),
        "Noise_RMS_LSB": round(rms_lsb, 6),
        "Noise_PeakToPeak_LSB": round(pp_lsb, 6),
        "Noise_P99_PeakToPeak_LSB": round(pp_99_lsb, 6),
        "Noise_6Sigma_LSB": round(6.0 * std_lsb, 6),
        "Noise_Std_Visual_mV": round(std_mv, 6),
        "Noise_6Sigma_Visual_mV": round(float(raw16_to_visual_mv(6.0 * std_lsb)), 6),
    }


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def format_number(value: float) -> str:
    if np.isfinite(value) and abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.6f}"


def write_detrended_source_csv(input_path: Path, output_path: Path) -> None:
    columns = read_csv_columns(input_path)
    if "TemperatureRaw16" not in columns:
        raise ValueError("Expected column TemperatureRaw16 in the input CSV.")

    temperature_raw = columns["TemperatureRaw16"]
    temp_noise, temp_baseline = detrended_noise(temperature_raw)
    temp_filtered = temp_noise + float(np.mean(temperature_raw))

    ground_raw = columns.get("VGNDRaw16")
    ground_noise = ground_baseline = ground_filtered = None
    if ground_raw is not None:
        ground_noise, ground_baseline = detrended_noise(ground_raw)
        ground_filtered = ground_noise + float(np.mean(ground_raw))

    with input_path.open("r", newline="", encoding="utf-8-sig") as src, output_path.open(
        "w", newline="", encoding="utf-8"
    ) as dst:
        reader = csv.DictReader(src)
        source_fields = reader.fieldnames or []
        extra_fields = [
            "TemperatureBaselineRaw16",
            "TemperatureRaw16_Filtered",
            "TemperatureNoiseRaw16",
        ]
        if ground_raw is not None:
            extra_fields.extend(["VGNDBaselineRaw16", "VGNDRaw16_Filtered", "VGNDNoiseRaw16"])

        writer = csv.DictWriter(dst, fieldnames=source_fields + extra_fields)
        writer.writeheader()

        for index, row in enumerate(reader):
            row["TemperatureBaselineRaw16"] = format_number(float(temp_baseline[index]))
            row["TemperatureRaw16_Filtered"] = format_number(float(temp_filtered[index]))
            row["TemperatureNoiseRaw16"] = format_number(float(temp_noise[index]))
            if ground_raw is not None:
                row["VGNDBaselineRaw16"] = format_number(float(ground_baseline[index]))
                row["VGNDRaw16_Filtered"] = format_number(float(ground_filtered[index]))
                row["VGNDNoiseRaw16"] = format_number(float(ground_noise[index]))
            writer.writerow(row)


def make_noise_figure(
    input_path: Path,
    temperature_raw: np.ndarray,
    ground_raw: np.ndarray | None,
    summary: list[dict[str, object]],
) -> Path:
    temp_noise, temp_baseline = detrended_noise(temperature_raw)
    time_s = np.arange(len(temperature_raw), dtype=float) / FS

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3))

    ax = axes[0]
    ax.plot(time_s, temp_noise, color="#3366cc", linewidth=0.55, label="Temperature detrended noise")
    if ground_raw is not None:
        ground_noise, _ = detrended_noise(ground_raw)
        ax.plot(time_s, ground_noise, color="#cc5533", linewidth=0.45, alpha=0.55, label="VGND detrended noise")
    temp_std = float(summary[0]["Noise_Std_LSB"])
    ax.axhline(0, color="#777777", linewidth=0.8, linestyle="--")
    ax.fill_between(time_s, -temp_std, temp_std, color="#3366cc", alpha=0.12, label="Temp ±1σ")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Noise after drift removal (LSB)")
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.45)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    bins = np.arange(np.floor(temp_noise.min()) - 0.5, np.ceil(temp_noise.max()) + 1.5, 1.0)
    ax.hist(temp_noise, bins=bins, color="#3366cc", alpha=0.72, edgecolor="none")
    ax.axvline(0, color="#777777", linewidth=0.8, linestyle="--")
    ax.set_xlabel("TemperatureRaw16 detrended noise (LSB)")
    ax.set_ylabel("Count")
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.45)
    ax.text(
        0.97,
        0.95,
        f"σ = {summary[0]['Noise_Std_LSB']:.3f} LSB\n"
        f"6σ = {summary[0]['Noise_6Sigma_LSB']:.3f} LSB\n"
        f"P-P = {summary[0]['Noise_PeakToPeak_LSB']:.1f} LSB\n"
        f"drift P-P = {summary[0]['Estimated_Baseline_PeakToPeak_LSB']:.1f} LSB",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.85),
    )

    fig.tight_layout()

    png_path = PNG_DIR / "01_noise_overview.png"
    cn_png_path = CN_PNG_DIR / "01_noise_overview.png"
    pdf_path = PDF_DIR / "01_noise_overview.pdf"
    fig.savefig(png_path, dpi=300)
    save_chinese_png(fig, cn_png_path)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path


def main() -> None:
    input_path = resolve_input_path(sys.argv[1] if len(sys.argv) > 1 else None)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    columns = read_csv_columns(input_path)
    if "TemperatureRaw16" not in columns:
        raise ValueError("Expected column TemperatureRaw16 in the input CSV.")

    temperature_raw = columns["TemperatureRaw16"]
    ground_raw = columns.get("VGNDRaw16")

    summary = [summarize_signal("TemperatureRaw16", temperature_raw)]
    if ground_raw is not None:
        summary.append(summarize_signal("VGNDRaw16", ground_raw))

    summary_path = CSV_DIR / "01_noise_summary.csv"
    filtered_path = input_path.with_name(f"{input_path.stem}_detrended.csv")
    write_summary(summary_path, summary)
    write_detrended_source_csv(input_path, filtered_path)
    figure_path = make_noise_figure(input_path, temperature_raw, ground_raw, summary)

    print("Noise analysis completed")
    print(f"Input: {input_path}")
    print(f"Samples: {len(temperature_raw)}")
    print(f"Temperature noise std: {summary[0]['Noise_Std_LSB']:.4f} LSB")
    print(f"Temperature noise 6sigma: {summary[0]['Noise_6Sigma_LSB']:.4f} LSB")
    print(f"Saved figure: {figure_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved detrended CSV: {filtered_path}")


if __name__ == "__main__":
    main()
