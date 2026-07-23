"""
Plot normalized patch temperature against liquid position.

Input CSV:
    Workspace/B_experiments/B11_temperature_diffpercent/diffper_test_main.csv
    Workspace/B_experiments/B11_temperature_diffpercent/diffper_test_forcemove.csv
    Workspace/B_experiments/B11_temperature_diffpercent/diffper_test_xmove.csv

Expected columns:
    group       repeated experiment id
    value1      environment temperature, T_env
    value2      patch temperature, T_patch
    volume_ml   liquid position / liquid volume

If a liquid-temperature column is present, it will be used. Otherwise the
liquid temperature for each group is estimated from that group's maximum
observed patch temperature.

Normalization:
    theta = (T_patch - T_env) / (T_liquid - T_env)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


SCRIPT_DIR = Path(__file__).resolve().parent
PLOT_ROOT = SCRIPT_DIR.parent
REPO_ROOT = PLOT_ROOT.parent
sys.path.insert(0, str(PLOT_ROOT))

from plot_style import COLORS, apply_measurement_style, save_figure  # noqa: E402


DATA_DIR = REPO_ROOT / "Workspace" / "B_experiments" / "B11_temperature_diffpercent"
DATASETS = (
    {
        "key": "main",
        "label": "Baseline condition",
        "path": DATA_DIR / "diffper_test_main.csv",
        "color": COLORS["red"],
        "fit": True,
    },
    {
        "key": "forcemove",
        "label": "Changed contact force moment",
        "path": DATA_DIR / "diffper_test_forcemove.csv",
        "color": COLORS["blue"],
        "fit": False,
    },
    {
        "key": "xmove",
        "label": "Changed contact position",
        "path": DATA_DIR / "diffper_test_xmove.csv",
        "color": COLORS["green"],
        "fit": False,
    },
)
OUTPUT_ROOT = PLOT_ROOT / "result_display" / "B_experiment" / "test11_temp_diffpercent"
SUMMARY_PATH = OUTPUT_ROOT / "normalized_temperature_summary.csv"
FIT_RESULTS_PATH = OUTPUT_ROOT / "normalized_temperature_fit_results.csv"

GROUP_COL = "group"
ENV_COL = "value1"
PATCH_COL = "value2"
POSITION_COL = "volume_ml"

LIQUID_TEMP_CANDIDATES = (
    "value3",
    "T_liquid_c",
    "t_liquid_c",
    "liquid_temp_c",
    "liquid_temperature_c",
    "T_liquid",
)


def exponential_saturation(x: np.ndarray, k: float) -> np.ndarray:
    return 1.0 - np.exp(-k * x)


def logistic_curve(x: np.ndarray, L: float, k: float, x0: float) -> np.ndarray:
    return L / (1.0 + np.exp(-k * (x - x0)))


def power_law(x: np.ndarray, a: float, b: float) -> np.ndarray:
    return a * np.power(np.clip(x, 0.0, None), b)


def sigmoid_curve(x: np.ndarray, y0: float, L: float, k: float, x0: float) -> np.ndarray:
    return y0 + L / (1.0 + np.exp(-k * (x - x0)))


def find_liquid_temperature_column(df: pd.DataFrame) -> str | None:
    """Return the first available liquid-temperature column, if present."""
    for col in LIQUID_TEMP_CANDIDATES:
        if col in df.columns:
            return col
    return None


def load_normalized_data(csv_path: Path) -> pd.DataFrame:
    """Load raw data and add T_liquid_c plus normalized theta columns."""
    df = pd.read_csv(csv_path)
    required = {GROUP_COL, ENV_COL, PATCH_COL, POSITION_COL}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    df = df.copy()
    for col in (GROUP_COL, ENV_COL, PATCH_COL, POSITION_COL):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[GROUP_COL, ENV_COL, PATCH_COL, POSITION_COL])

    liquid_col = find_liquid_temperature_column(df)
    if liquid_col is not None:
        liquid_temp = pd.to_numeric(df[liquid_col], errors="coerce")
        group_max_patch = df.groupby(GROUP_COL)[PATCH_COL].transform("max")
        df["T_liquid_c"] = liquid_temp.where(liquid_temp > df[ENV_COL], group_max_patch)
    else:
        df["T_liquid_c"] = df.groupby(GROUP_COL)[PATCH_COL].transform("max")

    denominator = df["T_liquid_c"] - df[ENV_COL]
    df["theta"] = np.where(
        np.isclose(denominator, 0.0),
        np.nan,
        (df[PATCH_COL] - df[ENV_COL]) / denominator,
    )
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["theta"])
    return df.sort_values([POSITION_COL, GROUP_COL]).reset_index(drop=True)


def summarize_by_position(df: pd.DataFrame, dataset_key: str, dataset_label: str) -> pd.DataFrame:
    """Compute mean theta and the sample standard deviation at each position."""
    summary = (
        df.groupby(POSITION_COL, as_index=False)
        .agg(
            theta_mean=("theta", "mean"),
            theta_std=("theta", "std"),
            theta_sem=("theta", lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0),
            n=("theta", "count"),
            T_env_mean_c=(ENV_COL, "mean"),
            T_patch_mean_c=(PATCH_COL, "mean"),
            T_liquid_mean_c=("T_liquid_c", "mean"),
        )
        .sort_values(POSITION_COL)
    )
    summary["theta_std"] = summary["theta_std"].fillna(0.0)
    summary["theta_sem"] = summary["theta_sem"].fillna(0.0)
    summary.insert(0, "dataset", dataset_key)
    summary.insert(1, "dataset_label", dataset_label)
    return summary


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate coefficient of determination."""
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


def fit_curves(summary: pd.DataFrame) -> list[dict]:
    """Fit candidate curves to the mean normalized response."""
    x = summary[POSITION_COL].to_numpy(dtype=float)
    y = summary["theta_mean"].to_numpy(dtype=float)

    fit_specs = [
        {
            "name": "Exp saturation",
            "func": exponential_saturation,
            "p0": [0.03],
            "bounds": ([0.0], [1.0]),
            "param_names": ["k"],
            "color": COLORS["blue"],
            "linestyle": "--",
        },
        {
            "name": "Logistic",
            "func": logistic_curve,
            "p0": [1.0, 0.08, 50.0],
            "bounds": ([0.0, 0.0, -200.0], [2.0, 1.0, 200.0]),
            "param_names": ["L", "k", "x0"],
            "color": COLORS["green"],
            "linestyle": "-.",
        },
        {
            "name": "Power law",
            "func": power_law,
            "p0": [0.02, 0.9],
            "bounds": ([0.0, 0.0], [10.0, 5.0]),
            "param_names": ["a", "b"],
            "color": COLORS["orange"],
            "linestyle": ":",
        },
        {
            "name": "Sigmoid",
            "func": sigmoid_curve,
            "p0": [0.0, 1.0, 0.08, 50.0],
            "bounds": ([-1.0, 0.0, 0.0, -200.0], [1.0, 2.0, 1.0, 200.0]),
            "param_names": ["y0", "L", "k", "x0"],
            "color": COLORS["purple"],
            "linestyle": (0, (5, 2)),
        },
    ]

    results = []
    for spec in fit_specs:
        try:
            popt, _ = curve_fit(
                spec["func"],
                x,
                y,
                p0=spec["p0"],
                bounds=spec["bounds"],
                maxfev=20000,
            )
            y_fit = spec["func"](x, *popt)
            results.append(
                {
                    "name": spec["name"],
                    "func": spec["func"],
                    "params": popt,
                    "param_names": spec["param_names"],
                    "r2": r_squared(y, y_fit),
                    "color": spec["color"],
                    "linestyle": spec["linestyle"],
                    "success": True,
                    "error": "",
                }
            )
        except Exception as exc:
            results.append(
                {
                    "name": spec["name"],
                    "func": spec["func"],
                    "params": np.array([], dtype=float),
                    "param_names": spec["param_names"],
                    "r2": np.nan,
                    "color": spec["color"],
                    "linestyle": spec["linestyle"],
                    "success": False,
                    "error": str(exc),
                }
            )
    return results


def evaluate_fits_on_summaries(summaries: list[dict], fit_results: list[dict]) -> pd.DataFrame:
    """Evaluate each main-fitted curve against every dataset summary."""
    rows = []
    for fit_result in fit_results:
        for item in summaries:
            summary = item["summary"]
            x = summary[POSITION_COL].to_numpy(dtype=float)
            y = summary["theta_mean"].to_numpy(dtype=float)
            if fit_result["success"]:
                y_pred = fit_result["func"](x, *fit_result["params"])
                r2 = r_squared(y, y_pred)
            else:
                r2 = np.nan
            rows.append(
                {
                    "dataset": item["key"],
                    "dataset_label": item["label"],
                    "model": fit_result["name"],
                    "R2_using_main_fit": r2,
                }
            )
    return pd.DataFrame(rows)


def fit_results_to_table(fit_results: list[dict], evaluation: pd.DataFrame | None = None) -> pd.DataFrame:
    """Flatten fit results and parameters for CSV output."""
    rows = []
    for result in fit_results:
        row = {
            "model": result["name"],
            "R2": result["r2"],
            "success": result["success"],
            "error": result["error"],
        }
        for name, value in zip(result["param_names"], result["params"]):
            row[name] = value
        rows.append(row)
    table = pd.DataFrame(rows)
    if evaluation is not None and not table.empty:
        for _, eval_row in evaluation.iterrows():
            col = f"R2_on_{eval_row['dataset']}"
            table.loc[table["model"] == eval_row["model"], col] = eval_row["R2_using_main_fit"]
    return table


def plot_normalized_temperature(
    summaries: list[dict],
    fit_results: list[dict],
    fit_evaluation: pd.DataFrame,
) -> plt.Figure:
    """Draw mean curves, SD bands, and fitted curves for the main dataset."""
    fig, ax = plt.subplots(figsize=(7.9, 4.9))

    all_x = []
    all_y = []
    all_upper = []
    terminal_annotations = []
    for item in summaries:
        summary = item["summary"]
        color = item["color"]
        label = item["label"]
        x = summary[POSITION_COL].to_numpy(dtype=float)
        y = summary["theta_mean"].to_numpy(dtype=float)
        yerr = summary["theta_std"].to_numpy(dtype=float)
        all_x.append(x)
        all_y.append(y - yerr)
        all_upper.append(y + yerr)

        ax.fill_between(
            x,
            y - yerr,
            y + yerr,
            color=color,
            alpha=0.13,
            linewidth=0,
            label="_nolegend_",
        )
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            fmt="o-",
            color=color,
            linewidth=2.0 if item["fit"] else 1.7,
            markersize=5,
            markerfacecolor="white",
            markeredgewidth=1.1,
            elinewidth=0.9,
            capsize=2.5,
            capthick=0.9,
            label=label,
        )
        terminal_annotations.append((item["key"], label, color, float(x[-1]), float(y[-1])))

    main_summary = next(item["summary"] for item in summaries if item["fit"])
    x = main_summary[POSITION_COL].to_numpy(dtype=float)
    x_fit = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 400)
    sigmoid_result = None
    for result in fit_results:
        if not result["success"] or result["name"] != "Sigmoid":
            continue
        sigmoid_result = result
        ax.plot(
            x_fit,
            result["func"](x_fit, *result["params"]),
            color=result["color"],
            linestyle=result["linestyle"],
            linewidth=1.9,
            label="_nolegend_",
        )

    if not fit_evaluation.empty and terminal_annotations:
        sigmoid_evaluation = fit_evaluation[fit_evaluation["model"] == "Sigmoid"].set_index("dataset")
        label_offsets = {
            "main": (8, 10),
            "forcemove": (8, -2),
            "xmove": (8, -14),
        }
        for dataset_key, label, color, x_end, y_end in terminal_annotations:
            if dataset_key not in sigmoid_evaluation.index:
                continue
            r2_value = sigmoid_evaluation.loc[dataset_key, "R2_using_main_fit"]
            r2_text = "n/a" if pd.isna(r2_value) else f"{r2_value:.3f}"
            ax.annotate(
                rf"$R^2$={r2_text}",
                xy=(x_end, y_end),
                xytext=label_offsets.get(dataset_key, (8, 0)),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=8.3,
                color=color,
            )

    ax.axhline(0.0, color="0.35", linewidth=0.8, linestyle=":")
    ax.axhline(1.0, color="0.35", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Liquid level")
    ax.set_ylabel("Normalized temperature")
    all_x_flat = np.concatenate(all_x)
    all_y_flat = np.concatenate(all_y)
    all_upper_flat = np.concatenate(all_upper)
    x_min = min(0.0, float(np.nanmin(all_x_flat)))
    x_max = float(np.nanmax(all_x_flat))
    ax.set_xlim(x_min, x_max + 0.18 * max(1.0, x_max - x_min))
    ax.set_ylim(
        min(-0.08, float(np.nanmin(all_y_flat)) - 0.04),
        max(1.12, float(np.nanmax(all_upper_flat)) + 0.08),
    )
    ax.legend(loc="upper left", frameon=False, handlelength=2.4)
    fig.tight_layout()
    return fig


def main() -> None:
    apply_measurement_style()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    summaries = []
    main_summary = None
    for dataset in DATASETS:
        print(f"Loading: {dataset['path']}")
        df = load_normalized_data(dataset["path"])
        summary = summarize_by_position(df, dataset["key"], dataset["label"])
        summaries.append({**dataset, "summary": summary})
        if dataset["fit"]:
            main_summary = summary

    if main_summary is None:
        raise RuntimeError("No dataset is configured for curve fitting.")

    fit_results = fit_curves(main_summary)
    fit_evaluation = evaluate_fits_on_summaries(summaries, fit_results)

    pd.concat([item["summary"] for item in summaries], ignore_index=True).to_csv(SUMMARY_PATH, index=False)
    fit_results_to_table(fit_results, fit_evaluation).to_csv(FIT_RESULTS_PATH, index=False)

    fig = plot_normalized_temperature(summaries, fit_results, fit_evaluation)
    figure_path = save_figure(fig, str(OUTPUT_ROOT), "normalized_temperature_vs_liquid_position")
    plt.close(fig)

    print(f"Saved figure: {figure_path}")
    print(f"Saved summary: {SUMMARY_PATH}")
    print(f"Saved fit results: {FIT_RESULTS_PATH}")
    for result in fit_results:
        status = f"R2={result['r2']:.4f}" if result["success"] else f"failed: {result['error']}"
        print(f"  {result['name']}: {status}")
    for _, row in fit_evaluation.iterrows():
        r2_value = row["R2_using_main_fit"]
        status = "n/a" if pd.isna(r2_value) else f"{r2_value:.4f}"
        print(f"  Main {row['model']} on {row['dataset_label']}: R2={status}")
    print("Liquid temperature source: CSV column if present; otherwise per-group max patch temperature.")


if __name__ == "__main__":
    main()
