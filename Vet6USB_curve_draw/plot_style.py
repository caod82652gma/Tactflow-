"""
Shared measurement-grade plot style for VET6USB calibration experiments.
Import at the top of each notebook:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from plot_style import apply_measurement_style, COLORS, add_subplot_label, plot_residuals, SAVE_DIR
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import font_manager
import numpy as np
import os
import glob
import re
from contextlib import contextmanager

# ── Output directories ────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
RESULT_ROOT = os.path.join(_HERE, "result_display")
CN_RESULT_ROOT = os.path.join(_HERE, "result_display_cn")

# Macro switch for extra Chinese PNG output. Keep False for the original
# publication-style English outputs only. It can also be enabled per run with:
#   $env:VET6USB_CN_FIGURES="1" False
ENABLE_CHINESE_FIGURES = True
_CN_ENV_VALUE = os.environ.get("VET6USB_CN_FIGURES", "").strip().lower()
if _CN_ENV_VALUE in {"1", "true", "yes", "on", "cn", "zh"}:
    ENABLE_CHINESE_FIGURES = True
SAVE_DIRS = {
    1: os.path.join(RESULT_ROOT, "A_calibration", "test1"),
    2: os.path.join(RESULT_ROOT, "A_calibration", "test2"),
    3: os.path.join(RESULT_ROOT, "A_calibration", "test3"),
    4: os.path.join(RESULT_ROOT, "A_calibration", "test4"),
    5: os.path.join(RESULT_ROOT, "A_calibration", "test5"),
    6: os.path.join(RESULT_ROOT, "A_calibration", "test6"),
}

_INSTALLED_FONT_NAMES = {font.name for font in font_manager.fontManager.ttflist}


def _installed_fonts(candidates, fallback):
    fonts = [name for name in candidates if name in _INSTALLED_FONT_NAMES]
    for name in fallback:
        if name not in fonts:
            fonts.append(name)
    return fonts


SERIF_FONTS = _installed_fonts(
    ("Times New Roman", "SimSun", "Noto Serif CJK SC", "Source Han Serif SC", "DejaVu Serif"),
    ("DejaVu Serif",),
)
CN_SANS_FONTS = _installed_fonts(
    (
        "Microsoft YaHei",
        "DengXian",
        "Microsoft JhengHei",
        "SimHei",
        "SimSun",
        "NSimSun",
        "KaiTi",
        "FangSong",
        "Arial Unicode MS",
        "Source Han Sans SC",
        "Noto Sans CJK SC",
        "DejaVu Sans",
    ),
    ("DejaVu Sans",),
)
CN_SERIF_FONTS = _installed_fonts(
    ("SimSun", "NSimSun", "Source Han Serif SC", "Noto Serif CJK SC", "DejaVu Serif"),
    ("DejaVu Serif",),
)
CN_TEXT_FONT = CN_SANS_FONTS[0]

# ── Color palette (Tableau muted / ColorBrewer Set1) ─────────────────────────
COLORS = {
    "blue":    "#4878CF",   # Tableau muted blue
    "red":     "#D65F5F",   # Tableau muted red
    "green":   "#6ACC65",   # Tableau muted green
    "orange":  "#E6A020",   # Tableau muted orange
    "purple":  "#B47CC7",   # Tableau muted purple
    "teal":    "#77BEDB",   # Tableau muted teal
    "gray":    "#8C8C8C",   # neutral gray
    "loading": "#4878CF",   # alias for hysteresis loading
    "unloading": "#D65F5F", # alias for hysteresis unloading
    "fit":     "#D65F5F",   # regression line
    "data":    "#4878CF",   # scatter / errorbar data
    "residual":"#E6A020",   # residual bars
    "noise":   "#4878CF",
    "signal":  "#D65F5F",
    "zero":    "#4878CF",
    "detect":  "#6ACC65",
}


def apply_measurement_style():
    """Apply global rcParams for measurement-grade figures."""
    mpl.rcParams.update({
        # Font
        "font.family":        ["serif"],
        "font.serif":         SERIF_FONTS,
        "font.sans-serif":    CN_SANS_FONTS,
        "font.size":          10,
        "axes.titlesize":     11,
        "axes.labelsize":     10,
        "legend.fontsize":    9,
        "xtick.labelsize":    9,
        "ytick.labelsize":    9,
        # Spines
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.linewidth":     0.8,
        # Grid
        "axes.grid":          True,
        "grid.linestyle":     ":",
        "grid.alpha":         0.3,
        "grid.linewidth":     0.5,
        # Lines / markers
        "lines.linewidth":    1.5,
        "lines.markersize":   5,
        # Figure / save
        "figure.dpi":         100,
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
        "axes.unicode_minus":  False,
    })


CN_TEXT_RULES = {
    # General labels and titles
    "Time (s)": "时间 (s)",
    "Time t [s]": "时间 t [s]",
    "Time since contact [s]": "接触后时间 [s]",
    "Time from rise onset (s)": "上升起点后时间 (s)",
    "Frequency (Hz)": "频率 (Hz)",
    "Force (N)": "力 (N)",
    "Force (mN)": "力 (mN)",
    "Trial #": "试次编号",
    "Cycle number": "循环编号",
    "Column": "列",
    "Row": "行",
    "Capacity [%]": "容量 [%]",
    "Distance group": "距离组",
    "Reference channel": "参考通道",
    "Compared channel": "比较通道",
    "Probability Density": "概率密度",
    "Normalized amplitude": "归一化幅值",
    "Normalized response": "归一化响应",
    "Normalized response T\u0302": "归一化响应 T\u0302",
    "Signed normalized response": "带符号归一化响应",
    "Layer count": "层数",
    "Layer 0": "0层",
    "Layer 1": "1层",
    "Layer 3": "3层",
    "Fast time constant": "快时间常数",
    "Slow time constant": "慢时间常数",
    "Liquid level": "液位",
    "Normalized temperature": "归一化温度",
    "Baseline condition": "基准工况",
    "Changed contact force moment": "改变接触力矩",
    "Changed contact position": "改变接触位置",
    "Baseline-corrected ADC (LSB)": "基线校正 ADC (LSB)",
    "ADC - Mean (LSB)": "ADC - 均值 (LSB)",
    "ADC Output (LSB)": "ADC 输出 (LSB)",
    "Array envelope (LSB)": "阵列包络 (LSB)",
    "Frame diff (LSB)": "帧差 (LSB)",
    "Zero Point (LSB)": "零点 (LSB)",
    "Residual (LSB)": "残差 (LSB)",
    "Residual (%FS)": "残差 (%FS)",
    "Residual (degC)": "残差 (degC)",
    "Measured - model [degC]": "实测 - 模型 [degC]",
    "Temperature [\u00b0C]": "温度 [\u00b0C]",
    "Temperature T(t) [\u00b0C]": "温度 T(t) [\u00b0C]",
    "Target liquid temperature [\u00b0C]": "目标液体温度 [\u00b0C]",
    "Measured steady-state temperature [\u00b0C]": "实测稳态温度 [\u00b0C]",
    "T_std (degC)": "T_std (degC)",
    "Mean Delta t (ns)": "平均 Delta t (ns)",
    "Delta t (ns)": "Delta t (ns)",
    "Delta t vs AD1 (ns)": "相对 AD1 的 Delta t (ns)",
    "Repeatability (%FS)": "重复性 (%FS)",
    "Max hysteresis (%FS)": "最大迟滞 (%FS)",
    "Max hysteresis =": "最大迟滞 =",
    "Avg hysteresis =": "平均迟滞 =",
    "Frequency centroid (Hz)": "频率质心 (Hz)",
    "Nominal speed (m/s)": "标称速度 (m/s)",
    "Active taxels (%)": "激活 taxels (%)",
    "tau_fast [s]": "tau_fast [s]",
    "tau_slow [s]": "tau_slow [s]",
    "tau_thermal [s]": "tau_thermal [s]",
    "|Delta T_steady| [\u00b0C]": "|Delta T_steady| [\u00b0C]",
    # Figure titles
    "Repeatability Test \u2014 10 Cycles": "重复性测试 \u2014 10 次循环",
    "Linearity & Sensitivity Calibration": "线性度与灵敏度校准",
    "Detection Limit Analysis": "检测限分析",
    "Hysteresis Across 8 Loading-Unloading Cycles": "8 次加载-卸载循环迟滞",
    "Contact establishment and short-hold response": "接触建立与短时保持响应",
    "Long-hold tail response": "长时保持尾部响应",
    "Long-hold response after 10 min": "10 min 后长时保持响应",
    "COP trajectory": "COP 轨迹",
    "Detected slip window": "检测到的滑移窗口",
    "Frequency centroid vs nominal speed": "频率质心与标称速度关系",
    "Selected-trial velocity and spectral summary": "选定试次速度与频谱汇总",
    "Max abs response": "最大绝对响应",
    "Contact mask": "接触掩膜",
    "Contact-mask frequency by force distance": "不同力距离下接触掩膜频率",
    "Contact-mask coverage distribution": "接触掩膜覆盖率分布",
    "Temperature Time Curves (Different Liquid Temperatures)": "温度-时间曲线（不同液体温度）",
    "Difference Model Residuals": "差值模型残差",
    "Steady-State Temperature Summary": "稳态温度汇总",
    "Normalized Response \u2014 Double-Exponential Fit": "归一化响应 \u2014 双指数拟合",
    "Normalized Response \u2014 Single-Exponential Fit": "归一化响应 \u2014 单指数拟合",
    "Signed Normalized Response": "带符号归一化响应",
    "P0/P1/P2 Signed Normalized Response Overlay": "P0/P1/P2 带符号归一化响应叠加",
    "Normalized Fit Time Constants by k": "归一化拟合时间常数随 k 变化",
    "First-order RC fit on temperature response (75% water)": "温度响应的一阶 RC 拟合（75% 水量）",
    "Time constant vs capacity": "时间常数与容量关系",
    "Steady-state offset vs capacity": "稳态偏移与容量关系",
    "Cross-Cycle Mean EMF": "跨循环平均 EMF",
    "8chip Temperature Calibration": "8chip 温度校准",
    # Legend and annotation words
    "Loading": "加载",
    "Unloading": "卸载",
    "loading": "加载",
    "unloading": "卸载",
    "Measured mean": "实测均值",
    "Measured": "实测",
    "Residual mean": "残差均值",
    "Gaussian fit": "Gaussian 拟合",
    "fit": "拟合",
    "Fit": "拟合",
    "mean": "均值",
    "Mean": "均值",
    "all rising curves": "全部上升曲线",
    "threshold": "阈值",
    "direction": "方向",
    "slip window": "滑移窗口",
    "Cold": "冷",
    "Room": "室温",
    "Hot": "热",
    "cooling": "冷却",
    "heating": "加热",
}

CN_REPLACE_RULES = (
    (r"\bFig\.", "图"),
    (r"\bFigure\b", "图"),
    (r"\bMain\b", "主图"),
    (r"\bResidual\b", "残差"),
    (r"\bResiduals\b", "残差"),
    (r"\bLoading-Unloading\b", "加载-卸载"),
    (r"\bLoading\b", "加载"),
    (r"\bUnloading\b", "卸载"),
    (r"\bload\b", "加载"),
    (r"\bunload\b", "卸载"),
    (r"\bTime Curves\b", "时间曲线"),
    (r"\bTemperature\b", "温度"),
    (r"\bDifferent Liquid Temperatures\b", "不同液体温度"),
    (r"\bSteady-State\b", "稳态"),
    (r"\bSummary\b", "汇总"),
    (r"\bCalibration\b", "校准"),
    (r"\bFrequency\b", "频率"),
    (r"\bcentroid\b", "质心"),
    (r"\bnominal speed\b", "标称速度"),
    (r"\bSelected-trial\b", "选定试次"),
    (r"\bvelocity\b", "速度"),
    (r"\bspectral\b", "频谱"),
    (r"\bContact establishment\b", "接触建立"),
    (r"\bshort-hold response\b", "短时保持响应"),
    (r"\bLong-hold response\b", "长时保持响应"),
    (r"\bLong-hold tail response\b", "长时保持尾部响应"),
    (r"\bDetected slip window\b", "检测到的滑移窗口"),
    (r"\bNo active COP\b", "无有效 COP"),
    (r"\bContact mask\b", "接触掩膜"),
    (r"\bContact-mask\b", "接触掩膜"),
    (r"\bcoverage distribution\b", "覆盖率分布"),
    (r"\bfrequency by force distance\b", "随力距离变化的频率"),
    (r"\ball rising curves\b", "全部上升曲线"),
    (r"\bRepresentative trial\b", "代表试次"),
    (r"\bMeasured\b", "实测"),
    (r"\bmodel\b", "模型"),
    (r"\bTarget liquid\b", "目标液体"),
    (r"\bresponse\b", "响应"),
    (r"\bNormalized\b", "归一化"),
    (r"\bSigned\b", "带符号"),
    (r"\bDouble-Exponential Fit\b", "双指数拟合"),
    (r"\bSingle-Exponential Fit\b", "单指数拟合"),
    (r"\bHeating -> \+1, cooling -> -1\b", "加热 -> +1，冷却 -> -1"),
    (r"\bTime constant\b", "时间常数"),
    (r"\bcapacity\b", "容量"),
    (r"\bCapacity\b", "容量"),
    (r"\bSteady-state offset\b", "稳态偏移"),
    (r"\bLiquid-sensor crossover\b", "液体-传感器交界"),
    (r"\broom-temp signal degraded\b", "室温信号减弱"),
    (r"\bgeometric coupling\b", "几何耦合"),
)

CN_SKIP_PATH_PARTS = (
    "diagnostic",
    "all_trials",
    "sliding trajectory",
    "_all_trajectory",
    "_cop_trajectory",
    "contact_masks",
    "process",
)


def chinese_output_enabled():
    """Return whether extra Chinese PNG output is enabled."""
    return bool(ENABLE_CHINESE_FIGURES)


def _has_cjk(text):
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def cn_label(text):
    """Translate chart text while preserving technical abbreviations and units."""
    if text is None:
        return text
    original = str(text)
    if not original or _has_cjk(original):
        return original

    translated = CN_TEXT_RULES.get(original, original)
    for english, chinese in sorted(CN_TEXT_RULES.items(), key=lambda item: len(item[0]), reverse=True):
        translated = translated.replace(english, chinese)
    for pattern, replacement in CN_REPLACE_RULES:
        translated = re.sub(pattern, replacement, translated)

    translated = translated.replace("+/-", "\u00b1")
    translated = translated.replace(" +/- ", " \u00b1 ")
    translated = translated.replace("degC", "\u00b0C")
    translated = translated.replace("Â°C", "\u00b0C")
    translated = translated.replace("Î”", "Delta ")
    translated = translated.replace("Ï„", "tau")
    translated = translated.replace("Î±", "alpha")
    translated = translated.replace("Î²", "beta")
    translated = translated.replace("â€”", "\u2014")
    translated = translated.replace("â†’", "->")
    return translated


def _iter_figure_texts(fig):
    for text in fig.findobj(match=mpl.text.Text):
        yield text
    for ax in fig.axes:
        legend = ax.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                yield text
            title = legend.get_title()
            if title is not None:
                yield title


@contextmanager
def chinese_figure_context(fig):
    """Temporarily localize visible text on a Matplotlib figure."""
    originals = []
    font_originals = []
    seen_artists = set()
    original_font_family = mpl.rcParams.get("font.family")
    original_sans = mpl.rcParams.get("font.sans-serif")
    original_serif = mpl.rcParams.get("font.serif")
    cn_font_family = CN_SANS_FONTS
    mpl.rcParams["font.family"] = ["sans-serif"]
    mpl.rcParams["font.sans-serif"] = cn_font_family
    mpl.rcParams["font.serif"] = CN_SERIF_FONTS
    try:
        for artist in _iter_figure_texts(fig):
            artist_id = id(artist)
            if artist_id not in seen_artists:
                seen_artists.add(artist_id)
                font_originals.append((artist, artist.get_fontfamily()))
                artist.set_fontfamily(CN_TEXT_FONT)
            text = artist.get_text()
            localized = cn_label(text)
            if localized != text:
                originals.append((artist, text))
                artist.set_text(localized)
        yield
    finally:
        for artist, text in originals:
            artist.set_text(text)
        for artist, fontfamily in font_originals:
            artist.set_fontfamily(fontfamily)
        mpl.rcParams["font.family"] = original_font_family
        mpl.rcParams["font.sans-serif"] = original_sans
        mpl.rcParams["font.serif"] = original_serif


def map_to_chinese_png_path(path):
    """Map a result_display PNG path to result_display_cn, preserving subfolders."""
    path = os.path.abspath(os.fspath(path))
    result_root = os.path.abspath(RESULT_ROOT)
    try:
        rel_path = os.path.relpath(path, result_root)
        if rel_path.startswith(".."):
            raise ValueError
    except ValueError:
        rel_path = os.path.basename(path)
    return os.path.join(CN_RESULT_ROOT, rel_path)


def save_chinese_png(fig, original_png_path):
    """Save one localized Chinese PNG if the macro switch is enabled."""
    if not chinese_output_enabled():
        return None
    path_text = os.fspath(original_png_path).replace("\\", "/").lower()
    if any(part in path_text for part in CN_SKIP_PATH_PARTS):
        return None
    cn_path = map_to_chinese_png_path(original_png_path)
    os.makedirs(os.path.dirname(cn_path), exist_ok=True)
    with chinese_figure_context(fig):
        fig.savefig(cn_path, format="png", dpi=mpl.rcParams.get("savefig.dpi", 300), bbox_inches="tight")
    return cn_path


def add_subplot_label(ax, label, x=0.02, y=0.96):
    """Add a subplot label like (a), (b) in the upper-left corner."""
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="top", ha="left")


def plot_residuals(ax, x, y_true, y_pred, full_scale=None,
                   xlabel="Force (N)", color=None):
    """
    Bar-plot residuals on ax.
    If full_scale is given, adds a secondary y-axis in %FS.
    """
    color = color or COLORS["residual"]
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    ax.bar(x, residuals, color=color, alpha=0.75, edgecolor="none", width=0.6)
    ax.axhline(0, color=COLORS["gray"], linewidth=0.8, linestyle="--")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Residual (LSB)")

    if full_scale is not None and full_scale != 0:
        ax2 = ax.twinx()
        ax2.set_ylabel("Residual (%FS)")
        ax2.set_ylim(
            ax.get_ylim()[0] / abs(full_scale) * 100,
            ax.get_ylim()[1] / abs(full_scale) * 100,
        )
        ax2.spines["top"].set_visible(False)
    return ax


def save_figure(fig, save_dir, stem, formats=("png", "pdf")):
    """Save figure to save_dir in both PNG and PDF."""
    png_dir = os.path.join(save_dir, "png")
    pdf_dir = os.path.join(save_dir, "pdf")
    os.makedirs(png_dir, exist_ok=True)
    os.makedirs(pdf_dir, exist_ok=True)
    for fmt in formats:
        if fmt == "png":
            path = os.path.join(png_dir, f"{stem}.{fmt}")
        elif fmt == "pdf":
            path = os.path.join(pdf_dir, f"{stem}.{fmt}")
        else:
            path = os.path.join(save_dir, f"{stem}.{fmt}")
        fig.savefig(path)
    save_chinese_png(fig, os.path.join(png_dir, f"{stem}.png"))
    return os.path.join(png_dir, f"{stem}.png")


def clear_experiment_outputs(data_dir, save_dir=None, prefixes=(), data_suffixes=(".csv", ".txt")):
    """
    Remove previous generated outputs for one experiment prefix.

    Only files whose basename starts with one of the given prefixes are removed.
    Figure files are cleared from save_dir/png and save_dir/pdf; table/text files
    are cleared from data_dir. Raw workspace data is never touched by this helper.
    """
    if isinstance(prefixes, str):
        prefixes = (prefixes,)
    prefixes = tuple(prefixes)
    if not prefixes:
        return []

    removed = []

    def remove_matching_files(folder, suffixes):
        if not folder or not os.path.isdir(folder):
            return
        for prefix in prefixes:
            for suffix in suffixes:
                pattern = os.path.join(folder, f"{prefix}*{suffix}")
                for path in glob.glob(pattern):
                    if os.path.isfile(path):
                        os.remove(path)
                        removed.append(path)

    if isinstance(data_dir, (str, bytes, os.PathLike)):
        data_dirs = (data_dir,)
    else:
        data_dirs = tuple(data_dir)
    for folder in data_dirs:
        remove_matching_files(os.fspath(folder), data_suffixes)

    if save_dir is not None:
        save_dir = os.fspath(save_dir)
        remove_matching_files(os.path.join(save_dir, "png"), (".png",))
        remove_matching_files(os.path.join(save_dir, "pdf"), (".pdf",))
        if chinese_output_enabled():
            save_dir_abs = os.path.abspath(save_dir)
            result_root_abs = os.path.abspath(RESULT_ROOT)
            rel_save_dir = os.path.relpath(save_dir_abs, result_root_abs)
            if not rel_save_dir.startswith(".."):
                cn_save_dir = os.path.join(CN_RESULT_ROOT, rel_save_dir)
                remove_matching_files(os.path.join(cn_save_dir, "png"), (".png",))

    if removed:
        print(f"Cleared {len(removed)} previous output files for prefix(es): {', '.join(prefixes)}")
    return removed
