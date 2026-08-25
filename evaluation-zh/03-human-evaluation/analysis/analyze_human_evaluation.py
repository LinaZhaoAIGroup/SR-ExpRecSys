#!/usr/bin/env python3
"""Analyze repeated human ratings for RAG versus No-RAG answers.

Expected design
---------------
* Sheet1, Sheet2, Sheet3: three repeated system runs.
* Answer 1: RAG; Answer 2: No-RAG.
* Three experts score correctness, usefulness, and safety on a 1-10 scale.

The primary inferential unit is the question (n=24). For each question and
dimension, the script first averages the 3 runs x 3 experts, then performs a
paired RAG versus No-RAG comparison. This avoids treating repeated ratings of
the same question as independent observations.

The script creates one manuscript figure, three supplementary figures,
analysis tables, processed long-form data, and draft figure/method text.
It never modifies the source workbook.

Dependencies: Python 3.10+, pandas, openpyxl, numpy, scipy, matplotlib,
seaborn, and Pillow. Run with ``python analyze_human_evaluation.py``; use
``python analyze_human_evaluation.py --help`` for optional paths/settings.
"""

from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


SYSTEM_ORDER = ["RAG", "No-RAG"]
DIMENSION_ORDER = ["Correctness", "Usefulness", "Safety"]
LEVEL_ORDER = ["L1", "L2", "L3", "L4"]
EXPERT_ORDER = [1, 2, 3]

SYSTEM_COLORS = {
    "RAG": "#0072B2",
    "No-RAG": "#D55E00",
}

ANSWER_TO_SYSTEM = {"1": "RAG", "2": "No-RAG"}
CHINESE_TO_DIMENSION = {
    "正确性": "Correctness",
    "有用性": "Usefulness",
    "安全性": "Safety",
}
SCORE_COLUMN_PATTERN = re.compile(
    r"回答(?P<answer>[12])-(?P<dimension>正确性|有用性|安全性)-专家(?P<expert>[123])"
)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Analyze paired human ratings for RAG versus No-RAG."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=script_dir / "Evaluation_Human .xlsx",
        help="Input Excel workbook (default: workbook beside this script).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=script_dir / "human_evaluation_results",
        help="Output directory.",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=20_000,
        help="Number of stratified bootstrap replicates (default: 20000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260812,
        help="Random seed used for bootstrap confidence intervals.",
    )
    return parser.parse_args()


def configure_plotting() -> None:
    sns.set_theme(style="ticks", context="paper")
    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 600,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 10.0,
            "axes.titlesize": 11.0,
            "axes.labelsize": 10.5,
            "axes.labelweight": "bold",
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.5,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def natural_scene_key(scene_id: str) -> tuple[int, int, str]:
    match = re.fullmatch(r"L(\d+)-(\d+)", str(scene_id).strip())
    if match:
        return int(match.group(1)), int(match.group(2)), str(scene_id)
    return 999, 999, str(scene_id)


def choose_run_sheets(excel_file: pd.ExcelFile) -> list[str]:
    lookup = {name.lower(): name for name in excel_file.sheet_names}
    preferred = [lookup.get(f"sheet{i}") for i in range(1, 4)]
    if all(preferred):
        return preferred
    if len(excel_file.sheet_names) < 3:
        raise ValueError("The workbook must contain at least three run sheets.")
    warnings.warn(
        "Sheet1-Sheet3 were not all found; using the first three sheets instead."
    )
    return excel_file.sheet_names[:3]


def load_long_data(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Input workbook not found: {input_path}")

    excel_file = pd.ExcelFile(input_path, engine="openpyxl")
    sheet_names = choose_run_sheets(excel_file)
    long_parts: list[pd.DataFrame] = []

    required_base_columns = ["轮次", "序号", "场景编号", "用户问题"]

    for fallback_run, sheet_name in enumerate(sheet_names, start=1):
        frame = pd.read_excel(
            excel_file,
            sheet_name=sheet_name,
            engine="openpyxl",
        )
        frame.columns = [str(column).strip() for column in frame.columns]

        missing_base = [c for c in required_base_columns if c not in frame.columns]
        if missing_base:
            raise ValueError(f"{sheet_name} is missing columns: {missing_base}")

        score_columns = []
        for column in frame.columns:
            match = SCORE_COLUMN_PATTERN.fullmatch(column)
            if match:
                score_columns.append((column, match.groupdict()))

        if len(score_columns) != 18:
            raise ValueError(
                f"{sheet_name} contains {len(score_columns)} recognized score "
                "columns; expected 18."
            )

        base = frame[required_base_columns].copy()
        base = base.rename(
            columns={
                "轮次": "run",
                "序号": "item_number",
                "场景编号": "scene_id",
                "用户问题": "question",
            }
        )
        base["run"] = pd.to_numeric(base["run"], errors="coerce")
        if base["run"].isna().all():
            base["run"] = fallback_run
        if base["run"].nunique(dropna=True) != 1:
            raise ValueError(f"{sheet_name} should contain one run number only.")
        base["run"] = base["run"].fillna(fallback_run).astype(int)
        base["scene_id"] = base["scene_id"].astype(str).str.strip()
        base["level"] = base["scene_id"].str.extract(r"^(L\d+)", expand=False)

        for score_column, metadata in score_columns:
            part = base.copy()
            part["system"] = ANSWER_TO_SYSTEM[metadata["answer"]]
            part["dimension"] = CHINESE_TO_DIMENSION[metadata["dimension"]]
            part["expert"] = int(metadata["expert"])
            part["score"] = pd.to_numeric(frame[score_column], errors="coerce")
            long_parts.append(part)

    long_data = pd.concat(long_parts, ignore_index=True)
    long_data["system"] = pd.Categorical(
        long_data["system"], categories=SYSTEM_ORDER, ordered=True
    )
    long_data["dimension"] = pd.Categorical(
        long_data["dimension"], categories=DIMENSION_ORDER, ordered=True
    )
    long_data["level"] = pd.Categorical(
        long_data["level"], categories=LEVEL_ORDER, ordered=True
    )

    duplicated = long_data.duplicated(
        ["run", "scene_id", "system", "dimension", "expert"], keep=False
    )
    if duplicated.any():
        examples = long_data.loc[
            duplicated,
            ["run", "scene_id", "system", "dimension", "expert"],
        ].head()
        raise ValueError(f"Duplicate rating cells detected:\n{examples}")

    observed_scores = long_data["score"].dropna()
    if observed_scores.empty:
        raise ValueError("No numeric scores were found in the workbook.")
    if not observed_scores.between(1, 10).all():
        bad_values = sorted(observed_scores.loc[~observed_scores.between(1, 10)].unique())
        raise ValueError(f"Scores outside the expected 1-10 range: {bad_values}")

    if long_data["score"].isna().any():
        warnings.warn(
            f"There are {long_data['score'].isna().sum()} missing/non-numeric scores. "
            "Available-case means will be used."
        )

    return long_data.sort_values(
        ["run", "item_number", "dimension", "system", "expert"]
    ).reset_index(drop=True)


def build_question_level(long_data: pd.DataFrame) -> pd.DataFrame:
    question_level = (
        long_data.groupby(
            ["scene_id", "level", "system", "dimension"],
            observed=True,
            as_index=False,
        )
        .agg(score=("score", "mean"), n_ratings=("score", "count"))
    )
    incomplete = question_level.loc[question_level["n_ratings"] != 9]
    if not incomplete.empty:
        warnings.warn(
            "Some question-system-dimension cells do not contain all 9 ratings "
            "(3 runs x 3 experts). See question_level_scores.csv."
        )
    return question_level


def make_paired_question_table(question_level: pd.DataFrame) -> pd.DataFrame:
    paired = question_level.pivot_table(
        index=["scene_id", "level", "dimension"],
        columns="system",
        values="score",
        observed=True,
    ).reset_index()
    paired.columns.name = None
    paired = paired.dropna(subset=SYSTEM_ORDER).copy()
    paired["difference"] = paired["RAG"] - paired["No-RAG"]
    return paired.sort_values(
        "scene_id", key=lambda s: s.map(natural_scene_key)
    ).reset_index(drop=True)


def bootstrap_mean_ci(
    values: np.ndarray | pd.Series,
    strata: np.ndarray | pd.Series | None,
    rng: np.random.Generator,
    n_bootstrap: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    values_array = np.asarray(values, dtype=float)
    if strata is None:
        strata_array = np.repeat("all", len(values_array))
    else:
        strata_array = np.asarray(strata).astype(str)

    valid = np.isfinite(values_array)
    values_array = values_array[valid]
    strata_array = strata_array[valid]
    if values_array.size < 2:
        return np.nan, np.nan

    sampled_blocks = []
    for stratum in pd.unique(strata_array):
        stratum_values = values_array[strata_array == stratum]
        draw_indices = rng.integers(
            0,
            len(stratum_values),
            size=(n_bootstrap, len(stratum_values)),
        )
        sampled_blocks.append(stratum_values[draw_indices])

    bootstrap_means = np.concatenate(sampled_blocks, axis=1).mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(bootstrap_means, [alpha, 1.0 - alpha])
    return float(lower), float(upper)


def holm_adjust(p_values: np.ndarray | pd.Series) -> np.ndarray:
    p_values_array = np.asarray(p_values, dtype=float)
    adjusted = np.full(p_values_array.shape, np.nan, dtype=float)
    finite_positions = np.flatnonzero(np.isfinite(p_values_array))
    if finite_positions.size == 0:
        return adjusted

    finite_p = p_values_array[finite_positions]
    order = np.argsort(finite_p)
    sorted_p = finite_p[order]
    multipliers = np.arange(len(sorted_p), 0, -1)
    sorted_adjusted = np.maximum.accumulate(sorted_p * multipliers)
    sorted_adjusted = np.minimum(sorted_adjusted, 1.0)
    adjusted_positions = finite_positions[order]
    adjusted[adjusted_positions] = sorted_adjusted
    return adjusted


def paired_wilcoxon(differences: np.ndarray) -> tuple[float, float]:
    differences = np.asarray(differences, dtype=float)
    differences = differences[np.isfinite(differences)]
    if differences.size == 0:
        return np.nan, np.nan
    if np.allclose(differences, 0):
        return 0.0, 1.0
    try:
        result = stats.wilcoxon(
            differences,
            alternative="two-sided",
            zero_method="pratt",
            method="auto",
        )
    except TypeError:  # SciPy < 1.9 used the keyword ``mode``.
        result = stats.wilcoxon(
            differences,
            alternative="two-sided",
            zero_method="pratt",
            mode="auto",
        )
    return float(result.statistic), float(result.pvalue)


def primary_statistics(
    paired: pd.DataFrame,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> pd.DataFrame:
    rows = []
    for dimension in DIMENSION_ORDER:
        subset = paired.loc[paired["dimension"] == dimension].copy()
        differences = subset["difference"].to_numpy(dtype=float)
        ci_low, ci_high = bootstrap_mean_ci(
            differences,
            subset["level"],
            rng,
            n_bootstrap,
        )
        wilcoxon_w, p_value = paired_wilcoxon(differences)
        difference_sd = np.std(differences, ddof=1)
        cohen_dz = (
            float(np.mean(differences) / difference_sd)
            if difference_sd > 0
            else np.nan
        )
        tolerance = 1e-12
        rows.append(
            {
                "dimension": dimension,
                "n_questions": len(subset),
                "RAG_mean": subset["RAG"].mean(),
                "RAG_SD_across_questions": subset["RAG"].std(ddof=1),
                "No_RAG_mean": subset["No-RAG"].mean(),
                "No_RAG_SD_across_questions": subset["No-RAG"].std(ddof=1),
                "mean_paired_difference": differences.mean(),
                "CI95_low": ci_low,
                "CI95_high": ci_high,
                "median_paired_difference": np.median(differences),
                "cohen_dz": cohen_dz,
                "RAG_wins": int(np.sum(differences > tolerance)),
                "ties": int(np.sum(np.abs(differences) <= tolerance)),
                "No_RAG_wins": int(np.sum(differences < -tolerance)),
                "wilcoxon_W": wilcoxon_w,
                "p_raw": p_value,
            }
        )

    results = pd.DataFrame(rows)
    results["p_Holm"] = holm_adjust(results["p_raw"])
    return results


def level_statistics(
    paired: pd.DataFrame,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> pd.DataFrame:
    rows = []
    for level in LEVEL_ORDER:
        for dimension in DIMENSION_ORDER:
            subset = paired.loc[
                (paired["level"] == level)
                & (paired["dimension"] == dimension)
            ]
            differences = subset["difference"].to_numpy(dtype=float)
            ci_low, ci_high = bootstrap_mean_ci(
                differences,
                None,
                rng,
                n_bootstrap,
            )
            rows.append(
                {
                    "level": level,
                    "dimension": dimension,
                    "n_questions": len(subset),
                    "RAG_mean": subset["RAG"].mean(),
                    "No_RAG_mean": subset["No-RAG"].mean(),
                    "mean_paired_difference": differences.mean(),
                    "CI95_low": ci_low,
                    "CI95_high": ci_high,
                }
            )
    return pd.DataFrame(rows)


def run_statistics(
    long_data: pd.DataFrame,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_question = (
        long_data.groupby(
            ["run", "scene_id", "level", "system", "dimension"],
            observed=True,
            as_index=False,
        )
        .agg(score=("score", "mean"))
    )
    rows = []
    for dimension in DIMENSION_ORDER:
        for system in SYSTEM_ORDER:
            for run in sorted(run_question["run"].unique()):
                subset = run_question.loc[
                    (run_question["dimension"] == dimension)
                    & (run_question["system"] == system)
                    & (run_question["run"] == run)
                ]
                ci_low, ci_high = bootstrap_mean_ci(
                    subset["score"],
                    subset["level"],
                    rng,
                    n_bootstrap,
                )
                rows.append(
                    {
                        "run": int(run),
                        "dimension": dimension,
                        "system": system,
                        "n_questions": len(subset),
                        "mean_score": subset["score"].mean(),
                        "CI95_low": ci_low,
                        "CI95_high": ci_high,
                    }
                )
    return run_question, pd.DataFrame(rows)


def expert_statistics(
    long_data: pd.DataFrame,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    expert_question = (
        long_data.groupby(
            ["scene_id", "level", "expert", "system", "dimension"],
            observed=True,
            as_index=False,
        )
        .agg(score=("score", "mean"))
    )
    rows = []
    for dimension in DIMENSION_ORDER:
        for system in SYSTEM_ORDER:
            for expert in EXPERT_ORDER:
                subset = expert_question.loc[
                    (expert_question["dimension"] == dimension)
                    & (expert_question["system"] == system)
                    & (expert_question["expert"] == expert)
                ]
                ci_low, ci_high = bootstrap_mean_ci(
                    subset["score"],
                    subset["level"],
                    rng,
                    n_bootstrap,
                )
                rows.append(
                    {
                        "expert": expert,
                        "dimension": dimension,
                        "system": system,
                        "n_questions": len(subset),
                        "mean_score": subset["score"].mean(),
                        "CI95_low": ci_low,
                        "CI95_high": ci_high,
                    }
                )
    return expert_question, pd.DataFrame(rows)


def icc_two_way_random_absolute(matrix: np.ndarray) -> tuple[float, float]:
    """Return ICC(2,1) and ICC(2,k) for targets x raters."""
    ratings = np.asarray(matrix, dtype=float)
    if ratings.ndim != 2:
        raise ValueError("ICC input must be a targets x raters matrix.")
    n_targets, n_raters = ratings.shape
    if n_targets < 2 or n_raters < 2 or not np.isfinite(ratings).all():
        return np.nan, np.nan

    grand_mean = ratings.mean()
    target_means = ratings.mean(axis=1)
    rater_means = ratings.mean(axis=0)

    ss_targets = n_raters * np.sum((target_means - grand_mean) ** 2)
    ss_raters = n_targets * np.sum((rater_means - grand_mean) ** 2)
    residual = (
        ratings
        - target_means[:, None]
        - rater_means[None, :]
        + grand_mean
    )
    ss_error = np.sum(residual**2)

    ms_targets = ss_targets / (n_targets - 1)
    ms_raters = ss_raters / (n_raters - 1)
    ms_error = ss_error / ((n_targets - 1) * (n_raters - 1))

    denominator_single = (
        ms_targets
        + (n_raters - 1) * ms_error
        + n_raters * (ms_raters - ms_error) / n_targets
    )
    denominator_average = (
        ms_targets + (ms_raters - ms_error) / n_targets
    )
    icc_single = (
        (ms_targets - ms_error) / denominator_single
        if denominator_single != 0
        else np.nan
    )
    icc_average = (
        (ms_targets - ms_error) / denominator_average
        if denominator_average != 0
        else np.nan
    )
    return float(icc_single), float(icc_average)


def calculate_icc(long_data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dimension in DIMENSION_ORDER:
        for system in SYSTEM_ORDER:
            subset = long_data.loc[
                (long_data["dimension"] == dimension)
                & (long_data["system"] == system)
            ]
            matrix = subset.pivot_table(
                index=["scene_id", "run"],
                columns="expert",
                values="score",
                aggfunc="mean",
            ).reindex(columns=EXPERT_ORDER)
            complete_matrix = matrix.dropna(axis=0, how="any")
            if len(complete_matrix) < len(matrix):
                warnings.warn(
                    f"ICC for {system}, {dimension}: dropped "
                    f"{len(matrix) - len(complete_matrix)} incomplete targets."
                )
            icc_single, icc_average = icc_two_way_random_absolute(
                complete_matrix.to_numpy()
            )
            rows.append(
                {
                    "system": system,
                    "dimension": dimension,
                    "n_targets_question_by_run": len(complete_matrix),
                    "n_experts": complete_matrix.shape[1],
                    "ICC_2_1_single_expert": icc_single,
                    "ICC_2_k_mean_of_3_experts": icc_average,
                }
            )
    return pd.DataFrame(rows)


def add_panel_label(ax: mpl.axes.Axes, label: str, x: float = -0.16) -> None:
    ax.text(
        x,
        1.08,
        f"({label.lower()})",
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def format_p_value(p_value: float) -> str:
    if not np.isfinite(p_value):
        return "NA"
    if p_value < 0.001:
        return "<0.001"
    return f"{p_value:.3f}"


def bold_tick_labels(figure: mpl.figure.Figure) -> None:
    for axis in figure.axes:
        plt.setp(axis.get_xticklabels(), fontweight="bold")
        plt.setp(axis.get_yticklabels(), fontweight="bold")


def save_figure(
    figure: mpl.figure.Figure,
    figure_dir: Path,
    stem: str,
    save_tiff: bool = False,
) -> None:
    bold_tick_labels(figure)
    figure.savefig(
        figure_dir / f"{stem}.pdf",
        bbox_inches="tight",
        pad_inches=0.04,
    )
    figure.savefig(
        figure_dir / f"{stem}.png",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.04,
    )
    if save_tiff:
        figure.savefig(
            figure_dir / f"{stem}.tiff",
            dpi=600,
            bbox_inches="tight",
            pad_inches=0.04,
            pil_kwargs={"compression": "tiff_lzw"},
        )
    plt.close(figure)


def make_main_figure(
    paired: pd.DataFrame,
    primary: pd.DataFrame,
    level_results: pd.DataFrame,
    rng: np.random.Generator,
    n_bootstrap: int,
    figure_dir: Path,
) -> None:
    figure = plt.figure(figsize=(7.15, 6.4), constrained_layout=True)
    outer_grid = figure.add_gridspec(
        nrows=2,
        ncols=2,
        height_ratios=[1.35, 1.0],
        width_ratios=[1.08, 0.92],
    )
    paired_grid = outer_grid[0, :].subgridspec(1, 3, wspace=0.12)
    paired_axes = [figure.add_subplot(paired_grid[0, index]) for index in range(3)]

    jitter_rng = np.random.default_rng(1001)
    for dimension_index, (dimension, ax) in enumerate(
        zip(DIMENSION_ORDER, paired_axes)
    ):
        subset = paired.loc[paired["dimension"] == dimension].copy()
        subset = subset.sort_values(
            "scene_id", key=lambda s: s.map(natural_scene_key)
        )
        offsets = np.clip(jitter_rng.normal(0, 0.025, len(subset)), -0.06, 0.06)

        for offset, (_, row) in zip(offsets, subset.iterrows()):
            ax.plot(
                [0 + offset, 1 + offset],
                [row["RAG"], row["No-RAG"]],
                color="#A9ADB2",
                alpha=0.48,
                linewidth=0.65,
                zorder=1,
            )
            ax.scatter(
                0 + offset,
                row["RAG"],
                s=17,
                color=SYSTEM_COLORS["RAG"],
                edgecolor="white",
                linewidth=0.35,
                alpha=0.82,
                zorder=2,
            )
            ax.scatter(
                1 + offset,
                row["No-RAG"],
                s=17,
                color=SYSTEM_COLORS["No-RAG"],
                edgecolor="white",
                linewidth=0.35,
                alpha=0.82,
                zorder=2,
            )

        for x_position, system in enumerate(SYSTEM_ORDER):
            values = subset[system].to_numpy(dtype=float)
            ci_low, ci_high = bootstrap_mean_ci(
                values,
                subset["level"],
                rng,
                n_bootstrap,
            )
            mean_value = values.mean()
            ax.errorbar(
                x_position,
                mean_value,
                yerr=np.array([[mean_value - ci_low], [ci_high - mean_value]]),
                fmt="D",
                markersize=5.3,
                markerfacecolor=SYSTEM_COLORS[system],
                markeredgecolor="black",
                markeredgewidth=0.55,
                ecolor="black",
                elinewidth=1.1,
                capsize=3,
                capthick=1.0,
                zorder=4,
            )

        ax.set_title(dimension, fontweight="bold", pad=4)
        ax.set_xlim(-0.25, 1.25)
        ax.set_ylim(0.8, 10.25)
        ax.set_xticks([0, 1], SYSTEM_ORDER)
        ax.set_yticks([1, 4, 7, 10])
        ax.grid(axis="y", color="#E2E5E8", linewidth=0.6)
        ax.grid(axis="x", visible=False)
        sns.despine(ax=ax)
        ax.text(
            0.5,
            0.98,
            f"n = {len(subset)} questions",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=9,
            color="#555555",
        )
        if dimension_index == 0:
            ax.set_ylabel("Question-level mean score (1-10)")
            add_panel_label(ax, "A", x=-0.20)
        else:
            ax.set_ylabel("")
            ax.tick_params(labelleft=False)

    forest_ax = figure.add_subplot(outer_grid[1, 0])
    forest_y = np.arange(len(DIMENSION_ORDER))[::-1]
    forest_all_differences = paired["difference"].to_numpy(dtype=float)
    largest = max(
        0.75,
        float(np.nanmax(np.abs(forest_all_differences))),
        float(np.nanmax(np.abs(primary[["CI95_low", "CI95_high"]].to_numpy()))),
    )
    x_limit = np.ceil((largest + 0.25) * 2) / 2

    for y_position, dimension in zip(forest_y, DIMENSION_ORDER):
        differences = paired.loc[
            paired["dimension"] == dimension, "difference"
        ].to_numpy(dtype=float)
        y_jitter = np.linspace(-0.13, 0.13, len(differences))
        forest_ax.scatter(
            differences,
            y_position + y_jitter,
            s=12,
            color="#9DA3A8",
            alpha=0.58,
            linewidth=0,
            zorder=2,
        )
        result = primary.loc[primary["dimension"] == dimension].iloc[0]
        mean_difference = result["mean_paired_difference"]
        forest_ax.errorbar(
            mean_difference,
            y_position,
            xerr=np.array(
                [
                    [mean_difference - result["CI95_low"]],
                    [result["CI95_high"] - mean_difference],
                ]
            ),
            fmt="o",
            markersize=5.5,
            markerfacecolor=SYSTEM_COLORS["RAG"],
            markeredgecolor="black",
            markeredgewidth=0.55,
            ecolor="black",
            elinewidth=1.2,
            capsize=3,
            zorder=4,
        )
        annotation = (
            f"{mean_difference:.2f} [{result['CI95_low']:.2f}, "
            f"{result['CI95_high']:.2f}]\n"
            rf"$p_{{\mathrm{{Holm}}}}$ {format_p_value(result['p_Holm'])}"
        )
        forest_ax.text(
            1.02,
            y_position,
            annotation,
            transform=forest_ax.get_yaxis_transform(),
            ha="left",
            va="center",
            fontsize=9.0,
            clip_on=False,
        )

    forest_ax.axvline(0, color="#4D4D4D", linewidth=0.8, linestyle="--")
    forest_ax.set_xlim(-x_limit, x_limit)
    forest_ax.set_ylim(-0.55, len(DIMENSION_ORDER) - 0.45)
    forest_ax.set_yticks(forest_y, DIMENSION_ORDER)
    forest_ax.set_xlabel("Paired mean difference (RAG - No-RAG)")
    forest_ax.set_title("Overall paired effect", fontweight="bold", loc="left")
    forest_ax.grid(axis="x", color="#E2E5E8", linewidth=0.6)
    forest_ax.grid(axis="y", visible=False)
    sns.despine(ax=forest_ax, left=True)
    add_panel_label(forest_ax, "B", x=-0.16)

    heatmap_ax = figure.add_subplot(outer_grid[1, 1])
    heatmap_data = level_results.pivot(
        index="level",
        columns="dimension",
        values="mean_paired_difference",
    ).reindex(index=LEVEL_ORDER, columns=DIMENSION_ORDER)
    heatmap_values = heatmap_data.to_numpy(dtype=float)
    heat_limit = max(0.5, float(np.nanmax(np.abs(heatmap_values))))
    heat_norm = TwoSlopeNorm(vmin=-heat_limit, vcenter=0, vmax=heat_limit)
    sns.heatmap(
        heatmap_data,
        ax=heatmap_ax,
        cmap="RdBu",
        norm=heat_norm,
        linewidths=0.8,
        linecolor="white",
        square=False,
        annot=True,
        fmt=".2f",
        annot_kws={"fontsize": 9.0},
        cbar_kws={
            "label": "RAG - No-RAG",
            "shrink": 0.78,
            "pad": 0.03,
        },
    )
    heatmap_ax.set_title(
        "Effect consistency across question levels",
        fontweight="bold",
        loc="left",
    )
    heatmap_ax.set_xlabel("")
    heatmap_ax.set_ylabel("Question level (n = 6 each)")
    heatmap_ax.tick_params(axis="x", rotation=25)
    heatmap_ax.tick_params(axis="y", rotation=0)
    add_panel_label(heatmap_ax, "C", x=-0.18)

    save_figure(
        figure,
        figure_dir,
        "Figure_1_main_human_evaluation",
        save_tiff=True,
    )


def make_item_heatmap(paired: pd.DataFrame, figure_dir: Path) -> None:
    ordered_scenes = sorted(paired["scene_id"].unique(), key=natural_scene_key)
    heatmap_data = paired.pivot(
        index="scene_id",
        columns="dimension",
        values="difference",
    ).reindex(index=ordered_scenes, columns=DIMENSION_ORDER)
    values = heatmap_data.to_numpy(dtype=float)
    heat_limit = max(0.5, float(np.nanmax(np.abs(values))))
    norm = TwoSlopeNorm(vmin=-heat_limit, vcenter=0, vmax=heat_limit)

    figure, ax = plt.subplots(figsize=(5.6, 7.8), constrained_layout=True)
    sns.heatmap(
        heatmap_data,
        ax=ax,
        cmap="RdBu",
        norm=norm,
        linewidths=0.35,
        linecolor="white",
        annot=True,
        fmt=".1f",
        annot_kws={"fontsize": 8.0},
        cbar_kws={
            "label": "Question-level paired difference (RAG - No-RAG)",
            "shrink": 0.70,
            "pad": 0.03,
        },
    )
    ax.set_xlabel("")
    ax.set_ylabel("Question ID")
    ax.set_title(
        "Item-level effects after averaging three runs and three experts",
        loc="left",
        fontweight="bold",
    )
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)

    level_boundaries = []
    for index in range(1, len(ordered_scenes)):
        previous_level = ordered_scenes[index - 1].split("-")[0]
        current_level = ordered_scenes[index].split("-")[0]
        if previous_level != current_level:
            level_boundaries.append(index)
    for boundary in level_boundaries:
        ax.axhline(boundary, color="#333333", linewidth=1.3)

    save_figure(figure, figure_dir, "Figure_S1_item_level_effects")


def make_run_stability_figure(
    run_summary: pd.DataFrame,
    figure_dir: Path,
) -> None:
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(7.15, 2.55),
        sharey=True,
        constrained_layout=True,
    )
    x_offsets = {"RAG": -0.035, "No-RAG": 0.035}
    for dimension_index, (dimension, ax) in enumerate(
        zip(DIMENSION_ORDER, axes)
    ):
        for system in SYSTEM_ORDER:
            subset = run_summary.loc[
                (run_summary["dimension"] == dimension)
                & (run_summary["system"] == system)
            ].sort_values("run")
            x_values = subset["run"].to_numpy(dtype=float) + x_offsets[system]
            means = subset["mean_score"].to_numpy(dtype=float)
            asymmetric_error = np.vstack(
                [
                    means - subset["CI95_low"].to_numpy(dtype=float),
                    subset["CI95_high"].to_numpy(dtype=float) - means,
                ]
            )
            ax.errorbar(
                x_values,
                means,
                yerr=asymmetric_error,
                color=SYSTEM_COLORS[system],
                marker="o",
                markersize=4.2,
                linewidth=1.4,
                elinewidth=0.9,
                capsize=2.5,
                label=system,
            )
        ax.set_title(dimension, fontweight="bold")
        ax.set_xlabel("Repeated run")
        ax.set_xticks([1, 2, 3])
        ax.set_xlim(0.75, 3.25)
        ax.set_ylim(0.8, 10.25)
        ax.set_yticks([1, 4, 7, 10])
        ax.grid(axis="y", color="#E2E5E8", linewidth=0.6)
        ax.grid(axis="x", visible=False)
        sns.despine(ax=ax)
        if dimension_index == 0:
            ax.set_ylabel("Mean score across questions")
        else:
            ax.set_ylabel("")
    axes[-1].legend(frameon=False, loc="lower left")
    save_figure(figure, figure_dir, "Figure_S2_run_stability")


def make_expert_agreement_figure(
    expert_summary: pd.DataFrame,
    icc_results: pd.DataFrame,
    figure_dir: Path,
) -> None:
    figure = plt.figure(figsize=(7.15, 5.0), constrained_layout=True)
    grid = figure.add_gridspec(2, 3, height_ratios=[1.0, 0.88])
    expert_axes = [figure.add_subplot(grid[0, i]) for i in range(3)]

    x_offsets = {"RAG": -0.055, "No-RAG": 0.055}
    for dimension_index, (dimension, ax) in enumerate(
        zip(DIMENSION_ORDER, expert_axes)
    ):
        for system in SYSTEM_ORDER:
            subset = expert_summary.loc[
                (expert_summary["dimension"] == dimension)
                & (expert_summary["system"] == system)
            ].sort_values("expert")
            x_values = subset["expert"].to_numpy(dtype=float) + x_offsets[system]
            means = subset["mean_score"].to_numpy(dtype=float)
            asymmetric_error = np.vstack(
                [
                    means - subset["CI95_low"].to_numpy(dtype=float),
                    subset["CI95_high"].to_numpy(dtype=float) - means,
                ]
            )
            ax.errorbar(
                x_values,
                means,
                yerr=asymmetric_error,
                color=SYSTEM_COLORS[system],
                marker="o",
                markersize=4.2,
                linewidth=1.35,
                elinewidth=0.9,
                capsize=2.5,
                label=system,
            )
        ax.set_title(dimension, fontweight="bold")
        ax.set_xlabel("Expert")
        ax.set_xticks(EXPERT_ORDER)
        ax.set_xlim(0.75, 3.25)
        ax.set_ylim(0.8, 10.25)
        ax.set_yticks([1, 4, 7, 10])
        ax.grid(axis="y", color="#E2E5E8", linewidth=0.6)
        ax.grid(axis="x", visible=False)
        sns.despine(ax=ax)
        if dimension_index == 0:
            ax.set_ylabel("Mean score across questions and runs")
            add_panel_label(ax, "A", x=-0.20)
        else:
            ax.set_ylabel("")
    expert_axes[-1].legend(frameon=False, loc="lower left")

    icc_ax = figure.add_subplot(grid[1, :])
    icc_wide = icc_results.pivot(
        index="dimension",
        columns="system",
        values=["ICC_2_1_single_expert", "ICC_2_k_mean_of_3_experts"],
    ).reindex(index=DIMENSION_ORDER)
    desired_columns = pd.MultiIndex.from_tuples(
        [
            ("ICC_2_1_single_expert", "RAG"),
            ("ICC_2_k_mean_of_3_experts", "RAG"),
            ("ICC_2_1_single_expert", "No-RAG"),
            ("ICC_2_k_mean_of_3_experts", "No-RAG"),
        ]
    )
    icc_wide = icc_wide.reindex(columns=desired_columns)
    icc_wide.columns = [
        "RAG\nICC(2,1)",
        "RAG\nICC(2,3)",
        "No-RAG\nICC(2,1)",
        "No-RAG\nICC(2,3)",
    ]
    sns.heatmap(
        icc_wide,
        ax=icc_ax,
        cmap="coolwarm_r",
        vmin=-0.2,
        vmax=1.0,
        center=0.4,
        annot=True,
        fmt=".2f",
        annot_kws={"fontsize": 9.5},
        linewidths=0.8,
        linecolor="white",
        cbar_kws={"label": "Intraclass correlation coefficient", "shrink": 0.8},
    )
    icc_ax.set_xlabel("")
    icc_ax.set_ylabel("")
    icc_ax.set_title(
        "Absolute-agreement ICC across experts (72 question-by-run targets)",
        loc="left",
        fontweight="bold",
    )
    icc_ax.tick_params(axis="x", rotation=0)
    icc_ax.tick_params(axis="y", rotation=0)
    add_panel_label(icc_ax, "B", x=-0.08)

    save_figure(figure, figure_dir, "Figure_S3_expert_agreement")


def write_caption_and_methods(
    output_path: Path,
    primary: pd.DataFrame,
    n_bootstrap: int,
    seed: int,
) -> None:
    stat_lines = []
    for _, row in primary.iterrows():
        stat_lines.append(
            f"- {row['dimension']}: RAG {row['RAG_mean']:.2f}, No-RAG "
            f"{row['No_RAG_mean']:.2f}; mean paired difference "
            f"{row['mean_paired_difference']:.2f} (95% CI "
            f"{row['CI95_low']:.2f} to {row['CI95_high']:.2f}), "
            f"Holm-adjusted p {format_p_value(row['p_Holm'])}, "
            f"Cohen's dz = {row['cohen_dz']:.2f}."
        )

    text = f"""DRAFT MAIN-FIGURE CAPTION

Figure 1. Human expert evaluation of RAG and No-RAG answers. (a) Paired
question-level scores for correctness, usefulness, and safety. Each small point
is the mean for one question after averaging three repeated runs and three
experts; gray lines connect scores for the same question. Diamonds and error
bars show the mean and stratified bootstrap 95% confidence interval. (b)
Question-level paired differences (RAG - No-RAG). Small gray points show the 24
individual question effects, and blue points with error bars show mean paired
differences and stratified bootstrap 95% confidence intervals. P values are
from two-sided paired Wilcoxon signed-rank tests with Holm correction across the
three dimensions. (c) Mean paired differences stratified by question level
(six questions per level). Positive values favor RAG.

DRAFT SUPPLEMENTARY-FIGURE CAPTIONS

Figure S1. Question-specific paired effects of RAG relative to No-RAG. Each
cell is the RAG - No-RAG difference after averaging three runs and three
experts. Positive values favor RAG.

Figure S2. Stability of human-evaluation results across the three repeated
system runs. Points show means across 24 questions and three experts; error
bars show stratified bootstrap 95% confidence intervals across questions.

Figure S3. Expert-specific scores and inter-rater agreement. (a) Mean scores by
expert after averaging the three repeated runs; error bars show stratified
bootstrap 95% confidence intervals across questions. (b) Two-way random-effects,
absolute-agreement intraclass correlation coefficients for a single expert
[ICC(2,1)] and the mean of three experts [ICC(2,3)]. ICCs use the 72
question-by-run combinations as targets and are descriptive because only three
experts were sampled.

DRAFT STATISTICAL METHODS

Human evaluation used 24 questions spanning four levels, three independent
system runs, three experts, two systems, and three rating dimensions on a
1-10 scale. The question was treated as the primary inferential unit. For each
question, system, and dimension, ratings were averaged across the three runs
and three experts, yielding one paired RAG and No-RAG value per question.
Effect sizes were expressed as the paired mean difference (RAG - No-RAG) with
95% percentile bootstrap confidence intervals. Bootstrap resampling was
stratified by question level to preserve the balanced L1-L4 composition and
used {n_bootstrap:,} replicates (random seed {seed}). Two-sided Wilcoxon
signed-rank tests assessed paired differences, and P values were Holm-adjusted
across the three prespecified rating dimensions. Cohen's dz was calculated as
the mean paired difference divided by the standard deviation of the paired
differences. Expert agreement was summarized using two-way random-effects,
absolute-agreement ICC(2,1) and ICC(2,3).

PRIMARY NUMERICAL RESULTS

{chr(10).join(stat_lines)}
"""
    output_path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    configure_plotting()

    if args.bootstrap < 1_000:
        warnings.warn(
            "Fewer than 1,000 bootstrap replicates can produce unstable "
            "confidence interval limits."
        )

    output_dir = args.output.resolve()
    figure_dir = output_dir / "figures"
    table_dir = output_dir / "tables"
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    long_data = load_long_data(args.input.resolve())
    question_level = build_question_level(long_data)
    paired = make_paired_question_table(question_level)

    observed_questions = paired["scene_id"].nunique()
    observed_runs = long_data["run"].nunique()
    observed_experts = long_data["expert"].nunique()
    if (observed_questions, observed_runs, observed_experts) != (24, 3, 3):
        warnings.warn(
            "Observed design differs from 24 questions x 3 runs x 3 experts: "
            f"{observed_questions} questions x {observed_runs} runs x "
            f"{observed_experts} experts. Figures will use observed data."
        )

    primary = primary_statistics(paired, rng, args.bootstrap)
    level_results = level_statistics(paired, rng, args.bootstrap)
    _, run_summary = run_statistics(
        long_data, rng, args.bootstrap
    )
    _, expert_summary = expert_statistics(
        long_data, rng, args.bootstrap
    )
    icc_results = calculate_icc(long_data)

    long_data.to_csv(
        table_dir / "processed_long_scores.csv",
        index=False,
        encoding="utf-8-sig",
    )
    question_level.to_csv(
        table_dir / "question_level_scores.csv",
        index=False,
        encoding="utf-8-sig",
    )
    paired.to_csv(
        table_dir / "question_level_paired_differences.csv",
        index=False,
        encoding="utf-8-sig",
    )
    primary.to_csv(
        table_dir / "Table_S1_primary_statistics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    level_results.to_csv(
        table_dir / "Table_S2_level_effects.csv",
        index=False,
        encoding="utf-8-sig",
    )
    run_summary.to_csv(
        table_dir / "Table_S3_run_stability.csv",
        index=False,
        encoding="utf-8-sig",
    )
    expert_summary.to_csv(
        table_dir / "Table_S4_expert_means.csv",
        index=False,
        encoding="utf-8-sig",
    )
    icc_results.to_csv(
        table_dir / "Table_S5_expert_ICC.csv",
        index=False,
        encoding="utf-8-sig",
    )

    make_main_figure(
        paired,
        primary,
        level_results,
        rng,
        args.bootstrap,
        figure_dir,
    )
    make_item_heatmap(paired, figure_dir)
    make_run_stability_figure(run_summary, figure_dir)
    make_expert_agreement_figure(expert_summary, icc_results, figure_dir)
    write_caption_and_methods(
        output_dir / "figure_captions_and_statistical_methods.txt",
        primary,
        args.bootstrap,
        args.seed,
    )

    print(f"Analysis complete. Results written to: {output_dir}")
    print("Main manuscript figure: Figure_1_main_human_evaluation.*")
    print("Supplementary figures: Figure_S1 through Figure_S3")


if __name__ == "__main__":
    main()
