#!/usr/bin/env python3
"""Create manuscript and supplementary figures from repeated GPT scores.

The parent directory must contain the three scoring-repeat workbooks:
    benchmark_llm_scores_1.xlsx
    benchmark_llm_scores_2.xlsx
    benchmark_llm_scores_3.xlsx

Each workbook contains Sheet1-Sheet3 (three response-generation repeats). The
script combines the resulting nine observations per question, system, and
dimension. It creates one concise main-text figure, detailed SI figures, and
CSV tables in the same directory as this script.

Statistical unit: question (n = 24). Wilcoxon signed-rank tests compare paired
question-level means after averaging the nine repeated observations. P values
for the three dimensions are adjusted with the Holm method.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch
from scipy import stats


SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR = SCRIPT_DIR.parent / "01-llm-scoring"
OUTPUT_DIR = SCRIPT_DIR

INPUT_FILE_RE = re.compile(r"^benchmark_llm_scores_([123])\.xlsx$")
EXPECTED_SCORING_REPEATS = (1, 2, 3)
EXPECTED_GENERATION_REPEATS = (1, 2, 3)
EXPECTED_SHEETS = tuple(f"Sheet{i}" for i in EXPECTED_GENERATION_REPEATS)
EXPECTED_QUESTIONS = tuple(range(1, 25))

DIMENSIONS = ("Correctness", "Usefulness", "Safety")
CONDITIONS = ("RAG", "No RAG")
SOURCE_PREFIX = {"RAG": "Answer 1", "No RAG": "Answer 2"}
COLORS = {"RAG": "#1F77B4", "No RAG": "#D62728"}

FIGURE_DPI = 600
FIGURE_FORMATS = ("png", "pdf", "svg")
CI_LEVEL = 0.95

REQUIRED_BASE_COLUMNS = {
    "Run",
    "Item Number",
    "Scenario ID",
    "Scoring Model",
}
REQUIRED_SCORE_COLUMNS = {
    f"{SOURCE_PREFIX[condition]}-{dimension}"
    for condition in CONDITIONS
    for dimension in DIMENSIONS
}


def set_scientific_style() -> None:
    """Use a compact publication style consistent across all figures."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 8.5,
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": FIGURE_DPI,
        }
    )


def discover_input_files() -> dict[int, Path]:
    """Find all three workbooks, accepting names with or without an underscore."""
    found: dict[int, Path] = {}
    for path in sorted(INPUT_DIR.glob("benchmark_llm_scores_*.xlsx")):
        if path.name.startswith("~$"):
            continue
        match = INPUT_FILE_RE.fullmatch(path.name)
        if match is None:
            continue
        scoring_repeat = int(match.group(1))
        if scoring_repeat in found:
            raise ValueError(
                f"Duplicate workbook for scoring repeat {scoring_repeat}: "
                f"{found[scoring_repeat].name} and {path.name}"
            )
        found[scoring_repeat] = path

    expected = set(EXPECTED_SCORING_REPEATS)
    missing = sorted(expected - set(found))
    if missing:
        raise FileNotFoundError(
            f"Missing scoring-repeat workbook(s) {missing} in {INPUT_DIR}. "
            "Expected benchmark_llm_scores_1.xlsx through "
            "benchmark_llm_scores_3.xlsx."
        )
    return {repeat: found[repeat] for repeat in EXPECTED_SCORING_REPEATS}


def _validate_sheet(
    frame: pd.DataFrame,
    path: Path,
    sheet_name: str,
    generation_repeat: int,
) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [str(column).strip() for column in frame.columns]

    required = REQUIRED_BASE_COLUMNS | REQUIRED_SCORE_COLUMNS
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path.name}/{sheet_name} is missing columns: {missing}")

    if len(frame) != len(EXPECTED_QUESTIONS):
        raise ValueError(
            f"{path.name}/{sheet_name} has {len(frame)} rows; expected 24."
        )

    question_number = pd.to_numeric(frame["Item Number"], errors="coerce")
    if question_number.isna().any() or not np.allclose(
        question_number, np.rint(question_number)
    ):
        raise ValueError(f"{path.name}/{sheet_name}: serial numbermust contain integers.")
    frame["Item Number"] = question_number.astype(int)
    if tuple(sorted(frame["Item Number"].tolist())) != EXPECTED_QUESTIONS:
        raise ValueError(
            f"{path.name}/{sheet_name}: serial numbermust be the integers 1 through 24."
        )

    if frame["Scenario ID"].isna().any() or frame["Scenario ID"].astype(str).duplicated().any():
        raise ValueError(
            f"{path.name}/{sheet_name}: scene numbermust be complete and unique."
        )

    round_values = pd.to_numeric(frame["Run"], errors="coerce").dropna().unique()
    if len(round_values) != 1 or int(round_values[0]) != generation_repeat:
        raise ValueError(
            f"{path.name}/{sheet_name}: roundsdoes not match {generation_repeat}."
        )

    model_values = frame["Scoring Model"].dropna().astype(str).str.strip().unique()
    if len(model_values) != 1:
        raise ValueError(
            f"{path.name}/{sheet_name}: expected one non-missing Scoring modelvalue."
        )

    for column in sorted(REQUIRED_SCORE_COLUMNS):
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any():
            bad_rows = (values.isna().to_numpy().nonzero()[0] + 2).tolist()
            raise ValueError(
                f"{path.name}/{sheet_name}/{column}: nonnumeric or missing scores "
                f"at Excel rows {bad_rows}."
            )
        if not values.between(1, 10, inclusive="both").all():
            raise ValueError(
                f"{path.name}/{sheet_name}/{column}: scores must be within 1-10."
            )
        if not np.allclose(values, np.rint(values)):
            raise ValueError(
                f"{path.name}/{sheet_name}/{column}: scores must be integers."
            )
        frame[column] = values.astype(float)

    return frame.sort_values("Item Number").reset_index(drop=True)


def load_scores(files: dict[int, Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read, validate, and reshape all workbooks to one tidy table."""
    records: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, object]] = []
    reference_scenarios: dict[int, str] | None = None
    reference_model: str | None = None

    for scoring_repeat, path in files.items():
        workbook = pd.ExcelFile(path, engine="openpyxl")
        missing_sheets = sorted(set(EXPECTED_SHEETS) - set(workbook.sheet_names))
        if missing_sheets:
            raise ValueError(f"{path.name} is missing sheets: {missing_sheets}")

        for generation_repeat, sheet_name in zip(
            EXPECTED_GENERATION_REPEATS, EXPECTED_SHEETS
        ):
            frame = pd.read_excel(workbook, sheet_name=sheet_name)
            frame = _validate_sheet(frame, path, sheet_name, generation_repeat)

            scenario_map = dict(
                zip(frame["Item Number"], frame["Scenario ID"].astype(str).str.strip())
            )
            if reference_scenarios is None:
                reference_scenarios = scenario_map
            elif scenario_map != reference_scenarios:
                raise ValueError(
                    f"{path.name}/{sheet_name}: Item Number-to-scene numbermapping differs "
                    "from the first sheet."
                )

            score_model = str(frame["Scoring Model"].iloc[0]).strip()
            if reference_model is None:
                reference_model = score_model
            elif score_model != reference_model:
                raise ValueError(
                    f"{path.name}/{sheet_name}: scoring model {score_model!r} "
                    f"differs from {reference_model!r}."
                )

            manifest_rows.append(
                {
                    "scoring_repeat": scoring_repeat,
                    "generation_repeat": generation_repeat,
                    "workbook": path.name,
                    "sheet": sheet_name,
                    "n_questions": len(frame),
                    "scoring_model": score_model,
                }
            )

            for condition in CONDITIONS:
                for dimension in DIMENSIONS:
                    score_column = f"{SOURCE_PREFIX[condition]}-{dimension}"
                    part = frame[["Item Number", "Scenario ID"]].copy()
                    part.columns = ["question_number", "scenario_id"]
                    part["scoring_repeat"] = scoring_repeat
                    part["generation_repeat"] = generation_repeat
                    part["workbook"] = path.name
                    part["sheet"] = sheet_name
                    part["scoring_model"] = score_model
                    part["condition"] = condition
                    part["dimension"] = dimension
                    part["score"] = frame[score_column].to_numpy(dtype=float)
                    records.append(part)

    scores = pd.concat(records, ignore_index=True)
    expected_rows = (
        len(EXPECTED_SCORING_REPEATS)
        * len(EXPECTED_GENERATION_REPEATS)
        * len(EXPECTED_QUESTIONS)
        * len(CONDITIONS)
        * len(DIMENSIONS)
    )
    if len(scores) != expected_rows:
        raise ValueError(f"Loaded {len(scores)} score rows; expected {expected_rows}.")

    unique_key = [
        "scoring_repeat",
        "generation_repeat",
        "question_number",
        "condition",
        "dimension",
    ]
    if scores.duplicated(unique_key).any():
        raise ValueError("Duplicate observations found after reshaping the workbooks.")

    scores["scenario_id"] = scores["scenario_id"].astype(str)
    scores["condition"] = pd.Categorical(
        scores["condition"], categories=CONDITIONS, ordered=True
    )
    scores["dimension"] = pd.Categorical(
        scores["dimension"], categories=DIMENSIONS, ordered=True
    )
    scores = scores.sort_values(unique_key).reset_index(drop=True)
    manifest = pd.DataFrame(manifest_rows).sort_values(
        ["scoring_repeat", "generation_repeat"]
    )
    return scores, manifest


def mean_ci(values: np.ndarray, confidence: float = CI_LEVEL) -> tuple[float, float, float]:
    """Return the sample mean and two-sided t confidence interval."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan, np.nan
    mean = float(np.mean(values))
    if values.size == 1:
        return mean, mean, mean
    sem = float(stats.sem(values))
    critical = float(stats.t.ppf((1.0 + confidence) / 2.0, values.size - 1))
    margin = critical * sem
    return mean, mean - margin, mean + margin


def build_question_summary(scores: pd.DataFrame) -> pd.DataFrame:
    """Summarize the nine raw scores for every question and condition."""
    group_columns = ["question_number", "scenario_id", "condition", "dimension"]
    summary = (
        scores.groupby(group_columns, observed=True)["score"]
        .agg(mean="mean", sd="std", n="size")
        .reset_index()
    )
    if not (summary["n"] == 9).all():
        raise ValueError("Every question/condition/dimension must have nine scores.")

    critical = stats.t.ppf(0.975, summary["n"] - 1)
    margin = critical * summary["sd"] / np.sqrt(summary["n"])
    summary["ci95_low"] = summary["mean"] - margin
    summary["ci95_high"] = summary["mean"] + margin
    return summary.sort_values(group_columns).reset_index(drop=True)


def build_paired_question_means(question_summary: pd.DataFrame) -> pd.DataFrame:
    """Create one paired RAG/No RAG observation per question and dimension."""
    paired = question_summary.pivot(
        index=["question_number", "scenario_id", "dimension"],
        columns="condition",
        values="mean",
    ).reset_index()
    paired.columns.name = None
    missing = [condition for condition in CONDITIONS if condition not in paired.columns]
    if missing:
        raise ValueError(f"Paired question table is missing conditions: {missing}")
    if paired[list(CONDITIONS)].isna().any().any():
        raise ValueError("Missing values found in the paired question table.")
    paired["difference_RAG_minus_NoRAG"] = paired["RAG"] - paired["No RAG"]
    return paired.sort_values(["dimension", "question_number"]).reset_index(drop=True)


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    """Holm family-wise error correction."""
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    ordered = p_values[order]
    multipliers = len(ordered) - np.arange(len(ordered))
    adjusted_ordered = np.minimum(1.0, np.maximum.accumulate(ordered * multipliers))
    adjusted = np.empty_like(adjusted_ordered)
    adjusted[order] = adjusted_ordered
    return adjusted


def rank_biserial(differences: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation; positive values favor RAG."""
    differences = np.asarray(differences, dtype=float)
    nonzero = differences[~np.isclose(differences, 0.0)]
    if nonzero.size == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(nonzero), method="average")
    positive = float(ranks[nonzero > 0].sum())
    negative = float(ranks[nonzero < 0].sum())
    return (positive - negative) / (positive + negative)


def run_paired_statistics(paired: pd.DataFrame) -> pd.DataFrame:
    """Run two-sided Wilcoxon tests on 24 paired question-level means."""
    rows: list[dict[str, float | int | str]] = []
    for dimension in DIMENSIONS:
        data = paired.loc[paired["dimension"] == dimension].sort_values(
            "question_number"
        )
        if len(data) != len(EXPECTED_QUESTIONS):
            raise ValueError(f"{dimension}: expected 24 paired questions.")

        rag = data["RAG"].to_numpy(dtype=float)
        no_rag = data["No RAG"].to_numpy(dtype=float)
        differences = rag - no_rag

        if np.allclose(differences, 0.0):
            statistic, p_value = 0.0, 1.0
        else:
            result = stats.wilcoxon(
                differences,
                zero_method="wilcox",
                correction=False,
                alternative="two-sided",
                method="auto",
            )
            statistic, p_value = float(result.statistic), float(result.pvalue)

        rag_mean, rag_low, rag_high = mean_ci(rag)
        no_mean, no_low, no_high = mean_ci(no_rag)
        diff_mean, diff_low, diff_high = mean_ci(differences)
        rows.append(
            {
                "dimension": dimension,
                "n_paired_questions": len(data),
                "RAG_mean": rag_mean,
                "RAG_sd": float(np.std(rag, ddof=1)),
                "RAG_ci95_low": rag_low,
                "RAG_ci95_high": rag_high,
                "NoRAG_mean": no_mean,
                "NoRAG_sd": float(np.std(no_rag, ddof=1)),
                "NoRAG_ci95_low": no_low,
                "NoRAG_ci95_high": no_high,
                "mean_difference_RAG_minus_NoRAG": diff_mean,
                "difference_ci95_low": diff_low,
                "difference_ci95_high": diff_high,
                "wilcoxon_statistic": statistic,
                "wilcoxon_p_raw": p_value,
                "paired_rank_biserial": rank_biserial(differences),
            }
        )

    table = pd.DataFrame(rows)
    table["wilcoxon_p_holm"] = holm_adjust(table["wilcoxon_p_raw"].to_numpy())
    return table


def build_repeat_differences(scores: pd.DataFrame) -> pd.DataFrame:
    """Compute RAG - No RAG for every question and repeat combination."""
    index_columns = [
        "scoring_repeat",
        "generation_repeat",
        "question_number",
        "scenario_id",
        "dimension",
    ]
    differences = scores.pivot(
        index=index_columns,
        columns="condition",
        values="score",
    ).reset_index()
    differences.columns.name = None
    if differences[list(CONDITIONS)].isna().any().any():
        raise ValueError("Cannot compute repeat differences because scores are missing.")
    differences["difference_RAG_minus_NoRAG"] = (
        differences["RAG"] - differences["No RAG"]
    )
    return differences.sort_values(index_columns).reset_index(drop=True)


def format_p_value(value: float) -> str:
    if value < 0.001:
        return r"$p_{Holm}<0.001$"
    return rf"$p_{{Holm}}={value:.3f}$"


def save_figure(fig: mpl.figure.Figure, stem: str) -> None:
    """Save each figure as high-resolution raster and editable vector files."""
    for extension in FIGURE_FORMATS:
        fig.savefig(
            OUTPUT_DIR / f"{stem}.{extension}",
            dpi=FIGURE_DPI,
            bbox_inches="tight",
            pad_inches=0.04,
            facecolor="white",
        )
    plt.close(fig)


def plot_main_figure(paired: pd.DataFrame, test_table: pd.DataFrame) -> None:
    """Main text: paired question means plus overall mean and 95% CI."""
    fig, axes = plt.subplots(1, 3, figsize=(7.35, 3.15), sharey=True)
    offsets = np.linspace(-0.055, 0.055, len(EXPECTED_QUESTIONS))

    for panel_index, (axis, dimension) in enumerate(zip(axes, DIMENSIONS)):
        data = paired.loc[paired["dimension"] == dimension].sort_values(
            "question_number"
        )
        test = test_table.loc[test_table["dimension"] == dimension].iloc[0]
        rag = data["RAG"].to_numpy(dtype=float)
        no_rag = data["No RAG"].to_numpy(dtype=float)

        for idx in range(len(data)):
            x_pair = np.array([0.0, 1.0]) + offsets[idx]
            axis.plot(
                x_pair,
                [rag[idx], no_rag[idx]],
                color="#8C8C8C",
                linewidth=0.65,
                alpha=0.32,
                zorder=1,
            )

        axis.scatter(
            offsets,
            rag,
            s=12,
            facecolor="white",
            edgecolor=COLORS["RAG"],
            linewidth=0.7,
            alpha=0.95,
            zorder=2,
        )
        axis.scatter(
            1.0 + offsets,
            no_rag,
            s=12,
            facecolor="white",
            edgecolor=COLORS["No RAG"],
            linewidth=0.7,
            alpha=0.95,
            zorder=2,
        )

        for x_position, condition, prefix in (
            (0.0, "RAG", "RAG"),
            (1.0, "No RAG", "NoRAG"),
        ):
            mean = float(test[f"{prefix}_mean"])
            ci_low = float(test[f"{prefix}_ci95_low"])
            ci_high = float(test[f"{prefix}_ci95_high"])
            axis.errorbar(
                x_position,
                mean,
                yerr=np.array([[mean - ci_low], [ci_high - mean]]),
                fmt="o",
                markersize=6.2,
                markerfacecolor=COLORS[condition],
                markeredgecolor="black",
                markeredgewidth=0.7,
                ecolor="black",
                elinewidth=1.1,
                capsize=3.0,
                capthick=1.1,
                zorder=4,
            )

        difference = float(test["mean_difference_RAG_minus_NoRAG"])
        difference_low = float(test["difference_ci95_low"])
        difference_high = float(test["difference_ci95_high"])
        annotation = (
            f"Mean Δ (RAG - No RAG) = {difference:+.2f}\n"
            f"95% CI: [{difference_low:+.2f}, {difference_high:+.2f}]; "
            f"{format_p_value(float(test['wilcoxon_p_holm']))}"
        )
        axis.text(
            0.5,
            0.985,
            annotation,
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=7.3,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.3},
            zorder=5,
        )

        axis.set_title(dimension, fontsize=10, pad=7)
        axis.set_xlim(-0.22, 1.22)
        axis.set_xticks([0, 1], CONDITIONS)
        # Reserve a clear band above the 1-10 score range for the statistics.
        axis.set_ylim(0.8, 11.7)
        axis.set_yticks(np.arange(1, 11, 1))
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.5, alpha=0.65)
        axis.set_axisbelow(True)
        if panel_index == 0:
            axis.set_ylabel("GPT expert score")

    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.10, top=0.91, wspace=0.20)
    save_figure(fig, "Figure_main_paired_scores")


def plot_si_question_bars(
    scores: pd.DataFrame,
    question_summary: pd.DataFrame,
) -> None:
    """SI: mean +/- SD bars with all nine raw values overlaid."""
    x = np.arange(len(EXPECTED_QUESTIONS), dtype=float)
    bar_width = 0.34
    condition_shift = {"RAG": -bar_width / 2.0, "No RAG": bar_width / 2.0}
    repeat_jitter = np.linspace(-bar_width * 0.30, bar_width * 0.30, 9)

    for si_number, dimension in enumerate(DIMENSIONS, start=1):
        fig, axis = plt.subplots(figsize=(11.2, 4.8))
        dimension_summary = question_summary.loc[
            question_summary["dimension"] == dimension
        ]

        for condition in CONDITIONS:
            summary = dimension_summary.loc[
                dimension_summary["condition"] == condition
            ].sort_values("question_number")
            if len(summary) != len(EXPECTED_QUESTIONS):
                raise ValueError(f"{dimension}/{condition}: incomplete question summary.")

            centers = x + condition_shift[condition]
            axis.bar(
                centers,
                summary["mean"],
                width=bar_width,
                yerr=summary["sd"],
                color=COLORS[condition],
                alpha=0.68,
                edgecolor="black",
                linewidth=0.55,
                error_kw={
                    "ecolor": "black",
                    "elinewidth": 0.75,
                    "capsize": 1.8,
                    "capthick": 0.75,
                },
                zorder=2,
            )

            repeat_index = 0
            for scoring_repeat in EXPECTED_SCORING_REPEATS:
                for generation_repeat in EXPECTED_GENERATION_REPEATS:
                    raw = scores.loc[
                        (scores["dimension"] == dimension)
                        & (scores["condition"] == condition)
                        & (scores["scoring_repeat"] == scoring_repeat)
                        & (scores["generation_repeat"] == generation_repeat)
                    ].sort_values("question_number")
                    if len(raw) != len(EXPECTED_QUESTIONS):
                        raise ValueError(
                            f"{dimension}/{condition}/scoring {scoring_repeat}/"
                            f"generation {generation_repeat}: expected 24 scores."
                        )
                    axis.scatter(
                        centers + repeat_jitter[repeat_index],
                        raw["score"],
                        s=9,
                        facecolor=COLORS[condition],
                        edgecolor="white",
                        linewidth=0.35,
                        alpha=0.92,
                        zorder=3,
                    )
                    repeat_index += 1

        handles = [
            Patch(
                facecolor=COLORS[condition],
                edgecolor="black",
                linewidth=0.55,
                alpha=0.68,
                label=condition,
            )
            for condition in CONDITIONS
        ]
        axis.legend(
            handles=handles,
            loc="upper right",
            bbox_to_anchor=(0.92, 0.995),
            borderaxespad=0.0,
            ncol=2,
            fontsize=12,
        )
        axis.set_title(dimension, fontsize=15, pad=9)
        axis.set_xlabel("Question", fontsize=13)
        axis.set_ylabel("GPT expert score", fontsize=13)
        axis.set_xlim(-0.65, len(EXPECTED_QUESTIONS) - 0.35)
        axis.set_ylim(0.5, 10.5)
        axis.set_xticks(x, [str(number) for number in EXPECTED_QUESTIONS])
        axis.set_yticks(np.arange(1, 11, 1))
        axis.tick_params(axis="x", labelsize=11)
        axis.tick_params(axis="y", labelsize=11)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.5, alpha=0.65)
        axis.set_axisbelow(True)
        fig.subplots_adjust(left=0.075, right=0.995, bottom=0.17, top=0.90)

        stem = f"Figure_SI{si_number}_{dimension.lower()}_by_question"
        save_figure(fig, stem)


def plot_si_repeat_heatmap(differences: pd.DataFrame) -> None:
    """SI: repeat-level RAG - No RAG differences for every question."""
    row_pairs = [
        (scoring_repeat, generation_repeat)
        for scoring_repeat in EXPECTED_SCORING_REPEATS
        for generation_repeat in EXPECTED_GENERATION_REPEATS
    ]
    row_labels = [f"G{s}-R{g}" for s, g in row_pairs]
    maximum = float(np.nanmax(np.abs(differences["difference_RAG_minus_NoRAG"])))
    maximum = max(1.0, float(np.ceil(maximum)))
    norm = TwoSlopeNorm(vmin=-maximum, vcenter=0.0, vmax=maximum)

    fig, axes = plt.subplots(3, 1, figsize=(10.3, 6.35), sharex=True)
    image = None
    for panel_index, (axis, dimension) in enumerate(zip(axes, DIMENSIONS)):
        matrix = np.full((len(row_pairs), len(EXPECTED_QUESTIONS)), np.nan)
        subset = differences.loc[differences["dimension"] == dimension]

        for row_index, (scoring_repeat, generation_repeat) in enumerate(row_pairs):
            row = subset.loc[
                (subset["scoring_repeat"] == scoring_repeat)
                & (subset["generation_repeat"] == generation_repeat)
            ].sort_values("question_number")
            if len(row) != len(EXPECTED_QUESTIONS):
                raise ValueError(
                    f"{dimension}/G{scoring_repeat}-R{generation_repeat}: "
                    "expected 24 differences."
                )
            matrix[row_index, :] = row["difference_RAG_minus_NoRAG"].to_numpy()

        image = axis.imshow(
            matrix,
            cmap="RdBu",
            norm=norm,
            aspect="auto",
            interpolation="nearest",
        )
        axis.set_title(dimension, fontsize=9.5, pad=5)
        axis.set_yticks(np.arange(len(row_pairs)), row_labels)
        axis.tick_params(axis="y", labelsize=7.2, length=0)
        axis.set_ylabel("Grading / response repeat", fontsize=8)
        axis.set_xticks(
            np.arange(-0.5, len(EXPECTED_QUESTIONS), 1), minor=True
        )
        axis.set_yticks(np.arange(-0.5, len(row_pairs), 1), minor=True)
        axis.grid(which="minor", color="white", linewidth=0.35, alpha=0.70)
        axis.tick_params(which="minor", bottom=False, left=False)
        for spine in axis.spines.values():
            spine.set_visible(False)
        if panel_index < len(DIMENSIONS) - 1:
            axis.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

    axes[-1].set_xticks(
        np.arange(len(EXPECTED_QUESTIONS)),
        [str(number) for number in EXPECTED_QUESTIONS],
    )
    axes[-1].tick_params(axis="x", labelsize=7.2, length=2.5)
    axes[-1].set_xlabel("Question")

    if image is None:
        raise RuntimeError("No heatmap image was created.")
    colorbar = fig.colorbar(image, ax=axes, location="right", fraction=0.025, pad=0.018)
    colorbar.set_label("RAG - No RAG score", fontweight="bold")
    colorbar.ax.tick_params(labelsize=7.2)
    fig.subplots_adjust(left=0.115, right=0.90, bottom=0.09, top=0.96, hspace=0.25)
    save_figure(fig, "Figure_SI4_repeat_difference_heatmap")


def export_tables(
    scores: pd.DataFrame,
    manifest: pd.DataFrame,
    question_summary: pd.DataFrame,
    paired: pd.DataFrame,
    test_table: pd.DataFrame,
    differences: pd.DataFrame,
) -> None:
    """Export all plotted values and inferential results for reproducibility."""
    manifest.to_csv(OUTPUT_DIR / "input_manifest.csv", index=False, encoding="utf-8-sig")
    scores.to_csv(OUTPUT_DIR / "all_scores_long.csv", index=False, encoding="utf-8-sig")
    question_summary.to_csv(
        OUTPUT_DIR / "question_summary_nine_repeats.csv",
        index=False,
        encoding="utf-8-sig",
    )
    paired.to_csv(
        OUTPUT_DIR / "paired_question_means.csv", index=False, encoding="utf-8-sig"
    )
    test_table.to_csv(
        OUTPUT_DIR / "wilcoxon_and_overall_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    differences.to_csv(
        OUTPUT_DIR / "repeat_level_differences.csv",
        index=False,
        encoding="utf-8-sig",
    )


def write_analysis_summary(test_table: pd.DataFrame) -> None:
    """Write a compact, auditable text summary of the analysis."""
    lines = [
        "GPT score analysis",
        "==================",
        "",
        "Design: 24 paired questions; RAG and No RAG; three response-generation "
        "repeats crossed with three GPT-scoring repeats (nine scores per question, "
        "system, and dimension).",
        "",
        "Descriptive display: SI bars show mean +/- sample SD across all nine raw "
        "scores, with every raw value overlaid. Main-figure points and 95% CIs are "
        "calculated across 24 question-level means.",
        "",
        "Inference: two-sided Wilcoxon signed-rank tests use the 24 paired "
        "question-level means. The three p values are corrected by the Holm method. "
        "The confidence interval for the mean paired difference is a t interval "
        "across questions.",
        "",
        "Results:",
    ]
    for row in test_table.itertuples(index=False):
        lines.append(
            f"- {row.dimension}: RAG {row.RAG_mean:.2f} "
            f"[95% CI {row.RAG_ci95_low:.2f}, {row.RAG_ci95_high:.2f}]; "
            f"No RAG {row.NoRAG_mean:.2f} "
            f"[95% CI {row.NoRAG_ci95_low:.2f}, {row.NoRAG_ci95_high:.2f}]; "
            f"difference {row.mean_difference_RAG_minus_NoRAG:+.2f} "
            f"[95% CI {row.difference_ci95_low:+.2f}, "
            f"{row.difference_ci95_high:+.2f}]; "
            f"Wilcoxon p={row.wilcoxon_p_raw:.4g}, "
            f"Holm-adjusted p={row.wilcoxon_p_holm:.4g}, "
            f"rank-biserial={row.paired_rank_biserial:+.3f}."
        )
    (OUTPUT_DIR / "analysis_summary.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    set_scientific_style()

    files = discover_input_files()
    scores, manifest = load_scores(files)
    question_summary = build_question_summary(scores)
    paired = build_paired_question_means(question_summary)
    test_table = run_paired_statistics(paired)
    differences = build_repeat_differences(scores)

    export_tables(
        scores,
        manifest,
        question_summary,
        paired,
        test_table,
        differences,
    )
    write_analysis_summary(test_table)
    plot_main_figure(paired, test_table)
    plot_si_question_bars(scores, question_summary)
    plot_si_repeat_heatmap(differences)

    print(f"Analysis complete. Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
