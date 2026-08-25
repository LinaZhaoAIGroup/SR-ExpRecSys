"""Plot dense similarity for RAG and no-RAG answers.

The workbook contains three repeated runs. The figure keeps the existing
four-panel point/error-bar layout and shows individual runs, means, standard
deviations, and 95% confidence intervals for the six questions in L1-L4.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy import stats


SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_FILE = SCRIPT_DIR / "bge_m3_combined_similarity_results.xlsx"
OUTPUT_DIR = SCRIPT_DIR / "dense_similarity_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FIGURE_PNG = OUTPUT_DIR / "dense_similarity_by_scenario.png"
FIGURE_PDF = OUTPUT_DIR / "dense_similarity_by_scenario.pdf"
FIGURE_SVG = OUTPUT_DIR / "dense_similarity_by_scenario.svg"
LONG_CSV = OUTPUT_DIR / "dense_similarity_long.csv"
STATS_CSV = OUTPUT_DIR / "dense_similarity_stats.csv"
GROUP_SUMMARY_CSV = OUTPUT_DIR / "dense_similarity_group_summary.csv"

RUN_SHEETS = ["Sheet1 Results", "Sheet2 Results", "Sheet3 Results"]
#SCENARIO_GROUPS = ["Level 1", "Level 2", "Level 3", "Level 4"]
SCENARIO_GROUPS = ["L1", "L2", "L3", "L4"]
SCORE_COLUMNS = {
    "RAG": "Answer 1-Dense Similarity",
    "No-RAG": "Answer 2-Dense Similarity",
}


def mean_and_ci(values, confidence=0.95):
    """Return mean, SD, variance, and a t-based confidence interval."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    mean = float(np.mean(values)) if n else np.nan
    if n < 2:
        return mean, np.nan, np.nan, mean, mean, n

    sd = float(np.std(values, ddof=1))
    variance = float(np.var(values, ddof=1))
    critical = stats.t.ppf((1 + confidence) / 2, n - 1)
    margin = critical * sd / np.sqrt(n)
    return mean, sd, variance, mean - margin, mean + margin, n


def load_long_data():
    if not INPUT_FILE.exists():
        if not LONG_CSV.exists():
            raise FileNotFoundError(
                f"Neither the processed workbook nor archived long-form data was found: "
                f"{INPUT_FILE}, {LONG_CSV}"
            )
        long_df = pd.read_csv(LONG_CSV, encoding="utf-8-sig")
        required = {
            "worksheet",
            "Repeat run",
            "Group",
            "Scenario ID",
            "Question number",
            "condition",
            "score",
        }
        missing = sorted(required - set(long_df.columns))
        if missing:
            raise ValueError(f"Archived long-form CSV is missing columns: {missing}")
        return long_df.sort_values(
            ["Group", "Question number", "Repeat run", "condition"]
        ).reset_index(drop=True)

    frames = []
    for replicate, sheet_name in enumerate(RUN_SHEETS, start=1):
        df = pd.read_excel(INPUT_FILE, sheet_name=sheet_name)
        required = {"Scenario ID", "Run", *SCORE_COLUMNS.values()}
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(f"{sheet_name} Missing columns:{missing}")

        df = df.copy()
        df["Scenario ID"] = df["Scenario ID"].astype(str)
        df["Group"] = df["Scenario ID"].str.extract(r"^(L[1-4])", expand=False)
        df["Question number"] = pd.to_numeric(
            df["Scenario ID"].str.extract(r"-(\d+)$", expand=False),
            errors="raise",
        ).astype(int)
        if df["Group"].isna().any():
            raise ValueError(f"{sheet_name} There is an unrecognized scene number")

        for condition, score_column in SCORE_COLUMNS.items():
            frames.append(
                pd.DataFrame(
                    {
                        "worksheet": sheet_name,
                        "Repeat run": replicate,
                        "Group": df["Group"],
                        "Scenario ID": df["Scenario ID"],
                        "Question number": df["Question number"],
                        "condition": condition,
                        "score": pd.to_numeric(df[score_column], errors="coerce"),
                    }
                )
            )

    long_df = pd.concat(frames, ignore_index=True)
    if long_df["score"].isna().any():
        raise ValueError("There are some input results that cannot be converted into numerical values.dense Similarity")

    expected = {
        f"L{group}-{question}"
        for group in range(1, 5)
        for question in range(1, 7)
    }
    actual = set(long_df["Scenario ID"])
    if actual != expected:
        raise ValueError(
            "Scene number is not completeL1-L4 × 1-6 structure;"
            f"Missing={sorted(expected - actual)}, Too much={sorted(actual - expected)}"
        )
    return long_df.sort_values(
        ["Group", "Question number", "Repeat run", "condition"]
    ).reset_index(drop=True)


def calculate_statistics(long_df):
    wide = (
        long_df.pivot_table(
            index=["Group", "Scenario ID", "Question number", "Repeat run"],
            columns="condition",
            values="score",
            aggfunc="mean",
        )
        .reset_index()
        .sort_values(["Group", "Question number", "Repeat run"])
    )

    rows = []
    for (group, scenario, question), subset in wide.groupby(
        ["Group", "Scenario ID", "Question number"], sort=False
    ):
        row = {
            "Group": group,
            "Scenario ID": scenario,
            "Question number": question,
        }
        for condition in SCORE_COLUMNS:
            mean, sd, variance, ci_low, ci_high, n = mean_and_ci(
                subset[condition].to_numpy(dtype=float)
            )
            row.update(
                {
                    f"{condition}Repeat times": n,
                    f"{condition}mean": mean,
                    f"{condition}standard deviation": sd,
                    f"{condition}variance": variance,
                    f"{condition}CI95lower limit": ci_low,
                    f"{condition}CI95upper limit": ci_high,
                }
            )
        rows.append(row)

    return wide, pd.DataFrame(rows)


def plot_figure(long_df, stats_df):
    # Plot settings follow evaluate.ipynb.
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["svg.fonttype"] = "none"

    colors = {"RAG": "#1f77b4", "No-RAG": "#d62728"}
    markers = {"RAG": "o", "No-RAG": "s"}

    # Compact full-width layout for manuscript typesetting.
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.8), dpi=600, sharey=True)
    axes = axes.flatten()

    y_values = [long_df["score"].to_numpy()]
    for ax, group in zip(axes, SCENARIO_GROUPS):
        group_stats = stats_df[stats_df["Group"] == group].sort_values("Question number")
        group_long = long_df[long_df["Group"] == group]
        x = np.arange(1, 7, dtype=float)
        offset = 0.16

        for condition, position_offset in [("RAG", -offset), ("No-RAG", offset)]:
            condition_long = group_long[group_long["condition"] == condition]
            for question, question_data in condition_long.groupby("Question number", sort=True):
                ax.scatter(
                    np.repeat(question + position_offset, len(question_data)),
                    question_data["score"],
                    s=18,
                    marker=markers[condition],
                    color=colors[condition],
                    alpha=0.30,
                    edgecolors="none",
                    zorder=2,
                )

            means = group_stats[f"{condition}mean"].to_numpy()
            sd = group_stats[f"{condition}standard deviation"].to_numpy()
            ci_low = group_stats[f"{condition}CI95lower limit"].to_numpy()
            ci_high = group_stats[f"{condition}CI95upper limit"].to_numpy()

            # Broad translucent line: mean +/- 1 SD.
            ax.vlines(
                x + position_offset,
                means - sd,
                means + sd,
                color=colors[condition],
                linewidth=5,
                alpha=0.22,
                zorder=3,
            )
            # Error bars: mean +/- 95% CI.
            ax.errorbar(
                x + position_offset,
                means,
                yerr=[means - ci_low, ci_high - means],
                fmt=markers[condition],
                color=colors[condition],
                markerfacecolor=colors[condition],
                markeredgecolor="white",
                markeredgewidth=0.6,
                markersize=5.4,
                capsize=2.5,
                capthick=0.8,
                linewidth=1.1,
                zorder=4,
                label=condition,
            )
            y_values.extend([means - sd, means + sd, ci_low, ci_high])

        # Thin paired lines retain the repeated-run structure without a test annotation.
        for question in range(1, 7):
            paired = group_long[group_long["Question number"] == question].pivot(
                index="Repeat run", columns="condition", values="score"
            )
            for _, values in paired.iterrows():
                if {"RAG", "No-RAG"}.issubset(values.index):
                    ax.plot(
                        [question - offset, question + offset],
                        [values["RAG"], values["No-RAG"]],
                        color="0.70",
                        linewidth=0.45,
                        alpha=0.7,
                        zorder=1,
                    )

        ax.set_title(group, fontsize=11, fontweight="bold", loc="left", pad=3)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{group}-{i}" for i in range(1, 7)])
        ax.set_xlim(0.45, 6.55)
        ax.grid(True, axis="y", linestyle="--", alpha=0.35, color="gray")
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="x", labelsize=9, rotation=0, pad=2)
        ax.tick_params(axis="y", labelsize=9, pad=2)
        ax.set_xlabel("Question", fontsize=10, fontweight="bold", labelpad=3)

    all_y = np.concatenate([np.asarray(values, dtype=float).ravel() for values in y_values])
    y_min = float(np.nanmin(all_y)) - 0.04
    y_max = float(np.nanmax(all_y)) + 0.04
    for ax in axes:
        ax.set_ylim(y_min, y_max)

    axes[0].set_ylabel("BGE-M3 dense similarity", fontsize=10, fontweight="bold")
    axes[2].set_ylabel("BGE-M3 dense similarity", fontsize=10, fontweight="bold")

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker=markers[condition],
            color=colors[condition],
            markerfacecolor=colors[condition],
            markeredgecolor="white",
            markersize=6,
            linewidth=1.2,
            label=condition,
        )
        for condition in ["RAG", "No-RAG"]
    ]
    legend_handles.extend(
        [
            Line2D([0], [0], color="0.35", linewidth=5, alpha=0.22, label="mean +/- SD"),
            Line2D([0], [0], color="0.35", linewidth=1.0, marker="|", markersize=7, label="95% CI"),
        ]
    )
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=4,
        frameon=False,
        fontsize=9,
        handlelength=2.0,
        columnspacing=1.3,
    )

    fig.subplots_adjust(
        left=0.09,
        right=0.99,
        bottom=0.09,
        top=0.91,
        wspace=0.09,
        hspace=0.38,
    )
    fig.savefig(
        FIGURE_PNG,
        bbox_inches="tight",
        dpi=600,
        facecolor="white",
        edgecolor="none",
    )
    fig.savefig(
        FIGURE_PDF,
        bbox_inches="tight",
        dpi=600,
        format="pdf",
        facecolor="white",
        edgecolor="none",
    )
    fig.savefig(
        FIGURE_SVG,
        bbox_inches="tight",
        dpi=600,
        format="svg",
        facecolor="white",
        edgecolor="none",
    )
    plt.close(fig)


def save_statistics(long_df, stats_df):
    long_df.to_csv(LONG_CSV, index=False, encoding="utf-8-sig")
    stats_df.to_csv(STATS_CSV, index=False, encoding="utf-8-sig")

    group_rows = []
    for group, subset in long_df.groupby("Group", sort=False):
        for condition, condition_data in subset.groupby("condition", sort=False):
            mean, sd, variance, ci_low, ci_high, n = mean_and_ci(condition_data["score"])
            group_rows.append(
                {
                    "Group": group,
                    "condition": condition,
                    "n": n,
                    "mean": mean,
                    "standard deviation": sd,
                    "variance": variance,
                    "CI95lower limit": ci_low,
                    "CI95upper limit": ci_high,
                }
            )
    pd.DataFrame(group_rows).to_csv(
        GROUP_SUMMARY_CSV,
        index=False,
        encoding="utf-8-sig",
    )


def main():
    long_df = load_long_data()
    _, stats_df = calculate_statistics(long_df)
    save_statistics(long_df, stats_df)
    plot_figure(long_df, stats_df)
    print(f"Input file:{INPUT_FILE}")
    print(f"Output folder:{OUTPUT_DIR}")
    print(f"Image:{FIGURE_PNG}")
    print(f"Statistics:{STATS_CSV}")


if __name__ == "__main__":
    main()
