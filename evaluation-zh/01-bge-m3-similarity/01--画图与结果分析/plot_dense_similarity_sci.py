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
INPUT_FILE = SCRIPT_DIR / "BGE-M3综合相似度计算结果.xlsx"
OUTPUT_DIR = SCRIPT_DIR / "dense_similarity_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FIGURE_PNG = OUTPUT_DIR / "dense_similarity_by_scenario.png"
FIGURE_PDF = OUTPUT_DIR / "dense_similarity_by_scenario.pdf"
FIGURE_SVG = OUTPUT_DIR / "dense_similarity_by_scenario.svg"
LONG_CSV = OUTPUT_DIR / "dense_similarity_long.csv"
STATS_CSV = OUTPUT_DIR / "dense_similarity_stats.csv"
GROUP_SUMMARY_CSV = OUTPUT_DIR / "dense_similarity_group_summary.csv"

RUN_SHEETS = ["Sheet1结果", "Sheet2结果", "Sheet3结果"]
#SCENARIO_GROUPS = ["Level 1", "Level 2", "Level 3", "Level 4"]
SCENARIO_GROUPS = ["L1", "L2", "L3", "L4"]
SCORE_COLUMNS = {
    "RAG": "回答1-dense相似度",
    "No-RAG": "回答2-dense相似度",
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
        raise FileNotFoundError(f"找不到输入文件：{INPUT_FILE}")

    frames = []
    for replicate, sheet_name in enumerate(RUN_SHEETS, start=1):
        df = pd.read_excel(INPUT_FILE, sheet_name=sheet_name)
        required = {"场景编号", "轮次", *SCORE_COLUMNS.values()}
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(f"{sheet_name} 缺少列：{missing}")

        df = df.copy()
        df["场景编号"] = df["场景编号"].astype(str)
        df["组别"] = df["场景编号"].str.extract(r"^(L[1-4])", expand=False)
        df["问题序号"] = pd.to_numeric(
            df["场景编号"].str.extract(r"-(\d+)$", expand=False),
            errors="raise",
        ).astype(int)
        if df["组别"].isna().any():
            raise ValueError(f"{sheet_name} 存在无法识别的场景编号")

        for condition, score_column in SCORE_COLUMNS.items():
            frames.append(
                pd.DataFrame(
                    {
                        "工作表": sheet_name,
                        "重复运行": replicate,
                        "组别": df["组别"],
                        "场景编号": df["场景编号"],
                        "问题序号": df["问题序号"],
                        "condition": condition,
                        "score": pd.to_numeric(df[score_column], errors="coerce"),
                    }
                )
            )

    long_df = pd.concat(frames, ignore_index=True)
    if long_df["score"].isna().any():
        raise ValueError("输入结果中存在无法转换为数值的 dense 相似度")

    expected = {
        f"L{group}-{question}"
        for group in range(1, 5)
        for question in range(1, 7)
    }
    actual = set(long_df["场景编号"])
    if actual != expected:
        raise ValueError(
            "场景编号不是完整的 L1-L4 × 1-6 结构；"
            f"缺失={sorted(expected - actual)}，多出={sorted(actual - expected)}"
        )
    return long_df.sort_values(
        ["组别", "问题序号", "重复运行", "condition"]
    ).reset_index(drop=True)


def calculate_statistics(long_df):
    wide = (
        long_df.pivot_table(
            index=["组别", "场景编号", "问题序号", "重复运行"],
            columns="condition",
            values="score",
            aggfunc="mean",
        )
        .reset_index()
        .sort_values(["组别", "问题序号", "重复运行"])
    )

    rows = []
    for (group, scenario, question), subset in wide.groupby(
        ["组别", "场景编号", "问题序号"], sort=False
    ):
        row = {
            "组别": group,
            "场景编号": scenario,
            "问题序号": question,
        }
        for condition in SCORE_COLUMNS:
            mean, sd, variance, ci_low, ci_high, n = mean_and_ci(
                subset[condition].to_numpy(dtype=float)
            )
            row.update(
                {
                    f"{condition}重复次数": n,
                    f"{condition}均值": mean,
                    f"{condition}标准差": sd,
                    f"{condition}方差": variance,
                    f"{condition}CI95下限": ci_low,
                    f"{condition}CI95上限": ci_high,
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
        group_stats = stats_df[stats_df["组别"] == group].sort_values("问题序号")
        group_long = long_df[long_df["组别"] == group]
        x = np.arange(1, 7, dtype=float)
        offset = 0.16

        for condition, position_offset in [("RAG", -offset), ("No-RAG", offset)]:
            condition_long = group_long[group_long["condition"] == condition]
            for question, question_data in condition_long.groupby("问题序号", sort=True):
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

            means = group_stats[f"{condition}均值"].to_numpy()
            sd = group_stats[f"{condition}标准差"].to_numpy()
            ci_low = group_stats[f"{condition}CI95下限"].to_numpy()
            ci_high = group_stats[f"{condition}CI95上限"].to_numpy()

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
            paired = group_long[group_long["问题序号"] == question].pivot(
                index="重复运行", columns="condition", values="score"
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
    for group, subset in long_df.groupby("组别", sort=False):
        for condition, condition_data in subset.groupby("condition", sort=False):
            mean, sd, variance, ci_low, ci_high, n = mean_and_ci(condition_data["score"])
            group_rows.append(
                {
                    "组别": group,
                    "condition": condition,
                    "n": n,
                    "均值": mean,
                    "标准差": sd,
                    "方差": variance,
                    "CI95下限": ci_low,
                    "CI95上限": ci_high,
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
    print(f"输入文件：{INPUT_FILE}")
    print(f"输出文件夹：{OUTPUT_DIR}")
    print(f"图像：{FIGURE_PNG}")
    print(f"统计：{STATS_CSV}")


if __name__ == "__main__":
    main()
