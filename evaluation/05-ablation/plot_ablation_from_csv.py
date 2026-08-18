"""Plot the BGE-M3 ablation comparison from saved CSV scores."""

import importlib.util
from pathlib import Path
import subprocess
import sys


if importlib.util.find_spec("seaborn") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "seaborn"])

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "bge_m3_ablation_figures"
INPUT_CSV = DATA_DIR / "bge_m3_ablation_similarity_scores.csv"

FIGURE_PNG = DATA_DIR / "bge_m3_ablation_similarity_from_csv.png"
FIGURE_PDF = DATA_DIR / "bge_m3_ablation_similarity_from_csv.pdf"
FIGURE_SVG = DATA_DIR / "bge_m3_ablation_similarity_from_csv.svg"

METHODS = ["单独KG回答", "单独向量检索回答", "RAG", "no RAG"]
DISPLAY_LABELS = ["KG-only RAG", "Vector-only RAG", "KG-Vector RAG", "No-RAG"]
COLORS = ["#377EB8", "#4DAF4A", "#E41A1C", "#999999"]
RANDOM_SEED = 42


def load_plot_data():
    """Read the wide score table and convert it to plotting format."""
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"找不到输入文件：{INPUT_CSV}")

    score_data = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")
    missing = [column for column in METHODS if column not in score_data.columns]
    if missing:
        raise ValueError(f"CSV 缺少列：{missing}")

    numeric_scores = score_data[METHODS].apply(pd.to_numeric, errors="coerce")
    if numeric_scores.isna().any().any():
        invalid_columns = numeric_scores.columns[numeric_scores.isna().any()].tolist()
        raise ValueError(f"以下列包含空值或非数值：{invalid_columns}")

    plot_data = numeric_scores.melt(
        var_name="Method",
        value_name="Similarity",
    )
    return plot_data


def plot_figure(plot_data):
    """Draw the saved-score figure in the established manuscript style."""
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.linewidth"] = 1.2
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["svg.fonttype"] = "none"

    sns.set_style("white")
    np.random.seed(RANDOM_SEED)

    fig, ax = plt.subplots(figsize=(7, 4), dpi=600)
    palette = dict(zip(METHODS, COLORS))

    sns.boxplot(
        x="Method",
        y="Similarity",
        data=plot_data,
        order=METHODS,
        hue="Method",
        palette=palette,
        dodge=False,
        width=0.45,
        linewidth=1.5,
        fliersize=0,
        ax=ax,
    )

    if ax.get_legend() is not None:
        ax.get_legend().remove()

    for patch in ax.patches:
        red, green, blue, _ = patch.get_facecolor()
        patch.set_facecolor((red, green, blue, 0.62))

    sns.stripplot(
        x="Method",
        y="Similarity",
        data=plot_data,
        order=METHODS,
        color="#252525",
        size=4,
        alpha=0.60,
        jitter=0.15,
        zorder=10,
        ax=ax,
    )

    means = plot_data.groupby("Method")["Similarity"].mean().reindex(METHODS)
    ax.scatter(
        range(len(METHODS)),
        means.values,
        color="black",
        marker="X",
        s=55,
        zorder=12,
        label="Mean",
    )

    ax.set_ylabel("Cosine Similarity", fontweight="bold", fontsize=12)
    ax.set_xlabel("")
    ax.set_ylim(0.4, 1.1)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.set_axisbelow(True)

    ax.set_xticks(range(len(METHODS)))
    ax.set_xticklabels(DISPLAY_LABELS, fontsize=11, fontweight="bold")

    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)

    for side in ["bottom", "left"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(1.2)
        ax.spines[side].set_color("black")

    ax.tick_params(
        axis="x",
        which="both",
        bottom=True,
        top=False,
        direction="out",
        length=5,
        width=1.2,
        labelsize=11,
    )
    ax.tick_params(
        axis="y",
        which="both",
        left=True,
        right=False,
        direction="out",
        length=5,
        width=1.2,
        labelsize=11,
    )

    for index, value in enumerate(means.values):
        ax.text(
            index,
            value + 0.03,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="black",
            bbox={
                "boxstyle": "round,pad=0.15",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.85,
            },
            zorder=13,
        )

    ax.legend(loc="upper right", fontsize=10, frameon=False)

    fig.tight_layout()
    fig.savefig(FIGURE_PNG, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(FIGURE_PDF, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(FIGURE_SVG, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    plot_data = load_plot_data()
    plot_figure(plot_data)

    print(f"输入文件：{INPUT_CSV}")
    print(f"图片：{FIGURE_PNG}")


if __name__ == "__main__":
    main()
