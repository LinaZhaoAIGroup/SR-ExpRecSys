"""Calculate BGE-M3 similarity and plot the ablation comparison."""
import importlib.util
import subprocess
import sys

if importlib.util.find_spec("seaborn") is None:
    subprocess.check_call([
        sys.executable,
        "-m",
        "pip",
        "install",
        "seaborn",
    ])

import seaborn as sns
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from FlagEmbedding import BGEM3FlagModel


SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_FILE = SCRIPT_DIR / "ablation_responses.xlsx"
INPUT_SHEET = "Sheet2"
OUTPUT_DIR = SCRIPT_DIR / "bge_m3_ablation_figures"

MODEL_NAME = "BAAI/bge-m3"
BATCH_SIZE = 4
MAX_LENGTH = 1024
RANDOM_SEED = 42

METHODS = ["KG-only RAG", "Vector-only RAG", "RAG", "No RAG"]
DISPLAY_LABELS = ["KG-only RAG", "Vector-only RAG", "KG-Vector RAG", "No RAG"]
COLORS = ["#377EB8", "#4DAF4A", "#E41A1C", "#999999"]
GOLD_COLUMN = "Gold Standard"

FIGURE_PNG = OUTPUT_DIR / "bge_m3_ablation_similarity.png"
FIGURE_PDF = OUTPUT_DIR / "bge_m3_ablation_similarity.pdf"
FIGURE_SVG = OUTPUT_DIR / "bge_m3_ablation_similarity.svg"
SCORES_CSV = OUTPUT_DIR / "bge_m3_ablation_similarity_scores.csv"


def load_data():
    """Load and validate the four answers and their gold standards."""
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found:{INPUT_FILE}")

    df = pd.read_excel(INPUT_FILE, sheet_name=INPUT_SHEET)
    required_columns = [*METHODS, GOLD_COLUMN]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"{INPUT_SHEET} Missing columns:{missing}")

    return df


def calculate_similarities(df):
    """Calculate row-matched cosine similarity with BGE-M3 dense vectors."""
    model = BGEM3FlagModel(MODEL_NAME, use_fp16=True)
    gold_texts = df[GOLD_COLUMN].fillna("").astype(str).tolist()
    gold_embeddings = model.encode(
        gold_texts,
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
    )["dense_vecs"]

    score_table = pd.DataFrame(index=df.index)
    if "Scenario ID" in df.columns:
        score_table["Scenario ID"] = df["Scenario ID"]

    plot_frames = []
    for method in METHODS:
        answer_texts = df[method].fillna("").astype(str).tolist()
        answer_embeddings = model.encode(
            answer_texts,
            batch_size=BATCH_SIZE,
            max_length=MAX_LENGTH,
        )["dense_vecs"]

        # BGE-M3 dense vectors are normalized; paired dot products are cosine scores.
        similarities = np.einsum(
            "ij,ij->i",
            answer_embeddings,
            gold_embeddings,
        )
        score_table[method] = similarities
        plot_frames.append(
            pd.DataFrame(
                {
                    "Method": method,
                    "Similarity": similarities,
                }
            )
        )

    plot_data = pd.concat(plot_frames, ignore_index=True)
    return plot_data, score_table


def plot_figure(plot_data):
    """Draw the existing academic-style figure without a mean connection line."""
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

    # Keep the mean markers and labels, but do not connect the means with a line.
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
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data()
    plot_data, score_table = calculate_similarities(data)
    score_table.to_csv(SCORES_CSV, index=False, encoding="utf-8-sig")
    plot_figure(plot_data)

    print(f"Input file:{INPUT_FILE}")
    print(f"Model:{MODEL_NAME}")
    print(f"Output folder:{OUTPUT_DIR}")
    print(f"Picture:{FIGURE_PNG}")


if __name__ == "__main__":
    main()
