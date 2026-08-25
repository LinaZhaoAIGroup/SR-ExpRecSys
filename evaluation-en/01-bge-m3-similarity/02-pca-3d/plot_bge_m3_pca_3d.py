"""Visualize BGE-M3 answer embeddings in a shared three-dimensional PCA space.

The script reads Sheet1 from benchmark_questions.xlsx, where Answer 1 is the
RAG answer, Answer 2 is the No-RAG answer, and Gold Standard is the reference.
The RAG, No-RAG, and gold-standard embeddings are jointly fitted with one PCA
model so that the two panels use the same coordinate system.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from FlagEmbedding import BGEM3FlagModel
from sklearn.decomposition import PCA


SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_FILE = SCRIPT_DIR / "benchmark_questions.xlsx"
OUTPUT_DIR = SCRIPT_DIR / "bge_m3_pca_3d_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FIGURE_PNG = OUTPUT_DIR / "BGE-M3_PCA_3D_RAG_vs_No-RAG.png"
FIGURE_PDF = OUTPUT_DIR / "BGE-M3_PCA_3D_RAG_vs_No-RAG.pdf"
FIGURE_SVG = OUTPUT_DIR / "BGE-M3_PCA_3D_RAG_vs_No-RAG.svg"
PCA_COORDINATES_CSV = OUTPUT_DIR / "BGE-M3_PCA_3D_coordinates.csv"
DISTANCES_CSV = OUTPUT_DIR / "BGE-M3_embedding_distances.csv"

SHEET_NAME = "Sheet1"
MODEL_NAME = "BAAI/bge-m3"
BATCH_SIZE = 6
MAX_LENGTH = 1024
USE_FP16 = True

REQUIRED_COLUMNS = ["Item Number", "Scenario ID", "Answer 1", "Answer 2", "Gold Standard"]

COLOR_GOLD = "#8899A6"
COLOR_RAG = "#FF00FF"
COLOR_NO_RAG = "#FF00FF"


def load_data():
    """Load and validate the 24 paired answers from Sheet1."""
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found:{INPUT_FILE}")

    df = pd.read_excel(INPUT_FILE, sheet_name=SHEET_NAME)
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"{SHEET_NAME} Missing columns:{missing}")

    df = df[REQUIRED_COLUMNS].copy()
    for column in ["Answer 1", "Answer 2", "Gold Standard"]:
        if df[column].isna().any():
            rows = (df.index[df[column].isna()] + 2).tolist()
            raise ValueError(f"{column} inExcel OK{rows} Null value exists")
        df[column] = df[column].astype(str).str.strip()
        if df[column].eq("").any():
            rows = (df.index[df[column].eq("")] + 2).tolist()
            raise ValueError(f"{column} inExcel OK{rows} Empty text exists")

    if len(df) != 24:
        raise ValueError(f"{SHEET_NAME} should contain24 question, actually{len(df)} a")

    return df


def encode_dense(model, texts):
    """Generate normalized BGE-M3 dense embeddings."""
    output = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )
    embeddings = np.asarray(output["dense_vecs"], dtype=np.float32)

    # Normalize explicitly so Euclidean distance is comparable to cosine distance.
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("The model generated a zero-norm vector and the distance could not be calculated")
    return embeddings / norms


def calculate_embedding_distances(rag_embeddings, no_rag_embeddings, gold_embeddings):
    """Calculate paired Euclidean distances in the original embedding space."""
    rag_distances = np.linalg.norm(rag_embeddings - gold_embeddings, axis=1)
    no_rag_distances = np.linalg.norm(no_rag_embeddings - gold_embeddings, axis=1)
    return rag_distances, no_rag_distances


def reduce_to_shared_3d(rag_embeddings, no_rag_embeddings, gold_embeddings):
    """Fit one PCA model to all embeddings and return shared 3D coordinates."""
    all_embeddings = np.vstack(
        [rag_embeddings, no_rag_embeddings, gold_embeddings]
    )
    pca = PCA(n_components=3)
    all_reduced = pca.fit_transform(all_embeddings)

    n_samples = len(rag_embeddings)
    rag_3d = all_reduced[:n_samples]
    no_rag_3d = all_reduced[n_samples : 2 * n_samples]
    gold_3d = all_reduced[2 * n_samples :]
    return rag_3d, no_rag_3d, gold_3d, pca.explained_variance_ratio_


def get_text_offset(index, dx, dy, dz, mode):
    """Use alternating offsets to reduce overlap among point labels."""
    gold_patterns = [
        (dx, dy, dz),
        (-dx, dy, dz),
        (dx, -dy, dz),
        (-dx, -dy, dz),
    ]
    prediction_patterns = [
        (dx, dy, -dz),
        (-dx, dy, -dz),
        (dx, -dy, -dz),
        (-dx, -dy, -dz),
    ]
    patterns = gold_patterns if mode == "gold" else prediction_patterns
    return patterns[index % len(patterns)]


def plot_labeled_subplot(
    ax,
    predicted_3d,
    predicted_label,
    predicted_color,
    gold_3d,
    point_labels,
    title,
    average_distance,
    axis_limits,
    spans,
):
    """Draw one paired 3D embedding panel."""
    x_limits, y_limits, z_limits = axis_limits
    x_span, y_span, z_span = spans

    ax.scatter(
        gold_3d[:, 0],
        gold_3d[:, 1],
        gold_3d[:, 2],
        c=COLOR_GOLD,
        marker="o",
        s=42,
        alpha=0.88,
        edgecolors="white",
        linewidth=0.5,
        label="Gold standard",
        depthshade=False,
    )
    ax.scatter(
        predicted_3d[:, 0],
        predicted_3d[:, 1],
        predicted_3d[:, 2],
        c=predicted_color,
        marker="^",
        s=42,
        alpha=0.88,
        edgecolors="white",
        linewidth=0.5,
        label=predicted_label,
        depthshade=False,
    )

    dx = x_span * 0.024 if x_span > 0 else 0.05
    dy = y_span * 0.024 if y_span > 0 else 0.05
    dz = z_span * 0.036 if z_span > 0 else 0.05

    for index, label in enumerate(point_labels):
        ax.plot(
            [predicted_3d[index, 0], gold_3d[index, 0]],
            [predicted_3d[index, 1], gold_3d[index, 1]],
            [predicted_3d[index, 2], gold_3d[index, 2]],
            color="0.55",
            linestyle="--",
            linewidth=0.55,
            alpha=0.38,
            zorder=1,
        )

        gx, gy, gz = get_text_offset(index, dx, dy, dz, mode="gold")
        ax.text(
            gold_3d[index, 0] + gx,
            gold_3d[index, 1] + gy,
            gold_3d[index, 2] + gz,
            str(label),
            fontsize=6.2,
            color=COLOR_GOLD,
            ha="center",
            va="center",
            fontweight="bold",
        )

        px, py, pz = get_text_offset(index, dx, dy, dz, mode="prediction")
        ax.text(
            predicted_3d[index, 0] + px,
            predicted_3d[index, 1] + py,
            predicted_3d[index, 2] + pz,
            str(label),
            fontsize=6.2,
            color=predicted_color,
            ha="center",
            va="center",
            fontweight="bold",
        )

    ax.set_xlabel("Dimension 1", fontsize=9.5, fontweight="bold", labelpad=6)
    ax.set_ylabel("Dimension 2", fontsize=9.5, fontweight="bold", labelpad=6)
    ax.set_zlabel("Dimension 3", fontsize=9.5, fontweight="bold", labelpad=7)

    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    ax.set_zlim(*z_limits)
    ax.tick_params(axis="x", labelsize=7, pad=0)
    ax.tick_params(axis="y", labelsize=7, pad=0)
    ax.tick_params(axis="z", labelsize=7, pad=0)
    ax.view_init(elev=20, azim=130)
    ax.set_box_aspect((1, 1, 0.9))

    ax.text2D(
        0.02,
        0.98,
        title,
        transform=ax.transAxes,
        fontsize=9.5,
        fontweight="bold",
        ha="left",
        va="top",
    )
    ax.text2D(
        0.02,
        0.90,
        f"Avg. Dist = {average_distance:.3f}",
        transform=ax.transAxes,
        fontsize=8.5,
        fontweight="bold",
        ha="left",
        va="top",
    )

    ax.legend(
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
        fontsize=7.5,
        frameon=True,
        borderpad=0.3,
        handletextpad=0.4,
    )
    ax.grid(True, linestyle="--", alpha=0.45)

    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor("0.88")


def save_analysis_tables(
    df,
    rag_3d,
    no_rag_3d,
    gold_3d,
    rag_distances,
    no_rag_distances,
    explained_variance_ratio,
):
    """Save coordinates and original-space distances for reproducibility."""
    coordinate_frames = []
    for condition, coordinates in [
        ("RAG", rag_3d),
        ("No-RAG", no_rag_3d),
        ("Gold standard", gold_3d),
    ]:
        coordinate_frames.append(
            pd.DataFrame(
                {
                    "Item Number": df["Item Number"].to_numpy(),
                    "Scenario ID": df["Scenario ID"].to_numpy(),
                    "condition": condition,
                    "Dimension 1": coordinates[:, 0],
                    "Dimension 2": coordinates[:, 1],
                    "Dimension 3": coordinates[:, 2],
                }
            )
        )
    coordinates_df = pd.concat(coordinate_frames, ignore_index=True)
    coordinates_df.to_csv(PCA_COORDINATES_CSV, index=False, encoding="utf-8-sig")

    distance_df = pd.DataFrame(
        {
            "Item Number": df["Item Number"],
            "Scenario ID": df["Scenario ID"],
            "RAG-GoldOriginal embedding Euclidean distance": rag_distances,
            "No-RAG-GoldOriginal embedding Euclidean distance": no_rag_distances,
            "distance difference(No-RAG-RAG)": no_rag_distances - rag_distances,
        }
    )
    distance_df.loc[len(distance_df)] = {
        "Item Number": "Mean",
        "Scenario ID": "All questions",
        "RAG-GoldOriginal embedding Euclidean distance": float(np.mean(rag_distances)),
        "No-RAG-GoldOriginal embedding Euclidean distance": float(np.mean(no_rag_distances)),
        "distance difference(No-RAG-RAG)": float(
            np.mean(no_rag_distances) - np.mean(rag_distances)
        ),
    }
    for index, ratio in enumerate(explained_variance_ratio, start=1):
        distance_df[f"PCA Dimension {index} explained variance ratio"] = np.nan
        distance_df.loc[0, f"PCA Dimension {index} explained variance ratio"] = ratio
    distance_df.to_csv(DISTANCES_CSV, index=False, encoding="utf-8-sig")


def create_figure(
    rag_3d,
    no_rag_3d,
    gold_3d,
    point_labels,
    mean_rag_distance,
    mean_no_rag_distance,
):
    """Create and save the two-panel 3D PCA figure."""
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["svg.fonttype"] = "none"

    all_data = np.vstack([gold_3d, rag_3d, no_rag_3d])
    minima = np.min(all_data, axis=0)
    maxima = np.max(all_data, axis=0)
    spans = maxima - minima
    padding = np.where(spans > 0, spans * np.array([0.08, 0.08, 0.10]), 0.5)
    axis_limits = tuple(
        (float(minima[index] - padding[index]), float(maxima[index] + padding[index]))
        for index in range(3)
    )

    fig = plt.figure(figsize=(9.0, 4.2), dpi=600)
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    plot_labeled_subplot(
        ax=ax1,
        predicted_3d=rag_3d,
        predicted_label="RAG",
        predicted_color=COLOR_RAG,
        gold_3d=gold_3d,
        point_labels=point_labels,
        title="(a) RAG vs. Gold standard",
        average_distance=mean_rag_distance,
        axis_limits=axis_limits,
        spans=spans,
    )

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    plot_labeled_subplot(
        ax=ax2,
        predicted_3d=no_rag_3d,
        predicted_label="No-RAG",
        predicted_color=COLOR_NO_RAG,
        gold_3d=gold_3d,
        point_labels=point_labels,
        title="(b) No-RAG vs. Gold standard",
        average_distance=mean_no_rag_distance,
        axis_limits=axis_limits,
        spans=spans,
    )

    # Extra center spacing keeps the z-axis label of panel (a) unobstructed.
    fig.subplots_adjust(left=0.03, right=0.985, bottom=0.06, top=0.97, wspace=0.22)
    fig.savefig(
        FIGURE_PNG,
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    fig.savefig(
        FIGURE_PDF,
        dpi=600,
        bbox_inches="tight",
        format="pdf",
        facecolor="white",
        edgecolor="none",
    )
    fig.savefig(
        FIGURE_SVG,
        dpi=600,
        bbox_inches="tight",
        format="svg",
        facecolor="white",
        edgecolor="none",
    )
    plt.close(fig)


def main():
    df = load_data()

    model = BGEM3FlagModel(MODEL_NAME, use_fp16=USE_FP16)
    rag_embeddings = encode_dense(model, df["Answer 1"].tolist())
    no_rag_embeddings = encode_dense(model, df["Answer 2"].tolist())
    gold_embeddings = encode_dense(model, df["Gold Standard"].tolist())

    rag_distances, no_rag_distances = calculate_embedding_distances(
        rag_embeddings,
        no_rag_embeddings,
        gold_embeddings,
    )
    rag_3d, no_rag_3d, gold_3d, explained_variance_ratio = reduce_to_shared_3d(
        rag_embeddings,
        no_rag_embeddings,
        gold_embeddings,
    )

    save_analysis_tables(
        df,
        rag_3d,
        no_rag_3d,
        gold_3d,
        rag_distances,
        no_rag_distances,
        explained_variance_ratio,
    )
    create_figure(
        rag_3d,
        no_rag_3d,
        gold_3d,
        point_labels=df["Item Number"].tolist(),
        mean_rag_distance=float(np.mean(rag_distances)),
        mean_no_rag_distance=float(np.mean(no_rag_distances)),
    )

    print(f"Input file:{INPUT_FILE}")
    print(f"Use a worksheet:{SHEET_NAME}")
    print(f"Model:{MODEL_NAME}")
    print(f"RAG vs. Gold Average distance:{np.mean(rag_distances):.6f}")
    print(f"No-RAG vs. Gold Average distance:{np.mean(no_rag_distances):.6f}")
    print(f"Output folder:{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
