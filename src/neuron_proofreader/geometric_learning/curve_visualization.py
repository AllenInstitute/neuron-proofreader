"""
Created on Mon June 8 17:00:00 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code for visualizing 3D space curves and their embeddings.

"""

from colorsys import hsv_to_rgb
from matplotlib.colors import LogNorm
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

from neuron_proofreader.utils import geometry_util


# --- Plot Curves ---
def plot_curves(curve1, curve2, name1=None, name2=None):
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=curve1[:, 0],
                y=curve1[:, 1],
                z=curve1[:, 2],
                mode="lines",
                name=name1,
                line=dict(width=5, color="blue"),
            ),
            go.Scatter3d(
                x=curve2[:, 0],
                y=curve2[:, 1],
                z=curve2[:, 2],
                mode="lines",
                name=name2,
                line=dict(width=5, color="green"),
            ),
            go.Scatter3d(
                x=[0],
                y=[0],
                z=[0],
                mode="markers",
                name="Origin",
                marker=dict(size=3, color="red"),
            ),
        ]
    )
    fig.update_layout(
        scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z")
    )
    fig.show()


def plot_length_distribution(dataset_collection, title=None, output_path=None):
    # Compute path length stats
    lengths = dataset_collection.examples_df["length"]
    p50 = round(np.percentile(lengths, 50), 2)
    p99 = round(np.percentile(lengths, 99.9), 2)
    thr_lengths = [l for l in lengths if l < p99]

    # Plot path lengths
    plt.figure(figsize=(8, 5))
    plt.hist(thr_lengths, bins=50, edgecolor="white", linewidth=0.5, zorder=2)
    add_line(p50, color="r", label=f"50th perc = {p50}")
    add_line(p99, color="g", label=f"99.9th perc = {p99}")

    # Plot labels
    plt.grid(axis="y", color="lightgrey", linewidth=0.5)
    plt.legend(loc="upper right")
    plt.title(title, fontsize=13)
    plt.xlabel("Path Length (μm)", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.yscale("log")

    plt.figtext(
        0.01,
        -0.02,
        "Note: Path lengths thresholded at 99.9th percentile in this plot",
        fontsize=9,
        color="tab:grey",
        va="bottom",
    )
    plt.subplots_adjust(bottom=0.8)
    visualize_result(output_path=output_path)


# --- Plot Curve Embeddings ---
def plot_error_vs_length(lengths, rmse_results, output_path=None):
    # Set colors
    norm = LogNorm(vmin=lengths.min(), vmax=lengths.max())
    colors = plt.cm.viridis(norm(lengths))

    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(lengths, rmse_results, c=colors, s=10, alpha=0.8)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Path Length (μm)", fontsize=12)
    plt.ylabel("RMSE", fontsize=12)
    visualize_result(output_path=output_path)


def plot_latents_by_pca(curves, latents, output_path=None):
    # Set output paths
    if output_path:
        dir_output_path = output_path.replace("pca", "pca_direction")
        len_output_path = output_path.replace("pca", "pca_len")
    else:
        dir_output_path = None
        len_output_path = None

    # PCA of latents
    pca = PCA(n_components=2)
    latents_2d = pca.fit_transform(latents)
    lengths = np.array([geometry_util.compute_length(c) for c in curves])

    # Visualize results
    _plot_latents_by_direction(curves, latents_2d, pca, dir_output_path)
    _plot_latents_by_length(lengths, latents_2d, pca, len_output_path)


def _plot_latents_by_direction(curves, latents_2d, pca, output_path=None):
    # Compute directions and colors for each curve
    directions = np.array([curve_principal_direction(c) for c in curves])
    colors = np.array([direction_to_color(d) for d in directions])

    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(latents_2d[:, 0], latents_2d[:, 1], c=colors, s=10, alpha=0.8)
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    plt.title("Curve Embeddings Colored by Principal Direction")
    visualize_result(output_path=output_path)


def _plot_latents_by_length(lengths, latents_2d, pca, output_path=None):
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(
        latents_2d[:, 0],
        latents_2d[:, 1],
        c=lengths,
        cmap="viridis",
        s=10,
        alpha=0.7,
        norm=LogNorm(vmin=lengths.min(), vmax=lengths.max()),
    )
    plt.colorbar(sc, label="Path length (μm)")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    plt.title("PCA of curve embeddings")
    visualize_result(output_path=output_path)


# --- Helpers ---
def add_line(p, color=None, label=None):
    plt.axvline(p, color=color, linestyle="--", label=label, zorder=2)


def curve_principal_direction(curve):
    curve_pca = PCA(n_components=1)
    curve_pca.fit(curve)
    direction = curve_pca.components_[0]
    if direction[2] < 0:
        direction = -direction
    return direction / np.linalg.norm(direction)


def direction_to_color(direction):
    x, y, z = direction
    azimuth = np.arctan2(y, x)
    hue = (azimuth + np.pi) / (2 * np.pi)
    polar = np.arccos(np.clip(z, 0, 1))
    saturation = polar / (np.pi / 2)
    value = 1.0
    return hsv_to_rgb(hue, saturation, value)


def visualize_result(output_path=None):
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()
