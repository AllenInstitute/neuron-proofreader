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
    """
    Plots two 3D curves using Plotly.

    Parameters
    ----------
    curve1 : numpy.ndarray
        Array with shape (N, 3) containing coordinates of the first curve.
    curve2 : numpy.ndarray
        Array with shape (M, 3) containing coordinates of the second curve.
    name1 : str, optional
        Label for the first curve in the plot legend. Default is None.
    name2 : str, optional
        Label for the second curve in the plot legend. Default is None.
    """
    pt = np.zeros((1, 3))
    fig = go.Figure(
        data=[
            create_scatter3d(curve1, color="blue", name=name1),
            create_scatter3d(curve2, color="green", name=name2),
            create_scatter3d(pt, color="red", mode="markers", name="Origin"),
        ]
    )
    fig.update_layout(
        scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z")
    )
    fig.show()


def plot_length_distribution(dataset, output_path=None, title=None):
    """
    Plots the distribution of path lengths in a dataset.

    Parameters
    ----------
    dataset : PathsDataset
        Dataset containing an "examples_df" attribute with a "length" column
        specifying the path lengths.
    output_path : str, optional
        If provided, the figure is saved to this location. Otherwise, it is
        displayed. Default is None.
    title : str, optional
        Title of the plot. Default is None.
    """
    # Compute path length stats
    lengths = dataset.examples_df["length"].to_numpy()
    p50, p99 = np.percentile(lengths, [50, 99.9])
    lengths = lengths[lengths <= p99]

    # Plot path lengths
    plt.figure(figsize=(8, 5))
    plt.hist(lengths, bins=50, edgecolor="white", linewidth=0.5, zorder=2)
    add_line(p50, color="r", label=f"50th perc = {p50:.2f}")
    add_line(p99, color="g", label=f"99.9th perc = {p99:.2f}")

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
    """
    Plots reconstruction error as a function of path length.

    ----------
    lengths : numpy.ndarray
        One-dimensional array of path lengths in mircons.
    rmse_results : numpy.ndarray
        One-dimensional array of root mean squared errors in mircons.
    output_path : str, optional
        If provided, the figure is saved to this location. Otherwise, it is
        displayed. Default is None.
    """
    # Set colors
    norm = LogNorm(vmin=lengths.min(), vmax=lengths.max())
    colors = plt.cm.viridis(norm(lengths))

    # Plot
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(lengths, rmse_results, c=colors, s=10, alpha=0.8)
    plt.colorbar(sc, label="Path Length (μm)")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Path Length (μm)", fontsize=12)
    plt.ylabel("RMSE (μm)", fontsize=12)
    visualize_result(output_path=output_path)


def plot_latents_by_pca(curves, latents, output_path=None):
    """
    Visualizes latent representations using PCA.

    Parameters
    ----------
    curves : List[numpy.ndarray]
        Curves such that each is an array with shape (N, 3).
    latents : numpy.ndarray
        Array of latent representations with shape (N, latent_dim).
    output_path : str, optional
        Base path for saving the figures. If provided, the figure is saved to
        this location. Otherwise, it is displayed. Default is None.
    """
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
    """
    Plots 2D latent colored by the principal direction of each curve.

    Parameters
    ----------
    curves : List[numpy.ndarray]
        Curves such that each is an array with shape (N, 3).
    latents_2d : numpy.ndarray
        Latent representations with shape (N, 2).
    pca : sklearn.decomposition.PCA
        Fitted PCA object used to generate "latents_2d".
    output_path : str, optional
        If provided, the figure is saved to this location. Otherwise, it is
        displayed. Default is None.
    """
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
    """
    Plots 2D latents colored by the length of each curve.

    Parameters
    ----------
    curves : List[numpy.ndarray]
        Curves such that each is an array with shape (N, 3).
    latents_2d : numpy.ndarray
        Latent representations with shape (N, 2).
    pca : sklearn.decomposition.PCA
        Fitted PCA object used to generate "latents_2d".
    output_path : str, optional
        If provided, the figure is saved to this location. Otherwise, it is
        displayed. Default is None.
    """
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
    plt.colorbar(sc, label="Path Length (μm)")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    plt.title("Curve Embeddings Colored by Path Length")
    visualize_result(output_path=output_path)


# --- Helpers ---
def add_line(x, color=None, label=None):
    """
    Adds a vertical reference line to the current Matplotlib axes.

    Parameters
    ----------
    x : float
        X-coordinate at which to draw the vertical line.
    color : str, optional
        Color of the line. Default is None.
    label : str, optional
        Label for the line displayed in the plot legend. Default is None.
    """
    plt.axvline(x, color=color, linestyle="--", label=label, zorder=2)


def create_scatter3d(pts, color=None, mode="lines", name=None, width=5):
    """
    Creates a Plotly 3D scatter trace.

    Parameters
    ----------
    pts : numpy.ndarray
        Array with shape (N, 3) containing the 3D coordinates to plot.
    color : str, optional
        Color of the line or markers. Default is None.
    mode : str, optional
        Rendering mode for the trace. Default is "lines".
    name : str, optional
        Name of the trace displayed in the plot legend. Default is None.
    width : float, optional
        Width of object to be plotted. Default is 5.

    Returns
    -------
    plotly.graph_objects.Scatter3d
        A Plotly 3D scatter trace.
    """
    # Create object dict
    if mode == "lines":
        line = dict(width=width, color=color)
        marker = None
    else:
        line = None
        marker = dict(size=width, color=color)

    # Create scatter plot
    return go.Scatter3d(
        x=pts[:, 0],
        y=pts[:, 1],
        z=pts[:, 2],
        mode=mode,
        name=name,
        line=line,
        marker=marker,
    )


def curve_principal_direction(curve):
    """
    Computes the principal direction of a 3D curve using PCA.

    Parameters
    ----------
    curve : numpy.ndarray
        Array with shape (N, 3) containing the 3D coordinates of the curve.

    Returns
    -------
    numpy.ndarray
        Unit vector of shape (3,) representing the principal direction of the
        curve.
    """
    curve_pca = PCA(n_components=1)
    curve_pca.fit(curve)
    direction = curve_pca.components_[0]
    if direction[2] < 0:
        direction = -direction
    return direction / np.linalg.norm(direction)


def direction_to_color(direction):
    """
    Converts a 3D direction vector into an RGB color representation, where the
    azimuth angle of the vector is mapped to a hue and polar angle is mapped
    to a saturation.

    Parameters
    ----------
    direction : numpy.ndarray
        Unit vector of shape (3,) representing a 3D direction.

    Returns
    -------
    tuple
        RGB color corresponding to the input direction.
    """
    x, y, z = direction
    azimuth = np.arctan2(y, x)
    hue = (azimuth + np.pi) / (2 * np.pi)
    polar = np.arccos(np.clip(z, 0, 1))
    saturation = polar / (np.pi / 2)
    value = 1.0
    return hsv_to_rgb(hue, saturation, value)


def visualize_result(output_path=None):
    """
    Displays or saves the current Matplotlib figure.

    Parameters
    ----------
    output_path : str, optional
        If provided, the figure is saved to this location. Otherwise, it is
        displayed. Default is None.
    """
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
