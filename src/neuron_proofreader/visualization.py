"""
Created on Sat Sep 30 10:00:00 2025

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code for visualizing SkeletonGraphs.

"""

import numpy as np
import plotly.graph_objects as go


def visualize(graph):
    """
    Visualizes the given graph using Plotly.

    Parameters
    ----------
    graph : SkeletonGraph
        Graph to be visualized.
    """
    # Initializations
    data = get_edge_trace(graph)
    layout = get_layout()

    # Generate plot
    fig = go.Figure(data=data, layout=layout)
    fig.show()


def visualize_proposals(graph, gt_graph, proposals=list()):
    """
    Visualizes a graph along with its proposals.

    Parameters
    ----------
    graph : networkx.Graph
        Graph to be visualized.
    groundtruth_graph : networkx.Graph, optional
        Graph generated from groundtruth tracings. Default is None.
    proposals : List[Frozenset[int]], optional
        List of proposals to visualize. Default is an empty list.
    """
    # Generate traces
    data = list()
    proposals = proposals or graph.list_proposals()
    data = [get_edge_trace(graph, color="black")]
    data.extend(get_proposal_traces(graph, proposals))

    # Plot traces
    layout = get_layout()
    fig = go.Figure(data=data, layout=layout)
    fig.show()


def get_edge_trace(graph, color="blue", name=""):
    """
    Generates a 3D edge trace for visualizing the edges of a graph.

    Parameters
    ----------
    graph : SkeletonGraph
        Graph to be visualized.
    color : str, optional
        Color to use for the edge lines in the plot. Default is "blue".
    name : str, optional
        Name of the edge trace. Default is an empty string.

    Returns
    -------
    edge_trace : plotly.graph_objects.Scatter3d
        Scatter3d object that represents the 3D trace of the graph edges.
    """
    # Build coordinate lists
    x, y, z = list(), list(), list()
    for i, j in graph.edges():
        x0, y0, z0 = graph.node_xyz[i]
        x1, y1, z1 = graph.node_xyz[j]
        x.extend([x0, x1, None])
        y.extend([y0, y1, None])
        z.extend([z0, z1, None])

    # Set edge trace
    edge_trace = go.Scatter3d(
        x=x, y=y, z=z, mode="lines", line=dict(color=color, width=3), name=name
    )
    return edge_trace


def get_proposal_traces(graph, proposals):
    traces = []
    for proposal in map(tuple, proposals):
        xyz = graph.node_xyz[np.array(proposal)]
        trace = go.Scatter3d(
            x=xyz[:, 0],
            y=xyz[:, 1],
            z=xyz[:, 2],
            mode="lines",
            line=dict(width=5),
            name=str(proposal),
        )
        traces.append(trace)
    return traces


# --- Helpers ---
def get_layout():
    """
    Generates the layout for a 3D plot using Plotly.

    Returns
    -------
    plotly.graph_objects.Layout
        Layout object that defines the appearance and properties of the plot.
    """
    layout = go.Layout(
        scene=dict(aspectmode="manual", aspectratio=dict(x=1, y=1, z=1)),
        showlegend=True,
        template="plotly_white",
        height=700,
        width=1200,
    )
    return layout
