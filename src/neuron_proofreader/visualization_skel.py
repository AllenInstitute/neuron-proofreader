"""
Created on Sat Sep 30 10:00:00 2025

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code for visualizing SkeletonGraphs.

"""

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


def get_edge_trace(graph, color="blue", name=""):
    """
    Generates a 3D edge trace for visualizing the edges of a graph.

    Parameters
    ----------
    graph : SkeletonGraph
        Graph to be visualized.
    color : str, optional
        Color to use for the edge lines in the plot. Default is "black".
    name : str, optional
        Name of the edge trace. Default is an empty string.

    Returns
    -------
    plotly.graph_objects.Scatter3d
        Scatter3d object that represents the 3D trace of the graph edges.
    """
    # Build coordinate lists
    x, y, z = list(), list(), list()
    for u, v in graph.edges():
        x0, y0, z0 = graph.node_xyz[u]
        x1, y1, z1 = graph.node_xyz[v]
        x.extend([x0, x1, None])
        y.extend([y0, y1, None])
        z.extend([z0, z1, None])

    # Set edge trace
    edge_trace = go.Scatter3d(
        x=x, y=y, z=z, mode="lines", line=dict(color=color, width=3), name=name
    )
    return edge_trace


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
