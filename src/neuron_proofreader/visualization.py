"""
Created on Sat Sep 30 10:00:00 2025

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code for visualizing SkeletonGraphs.

"""

import networkx as nx
import numpy as np
import plotly.colors as plc
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


def visualize_proposals(graph, gt_graph=None, proposals=list()):
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
    # Initializations
    proposals = proposals or graph.list_proposals()
    gt_graph = gt_graph or nx.Graph

    # Generate traces
    data = [get_edge_trace(graph, color="black")]
    data.append(get_node_trace(graph, graph.leaf_nodes()))
    data.append(get_node_trace(graph, graph.branching_nodes()))
    data.extend(get_component_traces(gt_graph))
    data.extend(get_proposal_traces(graph, proposals))

    # Plot traces
    layout = get_layout()
    fig = go.Figure(data=data, layout=layout)
    fig.show()


def get_component_traces(graph, use_color=True):
    """
    Generates edge traces to visualize the connected components of a graph.

    Parameters
    ----------
    graph : SkeletonGraph
        Graph to be visualized.
    """
    colors = plc.qualitative.Bold
    traces = list()
    for nodes in map(list, nx.connected_components(graph)):
        # Extract data
        color = colors[len(traces) % len(colors)] if use_color else "black"
        name = graph.node_swc_id(nodes[0])
        subgraph = graph.subgraph(nodes)
        edges = subgraph.edges

        # Create trace
        traces.append(
            get_edge_trace(graph, color=color, edges=edges, name=name)
        )
    return traces


def get_edge_trace(graph, color="blue", edges=list(), name=None):
    """
    Gets the edge traces for visualizing a graph.

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
    edges = edges or graph.edges()
    x, y, z = list(), list(), list()
    for i, j in edges:
        x0, y0, z0 = graph.node_xyz[i]
        x1, y1, z1 = graph.node_xyz[j]
        x.extend([x0, x1, None])
        y.extend([y0, y1, None])
        z.extend([z0, z1, None])

    # Set edge trace
    line = dict(color=color, width=3)
    trace = go.Scatter3d(x=x, y=y, z=z, mode="lines", line=line, name=name)
    return trace


def get_node_trace(graph, nodes, color="black"):
    """
    Gets a scatter plot trace for the given nodes.

    Parameters
    ----------
    graph : SkeletonGraph
        Graph containing the given nodes.
    nodes : List[int]
        Nodes to be visualized.
    color : str, optional
        Color that nodes are plotted with. Default is "black".

    Returns
    -------
    trace : go.Scatter3d
        Scatter plot trace for the given nodes.
    """
    nodes = np.array(nodes)
    trace = go.Scatter3d(
        x=graph.node_xyz[nodes, 0],
        y=graph.node_xyz[nodes, 1],
        z=graph.node_xyz[nodes, 2],
        mode="markers",
        marker=dict(size=1.5, color=color),
        text=[str(int(n)) for n in nodes],
        hovertemplate="node: %{text}<extra></extra>",
    )
    return trace


def get_proposal_traces(graph, proposals):
    """
    Gets scatter plot traces for the given proposals.

    Parameters
    ----------
    graph : ProposalGraph
        Graph containing the given proposals.
    proposals : List[Frozenset[int]]
        Proposals to be visualized.

    Returns
    -------
    traces : List[go.Scatter3d]
        Scatter plots of the given proposals.
    """
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
