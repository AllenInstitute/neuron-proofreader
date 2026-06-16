"""
Created on Wed June 5 16:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code that loads and preprocesses neuron fragments stored as SWC files, then
constructs a custom graph object called a "FragmentsGraph".

"""

from collections import deque
from concurrent.futures import (
    FIRST_COMPLETED,
    ProcessPoolExecutor,
    wait,
)
from tqdm import tqdm

import networkx as nx
import numpy as np

from neuron_proofreader.utils import geometry_util as geometry, swc_util


class GraphLoader:
    """
    Class that loads SWC files and constructs a FragmentsGraph instance from
    the data.
    """

    def __init__(
        self,
        anisotropy=(1.0, 1.0, 1.0),
        min_cable_length=40.0,
        min_swc_pts=1,
        node_spacing=1,
        prefetch=128,
        prune_depth=24.0,
        verbose=False,
    ):
        """
        Builds a FragmentsGraph by reading swc files stored on either the
        cloud or local machine, then extracting the irreducible components.

        Parameters
        ----------
        anisotropy : Tuple[float], optional
            Image to physical coordinates scaling factors to account for the
            anisotropy of the microscope. Default is (1.0, 1.0, 1.0).
        min_cable_length : float, optional
            Minimum cable length (in microns) of SWC files that are loaded.
            Default is 40.
        min_swc_pts : int, optional
            ...
        node_spacing : int, optional
            Spacing (in microns) between neighboring nodes. Default is 1.
        prefetch : int, optional
            Number of jobs to prefetch. Default is 128.
        prune_depth : int, optional
            Branches with length less than "prune_depth" microns are pruned.
            Default is 24.
        verbose : bool, optional
            Indication of whether to display a progress bar while building
            FragmentsGraph. Default is True.
        """
        # Instance attributes
        self.min_cable_length = min_cable_length
        self.min_swc_pts = min_swc_pts
        self.node_spacing = node_spacing
        self.prefetch = prefetch
        self.prune_depth = prune_depth
        self.swc_reader = swc_util.Reader(anisotropy, min_swc_pts, verbose)
        self.verbose = verbose

    def __call__(self, swc_pointer):
        """
        Processes a list of SWC dictionaries in parallel and loads the
        components of the irreducible subgraph from each.

        Parameters
        ----------
        swc_pointer : str
            Path to SWC files to be read.

        Returns
        -------
        irreducibles : List[dict]
            Dictionaries containing components of the irreducible subgraph
            loaded from SWC files.
        """
        # Read SWC files
        swc_dicts = self.swc_reader(swc_pointer)
        swc_dicts = deque(
            d for d in swc_dicts if len(d["xyz"]) > self.min_swc_pts
        )
        if self.verbose:
            pbar = tqdm(total=len(swc_dicts), desc="Load Graphs")

        # Load graphs
        pending = set()
        with ProcessPoolExecutor() as executor:
            # Start processes
            while len(pending) < self.prefetch and swc_dicts:
                pending.add(executor.submit(self.load, swc_dicts.pop()))

            # Yield processes
            irreducibles = deque()
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    # Store completed processes
                    result = future.result()
                    if result:
                        irreducibles.append(result)

                    # Update progress bar
                    if self.verbose:
                        pbar.update(1)

                    # Continue submitting processes
                    if swc_dicts:
                        pending.add(
                            executor.submit(self.load, swc_dicts.pop())
                        )
        return irreducibles

    def load(self, swc_dict):
        """
        Loads irreducible components from "swc_dict", which is assumed to
        contain exactly one connected component.

        Parameters
        ----------
        swc_dict : dict
            Contents of an SWC file.

        Returns
        -------
        dict
            Dictionary that each contains the components of an irreducible
            subgraph.
        """
        # Build graph
        graph = swc_util.to_graph(swc_dict)
        prune_branches(graph, self.prune_depth)

        # Extract irreducible components (if applicable)
        irreducibles = self.get_irreducibles(graph)
        if irreducibles:
            irreducibles["is_soma"] = len(swc_dict["soma_nodes"]) > 0
            irreducibles["swc_id"] = swc_dict["swc_name"]
            return irreducibles
        else:
            return None

    def get_irreducibles(self, graph):
        """
        Identifies irreducible components of a connected graph.

        Parameters
        ----------
        graph : networkx.Graph
            Graph to be searched.

        Returns
        -------
        irreducibles : dict
            Dictionary containing the irreducible components of a connected
            graph.
        """
        # Initializations
        leaf = find_leaf(graph)
        irr_nodes = {leaf}
        irr_edges = dict()

        radius = graph.graph["radius"]
        xyz = graph.graph["xyz"]

        # Main
        root, cable_length = None, 0
        for i, j in nx.dfs_edges(graph, source=leaf):
            # Check for start of irreducible edge
            if root is None:
                root, edge_length = i, 0
                attrs = {"radius": [radius[i]], "xyz": [xyz[i]]}

            # Visit node
            edge_length += np.linalg.norm(xyz[i] - xyz[j])
            attrs["radius"].append(radius[j])
            attrs["xyz"].append(xyz[j])

            # Check for end of irreducible edge
            if graph.degree[j] != 2:
                cable_length += edge_length
                irr_nodes.add(j)

                attrs = to_numpy(attrs)
                n_pts = int(edge_length / self.node_spacing)
                self.resample_curve_3d(graph, attrs, (root, j), n_pts)

                irr_edges[(root, j)] = attrs
                root = None

        # Check for curvy line fragment
        if len(irr_nodes) == 2:
            t0, t1 = irr_nodes
            endpoint_dist = np.linalg.norm(xyz[t0] - xyz[t1])
            if endpoint_dist / cable_length < 0.5:
                return None

        # Store results
        if cable_length >= self.min_cable_length:
            irreducibles = {
                "nodes": set_node_attrs(graph, irr_nodes),
                "edges": set_edge_attrs(graph, irr_edges),
            }
        else:
            irreducibles = None
        return irreducibles

    # --- Helpers ---
    def resample_curve_3d(self, graph, attrs, edge, n_pts):
        """
        Smooths a 3D curve and update the corresponding edge endpoints in the
        graph.

        Parameters
        ----------
        graph : networkx.Graph
            Graph to be updated.
        attrs : dict
            Dictionary containing "xyz" (list of 3D points) and "radius" (list
            of scalars) representing the edge to be smoothed.
        edge : Tuple[int]
            Start and end node IDs of the edge.
        n_pts : int
            Number of points to use for the smoothed curve.
        """
        attrs["xyz"] = geometry.resample_curve_3d(attrs["xyz"], n_pts=n_pts)
        attrs["radius"] = geometry.resample_curve_1d(attrs["radius"], n_pts)
        graph.graph["xyz"][edge[0]] = attrs["xyz"][0]
        graph.graph["xyz"][edge[1]] = attrs["xyz"][-1]


# --- Helpers ---
def set_node_attrs(graph, nodes):
    """
    Extracts attributes for each node in the graph.

    Parameters
    ----------
    graph : networkx.Graph
        Graph that contains "nodes".
    nodes : List[int]
        Nodes whose attributes are to be extracted from the graph.

    Returns
    -------
    attrs : dict
        Dictionary where the keys are node ids and values are dictionaries
        containing the "radius" and "xyz" attributes of the nodes.
    """
    xyz, radius = graph.graph["xyz"], graph.graph["radius"]
    return {i: {"radius": radius[i], "xyz": xyz[i]} for i in nodes}


def set_edge_attrs(graph, attrs):
    """
    Sets the edge attributes of a given graph by updating node coordinates and
    resamples points in irreducible path.

    Parameters
    ----------
    graph : networkx.Graph
        Graph that attributes dictionary was built from.
    attrs : dict
        Dictionary where the keys are irreducible edge IDs and values are the
        corresponding attribute dictionaries.

    Returns
    -------
    attrs : dict
        Updated edge attribute dictionary.
    """
    for e in attrs:
        i, j = e
        attrs[e]["xyz"][0] = graph.graph["xyz"][i]
        attrs[e]["xyz"][-1] = graph.graph["xyz"][j]
    return attrs


# --- Miscellaneous ---
def count_nodes(irr_list):
    n = 0
    for irr in irr_list:
        n += len(irr["nodes"])
        for attrs in irr["edges"].values():
            n += len(attrs["xyz"]) - 2
    return n


def cycle_exists(graph):
    """
    Checks if the given graph has a cycle.

    Paramaters
    ----------
    graph : networkx.Graph
        Graph to be searched.

    Returns
    -------
    bool
        Indication of whether graph has a cycle.
    """
    try:
        nx.find_cycle(graph)
        return True
    except nx.exception.NetworkXNoCycle:
        return False


def edges_to_line_graph(edges):
    """
    Initializes a line graph from a list of edges.

    Parameters
    ----------
    edges : List[Tuple[int]]
        List of edges.

    Returns
    -------
    graph: networkx.Graph
        Line graph generated from a list of edges.
    """
    graph = nx.Graph()
    graph.add_edges_from(edges)
    return nx.line_graph(graph)


def find_leaf(graph):
    """
    Finds a leaf node in the given graph.

    Parameters
    ----------
    graph : networkx.Graph
        Graph to be searched.

    Returns
    -------
    i : int
        Leaf node.
    """
    for i in graph.nodes:
        if graph.degree[i] == 1:
            return i
    return None


def prune_branches(graph, depth):
    """
    Prunes paths between leaf and branching nodes with cable length less than
    "depth" microns.

    Parameters
    ----------
    graph : networkx.Graph
        Graph to be searched.
    depth : float
        Length of branches that are pruned.
    """
    xyz = graph.graph["xyz"]
    changed = True
    while changed:
        changed = False
        for leaf in [i for i in graph.nodes if graph.degree[i] == 1]:
            branch, length = [leaf], 0
            for i, j in nx.dfs_edges(graph, source=leaf):
                length += np.linalg.norm(xyz[i] - xyz[j])
                if length > depth:
                    break
                if graph.degree(j) == 2:
                    branch.append(j)
                elif graph.degree(j) > 2:
                    graph.remove_nodes_from(branch)
                    changed = True
                    break


def to_numpy(attrs):
    """
    Converts edge attributes from a list to NumPy array.

    Parameters
    ----------
    attrs : dict
        Edge attribute dictionary.

    Returns
    -------
    attrs : dict
        Updated edge attribute dictionary.
    """
    attrs["xyz"] = np.array(attrs["xyz"], dtype=np.float32)
    attrs["radius"] = np.array(attrs["radius"], dtype=np.float16)
    return attrs
