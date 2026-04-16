"""
Created on Wed June 5 16:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code that loads and preprocesses neuron fragments stored as SWC files, then
constructs a custom graph object called a "FragmentsGraph".

    Graph Loading Algorithm:
        1. Load Soma Locations (Optional)

        2. Extract Irreducibles from SWC files
            a. Build graph from SWC file
            b. Break soma merges (optional)
            c. Break high risk merges (optional)
            d. Find irreducible nodes
            e. Find irreducible edges


Note: We use the term "branch" to refer to a path in a graph from a branching
      node to a leaf.
"""

from collections import deque
from concurrent.futures import (
    FIRST_COMPLETED,
    ProcessPoolExecutor,
    wait,
)
from scipy.spatial.distance import euclidean
from tqdm import tqdm

import networkx as nx
import numpy as np

from neuron_proofreader.utils import geometry_util as geometry, swc_util, util


class GraphLoader:
    """
    Class that loads SWC files and constructs a FragmentsGraph instance from
    the data.
    """

    def __init__(
        self,
        anisotropy=(1.0, 1.0, 1.0),
        min_cable_length=40.0,
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
        self.node_spacing = node_spacing
        self.prefetch = prefetch
        self.prune_depth = prune_depth
        self.swc_reader = swc_util.Reader(anisotropy, min_cable_length, verbose)
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
        if self.verbose:
            pbar = tqdm(total=len(swc_dicts), desc="Load Graphs")

        # Load graphs
        with ProcessPoolExecutor() as executor:
            # Start processes
            pending = set()
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
                    pbar.update(1) if self.verbose else None

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
        graph = swc_util.to_graph(swc_dict, set_attrs=True)
        prune_branches(graph, self.prune_depth)

        # Extract irreducible components (if applicable)
        if self.satisfies_cable_length_condition(graph):
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

        def dist(i, j):
            """
            Computes distance between the given nodes.

            Parameters
            ----------
            i : int
                Node ID.
            j : int
                Node ID.

            Returns
            -------
            float
                Distance between nodes.
            """
            return euclidean(graph.graph["xyz"][i], graph.graph["xyz"][j])

        # Initializations
        leaf = find_leaf(graph)
        irreducible_nodes = {leaf}
        irreducible_edges = dict()

        # Main
        root, cable_length = None, 0
        for i, j in nx.dfs_edges(graph, source=leaf):
            # Check for start of irreducible edge
            if root is None:
                root, edge_length = i, 0
                attrs = {
                    "radius": [graph.graph["radius"][i]],
                    "xyz": [graph.graph["xyz"][i]],
                }

            # Visit node
            edge_length += dist(i, j)
            attrs["radius"].append(graph.graph["radius"][j])
            attrs["xyz"].append(graph.graph["xyz"][j])

            # Check for end of irreducible edge
            if graph.degree[j] != 2:
                cable_length += edge_length
                irreducible_nodes.add(j)

                attrs = to_numpy(attrs)
                n_pts = int(edge_length / self.node_spacing)
                self.resample_curve_3d(graph, attrs, (root, j), n_pts)

                irreducible_edges[(root, j)] = attrs
                root = None

        # Check for curvy line fragment
        if len(irreducible_nodes) == 2:
            endpoint_dist = dist(*tuple(irreducible_nodes))
            if endpoint_dist / cable_length < 0.5:
                return None

        # Store results
        if cable_length > self.min_cable_length:
            irreducibles = {
                "nodes": set_node_attrs(graph, irreducible_nodes),
                "edges": set_edge_attrs(graph, irreducible_edges),
            }
        else:
            irreducibles = None
        return irreducibles

    # --- Helpers ---
    def satisfies_cable_length_condition(self, graph):
        """
        Determines whether the cable length of the given graph is greater
        than "self.min_cable_length".

        Parameters
        ----------
        graph : networkx.Graph
            Graph to be checked.

        Returns
        -------
        bool
            Indication of whether the total cable length of the given graph is
            greater than "self.min_cable_length".
        """
        length = 0
        for i, j in nx.dfs_edges(graph):
            length += euclidean(graph.graph["xyz"][i], graph.graph["xyz"][j])
            if length > self.min_cable_length:
                return True
        return False

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
    attrs = dict()
    for i in nodes:
        attrs[i] = {
            "radius": graph.graph["radius"][i],
            "xyz": graph.graph["xyz"][i],
        }
    return attrs


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
        i, j = tuple(e)
        attrs[e]["xyz"][0] = graph.graph["xyz"][i]
        attrs[e]["xyz"][-1] = graph.graph["xyz"][j]
    return attrs


# --- Miscellaneous ---
def count_nodes(irreducibles):
    n = 0
    for irr in irreducibles:
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


def largest_components(graph, k):
    """
    Finds the "k" largest connected components in "graph".

    Parameters
    ----------
    graph : nx.Graph
        Graph to be searched.
    k : int
        Number of largest connected components to return.

    Returns
    -------
    node_ids : List[int]
        List where each entry is a random node from one of the k largest
        connected components.
    """
    component_cardinalities = k * [-1]
    node_ids = k * [-1]
    for nodes in nx.connected_components(graph):
        if len(nodes) > component_cardinalities[-1]:
            i = 0
            while i < k:
                if len(nodes) > component_cardinalities[i]:
                    component_cardinalities.insert(i, len(nodes))
                    component_cardinalities.pop(-1)
                    node_ids.insert(i, util.sample_singleton(nodes))
                    node_ids.pop(-1)
                    break
                i += 1
    return node_ids


def prune_branches(graph, depth):
    """
    Prunes branches with length less than "depth" microns.

    Parameters
    ----------
    graph : networkx.Graph
        Graph to be searched.
    depth : float
        Length of branches that are pruned.
    """
    for leaf in [i for i in graph.nodes if graph.degree[i] == 1]:
        branch = [leaf]
        length = 0
        for i, j in nx.dfs_edges(graph, source=leaf):
            # Visit edge
            length += euclidean(graph.graph["xyz"][i], graph.graph["xyz"][j])
            if length > depth:
                break

            # Check whether to continue search
            if graph.degree(j) == 2:
                branch.append(j)
            elif graph.degree(j) > 2:
                graph.remove_nodes_from(branch)
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
