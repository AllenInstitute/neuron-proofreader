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

from collections import defaultdict, deque
from concurrent.futures import (
    as_completed, ProcessPoolExecutor, ThreadPoolExecutor
)
from random import sample
from scipy.spatial import KDTree
from tqdm import tqdm

import multiprocessing
import networkx as nx
import numpy as np
import os

from neuron_proofreader.utils import (
    geometry_util as geometry, img_util, swc_util, util
)

os.environ["OPENBLAS_NUM_THREADS"] = "1"


class GraphLoader:
    """
    Class that loads SWC files and constructs a FragmentsGraph instance from
    the data.
    """

    def __init__(
        self,
        anisotropy=(1.0, 1.0, 1.0),
        min_size=40.0,
        node_spacing=1,
        prune_depth=24.0,
        remove_high_risk_merges=False,
        segmentation_path=None,
        smooth_bool=True,
        soma_centroids=None,
        verbose=False,
    ):
        """
        Builds a FragmentsGraph by reading swc files stored on either the
        cloud or local machine, then extracting the irreducible components.

        Parameters
        ----------
        anisotropy : List[float], optional
            Image to physical coordinates scaling factors to account for the
            anisotropy of the microscope. Default is (1.0, 1.0, 1.0).
        min_size : float, optional
            Minimum path length of swc files that are loaded into the
            FragmentsGraph. Default is 24.0 microns.
        node_spacing : int, optional
            Sampling rate for nodes in FragmentsGraph. Every "node_spacing"
            node is retained. Default is 1.
        prune_depth : int, optional
            Branches with length less than "prune_depth" microns are pruned.
            Default is 20.0 microns.
        remove_high_risk_merges : bool, optional
            Indication of whether to remove high risk merge sites (i.e. close
            branching points). Default is False.
        segmentation_path : str, optional
            Path to segmentation stored in GCS bucket. Default is None.
        smooth_bool : bool, optional
            Indication of whether to smooth xyz coordinates from SWC files.
            Default is True.
        soma_centroids : List[Tuple[float]] or None, optional
            Physcial coordinates of soma centroids. Default is None.
        verbose : bool, optional
            Indication of whether to display a progress bar while building
            FragmentsGraph. Default is True.
        """
        # Instance attributes
        self.id_to_soma = defaultdict(list)
        self.min_size = min_size
        self.node_spacing = node_spacing
        self.prune_depth = prune_depth
        self.remove_high_risk_merges_bool = remove_high_risk_merges
        self.smooth_bool = smooth_bool
        self.soma_centroids = soma_centroids
        self.verbose = verbose

        # SWC reader
        self.swc_reader = swc_util.Reader(anisotropy, min_size)

        # Load somas
        if segmentation_path and soma_centroids:
            self.soma_kdtree = KDTree(self.soma_centroids)
            self.ingest_somas(segmentation_path)
        else:
            self.soma_kdtree = None

    def ingest_somas(self, segmentation_path):
        """
        Loads soma locations from a specified file and search for interestions
        between soma locations and objects in segmentation mask.

        Parameters
        ----------
        segmentation_path : str
            Path to a segmentation stored in a GCS bucket.
        """
        reader = img_util.TensorStoreReader(segmentation_path)
        with ThreadPoolExecutor() as executor:
            # Assign threads
            threads = list()
            for xyz in self.soma_centroids:
                voxel = img_util.to_voxels(xyz, (0.748, 0.748, 1.0))
                threads.append(executor.submit(reader.read_voxel, voxel, xyz))

            # Store results
            for thread in as_completed(threads):
                xyz, segment_id = thread.result()
                if segment_id != 0:
                    self.id_to_soma[segment_id].append(xyz)

        # Report results
        if self.verbose:
            n = np.sum([1 for v in self.id_to_soma.values() if len(v) > 1])
            print("# Somas:", len(self.soma_centroids))
            print("# Soma-Fragment Intersections:", len(self.id_to_soma))
            print("# Soma Merges:", n)

    # --- Irreducibles Extraction ---
    def run(self, fragments_pointer):
        """
        Processes a list of swc dictionaries in parallel and extracts the
        components of the irreducible subgraph from each. Note: this routine
        also breaks fragments that intersect multiple somas if soma locations
        are provided.

        Parameters
        ----------
        fragments_pointer : Any
            Object that points to SWC files to be read.

        Returns
        -------
        irreducibles : List[dict]
            Dictionaries that contain components of the irreducible subgraph
            extracted from each SWC dictionary.
        """
        # Read SWC files
        swc_dicts = self.swc_reader.read(fragments_pointer)

        # Load graphs
        desc = "Extract Graphs"
        pbar = tqdm(total=len(swc_dicts), desc=desc) if self.verbose else None
        multiprocessing.set_start_method('spawn', force=True)
        with ProcessPoolExecutor() as executor:
            # Start processes
            pending = set()
            for _ in range(min(512, len(swc_dicts))):
                pending.add(executor.submit(self.extract, swc_dicts.pop()))

            # Yield processes
            irreducibles = deque()
            high_risk_cnt = 0
            while pending or swc_dicts:
                for process in as_completed(pending):
                    # Store completed processes
                    result, cnt = process.result()
                    pending.remove(process)
                    high_risk_cnt += cnt
                    irreducibles.extend(result)
                    pbar.update(1) if self.verbose else None

                    # Continue submitting processes
                    if swc_dicts:
                        pending.add(
                            executor.submit(self.extract, swc_dicts.pop())
                        )

        # Report results
        if self.verbose and high_risk_cnt > 0:
            print("# High Risk Merges Detected:", high_risk_cnt)
        return irreducibles

    def extract(self, swc_dict):
        """
        Breaks a graph built from "swc_dict" at high risk locations, then
        extracts the components of the irreducible subgraph from a given SWC
        dictionary.

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
        graph = self.to_graph(swc_dict)
        irreducibles_list = deque()
        high_risk_cnt = 0
        if self.satifies_path_length_condition(graph):
            # Check for soma merges
            if len(graph.graph["soma_nodes"]) > 1:
                self.remove_soma_merges(graph)

            # Check for high risk merges
            if self.remove_high_risk_merges_bool:
                high_risk_cnt = self.remove_high_risk_merges(graph)

            # Extract irreducibles
            i = 0
            leafs = set(get_leafs(graph))
            while leafs:
                # Extract for connected component
                leaf = util.sample_once(leafs)
                irreducibles, visited = self.get_irreducibles(graph, leaf)
                leafs -= visited

                # Store results
                if irreducibles:
                    swc_id = f"{graph.graph['segment_id']}.{i}"
                    irreducibles["swc_id"] = swc_id
                    irreducibles_list.append(irreducibles)
                    i += 1
        return irreducibles_list, high_risk_cnt

    def get_irreducibles(self, graph, source):
        """
        Identifies irreducible components of a connected graph.

        Parameters
        ----------
        graph : networkx.Graph
            Graph to be searched.
        source : Set[int]
            Leaf nodes in the given graph.

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
            return geometry.dist(graph.graph["xyz"][i], graph.graph["xyz"][j])

        # Initializations
        irreducible_nodes = set({source})
        irreducible_edges = dict()
        is_soma = source in graph.graph["soma_nodes"]
        leafs = set({source})

        # Main
        root, path_length = None, 0
        for (i, j) in nx.dfs_edges(graph, source=source):
            # Check for start of irreducible edge
            if root is None:
                root, edge_length = i, 0
                attrs = {
                    "radius": [graph.graph["radius"][i]],
                    "xyz": [graph.graph["xyz"][i]],
                }

            # Visit node
            edge_length += dist(i, j)
            is_soma = is_soma or j in graph.graph["soma_nodes"]
            attrs["radius"].append(graph.graph["radius"][j])
            attrs["xyz"].append(graph.graph["xyz"][j])

            # Check for end of irreducible edge
            if graph.degree[j] != 2:
                path_length += edge_length
                irreducible_nodes.add(j)
                attrs = to_numpy(attrs)
                if self.smooth_bool:
                    n_pts = int(edge_length / self.node_spacing)
                    self.smooth_curve_3d(graph, attrs, (root, j), n_pts)
                if graph.degree[j] == 1:
                    leafs.add(j)

                # Finish
                irreducible_edges[(root, j)] = attrs
                root = None

        # Check for curvy line fragment
        if len(irreducible_nodes) == 2:
            endpoint_dist = dist(*tuple(irreducible_nodes))
            if endpoint_dist / path_length < 0.5:
                return None, leafs

        # Store results
        if path_length > self.min_size - 10:
            irreducibles = {
                "nodes": set_node_attrs(graph, irreducible_nodes),
                "edges": set_edge_attrs(graph, irreducible_edges),
                "is_soma": is_soma,
            }
        else:
            irreducibles = None
        return irreducibles, leafs

    # --- Merge Removal ---
    def remove_high_risk_merges(self, graph, max_dist=7):
        """
        Removes high risk merge sites from a graph, which is defined to be
        either (1) two branching points within "max_dist" or (2) branching
        point with degree 4+. Note: if soma locations are provided, we skip
        branching points within 300um of a soma.

        Parameters
        ----------
        graph : networkx.Graph
            Graph to be searched.
        max_dist : float, optional
            Maximum distance between branching points that qualifies a site to
            be considered "high risk". The default is 7.

        Returns
        -------
        high_risk_cnt : int
            Number of high risk merges detected.
        """
        high_risk_cnt = 0
        nodes = set()
        branchings = set([i for i in graph.nodes if graph.degree[i] > 2])
        while branchings:
            # Initializations
            root = branchings.pop()
            hit_branchings = set()
            queue = [(root, 0)]
            visited = set(queue)

            # Check if close to soma
            soma_dist = self.dist_from_soma(graph.graph["xyz"][root])
            if graph.graph["soma_nodes"] and soma_dist < 300:
                continue

            # BFS
            while len(queue) > 0:
                # Visit node
                i, dist_i = queue.pop()
                if graph.degree[i] > 2 and i != root:
                    hit_branchings.add(i)

                # Update queue
                for j in graph.neighbors(i):
                    dist_j = dist_i + dist(graph, i, j)
                    if j not in visited and dist_j < max_dist:
                        queue.append((j, dist_j))
                        visited.add(j)

            # Determine whether to remove visited nodes
            if hit_branchings or graph.degree(root) > 3:
                nodes = nodes.union(visited)
                high_risk_cnt += 1
                branchings -= hit_branchings
        graph.remove_nodes_from(nodes)
        return high_risk_cnt

    def remove_soma_merges(self, graph):
        """
        Breaks a fragment that intersects with multiple somas so that nodes
        closest to soma locations are disconnected.

        Parameters
        ----------
        swc_dict : dict
            Contents of an SWC file.
        somas_xyz : List[Tuple[float]]
            Physical coordinates representing soma locations.
        """
        soma_nodes = graph.graph["soma_nodes"]
        if len(soma_nodes) <= 20:
            # Break connecting path
            nodes = set()
            for node in find_connecting_path(graph, list(soma_nodes)):
                if graph.degree(node) > 2:
                    nodes.add(node)
            remove_nearby_nodes(graph, nodes)

            # Associate each soma to one node
            graph.graph["soma_nodes"] = soma_nodes.intersection(graph.nodes)

    # --- Helpers ---
    def dist_from_soma(self, xyz):
        """
        Computes the distance between the given xyz coordinate and nearest
        soma location.

        Parameters
        ----------
        xyz : ArrayLike
            Coordinate to be queried.

        Returns
        -------
        float
            Distance between a given xyz coordinate and the nearest soma
            location.
        """
        return self.soma_kdtree.query(xyz)[0] if self.soma_kdtree else np.inf

    def find_soma_nodes(self, graph):
        """
        Find nodes in graph that correspond to soma locations.

        Parameters
        ----------
        graph : networkx.Graph
            Graph to be searched.

        Returns
        -------
        soma_nodes : Set[int]
            Node IDs that correspond to soma locations.
        """
        soma_nodes = set()
        for xyz in self.id_to_soma[graph.graph["segment_id"]]:
            node, dist = find_closest_node(graph, xyz)
            if dist < 20:
                soma_nodes.add(find_nearby_branching_node(graph, node, 20))
        return soma_nodes

    def satifies_path_length_condition(self, graph):
        """
        Determines whether the total path length of the given graph is greater
        than "self.min_size".

        Parameters
        ----------
        xyz : ArrayLike
            Physical coordinate to be queried.

        Returns
        -------
        bool
            Indication of whether the total path length of the given graph is
            greater than "self.min_size".
        """
        return path_length(graph, self.min_size) > self.min_size

    def smooth_curve_3d(self, graph, attrs, edge, n_pts):
        """
        Smooth a 3D curve and update the corresponding edge endpoints in the
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

        Returns
        -------
        None
        """
        attrs["xyz"] = geometry.smooth_curve_3d(attrs["xyz"], n_pts=n_pts)
        attrs["radius"] = geometry.smooth_curve_1d(attrs["radius"], n_pts)
        graph.graph["xyz"][edge[0]] = attrs["xyz"][0]
        graph.graph["xyz"][edge[1]] = attrs["xyz"][-1]

    def to_graph(self, swc_dict):
        """
        Converts a dictionary containing swc attributes to a graph.

        Parameters
        ----------
        swc_dict : dict
            Dictionaries whose keys and values are the attribute name and
            values from an SWC file.

        Returns
        -------
        graph : networkx.Graph
            Graph generated from "swc_dict".
        """
        # Build graph
        graph = swc_util.to_graph(swc_dict, set_attrs=True)
        prune_branches(graph, self.prune_depth)

        # Check if original segment intersects with soma
        graph.graph["segment_id"] = swc_util.get_segment_id(swc_dict["swc_name"])
        if graph.graph["segment_id"] in self.id_to_soma:
            graph.graph["soma_nodes"] = self.find_soma_nodes(graph)
        else:
            graph.graph["soma_nodes"] = set()
        return graph


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
            "radius": graph.graph["radius"][i], "xyz": graph.graph["xyz"][i]
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


def dist(graph, i, j):
    """
    Computes Euclidean distance between nodes i and j.

    Parameters
    ----------
    graph : netowrkx.Graph
        Graph containing nodes i and j.
    i : int
        Node ID.
    j : int
        Node ID.

    Returns
    -------
    float
        Euclidean distance between nodes i and j.
    """
    return geometry.dist(graph.graph["xyz"][i], graph.graph["xyz"][j])


def find_closest_node(graph, xyz):
    """
    Finds the node in the graph that is closest to the given coordinates.

    Parameters
    ----------
    graph : networkx.Graph
        Graph to be search.
    xyz : Tuple[float]
        Coordinate to which the closest node in the graph will be found.

    Returns
    -------
    best_node : int
        Node in the graph that is closest to the given xyz coordinate.
    best_dist : float
        Distance between "best_node" and the given xyz coordinate.
    """
    best_dist = np.inf
    best_node = None
    for i in graph.nodes:
        cur_dist = geometry.dist(xyz, graph.graph["xyz"][i])
        if cur_dist < best_dist:
            best_dist = cur_dist
            best_node = i
    return best_node, best_dist


def find_connecting_path(graph, nodes):
    """
    Finds path that connects a set of nodes.

    Parameters
    ----------
    graph : networkx.Graph
        Graph to be searched.
    nodes : List[int]
        List of nodes to be connected.

    Returns
    -------
    path : Set[int]
        Nodes along the shortest paths between somas.
    """
    # Break merges between somas
    path = set()
    if len(nodes) > 1:
        for i in range(1, len(nodes)):
            subpath = nx.shortest_path(
                graph, source=nodes[0], target=nodes[i]
            )
            path = path.union(set(subpath))
    return path


def find_leaf(graph):
    """
    Finds a leaf node in the given graph.

    Parameters
    ----------
    graph : networkx.Graph
        Graph to be searched.

    Returns
    -------
    int
        Leaf node.
    """
    for i in graph.nodes:
        if graph.degree[i] == 1:
            return i


def find_nearby_branching_node(graph, root, max_depth=10):
    """
    Search for the nearest branching node starting from a given root node.

    Parameters
    ----------
    graph : networkx.Graph
        Graph to be searched.
    root : int
        Node from which to begin the search.
    max_depth : float
        Maximum distance to search from the root node.

    Returns
    -------
    int
        Nearest branching node if one is found; otherwise, returns "root".
    """
    queue = [(root, 0)]
    visited = set(queue)
    while queue:
        # Visit node
        i, dist_i = queue.pop()
        if graph.degree[i] >= 3:
            return i

        # Update queue
        for j in graph.neighbors(i):
            dist_j = dist_i + dist(graph, i, j)
            if dist_j < max_depth and j not in visited:
                queue.append((j, dist_j))
                visited.add(j)
    return root


def get_component(graph, root):
    """
    Gets the connected component corresponding to "root" from "graph".

    Parameters
    ----------
    graph : networkx.Graph
        Graph to be searched.
    root : int
        Node that breadth-first search starts from.

    Returns
    -------
    visited : Set[int]
        Nodes in the connected component corresponding to "root".
    """
    queue = [root]
    visited = set()
    while len(queue):
        i = queue.pop()
        visited.add(i)
        for j in [j for j in graph.neighbors(i) if j not in visited]:
            queue.append(j)
    return visited


def get_leafs(graph):
    """
    Gets leaf nodes of the given graph.

    Parameters
    ----------
    graph : networkx.Graph
        Graph to be searched

    Returns
    -------
    List[int]
        Leaf nodes of the given graph.
    """
    return [i for i in graph.nodes if graph.degree[i] == 1]


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


def path_length(graph, max_length=np.inf):
    """
    Computes the path length of the given graph.

    Parameters
    ----------
    graph : networkx.Graph
        Graph whose nodes have an attribute called "xyz" which represents
        a 3d coordinate.
    max_length : float
        Maximum physical distance to search along the graph. Limits traversal
        depth and can improve performance.

    Returns
    -------
    length : float
        Path length of graph.
    """
    length = 0
    for i, j in nx.dfs_edges(graph):
        length += dist(graph, i, j)
        if length > max_length:
            break
    return length


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
    for leaf in get_leafs(graph):
        branch = [leaf]
        length = 0
        for (i, j) in nx.dfs_edges(graph, source=leaf):
            # Visit edge
            length += dist(graph, i, j)
            if length > depth:
                break

            # Check whether to continue search
            if graph.degree(j) == 2:
                branch.append(j)
            elif graph.degree(j) > 2:
                graph.remove_nodes_from(branch)
                break


def remove_nearby_nodes(graph, roots, max_dist=5.0):
    """
    Removes nodes from graph within a given radius from a set of root nodes.

    Parameters
    ----------
    graph : networkx.Graph
        Graph to be searched.
    roots : List[int]
        Root nodes.
    max_dist : float, optional
        Maximum distance within which nodes are removed. The default is 5.0.
    """
    nodes = set()
    while len(roots) > 0:
        root = roots.pop()
        queue = [(root, 0)]
        visited = set()
        while len(queue) > 0:
            # Visit node
            i, dist_i = queue.pop()
            visited.add(i)

            # Update queue
            for j in graph.neighbors(i):
                dist_j = dist_i + dist(graph, i, j)
                if j not in visited and dist_j <= max_dist:
                    queue.append((j, dist_j))
                elif j not in visited and graph.degree[j] > 2:
                    queue.append((j, dist_i))
        nodes = nodes.union(visited)
    graph.remove_nodes_from(nodes)


def sample_node(graph):
    """
    Samples a single node from a graph.

    Parameters
    ----------
    graph : networkx.Graph
        Graph to be sampled from.

    Returns
    -------
    int
        Node ID.
    """
    nodes = list(graph.nodes)
    return sample(nodes, 1)[0]


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
