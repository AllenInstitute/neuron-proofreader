"""
Created on Wed July 2 14:00:00 2025

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Implementation of a custom subclass of Networkx.Graph called "SkeletonGraph".
The graph is constructed by reading and processing SWC files (i.e. neuron
fragments). It then stores the relevant information into the graph structure.

"""

from collections import defaultdict
from io import StringIO
from scipy.spatial import KDTree
from scipy.spatial import distance

import networkx as nx
import numpy as np
import zipfile

from neuron_proofreader.utils import graph_util as gutil, img_util, util


class SkeletonGraph(nx.Graph):
    """
    A custom subclass of NetworkX tailored for graphs constructed from SWC
    files, where each connected component represents a single SWC file. The
    graph is organized hierarchically into two levels of representation:

        1. Irreducible structure — reduced graph where edges represent paths
           between leaf and branching nodes in the original morphology.

        2. Dense structure — the full, fine-grained graph reconstructed from
           the SWC files, preserving detailed spatial geometry.

    Attributes
    ----------
    ...
    """

    def __init__(
        self,
        anisotropy=(1.0, 1.0, 1.0),
        min_size=0,
        node_spacing=1,
        prune_depth=20,
        use_anisotropy=True,
        verbose=False,
    ):
        """
        Instantiates a SkeletonGraph object.

        Parameters
        ----------
        anisotropy : Tuple[float], optional
            Image to physical coordinates scaling factors to account for the
            anisotropy of the microscope. Default is (1.0, 1.0, 1.0).
        min_size : float, optional
            Minimum path length of fragments loaded into graph. Default is 0.
        node_spacing : float, optional
            Distance (in microns) between neighboring nodes. Default 1μm.
        prune_depth : float, optional
            Branches with length less than "prune_depth" microns are removed.
            Default is 20μm.
        use_anisotropy : bool, optional
            Indication of whether to apply anisotropy to SWC files. Note: set
            to False if the SWC files are saved in physical coordinates.
            Default is False.
        verbose : bool, optional
            Indication of whether to display a progress bar while building
            graph. Default is True.
        """
        # Call parent class
        super().__init__()

        # Instance attributes
        self.anisotropy = anisotropy
        self.component_id_to_swc_id = dict()
        self.kdtree = None
        self.node_spacing = node_spacing

        # Graph Loader
        anisotropy = anisotropy if use_anisotropy else (1.0, 1.0, 1.0)
        self.graph_loader = gutil.GraphLoader(
            anisotropy=anisotropy,
            min_size=min_size,
            node_spacing=node_spacing,
            prune_depth=prune_depth,
            verbose=verbose,
        )

    # --- Build Graph ---
    def load(self, swc_pointer):
        """
        Loads SWC files into graph.

        Parameters
        ----------
        swc_pointer : str
            Object that points to SWC files to be read.
        """
        # Extract irreducible components from SWC files
        irreducibles = self.graph_loader(swc_pointer)
        n = 0
        for irr in irreducibles:
            n += len(irr["nodes"])
            for attrs in irr["edges"].values():
                n += len(attrs["xyz"]) - 2

        # Initialize node attribute data structures
        self.node_component_id = np.zeros((n), dtype=int)
        self.node_radius = np.zeros((n), dtype=np.float16)
        self.node_xyz = np.zeros((n, 3), dtype=np.float32)
        self.set_kdtree()

        # Add irreducibles to graph
        component_id = 0
        while irreducibles:
            self.add_connected_component(irreducibles.pop(), component_id)
            component_id += 1

    def add_connected_component(self, irreducibles, component_id):
        """
        Adds a new connected component to the graph.

        Parameters
        ----------
        irreducibles : dict
            Dictionary with the following required fields:
                - "swc_id": SWC ID of the component.
                - "nodes": dictionary of node attributes.
                - "edges": dictionary of edge attributes.
        component_id : int
            Unique identifier for the connected component being added.
        """
        # Set component id
        self.component_id_to_swc_id[component_id] = irreducibles["swc_id"]

        # Add nodes
        node_id_mapping = self._add_nodes(irreducibles["nodes"], component_id)

        # Add edges
        for (i, j), attrs in irreducibles["edges"].items():
            edge_id = (node_id_mapping[i], node_id_mapping[j])
            self._add_edge(edge_id, attrs, component_id)

    def _add_nodes(self, node_dict, component_id):
        """
        Adds nodes to the graph from a dictionary of node attributes and
        returns a mapping from original node IDs to the new graph node IDs.

        Parameters
        ----------
        node_dict : dict
            Dictionary mapping original node IDs to their attributes. Each
            value must be a dictionary containing the keys "radius" and "xyz".
        component_id : str
            Connected component ID used to map node IDs back to SWC IDs via
            "self.component_id_to_swc_id".

        Returns
        -------
        node_id_mapping : Dict[int, int]
            Dictionary mapping the original node IDs from "node_dict" to the
            new node IDs assigned in the graph.
        """
        node_id_mapping = dict()
        for node_id, attrs in node_dict.items():
            new_id = self.number_of_nodes()
            self.node_xyz[new_id] = attrs["xyz"]
            self.node_radius[new_id] = attrs["radius"]
            self.node_component_id[new_id] = component_id
            self.add_node(new_id)
            node_id_mapping[node_id] = new_id
        return node_id_mapping

    def _add_edge(self, edge_id, attrs, component_id):
        """
        Adds an edge to the graph.

        Parameters
        ----------
        edge : Tuple[int]
            Edge to be added.
        attrs : dict
            Dictionary of attributes of "edge" obtained from an SWC file.
        component_id : int
            Connected component ID used to map node IDs back to SWC IDs via
            "self.component_id_to_swc_id".
        """
        # Determine orientation of attributes
        i, j = tuple(edge_id)
        dist_i = distance.euclidean(self.node_xyz[i], attrs["xyz"][0])
        dist_j = distance.euclidean(self.node_xyz[j], attrs["xyz"][0])
        if dist_i < dist_j:
            start = i
            end = j
        else:
            start = j
            end = i

        # Populate graph
        iterator = zip(attrs["radius"], attrs["xyz"])
        for cnt, (radius, xyz) in enumerate(iterator):
            if cnt > 0 and cnt < len(attrs["xyz"]) - 1:
                # Add edge
                new_id = self.number_of_nodes()
                if cnt == 1:
                    self.add_edge(start, new_id)
                else:
                    self.add_edge(new_id, new_id - 1)

                # Store attributes
                self.node_xyz[new_id] = xyz
                self.node_radius[new_id] = radius
                self.node_component_id[new_id] = component_id
        self.add_edge(new_id, end)

    # --- Update Structure ---
    def reassign_component_ids(self):
        """
        Reassigns component IDs for all connected components in the graph.
        """
        component_id_to_swc_id = dict()
        for i, nodes in enumerate(nx.connected_components(self)):
            nodes = np.array(list(nodes), dtype=int)
            component_id_to_swc_id[i + 1] = self.node_swc_id(nodes[0])
            self.node_component_id[nodes] = i + 1
        self.component_id_to_swc_id = component_id_to_swc_id

    def relabel_nodes(self):
        """
        Reassigns contiguous node IDs and update all dependent structures.
        """
        # Set node ids
        old_node_ids = np.array(self.nodes, dtype=int)
        new_node_ids = np.arange(len(old_node_ids))

        # Set edge ids
        old_to_new = dict(zip(old_node_ids, new_node_ids))
        old_edge_ids = list(self.edges)

        # Reset graph
        self.clear()
        for (i, j) in old_edge_ids:
            self.add_edge(old_to_new[i], old_to_new[j])

        # Update attributes
        self.node_radius = self.node_radius[old_node_ids]
        self.node_xyz = self.node_xyz[old_node_ids]
        self.node_component_id = self.node_component_id[old_node_ids]

        self.reassign_component_ids()
        if self.kdtree:
            self.set_kdtree()
        return old_to_new

    def remove_nodes(self, nodes, relabel_nodes=True):
        """
        Removes nodes from both the main graph and the irreducible subgraph.

        Parameters
        ----------
        nodes : container
            Node IDs to remove from the graph.
        relabel_nodes : bool, optional
            Indication of whether to relabel nodes. Default is True.
        """
        self.remove_nodes_from(nodes)
        if relabel_nodes:
            self.relabel_nodes()

    # --- Getters ---
    def branching_nodes(self):
        """
        Gets all branching nodes in the graph.

        Returns
        -------
        List[int]
            Branching nodes in the graph.
        """
        return [i for i in self.nodes if self.degree[i] > 2]

    def component_id_from_swc_id(self, query_swc_id):
        for component_id, swc_id in self.component_id_to_swc_id.items():
            if query_swc_id == swc_id:
                return component_id
        raise ValueError(f"SWC ID={query_swc_id} not found")

    def connected_nodes(self, root):
        """
        Gets all nodes connected to the given root node.

        Parameters
        ----------
        root : int
            Node ID.

        Returns
        -------
        visited : List[int]
            Nodes connected to the given root.
        """
        queue = [root]
        visited = set({root})
        while queue:
            i = queue.pop()
            for j in self.neighbors(i):
                if j not in visited:
                    queue.append(j)
                    visited.add(j)
        return visited

    def leaf_nodes(self):
        """
        Gets all leaf nodes in the graph.

        Returns
        -------
        List[int]
            Leaf nodes in the graph.
        """
        return [i for i in self.nodes if self.degree[i] == 1]

    def node_local_voxel(self, node, offset):
        """
        Computes the local voxel coordinate of the given node within the image
        patch defined by "center" and "patch_shape".

        Parameters
        ----------
        node : int
            Node in image patch.
        offset : Tuple[int]
            Shift to be applied to the node's voxel coordinate.

        Returns
        -------
        Tuple[int]
            Local voxel coordinate of the given node within the image patch
            defined by "center" and "patch_shape".
        """
        return tuple([v - o for v, o in zip(self.node_voxel(node), offset)])

    def node_segment_id(self, node):
        """
        Gets the segment ID corresponding to the given node.

        Parameters
        ----------
        node : int
            Node ID.

        Returns
        -------
        str
            Segment ID corresponding to the given node.
        """
        return self.get_swc_id(node).split(".")[0]

    def node_swc_id(self, i):
        """
        Gets the SWC ID of the given node.

        Parameters
        ----------
        i : int
            Node ID.

        Returns
        -------
        str
            SWC ID of the given node.
        """
        component_id = self.node_component_id[i]
        return self.component_id_to_swc_id[component_id]

    def node_voxel(self, i):
        """
        Gets the voxel coordinate of the given node.

        Parameters
        ----------
        i : int
            Node ID.

        Returns
        -------
        float
            Voxel coordinate of the given node.
        """
        return img_util.to_voxels(self.node_xyz[i], self.anisotropy)

    def nodes_with_component_id(self, component_id):
        """
        Gets all nodes with the given componenet ID.

        Parameters
        ----------
        component_id : int
            Unique identifier of connected component to be queried.

        Returns
        -------
        Set[int]
            Nodes with the given component ID.
        """
        return set(np.where(self.node_component_id == component_id)[0])

    def nodes_with_segment_id(self, segment_id):
        """
        Gets all nodes with the given segment ID.

        Parameters
        ----------
        segment_id : int
            Unique identifier of a segment to be queried.

        Returns
        -------
        Set[int]
            Nodes with the given segment ID.
        """
        nodes = set()
        query_id = f"{segment_id}."
        for swc_id in self.get_swc_ids():
            segment_id = int(swc_id.replace(".0", ""))
            if segment_id == query_id:
                component_id = self.component_id_from_swc_id(swc_id)
                nodes = nodes.union(
                    self.nodes_with_component_id(component_id)
                )
        return nodes

    def nodes_within_distance(self, root, max_depth):
        queue = [(root, 0)]
        visited = {root}
        while queue:
            # Visit node
            i, dist_i = queue.pop()

            # Populate queue
            for j in self.neighbors(i):
                dist_j = dist_i + self.dist(i, j)
                if dist_j < max_depth and j not in visited:
                    queue.append((j, dist_j))
                    visited.add(j)
        return list(visited)

    def path_from_leaf(self, leaf, max_depth=np.inf):
        queue = [(leaf, 0)]
        path = [leaf]
        while queue:
            # Visit node
            i, dist_i = queue.pop()
            if self.degree[i] != 2 and dist_i > 0:
                return path

            # Update queue
            for j in self.neighbors(i):
                dist_j = dist_i + self.dist(i, j)
                if dist_j < max_depth and j not in path:
                    queue.append((j, dist_j))
                    path.append(j)
        return path

    def rooted_subgraph(self, root, radius):
        """
        Gets a rooted subgraph with the given radius (in microns).

        Parameters
        ----------
        root : int
            Node ID of root.
        radius : float
            Depth (in microns) of subgraph.

        Returns
        -------
        subgraph : SkeletonGraph
            Rooted subgraph.
        """
        # Initializations
        subgraph = SkeletonGraph(anisotropy=self.anisotropy)
        subgraph.add_node(0)
        idxs = [root]

        # Extract graph
        node_mapping = {root: 0}
        queue = [(root, 0)]
        visited = {root}
        while queue:
            # Visit node
            i, dist_i = queue.pop()

            # Populate queue
            for j in self.neighbors(i):
                dist_j = dist_i + self.dist(i, j)
                if j not in visited and dist_j < radius:
                    node_mapping[j] = subgraph.number_of_nodes()
                    subgraph.add_edge(node_mapping[i], node_mapping[j])
                    queue.append((j, dist_j))
                    visited.add(j)
                    idxs.append(j)

        # Store coordinates
        idxs = np.array(idxs, dtype=int)
        subgraph.node_radius = self.node_radius[idxs]
        subgraph.node_xyz = self.node_xyz[idxs]
        return subgraph

    def get_swc_ids(self):
        """
        Gets the set of all unique SWC IDs of nodes in the graph.

        Returns
        -------
        Set[str]
            Set of all unique SWC IDs of nodes in the graph.
        """
        return set(self.component_id_to_swc_id.values())

    # --- Writer ---
    def to_zipped_swcs(self, zip_path, preserve_radius=False):
        """
        Writes the graph to a ZIP archive of SWC files, where each file
        corresponds to a connected component.

        Parameters
        ----------
        zip_path : str
            Path to ZIP archive that SWC files are to be written to.
        preserve_radius : bool, optional
            Indication of whether to set radius as node radius or 2μm.
            Default is False.
        """
        with zipfile.ZipFile(zip_path, "w") as zip_writer:
            for nodes in map(list, nx.connected_components(self)):
                root = util.sample_once(nodes)
                self.component_to_zipped_swc(
                    zip_writer, root, preserve_radius=preserve_radius
                )

    def component_to_zipped_swc(
        self, zip_writer, root, preserve_radius=False
    ):
        """
        Writes the connected component containing the given root node to a
        zipped SWC file.

        Parameters
        ----------
        zip_writer : zipfile.ZipFile
            A ZipFile object that will store the generated SWC file.
        root : int
            Root node of connected component to be written to an SWC file.
        preserve_radius : bool, optional
            Indication of whether to preserve radii of nodes or use default
            radius of 2μm. Default is False.
        """
        # Subroutines
        def write_entry(node, parent):
            """
            Writes a line of an SWC file for the given node.

            Parameters
            ----------
            node : int
                Node ID.
            parent : int
                Node ID of parent.
            """
            x, y, z = tuple(self.node_xyz[node])
            r = self.node_radius[node] if preserve_radius else 2
            node_id = cnt
            parent_id = node_to_idx[parent]
            node_to_idx[node] = node_id
            text_buffer.write(f"\n{node_id} 2 {x} {y} {z} {r} {parent_id}")

        # Main
        with StringIO() as text_buffer:
            # Preamble
            text_buffer.write("# id, type, z, y, x, r, pid")

            # Write entries
            cnt = 1
            node_to_idx = defaultdict(lambda: -1)
            for i, j in nx.dfs_edges(self, source=root):
                # Special Case: Root
                if len(node_to_idx) == 0:
                    write_entry(i, -1)

                # General Case: Non-Root
                cnt += 1
                write_entry(j, i)

            # Finish
            filename = self.node_swc_id(i)
            filename = util.set_zip_path(zip_writer, filename, ".swc")
            zip_writer.writestr(filename, text_buffer.getvalue())

    # --- Helpers ---
    def cable_length(self, max_depth=np.inf, root=None):
        """
        Computes the cable length of the graph. If a root is provided, then
        cable length of the connected component containing the given root is
        computed.

        Parameters
        ----------
        max_depth : float, optional
            Maximum depth (in microns) to traverse before stopping. Useful for
            checking if cable length is above a threshold. Default is np.inf.
        root : int
            Node contained in connected component to be searched. Default is
            None.

        Returns
        -------
        cable_length : float
            Cable length of the connected component containing the given root
            node.
        """
        cable_length = 0
        for i, j in nx.dfs_edges(self, source=root):
            cable_length += self.dist(i, j)
            if cable_length > max_depth:
                break
        return cable_length

    def clip_to_bbox(self, metadata_path):
        bucket_name, path = util.parse_cloud_path(metadata_path)
        if util.check_gcs_file_exists(bucket_name, path):
            # Extract bounding box
            metadata = util.read_json_from_gcs(bucket_name, path)
            origin = metadata["chunk_origin"][::-1]
            shape = metadata["chunk_shape"][::-1]

            # Clip graph
            nodes = list()
            for i in self.nodes:
                voxel = np.array(self.node_voxel(i))
                if not img_util.is_contained(voxel - origin, shape):
                    nodes.append(i)
            self.remove_nodes_from(nodes)
            self.relabel_nodes()

    def clip_to_groundtruth(self, gt_graph, dist):
        """
        Removes nodes that are more than "dist" microns from "gt_graph".

        Parameters
        ----------
        gt_graph : SkeletonGraph
            Ground truth graph used as clipping reference.
        dist : float
            Distance threshold (in microns) that determines what nodes to
            remove.
        """
        # Remove nodes too far from ground truth
        d_gt, _ = gt_graph.kdtree.query(self.node_xyz)
        nodes = np.where(d_gt > dist)[0]
        self.remove_nodes_from(nodes)

        # Remove resulting small connected components
        for nodes in list(nx.connected_components(self)):
            if len(nodes) < 30:
                self.remove_nodes_from(nodes)
        self.relabel_nodes()

    def closest_node(self, xyz):
        """
        Finds the closest node to the given xyz coordinate.

        Parameters
        ----------
        xyz : ArrayLike
            Coordinate to be queried.

        Returns
        -------
        node : int
            Closest node to the given xyz coordinate.
        """
        assert self.kdtree, "KD-Tree attribute has not be set!"
        _, node = self.kdtree.query(xyz)
        return node

    def dist(self, i, j):
        """
        Computes the Euclidean distance between nodes "i" and "j".

        Parameters
        ----------
        i : int
            Node ID.
        j : int
            Node ID.

        Returns
        -------
        float
            Euclidean distance between nodes "i" and "j".
        """
        return distance.euclidean(self.node_xyz[i], self.node_xyz[j])

    def get_irreducible_edge(self, node):
        # Check node is non-branhching
        assert self.degree[node] < 3

        # Search
        edge = list()
        queue = [node]
        visited = set(queue)
        while queue:
            # Visit node
            i = queue.pop()
            if self.degree[i] != 2:
                edge.append(i)
                continue

            # Update queue
            for j in self.neighbors(i):
                if j not in visited:
                    queue.append(j)
                    visited.add(j)
        assert len(edge) == 2
        return edge

    def get_summary(self, prefix=""):
        # Compute values
        n_components = format(nx.number_connected_components(self), ",")
        n_nodes = format(self.number_of_nodes(), ",")
        n_edges = format(self.number_of_edges(), ",")
        memory = util.get_memory_usage()

        # Compile results
        summary = [f"{prefix} Graph"]
        summary.append(f"# Connected Components: {n_components}")
        summary.append(f"# Nodes: {n_nodes}")
        summary.append(f"# Edges: {n_edges}")
        summary.append(f"Memory Consumption: {memory:.2f} GBs")
        return "\n".join(summary)

    def path_length(self, path):
        """
        Computes the length of the given path.

        Parameters
        ----------
        path : List[int]
            List of nodes that forms a path.

        Returns
        -------
        Length of the given path.
        """
        path_length = 0
        if len(path) > 1:
            for i, j in zip(path[1:], path[:-1]):
                path_length += self.dist(i, j)
        return path_length

    def set_kdtree(self):
        """
        Initializes KD-Tree from node xyz coordinates.
        """
        self.kdtree = KDTree(self.node_xyz)
