"""
Created on Wed July 2 14:00:00 2025

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Implementation of a custom subclass of Networkx.Graph called "SkeletonGraph".
The graph is constructed by reading and processing SWC files (i.e. neuron
fragments). It then stores the relevant information into the graph structure.

"""

from io import StringIO
from scipy.spatial import KDTree

import networkx as nx
import numpy as np
import zipfile

from neuron_proofreader.utils import (
    geometry_util, graph_util as gutil, img_util, util
)


class SkeletonGraph(nx.Graph):

    def __init__(
        self,
        anisotropy=(1.0, 1.0, 1.0),
        min_size=0,
        node_spacing=1,
        prune_depth=20.0,
        smooth_bool=True,
        verbose=False,
    ):
        # Call parent class
        super().__init__()

        # Instance attributes
        self.anisotropy = anisotropy
        self.component_id_to_swc_id = dict()
        self.irreducible = nx.Graph()
        self.node_spacing = node_spacing

        # Graph Loader
        self.graph_loader = gutil.GraphLoader(
            anisotropy=anisotropy,
            min_size=min_size,
            node_spacing=node_spacing,
            smooth_bool=smooth_bool,
            prune_depth=prune_depth,
            verbose=verbose,
        )

    # --- Build Graph ---
    def load(self, swc_pointer):
        # Extract irreducible components from SWC files
        irreducibles = self.graph_loader.run(swc_pointer)
        n = 0
        for irr in irreducibles:
            n += len(irr["nodes"])
            for attrs in irr["edges"].values():
                n += len(attrs["xyz"]) - 2

        # Initialize node attribute data structures
        self.node_component_id = np.zeros((n), dtype=int)
        self.node_radius = np.zeros((n), dtype=np.float16)
        self.node_xyz = np.zeros((n, 3), dtype=np.float32)

        # Add irreducibles to graph
        component_id = 0
        while irreducibles:
            self.add_connected_component(irreducibles.pop(), component_id)
            component_id += 1

    def add_connected_component(self, irreducibles, component_id):
        # Set component id
        self.component_id_to_swc_id[component_id] = irreducibles["swc_id"]

        # Add nodes
        node_id_mapping = self._add_nodes(irreducibles["nodes"], component_id)

        # Add edges
        for (i, j), attrs in irreducibles["edges"].items():
            edge_id = (node_id_mapping[i], node_id_mapping[j])
            self._add_edge(edge_id, attrs, component_id)
            self.irreducible.add_edge(*edge_id)

    def _add_nodes(self, node_dict, component_id):
        """
        Adds nodes to the graph from a dictionary of node attributes and
        returns a mapping from original node IDs to the new graph node IDs.

        Parameters
        ----------
        node_dict : dict
            Dictionary mapping original node IDs (e.g., from an SWC file) to
            their attributes. Each value must be a dictionary containing:
                - "radius" : float
                - "xyz"    : array-like of shape (3,)
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
        Adds an edge to "self".

        Parameters
        ----------
        edge : Tuple[int]
            Edge to be added.
        attrs : dict
            Dictionary of attributes of "edge" obtained from an SWC file.
        component_id : str
            Connected component ID used to map node IDs back to SWC IDs via
            "self.component_id_to_swc_id".
        """
        # Determine orientation of attributes
        i, j = tuple(edge_id)
        dist_i = geometry_util.dist(self.node_xyz[i], attrs["xyz"][0])
        dist_j = geometry_util.dist(self.node_xyz[j], attrs["xyz"][0])
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

    def relabel_nodes(self):
        # Set node ids
        old_node_ids = np.array(self.nodes, dtype=int)
        new_node_ids = np.arange(len(old_node_ids))

        # Set edge ids
        old_to_new = dict(zip(old_node_ids, new_node_ids))
        old_edge_ids = list(self.edges)
        old_irr_edge_ids = self.irreducible.edges

        # Reset graph
        self.clear()
        for (i, j) in old_edge_ids:
            self.add_edge(old_to_new[i], old_to_new[j])

        self.irreducible.clear()
        for (i, j) in old_irr_edge_ids:
            self.irreducible.add_edge(old_to_new[i], old_to_new[j])

        self.node_radius = self.node_radius[old_node_ids]
        self.node_xyz = self.node_xyz[old_node_ids]
        self.node_component_id = self.node_component_id[old_node_ids]
        self.reassign_component_ids()

    def reassign_component_ids(self):
        """
        Reassigns component IDs for all connected components in the graph.
        """
        component_id_to_swc_id = dict()
        for i, nodes in enumerate(nx.connected_components(self)):
            nodes = np.array(list(nodes), dtype=int)
            component_id_to_swc_id[i + 1] = self.get_swc_id(nodes[0])
            self.node_component_id[nodes] = i + 1
        self.component_id_to_swc_id = component_id_to_swc_id

    # --- Getters ---
    def get_branchings(self):
        """
        Gets all branching nodes in the graph.

        Returns
        -------
        List[int]
            Branching nodes in the graph.
        """
        return [i for i in self.nodes if self.degree[i] > 2]

    def get_connected_nodes(self, root):
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

    def get_leafs(self):
        """
        Gets all leaf nodes in the graph.

        Returns
        -------
        List[int]
            Leaf nodes in the graph.
        """
        return [i for i in self.nodes if self.degree[i] == 1]

    def get_nodes_with_component_id(self, component_id):
        return set(np.where(self.node_component_id == component_id)[0])

    def get_rooted_subgraph(self, root, radius):
        # Initializations
        subgraph = SkeletonGraph()
        node_mapping = {root: 0}

        # Extract graph
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

        # Store coordinates
        idxs = np.array(list(node_mapping.keys()), dtype=int)
        subgraph.node_xyz = self.node_xyz[idxs]
        return subgraph

    def get_swc_id(self, i):
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

    def get_swc_ids(self):
        """
        Gets the set of all unique SWC IDs of nodes in the graph.

        Returns
        -------
        Set[str]
            Set of all unique SWC IDs of nodes in the graph.
        """
        return np.unique(list(self.component_id_to_swc_id.values()))

    # --- Writer ---
    def to_zipped_swcs(self, zip_path, preserve_radius=False):
        with zipfile.ZipFile(zip_path, "w") as zip_writer:
            for nodes in nx.connected_components(self):
                root = util.sample_once(nodes)
                self.component_to_zipped_swc(
                    zip_writer, root, preserve_radius=preserve_radius
                )

    def component_to_zipped_swc(
        self, zip_writer, root, preserve_radius=False
    ):
        """
        Writes the graph to an SWC file format, which is then stored in a ZIP
        archive.

        Parameters
        ----------
        zip_writer : zipfile.ZipFile
            A ZipFile object that will store the generated SWC file.
        root : int
            Root node of connected component to be written to an SWC file.
        preserve_radius : bool, optional
            Indication of whether to preserve radii of nodes or use default
            radius of 2um. Default is False.
        """
        # Subroutines
        def write_entry(node, parent):
            x, y, z = tuple(self.node_xyz[node])
            r = self.node_radius[node] if preserve_radius else 2
            node_id = len(node_to_idx) + 1
            node_to_idx[node] = node_id
            text_buffer.write("\n" + f"{node_id} 2 {x} {y} {z} {r} {parent}")

        # Main
        with StringIO() as text_buffer:
            # Preamble
            text_buffer.write("\n" + "# id, type, z, y, x, r, pid")

            # Write entries
            node_to_idx = dict()
            for i, j in nx.dfs_edges(self, source=root):
                # Special Case: Root
                if len(node_to_idx) == 0:
                    write_entry(i, -1)

                # General Case: Non-Root
                write_entry(j, node_to_idx[i])

            # Finish
            filename = self.get_swc_id(i)
            filename = util.set_zip_path(zip_writer, filename, ".swc")
            zip_writer.writestr(filename, text_buffer.getvalue())

    # --- Helpers ---
    def path_length(self, max_depth=np.inf, root=None):
        """
        Computes the path length of the connected component that contains the
        given root node.

        Parameters
        ----------
        root : int
            Node that belongs to connected component to be searched.
        max_depth : float
            Maximum depth (in microns) to search in the connected component.
            Useful if only checking that a connected component is has path
            length above some threshold.

        Returns
        -------
        length : float
            Path length of the connected component containing the given root
            node.
        """
        length = 0
        for i, j in nx.dfs_edges(self, source=root):
            length += self.dist(i, j)
            if length > max_depth:
                break
        return length

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
        return geometry_util.dist(self.node_xyz[i], self.node_xyz[j])

    def get_voxel(self, i):
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

    def init_kdtree(self):
        """
        Initializes KD-Tree from node xyz coordinates.
        """
        self.kdtree = KDTree(self.node_xyz)

    def query_node(self, xyz):
        # this breaks if node is deleted after kdtree was built
        _, idx = self.kdtree.query(xyz)
        return idx
