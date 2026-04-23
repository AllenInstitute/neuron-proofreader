"""
Created on Wed July 2 14:00:00 2025

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Implementation of a custom subclass of Networkx.Graph called "SkeletonGraph".
The graph is constructed by reading and processing SWC files (i.e. neuron
fragments). It then stores the relevant information into the graph structure.

"""

from collections import defaultdict
from concurrent.futures import as_completed, ThreadPoolExecutor
from io import StringIO
from scipy.spatial import KDTree
from scipy.spatial import distance
from tqdm import tqdm

import networkx as nx
import numpy as np
import os
import zipfile as zf

from neuron_proofreader.utils import geometry_util, graph_util, img_util, util


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
        min_cable_length=0,
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
        min_cable_length : float, optional
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
        self.anisotropy = np.array(anisotropy)
        self.component_id_to_swc_id = dict()
        self.kdtree = None
        self.node_spacing = node_spacing
        self.soma_centroids = list()
        self.soma_component_ids = list()

        # Graph Loader
        anisotropy = anisotropy if use_anisotropy else (1.0, 1.0, 1.0)
        self.graph_loader = graph_util.GraphLoader(
            anisotropy=anisotropy,
            min_cable_length=min_cable_length,
            node_spacing=node_spacing,
            prune_depth=prune_depth,
            verbose=verbose,
        )

    # --- Load SWC Files ---
    def load(self, swc_pointer):
        """
        Loads SWC files into graph.

        Parameters
        ----------
        swc_pointer : str
            Object that points to SWC files to be loaded.
        """
        # Extract irreducible components from SWC files
        irreducibles = self.graph_loader(swc_pointer)

        # Initialize node attribute data structures
        num_nodes = graph_util.count_nodes(irreducibles)
        self.node_component_id = np.zeros((num_nodes), dtype=int)
        self.node_radius = np.zeros((num_nodes), dtype=np.float16)
        self.node_xyz = np.zeros((num_nodes, 3), dtype=np.float32)

        # Add irreducibles to graph
        component_id = 0
        while irreducibles:
            self.add_connected_component(irreducibles.pop(), component_id)
            component_id += 1

        self.check_swc_ids()
        self.set_kdtree()

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
            self.node_component_id[new_id] = component_id
            self.node_radius[new_id] = attrs["radius"]
            self.node_xyz[new_id] = attrs["xyz"]
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
        i, j = edge_id
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
                self.node_component_id[new_id] = component_id
                self.node_radius[new_id] = radius
                self.node_xyz[new_id] = xyz
        self.add_edge(new_id, end)

    # --- Load Somas ---
    def load_somas(self, soma_centroids):
        # Resize attributes
        num_components = nx.number_connected_components(self)
        num_nodes = self.number_of_nodes()
        num_somas = len(soma_centroids)

        self.resize_node_attr((num_nodes + num_somas), "node_component_id")
        self.resize_node_attr((num_nodes + num_somas), "node_radius")
        self.resize_node_attr((num_nodes + num_somas, 3), "node_xyz")

        # Add soma nodes
        for idx, xyz in enumerate(soma_centroids, start=1):
            # Create node ID
            node_id = self.number_of_nodes()
            assert node_id not in self.nodes

            # Connect closest component (if applicable)
            dist_i, i = self.kdtree.query(xyz)
            if dist_i < 25:
                self.add_edge(i, node_id)
                component_id = self.node_component_id[i]
                swc_id = self.node_swc_id(i)
            elif dist_i < 50:
                self.add_node(node_id)
                component_id = num_components + idx
                swc_id = f"soma-component-{idx}"
            else:
                continue

            # Store attributes
            self.component_id_to_swc_id[component_id] = swc_id
            self.node_component_id[node_id] = component_id
            self.node_radius[node_id] = 20
            self.node_xyz[node_id] = xyz
            self.soma_centroids.append(xyz)

        self.relabel_nodes()

    def connect_soma_fragments(self, max_dist=25):
        merge_cnt, somas_connected = 0, list()
        for soma_node in self.soma_nodes():
            # Find nearby nodes
            soma_xyz = self.node_xyz[soma_node]
            nodes = self.kdtree.query_ball_point(soma_xyz, max_dist)
            nodes = np.array(nodes, dtype=int)

            # Connect soma to nearby components
            for cid in np.unique(self.node_component_id[nodes]):
                soma_component_id = self.node_component_id[soma_node]
                if cid != self.node_component_id[soma_node]:
                    # Find node closest to soma
                    idxs = np.where(self.node_component_id[nodes] == cid)[0]
                    dists = np.sum(
                        (self.node_xyz[nodes[idxs]] - soma_xyz) ** 2, axis=1
                    )
                    node = nodes[idxs[np.argmin(dists)]]

                    # Connect closest node to soma
                    if not nx.has_path(self, node, soma_node):
                        self.add_edge(node, soma_node)
                        self.update_component_ids(soma_component_id, node)
                        merge_cnt += 1
                        somas_connected.append(soma_component_id)

        results = [
            f"# Somas Connected: {len(np.unique(somas_connected))}",
            f"# Connections Added: {merge_cnt}",
        ]
        return "\n".join(results)

    def remove_soma_merges(self):
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
        # Check whether to exit
        if len(self.soma_centroids) == 0:
            return ""

        # Extract soma info
        component_id_to_soma_nodes = defaultdict(set)
        somas_kdtree = KDTree(self.soma_centroids)
        for cnt, i in enumerate(self.soma_nodes(), start=1):
            component_id_to_soma_nodes[self.node_component_id[i]].add(i)

        # Check for connected components with multiple soma nodes
        n_soma_merges = 0
        for component_id, soma_nodes in component_id_to_soma_nodes.items():
            if len(soma_nodes) > 1 and len(soma_nodes) < 20:
                n_soma_merges += len(soma_nodes)
                rm_nodes = set()
                for i in self.find_connecting_path(list(soma_nodes)):
                    dist, _ = somas_kdtree.query(self.node_xyz[i])
                    if self.degree[i] > 2 and dist > 25:
                        rm_nodes.add(i)
                self.remove_nearby_nodes(rm_nodes)

        # Finish
        self.remove_small_components(relabel_nodes=False)
        self.relabel_nodes()
        results = [
            f"# Soma Fragments: {len(self.soma_centroids)}",
            f"# Soma Merges: {n_soma_merges}"
        ]
        return "\n".join(results)

    def remove_nearby_nodes(self, roots, max_dist=5.0):
        """
        Removes nodes within a given radius from a set of root nodes.

        Parameters
        ----------
        roots : List[int]
            Root nodes.
        max_dist : float, optional
            Maximum distance within which nodes are removed. Default is 5.0.
        """
        nodes = set()
        while len(roots) > 0:
            root = roots.pop()
            queue = [(root, 0)]
            visited = {root}
            while len(queue) > 0:
                # Visit node
                i, dist_i = queue.pop()
                visited.add(i)

                # Update queue
                for j in self.neighbors(i):
                    dist_j = dist_i + self.dist(i, j)
                    if j not in visited and dist_j <= max_dist:
                        queue.append((j, dist_j))
                    elif j not in visited and self.degree[j] > 2:
                        queue.append((j, dist_i))
            nodes = nodes.union(visited)
        self.remove_nodes_from(nodes)

    # --- Update Structure ---
    def check_swc_ids(self):
        visited = set()
        for nodes in map(list, nx.connected_components(self)):
            # Check if SWC ID exists
            swc_id = self.node_swc_id(nodes[0])
            if swc_id not in visited:
                visited.add(swc_id)
                continue

            # Find new SWC ID
            swc_ids = set(self.component_id_to_swc_id.values())
            while swc_id in visited and swc_id in swc_ids:
                swc_name, cnt = swc_id.split(".")
                swc_id = f"{swc_name}.{int(cnt) + 1}"

            # Update graph
            component_id = self.node_component_id[nodes[0]]
            self.component_id_to_swc_id[component_id] = swc_id
            visited.add(swc_id)

    def reassign_component_ids(self):
        """
        Reassigns component IDs for all connected components in the graph.
        """
        self.check_swc_ids()
        component_id_to_swc_id = dict()
        for i, nodes in enumerate(nx.connected_components(self), start=1):
            nodes = np.array(list(nodes), dtype=int)
            component_id_to_swc_id[i] = self.node_swc_id(nodes[0])
            self.node_component_id[nodes] = i
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
        self.add_nodes_from(new_node_ids)
        for i, j in old_edge_ids:
            self.add_edge(old_to_new[i], old_to_new[j])

        # Update attributes
        self.node_radius = self.node_radius[old_node_ids]
        self.node_xyz = self.node_xyz[old_node_ids]
        self.node_component_id = self.node_component_id[old_node_ids]

        self.reassign_component_ids()
        self.set_kdtree()
        assert len(self.node_xyz) == self.number_of_nodes()
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

    def remove_small_components(self, min_size=20, relabel_nodes=True):
        rm_nodes = list()
        for nodes in map(list, nx.connected_components(self)):
            length = self.cable_length(max_depth=min_size, root=nodes[0])
            if length < min_size:
                rm_nodes.extend(nodes)
        self.remove_nodes_from(rm_nodes)

    # --- Writer ---
    def to_zipped_swcs(self, zip_path, use_radius=False):
        """
        Writes the graph to a ZIP archive of SWC files, where each file
        corresponds to a single connected component.

        Parameters
        ----------
        zip_path : str
            Path to ZIP archive that SWC files are to be written to.
        use_radius : bool, optional
            Indication of whether to set radius as node radius or 2μm.
            Default is False.
        """
        self.check_swc_ids()
        with zf.ZipFile(zip_path, "w") as zipfile:
            for nodes in map(list, nx.connected_components(self)):
                self.component_to_zipped_swc(zipfile, nodes[0], use_radius)

    def to_zipped_swcs_multithreaded(self, output_dir, use_radius=True):
        """
        Writes the graph to a ZIP archive of SWC files using mutlithreading,
        where each file corresponds to a single connected component.

        Parameters
        ----------
        output_dir : str
            Path to directory that ZIP archives with SWC files are written to.
        use_radius : bool, optional
            Indication of whether to set radius as node radius or 2μm.
            Default is False.
        """

        def create_job(batch, i):
            zip_path = os.path.join(output_dir, f"{i}.zip")
            thread = executor.submit(
                self._batch_to_zipped_swcs,
                batch,
                zip_path,
                use_radius,
            )
            return thread

        # Set batch size
        n = nx.number_connected_components(self)
        batch_size = max(1, n / 1000) if n > 10**4 else n
        util.mkdir(output_dir)

        # Main
        self.check_swc_ids()
        with ThreadPoolExecutor() as executor:
            # Assign threads
            batch = list()
            threads = list()
            zip_cnt = 0
            for i, nodes in enumerate(nx.connected_components(self)):
                batch.append(nodes)
                if len(batch) >= batch_size:
                    threads.append(create_job(list(batch), zip_cnt))
                    batch = list()
                    zip_cnt += 1

            # Submit last batch
            if batch:
                threads.append(create_job(list(batch), zip_cnt + 1))

            # Watch progress
            pbar = tqdm(total=len(threads), desc="Write SWCs")
            for thread in as_completed(threads):
                thread.result()
                pbar.update(1)

    def _batch_to_zipped_swcs(self, nodes_list, zip_path, use_radius):
        with zf.ZipFile(zip_path, "w") as zipfile:
            for nodes in map(list, nodes_list):
                self.component_to_zipped_swc(zipfile, nodes[0], use_radius)

    def component_to_zipped_swc(self, zipfile, root, use_radius=False):
        """
        Writes the connected component containing the given root node to a
        zipped SWC file.

        Parameters
        ----------
        zipfile : zipfile.ZipFile
            ZipFile object that will store the generated SWC file.
        root : int
            Root node of connected component to be written to an SWC file.
        ususe_radius : bool, optional
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
            x, y, z = self.node_xyz[node]
            r = self.node_radius[node] if use_radius else 2
            node_id = cnt
            parent_id = node_to_idx[parent]
            node_to_idx[node] = node_id
            text_buffer.write(f"\n{node_id} 2 {x} {y} {z} {r} {parent_id}")

        # Main
        with StringIO() as text_buffer:
            # Preamble
            text_buffer.write("# id, type, z, y, x, r, pid")

            # Special case: root
            cnt = 1
            node_to_idx = defaultdict(lambda: -1)
            write_entry(root, -1)

            # General case: non-root
            for i, j in nx.dfs_edges(self, source=root):
                cnt += 1
                write_entry(j, i)

            # Finish
            name = util.set_filename_in_zip(zipfile, self.node_swc_id(root))
            util.set_filename_in_zip(zipfile, self.node_swc_id(root))
            zipfile.writestr(name, text_buffer.getvalue())

    # --- Helpers ---
    def branching_nodes(self):
        """
        Gets all branching nodes in the graph.

        Returns
        -------
        List[int]
            Branching nodes in the graph.
        """
        return [i for i in self.nodes if self.degree[i] > 2]

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
        """
        Clips skeletons to the bounding box defined by an origin and shape
        stored in the file at the given path.

        Parameters
        ----------
        metadata_path : str
            Path to JSON file containing origin and shape of bounding box to
            clip skeletons to.
        """
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

    def clip_to_skeleton(self, gt_graph, dist):
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

    def component_id_from_swc_id(self, query_swc_id):
        """
        Gets the component ID corresponding to the given SWC ID.

        Parameters
        ----------
        query_swc_id : str
            SWC ID to query.

        Returns
        -------
        component_id : int
            Component ID corresponding to the given SWC ID.
        """
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

    def directed_path(self, start_node, next_node, max_depth=np.inf):
        queue = [(next_node, 0)]
        visited = [start_node, next_node]
        while queue:
            # Visit node
            i, dist_i = queue.pop()
            if self.degree[i] != 2:
                return visited

            # Update queue
            for j in self.neighbors(i):
                dist_j = dist_i + self.dist(i, j)
                if dist_j < max_depth and j not in visited:
                    queue.append((j, dist_j))
                    visited.append(j)
        return visited

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

    def find_connecting_path(self, nodes):
        """
        Finds path connecting the given set of nodes.

        Parameters
        ----------
        nodes : List[int]
            Nodes to be connected.

        Returns
        -------
        path : Set[int]
            Set of node IDs that connects the given nodes.
        """
        connecting_nodes = set()
        for target in nodes[1:]:
            path = nx.shortest_path(self, source=nodes[0], target=target)
            connecting_nodes = connecting_nodes.union(set(path))
        return connecting_nodes

    def get_irreducible_edge(self, node):
        """
        Finds the irreducible edge containing the given node.

        Parameters
        ----------
        node : int
            Node ID.

        Returns
        -------
        edge : Tuple[int]
            Irreducible edge containing the given node.
        """
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

    def find_nearby_branching_node(self, root, max_depth=16):
        queue = [(root, 0)]
        visited = {root}
        while queue:
            # Visit node
            i, d_i = queue.pop(0)
            if self.degree[i] >= 3:
                return i

            # Update queue
            for j in self.neighbors(i):
                d_j = d_i + self.dist(i, j)
                if j not in visited and d_j < max_depth:
                    queue.append((j, d_j))
                    visited.add(j)
        return root

    def irreducible_nodes(self):
        """
        Gets the set of irreducible nodes.

        Returns
        -------
        Set[int]
            Irreducible nodes.
        """
        return {i for i in map(int, self.nodes) if self.degree[i] != 2}

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
        return self.node_swc_id(node).split(".")[0]

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
        swc_id = self.component_id_to_swc_id[component_id]
        return swc_id if "." in swc_id else f"{swc_id}.0"

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

    def nodes_with_segment_id(self, query_segment_id):
        """
        Gets all nodes with the given segment ID.

        Parameters
        ----------
        query_segment_id : int
            Unique identifier of a segment to be queried.

        Returns
        -------
        Set[int]
            Nodes with the given segment ID.
        """
        nodes = set()
        query_id = str(query_segment_id)
        for swc_id in self.swc_ids():
            segment_id = swc_id.split(".")[0]
            if segment_id == query_id:
                component_id = self.component_id_from_swc_id(swc_id)
                nodes = nodes.union(self.nodes_with_component_id(component_id))
        return nodes

    def nodes_within_distance(self, root, max_dist):
        """
        Gets nodes connected to the given root node up to a certain depth.

        Parameters
        ----------
        root : int
            Node ID and root of search.
        max_dist : float
            Maximum distance (in microns) between returned nodes and root.

        Returns
        -------
        visited : List[int]
            Nodes connected to the given root node up to a certain depth.
        """
        queue = [(root, 0)]
        visited = {root}
        while queue:
            # Visit node
            i, dist_i = queue.pop()

            # Populate queue
            for j in self.neighbors(i):
                dist_j = dist_i + self.dist(i, j)
                if dist_j < max_dist and j not in visited:
                    queue.append((j, dist_j))
                    visited.add(j)
        return list(visited)

    def path_from_leaf(self, leaf, max_depth=np.inf):
        """
        Gets the path of nodes emanating from the given leaf up to a certain
        depth if "max_depth" is provided. Note that the path is truncated
        before reaching a branching node.

        Parameters
        ----------
        leaf : int
            Node ID and start of path.
        max_depth : float, optional
            Maximum length (in microns) of path returned. Default is np.inf.

        Returns
        -------
        path : List[int]
            Path of nodes emanating from the given leaf up to a certain depth
            if "max_depth" is provided.
        """
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
        if len(path) > 1:
            diffs = self.node_xyz[path[1:]] - self.node_xyz[path[:-1]]
            return np.sqrt(np.sum(diffs**2))
        else:
            return 0

    def path_thru_node(self, i, max_depth=np.inf):
        if self.degree[i] == 0:
            return [i]
        elif self.degree[i] == 1:
            return self.path_from_leaf(i, max_depth)
        else:
            assert self.degree[i] == 2
            j, k = self.neighbors(i)
            path_ij = self.directed_path(i, j, max_depth=max_depth)
            path_ik = self.directed_path(i, k, max_depth=max_depth)
            return path_ij[::-1] + path_ik[1:]

    def remove_high_risk_merges(self, max_dist=7):
        """
        Removes high risk merge sites from a graph, which are defined to be
        either (1) two branching points within "max_dist" or (2) branching
        point with degree 4+. Note: if soma locations are provided, we skip
        branching points within 300um of a soma.

        Parameters
        ----------
        max_dist : float, optional
            Maximum distance between branching points that qualifies a site to
            be considered "high risk". Default is 7.
        """
        # Initializations
        branching_nodes = set(self.branching_nodes())
        soma_nodes = np.array(self.soma_nodes(), dtype=int)
        if len(soma_nodes) > 0:
            somas_kdtree = KDTree(self.node_xyz[soma_nodes])

        # Iterate over branching nodes
        cnt = 0
        rm_nodes = set()
        while branching_nodes:
            # Set root of search
            root = branching_nodes.pop()

            # Check if close to soma
            if len(soma_nodes):
                dist, _ = somas_kdtree.query(self.node_xyz[root])
                if dist < 300:
                    continue

            # Search for high risk sites
            hit_branching_nodes = set()
            queue = [(root, 0)]
            visited = {root}
            while queue:
                # Visit node
                i, dist_i = queue.pop()
                if self.degree[i] > 2 and i != root:
                    hit_branching_nodes.add(i)

                # Update queue
                for j in self.neighbors(i):
                    dist_j = dist_i + self.dist(i, j)
                    if j not in visited and dist_j < max_dist:
                        queue.append((j, dist_j))
                        visited.add(j)

            # Determine whether to remove visited nodes
            if hit_branching_nodes or self.degree(root) > 3:
                rm_nodes = rm_nodes.union(visited)
                branching_nodes -= hit_branching_nodes
                cnt += 1
        self.remove_nodes(rm_nodes)
        return f"# High Risk Merges: {cnt}"

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

    def set_kdtree(self):
        """
        Initializes KD-Tree from node xyz coordinates.
        """
        self.kdtree = KDTree(self.node_xyz)

    def soma_nodes(self):
        soma_nodes = list()
        for dist_i, i in map(self.kdtree.query, self.soma_centroids):
            if dist_i < 5:
                soma_nodes.append(i)
        return soma_nodes

    def summary(self, prefix=""):
        """
        Generate a human-readable summary of the graph.

        Parameters
        ----------
        prefix : str, optional
            Optional string to prepend to the summary title.

        Returns
        -------
        summary : str
            Formatted multi-line string containing:
            - Graph Name
            - Number of connected components
            - Number of nodes
            - Number of edges
            - Memory consumption (in GBs)
        """
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

    def swc_ids(self):
        """
        Gets the set of all unique SWC IDs of nodes in the graph.

        Returns
        -------
        Set[str]
            Set of all unique SWC IDs of nodes in the graph.
        """
        return set(self.component_id_to_swc_id.values())

    def tangent_from_leaf(self, leaf, max_depth=np.inf):
        """
        Computes the tangent vector of the path emanating from the given leaf.

        Parameters
        ----------
        leaf : int
            Node ID and starting point of path extracted.
        max_depth : float
            Maximum depth (in microns) of path extracted.

        Returns
        -------
        numpy.ndarray
            Tangent vector of the path emanating from the given leaf.
        """
        path = self.path_from_leaf(leaf, max_depth=max_depth)
        return geometry_util.tangent(self.node_xyz[np.array(path)])
