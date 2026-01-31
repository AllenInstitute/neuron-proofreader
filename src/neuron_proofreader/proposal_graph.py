"""
Created on Sat July 15 9:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Implementation of a custom subclass of Networkx.Graph called "ProposalGraph".
After initializing an instance of this subclass, the graph is built by reading
and processing SWC files (i.e. neuron fragments). It then stores the relevant
information into the graph structure.

"""

from collections import defaultdict
from concurrent.futures import as_completed, ThreadPoolExecutor
from io import StringIO
from scipy.spatial import KDTree
from tqdm import tqdm

import networkx as nx
import numpy as np
import os
import zipfile

from neuron_proofreader.skeleton_graph import SkeletonGraph
from neuron_proofreader.split_proofreading import (
    groundtruth_generation
)
from neuron_proofreader.split_proofreading.proposal_generation import (
    ProposalGenerator,
    trim_endpoints_at_proposal
)
from neuron_proofreader.utils import (
    geometry_util as geometry,
    graph_util as gutil,
    util,
)


class ProposalGraph(SkeletonGraph):
    """
    Custom subclass of NetworkX.Graph constructed from neuron fragments. The
    graph's nodes are irreducible, meaning each node has either degree 1
    (leaf) or degree 3+ (branching points). Each edge has an attribute that
    stores a dense path of points connecting irreducible nodes. Additionally,
    the graph has an attribute called "proposals", which is a set of potential
    connections between pairs of neuron fragments.
    """

    def __init__(
        self,
        anisotropy=(1.0, 1.0, 1.0),
        gt_path=None,
        max_proposals_per_leaf=3,
        min_size=0,
        min_size_with_proposals=0,
        node_spacing=1,
        prune_depth=20.0,
        remove_high_risk_merges=False,
        segmentation_path=None,
        soma_centroids=None,
        verbose=True,
    ):
        """
        Instantiates a ProposalGraph object.

        Parameters
        ----------
        anisotropy : Tuple[int], optional
            Image to physical coordinates scaling factors to account for the
            anisotropy of the microscope. Default is (1.0, 1.0, 1.0).
        min_size : float, optional
            Minimum path length of fragments loaded into graph. Default is 0.
        min_size_with_proposals : float, optional
            Minimum fragment path length required for proposals. Default is 0.
        node_spacing : float, optional
            Distance between points in edges.
        prune_depth : float, optional
            Branches with length less than "prune_depth" microns are removed.
            Default is 20um.
        remove_high_risk_merges : bool, optional
            Indication of whether to remove high risk merge sites (i.e. close
            branching points). Default is False.
        segmentation_path : str, optional
            Path to segmentation stored in GCS bucket. Default is None.
        soma_centroids : List[Tuple[float]] or None, optional
            Phyiscal coordinates of soma centroids. Default is None.
        verbose : bool, optional
            Indication of whether to display a progress bar while building
            graph. Default is True.
        """
        # Call parent class
        super().__init__()

        # Instance attributes - Graph
        self.anisotropy = anisotropy
        self.component_id_to_swc_id = dict()
        self.gt_path = gt_path
        self.leaf_kdtree = None
        self.soma_ids = set()
        self.verbose = verbose
        self.xyz_to_edge = dict()

        # Instance attributes - Proposals
        self.gt_accepts = set()
        self.merged_ids = set()
        self.n_merges_blocked = 0
        self.n_proposals_blocked = 0
        self.node_proposals = defaultdict(set)
        self.proposals = set()

        self.proposal_generator = ProposalGenerator(
            self,
            max_proposals_per_leaf=max_proposals_per_leaf,
            min_size_with_proposals=min_size_with_proposals
        )

        # Graph Loader
        self.graph_loader = gutil.GraphLoader(
            anisotropy=anisotropy,
            min_size=min_size,
            node_spacing=node_spacing,
            prune_depth=prune_depth,
            remove_high_risk_merges=remove_high_risk_merges,
            segmentation_path=segmentation_path,
            soma_centroids=soma_centroids,
            verbose=verbose,
        )

    def load(self, swc_pointer):
        """
        Loads fragments into "self" by reading SWC files stored on either the
        cloud or local machine, then extracts the irreducible components from
        from each SWC file.

        Parameters
        ----------
        swc_pointer : Any
            Pointer to SWC files to be loaded, see "swc_util.Reader" for
            documentation.
        """
        # Extract irreducible components from SWC files
        irreducibles = self.graph_loader(swc_pointer)
        n = np.sum([len(irr["nodes"]) for irr in irreducibles])

        # Initialize node attribute data structures
        self.node_component_id = np.zeros((n), dtype=int)
        self.node_radius = np.zeros((n), dtype=np.float16)
        self.node_xyz = np.zeros((n, 3), dtype=np.float32)

        # Add irreducibles to graph
        component_id = 0
        while irreducibles:
            self.add_connected_component(irreducibles.pop(), component_id)
            component_id += 1

    # --- Update Structure ---
    def add_connected_component(self, irreducibles, component_id):
        """
        Adds the irreducibles from a single connected component to "self".

        Parameters
        ----------
        irreducibles : dict
            Dictionary containing the irreducibles of a connected component to
            add to "self". This dictionary must contain the keys: "nodes",
            "edges", "swc_id", and "is_soma".
        component_id : int
            Connected component ID used to map node IDs back to SWC IDs via
            "self.component_id_to_swc_id".
        """
        # Component ID
        self.component_id_to_swc_id[component_id] = irreducibles["swc_id"]
        if irreducibles["is_soma"]:
            self.soma_ids.add(component_id)

        # Add irreducibles
        node_id_mapping = self._add_nodes(irreducibles["nodes"], component_id)
        for (i, j), attrs in irreducibles["edges"].items():
            edge_id = (node_id_mapping[i], node_id_mapping[j])
            self._add_edge(edge_id, attrs)

    def _add_edge(self, edge_id, attrs):
        """
        Adds an edge to "self".

        Parameters
        ----------
        edge : Tuple[int]
            Edge to be added.
        attrs : dict
            Dictionary of attributes of "edge" obtained from an SWC file.
        """
        i, j = tuple(edge_id)
        self.add_edge(i, j, radius=attrs["radius"], xyz=attrs["xyz"])
        self.xyz_to_edge.update({tuple(xyz): edge_id for xyz in attrs["xyz"]})

    def connect_soma_fragments(self, soma_centroids):
        merge_cnt = 0
        soma_cnt = 0
        self.set_kdtree()
        for soma_xyz in soma_centroids:
            node_ids = self.find_fragments_near_xyz(soma_xyz, 25)
            if len(node_ids) > 1:
                # Find closest node to soma location
                soma_cnt += 1
                best_dist = np.inf
                best_node = None
                for i in node_ids:
                    dist = geometry.dist(soma_xyz, self.node_xyz[i])
                    if dist < best_dist:
                        best_dist = dist
                        best_node = i
                soma_component_id = self.node_component_id[best_node]
                self.soma_ids.add(soma_component_id)
                node_ids.remove(best_node)

                # Merge fragments to soma
                soma_xyz = self.node_xyz[best_node]
                for i in node_ids:
                    attrs = {
                        "radius": np.array([2, 2]),
                        "xyz": np.array([soma_xyz, self.node_xyz[i]]),
                    }
                    self._add_edge((best_node, i), attrs)
                    self.update_component_ids(soma_component_id, i)
                    merge_cnt += 1

        # Summarize results
        results = [f"# Somas Connected: {soma_cnt}"]
        results.append(f"# Soma Fragments Merged: {merge_cnt}")
        return "\n".join(results)

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
        old_irr_edge_ids = self.irreducible.edges
        edge_attrs = {(i, j): data for i, j, data in self.edges(data=True)}

        # Reset graph
        self.clear()
        self.xyz_to_edge = dict()
        for (i, j) in old_edge_ids:
            edge_id = (int(old_to_new[i]), int(old_to_new[j]))
            self._add_edge(edge_id, edge_attrs[(i, j)])

        self.irreducible.clear()
        for (i, j) in old_irr_edge_ids:
            self.irreducible.add_edge(old_to_new[i], old_to_new[j])

        # Update attributes
        self.node_radius = self.node_radius[old_node_ids]
        self.node_xyz = self.node_xyz[old_node_ids]
        self.node_component_id = self.node_component_id[old_node_ids]

        self.reassign_component_ids()

    def remove_line_fragment(self, i, j):
        """
        Deletes nodes "i" and "j" from "graph", where these nodes form a connected
        component.

        Parameters
        ----------
        i : int
            Node to be removed.
        j : int
            Node to be removed.
        """
        # Remove xyz entries
        self.xyz_to_edge.pop(tuple(self.node_xyz[i]), None)
        self.xyz_to_edge.pop(tuple(self.node_xyz[j]), None)
        for xyz in self.edges[i, j]["xyz"]:
            self.xyz_to_edge.pop(tuple(xyz), None)

        # Remove nodes
        del self.component_id_to_swc_id[self.node_component_id[i]]
        self.remove_nodes_from([i, j])

    # -- KDTree --
    def set_kdtree(self, node_type=None):
        """
        Builds a KD-Tree from the xyz coordinates of the subset of nodes
        indicated by "node_type".

        Parameters
        ----------
        node_type : None or str
            Type of node used to build kdtree.
        """
        if node_type == "leaf":
            self.leaf_kdtree = self.get_kdtree(node_type=node_type)
        elif node_type == "branching":
            self.branching_kdtree = self.get_kdtree(node_type=node_type)
        elif node_type == "proposal":
            self.proposal_kdtree = self.get_kdtree(node_type=node_type)
        else:
            self.kdtree = self.get_kdtree()

    def get_kdtree(self, node_type=None):
        """
        Builds KD-Tree from xyz coordinates across all nodes and edges.

        Parameters
        ----------
        node_type : None or str, optional
            Type of nodes used to build kdtree. Default is None.

        Returns
        -------
        KDTree
            KD-Tree generated from xyz coordinates across all nodes and edges.
        """
        # Get xyz coordinates
        if node_type == "leaf":
            leafs = np.array(self.get_leafs(), dtype=int)
            return KDTree(self.node_xyz[leafs])
        elif node_type == "branching":
            branchings = np.array(self.get_branchings(), dtype=int)
            return KDTree(self.node_xyz[branchings])
        elif node_type == "proposal":
            xyz_set = set()
            for p in self.proposals:
                xyz_i, xyz_j = self.proposal_attr(p, attr="xyz")
                xyz_set = xyz_set.union({tuple(xyz_i), tuple(xyz_j)})
            return KDTree(list(xyz_set))
        else:
            return KDTree(list(self.xyz_to_edge.keys()))

    # --- Proposal Generation ---
    def generate_proposals(self, search_radius):
        """
        Generates proposals from leaf nodes.

        Parameters
        ----------
        search_radius : float
            Search radius used to generate proposals.
        gt_graph : networkx.Graph, optional
            Ground truth graph. Default is None.
        """
        # Proposal pipeline
        proposals = self.proposal_generator(search_radius)
        self.search_radius = search_radius
        self.store_proposals(proposals)
        self.trim_proposals()

        # Set groundtruth
        if self.gt_path:
            gt_graph = ProposalGraph(anisotropy=self.anisotropy)
            gt_graph.load(self.gt_path)
            self.gt_accepts = groundtruth_generation.run(gt_graph, self)

    def add_proposal(self, i, j):
        """
        Adds proposal between nodes "i" and "j".

        Parameters
        ----------
        i : int
            Node ID.
        j : int
            Node ID
        """
        proposal = frozenset({i, j})
        self.node_proposals[i].add(j)
        self.node_proposals[j].add(i)
        self.proposals.add(proposal)

    def store_proposals(self, proposals):
        self.node_proposals = defaultdict(set)
        for proposal in proposals:
            i, j = tuple(proposal)
            self.add_proposal(i, j)

    def remove_proposal(self, proposal):
        """
        Removes an existing proposal between two nodes.

        Parameters
        ----------
        proposal : Frozenset[int]
            Pair of node IDs corresponding to a proposal.
        """
        i, j = tuple(proposal)
        self.node_proposals[i].remove(j)
        self.node_proposals[j].remove(i)
        self.proposals.remove(proposal)

    def trim_proposals(self):
        for proposal in self.list_proposals():
            is_simple = self.is_simple(proposal)
            is_single = self.is_single_proposal(proposal)
            if is_simple and is_single:
                trim_endpoints_at_proposal(self, proposal)

    def is_single_proposal(self, proposal):
        """
        Determines whether "proposal" is the only proposal generated for the
        corresponding nodes.

        Parameters
        ----------
        proposal : Frozenset[int]
            Pair of node IDs corresponding to a proposal.

        Returns
        -------
        bool
            Indiciation of "proposal" is the only proposal generated for the
            corresponding nodes.
        """
        i, j = tuple(proposal)
        single_i = len(self.node_proposals[i]) == 1
        single_j = len(self.node_proposals[j]) == 1
        return single_i and single_j

    def list_proposals(self):
        """
        Lists proposals in self.

        Returns
        -------
        List[Frozenset[int]]
            Proposals.
        """
        return list(self.proposals)

    # --- Proposal Helpers ---
    def get_sorted_proposals(self):
        """
        Return proposals sorted by physical length.

        Returns
        -------
        List[Frozenset[int]]
            List of proposals sorted by phyiscal length.
        """
        proposals = self.list_proposals()
        lengths = [self.proposal_length(p) for p in proposals]
        return [proposals[i] for i in np.argsort(lengths)]

    def n_proposals(self):
        """
        Counts the number of proposals in the graph.

        Returns
        -------
        int
            Number of proposals in the graph.
        """
        return len(self.proposals)

    def is_simple(self, proposal):
        """
        Checks if both nodes in a proposal are leafs.

        Parameters
        ----------
        proposal : Frozenset[int]
            Pair of nodes that form a proposal.

        Returns
        -------
        bool
            Indication of whether both nodes in a proposal are leafs.
        """
        i, j = tuple(proposal)
        return True if self.degree[i] == 1 and self.degree[j] == 1 else False

    def simple_proposals(self):
        return set([p for p in self.proposals if self.is_simple(p)])

    def complex_proposals(self):
        return set([p for p in self.proposals if not self.is_simple(p)])

    def proposal_length(self, proposal):
        return self.dist(*tuple(proposal))

    def proposal_midpoint(self, proposal):
        i, j = tuple(proposal)
        return geometry.midpoint(self.node_xyz[i], self.node_xyz[j])

    def proposal_attr(self, proposal, key):
        """
        Gets the attributes of nodes in "proposal".

        Parameters
        ----------
        proposal : Frozenset[int]
            Pair of nodes that form a proposal.
        key : str
            Name of attribute to be returned.

        Returns
        -------
        numpy.ndarray
            Attributes of nodes in "proposal".
        """
        i, j = tuple(proposal)
        if key == "xyz":
            return np.array([self.node_xyz[i], self.node_xyz[j]])
        elif key == "radius":
            return np.array([self.node_radius[i], self.node_radius[j]])

    def proposal_avg_radii(self, proposal):
        i, j = tuple(proposal)
        radii_i = self.edge_attr(i, key="radius")
        radii_j = self.edge_attr(j, key="radius")
        return np.array([avg_radius(radii_i), avg_radius(radii_j)])

    def proposal_directionals(self, proposal, depth):
        # Extract points along branches
        i, j = tuple(proposal)
        xyz_list_i = self.truncated_edge_attr_xyz(i, depth)
        xyz_list_j = self.truncated_edge_attr_xyz(j, depth)
        origin = self.proposal_midpoint(proposal)

        # Compute tangent vectors - branches
        direction_i = geometry.get_directional(xyz_list_i, origin, depth)
        direction_j = geometry.get_directional(xyz_list_j, origin, depth)
        direction = geometry.tangent(self.proposal_attr(proposal, "xyz"))
        if np.isnan(direction).any():
            direction[0] = 0
            direction[1] = 0

        # Compute features
        dot_i = abs(np.dot(direction, direction_i))
        dot_j = abs(np.dot(direction, direction_j))
        if self.is_simple(proposal):
            dot_ij = np.dot(direction_i, direction_j)
        else:
            dot_ij = np.dot(direction_i, direction_j)
            if not self.is_simple(proposal):
                dot_ij = max(dot_ij, -dot_ij)
        return np.array([dot_i, dot_j, dot_ij])

    def truncated_edge_attr_xyz(self, i, depth):
        xyz_path_list = self.edge_attr(i, "xyz")
        return [geometry.truncate_path(path, depth) for path in xyz_path_list]

    def merge_proposal(self, proposal):
        i, j = tuple(proposal)
        if self.is_mergeable(i, j):
            # Update attributes
            attrs = {
                "radius": self.node_radius[np.array([i, j], dtype=int)],
                "xyz": self.node_xyz[np.array([i, j], dtype=int)]
            }
            self.node_radius[i] = 5.3141592
            self.node_radius[j] = 5.3141592

            # Update component_ids
            self.merged_ids.add((self.get_swc_id(i), self.get_swc_id(j)))
            if self.is_soma(i):
                component_id = self.node_component_id[i]
                self.update_component_ids(component_id, j)
            else:
                component_id = self.node_component_id[j]
                self.update_component_ids(component_id, i)

            # Update graph
            self._add_edge((i, j), attrs)
            self.proposals.remove(proposal)
        else:
            self.n_merges_blocked += 1

    def is_mergeable(self, i, j):
        one_leaf = self.degree[i] == 1 or self.degree[j] == 1
        branching = self.degree[i] > 2 or self.degree[j] > 2
        somas_check = not (self.is_soma(i) and self.is_soma(j))
        return somas_check and (one_leaf and not branching)

    def update_component_ids(self, component_id, root):
        """
        Updates the component_id of all nodes connected to "root".

        Parameters
        ----------
        component_id : str
            Connected component id.
        root : int
            Node ID
        """
        queue = [root]
        visited = set(queue)
        while len(queue) > 0:
            i = queue.pop()
            self.node_component_id[i] = component_id
            visited.add(i)
            for j in [j for j in self.neighbors(i) if j not in visited]:
                queue.append(j)

    def n_nearby_leafs(self, proposal, radius):
        """
        Counts the number of nearby leaf nodes within a specified radius from
        a proposal.

        Parameters
        ----------
        proposal : frozenset
            Pproposal for which nearby leaf nodes are to be counted.
        radius : float
            The radius within which to search for nearby leaf nodes.

        Returns
        -------
        int
            Number of nearby leaf nodes within a specified radius from
            a proposal.
        """
        xyz = self.proposal_midpoint(proposal)
        return len(geometry.query_ball(self.leaf_kdtree, xyz, radius)) - 1

    # --- Helpers ---
    def node_attr(self, i, key):
        if key == "xyz":
            return self.node_xyz[i]
        elif key == "radius":
            return self.node_radius[i]
        else:
            return self.nodes[i][key]

    def edge_attr(self, i, key="xyz", ignore=False):
        """
        Gets the edge attribute specified by "key" for all edges connected to
        the given node.

        Parameters
        ----------
        i : int
            Node for which the edge attributes are to be retrieved.
        key : str, optional
            Key specifying the type of edge attribute to retrieve. The default
            is "xyz".
        ignore : bool, optional
            If True, it will only consider direct neighbors of node "i". If
            False, the method will follow add the edge attributes along the
            path of chain-like connections from node "i" to its neighbors,
            provided that the neighbor nodes have degree 2.

        Returns
        -------
        List[numpy.ndarray]
            Edge attribute specified by "key" for all edges connected to the
            given node.
        """
        attrs = list()
        for j in self.neighbors(i):
            attr_ij = self.orient_edge_attr((i, j), i, key=key)
            if not ignore:
                root = i
                while self.degree[j] == 2:
                    k = [k for k in self.neighbors(j) if k != root][0]
                    attr_jk = self.orient_edge_attr((j, k), j, key=key)
                    if key == "xyz":
                        attr_ij = np.vstack([attr_ij, attr_jk])
                    else:
                        attr_ij = np.concatenate((attr_ij, attr_jk))
                    root = j
                    j = k
            attrs.append(attr_ij)
        return attrs

    def edge_length(self, edge):
        length = 0
        for i in range(1, len(self.edges[edge]["xyz"])):
            length += geometry.dist(
                self.edges[edge]["xyz"][i], self.edges[edge]["xyz"][i - 1]
            )
        return length

    def find_fragments_near_xyz(self, query_xyz, max_dist):
        hits = dict()
        xyz_list = geometry.query_ball(self.kdtree, query_xyz, max_dist)
        for xyz in xyz_list:
            i, j = self.xyz_to_edge[tuple(xyz)]
            dist_i = geometry.dist(self.node_xyz[i], query_xyz)
            dist_j = geometry.dist(self.node_xyz[j], query_xyz)
            hits[self.node_component_id[i]] = i if dist_i < dist_j else j
        return list(hits.values())

    def is_soma(self, i):
        """
        Check whether a node belongs to a component containing a soma.

        Parameters
        ----------
        i : str
            Node ID.

        Returns
        -------
        bool
            True if the node belongs to a connected component with a soma;
            False otherwise.
        """
        return self.node_component_id[i] in self.soma_ids

    def orient_edge_attr(self, edge, i, key="xyz"):
        node_attr = self.node_attr(i, key)
        if (self.edges[edge][key][0] == node_attr).all():
            return self.edges[edge][key]
        else:
            return np.flip(self.edges[edge][key], axis=0)

    def xyz_to_component_id(self, xyz, return_node=False):
        if tuple(xyz) in self.xyz_to_edge.keys():
            edge = self.xyz_to_edge[tuple(xyz)]
            return self.node_component_id[edge[0]]
        else:
            return None

    # --- SWC Writer ---
    def to_zipped_swcs(self, swc_dir, preserve_radius=False, sampling_rate=1):
        # Initializations
        n = nx.number_connected_components(self)
        batch_size = max(1, n // 1000) if n > 10 ** 4 else n
        util.mkdir(swc_dir)

        # Main
        zip_cnt = 0
        with ThreadPoolExecutor() as executor:
            # Assign threads
            batch = list()
            threads = list()
            for i, nodes in enumerate(nx.connected_components(self)):
                batch.append(nodes)
                if len(batch) >= batch_size or i == n - 1:
                    # Zip batch
                    zip_path = os.path.join(swc_dir, f"{zip_cnt}.zip")
                    threads.append(
                        executor.submit(
                            self.batch_to_zipped_swcs,
                            batch,
                            zip_path,
                            preserve_radius,
                            sampling_rate
                        )
                    )

                    # Reset batch
                    batch = list()
                    zip_cnt += 1

            # Watch progress
            pbar = tqdm(total=len(threads), desc="Write SWCs")
            for _ in as_completed(threads):
                pbar.update(1)

    def batch_to_zipped_swcs(
        self, nodes_list, zip_path, preserve_radius=False, sampling_rate=1
    ):
        with zipfile.ZipFile(zip_path, "w") as zip_writer:
            for nodes in nodes_list:
                self.nodes_to_zipped_swc(
                    zip_writer,
                    nodes,
                    preserve_radius=preserve_radius,
                    sampling_rate=sampling_rate
                )

    def nodes_to_zipped_swc(
        self,
        zip_writer,
        nodes,
        preserve_radius=False,
        sampling_rate=1,
    ):
        with StringIO() as text_buffer:
            # Preamble
            text_buffer.write("# id, type, x, y, z, r, pid")

            # Write entries
            n_entries = 0
            node_to_idx = dict()
            for i, j in nx.dfs_edges(self.subgraph(nodes)):
                # Root entry
                if len(node_to_idx) == 0:
                    # Get attributes
                    x, y, z = tuple(self.node_xyz[i])
                    if preserve_radius:
                        r = self.node_radius[i]
                    else:
                        r = 6 if self.node_radius[i] == 5.3141592 else 2

                    # Write entry
                    text_buffer.write(f"\n1 2 {x} {y} {z} {r} -1")
                    node_to_idx[i] = 1
                    n_entries += 1

                # Remaining entries
                parent = node_to_idx[i]
                text_buffer, n_entries = self.branch_to_zip(
                    text_buffer,
                    n_entries,
                    i,
                    j,
                    parent,
                    preserve_radius=preserve_radius,
                    sampling_rate=sampling_rate
                )
                node_to_idx[j] = n_entries

            # Write SWC file
            filename = self.get_swc_id(i)
            filename = util.set_zip_path(zip_writer, filename, ".swc")
            zip_writer.writestr(filename, text_buffer.getvalue())

    def branch_to_zip(
        self,
        text_buffer,
        n_entries,
        i,
        j,
        parent,
        preserve_radius=False,
        sampling_rate=1,
    ):
        branch_xyz = self.orient_edge_attr((i, j), i, "xyz")
        branch_radius = self.orient_edge_attr((i, j), i, "radius")
        for k in util.spaced_idxs(len(branch_xyz), sampling_rate):
            # Get attributes
            node_id = n_entries + 1
            parent = n_entries if k > 1 else parent
            x, y, z = tuple(branch_xyz[k])
            if preserve_radius:
                r = branch_radius[k]
            else:
                r = 6 if branch_radius[k] == 5.3141592 else 2

            # Write entry
            text_buffer.write(f"\n{node_id} 2 {x} {y} {z} {r} {parent}")
            n_entries += 1
        return text_buffer, n_entries


# -- Helpers --
def avg_radius(radii_list):
    avg = 0
    for radii in radii_list:
        end = max(min(16, len(radii) - 1), 1)
        avg += np.mean(radii[0:end]) / len(radii_list)
    return avg
