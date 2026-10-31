"""
Created on Sat July 15 9:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code that generates edge proposals for a given graph.

"""

from collections import defaultdict
from scipy.spatial import KDTree
from tqdm import tqdm

import numpy as np

from neuron_proofreader.utils import geometry_util


class ProposalGenerator:
    """
    A class for generating proposals between fragments in a graph.
    """

    def __init__(
        self,
        graph,
        allow_nonleaf_proposals=False,
        cone_depth=64,
        max_angle=None,
        max_attempts=2,
        max_proposals_per_leaf=3,
        min_size_with_proposals=0,
        search_scaling_factor=1.5,
    ):
        """
        Instantiates a ProposalGenerator object.

        Parameters
        ----------
        graph : ProposalGraph
            Graph that proposals will be generated for.
        allow_nonleaf_proposals : bool, optional
            Indication of whether to generate proposals between leaf and nodes
            with degree 2. Default is False.
        cone_depth : float, optional
            Path length (in microns) from a leaf used to estimate its tangent
            when "max_angle" is set. Default is 64.
        max_angle : float, optional
            If set, proposals are only generated from leaves whose branch is
            straight and only to candidates within this angle (in degrees)
            of the leaf's outward tangent. Default is None.
        max_attempts : int, optional
            Number of attempts made to generate proposals from a node with
            increasing search radii. Default is 2.
        max_proposals_per_leaf : bool, optional
            Maximum number of proposals generated at each leaf. Default is 3.
        min_size_with_proposals : float, optional
            Minimum cable path length required for fragments that proposals
            are generated from. Default is 0.
        search_scaling_factor : 1.5, optional
            Scaling actor used to enlarge search radius for each search.
            Default is 2.
        """
        # Instance attributes
        self.allow_nonleaf_proposals = allow_nonleaf_proposals
        self.cone_depth = cone_depth
        self.excluded = defaultdict(set)
        self.graph = graph
        self.kdtree = None
        self.leaf_tangents = dict()
        self.max_angle = max_angle
        self.max_attempts = max_attempts
        self.max_proposals_per_leaf = max_proposals_per_leaf
        self.min_size_with_proposals = min_size_with_proposals
        self.search_scaling_factor = search_scaling_factor
        if max_angle is not None:
            self.cos_max_angle = np.cos(np.deg2rad(max_angle))

    def __call__(self, initial_radius, proposals=None):
        """
        Generates edge proposals between fragments within the given search
        radius.

        Parameters
        ----------
        initial_radius : float
            Initial search radius used to generate proposals between endpoints
            of proposal.
        proposals : Iterable[Frozenset[int]], optional
            Existing proposals to keep. Each component pair retains a single
            (shortest) proposal, so these are seeded before searching.
            Default is None.
        """
        # Initializations
        self.set_kdtree()
        self.set_excluded()
        iterator = self.graph.leaf_nodes()
        if self.graph.verbose:
            iterator = tqdm(iterator, desc="Proposal Generation")

        # Cable length of every fragment in one vectorized pass, instead of
        # a DFS from each leaf.
        node_cable_length = self.graph.node_cable_lengths()

        # Main
        connections = dict()
        proposals = set(proposals or ())
        for proposal in proposals:
            connections[self.component_pair(*proposal)] = proposal
        for leaf in iterator:
            # Check if fragment satisfies size requirement
            if node_cable_length[leaf] < self.min_size_with_proposals:
                continue
            n_open = self.max_proposals_per_leaf - len(
                self.graph.node_proposals[leaf]
            )
            if n_open <= 0:
                continue
            if self.max_angle is not None and not self.is_straight(leaf):
                continue

            # Generate proposals
            cnt = 0
            node_candidates = list()
            while len(node_candidates) == 0 and cnt < self.max_attempts:
                # Search for candidates
                radius = initial_radius * self.search_scaling_factor**cnt
                cnt += 1
                node_candidates = self.find_node_candidates(
                    leaf, radius, n_open
                )

                # Parse candidates
                for i in node_candidates:
                    # Candidate info
                    pair_id = frozenset({int(leaf), int(i)})
                    pair_component_id = self.component_pair(leaf, i)

                    # Determine whether to keep
                    if pair_component_id in connections:
                        cur_proposal = connections[pair_component_id]
                        cur_length = self.graph.proposal_length(cur_proposal)
                        if self.graph.dist(leaf, i) < cur_length:
                            proposals.remove(cur_proposal)
                            del connections[pair_component_id]
                        else:
                            continue

                    # Add proposal
                    proposals.add(pair_id)
                    connections[pair_component_id] = pair_id
        return proposals

    def find_node_candidates(self, leaf, radius, max_candidates=None):
        """
        Finds valid node proposal candidates for a given leaf within a search
        radius.

        Parameters
        ----------
        leaf : int
            Leaf node to generate proposals from.
        radius : float
            Search radius used to generate proposals.
        max_candidates : int, optional
            Maximum number of candidates to return. Default is
            "max_proposals_per_leaf".

        Returns
        -------
        node_candidates : List[int]
            Node IDs that are valid proposal candidates for the given leaf.
        """
        node_candidates = list()
        for node in self.get_nearby_nodes(leaf, radius, max_candidates):
            node = self.adjust_position(leaf, node, radius)
            if self.is_valid_proposal(leaf, node) and self.is_aligned(leaf, node):
                node_candidates.append(node)
        return node_candidates

    def get_nearby_nodes(self, leaf, radius, max_components=None):
        """
        Gets nearby spatial points for a leaf node within a given radius.

        Parameters
        ----------
        leaf : int
            Leaf node to generate proposals from.
        radius : float
            Search radius used to generate proposals.
        max_components : int, optional
            Maximum number of components to return a node from. Default is
            "max_proposals_per_leaf".

        Returns
        -------
        List[int]
            Node IDs that are valid proposal candidates for the given leaf.
        """
        # Search for nearby nodes
        pts_dict = dict()
        for node in self.query_nbhd(leaf, radius):
            component_id = self.graph.node_component_id[node]
            if component_id == self.graph.node_component_id[leaf]:
                continue
            if component_id in self.excluded[leaf]:
                continue
            if self.in_cone(leaf, node):
                dist = self.graph.dist(leaf, node)
                if dist > radius:
                    continue
                if component_id not in pts_dict:
                    pts_dict[component_id] = {"dist": dist, "node": node}
                elif dist < pts_dict[component_id]["dist"]:
                    pts_dict[component_id] = {"dist": dist, "node": node}

        # Choose best proposals wrt proposal limit
        pts_dict = self.select_closest_components(pts_dict, max_components)
        return [val["node"] for val in pts_dict.values()]

    # --- Tangent Cone ---
    def outward_tangent(self, leaf):
        """
        Unit vector pointing from a leaf away from its fragment, cached per
        leaf for the duration of one call.
        """
        if leaf not in self.leaf_tangents:
            tangent = self.graph.tangent_from_leaf(
                leaf, max_depth=self.cone_depth
            )
            self.leaf_tangents[leaf] = -tangent
        return self.leaf_tangents[leaf]

    def is_straight(self, leaf):
        """
        Checks that the branch ending at a leaf has near-zero curvature, i.e.
        its tangent over a short depth agrees with its tangent over
        "cone_depth" to within "max_angle".
        """
        short = self.graph.tangent_from_leaf(
            leaf, max_depth=self.cone_depth / 4
        )
        return np.dot(short, -self.outward_tangent(leaf)) >= self.cos_max_angle

    def in_cone(self, leaf, node):
        """
        Checks whether a node lies within "max_angle" of the outward tangent
        at a leaf. Always True when no angle is set.
        """
        if self.max_angle is None:
            return True
        direction = self.graph.node_xyz[node] - self.graph.node_xyz[leaf]
        norm = np.linalg.norm(direction)
        if norm == 0:
            return False
        cos = np.dot(self.outward_tangent(leaf), direction / norm)
        return cos >= self.cos_max_angle

    def is_aligned(self, leaf, node):
        """
        Checks the tangent cone constraint on a final candidate. When the
        candidate is itself a leaf, it must be straight and point back at
        "leaf" so that both fragments are collinear across the gap.
        """
        if self.max_angle is None:
            return True
        if not self.in_cone(leaf, node):
            return False
        if self.graph.degree(node) == 1:
            return self.is_straight(node) and self.in_cone(node, leaf)
        return True

    # --- Helpers ---
    def component_pair(self, i, j):
        return frozenset(
            (self.graph.node_component_id[i], self.graph.node_component_id[j])
        )

    def set_excluded(self):
        """
        Maps each node of a previously rejected proposal to the current
        component of its partner, so the pair is not proposed again.
        """
        self.excluded = defaultdict(set)
        for i, j in getattr(self.graph, "rejected", ()):
            self.excluded[i].add(self.graph.node_component_id[j])
            self.excluded[j].add(self.graph.node_component_id[i])

    def adjust_position(self, leaf, node, radius):
        """
        Adjusts a candidate node by searching for a nearby leaf, or falling
        back to angular alignment if none is found.

        Parameters
        ----------
        leaf : int
            Node ID of the leaf node.
        node : int
            Node ID of the candidate node.
        radius : float
            Maximum path distance (in microns) for searching neighboring
            nodes.

        Returns
        -------
        node : int
            Node ID of candidate.
        """
        # Check if node is close to another leaf
        queue = [(node, 0)]
        visited = {node}
        while queue:
            # Visit node
            i, dist_i = queue.pop()
            if self.graph.degree(i) == 1:
                return i

            # Update queue
            for j in self.graph.neighbors(i):
                dist_j = dist_i + self.graph.dist(i, j)
                if dist_j < radius and j not in visited:
                    queue.append((j, dist_j))
                    visited.add(j)
        return self.maximize_branch_angle(leaf, node)

    def maximize_branch_angle(self, leaf, node):
        """
        Selects a nearby node that maximizes the angular alignment with a
        leaf's tangent.

        Parameters
        ----------
        leaf : int
            Node ID of the leaf node.
        node : int
            Node ID of the candidate node. The search for better candidates is
            restricted to nodes within a fixed distance of this node.

        Returns
        -------
        node : int
            Node ID of the node that maximizes angular alignment with the leaf
            tangent.
        """
        max_inner_product = 0
        leaf_tangent = self.graph.tangent_from_leaf(leaf, max_depth=20)
        for i in self.graph.nodes_within_distance(node, 30):
            pts = [self.graph.node_xyz[i], self.graph.node_xyz[leaf]]
            proposal_tangent = geometry_util.tangent(pts)
            inner_product = abs(np.sum(leaf_tangent * proposal_tangent))
            if inner_product > max_inner_product:
                max_inner_product = inner_product
                node = i
        return node

    def is_valid_proposal(self, leaf, i):
        """
        Determines whether a pair of nodes satisfies the following:
            (1) "i" is not None
            (2) "leaf" and "i" do not have swc_ids contained in
                "self.graph.soma_ids"

        Parmeters
        ---------
        leaf : int
            Leaf node ID.
        i : int
            Node ID.

        Returns
        -------
        bool
            Indication of whether proposal is valid.
        """
        if i is not None:
            is_soma = self.graph.is_soma(i) and self.graph.is_soma(leaf)
            self.graph.n_proposals_blocked += 1 if is_soma else 0
            return not is_soma and self.graph.degree(i) < 3
        else:
            return False

    def query_nbhd(self, node, radius):
        """
        Queries nodes within a spatial neighborhood of a given node.

        Parameters
        ----------
        node : int
            Node ID.
        radius : float
            Search radius (in microns) of the neighborhood.

        Returns
        -------
        nodes : List[int]
            Node IDs within the specified radius.
        """
        xyz = self.graph.node_xyz[node]
        if self.max_angle is not None:
            # Smallest ball enclosing the truncated cone, centered halfway
            # along its axis; far smaller than the full ball for small angles
            xyz = xyz + 0.5 * radius * self.outward_tangent(node)
            radius = radius * np.sqrt(1.25 - self.cos_max_angle)
        if self.allow_nonleaf_proposals:
            return self.kdtree.query_ball_point(xyz, radius)
        else:
            nodes = list()
            for idx in self.kdtree.query_ball_point(xyz, radius):
                node = self.graph.closest_node(self.kdtree.data[idx])
                nodes.append(node)
            return nodes

    def select_closest_components(self, pts_dict, max_components=None):
        """
        Retains only the closest components up to "max_components".

        Parameters
        ----------
        pts_dict : Dict[int, dict]
            Dictionary that maps component IDs to a dictionary containing a
            node ID and its distance from the leaf from which proposals are
            generated.
        max_components : int, optional
            Number of components to keep. Default is "max_proposals_per_leaf".

        Returns
        -------
        pts_dict : Dict[int, dict]
            Filtered dictionary containing up to "max_components"
            components, corresponding to the smallest distances.
        """
        if max_components is None:
            max_components = self.max_proposals_per_leaf
        while len(pts_dict) > max_components:
            farthest_key = None
            for key, val in pts_dict.items():
                if farthest_key is None:
                    farthest_key = key
                elif val["dist"] > pts_dict[farthest_key]["dist"]:
                    farthest_key = key
            del pts_dict[farthest_key]
        return pts_dict

    def set_kdtree(self):
        """
        Sets the KD-Tree used to search for nearby nodes to generate proposals
        between.
        """
        if self.allow_nonleaf_proposals:
            self.kdtree = self.graph.kdtree
        else:
            leafs = np.array(self.graph.leaf_nodes())
            self.kdtree = KDTree(self.graph.node_xyz[leafs])


# --- Trim Endpoints ---
def trim_proposal_endpoints(graph, proposal, max_depth=20):
    """
    Trims branch endpoints corresponding to the given proposal.

    Parameters
    ----------
    graph : ProposalGraph
        Graph containing the given proposal.
    proposal : Frozenset[int]
        Proposal used to specify endpoints to be trimmed.
    """

    def get_path(leaf):
        """
        Gets path of nodes emanating from the given node and ensures that
        there are no nodes with a proposal in this path.

        Parameters
        ----------
        leaf : int
            Node ID.

        Returns
        -------
        path : List[int]

        """
        idx = 1
        path = graph.path_from_leaf(leaf, max_depth=max_depth)
        while idx < len(path) and len(graph.node_proposals[path[idx]]) == 0:
            idx += 1
        return np.array(path[:idx], dtype=int)

    # Extract paths from nodes
    i, j = proposal
    path_i = get_path(i)
    path_j = get_path(j)

    # Find closest pair of points
    pts_i = graph.node_xyz[path_i]
    pts_j = graph.node_xyz[path_j]
    ii, jj = geometry_util.closest_pair(pts_i, pts_j)

    # Check if branches should be trimmed
    if graph.dist(path_i[ii], path_j[jj]) < graph.dist(i, j):
        if compute_dot(pts_i, pts_j, ii, jj) < -0.35:
            # Update proposals
            graph.remove_proposal(proposal)
            graph.add_proposal(path_i[ii], path_j[jj])

            # Remove nodes
            graph.remove_nodes_from(path_i[0:ii])
            graph.remove_nodes_from(path_j[0:jj])


def compute_dot(branch1, branch2, idx1, idx2):
    """
    Computes dot product between principal components of "branch1" and
    "branch_2".

    Parameters
    ----------
    branch1 : numpy.ndarray
        xyz coordinates of some branch from a graph.
    branch_2 : numpy.ndarray
        xyz coordinates of some branch from a graph.
    idx1 : int
        Index that "branch1" would be trimmed to (i.e. xyz coordinates from 0
        to "idx1" would be deleted from "branch1").
    idx2 : int
        Index that "branch_2" would be trimmed to (i.e. xyz coordinates from 0
        to "idx2" would be deleted from "branch_2").

    Returns
    -------
    float
        Dot product between principal components of "branch1" and "branch_2".
    """
    # Initializations
    midpoint = geometry_util.midpoint(branch1[idx1], branch2[idx2])
    b1 = branch1 - midpoint
    b2 = branch2 - midpoint

    # Main
    dot10 = np.dot(tangent(b1, idx1, 10), tangent(b2, idx2, 10))
    dot20 = np.dot(tangent(b1, idx1, 20), tangent(b2, idx2, 20))
    dot40 = np.dot(tangent(b1, idx1, 40), tangent(b2, idx2, 40))
    return np.min([dot10, dot20, dot40])


def tangent(branch, idx, depth):
    """
    Computes tangent vector of "branch" after indexing from "idx".

    Parameters
    ----------
    branch : numpy.ndarray
        xyz coordinates that form a path.
    idx : int
        Index of a row in "branch".

    Returns
    -------
    numpy.ndarray
        Tangent vector of "branch".
    """
    end = min(idx + depth, len(branch))
    return geometry_util.tangent(branch[idx:end])
