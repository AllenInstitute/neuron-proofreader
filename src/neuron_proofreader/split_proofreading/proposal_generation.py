"""
Created on Sat July 15 9:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code that generates edge proposals for a given graph.

"""

from tqdm import tqdm

import numpy as np

from neuron_proofreader.utils import geometry_util

DOT_THRESHOLD = -0.35


class ProposalGenerator:
    """
    A class for generating proposals between fragments in a graph.
    """

    def __init__(
        self,
        graph,
        allow_nonleaf_targets=False,
        max_attempts=2,
        max_proposals_per_leaf=3,
        min_size_with_proposals=0,
        search_scaling_factor=1.5
    ):
        """
        Instantiates a ProposalGenerator object.

        Parameters
        ----------
        graph : ProposalGraph
            Graph that proposals will be generated for.
        allow_nonleaf_targets : bool, optional
            Indication of whether to generate proposals between leaf and nodes
            with degree 2. Default is False.
        max_attempts : int, optional
            Number of attempts made to generate proposals from a node with
            increasing search radii. Default is 2.
        max_proposals_per_leaf : bool, optional
            Maximum number of proposals generated at each leaf. Default is 3.
        min_size_with_proposals : float, optional
            Minimum fragment path length required for proposals. Default is 0.
        search_scaling_factor : 1.5, optional
            Scaling actor used to enlarge search radius for each search.
            Default is 2.
        """
        # Instance attributes
        self.allow_nonleaf_targets = allow_nonleaf_targets
        self.graph = graph
        self.kdtree = None
        self.max_attempts = max_attempts
        self.max_proposals_per_leaf = max_proposals_per_leaf
        self.min_size_with_proposals = min_size_with_proposals
        self.search_scaling_factor = search_scaling_factor

    def __call__(self, initial_radius):
        """
        Generates edge proposals between fragments within the given search
        radius.

        Parameters
        ----------
        initial_radius : float
            Initial search radius used to generate proposals between endpoints
            of proposal.
        """
        # Initializations
        self.set_kdtree()
        iterator = self.graph.get_leafs()
        if self.graph.verbose:
            iterator = tqdm(iterator, desc="Proposal Generation")

        # Main
        connections = dict()
        proposals = set()
        for leaf in iterator:
            # Check if fragment satisfies size requirement
            length = self.graph.path_length(
                max_depth=self.min_size_with_proposals, root=leaf
            )
            if length < self.min_size_with_proposals:
                continue

            # Generate proposals
            cnt = 0
            node_candidates = list()
            while len(node_candidates) == 0 and cnt < self.max_attempts:
                # Search for candidates
                cnt += 1
                radius = initial_radius * self.search_scaling_factor ** cnt
                node_candidates = self.find_node_candidates(leaf, radius)

                # Parse candidates
                for i in node_candidates:
                    # Candidate info
                    pair_id = frozenset({leaf, i})
                    leaf_component_id = self.graph.node_component_id[leaf]
                    node_component_id = self.graph.node_component_id[i]
                    pair_component_id = frozenset(
                        (leaf_component_id, node_component_id)
                    )

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

    def find_node_candidates(self, leaf, radius):
        """
        Finds valid node proposal candidates for a given leaf within a search
        radius.

        Parameters
        ----------
        leaf : int
            Leaf node to generate proposals from.
        radius : float
            Search radius used to generate proposals.

        Returns
        -------
        node_candidates : List[int]
            Node IDs that are valid proposal candidates for the given leaf.
        """
        node_candidates = list()
        for xyz in self.get_nearby_points(leaf, radius):
            i = self.get_connecting_node(leaf, radius, xyz)
            if self.is_valid_proposal(leaf, i):
                node_candidates.append(i)
        return node_candidates

    def get_nearby_points(self, leaf, radius):
        """
        Get nearby spatial points for a leaf node within a given radius.

        Parameters
        ----------
        leaf : int
            Leaf node to generate proposals from.
        radius : float
            Search radius used to generate proposals.

        Returns
        -------
        List[int]
            Node IDs that are valid proposal candidates for the given leaf.
        """
        pts_dict = self.query_closest_points_per_component(leaf, radius)
        pts_dict = self.select_closest_components(pts_dict)
        return [val["xyz"] for val in pts_dict.values()]

    # --- Helpers ---
    def get_closer_endpoint(self, edge, xyz):
        """
        Gets node from "edge" that is closer to "xyz".

        Parameters
        ----------
        edge : Tuple[int]
            Edge to be checked.
        xyz : numpy.ndarray
            xyz coordinate.

        Returns
        -------
        int
            Node closer to "xyz".
        """
        i, j = tuple(edge)
        d_i = geometry_util.dist(self.graph.node_xyz[i], xyz)
        d_j = geometry_util.dist(self.graph.node_xyz[j], xyz)
        return i if d_i < d_j else j

    def get_connecting_node(self, leaf, radius, xyz):
        """
        Gets node that proposal emanating from "leaf" will connect to.

        Parameters
        ----------
        leaf : int
            Leaf node to generate proposals from.
        radius : float
            Search radius used to generate proposals.
        xyz : numpy.ndarray
            Coordinate of potential proposal

        Returns
        -------
        int
            Node id that proposal will connect to.
        """
        # Check if edge exists
        try:
            edge = self.graph.xyz_to_edge[xyz]
        except KeyError:
            return None

        # Find connecting node
        node = self.get_closer_endpoint(edge, xyz)
        if self.graph.dist(leaf, node) < radius:
            return node
        elif self.allow_nonleaf_targets:
            attrs = self.graph.get_edge_data(*edge)
            idx = np.where(np.all(attrs["xyz"] == xyz, axis=1))[0][0]
            if type(idx) is int:
                return self.graph.split_edge(edge, attrs, idx)
        return None

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
            is_soma = (self.graph.is_soma(i) and self.graph.is_soma(leaf))
            self.graph.n_proposals_blocked += 1 if is_soma else 0
            return not is_soma
        else:
            return False

    def query_closest_points_per_component(self, leaf, radius):
        """
        Finds the closest points on other connected components within the
        search radius

        Parameters
        ----------
        leaf : int
            Leaf node to generate proposals from.
        radius : float
            Search radius used to generate proposals.
        """
        pts_dict = dict()
        leaf_xyz = self.graph.node_xyz[leaf]
        for xyz in geometry_util.query_ball(self.kdtree, leaf_xyz, radius):
            component_id = self.graph.xyz_to_component_id(xyz)
            if component_id != self.graph.node_component_id[leaf]:
                dist = geometry_util.dist(leaf_xyz, xyz)
                if component_id not in pts_dict:
                    pts_dict[component_id] = {"dist": dist, "xyz": tuple(xyz)}
                elif dist < pts_dict[component_id]["dist"]:
                    pts_dict[component_id] = {"dist": dist, "xyz": tuple(xyz)}
        return pts_dict

    def select_closest_components(self, pts_dict):
        """
        Retains only the closest components up to "max_proposals_per_leaf".

        Parameters
        ----------
        pts_dict : Dict[int, dict]
            Dictionary that maps component IDs to a dictionary containing a
            node ID and its distance from the leaf from which proposals are
            generated.

        Returns
        -------
        pts_dict : Dict[int, dict]
            Filtered dictionary containing up to "self.max_proposals_per_leaf"
            components, corresponding to the smallest distances.
        """
        while len(pts_dict) > self.max_proposals_per_leaf:
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
        if self.allow_nonleaf_targets:
            self.kdtree = self.graph.get_kdtree()
        else:
            self.kdtree = self.graph.get_kdtree(node_type="leaf")


# --- Trim Endpoints ---
def trim_endpoints_at_proposal(graph, proposal):
    """
    Trims branch endpoints corresponding to the given proposal.

    Parameters
    ----------
    graph : ProposalGraph
        Graph containing the given proposal.
    proposal : Frozenset[int]
        Proposal used to specify endpoints to be trimmed.
    """
    # Find closest points between proposal branches
    i, j = tuple(proposal)
    pts_i = graph.edge_attr(i, key="xyz", ignore=True)[0]
    pts_j = graph.edge_attr(j, key="xyz", ignore=True)[0]
    dist_ij, (idx_i, idx_j) = find_closest_pair(pts_i, pts_j)

    # Update branches (if applicable)
    if dist_ij < geometry_util.dist(pts_i[0], pts_j[0]):
        if compute_dot(pts_i, pts_j, idx_i, idx_j) < DOT_THRESHOLD:
            trim_to_idx(graph, i, idx_i)
            trim_to_idx(graph, j, idx_j)


def find_closest_pair(pts1, pts2):
    """
    Finds the closest pair of points from the given lists of points.

    Parameters
    ----------
    pts1 : List[Tuple[int]]
        First list of points.
    pts2 : List[Tuple[int]]
        Second list of points.

    Returns
    -------
    best_dist : float
        Distance between closest pair of points
    best_idxs : Tuple[int]
        Indices of closest pair of points from the given lists.
    """
    best_dist, best_idxs = np.inf, (0, 0)
    i, length1 = -1, 0
    while length1 < 20 and i < len(pts1) - 1:
        i += 1
        length1 += geometry_util.dist(pts1[i], pts1[i - 1]) if i > 0 else 0

        # Search other branch
        j, length2 = -1, 0
        while length2 < 20 and j < len(pts2) - 1:
            j += 1
            length2 += geometry_util.dist(pts2[j], pts2[j - 1]) if j > 0 else 0

            # Check distance between points
            dist = geometry_util.dist(pts1[i], pts2[j])
            if dist < best_dist:
                best_dist = dist
                best_idxs = (i, j)
    return best_dist, best_idxs


def trim_to_idx(graph, i, idx):
    """
    Trims the end of a branch specified by the leaf node "i".

    Parameters
    ----------
    graph : FragmentsGraph
        Graph containing node "i"
    i : int
        Leaf node ID.
    idx : int
        Branch is trimmed to the index "idx".
    """
    # Update node
    edge_xyz = graph.edge_attr(i, key="xyz", ignore=True)[0]
    edge_radii = graph.edge_attr(i, key="radius", ignore=True)[0]
    graph.node_xyz[i] = edge_xyz[idx]
    graph.node_radius[i] = edge_radii[idx]

    # Update edge
    nb = list(graph.neighbors(i))[0]
    graph.edges[i, nb]["xyz"] = edge_xyz[idx:]
    graph.edges[i, nb]["radius"] = edge_radii[idx:]
    for k in range(idx):
        try:
            del graph.xyz_to_edge[tuple(edge_xyz[k])]
        except KeyError:
            pass


# --- Helpers ---
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


def get_sorted_proposals(graph, proposals):
    """
    Sorts the given proposals by physical length.

    Parameters
    ----------
    graph : ProposalGraph
        Graph used to generate proposals.
    proposals : List[Frozenset[int]]
        Proposals to be sorted.

    Returns
    -------
    proposals : List[Frozenset[int]]
        Sorted proposals.
    """
    # Compute lengths
    lengths = list()
    proposals = list(proposals)
    for i, j in proposals:
        lengths.append(graph.dist(i, j))

    # Sort by distance
    lengths = [graph.proposal_length(p) for p in proposals]
    return [proposals[i] for i in np.argsort(lengths)]


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
