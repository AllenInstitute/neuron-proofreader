"""
Created on Sat July 15 9:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Generates features for training a machine learning model and performing
inference.

Note: We assume that a segmentation mask corresponds to multiscale 0. Thus,
      the instance attribute "self.multiscale" corresponds to the multiscale
      of the input image.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from torch_geometric.data import HeteroData
from torch_geometric.utils import to_undirected

import numpy as np
import torch

from neuron_proofreader.utils import geometry_util, graph_util, img_util, util


# --- Feature Extractors ---
class FeaturePipeline:
    """
    A class that generates features from a given graph.
    """

    def __init__(
        self,
        img_path,
        search_radius,
        padding=40,
        patch_shape=(96, 96, 96),
        segmentation_path=None,
    ):
        """
        Instantiates a FeaturePipeline object.

        Parameters
        ----------
        img_path : str
            Path to image of whole-brain dataset.
        search_radius : float
            Search radius used to generate proposals.
        padding : int, optional
            Number of voxels to be added in each dimension from start and end
            point of proposal for image patch extraction. Default is 40.
        patch_shape : Tuple[int], optional
            Shape of image patch expected by the vision model. Default is (96,
            96, 96).
        segmentation_path : str, optional
            Path to segmentation of whole-brain dataset.
        """
        self.extractors = [
            SkeletonFeatureExtractor(search_radius),
            ImageFeatureExtractor(
                img_path,
                patch_shape=patch_shape,
                padding=padding,
                segmentation_path=segmentation_path,
            ),
        ]

    def __call__(self, graph):
        """
        Runs the feature extraction pipeline.

        Parameters
        ----------
        graph : ProposalGraph
            Graph to extract features from.
        """
        features = FeatureSet(graph)
        for extractor in self.extractors:
            extractor(graph, features)
        return features


class SkeletonFeatureExtractor:
    """
    A class for extracting skeleton-based features.
    """

    def __init__(self, search_radius):
        """
        Instantiates a SkeletonFeatureExtractor object.

        Parameters
        ----------
        search_radius : float
            Search radius used to generate edge proposals.
        """
        # Instance attributes
        self.search_radius = search_radius

    def __call__(self, graph, features):
        """
        Extracts skeleton-based features for nodes, edges, and proposals.

        Parameters
        ----------
        graph : ProposalGraph
            Graph to extract features for.
        features : FeatureSet
            Data structure that stores features.
        """
        self.extract_node_features(graph, features)
        self.extract_edge_features(graph, features)
        self.extract_proposal_features(graph, features)

    def extract_node_features(self, graph, features):
        """
        Extracts skeleton-based features for nodes.

        Parameters
        ----------
        graph : ProposalGraph
            Graph to generate features for.
        """
        node_features = dict()
        for u in graph.nodes:
            node_features[u] = np.array(
                [
                    graph.degree[u],
                    graph.node_radius[u],
                    len(graph.node_proposals[u]),
                ]
            )
        features.set_features(node_features, "node")

    def extract_edge_features(self, graph, features):
        """
        Extracts skeleton-based features for edges.

        Parameters
        ----------
        graph : ProposalGraph
            Graph to generate features for.

        Returns
        -------
        features : Dict[Frozenset[int], numpy.ndarray]
            Dictionary that maps an edge to its feature vector.
        """
        edge_features = dict()
        for edge in graph.edges:
            edge_features[frozenset(edge)] = np.array(
                [
                    np.mean(graph.edges[edge]["radius"]),
                    min(graph.edge_length(edge), 5000) / 5000,
                ],
            )
        features.set_features(edge_features, "edge")

    def extract_proposal_features(self, graph, features):
        """
        Extracts skeleton-based features for proposals.

        Parameters
        ----------
        graph : ProposalGraph
            Graph to generate features for.

        Returns
        -------
        features : Dict[Frozenset[int], numpy.ndarray]
            Dictionary that maps a proposal to its feature vector.
        """
        # Build KD-Tree from leaf node coordinates
        graph.set_kdtree(node_type="leaf")

        # Extract features
        proposal_features = dict()
        for p in graph.proposals:
            proposal_features[p] = np.concatenate(
                (
                    graph.proposal_length(p) / self.search_radius,
                    graph.n_nearby_leafs(p, self.search_radius),
                    graph.proposal_attr(p, "radius"),
                    graph.proposal_directionals(p, 16),
                    graph.proposal_directionals(p, 32),
                    graph.proposal_directionals(p, 64),
                    graph.proposal_directionals(p, 128),
                ),
                axis=None,
            )
        features.set_features(proposal_features, "proposal")


class ImageFeatureExtractor:
    """
    A class for extracting image patches, image profiles along proposals, and
    generating masks that indicate the spatial locations of proposals.
    """

    def __init__(
        self,
        img_path,
        brightness_clip=400,
        patch_shape=(96, 96, 96),
        padding=40,
        segmentation_path=None,
    ):
        """
        Instantiates an ImageExtractor object.

        Parameters
        ----------
        img_path : str
            Path to image of whole-brain dataset.
        brightness_clip : int, optional
            Intensity value that voxel brightnesses are clipped to.
        patch_shape : Tuple[int], optional
            Shape of image patch expected by the vision model. Default is (96,
            96, 96).
        padding : int, optional
            Number of voxels to be added in each dimension from start and end
            point of proposal for image patch extraction. Default is 40.
        segmentation_path : str, optional
            Path to segmentation of whole-brain dataset.
        """
        # Instance attributes
        self.brightness_clip = brightness_clip
        self.patch_shape = patch_shape
        self.padding = padding

        # Image reader
        self.img = img_util.TensorStoreReader(img_path)
        if segmentation_path:
            self.segmentation = img_util.TensorStoreReader(segmentation_path)
        else:
            self.segmentation = None

    def __call__(self, graph, features):
        """
        Extracts image patches and profiles for each proposal in the graph.

        Parameters
        ----------
        graph : ProposalGraph
            Graph to extract features for.
        features : FeatureSet
            Data structure that stores features.
        """
        with ThreadPoolExecutor() as executor:
            # Assign threads
            pending = dict()
            for p in graph.proposals:
                thread = executor.submit(self.get_patches, graph, p)
                pending[thread] = p

            # Store results
            proposal_patches = dict()
            proposal_profiles = dict()
            for thread in as_completed(pending.keys()):
                p = pending.pop(thread)
                patches, profile = thread.result()
                proposal_patches[p] = patches
                proposal_profiles[p] = profile

        # Update features
        features.set_features(proposal_patches, "proposal_patches")
        features.integrate_proposal_profiles(proposal_profiles)

    def get_patches(self, graph, proposal):
        # Read images
        center, shape = self.compute_proposal_crop(graph, proposal)
        img_patch = self.read_image_patch(center, shape)
        offset = img_util.get_offset(center, shape)

        # Generate image profile
        voxels = self.get_profile_line(graph, proposal, offset, 16)
        profile = np.array([img_patch[tuple(voxel)] for voxel in voxels])
        profile = np.append(profile, [profile.mean(), profile.std()])

        # Set patches
        proposal_mask = self.get_proposal_mask(graph, proposal, center, shape)
        img_path = img_util.resize(img_patch, self.patch_shape)
        patches = np.stack([img_path, proposal_mask], axis=0)
        return patches, profile

    def get_proposal_mask(self, graph, proposal, center, shape):
        # Read segmentation
        segmentation_mask = self.read_segmentation_mask(center, shape)

        # Annotate label patch
        u, v = tuple(proposal)
        offset = img_util.get_offset(center, shape)
        self.annotate_edge(graph, u, segmentation_mask, offset)
        self.annotate_edge(graph, v, segmentation_mask, offset)
        self.annotate_proposal(graph, proposal, segmentation_mask, offset)
        return img_util.resize(segmentation_mask, self.patch_shape, True)

    def annotate_edge(self, graph, node, patch, offset):
        voxels = self.get_local_coordinates(graph, node, offset)
        voxels = geometry_util.make_voxels_connected(voxels)
        img_util.annotate_voxels(patch, voxels, val=0.5)

    def annotate_proposal(self, graph, proposal, patch, offset):
        profile_line = self.get_profile_line(graph, proposal, offset)
        img_util.annotate_voxels(patch, profile_line, val=1)

    # --- Helpers ---
    def compute_proposal_crop(self, graph, proposal):
        # Compute bounds
        u1, u2 = proposal
        voxel1 = graph.get_voxel(u1)
        voxel2 = graph.get_voxel(u2)
        bounds = img_util.get_minimal_bbox([voxel1, voxel2], self.padding)

        # Transform into square
        center = [int((v1 + v2) / 2) for v1, v2 in zip(voxel1, voxel2)]
        length = np.max([u - l for u, l in zip(bounds["max"], bounds["min"])])
        return center, (length, length, length)

    def get_profile_line(self, graph, proposal, offset, n_pts=None):
        node1, node2 = proposal
        voxel1 = graph.get_local_voxel(node1, offset)
        voxel2 = graph.get_local_voxel(node2, offset)
        if n_pts:
            return geometry_util.make_line(voxel1, voxel2, n_pts)
        else:
            return geometry_util.make_digital_line(voxel1, voxel2)

    def get_local_coordinates(self, graph, node, offset):
        pts = np.vstack(graph.edge_attr(node, "xyz"))
        voxels = [img_util.to_voxels(xyz, graph.anisotropy) for xyz in pts]
        voxels = geometry_util.shift_path(voxels, offset)
        return voxels

    def read_image_patch(self, center, shape):
        patch = self.img.read(center, shape)
        patch = img_util.normalize(np.minimum(patch, self.brightness_clip))
        return patch

    def read_segmentation_mask(self, center, shape):
        if self.segmentation:
            patch = self.segmentation.read(center, shape)
            return 0.25 * (patch > 0).astype(float)
        else:
            return np.zeros(shape)


# --- Feature Data Structures ---
class FeatureSet:
    _FEATURE_TABLE = {
        "node": ("node_features", "node_index_mapping"),
        "edge": ("edge_features", "edge_index_mapping"),
        "proposal": ("proposal_features", "proposal_index_mapping"),
        "proposal_patches": ("proposal_patches", "proposal_index_mapping"),
    }

    def __init__(self, graph):
        # Instance Attributes
        self.graph = graph
        self.node_index_mapping = IndexMapping(graph.nodes)
        self.edge_index_mapping = IndexMapping(graph.edges)
        self.proposal_index_mapping = IndexMapping(graph.proposals)

        self.node_features = None
        self.edge_features = None
        self.proposal_features = None
        self.proposal_patches = None
        self.targets = self.get_targets()

    def set_features(self, feature_dict, feature_type):
        # Determine feature type
        if feature_type not in self._FEATURE_TABLE:
            raise ValueError(f"Unknown feature type: {feature_type}")
        feat_attr, index_mappping_attr = self._FEATURE_TABLE[feature_type]

        # Store features
        index_mapping = getattr(self, index_mappping_attr)
        feature_matrix = self.to_matrix(feature_dict, index_mapping)
        setattr(self, feat_attr, feature_matrix)

    def to_heterograph_data(self):
        return HeteroGraphData(self.graph, self.features)

    # --- Helpers ---
    def get_targets(self):
        y = np.zeros((self.graph.n_proposals(), 1))
        idx_to_id = self.proposal_index_mapping.idx_to_id
        for idx, object_id in idx_to_id.items():
            if object_id in self.graph.gt_accepts:
                y[idx] = 1
        return y

    @staticmethod
    def init_matrix(feature_dict):
        key = util.sample_once(feature_dict.keys())
        shape = (len(feature_dict.keys()),) + feature_dict[key].shape
        return np.zeros(shape)

    def integrate_proposal_profiles(self, profiles_dict):
        x = self.init_matrix(profiles_dict)
        for object_id in profiles_dict:
            idx = self.proposal_index_mapping.id_to_idx[object_id]
            x[idx] = profiles_dict[object_id]
        self.proposal_features = np.concatenate(
            (self.proposal_features, x), axis=1
        )

    def to_matrix(self, feature_dict, index_mapping):
        x = self.init_matrix(feature_dict)
        for object_id in feature_dict:
            idx = index_mapping.id_to_idx[object_id]
            x[idx] = feature_dict[object_id]
        return x


class HeteroGraphData(HeteroData):
    """
    A custom data class for heterogenous graphs. The graph is internally
    represented as a line graph to facilitate edge-based message passing in
    a GNN.
    """

    def __init__(self, graph, features):
        # Call parent class
        super().__init__()

        # Index mappings
        self.idxs_branches = features.edge_index_mapping
        self.idxs_proposals = features.proposal_index_mapping

        # Node features
        self["branch"].x = torch.tensor(features.edge_features)
        self["proposal"].x = torch.tensor(features.proposal_features)
        self["proposal"].y = torch.tensor(features.targets)
        self["patch"].x = torch.tensor(features.proposal_features)

        # Edge indices
        self.build_proposal_adjacency(graph)
        self.build_branch_adjacency(graph)
        self.build_proposal_branch_adjacency(graph)

    # --- Core Routines ---
    def build_proposal_adjacency(self, graph):
        """
        Builds proposal–proposal adjacency based on shared node incidence.

        Parameters
        ----------
        graph : ProposalGraph
            Graph containing proposals.
        """
        edges = graph.proposals
        edge_index = self._build_adjacency(edges, self.idxs_proposals)
        self.set_edge_index(edge_index, ("proposal", "to", "proposal"))

    def build_branch_adjacency(self, graph):
        """
        Builds branch–branch adjacency based on shared node incidence.

        Parameters
        ----------
        graph : ProposalGraph
            Graph containing branches.
        """
        edge_index = self._build_adjacency(graph.edges, self.idxs_branches)
        self.set_edge_index(edge_index, ("branch", "to", "branch"))

    def build_branch_proposal_adjacency(self, graph):
        """
        Builds branch–proposal adjacency based on shared node incidence.

        Parameters
        ----------
        graph : ProposalGraph
            Graph containing branches and proposals.
        """
        edge_index = []
        for proposal in graph.proposals:
            idx_proposal = self.idxs_proposals["id_to_idx"][proposal]
            for i in proposal:
                for j in graph.neighbors(i):
                    branch = frozenset((i, j))
                    idx_branch = self.idxs_branches["id_to_idx"][branch]
                    edge_index.append([idx_proposal, idx_branch])
        self.set_edge_index(edge_index, ("branch", "to", "proposal"))

    # --- Helpers ---
    @staticmethod
    def _build_adjacency(self, edges, index_mapping):
        # Build edge index
        edge_index = []
        line_graph = graph_util.edges_to_line_graph(edges)
        for e1, e2 in line_graph.edges:
            v1 = index_mapping["id_to_idx"][frozenset(e1)]
            v2 = index_mapping["id_to_idx"][frozenset(e2)]
            edge_index.append([v1, v2])
        return edge_index

    def set_edge_index(self, edge_index, edge_type):
        # Check if edge index is empty
        if len(edge_index) == 0:
            self[edge_type].edge_index = torch.empty((2, 0), dtype=torch.long)

        # Reformat edge index
        edge_index = to_undirected(edge_index)
        edge_index = torch.Tensor(edge_index).t().contiguous().long()
        self[edge_type].edge_index = edge_index
        self[edge_type[::-1]].edge_index = edge_index


class IndexMapping:
    """
    A class that stores data structures for mapping between object IDs and
    indices.

    Attributes
    ----------
    id_to_idx : Dict[hashable, int]
        Dictionary that maps object IDs to indices.
    idx_to_id : Dict[hashable, int]
        Dictionary that maps indices to object IDs.
    """

    def __init__(self, object_ids):
        """
        Instantiates an IndexMapper object.

        Parameters
        ----------
        object_ids : List[hashable]
            Object IDs to create index mapping from.
        """
        self.id_to_idx = dict()
        self.idx_to_id = dict()
        for idx, object_id in enumerate(object_ids):
            # Check object ID datatype
            if isinstance(object_id, tuple):
                object_id = frozenset(object_id)

            # Populate dictionary
            self.id_to_idx[object_id] = idx
            self.idx_to_id[idx] = object_id


# --- Helpers ---
def get_node_dict():
    """
    Returns the number of features for different node types.

    Returns
    -------
    dict
        Dictionary containing the number of features for each node type
    """
    return {"branch": 2, "proposal": 34}


def get_edge_dict():
    """
    Returns the number of features for different edge types.

    Returns
    -------
    dict
        A dictionary containing the number of features for each edge type
    """
    edge_dict = {
        ("proposal", "edge", "proposal"): 3,
        ("branch", "edge", "branch"): 3,
        ("branch", "edge", "proposal"): 3,
    }
    return edge_dict
