"""
Created on Tue Jan 13 15:00:00 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code for generating features used by machine learning models to perform split
correction.

"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from torch_geometric.data import HeteroData

import numpy as np
import torch

from neuron_proofreader.utils import geometry_util, graph_util, img_util, util
from neuron_proofreader.utils.ml_util import TensorDict


# --- Feature Extractors ---
class FeaturePipeline:
    """
    A class that generates features from a given graph.
    """

    def __init__(
        self,
        graph,
        img_path,
        brightness_clip=400,
        padding=50,
        patch_shape=(96, 96, 96),
        segmentation_path=None,
    ):
        """
        Instantiates a FeaturePipeline object.

        Parameters
        ----------
        graph : ProposalGraph
            Graph to extract features from.
        img_path : str
            Path to image of whole-brain dataset.
        brightness_clip : int, optional
            ...
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
            SkeletonFeatureExtractor(graph),
            ImageFeatureExtractor(
                graph,
                img_path,
                brightness_clip=brightness_clip,
                patch_shape=patch_shape,
                padding=padding,
                segmentation_path=segmentation_path,
            ),
        ]

    def __call__(self, subgraph):
        """
        Runs the feature extraction pipeline.

        Parameters
        ----------
        subgraph : ProposalGraph
            Subgraph of "graph" attribute to extract features for.
        """
        features = FeatureSet(subgraph)
        for extractor in self.extractors:
            extractor(subgraph, features)
        return features


class SkeletonFeatureExtractor:
    """
    A class for extracting skeleton-based features.
    """

    def __init__(self, graph):
        """
        Instantiates a SkeletonFeatureExtractor object.

        Parameters
        ----------
        graph : ProposalGraph
            Graph to extract features from.
        """
        # Instance attributes
        self.graph = graph

        # Build KD-tree from leaf nodes
        self.graph.set_kdtree(node_type="leaf")

    def __call__(self, subgraph, features):
        """
        Extracts skeleton-based features for nodes, edges, and proposals.

        Parameters
        ----------
        subgraph : ProposalGraph
            Subgraph of "graph" attribute to extract features for.
        features : FeatureSet
            Data structure that stores features.
        """
        self.extract_node_features(subgraph, features)
        self.extract_edge_features(subgraph, features)
        self.extract_proposal_features(subgraph, features)

    def extract_node_features(self, subgraph, features):
        """
        Extracts skeleton-based features for nodes.

        Parameters
        ----------
        subgraph : ProposalGraph
            Subgraph of "graph" attribute to extract features for.
        """
        node_features = dict()
        for u in subgraph.nodes:
            node_features[u] = np.array(
                [
                    self.graph.degree[u],
                    self.graph.node_radius[u],
                    len(self.graph.node_proposals[u]),
                ]
            )
        features.set_features(node_features, "node")

    def extract_edge_features(self, subgraph, features):
        """
        Extracts skeleton-based features for edges.

        Parameters
        ----------
        subgraph : ProposalGraph
            Subgraph of "graph" attribute to extract features for.

        Returns
        -------
        features : Dict[Frozenset[int], numpy.ndarray]
            Dictionary that maps an edge to its feature vector.
        """
        edge_features = dict()
        for edge in subgraph.edges:
            edge_features[frozenset(edge)] = np.array(
                [
                    np.mean(self.graph.edges[edge]["radius"]),
                    min(self.graph.edge_length(edge), 5000) / 5000,
                ],
            )
        features.set_features(edge_features, "edge")

    def extract_proposal_features(self, subgraph, features):
        """
        Extracts skeleton-based features for proposals.

        Parameters
        ----------
        subgraph : ProposalGraph
            Subgraph of "graph" attribute to extract features for.

        Returns
        -------
        features : Dict[Frozenset[int], numpy.ndarray]
            Dictionary that maps a proposal to its feature vector.
        """
        proposal_features = dict()
        for p in subgraph.proposals:
            proposal_features[p] = np.concatenate(
                (
                    self.graph.proposal_length(p) / self.graph.search_radius,
                    self.graph.n_nearby_leafs(p, self.graph.search_radius),
                    self.graph.proposal_attr(p, "radius"),
                    self.graph.proposal_directionals(p, 16),
                    self.graph.proposal_directionals(p, 32),
                    self.graph.proposal_directionals(p, 64),
                    self.graph.proposal_directionals(p, 128),
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
        graph,
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
        graph : ProposalGraph
            Graph to extract features from.
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
        self.graph = graph
        self.patch_shape = patch_shape
        self.padding = padding

        # Image reader
        self.img = img_util.TensorStoreReader(img_path)
        if segmentation_path:
            self.segmentation = img_util.TensorStoreReader(segmentation_path)
        else:
            self.segmentation = None

    def __call__(self, subgraph, features):
        """
        Extracts image patches and profiles for each proposal in the graph.

        Parameters
        ----------
        subgraph : ProposalGraph
            Subgraph of "graph" attribute to extract features for.
        features : FeatureSet
            Data structure that stores features.
        """
        with ThreadPoolExecutor() as executor:
            # Assign threads
            pending = dict()
            for proposal in subgraph.proposals:
                thread = executor.submit(
                    self.init_extractor, subgraph, proposal
                )
                pending[thread] = proposal

            # Store results
            patches, profiles = dict(), dict()
            for thread in as_completed(pending.keys()):
                proposal = pending.pop(thread)
                extractor = thread.result()
                profiles[proposal] = extractor.get_intensity_profile()
                patches[proposal] = extractor.get_input_patch()

        # Update features
        features.set_features(patches, "proposal_patches")
        features.integrate_proposal_profiles(profiles)

    def init_extractor(self, subgraph, proposal):
        """
        Initializes a PatchFeatureExtractor for a given subgraph and proposal.

        Parameters
        ----------
        subgraph : nx.Graph or similar
            Subgraph containing the given proposal.
        proposal : Any
            Proposal that image patches are centered about.

        Returns
        -------
        extractor : PatchFeatureExtractor
            Feature extractor configured with the cropped image, segmentation
            mask, spatial offset, and patch shape.
        """
        # Compute patch specs
        center, shape = self.compute_crop(proposal)
        offset = img_util.get_offset(center, shape)

        # Read images
        img = self.read_image(center, shape)
        mask = self.read_segmentation(center, shape)

        # Create patch feature extractor
        extractor = PatchFeatureExtractor(
            self.graph,
            img,
            mask,
            proposal,
            offset,
            self.patch_shape
        )
        return extractor

    # --- Helpers ---
    def read_image(self, center, shape):
        """
        Reads the image patch specified by the given center and shape.

        Parameters
        ----------
        center : Tuple[int]
            Center of image patch to be read.
        shape : Tuple[int]
            Center of image patch to be read.
        """
        patch = self.img.read(center, shape)
        patch = img_util.normalize(np.minimum(patch, self.brightness_clip))
        return patch

    def read_segmentation(self, center, shape):
        """
        Reads the segmentation patch specified by the given center and shape.

        Parameters
        ----------
        center : Tuple[int]
            Center of segmentation patch to be read.
        shape : Tuple[int]
            Center of segmentation patch to be read.
        """
        if self.segmentation:
            patch = self.segmentation.read(center, shape)
            return 0.25 * (patch > 0).astype(float)
        else:
            return np.zeros(shape)

    def compute_crop(self, proposal):
        """
        Extracts an intensity profile along a set of voxel coordinates.

        Returns
        -------
        profile : numpy.ndarray
            Image with shape (2, H, W, D) containing a raw image and proposal
            mask channels.
        """
        # Get info
        node1, node2 = tuple(proposal)
        voxel1 = self.graph.get_voxel(node1)
        voxel2 = self.graph.get_voxel(node2)

        # Compute bounds
        bounds = img_util.get_minimal_bbox([voxel1, voxel2], self.padding)
        center = tuple([int((v1 + v2) / 2) for v1, v2 in zip(voxel1, voxel2)])
        length = np.max([u - l for u, l in zip(bounds["max"], bounds["min"])])
        return center, (length, length, length)


class PatchFeatureExtractor:
    """
    A class that extracts features from an image patch that is centered at a
    proposal.
    """

    def __init__(
        self, graph, img, mask, proposal, offset, patch_shape=(96, 96, 96)
    ):
        """
        Instantiates a PatchFeatureExtractor object.

        Parameters
        ----------
        graph : ProposalGraph
            Graph to extract features from.
        img : numpy.ndarray
            Image patch centered at proposal coordinates.
        mask : numpy.ndarray
            Segmentation patch centered at proposal coordinates.
        proposal : Frozenset[int]
            Proposal that patch is centered about.
        offset : numpy.ndarray
            Offset used to map global coordinates into the local mask.
        """
        # Instance attributes
        self.graph = graph
        self.img = img
        self.mask = mask
        self.proposal = proposal
        self.offset = offset
        self.patch_shape = patch_shape

        # Annotate mask
        node1, node2 = tuple(self.proposal)
        self.annotate_edge(node1)
        self.annotate_edge(node2)
        self.annotate_proposal()

    # --- Core Routines ---
    def get_input_patch(self):
        """
        Gets input patch from the image and segmentation mask.

        Returns
        -------
        numpy.ndarray
            Array with shape (2, *patch_shape), where channel 0 contains
            raw image data and channel 1 contains segmentation data.
        """
        img = img_util.resize(self.img, self.patch_shape)
        mask = img_util.resize(self.mask, self.patch_shape, True)
        return np.stack([img, mask], axis=0)

    def get_intensity_profile(self):
        """
        Gets an intensity profile along the branches and proposal.

        Returns
        -------
        profile : numpy.ndarray
            Intensity profile along the branches and proposal.
        """
        # Branch profiles
        node1, node2 = tuple(self.proposal)
        branch1_profile = self.get_branch_profile(node1)
        branch2_profile = self.get_branch_profile(node2)

        # Proposal profile
        voxels = self.get_profile_line(16)
        proposal_profile = self._extract_profile(voxels)

        # Combine profiles
        profile = np.concatenate(
            (branch1_profile, proposal_profile, branch2_profile)
        )

        # Adjust intensities
        max_intensity = np.max(profile)
        self.img = np.minimum(max_intensity, self.img) / max_intensity
        profile /= max_intensity
        return profile

    def get_branch_profile(self, node):
        """
        Gets an intensity profile along the branch containing the given node.

        Parameters
        ----------
        node : int
            Identifier of the node whose incident branch coordinates are
            extracted.

        Returns
        -------
        profile : numpy.ndarray
            Intensity profile along the branch containing the given node.
        """
        def check_emptiness():
            """
            Checks if voxels is empty.
            """
            if len(voxels) < 2:
                voxels.append(self.graph.get_local_voxel(node, self.offset))

        # Get branch voxel coordinates
        voxels = self.get_branch_voxels(node)
        voxels = geometry_util.make_voxels_connected(voxels)
        voxels = img_util.get_contained_voxels(voxels, self.mask.shape)
        check_emptiness()

        # Resample voxels
        voxels = np.array(voxels)
        voxels = geometry_util.resample_curve_3d(voxels, 16).astype(int)
        voxels = img_util.get_contained_voxels(voxels, self.mask.shape)
        check_emptiness()
        return self._extract_profile(voxels)

    def _extract_profile(self, voxels):
        """
        Extracts an intensity profile along a set of voxel coordinates.

        Parameters
        ----------
        voxels : numpy.ndarray
            Voxel coordinates at which to sample the image.

        Returns
        -------
        profile : numpy.ndarray
            Image with shape (2, H, W, D) containing a raw image and proposal
            mask channels.
        """
        check_list_length(voxels, min_length=16)
        profile = np.array([self.img[tuple(voxel)] for voxel in voxels])
        profile = np.append(profile, [profile.mean(), profile.std()])
        return profile

    # --- Helpers ---
    def annotate_edge(self, node):
        """
        Annotates the neuron branch containing the specified node within the
        given mask.

        Parameters
        ----------
        node : int
            Node ID used to get branch to be annotated.
        """
        voxels = self.get_branch_voxels(node)
        voxels = geometry_util.make_voxels_connected(voxels)
        img_util.annotate_voxels(self.mask, voxels, val=0.5)

    def annotate_proposal(self):
        """
        Annotates the proposal within the given mask.
        """
        voxels = self.get_profile_line()
        img_util.annotate_voxels(self.mask, voxels, val=1)

    def get_profile_line(self, n_pts=None):
        """
        Generates a voxel line between the two nodes of a proposal.

        Parameters
        ----------
        n_pts : int, optional
            Number of points to sample along the line. If not provided, a
            dense voxel line is returned.

        Returns
        -------
        numpy.ndarray
            Voxel line between the two nodes of a proposal.
        """
        node1, node2 = self.proposal
        voxel1 = self.graph.get_local_voxel(node1, self.offset)
        voxel2 = self.graph.get_local_voxel(node2, self.offset)
        if n_pts:
            return geometry_util.make_line(voxel1, voxel2, n_pts)
        else:
            return geometry_util.make_digital_line(voxel1, voxel2)

    def get_branch_voxels(self, node):
        """
        Gets the voxel coordinates of the branch containing the given node.

        Parameters
        ----------
        node : int
            Identifier of the node whose incident branch coordinates are
            extracted.

        Returns
        -------
        voxels : List[Tuple[int]]
            Voxel coordinates representing the edge path in local patch
            coordinates.
        """
        pts = np.vstack(self.graph.edge_attr(node, "xyz"))
        anisotropy = self.graph.anisotropy
        voxels = [img_util.to_voxels(xyz, anisotropy) for xyz in pts]
        return geometry_util.shift_path(voxels, self.offset)


# --- Feature Data Structures ---
class FeatureSet:
    """
    A class for storing features and reformatting them into a form suitable
    for GNN input.
    """
    _FEATURE_TABLE = {
        "node": ("node_features", "node_index_mapping"),
        "edge": ("edge_features", "edge_index_mapping"),
        "proposal": ("proposal_features", "proposal_index_mapping"),
        "proposal_patches": ("proposal_patches", "proposal_index_mapping"),
    }
    n_branch_features = 2
    n_proposal_features = 70

    def __init__(self, graph):
        """
        Instantiates a FeatureSet object.

        Parameters
        ----------
        graph : ProposalGraph
            Graph to extract features from.
        """
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
        """
        Sets and stores a feature matrix for a given feature type.

        Parameters
        ----------
        feature_dict : Dict[hashable, numpy.ndarray]
            Mapping from object IDs to feature arrays.
        feature_type : str
            Key identifying the feature category to set. Note: must exist in
            "self._FEATURE_TABLE".
        """
        # Determine feature type
        if feature_type not in self._FEATURE_TABLE:
            raise ValueError(f"Unknown feature type: {feature_type}")
        feat_attr, index_mappping_attr = self._FEATURE_TABLE[feature_type]

        # Store features
        index_mapping = getattr(self, index_mappping_attr)
        feature_matrix = self.to_matrix(feature_dict, index_mapping)
        setattr(self, feat_attr, feature_matrix)

    # --- Helpers ---
    def get_targets(self):
        """
        Generates a binary target vector for proposals.

        Returns
        -------
        targets : numpy.ndarray
            Binary target vector for proposals.
        """
        targets = np.zeros((self.graph.n_proposals(), 1))
        idx_to_id = self.proposal_index_mapping.idx_to_id
        for idx, object_id in idx_to_id.items():
            if object_id in self.graph.gt_accepts:
                targets[idx] = 1
        return targets

    @staticmethod
    def init_matrix(feature_dict):
        """
        Initializes a  from a feature dictionary.

        Parameters
        ----------
        feature_dict : Dict[hashable, numpy.ndarray]
            Mapping from object IDs to feature arrays.

        Returns
        -------
        numpy.ndarray
            Zero-valued feature matrix with shape
                (num_objects, *feature_shape),
            where `feature_shape` is inferred from the feature dictionary.
        """
        key = util.sample_once(feature_dict.keys())
        shape = (len(feature_dict.keys()),) + feature_dict[key].shape
        return np.zeros(shape)

    def integrate_proposal_profiles(self, profiles_dict):
        """
        Integrates proposal profiles into the proposal feature matrix.

        Parameters
        ----------
        profiles_dict : dict
            Mapping from proposal IDs to profile feature arrays.
        """
        x = self.init_matrix(profiles_dict)
        for object_id in profiles_dict:
            idx = self.proposal_index_mapping.id_to_idx[object_id]
            x[idx] = profiles_dict[object_id]
        self.proposal_features = np.concatenate(
            (self.proposal_features, x), axis=1
        )

    def to_matrix(self, feature_dict, index_mapping):
        """
        Converts a dictionary of features into a dense feature matrix.

        Parameters
        ----------
        feature_dict : dict
            Mapping from object IDs to feature arrays.
        index_mapping : IndexMapping
            Data structure for mapping between object IDs and indices.

        Returns
        -------
        x : numpy.ndarray
            Dense feature matrix with shape (num_objects, feature_dim).
        """
        x = self.init_matrix(feature_dict)
        for object_id in feature_dict:
            idx = index_mapping.id_to_idx[object_id]
            x[idx] = feature_dict[object_id]
        return x


class HeteroGraphData(HeteroData):
    """
    A class for storing heterogenous graphs and reformatting them into a form
    suitable for GNN input. The graph is internally represented as a line
    graph to facilitate edge-based message passing in a GNN.
    """

    def __init__(self, features):
        """
        Instantiates a HeteroGraphData object.

        Parameters
        ----------
        features : FeatureSet
            Data structure that stores features.
        """
        # Call parent class
        super().__init__()

        # Index mappings
        self.idxs_branches = features.edge_index_mapping
        self.idxs_proposals = features.proposal_index_mapping

        # Node features
        self.x_img = torch.tensor(features.proposal_patches)
        self["branch"].x = torch.tensor(features.edge_features)
        self["proposal"].x = torch.tensor(features.proposal_features)
        self["proposal"].y = torch.tensor(features.targets)

        # Edge indices
        self.build_proposal_adjacency(features.graph)
        self.build_branch_adjacency(features.graph)
        self.build_branch_proposal_adjacency(features.graph)

    # --- Core Routines ---
    def build_proposal_adjacency(self, graph):
        """
        Builds proposal to proposal adjacency based on shared node incidence.

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
        Builds branch to branch adjacency based on shared node incidence.

        Parameters
        ----------
        graph : ProposalGraph
            Graph containing branches.
        """
        edge_index = self._build_adjacency(graph.edges, self.idxs_branches)
        self.set_edge_index(edge_index, ("branch", "to", "branch"))

    def build_branch_proposal_adjacency(self, graph):
        """
        Builds branch to proposal adjacency based on shared node incidence.

        Parameters
        ----------
        graph : ProposalGraph
            Graph containing branches and proposals.
        """
        edge_index_b2p, edge_index_p2b = list(), list()
        for proposal in graph.proposals:
            idx_proposal = self.idxs_proposals.id_to_idx[proposal]
            for i in proposal:
                for j in graph.neighbors(i):
                    branch = frozenset((i, j))
                    idx_branch = self.idxs_branches.id_to_idx[branch]
                    edge_index_b2p.append([idx_branch, idx_proposal])
                    edge_index_p2b.append([idx_proposal, idx_branch])
        self.set_edge_index(edge_index_b2p, ("branch", "to", "proposal"))
        self.set_edge_index(edge_index_p2b, ("proposal", "to", "branch"))

    # --- Helpers ---
    @staticmethod
    def _build_adjacency(edges, index_mapping):
        """
        Constructs an adjacency list for the line graph induced by the given
        set of edges. Note: the adjacency list is between edges and specifies
        whether two edges share a common vertex.

        Parameters
        ----------
        edges : List[Tuple[int]]
            Edges to determine adjacency of.
        index_mapping : IndexMapping
            Data structure for mapping between object IDs and indices.

        Returns
        -------
        edge_index : List[List[int]]
            Adjacency list for the line graph induced by the given set of
            edges.
        """
        # Build edge index
        edge_index = []
        line_graph = graph_util.edges_to_line_graph(edges)
        for e1, e2 in line_graph.edges:
            v1 = index_mapping.id_to_idx[frozenset(e1)]
            v2 = index_mapping.id_to_idx[frozenset(e2)]
            edge_index.extend([[v1, v2], [v2, v1]])
        return edge_index

    def get_feature_dict(self):
        """
        Gets a dictionary that contains the number of features for branchs and
        proposals.

        Returns
        -------
        feature_dict : Dict[str, int]
            Dictionary that contains the number of features for branchs and
            proposals.
        """
        feature_dict = {
            "branch": FeatureSet.n_branch_features,
            "proposal": FeatureSet.n_proposal_features
        }
        return feature_dict

    def get_inputs(self):
        """
        Gets inputs in a format that can passed through a GNN.

        Returns
        -------
        inputs_dict : TensorDict
            Inputs in a format that can pass through a GNN.
        """
        inputs_dict = TensorDict(
            {
                "x_dict": self.x_dict,
                "img": self.x_img,
                "edge_index_dict": self.edge_index_dict
            }
        )
        return inputs_dict

    def get_targets(self):
        """
        Gets targets in a format that can passed through a GNN.

        Returns
        -------
        TensorDict
            Targets in a format that can pass through a GNN.
        """
        return self.y_dict["proposal"]

    def n_proposals(self):
        """
        Counts the number of proposals in this data object.

        Returns
        -------
        int
            Number of proposals in the graph.
        """
        return len(self["proposal"].y)

    def set_edge_index(self, edge_index, edge_type):
        """
        Sets the edge index for a given heterogeneous edge type.

        Parameters
        ----------
        edge_index : List[Tuple[int]]
            Edge list specifying source and target node indices.
        edge_type : Tuple[str]
            Heterogeneous edge type of the form:
                (src_node_type, relation, dst_node_type).
        """
        if edge_index is None or len(edge_index) == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            edge_index = torch.as_tensor(edge_index, dtype=torch.long)
            edge_index = edge_index.t().contiguous()
            assert edge_index.shape[0] == 2, edge_index.shape
        self[edge_type].edge_index = edge_index


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
def check_list_length(my_list, min_length=2):
    """
    Checks that the list contains at least "min_length" items.

    Parameters
    ----------
    my_list : list
        List to be checked.
    min_length : int
        Minimum items that must be contained in the list
    """
    while len(my_list) < min_length:
        my_list.append(my_list[-1])


def get_feature_dict():
    """
    Gets a dictionary that contains the number of features for branchs and
    proposals.

    Returns
    -------
    Dict[str, int]
        Dictionary that contains the number of features for branchs and
        proposals.
    """
    return {"branch": 2, "proposal": 70}
