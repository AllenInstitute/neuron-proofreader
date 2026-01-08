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

import numpy as np

from neuron_proofreader.utils import geometry_util, img_util, util


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
        is_multimodal=True,
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
        is_multimodal : bool, optional
            Indication of whether model inputs are both feature vectors and
            image patches. Default is True.
        patch_shape : Tuple[int], optional
            Shape of image patch expected by the vision model. Default is (96,
            96, 96).
        segmentation_path : str, optional
            Path to segmentation of whole-brain dataset.
        """
        # Instance attributes
        if is_multimodal:
            self.extractors = [
                SkeletonFeatureExtractor(search_radius),
                ImageFeatureExtractor(
                    img_path,
                    patch_shape=patch_shape,
                    padding=padding,
                    segmentation_path=segmentation_path,
                ),
            ]
        else:
            self.extractors = [
                SkeletonFeatureExtractor(search_radius),
                ImageProfileExtractor(img_path),
            ]

    def run(self, graph):
        """
        Runs the feature extraction pipeline.

        Parameters
        ----------
        graph : ProposalGraph
            Graph to extract features from.
        """
        features = FeatureSet()
        for extractor in self.extractors:
            extractor.extract(graph, features)
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

    def extract(self, graph, features):
        """
        Extracts skeleton-based features for nodes, edges, and proposals.

        Parameters
        ----------
        graph : ProposalGraph
            Graph to extract features for.
        features : FeatureSet
            Data structure that stores features.
        """
        features.node_features = self.extract_node_features(graph)
        features.edge_features = self.extract_edge_features(graph)
        features.proposal_features = self.extract_proposal_features(graph)

    def extract_node_features(self, graph):
        """
        Extracts skeleton-based features for nodes.

        Parameters
        ----------
        graph : ProposalGraph
            Graph to generate features for.

        Returns
        -------
        features : Dict[int, numpy.ndarray]
            Dictionary that maps a node to its feature vector.
        """
        features = dict()
        for u in graph.nodes:
            features[u] = np.array(
                [
                    graph.degree[u],
                    graph.node_radius[u],
                    len(graph.node_proposals[u]),
                ]
            )
        return features

    def extract_edge_features(self, graph):
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
        features = dict()
        for edge in graph.edges:
            features[frozenset(edge)] = np.array(
                [
                    np.mean(graph.edges[edge]["radius"]),
                    min(graph.edge_length(edge), 5000) / 5000,
                ],
            )
        return features

    def extract_proposal_features(self, graph):
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
        features = dict()
        for p in graph.proposals:
            features[p] = np.concatenate(
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
        return features


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
        self.img_reader = img_util.TensorStoreReader(img_path)
        if segmentation_path:
            self.segmentation_reader = img_util.TensorStoreReader(
                segmentation_path
            )
        else:
            self.segmentation_reader = None

    def extract(self, graph, features):
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
            for thread in as_completed(pending.keys()):
                p = pending.pop(thread)
                patches, profile = thread.result()
                features.proposal_patches[p] = patches
                features.proposal_profiles[p] = profile

    def get_patches(self, graph, proposal):
        # Read images
        center, shape = self.compute_proposal_crop(graph, proposal)
        img_patch = self.read_image_patch(center, shape)
        offset = img_util.get_offset(center, shape)

        # Generate image profile
        profile_line = self.get_profile_line(graph, proposal, offset, 16)
        profile = [img_patch[tuple(voxel)] for voxel in profile_line]
        profile.extend([np.mean(profile), np.std(profile)])

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
        patch = self.img_reader.read(center, shape)
        patch = img_util.normalize(np.minimum(patch, self.brightness_clip))
        return patch

    def read_segmentation_mask(self, center, shape):
        if self.segmentation_reader:
            patch = self.segmentation_reader.read(center, shape)
            return 0.25 * (patch > 0).astype(float)
        else:
            return np.zeros(shape)


class ImageProfileExtractor:
    """
    A class for extracting image profiles along proposals.
    """

    def __init__(self):
        pass

    def extract(self):
        pass


# --- Feature Data Structures ---
class FeatureSet:

    def __init__(self):
        # Instance Attributes
        self.node_features = dict()
        self.edge_features = dict()
        self.proposal_features = dict()
        self.proposal_patches = dict()
        self.proposal_profiles = dict()

    def to_matrix(self):
        pass


# --- Build feature matrix ---
def get_matrix(features, gt_accepts=set()):
    # Initialize matrices
    key = util.sample_once(list(features.keys()))
    x = np.zeros((len(features.keys()), len(features[key])))
    y = np.zeros((len(features.keys())))

    # Populate
    idx_to_id = dict()
    for i, id_i in enumerate(features):
        idx_to_id[i] = id_i
        x[i, :] = features[id_i]
        y[i] = 1 if id_i in gt_accepts else 0
    return x, y, init_idx_mapping(idx_to_id)


def get_patches_matrix(patches, id_to_idx):
    patch = util.sample_once(list(patches.values()))
    x = np.zeros((len(id_to_idx),) + patch.shape)
    for key, patch in patches.items():
        x[id_to_idx[key], ...] = patch
    return x


def init_idx_mapping(idx_to_id):
    """
    Adds dictionary item called "edge_to_index" which maps a branch/proposal
    in a FragmentsGraph to an idx that represents it's position in the feature
    matrix.

    Parameters
    ----------
    idxs : dict
        Dictionary that maps indices to edges in a FragmentsGraph.

    Returns
    -------
    dict
        Updated dictionary.
    """
    idx_mapping = {
        "idx_to_id": idx_to_id,
        "id_to_idx": {v: k for k, v in idx_to_id.items()},
    }
    return idx_mapping


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
