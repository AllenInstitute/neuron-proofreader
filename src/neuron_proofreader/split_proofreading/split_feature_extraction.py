"""
Created on Tue Jan 13 15:00:00 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code for generating features used by machine learning models to perform split
correction.

"""

from concurrent.futures import as_completed, ThreadPoolExecutor
from scipy.spatial import KDTree

from torch_geometric.data import HeteroData

import numpy as np
import torch

from neuron_proofreader.configs import ImageConfig
from neuron_proofreader.machine_learning.image_dataloader import (
    ProposalPatchLoader,
)
from arborist.utils.graph_utils import edges_to_line_graph
from neuron_proofreader.utils import geometry_util, img_util, util
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
        percentiles=(1, 99.5),
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
        percentiles : Tuple[float], optional
            Upper and lower percentiles used to normalize image patches.
            Default is (1, 99.5).
        """
        self.skeleton_extractor = SkeletonFeatureExtractor(graph)
        self.image_extractor = ImageFeatureExtractor(
            graph,
            img_path,
            brightness_clip=brightness_clip,
            patch_shape=patch_shape,
            padding=padding,
            percentiles=percentiles,
        )

    def __call__(self, subgraph, reads=None):
        """
        Runs the feature extraction pipeline.

        Note: the image patch reads are launched before the skeleton features
        are computed so that the reads overlap that computation. The image
        features are stored afterwards, since the proposal profiles are
        concatenated onto the skeleton-based proposal features.

        Parameters
        ----------
        subgraph : ProposalComputationGraph
            Subgraph of "graph" attribute to extract features for.
        reads : dict, optional
            Image reads for this subgraph that were already started with
            "issue_reads" (e.g. while the previous batch was being
            processed). Default is None.
        """
        features = FeatureSet(subgraph)
        with ThreadPoolExecutor(max_workers=1) as executor:
            thread = executor.submit(
                self.image_extractor.extract_batch, subgraph, reads
            )
            self.skeleton_extractor(subgraph, features)
            patches, profiles = thread.result()
        self.image_extractor.update_features(features, patches, profiles)
        return features

    def issue_reads(self, subgraph):
        """
        Starts the image reads for a subgraph without waiting for them; pass
        the result to "__call__" later.
        """
        return self.image_extractor.issue_reads(subgraph)


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
        self.kdtree = KDTree(graph.node_xyz[np.array(graph.leaf_nodes())])

    def __call__(self, subgraph, features):
        """
        Extracts skeleton-based features for nodes, edges, and proposals.

        Parameters
        ----------
        subgraph : ProposalComputationGraph
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
        subgraph : ProposalComputationGraph
            Subgraph of "graph" attribute to extract features for.
        """
        node_features = dict()
        for i in subgraph.pg_nodes:
            node_features[i] = np.array(
                [
                    self.graph.degree(i),
                    self.graph.node_feats["radius"][i],
                    len(self.graph.node_proposals[i]),
                ]
            )
        features.set_features(node_features, "node")

    def extract_edge_features(self, subgraph, features):
        """
        Extracts skeleton-based features for edges.

        Parameters
        ----------
        subgraph : ProposalComputationGraph
            Subgraph of "graph" attribute to extract features for.

        Returns
        -------
        features : Dict[Frozenset[int], numpy.ndarray]
            Dictionary that maps an edge to its feature vector.
        """
        edge_features = dict()
        for edge in subgraph.pg_edges:
            path = subgraph.edge_to_path[edge]
            edge_features[edge] = np.array(
                [
                    np.mean(self.graph.node_feats["radius"][path]),
                    min(self.graph.path_length(path), 5000) / 5000,
                ],
            )
        features.set_features(edge_features, "edge")

    def extract_proposal_features(self, subgraph, features):
        """
        Extracts skeleton-based features for proposals.

        Parameters
        ----------
        subgraph : ProposalComputationGraph
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
                    int(self.graph.is_leaf2leaf(p)),
                    self.count_nearby_leafs(p),
                    self.graph.proposal_length(p) / self.graph.search_radius,
                    self.graph.proposal_radius(p),
                    self.graph.proposal_directionals(p, 16),
                    self.graph.proposal_directionals(p, 32),
                    self.graph.proposal_directionals(p, 64),
                    self.graph.proposal_directionals(p, 128),
                ),
                axis=None,
            )
        features.set_features(proposal_features, "proposal")

    def count_nearby_leafs(self, proposal):
        """
        Counts the number of nearby leaf nodes.

        Parameters
        ----------
        proposal : Frozenset[int]
            Proposal to generate feature for.

        Returns
        -------
        int
            Number of leaf nodes close to the nodes forming the given
            proposal.
        """
        xyz_i, xyz_j = self.graph.proposal_xyz(proposal)
        pts_i = self.kdtree.query_ball_point(xyz_i, self.graph.search_radius)
        pts_j = self.kdtree.query_ball_point(xyz_j, self.graph.search_radius)
        return len(pts_i) + len(pts_j)


class ImageFeatureExtractor:
    """
    A class for extracting image patches, image profiles along proposals, and
    generating masks that indicate the spatial locations of proposals.

    Attributes
    ----------
    max_workers : int
        Number of threads that turn image patches into features. Image reads
        are not bound by this: every read of a batch is issued up front and
        runs on TensorStore's own thread pool, so these threads only do CPU
        work. Keeping this small matters because the threads hold the GIL
        while annotating masks, and the GIL is shared with the thread that
        launches the GPU forward pass.
    """

    max_workers = 6

    def __init__(
        self,
        graph,
        img_path,
        brightness_clip=400,
        patch_shape=(96, 96, 96),
        padding=40,
        percentiles=(1, 99.5),
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
        percentiles : Tuple[float], optional
            Upper and lower percentiles used to normalize image patches.
            Default is (1, 99.5).
        """
        img_config = ImageConfig(
            brightness_clip=brightness_clip,
            img_path=img_path,
            patch_shape=patch_shape,
            percentiles=percentiles,
        )
        self.graph = graph
        self.patch_loader = ProposalPatchLoader(
            graph, img_config, padding=padding
        )
        self.patch_shape = patch_shape
        self.padding = padding

        # Patches are resized with torch ops from "max_workers" Python
        # threads. Each thread would otherwise spin up its own intra-op
        # thread pool (max_workers x cores threads), which starves the rest
        # of the pipeline; one thread per op is fastest here.
        torch.set_num_threads(1)

    def __call__(self, subgraph, features):
        """
        Extracts image patches and profiles for each proposal in the graph.

        Parameters
        ----------
        subgraph : ProposalComputationGraph
            Subgraph of "graph" attribute to extract features for.
        features : FeatureSet
            Data structure that stores features.
        """
        patches, profiles = self.extract_batch(subgraph)
        self.update_features(features, patches, profiles)

    def issue_reads(self, subgraph):
        """
        Starts the image read of every proposal in the subgraph. The reads
        run on TensorStore's thread pool, so issuing them early (ideally
        while the previous batch is still being processed) hides most of the
        read latency.

        Parameters
        ----------
        subgraph : ProposalComputationGraph
            Subgraph whose proposals' patches are to be read.

        Returns
        -------
        Dict[Frozenset[int], tuple]
            Pending read per proposal, see "ProposalPatchLoader.read_async".
        """
        return {p: self.patch_loader.read_async(p) for p in subgraph.proposals}

    def extract_batch(self, subgraph, reads=None):
        """
        Extracts an image patch and intensity profile for each proposal in the
        given subgraph.

        Parameters
        ----------
        subgraph : ProposalComputationGraph
            Subgraph of "graph" attribute to extract features for.
        reads : Dict[Frozenset[int], tuple], optional
            Pending image reads from "issue_reads". Default is None, in which
            case they are issued here.

        Returns
        -------
        patches : Dict[Frozenset[int], numpy.ndarray]
            Dictionary that maps a proposal to its image patch.
        profiles : Dict[Frozenset[int], numpy.ndarray]
            Dictionary that maps a proposal to its intensity profile.
        """
        # All reads are in flight before any CPU work starts (see max_workers)
        reads = reads if reads is not None else self.issue_reads(subgraph)
        proposals = list(subgraph.proposals)

        patches, profiles = dict(), dict()
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            pending = {
                executor.submit(self.extract, p, reads[p]): p
                for p in proposals
            }
            for thread in as_completed(pending):
                proposal = pending[thread]
                profiles[proposal], patches[proposal] = thread.result()
        return patches, profiles

    @staticmethod
    def update_features(features, patches, profiles):
        """
        Stores the extracted image patches and profiles.

        Parameters
        ----------
        features : FeatureSet
            Data structure that stores features.
        patches : Dict[Frozenset[int], numpy.ndarray]
            Dictionary that maps a proposal to its image patch.
        profiles : Dict[Frozenset[int], numpy.ndarray]
            Dictionary that maps a proposal to its intensity profile.
        """
        features.set_features(patches, "proposal_patches")
        features.integrate_proposal_profiles(profiles)

    def extract(self, proposal, read=None):
        """
        Extracts the intensity profile and input patch for a proposal.

        Parameters
        ----------
        proposal : Frozenset[int]
            Proposal that image patches are centered about.
        read : tuple, optional
            Pending image read for this proposal, see
            "ProposalPatchLoader.read_async". Default is None.

        Returns
        -------
        Tuple[numpy.ndarray]
            Intensity profile and input patch.
        """
        img, offset = self.patch_loader(proposal, read=read)
        mask = self.create_segment_mask(proposal, img.shape, offset)
        extractor = PatchFeatureExtractor(
            self.graph, img, mask, proposal, offset, self.patch_shape
        )
        return extractor.get_intensity_profile(), extractor.get_input_patch()

    # --- Helpers ---
    def create_segment_mask(self, proposal, shape, offset):
        # Find edges between nearby nodes
        center = self.graph.proposal_midpoint(proposal)
        nodes = self.graph.kdtree.query_ball_point(center, self.padding + 10)
        node_set = set(nodes)
        edges = [
            (i, j)
            for i in nodes
            for j in self.graph.neighbors(i)
            if i < j and j in node_set
        ]

        # Rasterize all edges at once
        mask = np.zeros(shape, dtype=np.float32)
        if edges:
            edges = np.asarray(edges, dtype=int)
            v = self.graph.nodes_local_voxels(edges.ravel(), offset)
            v = v.reshape(-1, 2, 3)
            voxels = geometry_util.make_digital_lines(v[:, 0], v[:, 1])
            img_util.annotate_voxels(mask, voxels, fill_val=0.25)
        return mask


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
        patch_shape : Tuple[int], optional
            Shape of image patch expected by model. Default is (96, 96, 96).
        """
        # Instance attributes
        self.graph = graph
        self.img = img
        self.mask = mask
        self.proposal = proposal
        self.offset = offset
        self.patch_shape = patch_shape

        # Annotate mask
        i, j = self.proposal
        self.voxels = {u: self.get_branch_voxels(u) for u in [i, j]}
        self.annotate_edge(i)
        self.annotate_edge(j)
        self.annotate_proposal()

    # --- Core Routines ---
    def get_input_patch(self):
        """
        Gets input patch from the image and segmentation mask.

        Returns
        -------
        numpy.ndarray
            Float16 array with shape (2, *patch_shape), where channel 0
            contains raw image data and channel 1 contains segmentation data.
            The cast to float16 is done here with torch because numpy's
            float16 conversion is slow enough to dominate batch assembly.
        """
        img = img_util.resize(self.img, self.patch_shape)
        mask = resize_segmentation(self.mask, self.patch_shape)
        patch = torch.from_numpy(np.stack([img, mask], axis=0))
        return patch.to(torch.float16).numpy()

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
        profile = self._extract_profile(self.voxels[node])
        profile = geometry_util.resample_curve_1d(profile, 16)
        return profile

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
        voxels = np.asarray(check_list_length(voxels, min_length=16))
        profile = self.img[voxels[:, 0], voxels[:, 1], voxels[:, 2]]
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
        img_util.annotate_voxels(self.mask, self.voxels[node], fill_val=0.5)

    def annotate_proposal(self):
        """
        Annotates the proposal within the given mask.
        """
        voxels = self.get_profile_line()
        img_util.annotate_voxels(self.mask, voxels, fill_val=1)

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
        voxel1 = self.graph.node_local_voxel(node1, self.offset)
        voxel2 = self.graph.node_local_voxel(node2, self.offset)
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
        queue = [(node, self.graph.node_local_voxel(node, self.offset))]
        visited = {node}
        voxels = list()
        while queue:
            # Visit node
            i, voxel_i = queue.pop()
            voxels.append(voxel_i)

            # Update queue
            for j in self.graph.neighbors(i):
                voxel_j = self.graph.node_local_voxel(j, self.offset)
                in_j = img_util.is_contained(voxel_j, self.img.shape)
                if in_j and j not in visited:
                    queue.append((j, voxel_j))
                    visited.add(j)
        return geometry_util.make_voxels_connected(voxels)


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
        graph : ProposalComputationGraph
            Graph to extract features from.
        """
        # Instance Attributes
        self.graph = graph
        self.node_index_mapping = IndexMapping(graph.pg_nodes)
        self.edge_index_mapping = IndexMapping(graph.pg_edges)
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
        dtype = np.float16 if feature_type == "proposal_patches" else np.float32
        feature_matrix = self.to_matrix(feature_dict, index_mapping, dtype)
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
    def init_matrix(feature_dict, dtype=np.float32):
        """
        Initializes a feature matrix from a feature dictionary.

        Parameters
        ----------
        feature_dict : Dict[hashable, numpy.ndarray]
            Mapping from object IDs to feature arrays.
        dtype : numpy.dtype, optional
            Data type of the matrix. Default is float32. Image patches are
            stored as float16: the model runs under fp16 autocast, so the
            first convolution would cast them to fp16 anyway, and this halves
            host memory and the host-to-device transfer.

        Returns
        -------
        numpy.ndarray
            Zero-valued feature matrix with shape
                (num_objects, *feature_shape),
            where `feature_shape` is inferred from the feature dictionary.
        """
        key = util.sample_once(feature_dict.keys())
        shape = (len(feature_dict.keys()),) + feature_dict[key].shape
        return np.zeros(shape, dtype=dtype)

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

    def to_matrix(self, feature_dict, index_mapping, dtype=np.float32):
        """
        Converts a dictionary of features into a dense feature matrix.

        Parameters
        ----------
        feature_dict : dict
            Mapping from object IDs to feature arrays.
        index_mapping : IndexMapping
            Data structure for mapping between object IDs and indices.
        dtype : numpy.dtype, optional
            Data type of the matrix. Default is float32.

        Returns
        -------
        x : numpy.ndarray
            Dense feature matrix with shape (num_objects, feature_dim).
        """
        x = self.init_matrix(feature_dict, dtype)
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

        # Node features (from_numpy avoids a second copy of the patches)
        self.x_img = torch.from_numpy(features.proposal_patches)
        self["branch"].x = torch.tensor(features.edge_features)
        self["proposal"].x = torch.tensor(features.proposal_features)
        self["proposal"].y = torch.tensor(features.targets)

        # Edge indices
        self.build_proposal_adjacency(features.graph.proposals)
        self.build_branch_adjacency(features.graph)
        self.build_branch_proposal_adjacency(features.graph)

    # --- Core Routines ---
    def build_proposal_adjacency(self, proposals):
        """
        Builds proposal to proposal adjacency based on shared node incidence.

        Parameters
        ----------
        proposals : Set[Frozenset[int]]
            Proposals to be predicted.
        """
        edge_index = self._build_adjacency(proposals, self.idxs_proposals)
        self.set_edge_index(edge_index, ("proposal", "to", "proposal"))

    def build_branch_adjacency(self, graph):
        """
        Builds branch to branch adjacency based on shared node incidence.

        Parameters
        ----------
        graph : networkx.Graph
            Irreducible graph containing branches.
        """
        edge_index = self._build_adjacency(graph.pg_edges, self.idxs_branches)
        self.set_edge_index(edge_index, ("branch", "to", "branch"))

    def build_branch_proposal_adjacency(self, graph):
        """
        Builds branch to proposal adjacency based on shared node incidence.

        Parameters
        ----------
        graph : ProposalComputationGraph
            Irreducible graph containing branches.
        proposals : Set[Frozenset[int]]
            Proposals to be predicted.
        """
        edge_index_b2p, edge_index_p2b = list(), list()
        for proposal in graph.proposals:
            idx_proposal = self.idxs_proposals.id_to_idx[proposal]
            for i in proposal:
                for j in graph.pg_neighbors(i):
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
        line_graph = edges_to_line_graph(edges)
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
            "proposal": FeatureSet.n_proposal_features,
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
                "edge_index_dict": self.edge_index_dict,
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
def check_list_length(arr, min_length=2):
    """
    Checks that the array contains at least "min_length" items.

    Parameters
    ----------
    arr : list
        Array to be checked.
    min_length : int
        Minimum length of the array.
    """
    if arr.shape[0] < min_length and arr.shape[0] > 0:
        pad_size = min_length - arr.shape[0]
        padding = np.repeat(arr[-1:], pad_size, axis=0)
        return np.concatenate([arr, padding], axis=0)
    elif arr.shape[0] == 0:
        return np.array([(0, 0, 0)] * min_length)
    else:
        return arr


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
    return {"branch": 2, "proposal": 67}


def resize_segmentation(mask, new_shape):
    """
    Resizes a segmentation mask to the given new shape.

    Parameters
    ----------
    mask : numpy.ndarray
        Segmentation mask to be resized.
    new_shape : Tuple[int]
        New shape of segmentation mask.

    Returns
    -------
    mask : numpy.ndarray
        Resized segmentation mask.
    """
    return img_util.resize_nearest(mask, new_shape)
