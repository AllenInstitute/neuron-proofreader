"""
Created on Wed July 2 14:00:00 2025

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Implementation of FragmentsGraph, a subclass of SkeletonGraph that represents
a collection of neuron fragments and provides proofreading-specific operations.

"""

from collections import defaultdict
from scipy.spatial import KDTree
from tqdm import tqdm

import networkx as nx
import numpy as np

from arborist.skeleton_graph import SkeletonGraph
from arborist.utils.graph_loading import GraphLoader, count_nodes
from neuron_proofreader.utils import geometry_util, img_util, util


class FragmentsGraph(SkeletonGraph):
    """
    Subclass of SkeletonGraph for neuron fragment graphs. Extends the base
    class with SWC loading, soma handling, and proofreading-specific
    operations.
    """

    def __init__(
        self,
        anisotropy=(1.0, 1.0, 1.0),
        min_cable_length=0,
        min_swc_pts=1,
        node_spacing=1,
        prune_depth=20,
        use_anisotropy=True,
        verbose=False,
    ):
        """
        Instantiates a FragmentsGraph object.

        Parameters
        ----------
        anisotropy : Tuple[float], optional
            Image to physical coordinates scaling factors. Default is
            (1.0, 1.0, 1.0).
        min_cable_length : float, optional
            Minimum path length of fragments loaded into graph. Default is 0.
        node_spacing : float, optional
            Distance (in microns) between neighboring nodes. Default is 1μm.
        prune_depth : float, optional
            Branches shorter than this (in microns) are removed. Default is
            20μm.
        use_anisotropy : bool, optional
            Whether to apply anisotropy to SWC coordinates. Default is True.
        verbose : bool, optional
            Whether to display progress bars. Default is False.
        """
        super().__init__(
            anisotropy=anisotropy,
            node_spacing=node_spacing,
            verbose=verbose,
        )
        self.soma_centroids = list()
        self.soma_component_ids = list()

        anisotropy_actual = anisotropy if use_anisotropy else (1.0, 1.0, 1.0)
        self.graph_loader = GraphLoader(
            anisotropy=anisotropy_actual,
            min_cable_length=min_cable_length,
            min_swc_pts=min_swc_pts,
            node_spacing=node_spacing,
            prune_depth=prune_depth,
            verbose=verbose,
        )

    # --- Node Attribute Helpers ---
    def resize_node_attr(self, new_shape, attr_name):
        node_attr = getattr(self, attr_name)
        new_node_attr = np.empty(new_shape, dtype=node_attr.dtype)
        new_node_attr[: len(node_attr)] = node_attr
        setattr(self, attr_name, new_node_attr)

    # --- Load ---
    def load(self, swc_pointer):
        """
        Loads SWC files into the graph.

        Parameters
        ----------
        swc_pointer : str
            Object that points to SWC files to be loaded.
        """
        irreducibles = self.graph_loader(swc_pointer)

        num_nodes = count_nodes(irreducibles)
        self.node_component_id = np.zeros((num_nodes), dtype=int)
        self.node_radius = np.zeros((num_nodes), dtype=np.float16)
        self.node_xyz = np.zeros((num_nodes, 3), dtype=np.float32)

        component_id = 0
        while irreducibles:
            self.add_connected_component(irreducibles.pop(), component_id)
            component_id += 1

        self.check_swc_ids()
        self.set_kdtree()

    # --- Soma Operations ---
    def load_somas(self, soma_centroids):
        num_components = nx.number_connected_components(self)
        num_nodes = self.number_of_nodes()
        num_somas = len(soma_centroids)

        self.resize_node_attr((num_nodes + num_somas), "node_component_id")
        self.resize_node_attr((num_nodes + num_somas), "node_radius")
        self.resize_node_attr((num_nodes + num_somas, 3), "node_xyz")

        for idx, xyz in enumerate(soma_centroids, start=1):
            node_id = self.number_of_nodes()
            assert node_id not in self.nodes
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
            self.component_id_to_swc_id[component_id] = swc_id
            self.node_component_id[node_id] = component_id
            self.node_radius[node_id] = 20
            self.node_xyz[node_id] = xyz
            self.soma_centroids.append(xyz)

        self.relabel_nodes()

    def connect_soma_fragments(self, max_dist=25):
        merge_cnt, somas_connected = 0, list()
        for soma_node in self.soma_nodes():
            soma_xyz = self.node_xyz[soma_node]
            nodes = self.kdtree.query_ball_point(soma_xyz, max_dist)
            nodes = np.array(nodes, dtype=int)
            for cid in np.unique(self.node_component_id[nodes]):
                soma_component_id = self.node_component_id[soma_node]
                if cid != soma_component_id:
                    idxs = np.where(self.node_component_id[nodes] == cid)[0]
                    dists = np.sum(
                        (self.node_xyz[nodes[idxs]] - soma_xyz) ** 2, axis=1
                    )
                    node = nodes[idxs[np.argmin(dists)]]
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

    def soma_nodes(self):
        soma_nodes = list()
        for dist_i, i in map(self.kdtree.query, self.soma_centroids):
            if dist_i < 5:
                soma_nodes.append(i)
        return soma_nodes

    def remove_merge_sites(self, merge_site_nodes, max_depth=10):
        """
        Removes detected merge sites and their local neighborhoods from the
        graph.

        Parameters
        ----------
        merge_site_nodes : list[int]
            Node IDs identified as merge sites.
        max_depth : float, optional
            Radius (in microns) around each merge site to remove. Default
            is 10.
        """
        rm_nodes = set()
        for root in tqdm(merge_site_nodes, desc="Remove Merge Sites"):
            root = self.find_nearby_branching_node(root)
            nbhd = self.nodes_within_distance(root, max_depth)
            for i in list(nbhd):
                if i != root and self.degree[i] >= 3:
                    nbhd.extend(self.nodes_within_distance(root, 8))
            rm_nodes.update(set(nbhd))
        self.remove_nodes(rm_nodes)
        print("# Nodes Deleted:", len(rm_nodes))

    # --- Image Coordinate Helpers ---
    def node_voxel(self, i):
        """
        Gets the voxel coordinate of the given node.
        """
        return img_util.to_voxels(self.node_xyz[i], self.anisotropy)

    def node_local_voxel(self, node, offset):
        """
        Computes the local voxel coordinate of the given node within a patch.
        """
        return tuple([v - o for v, o in zip(self.node_voxel(node), offset)])

    def clip_to_bbox(self, metadata_path):
        """
        Clips skeletons to the bounding box defined in a metadata JSON file.
        """
        if util.check_gcs_file_exists(metadata_path):
            metadata = util.read_json(metadata_path)
            origin = metadata["chunk_origin"][::-1]
            shape = metadata["chunk_shape"][::-1]
            nodes = list()
            for i in self.nodes:
                voxel = np.array(self.node_voxel(i))
                if not img_util.is_contained(voxel - origin, shape):
                    nodes.append(i)
            self.remove_nodes_from(nodes)
            self.relabel_nodes()

    def tangent_from_leaf(self, leaf, max_depth=np.inf):
        """
        Computes the tangent vector of the path emanating from a leaf.
        """
        path = self.path_from_leaf(leaf, max_depth=max_depth)
        return geometry_util.tangent(self.node_xyz[np.array(path)])

    def __repr__(self):
        n_components = format(nx.number_connected_components(self), ",")
        n_nodes = format(self.number_of_nodes(), ",")
        n_edges = format(self.number_of_edges(), ",")
        return (
            f"   FragmentsGraph(\n"
            f"      num_connected_components={n_components},\n"
            f"      num_nodes={n_nodes},\n"
            f"      num_edges={n_edges},\n"
            f"   )"
        )
