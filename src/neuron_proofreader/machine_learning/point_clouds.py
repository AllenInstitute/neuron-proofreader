import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from neuron_proofreader.machine_learning.vision_models import (
    CNN3D,
    init_feedforward,
)


# --- Architectures ---
class DGCNN(nn.Module):

    def __init__(self, input_dim=3, output_dim=64, k=16):
        super().__init__()
        self.k = k  # number of neighbors

        # BatchNorm layers
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)

        # Conv layers (Conv2d on edge features)
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_dim * 2, 32, kernel_size=1, bias=False),
            self.bn1,
            nn.LeakyReLU(0.2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32 * 2, 32, kernel_size=1, bias=False),
            self.bn2,
            nn.LeakyReLU(0.2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32 * 2, 64, kernel_size=1, bias=False),
            self.bn3,
            nn.LeakyReLU(0.2),
        )

        # Final fully-connected layer for embedding
        self.fc = nn.Linear(128, output_dim)

    def get_graph_feature(self, x, k):
        # x: (B, C, N)
        # Compute pairwise distance
        B, C, N = x.size()
        x = x.transpose(2, 1)  # (B, N, C)
        idx = self.knn(x, k)  # (B, N, k)

        # Gather neighbors
        neighbors = self.index_points(x, idx)  # (B, N, k, C)
        x = x.unsqueeze(2).expand_as(neighbors)  # (B, N, k, C)
        edge_feature = torch.cat((x, neighbors - x), dim=-1)  # (B, N, k, 2C)
        return edge_feature.permute(0, 3, 1, 2).contiguous()  # (B, 2C, N, k)

    @staticmethod
    def knn(x, k):
        # Compute pairwise distance
        inner = -2 * torch.matmul(x, x.transpose(2, 1))
        xx = torch.sum(x**2, dim=2, keepdim=True)
        pairwise_distance = -xx - inner - xx.transpose(2, 1)
        _, idx = pairwise_distance.topk(k=k, dim=-1)  # (B, N, k)
        return idx

    @staticmethod
    def index_points(x, idx):
        # x: (B, N, C), idx: (B, N, k)
        B = x.size(0)
        batch_indices = (
            torch.arange(B, device=x.device)
            .view(B, 1, 1)
            .repeat(1, idx.size(1), idx.size(2))
        )
        return x[batch_indices, idx, :]

    def forward(self, x):
        x1 = F.relu(self.conv1(self.get_graph_feature(x, self.k)))
        x1 = x1.max(dim=-1)[0]

        x2 = F.relu(self.conv2(self.get_graph_feature(x1, self.k)))
        x2 = x2.max(dim=-1)[0]

        x3 = F.relu(self.conv3(self.get_graph_feature(x2, self.k)))
        x3 = x3.max(dim=-1)[0]

        x_cat = torch.cat((x1, x2, x3), dim=1)
        x_global = x_cat.max(dim=-1)[0]
        return self.fc(x_global)


class VisionDGCNN(nn.Module):

    def __init__(self, patch_shape, output_dim=128):
        super().__init__()

        self.dgcnn = DGCNN(output_dim=output_dim)
        self.vision_model = CNN3D(
            patch_shape,
            n_conv_layers=6,
            n_feat_channels=20,
            output_dim=output_dim,
            use_double_conv=True,
        )
        self.output = init_feedforward(2 * output_dim, 1, 3)

    def forward(self, x):
        """
        Passes the given input through this neural network.

        Parameters
        ----------
        x : torch.Tensor
            Input vector of features.

        Returns
        -------
        x : torch.Tensor
            Output of the neural network.
        """
        x_img = self.vision_model(x["img"])
        x_pc = self.dgcnn(x["point_cloud"])
        x = torch.cat((x_img, x_pc), dim=1)
        x = self.output(x)
        return x


# --- Point Cloud Generation ---
def subgraph_to_point_cloud(graph, n_points=3200):
    point_cloud = list()
    for n1, n2 in graph.edges:
        # Use average radius
        r1 = graph.node_radius[n1]
        r2 = graph.node_radius[n2]
        r = (r1 + r2) / 2

        pts = sample_cylinder_between_points(
            graph.node_xyz[n1], graph.node_xyz[n2], r
        )
        point_cloud.append(pts)
    point_cloud = np.vstack(point_cloud)

    total_points = point_cloud.shape[0]
    idxs = np.arange(len(point_cloud))
    if total_points >= n_points:
        # Downsample randomly
        sampled_idxs = np.random.choice(idxs, n_points, replace=False)
        point_cloud = point_cloud[sampled_idxs]
    else:
        # Upsample with replacement
        sampled_idxs = np.random.choice(idxs, n_points, replace=True)
        point_cloud = point_cloud[sampled_idxs]
    return point_cloud.T


def sample_cylinder_between_points(p1, p2, r, n_samples=25):
    p1 = np.array(p1)
    p2 = np.array(p2)
    axis = p2 - p1
    axis_length = np.linalg.norm(axis)
    axis_dir = axis / axis_length

    # Build orthogonal basis
    if np.allclose(axis_dir, [0, 0, 1]):
        ortho1 = np.array([1, 0, 0])
    else:
        ortho1 = np.cross(axis_dir, [0, 0, 1])
        ortho1 /= np.linalg.norm(ortho1)
    ortho2 = np.cross(axis_dir, ortho1)

    # Random samples along axis
    t_values = np.random.rand(n_samples)
    centers = p1 + np.outer(t_values, axis)

    # Random samples in circular cross-section
    angles = np.random.rand(n_samples) * 2 * np.pi
    radii = r * np.sqrt(np.random.rand(n_samples))
    offsets = np.outer(radii * np.cos(angles), ortho1) + np.outer(
        radii * np.sin(angles), ortho2
    )
    points = centers + offsets
    return points
