"""
Created on Sat July 15 12:00:00 2025

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code for graph neural network models that perform machine learning tasks
within NeuronProofreader pipelines.

"""

from torch import nn

import torch

from neuron_proofreader.machine_learning.vision_models import (
    CNN3D,
    init_feedforward,
)


# --- Architectures ---
class EGNN(nn.Module):

    def __init__(
        self,
        in_node_dim,
        hidden_dim,
        out_node_dim,
        in_edge_dim=0,
        device="cuda",
        act_fn=nn.SiLU(),
        n_layers=4,
        residual=True,
        attention=False,
        normalize=False,
        tanh=False,
    ):
        """
        Instantiates an EGNN object.

        Parameters
        ----------
        in_node_dim : int
            Number of features for 'h' at the input.
        hidden_dim : int
            Number of hidden features.
        out_node_dim : int
            Number of features for 'h' at the output.
        in_edge_dim : int, optional
            Number of features for the edge features.
        device : str
            Device to load model and inputs. Default is "cuda".
        act_fn : ...
            Non-linearity
        n_layers : int
            Number of layer for the EGNN.
        residual : bool
            Indication of whether to use residual connections.
        attention : bool
            Indication of whether using attention mechanism.
        normalize : bool
            Normalizes the coordinates messages such that:
                x^{l+1}_i = x^{l}_i + Σ(x_i - x_j)phi_x(m_ij)
        tanh : ...
            Sets a tanh activation function at the output of phi_x(m_ij).
        """
        # Call parent class
        super(EGNN, self).__init__()

        # Instance attributes
        self.hidden_dim = hidden_dim
        self.device = device
        self.n_layers = n_layers
        self.embedding_in = nn.Linear(in_node_dim, self.hidden_dim)
        self.embedding_out = nn.Linear(self.hidden_dim, out_node_dim)

        # Build architecture
        for i in range(0, n_layers):
            self.add_module(
                "gcl_%d" % i,
                E_GCL(
                    self.hidden_dim,
                    self.hidden_dim,
                    self.hidden_dim,
                    edges_in_dim=in_edge_dim,
                    act_fn=act_fn,
                    residual=residual,
                    attention=attention,
                    normalize=normalize,
                    tanh=tanh,
                ),
            )
        self.to(self.device)

    # --- Core Routines ---
    def forward(self, h, x, edge_index, batch):
        # Node embeddings
        h = self.embedding_in(h)
        for i in range(0, self.n_layers):
            h, x, _ = self._modules["gcl_%d" % i](h, edge_index, x)
        h = self.embedding_out(h)

        # Pool embeddings
        h = self.pool(h, x, edge_index, batch)
        print("h.size():", h.size())
        return h

    def pool(self, h, x, edge_index, batch):
        # Move to CPU
        batch = batch.detach().cpu()
        edge_index = edge_index.detach().cpu()

        # Iterate over graphs
        h_graphs = list()
        num_graphs = int(batch.max().item()) + 1
        for graph_id in range(num_graphs):
            # Build subgraph
            node_mask = batch == graph_id
            node_ids = (node_mask).nonzero(as_tuple=True)[0]
            h_g = h[node_ids]
            x_g = x[node_ids]

            # Remap nodes and edges
            id_map = {int(n): i for i, n in enumerate(node_ids.tolist())}
            edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
            edge_index_g = edge_index[:, edge_mask]
            edge_index_g = torch.stack([
                torch.tensor([id_map[int(u)] for u in edge_index_g[0]]),
                torch.tensor([id_map[int(v)] for v in edge_index_g[1]])
            ], dim=0)

            # Pool nonbranching paths for single graph
            h_graph = self._pool_single_graph(h_g, x_g, edge_index_g)
            h_graphs.append(h_graph)
        return torch.cat(h_graphs, dim=0)

    def _pool_single_graph(self, h, x, edge_index):
        # Initializations
        num_nodes = h.size(0)
        adj, deg = self.get_adj_and_deg(edge_index, num_nodes)
        visited = torch.zeros(num_nodes, dtype=torch.bool, device=h.device)

        # Pool over non-branching paths
        h_paths = list()
        x_paths = list()
        for start in range(num_nodes):
            # Start paths at endpoints or branch points
            if visited[start]:
                continue
            if deg[start] > 2:
                # branch nodes are their own "path"
                visited[start] = True
                h_paths.append(h[start])
                x_paths.append(x[start])
                continue

            # Walk non-branching path
            path = [start]
            visited[start] = True
            prev = None
            curr = start
            while True:
                nbs = [n for n in adj[curr] if n != prev]
                if len(nbs) != 1:
                    break
                nxt = nbs[0]
                if visited[nxt] or deg[nxt] > 2:
                    break
                path.append(nxt)
                visited[nxt] = True
                prev, curr = curr, nxt

            # Pool along path
            h_paths.append(self._pool(h, path))
            x_paths.append(self._pool(x, path))

        # Global pool over graph
        h_paths = torch.stack(h_paths, dim=0)
        h_graph = h_paths.mean(dim=0, keepdim=True)

        x_paths = torch.stack(x_paths, dim=0)
        x_graph = x_paths.mean(dim=0, keepdim=True)
        return torch.cat((h_graph, x_graph), dim=1)

    # --- Helpers ---
    def get_adj_and_deg(self, edge_index, num_nodes):
        # Compute node degrees
        deg = torch.zeros(num_nodes, dtype=torch.long)
        ones = torch.ones(edge_index.shape[1], dtype=torch.long)
        deg.scatter_add_(0, edge_index[0], ones)
        deg.scatter_add_(0, edge_index[1], ones)

        # Build adjacency list
        adj = [[] for _ in range(num_nodes)]
        for u, v in edge_index.t().tolist():
            adj[u].append(v)
            adj[v].append(u)
        return adj, deg

    @staticmethod
    def _pool(x, idxs):
        return x[idxs].mean(dim=0)


class E_GCL(nn.Module):
    """
    Class that implements an equivariant convolutional layer (i.e. E(n)).
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim,
        edges_in_dim=0,
        act_fn=nn.SiLU(),
        residual=True,
        attention=False,
        normalize=False,
        coords_agg="mean",
        tanh=False,
    ):
        # Call parent class
        super(E_GCL, self).__init__()

        # Instance attributes
        input_edge = input_dim * 2
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.tanh = tanh
        self.epsilon = 1e-8
        edge_coords_dim = 1

        # Architecture
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim + input_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, output_dim),
        )
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, 1, bias=False),
        )
        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_dim + edges_in_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn,
        )
        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_dim, 1), nn.Sigmoid()
            )

    def edge_model(self, source, target, radial, edge_attr):
        if edge_attr is None:
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)
        out = self.edge_mlp(out)
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        return out

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = self.node_mlp(agg)
        if self.residual:
            out = x + out
        return out, agg

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat)
        if self.coords_agg == "sum":
            agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        elif self.coords_agg == "mean":
            agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        else:
            raise Exception("Wrong coords_agg parameter" % self.coords_agg)
        coord += agg
        return coord

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum(coord_diff**2, 1).unsqueeze(1)

        if self.normalize:
            norm = torch.sqrt(radial).detach() + self.epsilon
            coord_diff = coord_diff / norm

        return radial, coord_diff

    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)

        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)

        return h, coord, edge_attr


class VisionEGNN(nn.Module):

    def __init__(self, patch_shape, output_dim=128):
        # Call parent class
        super().__init__()

        # Architecture
        self.egnn = EGNN(in_node_dim=1, hidden_dim=64, out_node_dim=output_dim)
        self.vision_model = CNN3D(
            patch_shape,
            n_conv_layers=6,
            n_feat_channels=20,
            output_dim=output_dim,
            use_double_conv=True,
        )
        self.output = init_feedforward(2 * output_dim + 3, 1, 3)

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
        # Modality-based embeddings
        x_img = self.vision_model(x["img"])
        x_graph = self.egnn(*x["graph"])
        print(x_img.size(), x_graph.size())
        x = torch.cat((x_img, x_graph), dim=1)

        # Output layer
        x = self.output(x)
        return x


def unsorted_segment_sum(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result


def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)


# --- Helpers ---
def get_edges(n_nodes):
    rows, cols = [], []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                rows.append(i)
                cols.append(j)

    edges = [rows, cols]
    return edges


def get_edges_batch(n_nodes, batch_size):
    edges = get_edges(n_nodes)
    edge_attr = torch.ones(len(edges[0]) * batch_size, 1)
    edges = [torch.LongTensor(edges[0]), torch.LongTensor(edges[1])]
    if batch_size == 1:
        return edges, edge_attr
    elif batch_size > 1:
        rows, cols = [], []
        for i in range(batch_size):
            rows.append(edges[0] + n_nodes * i)
            cols.append(edges[1] + n_nodes * i)
        edges = [torch.cat(rows), torch.cat(cols)]
    return edges, edge_attr


def subgraph_to_data(subgraph):
    h = torch.tensor(subgraph.node_radius[:, None], dtype=torch.float32)
    x = torch.tensor(subgraph.node_xyz, dtype=torch.float32)
    edges = torch.tensor(list(subgraph.edges), dtype=torch.long).T
    return h, x, edges


if __name__ == "__main__":
    # Dummy parameters
    batch_size = 8
    n_nodes = 4
    n_feat = 1
    x_dim = 3

    # Dummy variables h, x and fully connected edges
    h = torch.ones(batch_size * n_nodes, n_feat)
    x = torch.ones(batch_size * n_nodes, x_dim)
    edges, edge_attr = get_edges_batch(n_nodes, batch_size)

    # Initialize EGNN
    egnn = EGNN(
        in_node_dim=n_feat, hidden_dim=32, out_node_dim=1, in_edge_dim=1
    )

    # Run EGNN
    h, x = egnn(h, x, edges)
