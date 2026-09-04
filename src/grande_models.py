from typing import Optional

import torch
import torch.nn.functional as F
import torch_sparse
from torch import Tensor
from torch_scatter import scatter_add
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.utils import (
    add_remaining_self_loops,
    add_self_loops as add_self_loops_fn,
    is_torch_sparse_tensor,
    spmm,
    to_edge_index,
)
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils.sparse import set_sparse_value
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, SparseTensor


def min_max_edge_distances_normalization(edge_distances):
    """Min-max normalize a tensor of edge distances to the ``[0, 1]`` range.

    Args:
        edge_distances: 1D tensor with one distance per edge.

    Returns:
        Tensor with the normalized distances.
    """
    min_distance = torch.min(edge_distances).item()
    max_distance = torch.max(edge_distances).item()

    normalized_distances = (edge_distances - min_distance) / (max_distance - min_distance)
    return normalized_distances


def grande(x, edge_index, sigma=0.2):
    """Compute the GRaNDe score for every node.

    GRaNDe (Gaussian Rank-Based Neighborhood Degree) augments the plain node
    degree with a dissimilarity term derived from an RBF kernel over the
    (min-max normalized) Euclidean distances between connected nodes.

    Args:
        x: Node feature matrix ``(num_nodes, num_features)``.
        edge_index: Edge index tensor ``(2, num_edges)``.
        sigma: Bandwidth of the Gaussian (RBF) kernel.

    Returns:
        Tensor ``(num_nodes,)`` with the GRaNDe score of each node.
    """
    num_nodes = x.size(0)
    src, dst = edge_index[0], edge_index[1]

    # Node degree.
    degree = scatter_add(torch.ones_like(src), src, dim=0, dim_size=num_nodes)

    # Euclidean distance along each edge.
    edge_distances = torch.norm(x[src] - x[dst], p=2, dim=1)

    # Avoid division by zero.
    epsilon = 1e-12
    edge_distances = torch.clamp(edge_distances, min=epsilon)

    # Min-max normalization.
    normalized_distance = min_max_edge_distances_normalization(edge_distances)

    # Gaussian weights (adapted RBF kernel).
    gaussian_weights = torch.exp(-(normalized_distance ** 2) / sigma)

    # Turn the Gaussian weights into a dissimilarity measure (larger distance
    # means larger dissimilarity).
    dissimilarity_ratio = 1 / (gaussian_weights + 1e-12)  # avoid division by zero

    # Sum the dissimilarity contributions per source node.
    sum_gaussian_weights = scatter_add(dissimilarity_ratio, src, dim=0, dim_size=num_nodes)

    # Dissimilarity score.
    dissimilarity_scores = sum_gaussian_weights / degree

    # Final GRaNDe score.
    grande_scores = degree + dissimilarity_scores

    return grande_scores

def grande_norm(
    edge_index,
    edge_weight=None,
    num_nodes=None,
    improved=False,
    add_self_loops=True,
    flow="source_to_target",
    dtype=None, x=None,
    sigma=0.2
):
    """GRaNDe-based normalization, a drop-in replacement for ``gcn_norm``.

    Instead of the standard symmetric degree normalization, edge weights are
    normalized using the GRaNDe score (see :func:`grande`). Supports dense
    ``edge_index`` tensors and ``SparseTensor``/torch-sparse adjacencies.

    Args:
        edge_index: Edge index tensor or sparse adjacency.
        edge_weight: Optional edge weights.
        num_nodes: Number of nodes (inferred when ``None``).
        improved: If True, uses a self-loop fill value of ``2.`` instead of ``1.``.
        add_self_loops: Whether to add self-loops before normalizing.
        flow: Message passing flow direction.
        dtype: Optional dtype for the edge weights.
        x: Node feature matrix, required to compute the GRaNDe score.
        sigma: Bandwidth of the Gaussian kernel used by GRaNDe.

    Returns:
        The normalized ``edge_index``/``edge_weight`` (or adjacency), matching
        the input representation.
    """
    fill_value = 2. if improved else 1.

    if isinstance(edge_index, SparseTensor):
        assert edge_index.size(0) == edge_index.size(1)

        adj_t = edge_index

        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1., dtype=dtype)
        if add_self_loops:
            adj_t = torch_sparse.fill_diag(adj_t, fill_value)

        edge_index2, edge_weight2 = to_edge_index(adj_t)
        edge_index_tensor = torch.tensor(edge_index2, dtype=torch.long, device=x.device)
        deg = grande(x, edge_index_tensor, sigma)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        adj_t = torch_sparse.mul(adj_t, deg_inv_sqrt.view(-1, 1))
        adj_t = torch_sparse.mul(adj_t, deg_inv_sqrt.view(1, -1))

        return adj_t

    if is_torch_sparse_tensor(edge_index):
        assert edge_index.size(0) == edge_index.size(1)

        if edge_index.layout == torch.sparse_csc:
            raise NotImplementedError("Sparse CSC matrices are not yet supported in 'gcn_norm'")

        adj_t = edge_index
        if add_self_loops:
            adj_t, _ = add_self_loops_fn(adj_t, None, fill_value, num_nodes)

        edge_index, value = to_edge_index(adj_t)
        col, row = edge_index[0], edge_index[1]

        deg = grande(x, edge_index, sigma)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        value = deg_inv_sqrt[row] * value * deg_inv_sqrt[col]

        return set_sparse_value(adj_t, value), None

    assert flow in ['source_to_target', 'target_to_source']
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    if add_self_loops:
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                 device=edge_index.device)

    row, col = edge_index[0], edge_index[1]
    idx = col if flow == 'source_to_target' else row
    deg = grande(x, edge_index, sigma)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    return edge_index, edge_weight

class GrandeAPPNP(MessagePassing):
    """APPNP propagation using GRaNDe-based edge normalization.

    Mirrors :class:`torch_geometric.nn.APPNP` but replaces ``gcn_norm`` with
    :func:`grande_norm`, so the personalized PageRank propagation is weighted by
    the GRaNDe score instead of the plain symmetric degree.

    Args:
        K: Number of propagation steps.
        alpha: Teleport (restart) probability.
        dropout: Dropout applied to the edge weights during propagation.
        cached: Whether to cache the normalized edges.
        add_self_loops: Whether to add self-loops before normalizing.
        normalize: Whether to apply GRaNDe normalization.
        sigma: Bandwidth of the Gaussian kernel used by GRaNDe.
    """

    _cached_edge_index: Optional[OptPairTensor]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, K: int, alpha: float, dropout: float = 0.,
                 cached: bool = False, add_self_loops: bool = True,
                 normalize: bool = True, sigma: float = 0.2, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.K = K
        self.alpha = alpha
        self.dropout = dropout
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self._cached_edge_index = None
        self._cached_adj_t = None
        self.sigma = sigma
                


    def reset_parameters(self):
        super().reset_parameters()
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        edge_weight: OptTensor = None,
    ) -> Tensor:

        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = grande_norm(  # GRaNDe replacement for gcn_norm
                        edge_index, edge_weight, x.size(self.node_dim), False,
                        self.add_self_loops, self.flow, dtype=x.dtype, x=x, sigma=self.sigma)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = grande_norm(  # GRaNDe replacement for gcn_norm
                        edge_index, edge_weight, x.size(self.node_dim), False,
                        self.add_self_loops, self.flow, dtype=x.dtype, x=x, sigma=self.sigma)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        h = x
        for _ in range(self.K):
            if self.dropout > 0 and self.training:
                if isinstance(edge_index, Tensor):
                    if is_torch_sparse_tensor(edge_index):
                        _, edge_weight = to_edge_index(edge_index)
                        edge_weight = F.dropout(edge_weight, p=self.dropout)
                        edge_index = set_sparse_value(edge_index, edge_weight)
                    else:
                        assert edge_weight is not None
                        edge_weight = F.dropout(edge_weight, p=self.dropout)
                else:
                    value = edge_index.storage.value()
                    assert value is not None
                    value = F.dropout(value, p=self.dropout)
                    edge_index = edge_index.set_value(value, layout='coo')

            # propagate_type: (x: Tensor, edge_weight: OptTensor)
            x = self.propagate(edge_index=edge_index, x=x, edge_weight=edge_weight)
            x = x * (1 - self.alpha)
            x = x + self.alpha * h

        return x

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: Adj, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(K={self.K}, alpha={self.alpha})'


class GrandeSGConv(MessagePassing):
    """Simplified Graph Convolution (SGC) with GRaNDe-based normalization.

    Mirrors :class:`torch_geometric.nn.SGConv` but replaces ``gcn_norm`` with
    :func:`grande_norm`.

    Args:
        in_channels: Size of each input sample.
        out_channels: Size of each output sample.
        K: Number of propagation hops.
        cached: Whether to cache the propagated features.
        add_self_loops: Whether to add self-loops before normalizing.
        bias: Whether the internal linear layer learns an additive bias.
        sigma: Bandwidth of the Gaussian kernel used by GRaNDe.
    """

    def __init__(self, in_channels: int,
                 out_channels: int, 
                 K: int = 1,
                 cached: bool = False, 
                 add_self_loops: bool = True,
                 bias: bool = True, 
                 sigma: float = 0.2,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.cached = cached
        self.add_self_loops = add_self_loops

        self._cached_x = None

        self.lin = Linear(in_channels, out_channels, bias=bias)

        self.reset_parameters()
        self.sigma = sigma

    def reset_parameters(self):
        super().reset_parameters()
        self.lin.reset_parameters()
        self._cached_x = None

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        cache = self._cached_x
        x = self.lin(x)
        if cache is None:
            if isinstance(edge_index, Tensor):
                edge_index, edge_weight = grande_norm(
                    edge_index,
                    edge_weight=edge_weight,
                    num_nodes=x.size(0),
                    x=x,
                    sigma=self.sigma)
                
            elif isinstance(edge_index, SparseTensor):
                edge_index, edge_weight = grande_norm(
                    edge_index,
                    edge_weight=edge_weight,
                    num_nodes=x.size(0),
                    x=x,
                    sigma=self.sigma)

            for k in range(self.K):
                # propagate_type: (x: Tensor, edge_weight: OptTensor)
                x = self.propagate(edge_index, x=x, edge_weight=edge_weight)
                if self.cached:
                    self._cached_x = x
        else:
            x = cache.detach()
        return x

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: Adj, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, K={self.K})')