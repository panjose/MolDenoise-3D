from typing import Optional, Tuple, List, Union

import re
import math
import warnings
import torch
from torch import nn
from torch.autograd import grad
from torch.nn.functional import mse_loss
from torch.nn.parameter import Parameter
from torch_scatter import scatter
from torch_geometric.nn import MessagePassing
from torch_cluster import radius_graph


class Distance(nn.Module):
    '''Calculate the distance and connection relationship between atoms in atomic coordinates,
    and filter out edges that meet the conditions based on the distance.
    Args:
        cutoff_lower (float): lower cutoff distance for the distance-dependent weighting function in angstroms
        cutoff_upper (float): upper cutoff distance for the distance-dependent weighting function in angstroms
        max_num_neighbors (int): maximum number of neighbors to consider for each atom (default: 32)
        return_vecs (bool): whether to return the vectors of the edges (default: False)
        loop (bool): whether to include self loops in the graph (default: False)'''
    def __init__(
        self,
        cutoff_lower:float,
        cutoff_upper:float,
        max_num_neighbors:int = 32,
        return_vecs:bool = False,
        loop:bool = False,
    ):
        super(Distance, self).__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.max_num_neighbors = max_num_neighbors
        self.return_vecs = return_vecs
        self.loop = loop

    def forward(self, pos, batch)-> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        '''Compute the edge indices, edge weights, and optionally edge vectors for a given set of atomic coordinates.
        Args:
            pos (torch.Tensor): atomic coordinates in the graph (num_nodes, 3)
            batch (torch.Tensor): batch indices of the graph (num_nodes)
        Returns:
            edge_index (torch.Tensor): edge indices of the graph in COO format (2, num_edges)
            edge_weight (torch.Tensor): edge weights of the graph in COO format 
                                        also the distances between atoms in angstroms (num_edges)
            edge_vec (torch.Tensor): edge vectors of the graph in COO format (num_edges, 3)'''
        edge_index = radius_graph(
            x=pos,
            r=self.cutoff_upper,
            batch=batch,
            loop=self.loop,
            max_num_neighbors=self.max_num_neighbors,
        )
        edge_vec = pos[edge_index[0]] - pos[edge_index[1]]

        if self.loop:
            # mask out self loops when computing distances because
            # the norm of 0 produces NaN gradients
            mask = edge_index[0] != edge_index[1]
            edge_weight = torch.zeros(edge_vec.size(0), device=edge_vec.device)
            edge_weight[mask] = torch.norm(edge_vec[mask], dim=-1)
        else:
            edge_weight = torch.norm(edge_vec, dim=-1)

        lower_mask = edge_weight >= self.cutoff_lower
        edge_index = edge_index[:, lower_mask]
        edge_weight = edge_weight[lower_mask]

        if self.return_vecs:
            edge_vec = edge_vec[lower_mask]
            return edge_index, edge_weight, edge_vec
        return edge_index, edge_weight, None


class CosineCutoff(nn.Module):
    '''Cosine cutoff function for distances.
    This function returns a value between 0 and 1 based on the distance between two atoms.
    Args:
        cutoff_lower (float): lower cutoff distance in angstroms (default: 0.0)
        cutoff_upper (float): upper cutoff distance in angstroms (default: 5.0)'''
    def __init__(self, cutoff_lower=0.0, cutoff_upper=5.0):
        super(CosineCutoff, self).__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            distances (torch.Tensor): distances between atoms in angstroms
        Returns:
            cutoffs (torch.Tensor): cutoff values between 0 and 1
        '''
        if self.cutoff_lower > 0:
            cutoffs = 0.5 * (torch.cos(math.pi * (2 * (distances - self.cutoff_lower) / (self.cutoff_upper - self.cutoff_lower) + 1.0) ) + 1.0)
            # remove contributions below the cutoff radius
            cutoffs = cutoffs * (distances < self.cutoff_upper).float()
            cutoffs = cutoffs * (distances > self.cutoff_lower).float()
            return cutoffs
        else:
            cutoffs = 0.5 * (torch.cos(distances * math.pi / self.cutoff_upper) + 1.0)
            # remove contributions beyond the cutoff radius
            cutoffs = cutoffs * (distances < self.cutoff_upper).float()
            return cutoffs


class ExpNormalSmearing(nn.Module):
    '''Convert the distances between atoms into a set of high-dimensional feature vectors to encode the features of the edges.
    Radial basis function (RBF) for distance-dependent weighting.
    The RBF is a Gaussian function with a cutoff function to ensure that the function is 0 outside of the cutoff range.
    The RBF is used to weight the atomic embeddings of neighboring atoms based on the distance between them.
    The RBF is defined as:
        f(r) = exp(-beta * (exp(alpha * (r - cutoff_lower)) - mean) ** 2)
    Args:
        cutoff_lower (float): lower cutoff distance in angstroms (default: 0.0)
        cutoff_upper (float): upper cutoff distance in angstroms (default: 5.0)
        num_rbf (int): number of radial basis functions (default: 50)
        trainable (bool): whether the parameters of the RBF are trainable (default: True) '''
    def __init__(self, cutoff_lower=0.0, cutoff_upper=5.0, num_rbf=50, trainable=True):
        super(ExpNormalSmearing, self).__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.num_rbf = num_rbf
        self.trainable = trainable

        self.cutoff_fn = CosineCutoff(0, cutoff_upper)
        self.alpha = 5.0 / (cutoff_upper - cutoff_lower)

        means, betas = self._initial_params()
        if trainable:
            self.register_parameter("means", nn.Parameter(means))
            self.register_parameter("betas", nn.Parameter(betas))
        else:
            self.register_buffer("means", means)
            self.register_buffer("betas", betas)

    def _initial_params(self) -> Tuple[torch.Tensor, torch.Tensor]:
        '''Initialize means and betas for the RBF.
        The means are initialized linearly between exp(-cutoff_upper + cutoff_lower) and 1.
        The betas are initialized to (2 / num_rbf * (1 - exp(-cutoff_upper + cutoff_lower))) ** -2.
        Returns:
            means (torch.Tensor): means of the RBF (num_rbf)
            betas (torch.Tensor): betas of the RBF (num_rbf)
        '''
        # initialize means and betas according to the default values in PhysNet
        # https://pubs.acs.org/doi/10.1021/acs.jctc.9b00181
        start_value = torch.exp(torch.tensor(-self.cutoff_upper + self.cutoff_lower))
        means = torch.linspace(start_value, 1, self.num_rbf)
        betas = torch.tensor(
            [(2 / self.num_rbf * (1 - start_value)) ** -2] * self.num_rbf
        )
        return means, betas

    def reset_parameters(self) -> None:
        '''Reset the parameters of the RBF.'''
        means, betas = self._initial_params()
        self.means.data.copy_(means)
        self.betas.data.copy_(betas)

    def forward(self, dist) -> torch.Tensor:
        '''Compute the RBF for a given distance.
        Args:
            dist (torch.Tensor): distance between atoms in angstroms (num_edges)
        Returns:
            rbf (torch.Tensor): RBF values (num_edges, num_rbf)'''
        dist = dist.unsqueeze(-1) # [num_edges] -> [num_edges, 1]
        return self.cutoff_fn(dist) * torch.exp(
            -self.betas
            * (torch.exp(self.alpha * (-dist + self.cutoff_lower)) - self.means) ** 2
        ) # [num_edges, 1] * [num_edges, num_rbf] -> [num_edges, num_rbf]


class NeighborEmbedding(MessagePassing):
    '''Through the message passing mechanism, the information of neighbor nodes is embedded into the representation of the central node.
    This module takes in atomic coordinates and atomic numbers, and outputs a tensor of atomic embeddings.
    The embeddings are computed by aggregating the atomic embeddings of neighboring atoms, using a distance-dependent weighting function.
    The distance-dependent weighting function is a cosine cutoff function, which smoothly transitions from 1 to 0 as the distance between atoms increases.
    Args:
        hidden_channels (int): number of hidden channels for the embedding layer
        num_rbf (int): number of radial basis functions for the distance-dependent weighting function
                        also used to determine the size of the edge attributes
        cutoff_lower (float): lower cutoff distance for the distance-dependent weighting function in angstroms
        cutoff_upper (float): upper cutoff distance for the distance-dependent weighting function in angstroms
        max_z (int): maximum atomic number (default: 100) '''
    def __init__(self, hidden_channels, num_rbf, cutoff_lower, cutoff_upper, max_z=100):
        super(NeighborEmbedding, self).__init__(aggr="add")
        self.embedding = nn.Embedding(max_z, hidden_channels)
        self.distance_proj = nn.Linear(num_rbf, hidden_channels)
        self.combine = nn.Linear(hidden_channels * 2, hidden_channels)
        self.cutoff = CosineCutoff(cutoff_lower, cutoff_upper)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        '''Reset the parameters of the embedding and linear layers.'''
        self.embedding.reset_parameters()
        nn.init.xavier_uniform_(self.distance_proj.weight)
        nn.init.xavier_uniform_(self.combine.weight)
        self.distance_proj.bias.data.fill_(0)
        self.combine.bias.data.fill_(0)

    def forward(self, z, x, edge_index, edge_weight, edge_attr) -> torch.Tensor:
        '''Compute the neighbor embeddings for a given set of atomic coordinates and atomic numbers.
        Args:
            z (torch.Tensor): atomic numbers of the atoms in the graph (num_nodes)
            x (torch.Tensor): the attribute of the nodes (num_nodes, hidden_channels) 
            edge_index (torch.Tensor): edge indices of the graph in COO format (2, num_edges)
            edge_weight (torch.Tensor): edge weights of the graph in COO format
                                        also the distances between atoms in angstroms (num_edges)
            edge_attr (torch.Tensor): edge attributes of the graph in COO format (num_edges, num_rbf)
        Returns:
            x_neighbors (torch.Tensor): neighbor embeddings of shape (num_nodes, hidden_channels)'''
        # remove self loops
        mask = edge_index[0] != edge_index[1]
        if not mask.all():
            edge_index = edge_index[:, mask]
            edge_weight = edge_weight[mask]
            edge_attr = edge_attr[mask]

        C = self.cutoff(edge_weight) # [num_edges]
        W = self.distance_proj(edge_attr) * C.view(-1, 1)
        # [num_edges, hidden_channels] * [num_edges, 1] = [num_edges, hidden_channels]
        # broadcast mutiply : (C to match the size of W) [num_edges, 1] -> [num_edges, hidden_channels]

        x_neighbors = self.embedding(z) # [num_nodes, hidden_channels]
        # propagate_type: (x: Tensor, W: Tensor)
        x_neighbors = self.propagate(edge_index, x=x_neighbors, W=W, size=None)
        # propagate = message + aggregate + update -> [num_nodes, hidden_channels]
        x_neighbors = self.combine(torch.cat([x, x_neighbors], dim=1))
        # torch.cat : [num_nodes, hidden_channels * 2]
        # combine -> [num_nodes, hidden_channels]
        return x_neighbors

    def message(self, x_j, W) -> torch.Tensor:
        '''Message function for the message passing layer.
        Args:
            x_j (torch.Tensor): atomic embeddings of neighboring atoms
                                also the set of all source node features (num_edges, hidden_channels)
            W (torch.Tensor): distance-dependent weighting function (num_edges, hidden_channels)
        Returns:
            x_j * W (torch.Tensor): weighted atomic embeddings of neighboring atoms (num_edges, hidden_channels)'''
        return x_j * W


class GatedEquivariantBlock(nn.Module):
    '''Gated Equivariant Block is a special neural network layer to deal with the data containing both scalar and vector attributes
    such as the nodes containing both atomic number and atomic coordinates in a molecule graph.
    It is as defined in Schütt et al. (2021):
    Equivariant message passing for the prediction of tensorial properties and molecular spectra.
    Args:
        hidden_channels (int): number of hidden channels in the layer
        out_channels (int): number of output channels in the layer
        intermediate_channels (int): number of intermediate channels in the update network (default: None)
        scalar_activation (bool): whether to apply the activation function to the scalar output (default: False)'''

    def __init__(
        self,
        hidden_channels:int,
        out_channels:int,
        intermediate_channels:int = None,
        scalar_activation:bool = False,
    ):
        super(GatedEquivariantBlock, self).__init__()
        self.out_channels = out_channels

        if intermediate_channels is None:
            intermediate_channels = hidden_channels

        self.vec1_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.vec2_proj = nn.Linear(hidden_channels, out_channels, bias=False)

        self.update_net = nn.Sequential(
            nn.Linear(hidden_channels * 2, intermediate_channels),
            nn.SiLU(),
            nn.Linear(intermediate_channels, out_channels * 2),
        )

        self.act = nn.SiLU() if scalar_activation else None

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.vec1_proj.weight)
        nn.init.xavier_uniform_(self.vec2_proj.weight)
        nn.init.xavier_uniform_(self.update_net[0].weight)
        self.update_net[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.update_net[2].weight)
        self.update_net[2].bias.data.fill_(0)

    def forward(self, x, v) -> tuple[torch.Tensor, torch.Tensor]:
        '''Forward pass of the Gated Equivariant Block.
        Args:
            x (torch.Tensor): input tensor of shape (num_nodes, hidden_channels)
            v (torch.Tensor): input tensor of shape (num_nodes, 3, hidden_channels)
        Returns:
            x (torch.Tensor): output tensor of shape (num_nodes, out_channels)
            v (torch.Tensor): output tensor of shape (num_nodes, 3, out_channels)
        '''
        vec1 = torch.norm(self.vec1_proj(v), dim=-2) # [num_nodes, hidden_channels]
        vec2 = self.vec2_proj(v) # [num_nodes, 3, hidden_channels]

        x = torch.cat([x, vec1], dim=-1) # [num_nodes, hidden_channels * 2]
        x, v = torch.split(self.update_net(x), self.out_channels, dim=-1)
        # self.update_net(x) -> [num_nodes, out_channels * 2]
        # torch.split : [num_nodes, out_channels] + [num_nodes, out_channels]
        v = v.unsqueeze(1) * vec2
        # [num_nodes, 1, out_channels] * [num_nodes, 3, out_channels]
        # -> [num_nodes, 3, out_channels]

        if self.act is not None:
            x = self.act(x)
        return x, v


class EquivariantScalar(nn.Module):
    '''Process the input scalar and vector features and finally reduce them to a scalar output.
    It is used in the output layer of graph neural networks, to predict molecular energy, band gap, 
    or other properties represented by only one value.
    Args:
        hidden_channels (int): number of hidden channels in the layer
        allow_prior_model (bool): whether to allow the prior model (default: True)
    '''
    def __init__(self, hidden_channels, allow_prior_model=True):
        # super(EquivariantScalar, self).__init__(allow_prior_model=allow_prior_model)
        super(EquivariantScalar, self).__init__()
        self.output_network = nn.ModuleList(
            [
                GatedEquivariantBlock(
                    hidden_channels=hidden_channels,
                    out_channels=hidden_channels // 2,
                    scalar_activation=True,
                ),
                GatedEquivariantBlock(
                    hidden_channels=hidden_channels // 2, 
                    out_channels=1
                ),
            ]
        )
        self.allow_prior_model = allow_prior_model

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for layer in self.output_network:
            layer.reset_parameters()

    def pre_reduce(self, x, v, z, pos, batch) -> torch.Tensor:
        '''process the input scalar and vector features
        Args:
            x (torch.Tensor): input tensor of shape (num_nodes, hidden_channels)
            v (torch.Tensor): input tensor of shape (num_nodes, 3, hidden_channels)
        Returns:
            torch.Tensor: output tensor of shape (num_nodes, 1)'''
        for layer in self.output_network:
            x, v = layer(x, v)
        # include v in output to make sure all parameters have a gradient
        return x + v.sum() * 0
    
    def post_reduce(self, x) -> torch.Tensor:
        return x


class EquivariantVectorOutput(EquivariantScalar):
    '''Process the input scalar and vector features and reduce them to a vector output instead of a scalar.
    This module is used to predict vector properties like forces and dipole moments.'''
    def __init__(self, hidden_channels):
        super(EquivariantVectorOutput, self).__init__(
            hidden_channels, allow_prior_model=False
        )

    def pre_reduce(self, x, v, z, pos, batch) -> torch.Tensor:
        '''process the input scalar and vector features
        Args:
            x (torch.Tensor): input tensor of shape (num_nodes, hidden_channels)
            v (torch.Tensor): input tensor of shape (num_nodes, 3, hidden_channels)
        Returns:
            torch.Tensor: output tensor of shape (num_nodes, 3)'''
        for layer in self.output_network:
            x, v = layer(x, v)
        return v.squeeze()


class EquivariantMultiHeadAttention(MessagePassing):
    '''Handles both scalar and vector features, ensuring equivariance under spatial rotation and reflection.
    This module combines the attention mechanism in Transformer with the equivariance constraints of the physical system, 
    enabling the model to more accurately predict the properties of 3D objects.
    Args:
        hidden_channels (int): number of hidden channels in the layer
        num_rbf (int): number of radial basis functions
        distance_influence (str): how distance influences the attention mechanism
        num_heads (int): number of attention heads
        cutoff_lower (float): lower cutoff distance for the attention mechanism
        cutoff_upper (float): upper cutoff distance for the attention mechanism
    '''
    def __init__(
        self,
        hidden_channels,
        num_rbf,
        distance_influence,
        num_heads,
        cutoff_lower,
        cutoff_upper,
    ):
        super(EquivariantMultiHeadAttention, self).__init__(aggr="add", node_dim=0)
        assert hidden_channels % num_heads == 0, (
            f"The number of hidden channels ({hidden_channels}) "
            f"must be evenly divisible by the number of "
            f"attention heads ({num_heads})"
        )

        self.distance_influence = distance_influence
        self.num_heads = num_heads
        self.hidden_channels = hidden_channels
        self.head_dim = hidden_channels // num_heads

        self.layernorm = nn.LayerNorm(hidden_channels)
        self.act = nn.SiLU()
        self.attn_activation = nn.SiLU()
        self.cutoff = CosineCutoff(cutoff_lower, cutoff_upper)

        self.q_proj = nn.Linear(hidden_channels, hidden_channels)
        self.k_proj = nn.Linear(hidden_channels, hidden_channels)
        self.v_proj = nn.Linear(hidden_channels, hidden_channels * 3)
        self.o_proj = nn.Linear(hidden_channels, hidden_channels * 3)

        self.vec_proj = nn.Linear(hidden_channels, hidden_channels * 3, bias=False)

        self.dk_proj = None
        if distance_influence in ["keys", "both"]:
            self.dk_proj = nn.Linear(num_rbf, hidden_channels)

        self.dv_proj = None
        if distance_influence in ["values", "both"]:
            self.dv_proj = nn.Linear(num_rbf, hidden_channels * 3)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.layernorm.reset_parameters()
        nn.init.xavier_uniform_(self.q_proj.weight)
        self.q_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.k_proj.weight)
        self.k_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.v_proj.weight)
        self.v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.vec_proj.weight)
        if self.dk_proj:
            nn.init.xavier_uniform_(self.dk_proj.weight)
            self.dk_proj.bias.data.fill_(0)
        if self.dv_proj:
            nn.init.xavier_uniform_(self.dv_proj.weight)
            self.dv_proj.bias.data.fill_(0)

    def forward(self, x, vec, edge_index, r_ij, f_ij, d_ij) -> Tuple[torch.Tensor, torch.Tensor]:
        '''Performs the forward pass of the equivariant multi-head attention block.
        Args:
            x (torch.Tensor): Scalar node features. Shape: (num_nodes, hidden_channels)
            vec (torch.Tensor): Vector node features. Shape: (num_nodes, 3, hidden_channels)
            edge_index (torch.Tensor): Graph edges in COO format. Shape: (2, num_edges)
            r_ij (torch.Tensor): Interatomic distances. Shape: (num_edges,)
            f_ij (torch.Tensor): Radial basis functions (RBF) features for edges. Shape: (num_edges, num_rbf)
            d_ij (torch.Tensor): Normalized edge vectors. Shape: (num_edges, 3)
        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing updated scalar and vector features.
            The shapes are (num_nodes, hidden_channels) and (num_nodes, 3, hidden_channels) respectively.'''
        x = self.layernorm(x)
        q = self.q_proj(x).reshape(-1, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(-1, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(-1, self.num_heads, self.head_dim * 3)

        vec1, vec2, vec3 = torch.split(self.vec_proj(vec), self.hidden_channels, dim=-1)
        vec = vec.reshape(-1, 3, self.num_heads, self.head_dim)
        vec_dot = (vec1 * vec2).sum(dim=1)

        dk = (
            self.act(self.dk_proj(f_ij)).reshape(-1, self.num_heads, self.head_dim)
            if self.dk_proj is not None
            else None
        )
        dv = (
            self.act(self.dv_proj(f_ij)).reshape(-1, self.num_heads, self.head_dim * 3)
            if self.dv_proj is not None
            else None
        )

        # propagate_type: (q: Tensor, k: Tensor, v: Tensor, vec: Tensor, dk: Tensor, dv: Tensor, r_ij: Tensor, d_ij: Tensor)
        x, vec = self.propagate(
            edge_index,
            q=q,
            k=k,
            v=v,
            vec=vec,
            dk=dk,
            dv=dv,
            r_ij=r_ij,
            d_ij=d_ij,
            size=None,
        )
        x = x.reshape(-1, self.hidden_channels)
        vec = vec.reshape(-1, 3, self.hidden_channels)

        o1, o2, o3 = torch.split(self.o_proj(x), self.hidden_channels, dim=1)
        dx = vec_dot * o2 + o3
        dvec = vec3 * o1.unsqueeze(1) + vec
        return dx, dvec

    def message(self, q_i, k_j, v_j, vec_j, dk, dv, r_ij, d_ij) -> Tuple[torch.Tensor, torch.Tensor]:
        '''Computes the message from source nodes to target nodes.
        Args:
            q_i (torch.Tensor): Query tensor of target nodes. Shape: (num_edges, num_heads, head_dim)
            k_j (torch.Tensor): Key tensor of source nodes. Shape: (num_edges, num_heads, head_dim)
            v_j (torch.Tensor): Value tensor of source nodes. Shape: (num_edges, num_heads, head_dim * 3)
            vec_j (torch.Tensor): Vector features of source nodes. Shape: (num_edges, 3, num_heads, head_dim)
            dk (torch.Tensor): Distance-influenced keys. Shape: (num_edges, num_heads, head_dim) or None
            dv (torch.Tensor): Distance-influenced values. Shape: (num_edges, num_heads, head_dim * 3) or None
            r_ij (torch.Tensor): Interatomic distances. Shape: (num_edges,)
            d_ij (torch.Tensor): Normalized edge vectors. Shape: (num_edges, 3)
        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing the scalar and vector messages to be aggregated.
            The shapes are (num_edges, num_heads, head_dim) and (num_edges, 3, num_heads, head_dim).'''
        # attention mechanism
        if dk is None:
            attn = (q_i * k_j).sum(dim=-1)
        else:
            attn = (q_i * k_j * dk).sum(dim=-1)

        # attention activation function
        attn = self.attn_activation(attn) * self.cutoff(r_ij).unsqueeze(1)

        # value pathway
        if dv is not None:
            v_j = v_j * dv
        x, vec1, vec2 = torch.split(v_j, self.head_dim, dim=2)

        # update scalar features
        x = x * attn.unsqueeze(2)
        # update vector features
        vec = vec_j * vec1.unsqueeze(1) + vec2.unsqueeze(1) * d_ij.unsqueeze(
            2
        ).unsqueeze(3)
        return x, vec

    def aggregate(
        self,
        features: Tuple[torch.Tensor, torch.Tensor],
        index: torch.Tensor,
        ptr: Optional[torch.Tensor],
        dim_size: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        '''Aggregates messages from neighboring nodes.
        Args:
            features (tuple[torch.Tensor, torch.Tensor]): A tuple of scalar and vector messages from the message function.
                Shapes are (num_edges, num_heads, head_dim) and (num_edges, 3, num_heads, head_dim).
            index (torch.Tensor): The target node indices for each message. Shape: (num_edges,)
            ptr (torch.Tensor): Pointer to the start of each graph in a batch. Used for scatter operations.
                Shape: (num_graphs + 1,) or None.
            dim_size (int): The number of nodes in the graph. Shape: int or None.
        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing the aggregated scalar and vector features for each node.
            The shapes are (num_nodes, num_heads, head_dim) and (num_nodes, 3, num_heads, head_dim).'''
        x, vec = features
        x = scatter(x, index, dim=self.node_dim, dim_size=dim_size)
        vec = scatter(vec, index, dim=self.node_dim, dim_size=dim_size)
        return x, vec

    def update(
        self, inputs: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        '''Updates the node features after aggregation.
        This method simply passes through the aggregated features as the final update step is performed in the forward pass.'''
        return inputs


class EquivariantLayerNorm(nn.Module):
    '''Rotationally-equivariant Vector Layer Normalization.
    The module implements rotation-equivariant layer normalization specifically for processing vector features. 
    It normalizes vectors by adjusting their mean and covariance while maintaining their equivariance under spatial rotation.
    Args:
        normalized_shape (int): The shape of the input tensor.
        eps (float, optional): A value added to the denominator for numerical stability. Default: 1e-5.
        elementwise_linear (bool, optional): If True, applies elementwise linear transformation to the normalized vectors.
            Default: True.
        device (torch.device, optional): The device on which the module is allocated. Default: None.
        dtype (torch.dtype, optional): The data type of the module. Default: None.''' 
       
    __constants__ = ["normalized_shape", "elementwise_linear"]
    normalized_shape: Tuple[int, ...]
    eps: float
    elementwise_linear: bool

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-5,
        elementwise_linear: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(EquivariantLayerNorm, self).__init__()

        self.normalized_shape = (int(normalized_shape),)
        self.eps = eps
        self.elementwise_linear = elementwise_linear
        if self.elementwise_linear:
            self.weight = Parameter(
                torch.empty(self.normalized_shape, **factory_kwargs)
            )
        else:
            self.register_parameter("weight", None) # Without bias term to preserve equivariance!

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_linear:
            nn.init.ones_(self.weight)

    def mean_center(self, input:torch.Tensor) -> torch.Tensor:
        '''Mean center the input tensor along the specified dimension.
        Args:
            input (torch.Tensor): The input tensor to be mean centered.
        Returns:
            torch.Tensor: The mean centered tensor.'''
        return input - input.mean(-1, keepdim=True)

    def covariance(self, input:torch.Tensor) -> torch.Tensor:
        '''Compute the covariance matrix of the input tensor.
        Args:
            input (torch.Tensor): The input tensor.
        Returns:
            torch.Tensor: The covariance matrix.'''
        return 1 / self.normalized_shape[0] * input @ input.transpose(-1, -2)

    def symsqrtinv(self, matrix:torch.Tensor) -> torch.Tensor:
        '''Compute the inverse square root of a positive definite matrix.
        Based on https://github.com/pytorch/pytorch/issues/25481
        Args:
            matrix (torch.Tensor): The input matrix.
        Returns:
            torch.Tensor: The inverse square root of the input matrix.'''
        _, s, v = matrix.svd()
        good = (
            s > s.max(-1, True).values * s.size(-1) * torch.finfo(s.dtype).eps
        )
        components = good.sum(-1)
        common = components.max()
        unbalanced = common != components.min()
        if common < s.size(-1):
            s = s[..., :common]
            v = v[..., :common]
            if unbalanced:
                good = good[..., :common]
        if unbalanced:
            s = s.where(good, torch.zeros((), device=s.device, dtype=s.dtype))
        return (v * 1 / torch.sqrt(s + self.eps).unsqueeze(-2)) @ v.transpose(
            -2, -1
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        '''Forward pass of the EquivariantLayerNorm module.
        Args:
            input (torch.Tensor): The input tensor.
        Returns:
            torch.Tensor: The normalized tensor.'''
        input = input.to(torch.float64) # Need double precision for accurate inversion.
        input = self.mean_center(input)
        # We use different diagonal elements in case input matrix is approximately zero,
        # in which case all singular values are equal which is problematic for backprop.
        # See e.g. https://pytorch.org/docs/stable/generated/torch.svd.html
        reg_matrix = (
            torch.diag(torch.tensor([1.0, 2.0, 3.0]))
            .unsqueeze(0)
            .to(input.device)
            .type(input.dtype)
        )
        covar = self.covariance(input) + self.eps * reg_matrix
        covar_sqrtinv = self.symsqrtinv(covar)
        return (covar_sqrtinv @ input).to(
            self.weight.dtype
        ) * self.weight.reshape(1, 1, self.normalized_shape[0])

    def extra_repr(self) -> str:
        '''Return a string representation of the EquivariantLayerNorm module.'''
        return (
            "{normalized_shape}, "
            "elementwise_linear={elementwise_linear}".format(**self.__dict__)
        )
    
    
class AccumulatedNormalization(nn.Module):
    '''Running normalization of a tensor.
    Its core functionality is to implement runtime normalization, 
    which accumulates statistics (mean and standard deviation) over multiple batches of data 
    and then uses these statistics to normalize the incoming tensors.
    Args:
        accumulator_shape (Tuple[int, ...]): The shape of the accumulator tensor.
        epsilon (float, optional): A small value to prevent division by zero. (default: 1e-8)'''
    def __init__(self, accumulator_shape: Tuple[int, ...], epsilon: float = 1e-8):
        super(AccumulatedNormalization, self).__init__()

        self._epsilon = epsilon
        self.register_buffer("acc_sum", torch.zeros(accumulator_shape))
        self.register_buffer("acc_squared_sum", torch.zeros(accumulator_shape))
        self.register_buffer("acc_count", torch.zeros((1,)))
        self.register_buffer("num_accumulations", torch.zeros((1,)))

    def update_statistics(self, batch: torch.Tensor):
        batch_size = batch.shape[0]
        self.acc_sum += batch.sum(dim=0)
        self.acc_squared_sum += batch.pow(2).sum(dim=0)
        self.acc_count += batch_size
        self.num_accumulations += 1

    @property
    def acc_count_safe(self):
        return self.acc_count.clamp(min=1)

    @property
    def mean(self):
        return self.acc_sum / self.acc_count_safe

    @property
    def std(self):
        return torch.sqrt(
            (self.acc_squared_sum / self.acc_count_safe) - self.mean.pow(2)
        ).clamp(min=self._epsilon)

    def forward(self, batch: torch.Tensor):
        if self.training:
            self.update_statistics(batch)
        return ((batch - self.mean) / self.std)


class TorchMD_ET(nn.Module):
    '''The TorchMD Equivariant Transformer architecture.
    This class implements an equivariant Transformer, a type of graph neural network
    that simultaneously processes scalar and vector features while maintaining
    rotational and translational equivariance. It is particularly suited for
    tasks in physics and chemistry, such as molecular dynamics simulations or
    the prediction of molecular properties.
    Args:
        hidden_channels (int, optional): The number of hidden features for each node.
            (default: 128)
        num_layers (int, optional): The number of stacked equivariant attention layers.
            (default: 6)
        num_rbf (int, optional): The number of radial basis functions used to
            encode interatomic distances. (default: 50)
        trainable_rbf (bool, optional): If True, the RBF parameters will be
            trainable. (default: True)
        neighbor_embedding (bool, optional): If True, an initial neighbor embedding
            step will be performed to embed neighbors into node features.
            (default: True)
        num_heads (int, optional): The number of attention heads.
            (default: 8)
        distance_influence (string, optional): Specifies where distance information
            is used within the attention mechanism. Options are "keys", "values",
            "both", or "none". (default: "both")
        cutoff_lower (float, optional): The lower bound for considering interatomic
            interactions. (default: 0.0)
        cutoff_upper (float, optional): The upper bound for considering interatomic
            interactions. (default: 5.0)
        max_z (int, optional): The maximum possible atomic number, used for
            initializing the embedding layer. (default: 100)
        max_num_neighbors (int, optional): The maximum number of neighbors to consider
            per node. This parameter is passed to the graph construction routine.
            (default: 32)
        layernorm_on_vec (str, optional): The type of Layer Normalization to apply to
            the vector features. If set to "whitened", it uses an EquivariantLayerNorm.
            (default: None)'''

    def __init__(
        self,
        hidden_channels=128,
        num_layers=6,
        num_rbf=50,
        trainable_rbf=True,
        neighbor_embedding=True,
        num_heads=8,
        distance_influence="both",
        cutoff_lower=0.0,
        cutoff_upper=5.0,
        max_z=100,
        max_num_neighbors=32,
        layernorm_on_vec=None,
    ):
        super(TorchMD_ET, self).__init__()

        assert distance_influence in ["keys", "values", "both", "none"]

        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.num_rbf = num_rbf
        self.trainable_rbf = trainable_rbf
        self.neighbor_embedding = neighbor_embedding
        self.num_heads = num_heads
        self.distance_influence = distance_influence
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.max_z = max_z
        self.layernorm_on_vec = layernorm_on_vec

        self.embedding = nn.Embedding(self.max_z, hidden_channels)

        self.distance = Distance(
            cutoff_lower,
            cutoff_upper,
            max_num_neighbors=max_num_neighbors,
            return_vecs=True,
            loop=True,
        )
        self.distance_expansion = ExpNormalSmearing(
            cutoff_lower, cutoff_upper, num_rbf, trainable_rbf
        )
        self.neighbor_embedding = (
            NeighborEmbedding(
                hidden_channels, num_rbf, cutoff_lower, cutoff_upper, self.max_z
            ).jittable()
            if neighbor_embedding
            else None
        )

        self.attention_layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = EquivariantMultiHeadAttention(
                hidden_channels,
                num_rbf,
                distance_influence,
                num_heads,
                cutoff_lower,
                cutoff_upper,
            ).jittable()
            self.attention_layers.append(layer)

        self.out_norm = nn.LayerNorm(hidden_channels)
        if self.layernorm_on_vec:
            if self.layernorm_on_vec == "whitened":
                self.out_norm_vec = EquivariantLayerNorm(hidden_channels)
            else:
                raise ValueError(f"{self.layernorm_on_vec} not recognized.")
            
        self.reset_parameters()

    def reset_parameters(self):
        self.embedding.reset_parameters()
        self.distance_expansion.reset_parameters()
        if self.neighbor_embedding is not None:
            self.neighbor_embedding.reset_parameters()
        for attn in self.attention_layers:
            attn.reset_parameters()
        self.out_norm.reset_parameters()
        if self.layernorm_on_vec:
            self.out_norm_vec.reset_parameters()

    def forward(self, z, pos, batch):
        x = self.embedding(z)

        edge_index, edge_weight, edge_vec = self.distance(pos, batch)
        assert (
            edge_vec is not None
        ), "Distance module did not return directional information"

        edge_attr = self.distance_expansion(edge_weight)
        mask = edge_index[0] != edge_index[1]
        edge_vec[mask] = edge_vec[mask] / torch.norm(edge_vec[mask], dim=1).unsqueeze(1)

        if self.neighbor_embedding is not None:
            x = self.neighbor_embedding(z, x, edge_index, edge_weight, edge_attr)

        vec = torch.zeros(x.size(0), 3, x.size(1), device=x.device)

        for attn in self.attention_layers:
            dx, dvec = attn(x, vec, edge_index, edge_weight, edge_attr, edge_vec)
            x = x + dx
            vec = vec + dvec
        x = self.out_norm(x)
        if self.layernorm_on_vec:
            vec = self.out_norm_vec(vec)

        return x, vec, z, pos, batch

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"hidden_channels={self.hidden_channels}, "
            f"num_layers={self.num_layers}, "
            f"num_rbf={self.num_rbf}, "
            f"trainable_rbf={self.trainable_rbf}, "
            f"neighbor_embedding={self.neighbor_embedding}, "
            f"num_heads={self.num_heads}, "
            f"distance_influence={self.distance_influence}, "
            f"cutoff_lower={self.cutoff_lower}, "
            f"cutoff_upper={self.cutoff_upper})"
        )


class TorchMD_Net(nn.Module):
    '''The main TorchMD-Net model.
    This class serves as a wrapper for the entire model architecture,
    orchestrating the flow of data through the representation model,
    the output model, and optional prior models. It handles data-level
    operations like normalization, noise injection, and force derivative calculation.
    Args:
        representation_model (nn.Module): The core GNN model (e.g., TorchMD_ET)
            that learns to embed scalar and vector features.
        output_model (nn.Module): The output head that transforms the
            learned features into the final prediction (e.g., EquivariantScalar
            or EquivariantVectorOutput).
        prior_model (nn.Module, optional): An optional prior model to add
            physical priors to the prediction. (default: None)
        reduce_op (str, optional): The aggregation method to reduce
            node-level predictions to a single graph-level output.
            (default: "add")
        mean (torch.Tensor, optional): The mean of the training data for
            denormalization. (default: None)
        std (torch.Tensor, optional): The standard deviation of the
            training data for denormalization. (default: None)
        derivative (bool, optional): Whether to compute the derivative of the
            output with respect to atomic positions. (default: False)
        output_model_noise (nn.Module, optional): An optional output head for
            predicting noise, used in generative models. (default: None)
        position_noise_scale (float, optional): The scale of Gaussian noise
            to add to the atomic positions for data augmentation. (default: 0.)'''
    def __init__(
        self,
        representation_model,
        output_model,
        prior_model=None,
        reduce_op="add",
        mean=None,
        std=None,
        derivative=False,
        output_model_noise=None,
        position_noise_scale=0.,
    ):
        super(TorchMD_Net, self).__init__()
        self.representation_model = representation_model
        self.output_model = output_model

        self.prior_model = prior_model
        if not output_model.allow_prior_model and prior_model is not None:
            self.prior_model = None
            warnings.warn(
                (
                    "Prior model was given but the output model does "
                    "not allow prior models. Dropping the prior model."
                )
            )

        self.reduce_op = reduce_op
        self.derivative = derivative
        self.output_model_noise = output_model_noise        
        self.position_noise_scale = position_noise_scale

        mean = torch.scalar_tensor(0) if mean is None else mean
        self.register_buffer("mean", mean)
        std = torch.scalar_tensor(1) if std is None else std
        self.register_buffer("std", std)

        if self.position_noise_scale > 0:
            self.pos_normalizer = AccumulatedNormalization(accumulator_shape=(3,))

        self.reset_parameters()

    def reset_parameters(self):
        self.representation_model.reset_parameters()
        self.output_model.reset_parameters()
        if self.prior_model is not None:
            self.prior_model.reset_parameters()

    def forward(self, z, pos, batch: Optional[torch.Tensor] = None):
        assert z.dim() == 1 and z.dtype == torch.long
        batch = torch.zeros_like(z) if batch is None else batch

        if self.derivative:
            pos.requires_grad_(True)

        # run the potentially wrapped representation model
        x, v, z, pos, batch = self.representation_model(z, pos, batch=batch)

        # predict noise
        noise_pred = None
        if self.output_model_noise is not None:
            noise_pred = self.output_model_noise.pre_reduce(x, v, z, pos, batch) 

        # apply the output network
        x = self.output_model.pre_reduce(x, v, z, pos, batch)

        # scale by data standard deviation
        if self.std is not None:
            x = x * self.std

        # apply prior model
        if self.prior_model is not None:
            x = self.prior_model(x, z, pos, batch)

        # aggregate atoms
        out = scatter(x, batch, dim=0, reduce=self.reduce_op)

        # shift by data mean
        if self.mean is not None:
            out = out + self.mean

        # apply output model after reduction
        out = self.output_model.post_reduce(out)

        # compute gradients with respect to coordinates
        if self.derivative:
            grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(out)]
            dy = grad(
                [out],
                [pos],
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True,
            )[0]
            if dy is None:
                raise RuntimeError("Autograd returned None for the force prediction.")
            return out, noise_pred, -dy
        return out, noise_pred, None


def create_model(data_args, model_args, prior_model=None, mean=None, std=None) -> TorchMD_Net:
    '''Creates a model instance based on a dictionary of hyperparameters.
    This function acts as a factory, dynamically constructing the entire model
    based on the provided hyperparameters. It typically selects the correct
    model architecture (e.g., TorchMD_Net) and initializes it with the
    corresponding arguments.
    Args:
        data_args (DataArguments): Arguments related to the data.
        model_args (ModelArguments): Arguments related to the model.
    Returns:
        TorchMD_Net: A complete PyTorch model instance.'''
    # representation network
    if model_args.model == "equivariant-transformer":
        representation_model = TorchMD_ET(
            hidden_channels=model_args.embedding_dimension,
            num_layers=model_args.num_layers,
            num_rbf=model_args.num_rbf,
            trainable_rbf=False,
            neighbor_embedding=True,
            num_heads=model_args.num_heads,
            distance_influence="both",
            cutoff_lower=model_args.cutoff_lower,
            cutoff_upper=model_args.cutoff_upper,
            max_z=model_args.max_z,
            max_num_neighbors=model_args.max_num_neighbors,
            layernorm_on_vec="whitened",
        )
    else:
        raise ValueError(f'Unknown architecture: {model_args.model}')

    # create output network
    output_model = EquivariantScalar(model_args.embedding_dimension)

    # create the denoising output network
    output_model_noise = EquivariantVectorOutput(model_args.embedding_dimension)
        
    # combine representation and output network
    model = TorchMD_Net(
        representation_model=representation_model,
        output_model=output_model,
        prior_model=prior_model,
        reduce_op="add",
        mean=mean,
        std=std,
        derivative=model_args.derivative,
        output_model_noise=output_model_noise,
        position_noise_scale=data_args.position_noise_scale,
    )
    return model


def load_model(filepath, data_args, model_args, device="cpu", mean=None, std=None) -> TorchMD_Net:
    '''Loads a model from a checkpoint file.
    This function handles the loading of a saved model checkpoint,
    reconstructing the model architecture and loading the saved weights.
    Args:
        filepath (str): The path to the checkpoint file.
        data_args (DataArguments): Arguments related to the data.
        model_args (ModelArguments): Arguments related to the model.
        device (str, optional): The device to map the model to. (default: "cpu")
        mean (torch.Tensor, optional): The mean of the training data for
            denormalization. (default: None)
        std (torch.Tensor, optional): The standard deviation of the
            training data for denormalization. (default: None)
    Returns:
        TorchMD_Net: The loaded PyTorch model instance.'''
    device = torch.device(device)
    pretrained_dict = torch.load(filepath, map_location=device) 
    model = create_model(data_args, model_args)
    model.load_state_dict(pretrained_dict, strict=False)

    if mean:
        model.mean = mean
    if std:
        model.std = std

    return model.to(device)


class LNNP(nn.Module):
    def __init__(self, data_args, model_args, training_args, prior_model=None, mean=None, std=None):
        super(LNNP, self).__init__()
        self.data_args = data_args
        self.model_args = model_args
        self.training_args = training_args 
        
        if self.training_args.pretrained_model:
            self.model = load_model(
                self.training_args.pretrained_model, 
                self.data_args, 
                self.model_args, 
                device=self.training_args.device, 
                mean=mean, 
                std=std
            )
        else:
            self.model = create_model(self.data_args, self.model_args, prior_model, mean, std)

        self.ema = {"train_y": None, "val_y": None, "train_dy": None, "val_dy": None}

    def forward(self, z, pos, batch=None):
        return self.model(z, pos, batch=batch)

    def compute_loss(self, batch, stage):
        with torch.set_grad_enabled(stage == "train" or self.model_args.derivative):
            pred, noise_pred, deriv = self(batch.z, batch.pos, batch.batch)

        denoising_is_on = ("pos_target" in batch) and (self.data_args.denoising_weight > 0) and (noise_pred is not None)
        
        loss_y, loss_dy, loss_pos = torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)

        if self.model_args.derivative:
            if "y" not in batch:
                deriv = deriv + pred.sum() * 0

            loss_dy = mse_loss(deriv, batch.dy)

            if stage in ["train", "val"] and self.training_args.ema_alpha_dy < 1:
                ema_key = stage + "_dy"
                if self.ema[ema_key] is None:
                    self.ema[ema_key] = loss_dy.detach()
                loss_dy = (
                    self.training_args.ema_alpha_dy * loss_dy
                    + (1 - self.training_args.ema_alpha_dy) * self.ema[ema_key]
                )
                self.ema[ema_key] = loss_dy.detach()

        if "y" in batch:
            if (noise_pred is not None) and not denoising_is_on:
                pred = pred + noise_pred.sum() * 0

            if batch.y.ndim == 1:
                batch.y = batch.y.unsqueeze(1)

            loss_y = mse_loss(pred, batch.y)

            if stage in ["train", "val"] and self.training_args.ema_alpha_y < 1:
                ema_key = stage + "_y"
                if self.ema[ema_key] is None:
                    self.ema[ema_key] = loss_y.detach()
                loss_y = (
                    self.training_args.ema_alpha_y * loss_y
                    + (1 - self.training_args.ema_alpha_y) * self.ema[ema_key]
                )
                self.ema[ema_key] = loss_y.detach()

        if denoising_is_on:
            if "y" not in batch:
                noise_pred = noise_pred + pred.sum() * 0
            
            normalized_pos_target = self.model.pos_normalizer(batch.pos_target)
            loss_pos = mse_loss(noise_pred, normalized_pos_target)

        total_loss = (
            loss_y * self.data_args.energy_weight 
            + loss_dy * self.data_args.force_weight 
            + loss_pos * self.data_args.denoising_weight
        )
        
        return {
            "total_loss": total_loss,
            "energy_loss": loss_y.detach(),
            "force_loss": loss_dy.detach(),
            "denoising_loss": loss_pos.detach()
        }

