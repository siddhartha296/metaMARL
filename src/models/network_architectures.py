"""
Neural Network Architectures for Traffic Signal Control
File: src/models/network_architectures.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional


class FRAPPlusPlus(nn.Module):
    """
    Improved FRAP++ architecture for traffic signal control.
    Uses separate phase and lane encoders with attention mechanism.
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        phase_dim: int = 8,
        lane_dim: int = 16
    ):
        """
        Initialize FRAP++ network.
        
        Args:
            obs_dim: Observation dimension
            action_dim: Action dimension (number of phases)
            hidden_dim: Hidden layer dimension
            phase_dim: Phase embedding dimension
            lane_dim: Lane embedding dimension
        """
        super(FRAPPlusPlus, self).__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Phase encoder
        self.phase_encoder = nn.Sequential(
            nn.Linear(action_dim, phase_dim),
            nn.ReLU(),
            nn.Linear(phase_dim, phase_dim)
        )
        
        # Lane encoder  
        # Assuming observation contains: phase_onehot + time + lane_features
        lane_feature_dim = obs_dim - action_dim - 1
        self.lane_encoder = nn.Sequential(
            nn.Linear(lane_feature_dim, lane_dim),
            nn.ReLU(),
            nn.Linear(lane_dim, lane_dim)
        )
        
        # Attention mechanism for phase-lane interaction
        self.attention = nn.MultiheadAttention(
            embed_dim=lane_dim,
            num_heads=2,
            batch_first=True
        )
        
        # Value head (critic)
        self.value_head = nn.Sequential(
            nn.Linear(phase_dim + lane_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Policy head (actor)
        self.policy_head = nn.Sequential(
            nn.Linear(phase_dim + lane_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            obs: Observation tensor [batch_size, obs_dim]
            
        Returns:
            value: State value [batch_size, 1]
            policy_logits: Action logits [batch_size, action_dim]
        """
        batch_size = obs.shape[0]
        
        # Split observation into components
        phase_onehot = obs[:, :self.action_dim]
        time_feature = obs[:, self.action_dim:self.action_dim + 1]
        lane_features = obs[:, self.action_dim + 1:]
        
        # Encode phase
        phase_embedding = self.phase_encoder(phase_onehot)
        
        # Encode lane features
        # Reshape for attention if needed
        lane_embedding = self.lane_encoder(lane_features)
        
        # Apply attention (treat lane as sequence)
        lane_embedding = lane_embedding.unsqueeze(1)  # [batch, 1, lane_dim]
        attended_lane, _ = self.attention(
            lane_embedding, 
            lane_embedding, 
            lane_embedding
        )
        attended_lane = attended_lane.squeeze(1)  # [batch, lane_dim]
        
        # Concatenate all features
        combined = torch.cat([phase_embedding, attended_lane, time_feature], dim=1)
        
        # Compute value and policy
        value = self.value_head(combined)
        policy_logits = self.policy_head(combined)
        
        return value, policy_logits


class CoLightGAT(nn.Module):
    """
    CoLight architecture using Graph Attention Networks.
    Enables communication between neighboring intersections.
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        num_neighbors: int = 4,
        num_heads: int = 2
    ):
        """
        Initialize CoLight network.
        
        Args:
            obs_dim: Observation dimension
            action_dim: Action dimension
            hidden_dim: Hidden layer dimension
            num_neighbors: Maximum number of neighbors
            num_heads: Number of attention heads
        """
        super(CoLightGAT, self).__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_neighbors = num_neighbors
        self.num_heads = num_heads
        
        # Local feature encoder
        self.local_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Graph attention layer
        self.gat_layer = GATLayer(
            in_features=hidden_dim,
            out_features=hidden_dim,
            num_heads=num_heads
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim * num_heads, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim * num_heads, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(
        self, 
        obs: torch.Tensor,
        neighbor_obs: Optional[torch.Tensor] = None,
        adj_matrix: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with graph attention.
        
        Args:
            obs: Agent's own observation [batch_size, obs_dim]
            neighbor_obs: Neighbors' observations [batch_size, num_neighbors, obs_dim]
            adj_matrix: Adjacency matrix [batch_size, 1 + num_neighbors]
            
        Returns:
            value: State value [batch_size, 1]
            policy_logits: Action logits [batch_size, action_dim]
        """
        batch_size = obs.shape[0]
        
        # Encode local observation
        local_embedding = self.local_encoder(obs)  # [batch, hidden_dim]
        
        if neighbor_obs is not None and adj_matrix is not None:
            # Encode neighbor observations
            neighbor_embeddings = []
            for i in range(self.num_neighbors):
                if i < neighbor_obs.shape[1]:
                    neighbor_emb = self.local_encoder(neighbor_obs[:, i, :])
                    neighbor_embeddings.append(neighbor_emb)
            
            # Stack all embeddings (self + neighbors)
            all_embeddings = torch.stack(
                [local_embedding] + neighbor_embeddings, 
                dim=1
            )  # [batch, 1 + num_neighbors, hidden_dim]
            
            # Apply graph attention
            attended_embedding = self.gat_layer(
                all_embeddings, 
                adj_matrix
            )  # [batch, hidden_dim * num_heads]
            
            final_embedding = attended_embedding[:, 0, :]  # Use self-attended features
        else:
            # No neighbors, use only local embedding
            final_embedding = local_embedding.repeat(1, self.num_heads)
        
        # Compute value and policy
        value = self.value_head(final_embedding)
        policy_logits = self.policy_head(final_embedding)
        
        return value, policy_logits


class GATLayer(nn.Module):
    """Graph Attention Layer for CoLight."""
    
    def __init__(self, in_features: int, out_features: int, num_heads: int = 2):
        super(GATLayer, self).__init__()
        
        self.num_heads = num_heads
        self.out_features = out_features
        
        # Multi-head attention weights
        self.W = nn.ModuleList([
            nn.Linear(in_features, out_features, bias=False)
            for _ in range(num_heads)
        ])
        
        self.a = nn.ModuleList([
            nn.Linear(2 * out_features, 1, bias=False)
            for _ in range(num_heads)
        ])
        
        self.leaky_relu = nn.LeakyReLU(0.2)
        
    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: Node features [batch, num_nodes, in_features]
            adj: Adjacency matrix [batch, num_nodes, num_nodes]
            
        Returns:
            Multi-head attention output [batch, num_nodes, out_features * num_heads]
        """
        batch_size, num_nodes, _ = h.shape
        
        outputs = []
        
        for head in range(self.num_heads):
            # Linear transformation
            Wh = self.W[head](h)  # [batch, num_nodes, out_features]
            
            # Compute attention coefficients
            Wh_i = Wh.unsqueeze(2).repeat(1, 1, num_nodes, 1)
            Wh_j = Wh.unsqueeze(1).repeat(1, num_nodes, 1, 1)
            
            attention_input = torch.cat([Wh_i, Wh_j], dim=-1)
            e = self.leaky_relu(self.a[head](attention_input).squeeze(-1))
            
            # Mask attention with adjacency matrix
            e = e.masked_fill(adj == 0, float('-inf'))
            
            # Attention weights
            attention = F.softmax(e, dim=-1)
            
            # Apply attention
            h_prime = torch.matmul(attention, Wh)
            outputs.append(h_prime)
        
        # Concatenate multi-head outputs
        return torch.cat(outputs, dim=-1)


class SimpleActorCritic(nn.Module):
    """
    Simple Actor-Critic network for baseline comparisons.
    """
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super(SimpleActorCritic, self).__init__()
        
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Critic head
        self.critic = nn.Linear(hidden_dim, 1)
        
        # Actor head
        self.actor = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.shared(obs)
        value = self.critic(features)
        policy_logits = self.actor(features)
        return value, policy_logits
