"""
CoLight Baseline Agent Implementation
File: src/agents/colight_agent.py
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
import random


class CoLightAgent:
    """
    CoLight agent with Graph Attention Network.
    Baseline for comparison with MetaLight.
    """
    
    def __init__(
        self,
        agent_id: str,
        obs_dim: int,
        action_dim: int,
        model: nn.Module,
        neighbor_ids: List[str],
        gamma: float = 0.95,
        learning_rate: float = 1e-3,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize CoLight agent.
        
        Args:
            agent_id: Unique identifier for the agent
            obs_dim: Observation space dimension
            action_dim: Action space dimension
            model: CoLight GAT network
            neighbor_ids: List of neighboring agent IDs
            gamma: Discount factor
            learning_rate: Learning rate for optimizer
            entropy_coef: Entropy regularization coefficient
            value_coef: Value loss coefficient
            device: Device for computation
        """
        self.agent_id = agent_id
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.model = model.to(device)
        self.neighbor_ids = neighbor_ids
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.device = device
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=learning_rate
        )
        
        # Experience replay buffer
        self.replay_buffer = ReplayBuffer(capacity=10000)
        
        # Episode tracking
        self.episode_rewards = []
        self.episode_lengths = []
        
        # Neighbor observation cache
        self.neighbor_obs_cache = {}
        
    def select_action(
        self,
        obs: np.ndarray,
        neighbor_obs: Optional[Dict[str, np.ndarray]] = None,
        deterministic: bool = False
    ) -> Tuple[int, float, float]:
        """
        Select action using current policy with neighbor information.
        
        Args:
            obs: Current observation
            neighbor_obs: Dictionary of neighbor observations
            deterministic: If True, select argmax action
            
        Returns:
            action: Selected action index
            log_prob: Log probability of the action
            value: Estimated state value
        """
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        
        # Prepare neighbor observations and adjacency matrix
        neighbor_obs_tensor = None
        adj_matrix = None
        
        if neighbor_obs is not None and len(neighbor_obs) > 0:
            # Stack neighbor observations
            neighbor_list = []
            for neighbor_id in self.neighbor_ids:
                if neighbor_id in neighbor_obs:
                    neighbor_list.append(neighbor_obs[neighbor_id])
                else:
                    # Use zeros if neighbor observation not available
                    neighbor_list.append(np.zeros_like(obs))
            
            if neighbor_list:
                neighbor_obs_tensor = torch.FloatTensor(
                    np.stack(neighbor_list)
                ).unsqueeze(0).to(self.device)
                
                # Create adjacency matrix (1 for self and existing neighbors)
                num_nodes = 1 + len(neighbor_list)
                adj_matrix = torch.ones(1, num_nodes, num_nodes).to(self.device)
        
        with torch.no_grad():
            value, policy_logits = self.model(
                obs_tensor,
                neighbor_obs_tensor,
                adj_matrix
            )
        
        # Create action distribution
        action_probs = torch.softmax(policy_logits, dim=-1)
        
        if deterministic:
            action = torch.argmax(action_probs, dim=-1).item()
            log_prob = torch.log(action_probs[0, action]).item()
        else:
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample().item()
            log_prob = dist.log_prob(torch.tensor(action)).item()
        
        return action, log_prob, value.item()
    
    def store_transition(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        neighbor_obs: Optional[Dict[str, np.ndarray]] = None,
        next_neighbor_obs: Optional[Dict[str, np.ndarray]] = None
    ):
        """Store transition in replay buffer."""
        self.replay_buffer.add(
            obs, action, reward, next_obs, done,
            neighbor_obs, next_neighbor_obs
        )
    
    def train_step(self, batch_size: int = 32) -> Optional[Dict[str, float]]:
        """
        Perform one training step using experience replay.
        
        Args:
            batch_size: Batch size for training
            
        Returns:
            Dictionary of training metrics or None if buffer too small
        """
        if len(self.replay_buffer) < batch_size:
            return None
        
        # Sample batch from replay buffer
        batch = self.replay_buffer.sample(batch_size)
        
        states = torch.FloatTensor(batch['states']).to(self.device)
        actions = torch.LongTensor(batch['actions']).to(self.device)
        rewards = torch.FloatTensor(batch['rewards']).to(self.device)
        next_states = torch.FloatTensor(batch['next_states']).to(self.device)
        dones = torch.FloatTensor(batch['dones']).to(self.device)
        
        # Forward pass - current states
        values, policy_logits = self.model(states)
        
        # Forward pass - next states (for TD target)
        with torch.no_grad():
            next_values, _ = self.model(next_states)
            td_targets = rewards.unsqueeze(1) + self.gamma * next_values * (1 - dones.unsqueeze(1))
        
        # Compute advantages
        advantages = td_targets - values
        
        # Policy loss
        action_probs = torch.softmax(policy_logits, dim=-1)
        dist = torch.distributions.Categorical(action_probs)
        log_probs = dist.log_prob(actions)
        
        policy_loss = -(log_probs * advantages.detach().squeeze()).mean()
        
        # Value loss
        value_loss = nn.MSELoss()(values, td_targets)
        
        # Entropy bonus
        entropy = dist.entropy().mean()
        
        # Combined loss
        total_loss = (
            policy_loss + 
            self.value_coef * value_loss - 
            self.entropy_coef * entropy
        )
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'total_loss': total_loss.item()
        }
    
    def update_stats(self, episode_reward: float, episode_length: int):
        """Update episode statistics."""
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        
        # Keep last 100 episodes
        if len(self.episode_rewards) > 100:
            self.episode_rewards.pop(0)
            self.episode_lengths.pop(0)
    
    def get_stats(self) -> Dict[str, float]:
        """Get agent statistics."""
        if len(self.episode_rewards) == 0:
            return {'mean_reward': 0.0, 'mean_length': 0.0}
        
        return {
            'mean_reward': np.mean(self.episode_rewards),
            'mean_length': np.mean(self.episode_lengths),
            'std_reward': np.std(self.episode_rewards)
        }
    
    def save_model(self, path: str):
        """Save agent model."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths
        }, path)
    
    def load_model(self, path: str):
        """Load agent model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.episode_rewards = checkpoint.get('episode_rewards', [])
        self.episode_lengths = checkpoint.get('episode_lengths', [])


class ReplayBuffer:
    """Experience replay buffer for CoLight agent."""
    
    def __init__(self, capacity: int = 10000):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum buffer size
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        neighbor_obs: Optional[Dict] = None,
        next_neighbor_obs: Optional[Dict] = None
    ):
        """Add transition to buffer."""
        self.buffer.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'neighbor_obs': neighbor_obs,
            'next_neighbor_obs': next_neighbor_obs
        })
    
    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """
        Sample a batch from buffer.
        
        Args:
            batch_size: Number of samples
            
        Returns:
            Dictionary containing batched data
        """
        samples = random.sample(self.buffer, batch_size)
        
        batch = {
            'states': np.array([s['state'] for s in samples]),
            'actions': np.array([s['action'] for s in samples]),
            'rewards': np.array([s['reward'] for s in samples]),
            'next_states': np.array([s['next_state'] for s in samples]),
            'dones': np.array([s['done'] for s in samples], dtype=np.float32)
        }
        
        return batch
    
    def __len__(self):
        return len(self.buffer)


class CoLightMultiAgentController:
    """
    Controller for managing multiple CoLight agents.
    """
    
    def __init__(self, agents: Dict[str, CoLightAgent]):
        """
        Initialize multi-agent controller.
        
        Args:
            agents: Dictionary mapping agent IDs to CoLightAgent instances
        """
        self.agents = agents
        self.agent_ids = list(agents.keys())
    
    def select_actions(
        self,
        observations: Dict[str, np.ndarray],
        deterministic: bool = False
    ) -> Dict[str, int]:
        """
        Select actions for all agents with neighbor communication.
        
        Args:
            observations: Dictionary of observations for each agent
            deterministic: Whether to use deterministic policy
            
        Returns:
            Dictionary of actions for each agent
        """
        actions = {}
        
        for agent_id, obs in observations.items():
            # Get neighbor observations
            neighbor_obs = {
                neighbor_id: observations[neighbor_id]
                for neighbor_id in self.agents[agent_id].neighbor_ids
                if neighbor_id in observations
            }
            
            action, _, _ = self.agents[agent_id].select_action(
                obs,
                neighbor_obs,
                deterministic
            )
            actions[agent_id] = action
        
        return actions
    
    def store_transitions(
        self,
        observations: Dict[str, np.ndarray],
        actions: Dict[str, int],
        rewards: Dict[str, float],
        next_observations: Dict[str, np.ndarray],
        dones: Dict[str, bool]
    ):
        """Store transitions for all agents."""
        for agent_id in self.agent_ids:
            if agent_id in observations:
                # Get neighbor observations
                neighbor_obs = {
                    nid: observations[nid]
                    for nid in self.agents[agent_id].neighbor_ids
                    if nid in observations
                }
                next_neighbor_obs = {
                    nid: next_observations[nid]
                    for nid in self.agents[agent_id].neighbor_ids
                    if nid in next_observations
                }
                
                self.agents[agent_id].store_transition(
                    observations[agent_id],
                    actions[agent_id],
                    rewards[agent_id],
                    next_observations[agent_id],
                    dones[agent_id],
                    neighbor_obs,
                    next_neighbor_obs
                )
    
    def train_all_agents(self, batch_size: int = 32) -> Dict[str, Dict[str, float]]:
        """Train all agents and return metrics."""
        metrics = {}
        
        for agent_id, agent in self.agents.items():
            agent_metrics = agent.train_step(batch_size)
            if agent_metrics is not None:
                metrics[agent_id] = agent_metrics
        
        return metrics
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all agents."""
        return {
            agent_id: agent.get_stats()
            for agent_id, agent in self.agents.items()
        }
