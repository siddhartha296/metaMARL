"""
MetaLight Agent Implementation
File: src/agents/metalight_agent.py
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
import random


class MetaLightAgent:
    """
    MetaLight agent using meta-learning for fast adaptation.
    Implements Independent Advantage Actor-Critic (IA2C) with MAML.
    """
    
    def __init__(
        self,
        agent_id: str,
        obs_dim: int,
        action_dim: int,
        model: nn.Module,
        gamma: float = 0.95,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize MetaLight agent.
        
        Args:
            agent_id: Unique identifier for the agent
            obs_dim: Observation space dimension
            action_dim: Action space dimension
            model: Actor-Critic neural network
            gamma: Discount factor
            entropy_coef: Entropy regularization coefficient
            value_coef: Value loss coefficient
            device: Device for computation
        """
        self.agent_id = agent_id
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.model = model
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.device = device
        
        # Experience buffer
        self.buffer = ExperienceBuffer()
        
        # Episode tracking
        self.episode_rewards = []
        self.episode_lengths = []
        
    def select_action(
        self, 
        obs: np.ndarray, 
        deterministic: bool = False
    ) -> Tuple[int, float, float]:
        """
        Select action using current policy.
        
        Args:
            obs: Current observation
            deterministic: If True, select argmax action
            
        Returns:
            action: Selected action index
            log_prob: Log probability of the action
            value: Estimated state value
        """
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            value, policy_logits = self.model(obs_tensor)
            
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
        log_prob: float,
        value: float
    ):
        """Store transition in experience buffer."""
        self.buffer.add(obs, action, reward, next_obs, done, log_prob, value)
    
    def compute_returns(self, next_value: float = 0.0) -> np.ndarray:
        """
        Compute discounted returns using GAE (Generalized Advantage Estimation).
        
        Args:
            next_value: Bootstrap value for last state
            
        Returns:
            returns: Discounted returns for each step
        """
        rewards = self.buffer.rewards
        values = self.buffer.values + [next_value]
        
        returns = []
        gae = 0
        
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step + 1] - values[step]
            gae = delta + self.gamma * 0.95 * gae  # lambda = 0.95 for GAE
            returns.insert(0, gae + values[step])
        
        return np.array(returns)
    
    def get_training_batch(self, next_value: float = 0.0) -> Dict[str, torch.Tensor]:
        """
        Prepare training batch from buffer.
        
        Args:
            next_value: Bootstrap value for last state
            
        Returns:
            Dictionary containing training tensors
        """
        returns = self.compute_returns(next_value)
        
        batch = {
            'states': torch.FloatTensor(np.array(self.buffer.states)).to(self.device),
            'actions': torch.LongTensor(self.buffer.actions).to(self.device),
            'returns': torch.FloatTensor(returns).unsqueeze(1).to(self.device),
            'old_log_probs': torch.FloatTensor(self.buffer.log_probs).to(self.device),
            'values': torch.FloatTensor(self.buffer.values).unsqueeze(1).to(self.device)
        }
        
        return batch
    
    def compute_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns: torch.Tensor,
        old_log_probs: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute A2C loss.
        
        Args:
            states: State tensor
            actions: Action tensor
            returns: Return tensor
            old_log_probs: Old log probabilities (for logging)
            
        Returns:
            total_loss: Combined actor-critic loss
            metrics: Dictionary of loss components
        """
        # Forward pass
        values, policy_logits = self.model(states)
        
        # Compute advantages
        advantages = returns - values.detach()
        
        # Policy loss
        action_probs = torch.softmax(policy_logits, dim=-1)
        dist = torch.distributions.Categorical(action_probs)
        log_probs = dist.log_prob(actions)
        
        policy_loss = -(log_probs * advantages.squeeze()).mean()
        
        # Value loss
        value_loss = nn.MSELoss()(values, returns)
        
        # Entropy bonus (encourage exploration)
        entropy = dist.entropy().mean()
        
        # Combined loss
        total_loss = (
            policy_loss + 
            self.value_coef * value_loss - 
            self.entropy_coef * entropy
        )
        
        metrics = {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'total_loss': total_loss.item()
        }
        
        return total_loss, metrics
    
    def clear_buffer(self):
        """Clear experience buffer."""
        self.buffer.clear()
    
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


class ExperienceBuffer:
    """Buffer for storing agent experiences."""
    
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []
        self.values = []
    
    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        log_prob: float,
        value: float
    ):
        """Add transition to buffer."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
    
    def __len__(self):
        return len(self.states)
    
    def clear(self):
        """Clear all stored experiences."""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()


class MultiAgentController:
    """
    Controller for managing multiple MetaLight agents.
    Handles coordination and joint training.
    """
    
    def __init__(self, agents: Dict[str, MetaLightAgent]):
        """
        Initialize multi-agent controller.
        
        Args:
            agents: Dictionary mapping agent IDs to MetaLightAgent instances
        """
        self.agents = agents
        self.agent_ids = list(agents.keys())
    
    def select_actions(
        self, 
        observations: Dict[str, np.ndarray],
        deterministic: bool = False
    ) -> Dict[str, int]:
        """
        Select actions for all agents.
        
        Args:
            observations: Dictionary of observations for each agent
            deterministic: Whether to use deterministic policy
            
        Returns:
            Dictionary of actions for each agent
        """
        actions = {}
        
        for agent_id, obs in observations.items():
            action, log_prob, value = self.agents[agent_id].select_action(
                obs, 
                deterministic
            )
            actions[agent_id] = action
            
            # Store value for later use
            self.agents[agent_id]._last_value = value
            self.agents[agent_id]._last_log_prob = log_prob
        
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
                self.agents[agent_id].store_transition(
                    observations[agent_id],
                    actions[agent_id],
                    rewards[agent_id],
                    next_observations[agent_id],
                    dones[agent_id],
                    self.agents[agent_id]._last_log_prob,
                    self.agents[agent_id]._last_value
                )
    
    def get_all_training_batches(self) -> List[Dict[str, torch.Tensor]]:
        """
        Get training batches for all agents.
        
        Returns:
            List of training batches (one per agent)
        """
        batches = []
        
        for agent_id in self.agent_ids:
            # Compute next value for bootstrapping
            last_obs = self.agents[agent_id].buffer.next_states[-1]
            _, _, next_value = self.agents[agent_id].select_action(last_obs)
            
            batch = self.agents[agent_id].get_training_batch(next_value)
            batches.append(batch)
        
        return batches
    
    def clear_all_buffers(self):
        """Clear buffers for all agents."""
        for agent in self.agents.values():
            agent.clear_buffer()
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all agents."""
        return {
            agent_id: agent.get_stats() 
            for agent_id, agent in self.agents.items()
        }
