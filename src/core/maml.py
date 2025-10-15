"""
Model-Agnostic Meta-Learning (MAML) Implementation
File: src/core/maml.py
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional
from copy import deepcopy


class MAML:
    """
    Model-Agnostic Meta-Learning for fast adaptation in MARL.
    Implements the MAML algorithm for multi-agent traffic signal control.
    """
    
    def __init__(
        self,
        model: nn.Module,
        meta_lr: float = 1e-3,
        inner_lr: float = 1e-2,
        inner_steps: int = 3,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize MAML trainer.
        
        Args:
            model: The neural network model to meta-train
            meta_lr: Outer loop (meta) learning rate
            inner_lr: Inner loop (task-specific) learning rate
            inner_steps: Number of gradient steps in inner loop
            device: Device to run computations on
        """
        self.model = model.to(device)
        self.meta_lr = meta_lr
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.device = device
        
        # Meta-optimizer (outer loop)
        self.meta_optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=meta_lr
        )
        
    def inner_loop_update(
        self,
        task_data: Dict[str, torch.Tensor],
        model_copy: nn.Module
    ) -> Tuple[nn.Module, float]:
        """
        Perform inner loop adaptation on a specific task.
        
        Args:
            task_data: Dictionary containing states, actions, returns for the task
            model_copy: A copy of the model to adapt
            
        Returns:
            adapted_model: Model after inner loop updates
            task_loss: Final loss on the task
        """
        # Create optimizer for inner loop
        inner_optimizer = torch.optim.SGD(
            model_copy.parameters(),
            lr=self.inner_lr
        )
        
        states = task_data['states'].to(self.device)
        actions = task_data['actions'].to(self.device)
        returns = task_data['returns'].to(self.device)
        
        # Perform inner loop gradient steps
        for step in range(self.inner_steps):
            # Forward pass
            values, _ = model_copy(states)
            
            # Compute loss (MSE between predicted values and returns)
            loss = nn.MSELoss()(values, returns)
            
            # Backward pass and update
            inner_optimizer.zero_grad()
            loss.backward()
            inner_optimizer.step()
        
        # Compute final loss for meta-update
        with torch.no_grad():
            values, _ = model_copy(states)
            final_loss = nn.MSELoss()(values, returns)
        
        return model_copy, final_loss.item()
    
    def outer_loop_update(
        self,
        task_batch: List[Dict[str, torch.Tensor]],
        query_batch: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, float]:
        """
        Perform outer loop meta-update across a batch of tasks.
        
        Args:
            task_batch: List of task data for support set (inner loop training)
            query_batch: List of task data for query set (meta-loss calculation)
            
        Returns:
            Dictionary containing meta-training metrics
        """
        self.meta_optimizer.zero_grad()
        
        meta_losses = []
        task_losses = []
        
        # Process each task in the batch
        for support_data, query_data in zip(task_batch, query_batch):
            # Create a copy of model for task-specific adaptation
            model_copy = deepcopy(self.model)
            
            # Inner loop: adapt to support set
            adapted_model, support_loss = self.inner_loop_update(
                support_data, 
                model_copy
            )
            task_losses.append(support_loss)
            
            # Evaluate adapted model on query set
            query_states = query_data['states'].to(self.device)
            query_returns = query_data['returns'].to(self.device)
            
            query_values, _ = adapted_model(query_states)
            query_loss = nn.MSELoss()(query_values, query_returns)
            
            meta_losses.append(query_loss)
        
        # Compute mean meta-loss
        meta_loss = torch.stack(meta_losses).mean()
        
        # Backward pass and meta-update
        meta_loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.meta_optimizer.step()
        
        return {
            'meta_loss': meta_loss.item(),
            'mean_task_loss': np.mean(task_losses),
            'std_task_loss': np.std(task_losses)
        }
    
    def adapt_to_task(
        self,
        adaptation_data: Dict[str, torch.Tensor],
        num_steps: int = 10
    ) -> nn.Module:
        """
        Adapt the meta-trained model to a new task (for evaluation).
        
        Args:
            adaptation_data: Data from the new task for adaptation
            num_steps: Number of gradient steps for adaptation
            
        Returns:
            Adapted model
        """
        # Create a fresh copy of the meta-model
        adapted_model = deepcopy(self.model)
        
        # Create optimizer for adaptation
        adapt_optimizer = torch.optim.SGD(
            adapted_model.parameters(),
            lr=self.inner_lr
        )
        
        states = adaptation_data['states'].to(self.device)
        actions = adaptation_data['actions'].to(self.device)
        returns = adaptation_data['returns'].to(self.device)
        
        adapted_model.train()
        
        # Perform adaptation gradient steps
        for step in range(num_steps):
            values, _ = adapted_model(states)
            loss = nn.MSELoss()(values, returns)
            
            adapt_optimizer.zero_grad()
            loss.backward()
            adapt_optimizer.step()
        
        return adapted_model
    
    def save_model(self, path: str):
        """Save meta-trained model."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'meta_optimizer_state_dict': self.meta_optimizer.state_dict(),
            'meta_lr': self.meta_lr,
            'inner_lr': self.inner_lr,
            'inner_steps': self.inner_steps
        }, path)
        
    def load_model(self, path: str):
        """Load meta-trained model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.meta_optimizer.load_state_dict(checkpoint['meta_optimizer_state_dict'])
        self.meta_lr = checkpoint['meta_lr']
        self.inner_lr = checkpoint['inner_lr']
        self.inner_steps = checkpoint['inner_steps']


class FirstOrderMAML(MAML):
    """
    First-order approximation of MAML (FOMAML).
    Computationally more efficient by ignoring second-order derivatives.
    """
    
    def outer_loop_update(
        self,
        task_batch: List[Dict[str, torch.Tensor]],
        query_batch: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, float]:
        """
        Perform FOMAML outer loop update.
        Uses only first-order gradients for efficiency.
        """
        self.meta_optimizer.zero_grad()
        
        meta_losses = []
        task_losses = []
        
        for support_data, query_data in zip(task_batch, query_batch):
            # Create model copy
            model_copy = deepcopy(self.model)
            
            # Inner loop with no gradient tracking
            with torch.no_grad():
                _, support_loss = self.inner_loop_update(support_data, model_copy)
                task_losses.append(support_loss)
            
            # Evaluate on query set (this computes gradients)
            query_states = query_data['states'].to(self.device)
            query_returns = query_data['returns'].to(self.device)
            
            query_values, _ = model_copy(query_states)
            query_loss = nn.MSELoss()(query_values, query_returns)
            
            meta_losses.append(query_loss)
        
        # Compute and backprop mean meta-loss
        meta_loss = torch.stack(meta_losses).mean()
        meta_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.meta_optimizer.step()
        
        return {
            'meta_loss': meta_loss.item(),
            'mean_task_loss': np.mean(task_losses),
            'std_task_loss': np.std(task_losses)
        }
