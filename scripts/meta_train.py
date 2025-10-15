"""
Meta-Training Script for MetaMARL
File: scripts/meta_train.py
"""

import os
import sys
import json
import numpy as np
import torch
import random
from pathlib import Path
from typing import List, Dict
import argparse
from datetime import datetime
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.envs.cityflow_env import CityFlowEnv
from src.agents.metalight_agent import MetaLightAgent, MultiAgentController
from src.models.network_architectures import FRAPPlusPlus
from src.core.maml import MAML


def setup_logging(log_dir: str):
    """Setup logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'meta_train.log')),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_task_configs(data_dir: str) -> List[str]:
    """
    Load all task configuration files for meta-training.
    
    Args:
        data_dir: Directory containing task configurations
        
    Returns:
        List of configuration file paths
    """
    task_configs = []
    
    # Search for config files in subdirectories
    for city_dir in Path(data_dir).iterdir():
        if city_dir.is_dir():
            config_files = list(city_dir.glob('*.json'))
            # Filter to get only flow config files
            flow_configs = [
                str(f) for f in config_files 
                if 'flow' in f.name and 'roadnet' not in f.name
            ]
            task_configs.extend(flow_configs)
    
    return task_configs


def create_cityflow_config(roadnet_path: str, flow_path: str, output_path: str):
    """
    Create CityFlow configuration file.
    
    Args:
        roadnet_path: Path to road network file
        flow_path: Path to flow file
        output_path: Output path for configuration
    """
    config = {
        "interval": 1.0,
        "seed": 0,
        "dir": os.path.dirname(roadnet_path) + "/",
        "roadnetFile": os.path.basename(roadnet_path),
        "flowFile": os.path.basename(flow_path),
        "rlTrafficLight": True,
        "saveReplay": False,
        "roadnetLogFile": "roadnet.json",
        "replayLogFile": "replay.txt"
    }
    
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=4)


def collect_task_episodes(
    env: CityFlowEnv,
    agents: MultiAgentController,
    num_episodes: int = 1,
    max_steps: int = 3600
) -> List[Dict[str, torch.Tensor]]:
    """
    Collect episodes from environment for a task.
    
    Args:
        env: CityFlow environment
        agents: Multi-agent controller
        num_episodes: Number of episodes to collect
        max_steps: Maximum steps per episode
        
    Returns:
        List of training batches from collected episodes
    """
    all_batches = []
    
    for episode in range(num_episodes):
        # Reset environment
        observations = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            # Select actions
            actions = agents.select_actions(observations, deterministic=False)
            
            # Step environment
            next_observations, rewards, dones, infos = env.step(actions)
            
            # Store transitions
            agents.store_transitions(
                observations, actions, rewards, 
                next_observations, dones
            )
            
            observations = next_observations
            episode_reward += sum(rewards.values())
            
            if dones['__all__']:
                break
        
        # Get training batches from all agents
        batches = agents.get_all_training_batches()
        all_batches.extend(batches)
        
        # Clear buffers
        agents.clear_all_buffers()
    
    return all_batches


def meta_train_loop(
    task_configs: List[str],
    config: Dict,
    logger: logging.Logger
):
    """
    Main meta-training loop.
    
    Args:
        task_configs: List of task configuration paths
        config: Training configuration dictionary
        logger: Logger instance
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Create a temporary environment to get dimensions
    temp_config_path = '/tmp/temp_cityflow_config.json'
    roadnet_path = os.path.join(config['data_dir'], 'hangzhou_4x4', 'roadnet.json')
    flow_path = task_configs[0]
    create_cityflow_config(roadnet_path, flow_path, temp_config_path)
    
    temp_env = CityFlowEnv(temp_config_path, num_steps=100)
    sample_inter = temp_env.intersections[0]
    obs_dim = temp_env.get_observation_space(sample_inter)
    action_dim = temp_env.get_action_space(sample_inter)
    
    logger.info(f"Observation dim: {obs_dim}, Action dim: {action_dim}")
    logger.info(f"Number of intersections: {temp_env.num_agents}")
    
    # Create meta-model
    meta_model = FRAPPlusPlus(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=config['hidden_dim']
    )
    
    # Initialize MAML
    maml = MAML(
        model=meta_model,
        meta_lr=config['meta_lr'],
        inner_lr=config['inner_lr'],
        inner_steps=config['inner_steps'],
        device=device
    )
    
    logger.info("Starting meta-training...")
    
    # Meta-training loop
    for meta_iter in range(config['num_meta_iterations']):
        # Sample batch of tasks
        task_batch = random.sample(task_configs, config['meta_batch_size'])
        
        support_batches = []
        query_batches = []
        
        for task_config in task_batch:
            # Create environment for this task
            task_name = os.path.basename(task_config)
            city_dir = os.path.dirname(task_config)
            roadnet = os.path.join(city_dir, 'roadnet.json')
            
            config_path = f'/tmp/task_{task_name}'
            create_cityflow_config(roadnet, task_config, config_path)
            
            env = CityFlowEnv(config_path, num_steps=config['episode_length'])
            
            # Create agents
            agents_dict = {}
            for inter_id in env.intersections:
                agent = MetaLightAgent(
                    agent_id=inter_id,
                    obs_dim=env.get_observation_space(inter_id),
                    action_dim=env.get_action_space(inter_id),
                    model=meta_model,
                    gamma=config['gamma'],
                    device=device
                )
                agents_dict[inter_id] = agent
            
            agents = MultiAgentController(agents_dict)
            
            # Collect support set (for inner loop)
            support_episodes = collect_task_episodes(
                env, agents, 
                num_episodes=config['support_episodes'],
                max_steps=config['episode_length']
            )
            support_batches.extend(support_episodes)
            
            # Collect query set (for meta-update)
            query_episodes = collect_task_episodes(
                env, agents,
                num_episodes=config['query_episodes'],
                max_steps=config['episode_length']
            )
            query_batches.extend(query_episodes)
        
        # Perform meta-update
        metrics = maml.outer_loop_update(support_batches, query_batches)
        
        # Log progress
        if (meta_iter + 1) % config['log_interval'] == 0:
            logger.info(
                f"Meta-iteration {meta_iter + 1}/{config['num_meta_iterations']} | "
                f"Meta Loss: {metrics['meta_loss']:.4f} | "
                f"Task Loss: {metrics['mean_task_loss']:.4f} ± {metrics['std_task_loss']:.4f}"
            )
        
        # Save checkpoint
        if (meta_iter + 1) % config['save_interval'] == 0:
            save_path = os.path.join(
                config['save_dir'], 
                f"metalight_iter_{meta_iter + 1}.pth"
            )
            maml.save_model(save_path)
            logger.info(f"Saved checkpoint to {save_path}")
    
    # Save final model
    final_path = os.path.join(config['save_dir'], 'metalight_final.pth')
    maml.save_model(final_path)
    logger.info(f"Meta-training complete! Final model saved to {final_path}")


def main():
    parser = argparse.ArgumentParser(description='Meta-train MetaMARL for traffic signal control')
    parser.add_argument('--config', type=str, default='configs/metalight_config.json',
                        help='Path to configuration file')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Set seed
    set_seed(args.seed)
    
    # Setup logging
    log_dir = os.path.join(config['save_dir'], 'logs')
    logger = setup_logging(log_dir)
    
    # Load task configurations
    task_configs = load_task_configs(config['data_dir'])
    logger.info(f"Found {len(task_configs)} task configurations for meta-training")
    
    # Start meta-training
    meta_train_loop(task_configs, config, logger)


if __name__ == '__main__':
    main()
