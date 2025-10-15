"""
Baseline Training Script for CoLight
File: scripts/baseline_train.py
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
from src.agents.colight_agent import CoLightAgent, CoLightMultiAgentController
from src.models.network_architectures import CoLightGAT


def setup_logging(log_dir: str):
    """Setup logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'baseline_train.log')),
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


def get_neighbor_map(env: CityFlowEnv) -> Dict[str, List[str]]:
    """
    Create neighbor mapping for intersections based on road network.
    
    Args:
        env: CityFlow environment
        
    Returns:
        Dictionary mapping agent IDs to lists of neighbor IDs
    """
    # Simple adjacency based on grid structure
    # In a real implementation, parse the road network topology
    neighbor_map = {}
    
    for inter_id in env.intersections:
        # Placeholder: assume grid structure
        # You would parse roadnet.json to get actual neighbors
        neighbor_map[inter_id] = [
            nid for nid in env.intersections 
            if nid != inter_id
        ][:4]  # Limit to 4 neighbors
    
    return neighbor_map


def create_cityflow_config(roadnet_path: str, flow_path: str, output_path: str):
    """Create CityFlow configuration file."""
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


def train_episode(
    env: CityFlowEnv,
    agents: CoLightMultiAgentController,
    max_steps: int = 3600,
    train: bool = True
) -> Dict[str, float]:
    """
    Run one training episode.
    
    Args:
        env: CityFlow environment
        agents: Multi-agent controller
        max_steps: Maximum steps per episode
        train: Whether to perform training updates
        
    Returns:
        Dictionary of episode metrics
    """
    observations = env.reset()
    episode_reward = 0
    episode_length = 0
    
    travel_times = []
    queue_lengths = []
    
    for step in range(max_steps):
        # Select actions
        actions = agents.select_actions(observations, deterministic=not train)
        
        # Step environment
        next_observations, rewards, dones, infos = env.step(actions)
        
        # Store transitions
        if train:
            agents.store_transitions(
                observations, actions, rewards,
                next_observations, dones
            )
        
        # Collect metrics
        episode_reward += sum(rewards.values())
        episode_length += 1
        travel_times.append(infos.get('avg_travel_time', 0))
        queue_lengths.append(infos.get('total_queue_length', 0))
        
        observations = next_observations
        
        if dones['__all__']:
            break
    
    # Train agents
    training_metrics = {}
    if train:
        training_metrics = agents.train_all_agents(batch_size=32)
    
    return {
        'episode_reward': episode_reward,
        'episode_length': episode_length,
        'avg_travel_time': np.mean(travel_times) if travel_times else 0,
        'avg_queue_length': np.mean(queue_lengths) if queue_lengths else 0,
        'training_metrics': training_metrics
    }


def baseline_train_loop(
    task_configs: List[str],
    config: Dict,
    logger: logging.Logger
):
    """
    Main training loop for CoLight baseline.
    
    Args:
        task_configs: List of task configuration paths
        config: Training configuration dictionary
        logger: Logger instance
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Create environment to get dimensions
    temp_config_path = '/tmp/temp_colight_config.json'
    roadnet_path = os.path.join(config['data_dir'], 'hangzhou_4x4', 'roadnet.json')
    flow_path = task_configs[0]
    create_cityflow_config(roadnet_path, flow_path, temp_config_path)
    
    temp_env = CityFlowEnv(temp_config_path, num_steps=100)
    sample_inter = temp_env.intersections[0]
    obs_dim = temp_env.get_observation_space(sample_inter)
    action_dim = temp_env.get_action_space(sample_inter)
    
    logger.info(f"Observation dim: {obs_dim}, Action dim: {action_dim}")
    logger.info(f"Number of intersections: {temp_env.num_agents}")
    
    # Get neighbor mapping
    neighbor_map = get_neighbor_map(temp_env)
    
    # Create agents
    agents_dict = {}
    for inter_id in temp_env.intersections:
        model = CoLightGAT(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=config['hidden_dim'],
            num_neighbors=len(neighbor_map[inter_id])
        )
        
        agent = CoLightAgent(
            agent_id=inter_id,
            obs_dim=obs_dim,
            action_dim=action_dim,
            model=model,
            neighbor_ids=neighbor_map[inter_id],
            gamma=config['gamma'],
            learning_rate=config['learning_rate'],
            device=device
        )
        agents_dict[inter_id] = agent
    
    agents = CoLightMultiAgentController(agents_dict)
    
    logger.info("Starting baseline training...")
    
    # Training loop
    best_reward = float('-inf')
    
    for iteration in range(config['num_iterations']):
        # Sample a task
        task_config = random.choice(task_configs)
        task_name = os.path.basename(task_config)
        city_dir = os.path.dirname(task_config)
        roadnet = os.path.join(city_dir, 'roadnet.json')
        
        config_path = f'/tmp/colight_task_{task_name}'
        create_cityflow_config(roadnet, task_config, config_path)
        
        env = CityFlowEnv(config_path, num_steps=config['episode_length'])
        
        # Train episode
        metrics = train_episode(
            env, agents,
            max_steps=config['episode_length'],
            train=True
        )
        
        # Update stats
        for agent_id, agent in agents.agents.items():
            agent.update_stats(
                metrics['episode_reward'] / len(agents.agents),
                metrics['episode_length']
            )
        
        # Log progress
        if (iteration + 1) % config['log_interval'] == 0:
            stats = agents.get_all_stats()
            mean_reward = np.mean([s['mean_reward'] for s in stats.values()])
            
            logger.info(
                f"Iteration {iteration + 1}/{config['num_iterations']} | "
                f"Task: {task_name} | "
                f"Reward: {mean_reward:.2f} | "
                f"Travel Time: {metrics['avg_travel_time']:.2f} | "
                f"Queue Length: {metrics['avg_queue_length']:.2f}"
            )
            
            # Save best model
            if mean_reward > best_reward:
                best_reward = mean_reward
                for agent_id, agent in agents.agents.items():
                    save_path = os.path.join(
                        config['save_dir'],
                        f"colight_{agent_id}_best.pth"
                    )
                    agent.save_model(save_path)
        
        # Save checkpoint
        if (iteration + 1) % config['save_interval'] == 0:
            for agent_id, agent in agents.agents.items():
                save_path = os.path.join(
                    config['save_dir'],
                    f"colight_{agent_id}_iter_{iteration + 1}.pth"
                )
                agent.save_model(save_path)
            logger.info(f"Saved checkpoint at iteration {iteration + 1}")
    
    # Save final models
    for agent_id, agent in agents.agents.items():
        final_path = os.path.join(config['save_dir'], f'colight_{agent_id}_final.pth')
        agent.save_model(final_path)
    
    logger.info(f"Training complete! Models saved to {config['save_dir']}")


def main():
    parser = argparse.ArgumentParser(description='Train CoLight baseline')
    parser.add_argument('--config', type=str, default='configs/colight_config.json',
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
    
    # Load task configurations (same as meta-training)
    task_configs = []
    data_dir = Path(config['data_dir'])
    for city_dir in data_dir.iterdir():
        if city_dir.is_dir() and city_dir.name in ['hangzhou_4x4', 'jinan_3x4']:
            flow_configs = [
                str(f) for f in city_dir.glob('*.json')
                if 'flow' in f.name and 'roadnet' not in f.name
            ]
            task_configs.extend(flow_configs)
    
    logger.info(f"Found {len(task_configs)} task configurations for training")
    
    # Start training
    baseline_train_loop(task_configs, config, logger)


if __name__ == '__main__':
    main()
