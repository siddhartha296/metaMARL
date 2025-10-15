"""
Evaluation Script for MetaMARL vs CoLight
File: scripts/evaluate.py
"""

import os
import sys
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple
import argparse
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.envs.cityflow_env import CityFlowEnv
from src.agents.metalight_agent import MetaLightAgent, MultiAgentController
from src.agents.colight_agent import CoLightAgent, CoLightMultiAgentController
from src.models.network_architectures import FRAPPlusPlus, CoLightGAT
from src.core.maml import MAML


def setup_logging(log_dir: str):
    """Setup logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'evaluate.log')),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


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


def evaluate_episode(
    env: CityFlowEnv,
    agents,
    max_steps: int = 3600
) -> Dict[str, float]:
    """
    Evaluate agents for one episode.
    
    Args:
        env: CityFlow environment
        agents: Agent controller (MetaMARL or CoLight)
        max_steps: Maximum steps per episode
        
    Returns:
        Dictionary of performance metrics
    """
    observations = env.reset()
    
    travel_times = []
    queue_lengths = []
    delays = []
    total_reward = 0
    
    for step in range(max_steps):
        # Select actions deterministically
        actions = agents.select_actions(observations, deterministic=True)
        
        # Step environment
        next_observations, rewards, dones, infos = env.step(actions)
        
        # Collect metrics
        total_reward += sum(rewards.values())
        travel_times.append(infos.get('avg_travel_time', 0))
        queue_lengths.append(infos.get('total_queue_length', 0))
        
        observations = next_observations
        
        if dones['__all__']:
            break
    
    return {
        'avg_travel_time': np.mean(travel_times) if travel_times else 0,
        'avg_queue_length': np.mean(queue_lengths) if queue_lengths else 0,
        'total_reward': total_reward,
        'episode_length': step + 1
    }


def adapt_metalight(
    env: CityFlowEnv,
    meta_model,
    maml: MAML,
    num_adaptation_steps: int,
    config: Dict,
    logger: logging.Logger
) -> Tuple[List[Dict], MultiAgentController]:
    """
    Adapt MetaMARL to new task and track performance.
    
    Args:
        env: Test environment
        meta_model: Meta-trained model
        maml: MAML trainer
        num_adaptation_steps: Number of adaptation episodes
        config: Configuration dictionary
        logger: Logger instance
        
    Returns:
        adaptation_history: List of metrics during adaptation
        adapted_agents: Final adapted agent controller
    """
    device = maml.device
    adaptation_history = []
    
    # Create agents with meta-model
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
    
    logger.info(f"Adapting MetaMARL for {num_adaptation_steps} episodes...")
    
    # Create optimizer for adaptation
    adapt_optimizer = torch.optim.SGD(
        meta_model.parameters(),
        lr=config.get('adapt_lr', 1e-2)
    )
    
    for adapt_step in range(num_adaptation_steps):
        # Collect episode
        observations = env.reset()
        episode_data = {'states': [], 'actions': [], 'rewards': []}
        
        for step in range(config.get('episode_length', 3600)):
            actions = agents.select_actions(observations, deterministic=False)
            next_observations, rewards, dones, infos = env.step(actions)
            
            # Store for adaptation
            for agent_id in agents.agent_ids:
                episode_data['states'].append(observations[agent_id])
                episode_data['actions'].append(actions[agent_id])
                episode_data['rewards'].append(rewards[agent_id])
            
            # Store in agent buffers
            agents.store_transitions(
                observations, actions, rewards,
                next_observations, dones
            )
            
            observations = next_observations
            
            if dones['__all__']:
                break
        
        # Perform adaptation update
        batches = agents.get_all_training_batches()
        
        for batch in batches:
            states = batch['states']
            returns = batch['returns']
            
            values, _ = meta_model(states)
            loss = torch.nn.MSELoss()(values, returns)
            
            adapt_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(meta_model.parameters(), max_norm=1.0)
            adapt_optimizer.step()
        
        # Evaluate current performance
        metrics = evaluate_episode(env, agents, max_steps=config.get('episode_length', 3600))
        metrics['adaptation_step'] = adapt_step + 1
        adaptation_history.append(metrics)
        
        logger.info(
            f"Adaptation Step {adapt_step + 1}/{num_adaptation_steps} | "
            f"Travel Time: {metrics['avg_travel_time']:.2f} | "
            f"Queue Length: {metrics['avg_queue_length']:.2f}"
        )
        
        # Clear buffers
        agents.clear_all_buffers()
    
    return adaptation_history, agents


def evaluate_colight(
    env: CityFlowEnv,
    model_dir: str,
    neighbor_map: Dict[str, List[str]],
    config: Dict,
    logger: logging.Logger
) -> Dict[str, float]:
    """
    Evaluate CoLight baseline (zero-shot, no adaptation).
    
    Args:
        env: Test environment
        model_dir: Directory containing saved CoLight models
        neighbor_map: Neighbor mapping for agents
        config: Configuration dictionary
        logger: Logger instance
        
    Returns:
        Dictionary of performance metrics
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load agents
    agents_dict = {}
    for inter_id in env.intersections:
        model = CoLightGAT(
            obs_dim=env.get_observation_space(inter_id),
            action_dim=env.get_action_space(inter_id),
            hidden_dim=config['hidden_dim'],
            num_neighbors=len(neighbor_map.get(inter_id, []))
        )
        
        agent = CoLightAgent(
            agent_id=inter_id,
            obs_dim=env.get_observation_space(inter_id),
            action_dim=env.get_action_space(inter_id),
            model=model,
            neighbor_ids=neighbor_map.get(inter_id, []),
            gamma=config['gamma'],
            device=device
        )
        
        # Load trained model
        model_path = os.path.join(model_dir, f'colight_{inter_id}_final.pth')
        if os.path.exists(model_path):
            agent.load_model(model_path)
        else:
            logger.warning(f"Model not found: {model_path}, using random initialization")
        
        agents_dict[inter_id] = agent
    
    agents = CoLightMultiAgentController(agents_dict)
    
    logger.info("Evaluating CoLight (zero-shot)...")
    
    # Evaluate
    metrics = evaluate_episode(env, agents, max_steps=config.get('episode_length', 3600))
    
    logger.info(
        f"CoLight Performance | "
        f"Travel Time: {metrics['avg_travel_time']:.2f} | "
        f"Queue Length: {metrics['avg_queue_length']:.2f}"
    )
    
    return metrics


def plot_adaptation_curve(
    metalight_history: List[Dict],
    colight_metrics: Dict,
    save_path: str,
    task_name: str
):
    """
    Plot adaptation curves comparing MetaMARL and CoLight.
    
    Args:
        metalight_history: Adaptation history for MetaMARL
        colight_metrics: Final metrics for CoLight
        save_path: Path to save plot
        task_name: Name of the task
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Extract MetaMARL data
    steps = [m['adaptation_step'] for m in metalight_history]
    travel_times = [m['avg_travel_time'] for m in metalight_history]
    queue_lengths = [m['avg_queue_length'] for m in metalight_history]
    
    # Plot travel time
    axes[0].plot(steps, travel_times, marker='o', label='MetaMARL', linewidth=2)
    axes[0].axhline(y=colight_metrics['avg_travel_time'], 
                    color='r', linestyle='--', label='CoLight (Zero-Shot)')
    axes[0].set_xlabel('Adaptation Episodes')
    axes[0].set_ylabel('Average Travel Time (s)')
    axes[0].set_title(f'Travel Time Adaptation on {task_name}')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot queue length
    axes[1].plot(steps, queue_lengths, marker='o', label='MetaMARL', linewidth=2)
    axes[1].axhline(y=colight_metrics['avg_queue_length'], 
                    color='r', linestyle='--', label='CoLight (Zero-Shot)')
    axes[1].set_xlabel('Adaptation Episodes')
    axes[1].set_ylabel('Average Queue Length')
    axes[1].set_title(f'Queue Length Adaptation on {task_name}')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
def main():
    parser = argparse.ArgumentParser(description='Evaluate MetaMARL vs CoLight')
    parser.add_argument('--task_config', type=str, required=True,
                        help='Path to the test task configuration file (flow file)')
    parser.add_argument('--metalight_config', type=str, default='configs/metalight_config.json',
                        help='Path to MetaMARL configuration file')
    parser.add_argument('--colight_config', type=str, default='configs/colight_config.json',
                        help='Path to CoLight configuration file')
    parser.add_argument('--metalight_model', type=str, required=True,
                        help='Path to saved MetaMARL meta-model')
    parser.add_argument('--colight_model_dir', type=str, required=True,
                        help='Directory of saved CoLight models')
    parser.add_argument('--num_adapt_steps', type=int, default=10,
                        help='Number of adaptation steps for MetaMARL')
    parser.add_argument('--output_dir', type=str, default='results/plots',
                        help='Directory to save evaluation plots')
    args = parser.parse_args()

    # Setup logging
    log_dir = 'results/logs'
    logger = setup_logging(log_dir)

    # Load configurations
    with open(args.metalight_config, 'r') as f:
        metalight_config = json.load(f)
    with open(args.colight_config, 'r') as f:
        colight_config = json.load(f)

    # Create CityFlow environment for the test task
    task_name = Path(args.task_config).stem
    roadnet_path = str(Path(args.task_config).parent / 'roadnet.json')
    cityflow_config_path = f'/tmp/eval_{task_name}.json'
    create_cityflow_config(roadnet_path, args.task_config, cityflow_config_path)
    
    env = CityFlowEnv(cityflow_config_path, num_steps=metalight_config.get('episode_length', 3600))
    obs_dim = env.get_observation_space(env.intersections[0])
    action_dim = env.get_action_space(env.intersections[0])
    
    # --- Evaluate MetaMARL ---
    logger.info("--- Evaluating MetaMARL ---")
    meta_model = FRAPPlusPlus(obs_dim, action_dim, hidden_dim=metalight_config['hidden_dim'])
    maml = MAML(model=meta_model, inner_lr=metalight_config['inner_lr'], meta_lr=metalight_config['meta_lr'])
    maml.load_model(args.metalight_model)
    
    metalight_history, _ = adapt_metalight(env, meta_model, maml, args.num_adapt_steps, metalight_config, logger)

    # --- Evaluate CoLight ---
    logger.info("\n--- Evaluating CoLight ---")
    # A simple neighbor map for evaluation (replace with actual logic if available)
    neighbor_map = {inter_id: [n for n in env.intersections if n != inter_id][:4] for inter_id in env.intersections}
    colight_metrics = evaluate_colight(env, args.colight_model_dir, neighbor_map, colight_config, logger)

    # --- Plotting Results ---
    os.makedirs(args.output_dir, exist_ok=True)
    plot_path = os.path.join(args.output_dir, f'adaptation_comparison_{task_name}.png')
    plot_adaptation_curve(metalight_history, colight_metrics, plot_path, task_name)
    
    logger.info(f"\nEvaluation complete. Plot saved to {plot_path}")

if __name__ == '__main__':
    main()