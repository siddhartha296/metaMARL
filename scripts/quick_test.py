"""
Quick Test Script for MetaMARL with Synthetic Environment
File: scripts/quick_test.py
"""

import os
import sys
import numpy as np
import torch

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.envs.synthetic_env import SyntheticTrafficEnv
from src.agents.metalight_agent import MetaLightAgent, MultiAgentController
from src.models.network_architectures import FRAPPlusPlus
from src.core.maml import MAML


def test_single_episode():
    """Test a single episode with MetaLight agent."""
    print("=" * 60)
    print("Testing Single Episode with MetaLight")
    print("=" * 60)
    
    # Create environment
    env = SyntheticTrafficEnv(num_intersections=4, num_phases=4)
    
    # Get dimensions
    sample_inter = env.intersections[0]
    obs_dim = env.get_observation_space(sample_inter)
    action_dim = env.get_action_space(sample_inter)
    
    print(f"\nEnvironment Info:")
    print(f"  Intersections: {env.num_agents}")
    print(f"  Observation dim: {obs_dim}")
    print(f"  Action dim: {action_dim}")
    
    # Create model and agents
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device: {device}")
    
    model = FRAPPlusPlus(obs_dim, action_dim, hidden_dim=64)
    
    agents_dict = {}
    for inter_id in env.intersections:
        agent = MetaLightAgent(
            agent_id=inter_id,
            obs_dim=obs_dim,
            action_dim=action_dim,
            model=model,
            gamma=0.95,
            device=device
        )
        agents_dict[inter_id] = agent
    
    agents = MultiAgentController(agents_dict)
    
    # Run episode
    print("\nRunning episode...")
    observations = env.reset()
    episode_reward = 0
    
    for step in range(100):  # Short test
        # Select actions
        actions = agents.select_actions(observations, deterministic=False)
        
        # Step environment
        next_observations, rewards, dones, infos = env.step(actions)
        
        # Store transitions
        agents.store_transitions(
            observations, actions, rewards,
            next_observations, dones
        )
        
        episode_reward += sum(rewards.values())
        observations = next_observations
        
        if step % 20 == 0:
            print(f"  Step {step}: Reward={sum(rewards.values()):.2f}, "
                  f"Queue={infos['total_queue_length']:.1f}")
        
        if dones['__all__']:
            break
    
    print(f"\nEpisode Summary:")
    print(f"  Total Reward: {episode_reward:.2f}")
    print(f"  Episode Length: {step + 1}")
    print(f"  Final Queue Length: {infos['total_queue_length']:.1f}")
    print(f"  Buffer sizes: {[len(a.buffer) for a in agents.agents.values()]}")
    

def test_maml_meta_update():
    """Test MAML meta-update with multiple tasks."""
    print("\n" + "=" * 60)
    print("Testing MAML Meta-Update")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create environments for different tasks
    num_tasks = 3
    envs = [SyntheticTrafficEnv(num_intersections=4) for _ in range(num_tasks)]
    
    # Create meta-model
    sample_env = envs[0]
    obs_dim = sample_env.get_observation_space(sample_env.intersections[0])
    action_dim = sample_env.get_action_space(sample_env.intersections[0])
    
    model = FRAPPlusPlus(obs_dim, action_dim, hidden_dim=64)
    maml = MAML(
        model=model,
        meta_lr=1e-3,
        inner_lr=1e-2,
        inner_steps=2,
        device=device
    )
    
    print(f"\nCollecting data from {num_tasks} tasks...")
    
    support_batches = []
    query_batches = []
    
    for task_idx, env in enumerate(envs):
        print(f"\nTask {task_idx + 1}: flow_pattern={env.flow_pattern}, "
              f"difficulty={env.task_difficulty:.2f}")
        
        # Create agents for this task
        agents_dict = {}
        for inter_id in env.intersections:
            agent = MetaLightAgent(
                agent_id=inter_id,
                obs_dim=obs_dim,
                action_dim=action_dim,
                model=model,
                gamma=0.95,
                device=device
            )
            agents_dict[inter_id] = agent
        
        agents = MultiAgentController(agents_dict)
        
        # Collect support episode
        observations = env.reset()
        for step in range(50):  # Short episodes for testing
            actions = agents.select_actions(observations, deterministic=False)
            next_observations, rewards, dones, infos = env.step(actions)
            agents.store_transitions(
                observations, actions, rewards, next_observations, dones
            )
            observations = next_observations
            if dones['__all__']:
                break
        
        support_batch = agents.get_all_training_batches()
        support_batches.extend(support_batch)
        agents.clear_all_buffers()
        
        # Collect query episode
        observations = env.reset()
        for step in range(50):
            actions = agents.select_actions(observations, deterministic=False)
            next_observations, rewards, dones, infos = env.step(actions)
            agents.store_transitions(
                observations, actions, rewards, next_observations, dones
            )
            observations = next_observations
            if dones['__all__']:
                break
        
        query_batch = agents.get_all_training_batches()
        query_batches.extend(query_batch)
    
    # Perform meta-update
    print("\nPerforming meta-update...")
    metrics = maml.outer_loop_update(support_batches, query_batches)
    
    print(f"\nMeta-Update Results:")
    print(f"  Meta Loss: {metrics['meta_loss']:.4f}")
    print(f"  Mean Task Loss: {metrics['mean_task_loss']:.4f} ± {metrics['std_task_loss']:.4f}")
    

def test_adaptation():
    """Test fast adaptation to new task."""
    print("\n" + "=" * 60)
    print("Testing Fast Adaptation")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create meta-trained model (randomly initialized for test)
    env = SyntheticTrafficEnv(num_intersections=4)
    obs_dim = env.get_observation_space(env.intersections[0])
    action_dim = env.get_action_space(env.intersections[0])
    
    meta_model = FRAPPlusPlus(obs_dim, action_dim, hidden_dim=64)
    maml = MAML(model=meta_model, inner_lr=1e-2, device=device)
    
    print("\nTesting on new task (different traffic pattern)...")
    
    # Evaluate before adaptation
    print("\n--- Before Adaptation ---")
    agents_dict = {}
    for inter_id in env.intersections:
        agent = MetaLightAgent(
            agent_id=inter_id,
            obs_dim=obs_dim,
            action_dim=action_dim,
            model=meta_model,
            gamma=0.95,
            device=device
        )
        agents_dict[inter_id] = agent
    
    agents = MultiAgentController(agents_dict)
    
    observations = env.reset()
    reward_before = 0
    for step in range(50):
        actions = agents.select_actions(observations, deterministic=True)
        next_observations, rewards, dones, infos = env.step(actions)
        reward_before += sum(rewards.values())
        observations = next_observations
        if dones['__all__']:
            break
    
    print(f"  Performance: Reward={reward_before:.2f}")
    
    # Adapt for few steps
    print("\n--- Adapting (3 episodes) ---")
    adapt_optimizer = torch.optim.SGD(meta_model.parameters(), lr=1e-2)
    
    for adapt_episode in range(3):
        observations = env.reset()
        for step in range(50):
            actions = agents.select_actions(observations, deterministic=False)
            next_observations, rewards, dones, infos = env.step(actions)
            agents.store_transitions(
                observations, actions, rewards, next_observations, dones
            )
            observations = next_observations
            if dones['__all__']:
                break
        
        # Update
        batches = agents.get_all_training_batches()
        for batch in batches:
            values, _ = meta_model(batch['states'])
            loss = torch.nn.MSELoss()(values, batch['returns'])
            adapt_optimizer.zero_grad()
            loss.backward()
            adapt_optimizer.step()
        
        agents.clear_all_buffers()
        print(f"  Episode {adapt_episode + 1}: Loss={loss.item():.4f}")
    
    # Evaluate after adaptation
    print("\n--- After Adaptation ---")
    observations = env.reset()
    reward_after = 0
    for step in range(50):
        actions = agents.select_actions(observations, deterministic=True)
        next_observations, rewards, dones, infos = env.step(actions)
        reward_after += sum(rewards.values())
        observations = next_observations
        if dones['__all__']:
            break
    
    print(f"  Performance: Reward={reward_after:.2f}")
    print(f"  Improvement: {reward_after - reward_before:+.2f}")


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("MetaMARL Quick Test Suite")
    print("=" * 60)
    
    # Run tests
    test_single_episode()
    test_maml_meta_update()
    test_adaptation()
    
    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)
