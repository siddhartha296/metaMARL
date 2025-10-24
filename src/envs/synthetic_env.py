"""
Synthetic Traffic Environment for Testing MetaMARL
File: src/envs/synthetic_env.py
"""

import numpy as np
from typing import Dict, List, Tuple


class SyntheticTrafficEnv:
    """
    Simplified synthetic traffic environment that mimics CityFlow API.
    Useful for testing without actual CityFlow installation.
    """
    
    def __init__(self, num_intersections: int = 4, num_phases: int = 4, 
                 num_lanes_per_intersection: int = 12):
        """
        Initialize synthetic environment.
        
        Args:
            num_intersections: Number of traffic intersections
            num_phases: Number of signal phases per intersection
            num_lanes_per_intersection: Number of incoming lanes per intersection
        """
        self.num_intersections = num_intersections
        self.num_phases = num_phases
        self.num_lanes = num_lanes_per_intersection
        self.max_steps = 3600
        
        # Create intersection IDs
        self.intersections = [f"intersection_{i}" for i in range(num_intersections)]
        self.num_agents = num_intersections
        
        # Phase information
        self.phase_list = {i: num_phases for i in self.intersections}
        
        # State variables
        self.current_step = 0
        self.current_phases = {i: 0 for i in self.intersections}
        self.phase_times = {i: 0 for i in self.intersections}
        
        # Traffic state (vehicles per lane)
        self.vehicle_counts = {
            inter_id: np.random.randint(0, 10, self.num_lanes) 
            for inter_id in self.intersections
        }
        self.waiting_counts = {
            inter_id: np.random.randint(0, 5, self.num_lanes)
            for inter_id in self.intersections
        }
        
        # Task characteristics (for meta-learning diversity)
        self.task_difficulty = np.random.uniform(0.5, 1.5)  # Traffic flow multiplier
        self.flow_pattern = np.random.choice(['uniform', 'rush_hour', 'asymmetric'])
        
    def reset(self) -> Dict[str, np.ndarray]:
        """Reset environment to initial state."""
        self.current_step = 0
        self.current_phases = {i: 0 for i in self.intersections}
        self.phase_times = {i: 0 for i in self.intersections}
        
        # Reset traffic with some randomness based on task
        for inter_id in self.intersections:
            if self.flow_pattern == 'rush_hour':
                # Higher traffic in certain directions
                self.vehicle_counts[inter_id] = np.random.randint(
                    5, 20, self.num_lanes
                ) * self.task_difficulty
            elif self.flow_pattern == 'asymmetric':
                # Imbalanced traffic
                half = self.num_lanes // 2
                self.vehicle_counts[inter_id] = np.concatenate([
                    np.random.randint(10, 20, half),
                    np.random.randint(0, 5, half)
                ]) * self.task_difficulty
            else:
                # Uniform traffic
                self.vehicle_counts[inter_id] = np.random.randint(
                    3, 12, self.num_lanes
                ) * self.task_difficulty
            
            self.waiting_counts[inter_id] = (
                self.vehicle_counts[inter_id] * np.random.uniform(0.3, 0.7, self.num_lanes)
            ).astype(int)
        
        return self._get_observations()
    
    def step(self, actions: Dict[str, int]) -> Tuple[Dict, Dict, Dict, Dict]:
        """Execute one environment step."""
        # Apply phase changes
        for inter_id, action in actions.items():
            if action != self.current_phases[inter_id]:
                # Phase change: some waiting time penalty
                self.phase_times[inter_id] = 0
                self.current_phases[inter_id] = action
                # Switching penalty
                for lane_idx in range(self.num_lanes):
                    self.waiting_counts[inter_id][lane_idx] += 1
            else:
                self.phase_times[inter_id] += 1
        
        # Simulate traffic dynamics
        self._update_traffic(actions)
        
        self.current_step += 1
        
        # Get observations and rewards
        observations = self._get_observations()
        rewards = self._get_rewards()
        
        # Check if done
        done = self.current_step >= self.max_steps
        dones = {inter_id: done for inter_id in self.intersections}
        dones['__all__'] = done
        
        infos = self._get_info()
        
        return observations, rewards, dones, infos
    
    def _update_traffic(self, actions: Dict[str, int]):
        """Simulate traffic flow dynamics."""
        for inter_id in self.intersections:
            current_phase = self.current_phases[inter_id]
            
            # Lanes that have green light (simplified: assume phase controls certain lanes)
            green_lanes = self._get_green_lanes(current_phase)
            
            for lane_idx in range(self.num_lanes):
                if lane_idx in green_lanes:
                    # Green light: vehicles pass through
                    vehicles_passed = min(
                        self.vehicle_counts[inter_id][lane_idx],
                        np.random.randint(1, 4)  # 1-3 vehicles per step
                    )
                    self.vehicle_counts[inter_id][lane_idx] -= vehicles_passed
                    self.waiting_counts[inter_id][lane_idx] = max(
                        0, self.waiting_counts[inter_id][lane_idx] - vehicles_passed
                    )
                else:
                    # Red light: vehicles accumulate
                    new_vehicles = np.random.poisson(0.5 * self.task_difficulty)
                    self.vehicle_counts[inter_id][lane_idx] += new_vehicles
                    self.waiting_counts[inter_id][lane_idx] += new_vehicles * 0.8
            
            # Add some random vehicle arrivals
            arrival_lanes = np.random.choice(
                self.num_lanes, 
                size=np.random.randint(0, 3), 
                replace=False
            )
            for lane_idx in arrival_lanes:
                self.vehicle_counts[inter_id][lane_idx] += 1
    
    def _get_green_lanes(self, phase: int) -> List[int]:
        """Get lanes with green light for given phase."""
        # Simplified: rotate which lanes get green light
        lanes_per_phase = self.num_lanes // self.num_phases
        start_idx = phase * lanes_per_phase
        return list(range(start_idx, start_idx + lanes_per_phase))
    
    def _get_observations(self) -> Dict[str, np.ndarray]:
        """Get observations for all intersections."""
        observations = {}
        
        for inter_id in self.intersections:
            obs = []
            
            # 1. Current phase (one-hot)
            phase_onehot = np.zeros(self.num_phases)
            phase_onehot[self.current_phases[inter_id]] = 1
            obs.extend(phase_onehot)
            
            # 2. Time since phase change (normalized)
            obs.append(min(self.phase_times[inter_id] / 60.0, 1.0))
            
            # 3. Lane features (vehicle count and waiting count, normalized)
            for lane_idx in range(self.num_lanes):
                obs.append(min(self.vehicle_counts[inter_id][lane_idx] / 20.0, 1.0))
                obs.append(min(self.waiting_counts[inter_id][lane_idx] / 10.0, 1.0))
            
            observations[inter_id] = np.array(obs, dtype=np.float32)
        
        return observations
    
    def _get_rewards(self) -> Dict[str, float]:
        """Calculate rewards (negative waiting time)."""
        rewards = {}
        
        for inter_id in self.intersections:
            total_waiting = np.sum(self.waiting_counts[inter_id])
            rewards[inter_id] = -float(total_waiting)
        
        return rewards
    
    def _get_info(self) -> Dict:
        """Get environment info."""
        total_vehicles = sum(
            np.sum(counts) for counts in self.vehicle_counts.values()
        )
        total_waiting = sum(
            np.sum(counts) for counts in self.waiting_counts.values()
        )
        
        avg_travel_time = total_waiting / max(total_vehicles, 1) * 10  # Rough estimate
        
        return {
            'step': self.current_step,
            'vehicle_count': int(total_vehicles),
            'avg_travel_time': float(avg_travel_time),
            'total_queue_length': float(total_waiting),
            'flow_pattern': self.flow_pattern,
            'task_difficulty': self.task_difficulty
        }
    
    def get_observation_space(self, inter_id: str) -> int:
        """Get observation space dimension."""
        return self.num_phases + 1 + (self.num_lanes * 2)
    
    def get_action_space(self, inter_id: str) -> int:
        """Get action space dimension."""
        return self.num_phases
