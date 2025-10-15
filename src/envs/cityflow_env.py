"""
CityFlow Environment Wrapper for Multi-Agent Traffic Signal Control
File: src/envs/cityflow_env.py
"""

import cityflow
import numpy as np
import json
from typing import Dict, List, Tuple, Optional


class CityFlowEnv:
    """
    Multi-agent environment wrapper for CityFlow traffic simulator.
    Each intersection is controlled by an independent agent.
    """
    
    def __init__(self, config_file: str, num_steps: int = 3600):
        """
        Initialize the CityFlow environment.
        
        Args:
            config_file: Path to CityFlow configuration file
            num_steps: Maximum episode length in simulation steps
        """
        self.config_file = config_file
        self.num_steps = num_steps
        self.eng = cityflow.Engine(config_file, thread_num=1)
        
        # Load configuration to extract road network info
        with open(config_file, 'r') as f:
            self.config = json.load(f)
        
        # Parse intersection information
        self._parse_intersections()
        
        # State tracking
        self.current_step = 0
        self.phase_times = {i: 0 for i in self.intersections}
        self.current_phases = {i: 0 for i in self.intersections}
        
        # Metrics
        self.total_waiting_time = 0
        self.total_queue_length = 0
        
    def _parse_intersections(self):
        """Parse intersection IDs and available phases from road network."""
        roadnet_file = self.config['dir'] + self.config['roadnetFile']
        
        with open(roadnet_file, 'r') as f:
            roadnet = json.load(f)
        
        # Extract signalized intersections
        self.intersections = []
        self.phase_list = {}
        self.incoming_lanes = {}
        self.outgoing_lanes = {}
        
        for intersection in roadnet['intersections']:
            if not intersection['virtual']:
                inter_id = intersection['id']
                self.intersections.append(inter_id)
                
                # Store available phases
                self.phase_list[inter_id] = len(intersection['trafficLight']['lightphases'])
                
                # Store lane information
                roads_in = intersection.get('roads', [])
                self.incoming_lanes[inter_id] = []
                self.outgoing_lanes[inter_id] = []
                
                for road_id in roads_in:
                    # Find road in roadnet
                    for road in roadnet['roads']:
                        if road['id'] == road_id:
                            if road['endIntersection'] == inter_id:
                                self.incoming_lanes[inter_id].extend(
                                    [f"{road_id}_{i}" for i in range(len(road['lanes']))]
                                )
                            if road['startIntersection'] == inter_id:
                                self.outgoing_lanes[inter_id].extend(
                                    [f"{road_id}_{i}" for i in range(len(road['lanes']))]
                                )
        
        self.num_agents = len(self.intersections)
        
    def reset(self) -> Dict[str, np.ndarray]:
        """
        Reset the environment to initial state.
        
        Returns:
            observations: Dictionary mapping agent IDs to observation arrays
        """
        self.eng.reset()
        self.current_step = 0
        self.phase_times = {i: 0 for i in self.intersections}
        self.current_phases = {i: 0 for i in self.intersections}
        self.total_waiting_time = 0
        self.total_queue_length = 0
        
        return self._get_observations()
    
    def step(self, actions: Dict[str, int]) -> Tuple[Dict, Dict, Dict, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            actions: Dictionary mapping agent IDs to action indices
            
        Returns:
            observations: Next state observations
            rewards: Rewards for each agent
            dones: Done flags for each agent
            infos: Additional information
        """
        # Apply actions (phase changes)
        for inter_id, action in actions.items():
            if action != self.current_phases[inter_id]:
                self.eng.set_tl_phase(inter_id, action)
                self.current_phases[inter_id] = action
                self.phase_times[inter_id] = 0
            else:
                self.phase_times[inter_id] += 1
        
        # Step simulator
        self.eng.next_step()
        self.current_step += 1
        
        # Get next observations
        observations = self._get_observations()
        
        # Calculate rewards
        rewards = self._get_rewards()
        
        # Check if episode is done
        done = self.current_step >= self.num_steps
        dones = {inter_id: done for inter_id in self.intersections}
        dones['__all__'] = done
        
        # Collect info
        infos = self._get_info()
        
        return observations, rewards, dones, infos
    
    def _get_observations(self) -> Dict[str, np.ndarray]:
        """
        Get observations for all agents.
        
        Returns:
            Dictionary mapping agent IDs to observation vectors
        """
        observations = {}
        
        for inter_id in self.intersections:
            obs = []
            
            # 1. Current phase (one-hot encoding)
            phase_one_hot = np.zeros(self.phase_list[inter_id])
            phase_one_hot[self.current_phases[inter_id]] = 1
            obs.extend(phase_one_hot)
            
            # 2. Time since last phase change (normalized)
            obs.append(min(self.phase_times[inter_id] / 60.0, 1.0))
            
            # 3. Lane-based features for incoming lanes
            for lane in self.incoming_lanes[inter_id]:
                # Vehicle count
                vehicle_count = self.eng.get_lane_vehicle_count()[lane]
                obs.append(min(vehicle_count / 20.0, 1.0))
                
                # Waiting vehicle count
                waiting_count = self.eng.get_lane_waiting_vehicle_count()[lane]
                obs.append(min(waiting_count / 10.0, 1.0))
            
            observations[inter_id] = np.array(obs, dtype=np.float32)
        
        return observations
    
    def _get_rewards(self) -> Dict[str, float]:
        """
        Calculate rewards for all agents.
        Uses negative waiting time as reward signal.
        
        Returns:
            Dictionary mapping agent IDs to reward values
        """
        rewards = {}
        
        for inter_id in self.intersections:
            waiting_time = 0
            
            for lane in self.incoming_lanes[inter_id]:
                waiting_time += self.eng.get_lane_waiting_vehicle_count()[lane]
            
            # Negative waiting time as reward (minimize waiting)
            rewards[inter_id] = -waiting_time
        
        return rewards
    
    def _get_info(self) -> Dict:
        """
        Collect environment metrics.
        
        Returns:
            Dictionary with performance metrics
        """
        # Calculate average travel time
        vehicle_count = self.eng.get_vehicle_count()
        avg_travel_time = 0
        
        if vehicle_count > 0:
            vehicles = self.eng.get_vehicles(include_waiting=True)
            total_travel_time = sum(
                self.eng.get_vehicle_info(v_id)['speed'] 
                for v_id in vehicles
            )
            avg_travel_time = total_travel_time / vehicle_count if vehicle_count > 0 else 0
        
        # Queue length
        total_queue = sum(
            self.eng.get_lane_waiting_vehicle_count()[lane]
            for lanes in self.incoming_lanes.values()
            for lane in lanes
        )
        
        return {
            'step': self.current_step,
            'vehicle_count': vehicle_count,
            'avg_travel_time': avg_travel_time,
            'total_queue_length': total_queue,
        }
    
    def get_observation_space(self, inter_id: str) -> int:
        """Get the dimension of observation space for an agent."""
        # Phase one-hot + time + (vehicle_count + waiting_count) per lane
        num_phases = self.phase_list[inter_id]
        num_lanes = len(self.incoming_lanes[inter_id])
        return num_phases + 1 + (num_lanes * 2)
    
    def get_action_space(self, inter_id: str) -> int:
        """Get the number of available actions for an agent."""
        return self.phase_list[inter_id]
