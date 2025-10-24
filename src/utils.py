"""
Utility Functions for MetaMARL Project
File: src/utils.py
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
import torch


def load_json(file_path: str) -> Dict:
    """Load JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def save_json(data: Dict, file_path: str):
    """Save dictionary to JSON file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def plot_training_curves(
    metrics: Dict[str, List[float]],
    save_path: str,
    title: str = "Training Curves"
):
    """
    Plot training curves.
    
    Args:
        metrics: Dictionary mapping metric names to lists of values
        save_path: Path to save plot
        title: Plot title
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, (metric_name, values) in enumerate(metrics.items()):
        if idx >= 4:
            break
        
        ax = axes[idx]
        ax.plot(values, linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel(metric_name.replace('_', ' ').title())
        ax.set_title(metric_name.replace('_', ' ').title())
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def calculate_statistics(values: List[float]) -> Dict[str, float]:
    """Calculate statistics for a list of values."""
    if not values:
        return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}
    
    return {
        'mean': np.mean(values),
        'std': np.std(values),
        'min': np.min(values),
        'max': np.max(values),
        'median': np.median(values)
    }


def create_comparison_table(
    metalight_results: Dict,
    colight_results: Dict,
    save_path: str
):
    """
    Create comparison table between MetaMARL and CoLight.
    
    Args:
        metalight_results: MetaMARL metrics
        colight_results: CoLight metrics
        save_path: Path to save table
    """
    import pandas as pd
    
    # Create DataFrame
    data = {
        'Metric': [
            'Average Travel Time (s)',
            'Average Queue Length',
            'Total Reward',
            'Episode Length'
        ],
        'MetaMARL': [
            f"{metalight_results.get('avg_travel_time', 0):.2f}",
            f"{metalight_results.get('avg_queue_length', 0):.2f}",
            f"{metalight_results.get('total_reward', 0):.2f}",
            f"{metalight_results.get('episode_length', 0):.0f}"
        ],
        'CoLight': [
            f"{colight_results.get('avg_travel_time', 0):.2f}",
            f"{colight_results.get('avg_queue_length', 0):.2f}",
            f"{colight_results.get('total_reward', 0):.2f}",
            f"{colight_results.get('episode_length', 0):.0f}"
        ]
    }
    
    df = pd.DataFrame(data)
    
    # Calculate improvement
    improvements = []
    for i in range(len(data['Metric'])):
        try:
            metalight_val = float(data['MetaMARL'][i])
            colight_val = float(data['CoLight'][i])
            
            if 'Reward' in data['Metric'][i]:
                # Higher is better for reward
                improvement = ((metalight_val - colight_val) / abs(colight_val)) * 100
            else:
                # Lower is better for other metrics
                improvement = ((colight_val - metalight_val) / colight_val) * 100
            
            improvements.append(f"{improvement:+.1f}%")
        except:
            improvements.append("N/A")
    
    df['Improvement'] = improvements
    
    # Save to CSV
    df.to_csv(save_path, index=False)
    print(f"Comparison table saved to {save_path}")
    print("\n", df.to_string(index=False))


def validate_data_structure(data_dir: str) -> bool:
    """
    Validate that the data directory has the correct structure.
    
    Args:
        data_dir: Path to data directory
        
    Returns:
        True if structure is valid, False otherwise
    """
    required_dirs = ['hangzhou_4x4', 'jinan_3x4']
    test_dirs = ['atlanta_2x2']
    
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"Error: Data directory {data_dir} does not exist")
        return False
    
    # Check training data
    for city_dir in required_dirs:
        city_path = data_path / city_dir
        if not city_path.exists():
            print(f"Warning: Missing training data directory: {city_dir}")
            continue
        
        # Check for roadnet and flow files
        roadnet = city_path / 'roadnet.json'
        if not roadnet.exists():
            print(f"Error: Missing roadnet.json in {city_dir}")
            return False
        
        flow_files = list(city_path.glob('flow*.json'))
        if len(flow_files) == 0:
            print(f"Warning: No flow files found in {city_dir}")
    
    # Check test data
    for city_dir in test_dirs:
        city_path = data_path / city_dir
        if not city_path.exists():
            print(f"Info: Test data directory {city_dir} not found (optional)")
    
    print("Data structure validation complete!")
    return True


def create_synthetic_flow(
    roadnet_path: str,
    output_path: str,
    num_vehicles: int = 1000,
    time_range: Tuple[int, int] = (0, 3600)
):
    """
    Create a synthetic traffic flow file for testing.
    
    Args:
        roadnet_path: Path to road network file
        output_path: Path to save flow file
        num_vehicles: Number of vehicles to generate
        time_range: (start_time, end_time) in seconds
    """
    # Load roadnet to get available roads
    with open(roadnet_path, 'r') as f:
        roadnet = json.load(f)
    
    # Get all non-virtual intersections
    intersections = [i['id'] for i in roadnet['intersections'] if not i['virtual']]
    roads = [r['id'] for r in roadnet['roads']]
    
    # Generate vehicles
    vehicles = []
    start_time, end_time = time_range
    
    for i in range(num_vehicles):
        # Random start time
        time = np.random.uniform(start_time, end_time * 0.8)
        
        # Random origin and destination roads
        start_road = np.random.choice(roads)
        end_road = np.random.choice([r for r in roads if r != start_road])
        
        vehicle = {
            "vehicle": {
                "length": 5.0,
                "width": 2.0,
                "maxPosAcc": 2.0,
                "maxNegAcc": 4.5,
                "usualPosAcc": 2.0,
                "usualNegAcc": 4.5,
                "minGap": 2.5,
                "maxSpeed": 16.67,
                "headwayTime": 1.5
            },
            "route": [start_road, end_road],
            "interval": 1.0,
            "startTime": time
        }
        vehicles.append(vehicle)
    
    # Save flow file
    with open(output_path, 'w') as f:
        json.dump(vehicles, f, indent=2)
    
    print(f"Created synthetic flow with {num_vehicles} vehicles: {output_path}")


def log_experiment_info(
    config: Dict,
    model: torch.nn.Module,
    log_path: str
):
    """
    Log experiment information for reproducibility.
    
    Args:
        config: Configuration dictionary
        model: Model to log
        log_path: Path to save log
    """
    info = {
        'config': config,
        'model_parameters': count_parameters(model),
        'model_architecture': str(model),
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'num_gpus': torch.cuda.device_count() if torch.cuda.is_available() else 0
    }
    
    save_json(info, log_path)
    print(f"Experiment info saved to {log_path}")


if __name__ == '__main__':
    # Test utility functions
    print("Testing MetaMARL utilities...")
    
    # Test data structure validation
    print("\n1. Validating data structure...")
    validate_data_structure('data')
    
    print("\nUtility functions are ready!")