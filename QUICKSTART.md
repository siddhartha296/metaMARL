# MetaMARL Quick Start Guide

This guide will help you get MetaMARL running in 10 minutes.

## Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- 8GB+ RAM
- Ubuntu 18.04+ (or use Docker)

## Step-by-Step Setup

### 1. Clone and Setup (2 minutes)

```bash
# Clone the repository
git clone <your-repo-url>
cd MetaMARL

# Run setup script
bash setup.sh

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Data (3 minutes)

```bash
# Download from MetaLight repository
# https://github.com/MetaLight-Controller/MetaLight

# Or use wget (example)
cd data
wget https://github.com/MetaLight-Controller/MetaLight/releases/download/v1.0/datasets.tar.gz
tar -xzf datasets.tar.gz
cd ..

# Verify data structure
python -c "from src.utils import validate_data_structure; validate_data_structure('data')"
```

### 3. Create Configuration Files (1 minute)

Create `configs/metalight_config.json`:
```json
{
  "data_dir": "data",
  "save_dir": "saved_models/metalight",
  "meta_lr": 0.0001,
  "inner_lr": 0.001,
  "inner_steps": 3,
  "meta_batch_size": 4,
  "num_meta_iterations": 1000,
  "support_episodes": 2,
  "query_episodes": 2,
  "episode_length": 3600,
  "hidden_dim": 128,
  "gamma": 0.95,
  "log_interval": 10,
  "save_interval": 100
}
```

Create `configs/colight_config.json`:
```json
{
  "data_dir": "data",
  "save_dir": "saved_models/colight",
  "learning_rate": 0.0005,
  "num_iterations": 2000,
  "episode_length": 3600,
  "hidden_dim": 64,
  "gamma": 0.99,
  "log_interval": 10,
  "save_interval": 200
}
```

### 4. Quick Test (2 minutes)

Test if everything works:

```bash
# Test environment
python -c "
from src.envs.cityflow_env import CityFlowEnv
import json

# Create a minimal config
config = {
    'interval': 1.0,
    'seed': 0,
    'dir': 'data/hangzhou_4x4/',
    'roadnetFile': 'roadnet.json',
    'flowFile': 'flow_morning_peak.json',
    'rlTrafficLight': True,
    'saveReplay': False
}

with open('/tmp/test_config.json', 'w') as f:
    json.dump(config, f)

env = CityFlowEnv('/tmp/test_config.json', num_steps=100)
obs = env.reset()
print(f'Environment working! {len(obs)} agents detected.')
"
```

### 5. Train MetaMARL (30+ minutes)

```bash
# Start meta-training
python scripts/meta_train.py \
  --config configs/metalight_config.json \
  --seed 42

# Monitor progress
tail -f saved_models/metalight/logs/meta_train.log
```

Expected output:
```
Meta-iteration 10/1000 | Meta Loss: 2.3451 | Task Loss: 1.8723 ± 0.3421
Meta-iteration 20/1000 | Meta Loss: 1.9823 | Task Loss: 1.5234 ± 0.2891
...
```

### 6. Train CoLight Baseline (30+ minutes)

In a separate terminal:

```bash
# Start baseline training
python scripts/baseline_train.py \
  --config configs/colight_config.json \
  --seed 42

# Monitor progress
tail -f saved_models/colight/logs/baseline_train.log
```

### 7. Evaluate and Compare (5 minutes)

```bash
# Evaluate on in-distribution test task
python scripts/evaluate.py \
  --task_config data/hangzhou_4x4/flow_test.json \
  --metalight_config configs/metalight_config.json \
  --colight_config configs/colight_config.json \
  --metalight_model saved_models/metalight/metalight_final.pth \
  --colight_model_dir saved_models/colight/ \
  --num_adapt_steps 20 \
  --output_dir results/plots

# Evaluate on out-of-distribution task
python scripts/evaluate.py \
  --task_config data/atlanta_2x2/flow_config_1.json \
  --metalight_config configs/metalight_config.json \
  --colight_config configs/colight_config.json \
  --metalight_model saved_models/metalight/metalight_final.pth \
  --colight_model_dir saved_models/colight/ \
  --num_adapt_steps 20 \
  --output_dir results/plots
```

### 8. View Results

```bash
# Open generated plots
xdg-open results/plots/adaptation_flow_test.png
xdg-open results/plots/adaptation_flow_config_1.png

# View metrics
cat results/plots/results_flow_test.json
```

## Docker Quick Start

Even faster with Docker:

```bash
# Build image
docker build -t metamarl .

# Run with GPU support
docker run -it --gpus all \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/saved_models:/workspace/saved_models \
  -v $(pwd)/results:/workspace/results \
  metamarl

# Inside container
bash setup.sh
python scripts/meta_train.py --config configs/metalight_config.json
```

## Minimal Example

If you just want to test the code without full training:

```python
# test_minimal.py
import sys
import torch
from src.models.network_architectures import FRAPPlusPlus
from src.core.maml import MAML

# Create model
model = FRAPPlusPlus(obs_dim=32, action_dim=8, hidden_dim=64)

# Initialize MAML
maml = MAML(model=model, meta_lr=1e-3, inner_lr=1e-2, inner_steps=2)

# Create dummy data
dummy_batch = {
    'states': torch.randn(10, 32),
    'actions': torch.randint(0, 8, (10,)),
    'returns': torch.randn(10, 1)
}

# Test meta-update
metrics = maml.outer_loop_update([dummy_batch], [dummy_batch])
print(f"Meta-update successful! Loss: {metrics['meta_loss']:.4f}")
```

Run with: `python test_minimal.py`

## Troubleshooting

### Issue: CityFlow not found

```bash
# Install from source
git clone https://github.com/cityflow-project/CityFlow.git
cd CityFlow
pip install .
```

### Issue: CUDA out of memory

Reduce batch sizes in config:
```json
{
  "meta_batch_size": 2,  // Reduce from 4
  "episode_length": 1800  // Reduce from 3600
}
```

### Issue: No data files

Download manually from:
- [MetaLight Datasets](https://github.com/MetaLight-Controller/MetaLight)
- Place in appropriate `data/` subdirectories

### Issue: Import errors

```bash
# Ensure PYTHONPATH is set
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or add to ~/.bashrc
echo 'export PYTHONPATH="${PYTHONPATH}:'"$(pwd)"'"' >> ~/.bashrc
source ~/.bashrc
```

## Performance Tips

1. **Use GPU**: 10-20x faster training
2. **Reduce episode length**: Start with 1800 steps for faster iterations
3. **Use fewer meta-iterations**: 500 iterations often sufficient
4. **Parallel evaluation**: Run multiple evaluations simultaneously
5. **Monitor memory**: Use `nvidia-smi` to watch GPU usage

## Next Steps

1. **Experiment with hyperparameters**: Try different learning rates
2. **Add custom tasks**: Create synthetic traffic scenarios
3. **Extend architectures**: Modify network structures
4. **Compare algorithms**: Implement other meta-learning methods
5. **Visualize results**: Create custom plots and analysis

## Expected Timeline

- Setup: 10 minutes
- MetaMARL training: 1-4 hours (GPU) / 8-24 hours (CPU)
- CoLight training: 1-4 hours (GPU) / 8-24 hours (CPU)
- Evaluation: 10-30 minutes
- **Total: 3-9 hours for complete pipeline**

## Common Commands

```bash
# Check GPU status
nvidia-smi

# View training logs
tail -f saved_models/metalight/logs/meta_train.log

# Kill training
pkill -f meta_train.py

# Resume from checkpoint
python scripts/meta_train.py \
  --config configs/metalight_config.json \
  --resume saved_models/metalight/metalight_iter_500.pth

# Compare results
python -c "from src.utils import create_comparison_table; \
  import json; \
  ml = json.load(open('results/plots/results_flow_test.json'))['metalight_final']; \
  cl = json.load(open('results/plots/results_flow_test.json'))['colight']; \
  create_comparison_table(ml, cl, 'results/comparison.csv')"
```

## Resources

- **Paper**: [MetaLight: Value-Based Meta-Reinforcement Learning for Traffic Signal Control](https://arxiv.org/abs/2101.00702)
- **CityFlow Docs**: https://cityflow.readthedocs.io/
- **MAML Paper**: https://arxiv.org/abs/1703.03400
- **Issues**: Open an issue on GitHub

## Support

For help:
1. Check the full README.md
2. Search existing GitHub issues
3. Open a new issue with logs and error messages
4. Contact: [your-email]

Happy training! 🚦🤖
