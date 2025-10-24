#!/bin/bash

# MetaMARL Project Setup Script

echo "======================================"
echo "Setting up MetaMARL Project"
echo "======================================"

# Create directory structure
echo "Creating directory structure..."
mkdir -p data/{hangzhou_4x4,jinan_3x4,atlanta_2x2}
mkdir -p saved_models/{metalight,colight}
mkdir -p results/{logs,plots}
mkdir -p src/{agents,envs,models,core}
mkdir -p configs
mkdir -p scripts

# Create __init__.py files for Python packages
echo "Creating __init__.py files..."
touch src/__init__.py
touch src/agents/__init__.py
touch src/envs/__init__.py
touch src/models/__init__.py
touch src/core/__init__.py

# Set permissions
echo "Setting permissions..."
chmod +x scripts/*.py

echo ""
echo "======================================"
echo "Setup Complete!"
echo "======================================"
echo ""
echo "Next steps:"
echo "1. Download datasets from MetaLight GitHub:"
echo "   https://github.com/MetaLight-Controller/MetaLight"
echo ""
echo "2. Place datasets in the data/ directory:"
echo "   - data/hangzhou_4x4/"
echo "   - data/jinan_3x4/"
echo "   - data/atlanta_2x2/"
echo ""
echo "3. Install Python dependencies:"
echo "   pip install -r requirements.txt"
echo ""
echo "4. Or use Docker:"
echo "   docker build -t metamarl ."
echo "   docker run -it --gpus all -v $(pwd):/workspace metamarl"
echo ""
echo "5. Train MetaMARL:"
echo "   python scripts/meta_train.py --config configs/metalight_config.json"
echo ""
echo "6. Train CoLight baseline:"
echo "   python scripts/baseline_train.py --config configs/colight_config.json"
echo ""
echo "7. Evaluate:"
echo "   python scripts/evaluate.py \\"
echo "     --task_config data/atlanta_2x2/flow_config_1.json \\"
echo "     --metalight_model saved_models/metalight/metalight_final.pth \\"
echo "     --colight_model_dir saved_models/colight/"
echo ""
