# MetaMARL Makefile
# Simplifies common development and training tasks

.PHONY: help setup install test clean train-metalight train-colight evaluate docker-build docker-run

# Default target
help:
	@echo "MetaMARL - Available Commands:"
	@echo ""
	@echo "  make setup              - Create directory structure"
	@echo "  make install            - Install Python dependencies"
	@echo "  make test               - Run installation tests"
	@echo "  make validate-data      - Validate data directory structure"
	@echo ""
	@echo "  make train-metalight    - Train MetaMARL model"
	@echo "  make train-colight      - Train CoLight baseline"
	@echo "  make evaluate          - Evaluate models"
	@echo ""
	@echo "  make docker-build       - Build Docker image"
	@echo "  make docker-run         - Run Docker container"
	@echo ""
	@echo "  make clean              - Clean temporary files"
	@echo "  make clean-models       - Remove saved models"
	@echo "  make clean-all          - Remove all generated files"
	@echo ""

# Setup
setup:
	@echo "Creating directory structure..."
	@bash setup.sh
	@echo "Creating __init__.py files..."
	@touch src/__init__.py
	@touch src/agents/__init__.py
	@touch src/envs/__init__.py
	@touch src/models/__init__.py
	@touch src/core/__init__.py
	@echo "Setup complete!"

# Install dependencies
install:
	@echo "Installing Python dependencies..."
	pip install -r requirements.txt
	@echo "Installation complete!"
	@echo "Run 'make test' to verify installation"

# Test installation
test:
	@echo "Testing installation..."
	python test_installation.py

# Validate data
validate-data:
	@echo "Validating data structure..."
	python -c "from src.utils import validate_data_structure; validate_data_structure('data')"

# Create config files
create-configs:
	@echo "Creating configuration files..."
	@mkdir -p configs
	@echo '{\n  "data_dir": "data",\n  "save_dir": "saved_models/metalight",\n  "meta_lr": 0.0001,\n  "inner_lr": 0.001,\n  "inner_steps": 3,\n  "meta_batch_size": 4,\n  "num_meta_iterations": 1000,\n  "support_episodes": 2,\n  "query_episodes": 2,\n  "episode_length": 3600,\n  "hidden_dim": 128,\n  "gamma": 0.95,\n  "adapt_lr": 0.01,\n  "log_interval": 10,\n  "save_interval": 100\n}' > configs/metalight_config.json
	@echo '{\n  "data_dir": "data",\n  "save_dir": "saved_models/colight",\n  "learning_rate": 0.0005,\n  "num_iterations": 2000,\n  "episode_length": 3600,\n  "hidden_dim": 64,\n  "gamma": 0.99,\n  "entropy_coef": 0.01,\n  "value_coef": 0.5,\n  "log_interval": 10,\n  "save_interval": 200\n}' > configs/colight_config.json
	@echo "Configuration files created in configs/"

# Training commands
train-metalight:
	@echo "Starting MetaMARL training..."
	@echo "Logs: saved_models/metalight/logs/meta_train.log"
	python scripts/meta_train.py --config configs/metalight_config.json --seed 42

train-colight:
	@echo "Starting CoLight training..."
	@echo "Logs: saved_models/colight/logs/baseline_train.log"
	python scripts/baseline_train.py --config configs/colight_config.json --seed 42

# Evaluation
evaluate:
	@echo "Running evaluation..."
	@echo "Usage: make evaluate TASK=<task_config_path>"
	@if [ -z "$(TASK)" ]; then \
		echo "Error: TASK not specified"; \
		echo "Example: make evaluate TASK=data/atlanta_2x2/flow_config_1.json"; \
		exit 1; \
	fi
	python scripts/evaluate.py \
		--task_config $(TASK) \
		--metalight_config configs/metalight_config.json \
		--colight_config configs/colight_config.json \
		--metalight_model saved_models/metalight/metalight_final.pth \
		--colight_model_dir saved_models/colight/ \
		--num_adapt_steps 20 \
		--output_dir results/plots

# Docker commands
docker-build:
	@echo "Building Docker image..."
	docker build -t metamarl:latest .

docker-run:
	@echo "Running Docker container..."
	docker run -it --gpus all \
		-v $(PWD)/data:/workspace/data \
		-v $(PWD)/saved_models:/workspace/saved_models \
		-v $(PWD)/results:/workspace/results \
		metamarl:latest

# Monitoring
monitor-metalight:
	@tail -f saved_models/metalight/logs/meta_train.log

monitor-colight:
	@tail -f saved_models/colight/logs/baseline_train.log

# GPU monitoring
gpu:
	@watch -n 1 nvidia-smi

# Cleaning
clean:
	@echo "Cleaning temporary files..."
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -delete
	@find . -type f -name "*.log~" -delete
	@rm -rf /tmp/task_*.json
	@rm -rf /tmp/*cityflow*.json
	@echo "Cleanup complete!"

clean-models:
	@echo "Removing saved models..."
	@rm -rf saved_models/metalight/*.pth
	@rm -rf saved_models/colight/*.pth
	@echo "Models removed!"

clean-results:
	@echo "Removing results..."
	@rm -rf results/plots/*
	@rm -rf results/logs/*
	@echo "Results removed!"

clean-all: clean clean-models clean-results
	@echo "All generated files removed!"

# Development helpers
lint:
	@echo "Running code formatting check..."
	@which black > /dev/null || pip install black
	black --check src/ scripts/

format:
	@echo "Formatting code..."
	@which black > /dev/null || pip install black
	black src/ scripts/

# Quick start
quickstart: setup install create-configs test
	@echo ""
	@echo "======================================"
	@echo "Quick Start Complete!"
	@echo "======================================"
	@echo ""
	@echo "Next steps:"
	@echo "1. Download datasets into data/ directory"
	@echo "2. Run: make train-metalight"
	@echo "3. Run: make train-colight (in another terminal)"
	@echo "4. Run: make evaluate TASK=data/atlanta_2x2/flow_config_1.json"
	@echo ""

# Status check
status:
	@echo "======================================"
	@echo "MetaMARL Project Status"
	@echo "======================================"
	@echo ""
	@echo "Python: $$(python --version)"
	@echo "PyTorch: $$(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not installed')"
	@echo "CUDA Available: $$(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'N/A')"
	@echo ""
	@echo "Directories:"
	@ls -la | grep ^d | awk '{print "  " $$NF}'
	@echo ""
	@echo "Saved Models:"
	@find saved_models -name "*.pth" 2>/dev/null | wc -l | xargs echo "  "
	@echo ""
	@echo "Results:"
	@find results/plots -name "*.png" 2>/dev/null | wc -l | xargs echo "  Plots: "
	@find results/plots -name "*.json" 2>/dev/null | wc -l | xargs echo "  JSON: "
	@echo ""
