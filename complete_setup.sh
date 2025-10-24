#!/bin/bash

# Complete Setup Script for MetaMARL
# This script automates the entire setup process

set -e  # Exit on error

echo "=============================================="
echo "MetaMARL Complete Setup Script"
echo "=============================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "ℹ $1"
}

# Step 1: Create directory structure
echo "Step 1: Creating directory structure..."
mkdir -p data/{hangzhou_4x4,jinan_3x4,atlanta_2x2}
mkdir -p saved_models/{metalight,colight}
mkdir -p results/{logs,plots}
mkdir -p src/{agents,envs,models,core}
mkdir -p configs
mkdir -p scripts
print_success "Directory structure created"

# Step 2: Create __init__.py files
echo ""
echo "Step 2: Creating Python package files..."
touch src/__init__.py
touch src/agents/__init__.py
touch src/envs/__init__.py
touch src/models/__init__.py
touch src/core/__init__.py
print_success "Python package files created"

# Step 3: Create configuration files
echo ""
echo "Step 3: Creating configuration files..."

cat > configs/metalight_config.json << 'EOF'
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
  "adapt_lr": 0.01,
  "log_interval": 10,
  "save_interval": 100
}
EOF

cat > configs/colight_config.json << 'EOF'
{
  "data_dir": "data",
  "save_dir": "saved_models/colight",
  "learning_rate": 0.0005,
  "num_iterations": 2000,
  "episode_length": 3600,
  "hidden_dim": 64,
  "gamma": 0.99,
  "entropy_coef": 0.01,
  "value_coef": 0.5,
  "log_interval": 10,
  "save_interval": 200
}
EOF

print_success "Configuration files created"

# Step 4: Set permissions
echo ""
echo "Step 4: Setting file permissions..."
chmod +x scripts/*.py 2>/dev/null || true
chmod +x *.sh 2>/dev/null || true
print_success "Permissions set"

# Step 5: Check Python version
echo ""
echo "Step 5: Checking Python installation..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    print_success "Python found: $PYTHON_VERSION"
    
    # Check if version is 3.7+
    PYTHON_MAJOR=$(python3 -c 'import sys; print(sys.version_info.major)')
    PYTHON_MINOR=$(python3 -c 'import sys; print(sys.version_info.minor)')
    
    if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 7 ]; then
        print_success "Python version is compatible (3.7+)"
    else
        print_error "Python 3.7+ required. You have Python $PYTHON_MAJOR.$PYTHON_MINOR"
        exit 1
    fi
else
    print_error "Python 3 not found. Please install Python 3.7+"
    exit 1
fi

# Step 6: Install dependencies
echo ""
echo "Step 6: Installing Python dependencies..."
read -p "Install dependencies now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
        print_success "Dependencies installed"
    else
        print_warning "requirements.txt not found. Skipping..."
    fi
    
    # Try to install CityFlow
    echo ""
    echo "Installing CityFlow..."
    pip install cityflow 2>/dev/null && print_success "CityFlow installed" || print_warning "CityFlow installation failed. Install manually."
else
    print_info "Skipping dependency installation"
fi

# Step 7: Test installation
echo ""
echo "Step 7: Testing installation..."
if [ -f "test_installation.py" ]; then
    python test_installation.py
else
    print_warning "test_installation.py not found. Skipping tests..."
fi

# Step 8: Data setup information
echo ""
echo "=============================================="
echo "Setup Complete!"
echo "=============================================="
echo ""
print_info "Next steps:"
echo ""
echo "1. Download datasets:"
echo "   Visit: https://github.com/MetaLight-Controller/MetaLight"
echo "   Place files in:"
echo "   - data/hangzhou_4x4/"
echo "   - data/jinan_3x4/"
echo "   - data/atlanta_2x2/"
echo ""
echo "2. Verify data structure:"
echo "   python -c \"from src.utils import validate_data_structure; validate_data_structure('data')\""
echo ""
echo "3. Start training:"
echo "   python scripts/meta_train.py --config configs/metalight_config.json"
echo ""
echo "4. Train baseline:"
echo "   python scripts/baseline_train.py --config configs/colight_config.json"
echo ""
echo "5. Evaluate:"
echo "   python scripts/evaluate.py --task_config <path> --metalight_model <path> --colight_model_dir <path>"
echo ""
echo "For more details, see:"
echo "- README.md for comprehensive documentation"
echo "- QUICKSTART.md for quick start guide"
echo "- PROJECT_SUMMARY.md for complete file listing"
echo ""
echo "Use 'make help' to see available commands"
echo ""

# Create a status check script
cat > check_status.sh << 'EOF'
#!/bin/bash
echo "MetaMARL Project Status"
echo "======================="
echo ""
echo "Directory Structure:"
[ -d "src/agents" ] && echo "✓ src/agents/" || echo "✗ src/agents/"
[ -d "src/envs" ] && echo "✓ src/envs/" || echo "✗ src/envs/"
[ -d "src/models" ] && echo "✓ src/models/" || echo "✗ src/models/"
[ -d "src/core" ] && echo "✓ src/core/" || echo "✗ src/core/"
[ -d "configs" ] && echo "✓ configs/" || echo "✗ configs/"
[ -d "scripts" ] && echo "✓ scripts/" || echo "✗ scripts/"
[ -d "data" ] && echo "✓ data/" || echo "✗ data/"
echo ""
echo "Configuration Files:"
[ -f "configs/metalight_config.json" ] && echo "✓ metalight_config.json" || echo "✗ metalight_config.json"
[ -f "configs/colight_config.json" ] && echo "✓ colight_config.json" || echo "✗ colight_config.json"
echo ""
echo "Python Packages:"
[ -f "src/__init__.py" ] && echo "✓ src/__init__.py" || echo "✗ src/__init__.py"
[ -f "src/agents/__init__.py" ] && echo "✓ src/agents/__init__.py" || echo "✗ src/agents/__init__.py"
echo ""
echo "Data Files:"
DATA_COUNT=$(find data -name "*.json" 2>/dev/null | wc -l)
echo "Found $DATA_COUNT JSON files in data/"
echo ""
echo "Saved Models:"
MODEL_COUNT=$(find saved_models -name "*.pth" 2>/dev/null | wc -l)
echo "Found $MODEL_COUNT model files"
echo ""
EOF

chmod +x check_status.sh
print_success "Status check script created: ./check_status.sh"

echo ""
print_success "Setup script completed successfully!"
echo ""
