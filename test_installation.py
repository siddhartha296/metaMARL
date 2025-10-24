"""
Installation Test Script
Run this to verify MetaMARL is correctly installed
File: test_installation.py
"""

import sys
import os

def test_imports():
    """Test that all modules can be imported."""
    print("=" * 60)
    print("Testing MetaMARL Installation")
    print("=" * 60)
    
    tests_passed = 0
    tests_failed = 0
    
    # Test 1: Basic imports
    print("\n1. Testing basic Python imports...")
    try:
        import numpy as np
        import torch
        import matplotlib.pyplot as plt
        print("   ✓ NumPy, PyTorch, Matplotlib imported successfully")
        tests_passed += 1
    except ImportError as e:
        print(f"   ✗ Failed to import basic libraries: {e}")
        tests_failed += 1
    
    # Test 2: CityFlow
    print("\n2. Testing CityFlow...")
    try:
        import cityflow
        print("   ✓ CityFlow imported successfully")
        print(f"   CityFlow version: {cityflow.__version__ if hasattr(cityflow, '__version__') else 'unknown'}")
        tests_passed += 1
    except ImportError as e:
        print(f"   ✗ CityFlow not found: {e}")
        print("   Install with: pip install cityflow")
        tests_failed += 1
    
    # Test 3: Project modules
    print("\n3. Testing project modules...")
    try:
        from src.envs.cityflow_env import CityFlowEnv
        from src.agents.metalight_agent import MetaLightAgent
        from src.agents.colight_agent import CoLightAgent
        from src.models.network_architectures import FRAPPlusPlus, CoLightGAT
        from src.core.maml import MAML
        print("   ✓ All project modules imported successfully")
        tests_passed += 1
    except ImportError as e:
        print(f"   ✗ Failed to import project modules: {e}")
        print("   Make sure you're in the project root directory")
        tests_failed += 1
    
    # Test 4: Model creation
    print("\n4. Testing model creation...")
    try:
        from src.models.network_architectures import FRAPPlusPlus
        import torch
        
        model = FRAPPlusPlus(obs_dim=32, action_dim=8, hidden_dim=64)
        dummy_input = torch.randn(1, 32)
        value, policy = model(dummy_input)
        
        assert value.shape == (1, 1), "Value shape incorrect"
        assert policy.shape == (1, 8), "Policy shape incorrect"
        
        print("   ✓ Model creation and forward pass successful")
        print(f"   Model parameters: {sum(p.numel() for p in model.parameters())}")
        tests_passed += 1
    except Exception as e:
        print(f"   ✗ Model creation failed: {e}")
        tests_failed += 1
    
    # Test 5: MAML initialization
    print("\n5. Testing MAML...")
    try:
        from src.core.maml import MAML
        from src.models.network_architectures import FRAPPlusPlus
        
        model = FRAPPlusPlus(obs_dim=32, action_dim=8, hidden_dim=64)
        maml = MAML(model=model, meta_lr=1e-3, inner_lr=1e-2, inner_steps=2)
        
        print("   ✓ MAML initialized successfully")
        tests_passed += 1
    except Exception as e:
        print(f"   ✗ MAML initialization failed: {e}")
        tests_failed += 1
    
    # Test 6: Directory structure
    print("\n6. Testing directory structure...")
    required_dirs = [
        'src/agents',
        'src/envs',
        'src/models',
        'src/core',
        'configs',
        'scripts',
        'data',
        'saved_models',
        'results'
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            missing_dirs.append(dir_path)
    
    if not missing_dirs:
        print("   ✓ All required directories exist")
        tests_passed += 1
    else:
        print(f"   ✗ Missing directories: {', '.join(missing_dirs)}")
        print("   Run: bash setup.sh")
        tests_failed += 1
    
    # Test 7: CUDA availability
    print("\n7. Testing CUDA...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"   ✓ CUDA available: {torch.cuda.device_count()} GPU(s)")
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("   ⚠ CUDA not available (CPU mode)")
            print("   Training will be slower on CPU")
        tests_passed += 1
    except Exception as e:
        print(f"   ✗ CUDA check failed: {e}")
        tests_failed += 1
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"Tests Passed: {tests_passed}")
    print(f"Tests Failed: {tests_failed}")
    
    if tests_failed == 0:
        print("\n✓ All tests passed! MetaMARL is ready to use.")
        print("\nNext steps:")
        print("1. Download datasets: See README.md or QUICKSTART.md")
        print("2. Create config files in configs/")
        print("3. Run: python scripts/meta_train.py --config configs/metalight_config.json")
        return True
    else:
        print("\n✗ Some tests failed. Please fix the issues above.")
        return False


def test_minimal_training():
    """Test a minimal training loop."""
    print("\n" + "=" * 60)
    print("Testing Minimal Training Loop")
    print("=" * 60)
    
    try:
        import torch
        from src.models.network_architectures import FRAPPlusPlus
        from src.core.maml import MAML
        
        print("\nCreating model and MAML trainer...")
        model = FRAPPlusPlus(obs_dim=32, action_dim=8, hidden_dim=64)
        maml = MAML(model=model, meta_lr=1e-3, inner_lr=1e-2, inner_steps=2)
        
        print("Creating dummy training data...")
        support_batch = {
            'states': torch.randn(16, 32),
            'actions': torch.randint(0, 8, (16,)),
            'returns': torch.randn(16, 1)
        }
        
        query_batch = {
            'states': torch.randn(16, 32),
            'actions': torch.randint(0, 8, (16,)),
            'returns': torch.randn(16, 1)
        }
        
        print("Performing meta-update...")
        metrics = maml.outer_loop_update([support_batch], [query_batch])
        
        print(f"\n✓ Meta-update successful!")
        print(f"  Meta Loss: {metrics['meta_loss']:.4f}")
        print(f"  Task Loss: {metrics['mean_task_loss']:.4f} ± {metrics['std_task_loss']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Minimal training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("MetaMARL Installation Test")
    print("=" * 60)
    
    # Check Python version
    import sys
    print(f"\nPython version: {sys.version}")
    
    if sys.version_info < (3, 7):
        print("✗ Python 3.7+ required")
        return False
    
    # Run tests
    basic_tests_passed = test_imports()
    
    if basic_tests_passed:
        print("\n" + "=" * 60)
        minimal_training_passed = test_minimal_training()
        
        if minimal_training_passed:
            print("\n" + "=" * 60)
            print("🎉 Installation Complete!")
            print("=" * 60)
            print("\nYou're ready to train MetaMARL!")
            print("\nQuick commands:")
            print("  - Meta-train:  python scripts/meta_train.py --config configs/metalight_config.json")
            print("  - Train baseline: python scripts/baseline_train.py --config configs/colight_config.json")
            print("  - Evaluate:    python scripts/evaluate.py --task_config <path> ...")
            print("\nSee QUICKSTART.md for detailed instructions.")
            return True
    
    return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)