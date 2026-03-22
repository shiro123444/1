"""
Quick test to verify model implementations
"""
import torch
import numpy as np
from models.hdresnet import HDResNet
from models.hiernbeats import HierNBeats

def test_hdresnet():
    """Test HDResNet forward pass"""
    print("Testing HDResNet...")

    # Create dummy data
    batch_size = 4
    seq_len = 24
    num_series = 10
    num_bottom = 5

    # Dummy S matrix
    S = np.random.rand(num_series, num_bottom)

    # Dummy level structure
    level_structure = [2, 3]  # Example: 2 children for first parent, 3 for second

    # Initialize model
    model = HDResNet(
        input_size=num_series,
        hidden_size=64,
        num_blocks=3,
        level_structure=level_structure,
        hierarchy_matrix=S,
        output_horizon=1,
        dropout=0.1
    )

    # Create dummy input
    x = torch.randn(batch_size, seq_len, num_series)

    # Forward pass
    try:
        output = model(x)
        print(f"✓ HDResNet forward pass successful")
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
        return True
    except Exception as e:
        print(f"✗ HDResNet forward pass failed: {e}")
        return False


def test_hiernbeats():
    """Test HierNBeats forward pass"""
    print("\nTesting HierNBeats...")

    # Create dummy data
    batch_size = 4
    backcast_len = 24
    forecast_len = 1
    num_series = 10
    num_bottom = 5

    # Dummy S matrix
    S = np.random.rand(num_series, num_bottom)

    # Dummy hierarchy info
    hierarchy_info = {
        'level0': ['total'],
        'level1': ['A', 'B'],
        'level2': ['A1', 'A2', 'B1', 'B2', 'B3']
    }

    # Initialize model
    model = HierNBeats(
        backcast_length=backcast_len,
        forecast_length=forecast_len,
        hierarchy_structure=hierarchy_info,
        hierarchy_matrix=S,
        num_stacks=2,
        num_blocks=2,
        layer_size=128,
        interpretable=True
    )

    # Create dummy input
    x = torch.randn(batch_size, backcast_len, num_series)

    # Forward pass
    try:
        output, components = model(x)
        print(f"✓ HierNBeats forward pass successful")
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Components: {list(components.keys())}")
        print(f"  Branch weights: {components['weights'].detach().numpy()}")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
        return True
    except Exception as e:
        print(f"✗ HierNBeats forward pass failed: {e}")
        return False


def test_interpretability():
    """Test HierNBeats interpretable components"""
    print("\nTesting HierNBeats interpretability...")

    batch_size = 2
    backcast_len = 24
    forecast_len = 1
    num_series = 8
    num_bottom = 4

    S = np.eye(num_series, num_bottom)
    S[0, :] = 1  # Top level sums all

    hierarchy_info = {
        'level0': ['total'],
        'level1': ['A', 'B', 'C', 'D']
    }

    model = HierNBeats(
        backcast_length=backcast_len,
        forecast_length=forecast_len,
        hierarchy_structure=hierarchy_info,
        hierarchy_matrix=S,
        num_stacks=2,
        num_blocks=2,
        layer_size=64,
        interpretable=True
    )

    x = torch.randn(batch_size, backcast_len, num_series)

    try:
        components = model.get_interpretable_components(x)
        print(f"✓ Interpretable components extracted")
        print(f"  Individual forecast shape: {components['individual'].shape}")
        print(f"  Top-down forecast shape: {components['topdown'].shape}")
        print(f"  Bottom-up forecast shape: {components['bottomup'].shape}")
        return True
    except Exception as e:
        print(f"✗ Interpretability test failed: {e}")
        return False


def main():
    print("=" * 60)
    print("Model Implementation Tests")
    print("=" * 60)

    results = []

    # Test HDResNet
    results.append(("HDResNet", test_hdresnet()))

    # Test HierNBeats
    results.append(("HierNBeats", test_hiernbeats()))

    # Test interpretability
    results.append(("Interpretability", test_interpretability()))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{name:20s}: {status}")

    all_passed = all(r[1] for r in results)
    print("\n" + "=" * 60)
    if all_passed:
        print("All tests passed! ✓")
    else:
        print("Some tests failed! ✗")
    print("=" * 60)

    return all_passed


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
