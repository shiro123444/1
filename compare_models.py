"""
Compare HDResNet and HierNBeats performance
"""
import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from models.hdresnet import HDResNet
from models.hiernbeats import HierNBeats
from utils.data_loader import load_labour_data, create_dataloaders
from utils.trainer import evaluate


def compare_models():
    """Compare HDResNet and HierNBeats on test data"""

    # Configuration
    config = {
        'data_path': './data/labour/data.csv',
        'tags_path': './data/labour/tags.csv',
        'window_size': 24,
        'forecast_horizon': 1,
        'batch_size': 32,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    print("=" * 60)
    print("Model Comparison: HDResNet vs HierNBeats")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    df, hierarchy_info, S = load_labour_data(config['data_path'], config['tags_path'])

    # Create dataloaders
    train_loader, test_loader, level_weights, mean, std = create_dataloaders(
        df, hierarchy_info, S,
        window_size=config['window_size'],
        forecast_horizon=config['forecast_horizon'],
        batch_size=config['batch_size']
    )

    S_tensor = torch.FloatTensor(S).to(config['device'])
    level_weights = level_weights.to(config['device'])

    # Load HDResNet
    print("\nLoading HDResNet...")
    level_structure = []
    for i in range(len(hierarchy_info) - 1):
        parent_level = hierarchy_info[f'level{i}']
        child_level = hierarchy_info[f'level{i+1}']
        for parent in parent_level:
            num_children = sum(1 for child in child_level if child.startswith(parent))
            if num_children > 0:
                level_structure.append(num_children)

    hdresnet = HDResNet(
        input_size=df.shape[1],
        hidden_size=128,
        num_blocks=5,
        level_structure=level_structure,
        hierarchy_matrix=S,
        output_horizon=config['forecast_horizon'],
        dropout=0.1
    )

    if os.path.exists('./checkpoints/hdresnet_best.pt'):
        hdresnet.load_state_dict(torch.load('./checkpoints/hdresnet_best.pt'))
        hdresnet = hdresnet.to(config['device'])
        print("HDResNet loaded successfully")
    else:
        print("Warning: HDResNet checkpoint not found")

    # Load HierNBeats
    print("\nLoading HierNBeats...")
    hiernbeats = HierNBeats(
        backcast_length=config['window_size'],
        forecast_length=config['forecast_horizon'],
        hierarchy_structure=hierarchy_info,
        hierarchy_matrix=S,
        num_stacks=3,
        num_blocks=3,
        layer_size=512,
        interpretable=True
    )

    if os.path.exists('./checkpoints/hiernbeats_best.pt'):
        hiernbeats.load_state_dict(torch.load('./checkpoints/hiernbeats_best.pt'))
        hiernbeats = hiernbeats.to(config['device'])
        print("HierNBeats loaded successfully")
    else:
        print("Warning: HierNBeats checkpoint not found")

    # Evaluate both models
    print("\n" + "=" * 60)
    print("Evaluating HDResNet...")
    print("=" * 60)
    hdresnet_metrics, hdresnet_preds, targets = evaluate(
        hdresnet, test_loader, S_tensor, level_weights, config['device']
    )

    print("\nHDResNet Results:")
    print(f"  Loss: {hdresnet_metrics['loss']:.4f}")
    print(f"  RMSE: {hdresnet_metrics['rmse']:.4f}")
    print(f"  MAE: {hdresnet_metrics['mae']:.4f}")
    print(f"  MAPE: {hdresnet_metrics['mape']:.2f}%")

    print("\n" + "=" * 60)
    print("Evaluating HierNBeats...")
    print("=" * 60)
    hiernbeats_metrics, hiernbeats_preds, _ = evaluate(
        hiernbeats, test_loader, S_tensor, level_weights, config['device']
    )

    print("\nHierNBeats Results:")
    print(f"  Loss: {hiernbeats_metrics['loss']:.4f}")
    print(f"  RMSE: {hiernbeats_metrics['rmse']:.4f}")
    print(f"  MAE: {hiernbeats_metrics['mae']:.4f}")
    print(f"  MAPE: {hiernbeats_metrics['mape']:.2f}%")

    # Create comparison table
    print("\n" + "=" * 60)
    print("Comparison Summary")
    print("=" * 60)

    comparison_df = pd.DataFrame({
        'Model': ['HDResNet', 'HierNBeats'],
        'Loss': [hdresnet_metrics['loss'], hiernbeats_metrics['loss']],
        'RMSE': [hdresnet_metrics['rmse'], hiernbeats_metrics['rmse']],
        'MAE': [hdresnet_metrics['mae'], hiernbeats_metrics['mae']],
        'MAPE': [hdresnet_metrics['mape'], hiernbeats_metrics['mape']]
    })

    print("\n", comparison_df.to_string(index=False))

    # Determine winner
    print("\n" + "=" * 60)
    if hdresnet_metrics['rmse'] < hiernbeats_metrics['rmse']:
        print("Winner: HDResNet (lower RMSE)")
        improvement = (hiernbeats_metrics['rmse'] - hdresnet_metrics['rmse']) / hiernbeats_metrics['rmse'] * 100
        print(f"Improvement: {improvement:.2f}%")
    else:
        print("Winner: HierNBeats (lower RMSE)")
        improvement = (hdresnet_metrics['rmse'] - hiernbeats_metrics['rmse']) / hdresnet_metrics['rmse'] * 100
        print(f"Improvement: {improvement:.2f}%")
    print("=" * 60)

    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Metric comparison
    metrics = ['Loss', 'RMSE', 'MAE', 'MAPE']
    hdresnet_values = [hdresnet_metrics['loss'], hdresnet_metrics['rmse'],
                       hdresnet_metrics['mae'], hdresnet_metrics['mape']]
    hiernbeats_values = [hiernbeats_metrics['loss'], hiernbeats_metrics['rmse'],
                         hiernbeats_metrics['mae'], hiernbeats_metrics['mape']]

    x = np.arange(len(metrics))
    width = 0.35

    axes[0, 0].bar(x - width/2, hdresnet_values, width, label='HDResNet', alpha=0.8)
    axes[0, 0].bar(x + width/2, hiernbeats_values, width, label='HierNBeats', alpha=0.8)
    axes[0, 0].set_xlabel('Metrics')
    axes[0, 0].set_ylabel('Value')
    axes[0, 0].set_title('Model Comparison')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(metrics)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Sample predictions
    sample_idx = 0
    series_idx = 0
    axes[0, 1].plot(targets[sample_idx:sample_idx+50, series_idx], label='Ground Truth', linewidth=2)
    axes[0, 1].plot(hdresnet_preds[sample_idx:sample_idx+50, series_idx], label='HDResNet', alpha=0.7)
    axes[0, 1].plot(hiernbeats_preds[sample_idx:sample_idx+50, series_idx], label='HierNBeats', alpha=0.7)
    axes[0, 1].set_title(f'Sample Predictions (Series {series_idx})')
    axes[0, 1].set_xlabel('Time')
    axes[0, 1].set_ylabel('Value')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Error distribution
    hdresnet_errors = np.abs(hdresnet_preds - targets).flatten()
    hiernbeats_errors = np.abs(hiernbeats_preds - targets).flatten()

    axes[1, 0].hist(hdresnet_errors, bins=50, alpha=0.6, label='HDResNet', density=True)
    axes[1, 0].hist(hiernbeats_errors, bins=50, alpha=0.6, label='HierNBeats', density=True)
    axes[1, 0].set_title('Error Distribution')
    axes[1, 0].set_xlabel('Absolute Error')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Per-series RMSE
    num_series = targets.shape[1]
    hdresnet_rmse_per_series = np.sqrt(np.mean((hdresnet_preds - targets)**2, axis=0))
    hiernbeats_rmse_per_series = np.sqrt(np.mean((hiernbeats_preds - targets)**2, axis=0))

    axes[1, 1].plot(hdresnet_rmse_per_series, marker='o', label='HDResNet', alpha=0.7)
    axes[1, 1].plot(hiernbeats_rmse_per_series, marker='s', label='HierNBeats', alpha=0.7)
    axes[1, 1].set_title('RMSE per Series')
    axes[1, 1].set_xlabel('Series Index')
    axes[1, 1].set_ylabel('RMSE')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('./results/model_comparison.png', dpi=300)
    print("\nComparison plot saved to ./results/model_comparison.png")

    # Save comparison results
    comparison_df.to_csv('./results/comparison_results.csv', index=False)
    print("Comparison results saved to ./results/comparison_results.csv")


if __name__ == '__main__':
    compare_models()
