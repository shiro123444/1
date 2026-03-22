"""
Training script for HDResNet
Based on paper: Xiang et al. 2023
"""
import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from models.hdresnet import HDResNet
from utils.data_loader import load_labour_data, create_dataloaders
from utils.trainer import train_model, evaluate


def main():
    # Configuration
    config = {
        'data_path': './data/labour/data.csv',
        'tags_path': './data/labour/tags.csv',
        'window_size': 24,
        'forecast_horizon': 1,
        'hidden_size': 128,
        'num_blocks': 5,
        'batch_size': 32,
        'num_epochs': 100,
        'lr': 0.001,
        'lambda_coherence': 1.0,
        'dropout': 0.1,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_path': './checkpoints/hdresnet_best.pt'
    }

    print("=" * 50)
    print("HDResNet Training")
    print("=" * 50)
    print(f"Device: {config['device']}")

    # Create directories
    os.makedirs('./checkpoints', exist_ok=True)
    os.makedirs('./results', exist_ok=True)

    # Load data
    print("\nLoading data...")
    df, hierarchy_info, S = load_labour_data(config['data_path'], config['tags_path'])
    print(f"Data shape: {df.shape}")
    print(f"Hierarchy levels: {len(hierarchy_info)}")
    print(f"S matrix shape: {S.shape}")

    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, test_loader, level_weights, mean, std = create_dataloaders(
        df, hierarchy_info, S,
        window_size=config['window_size'],
        forecast_horizon=config['forecast_horizon'],
        batch_size=config['batch_size']
    )
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Build level structure for HDResNet
    # This defines how many children each parent has
    level_structure = []
    for i in range(len(hierarchy_info) - 1):
        parent_level = hierarchy_info[f'level{i}']
        child_level = hierarchy_info[f'level{i+1}']

        for parent in parent_level:
            num_children = sum(1 for child in child_level if child.startswith(parent))
            if num_children > 0:
                level_structure.append(num_children)

    print(f"\nLevel structure: {level_structure}")

    # Initialize model
    print("\nInitializing HDResNet...")
    model = HDResNet(
        input_size=df.shape[1],
        hidden_size=config['hidden_size'],
        num_blocks=config['num_blocks'],
        level_structure=level_structure,
        hierarchy_matrix=S,
        output_horizon=config['forecast_horizon'],
        dropout=config['dropout']
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Train model
    print("\nStarting training...")
    history = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        S=torch.FloatTensor(S),
        level_weights=level_weights,
        num_epochs=config['num_epochs'],
        lr=config['lr'],
        lambda_coherence=config['lambda_coherence'],
        device=config['device'],
        save_path=config['save_path']
    )

    # Plot training history
    print("\nPlotting results...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(history['train_loss'], label='Train')
    axes[0, 0].plot(history['test_loss'], label='Test')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    axes[0, 1].plot(history['train_forecast_loss'], label='Forecast')
    axes[0, 1].plot(history['train_coherence_loss'], label='Coherence')
    axes[0, 1].set_title('Training Loss Components')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    axes[1, 0].plot(history['test_rmse'])
    axes[1, 0].set_title('Test RMSE')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('RMSE')
    axes[1, 0].grid(True)

    axes[1, 1].plot(history['test_mae'])
    axes[1, 1].set_title('Test MAE')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('MAE')
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig('./results/hdresnet_training.png', dpi=300)
    print("Training plot saved to ./results/hdresnet_training.png")

    # Final evaluation
    print("\nFinal evaluation...")
    model.load_state_dict(torch.load(config['save_path']))
    test_metrics, predictions, targets = evaluate(
        model, test_loader, torch.FloatTensor(S).to(config['device']),
        level_weights.to(config['device']), config['device']
    )

    print("\nFinal Test Metrics:")
    print(f"  Loss: {test_metrics['loss']:.4f}")
    print(f"  RMSE: {test_metrics['rmse']:.4f}")
    print(f"  MAE: {test_metrics['mae']:.4f}")
    print(f"  MAPE: {test_metrics['mape']:.2f}%")

    # Save predictions
    np.save('./results/hdresnet_predictions.npy', predictions)
    np.save('./results/hdresnet_targets.npy', targets)
    print("\nPredictions saved to ./results/")

    print("\n" + "=" * 50)
    print("Training completed!")
    print("=" * 50)


if __name__ == '__main__':
    main()
