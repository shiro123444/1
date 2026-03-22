"""
Training script for HierNBeats
Based on paper: Sun et al. 2024 (ICANN 2024)
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

from models.hiernbeats import HierNBeats
from utils.data_loader import load_labour_data, create_dataloaders
from utils.trainer import train_model, evaluate


def main():
    # Configuration
    config = {
        'data_path': './data/labour/data.csv',
        'tags_path': './data/labour/tags.csv',
        'backcast_length': 24,
        'forecast_length': 1,
        'num_stacks': 3,
        'num_blocks': 3,
        'layer_size': 512,
        'batch_size': 32,
        'num_epochs': 100,
        'lr': 0.001,
        'lambda_coherence': 1.0,
        'interpretable': True,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_path': './checkpoints/hiernbeats_best.pt'
    }

    print("=" * 50)
    print("HierNBeats Training")
    print("=" * 50)
    print(f"Device: {config['device']}")
    print(f"Interpretable mode: {config['interpretable']}")

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
        window_size=config['backcast_length'],
        forecast_horizon=config['forecast_length'],
        batch_size=config['batch_size']
    )
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Initialize model
    print("\nInitializing HierNBeats...")
    model = HierNBeats(
        backcast_length=config['backcast_length'],
        forecast_length=config['forecast_length'],
        hierarchy_structure=hierarchy_info,
        hierarchy_matrix=S,
        num_stacks=config['num_stacks'],
        num_blocks=config['num_blocks'],
        layer_size=config['layer_size'],
        interpretable=config['interpretable']
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
    plt.savefig('./results/hiernbeats_training.png', dpi=300)
    print("Training plot saved to ./results/hiernbeats_training.png")

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

    # Analyze interpretable components
    if config['interpretable']:
        print("\nAnalyzing interpretable components...")
        model.eval()
        with torch.no_grad():
            sample_x, _ = next(iter(test_loader))
            sample_x = sample_x.to(config['device'])
            components = model.get_interpretable_components(sample_x)

            print(f"Branch weights: {components['weights'].cpu().numpy()}")
            print(f"  Individual: {components['weights'][0]:.3f}")
            print(f"  Top-down: {components['weights'][1]:.3f}")
            print(f"  Bottom-up: {components['weights'][2]:.3f}")

    # Save predictions
    np.save('./results/hiernbeats_predictions.npy', predictions)
    np.save('./results/hiernbeats_targets.npy', targets)
    print("\nPredictions saved to ./results/")

    print("\n" + "=" * 50)
    print("Training completed!")
    print("=" * 50)


if __name__ == '__main__':
    main()
