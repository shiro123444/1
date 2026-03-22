"""
Training script for HDResNet with data augmentation
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
from utils.data_loader import load_labour_data, create_dataloaders
from utils.trainer import evaluate
from utils.data_augmentation import augment_batch


def train_with_augmentation(model, train_loader, test_loader, S, level_weights,
                            num_epochs, device, save_path, augmentation_ratio=1.0):
    """Training with online data augmentation"""
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=30, T_mult=2, eta_min=1e-6
    )

    best_loss = float('inf')
    patience_counter = 0
    patience = 50

    history = {
        'train_loss': [], 'test_loss': [],
        'train_forecast_loss': [], 'train_coherence_loss': [],
        'test_rmse': [], 'test_mae': []
    }

    print(f"Training with {augmentation_ratio*100:.0f}% data augmentation")

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_forecast_loss = 0
        train_coherence_loss = 0

        for batch_x, batch_y in train_loader:
            # Apply data augmentation
            batch_x_np = batch_x.numpy()
            batch_y_np = batch_y.numpy()

            if augmentation_ratio > 0:
                batch_x_np, batch_y_np = augment_batch(
                    batch_x_np, batch_y_np, augmentation_ratio=augmentation_ratio
                )

            batch_x = torch.FloatTensor(batch_x_np).to(device)
            batch_y = torch.FloatTensor(batch_y_np).to(device)

            optimizer.zero_grad()
            output = model(batch_x)

            # Forecast loss
            forecast_loss = torch.mean(level_weights * torch.mean((output - batch_y) ** 2, dim=0))

            # Coherence loss
            num_bottom = S.size(1)
            bottom_pred = output[:, -num_bottom:]
            reconstructed = torch.matmul(bottom_pred, S.t())
            coherence_loss = torch.mean((output - reconstructed) ** 2)

            # Dynamic lambda
            lambda_coherence = min(1.0, 0.1 + epoch * 0.01)

            # Total loss
            loss = forecast_loss + lambda_coherence * coherence_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_forecast_loss += forecast_loss.item()
            train_coherence_loss += coherence_loss.item()

        train_forecast_loss /= len(train_loader)
        train_coherence_loss /= len(train_loader)
        train_loss = train_forecast_loss + lambda_coherence * train_coherence_loss

        # Evaluation
        test_metrics, _, _ = evaluate(model, test_loader, S, level_weights, device)

        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_metrics['loss'])
        history['train_forecast_loss'].append(train_forecast_loss)
        history['train_coherence_loss'].append(train_coherence_loss)
        history['test_rmse'].append(test_metrics['rmse'])
        history['test_mae'].append(test_metrics['mae'])

        scheduler.step()

        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f} (Forecast: {train_forecast_loss:.4f}, Coherence: {train_coherence_loss:.4f})")
            print(f"  Test Loss: {test_metrics['loss']:.4f}, RMSE: {test_metrics['rmse']:.4f}, MAE: {test_metrics['mae']:.4f}")
            print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Save best model
        if test_metrics['loss'] < best_loss:
            best_loss = test_metrics['loss']
            torch.save(model.state_dict(), save_path)
            patience_counter = 0
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  ✓ Model saved (best RMSE: {test_metrics['rmse']:.4f})")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    return history


def main():
    # Configuration with data augmentation
    config = {
        'data_path': './data/labour/data.csv',
        'tags_path': './data/labour/tags.csv',
        'window_size': 24,
        'forecast_horizon': 1,
        'hidden_size': 128,
        'num_blocks': 5,
        'batch_size': 32,
        'num_epochs': 200,
        'dropout': 0.1,
        'augmentation_ratio': 1.0,  # 100% augmentation
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_path': './checkpoints/hdresnet_augmented.pt'
    }

    print("=" * 50)
    print("HDResNet Training with Data Augmentation")
    print("=" * 50)
    print(f"Device: {config['device']}")
    print(f"Augmentation ratio: {config['augmentation_ratio']*100:.0f}%")

    os.makedirs('./checkpoints', exist_ok=True)
    os.makedirs('./results', exist_ok=True)

    # Load data
    print("\nLoading data...")
    df, hierarchy_info, S = load_labour_data(config['data_path'], config['tags_path'])
    print(f"Data shape: {df.shape}")

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
    print(f"Effective training samples per epoch: ~{len(train_loader) * config['batch_size'] * (1 + config['augmentation_ratio']):.0f}")

    # Build level structure
    level_structure = []
    for i in range(len(hierarchy_info) - 1):
        parent_level = hierarchy_info[f'level{i}']
        child_level = hierarchy_info[f'level{i+1}']
        for parent in parent_level:
            num_children = sum(1 for child in child_level if child.startswith(parent))
            if num_children > 0:
                level_structure.append(num_children)

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
    history = train_with_augmentation(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        S=torch.FloatTensor(S).to(config['device']),
        level_weights=level_weights.to(config['device']),
        num_epochs=config['num_epochs'],
        device=config['device'],
        save_path=config['save_path'],
        augmentation_ratio=config['augmentation_ratio']
    )

    # Plot training history
    print("\nPlotting results...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(history['train_loss'], label='Train')
    axes[0, 0].plot(history['test_loss'], label='Test')
    axes[0, 0].set_title('Total Loss (with Augmentation)')
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
    plt.savefig('./results/hdresnet_augmented_training.png', dpi=300)
    print("Training plot saved to ./results/hdresnet_augmented_training.png")

    # Final evaluation
    print("\nFinal evaluation...")
    model.load_state_dict(torch.load(config['save_path']))
    test_metrics, predictions, targets = evaluate(
        model, test_loader, torch.FloatTensor(S).to(config['device']),
        level_weights.to(config['device']), config['device']
    )

    print("\n" + "=" * 50)
    print("Final Test Metrics (with Data Augmentation):")
    print(f"  Loss: {test_metrics['loss']:.4f}")
    print(f"  RMSE: {test_metrics['rmse']:.4f}")
    print(f"  MAE: {test_metrics['mae']:.4f}")
    print(f"  MAPE: {test_metrics['mape']:.2f}%")
    print("=" * 50)

    # Compare with baseline
    baseline_rmse = 0.8245
    print("\nComparison with baseline:")
    print(f"  Baseline RMSE: {baseline_rmse:.4f}")
    print(f"  Augmented RMSE: {test_metrics['rmse']:.4f}")

    if test_metrics['rmse'] < baseline_rmse:
        improvement = (baseline_rmse - test_metrics['rmse']) / baseline_rmse * 100
        print(f"  ✓ Improvement: {improvement:.2f}%")
    else:
        degradation = (test_metrics['rmse'] - baseline_rmse) / baseline_rmse * 100
        print(f"  ✗ Degradation: {degradation:.2f}%")

    # Save predictions
    np.save('./results/hdresnet_augmented_predictions.npy', predictions)
    np.save('./results/hdresnet_augmented_targets.npy', targets)

    print("\nTraining completed!")


if __name__ == '__main__':
    main()
