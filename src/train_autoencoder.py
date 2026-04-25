"""
LSTM Autoencoder Training Script (Unsupervised)

This module trains an autoencoder ONLY on normal UAV telemetry data.
Training exclusively on normal samples allows the model to learn normal patterns,
making it sensitive to ANY deviation (including zero-day attacks).

Training Strategy:
==================
1. Load Dataset-1 (only normal samples, label=0)
2. Split into train/validation (80/20)
3. Train to minimize reconstruction error (MSE loss)
4. Select anomaly threshold using validation data (95th percentile)
5. Save model and threshold for inference

Why Train Only on Normal Data?
===============================
- Supervised models need attack examples → fail on zero-day attacks
- Autoencoders learn "what is normal" → detect anything abnormal
- GPS spoofing creates unusual patterns → high reconstruction error
- No attack labels needed → works even with unlabeled data

Key Insight:
If the model has only seen smooth GPS trajectories during training,
sudden 11km GPS jumps (spoofing) will produce very high reconstruction error.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime

from model import create_autoencoder, save_model
from preprocess import UAVDataPreprocessor, UAVSequenceDataset


class AutoencoderTrainer:
    """Trainer for LSTM Autoencoder (unsupervised)"""
    
    def __init__(self, model, device='cpu', learning_rate=0.001):
        """
        Initialize trainer
        
        Args:
            model: LSTM_Autoencoder instance
            device: 'cpu' or 'cuda'
            learning_rate: Learning rate for Adam optimizer
        """
        self.model = model
        self.device = device
        self.model.to(device)
        
        # Loss: Mean Squared Error (reconstruction error)
        self.criterion = nn.MSELoss(reduction='mean')
        
        # Optimizer: Adam with weight decay for regularization
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=1e-5
        )
        
        # Learning rate scheduler: Reduce LR when validation loss plateaus
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': []
        }
        
    def train_epoch(self, train_loader):
        """
        Train for one epoch
        
        Args:
            train_loader: DataLoader for training data
            
        Returns:
            avg_loss: Average training loss
        """
        self.model.train()
        total_loss = 0
        
        for data, _ in train_loader:  # Ignore labels (unsupervised)
            data = data.to(self.device)
            
            # Forward pass: Reconstruct input
            self.optimizer.zero_grad()
            reconstructed = self.model(data)
            
            # Loss: MSE between input and reconstruction
            loss = self.criterion(reconstructed, data)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping (prevents exploding gradients)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss
    
    def validate(self, val_loader):
        """
        Validate model
        
        Args:
            val_loader: DataLoader for validation data
            
        Returns:
            avg_loss: Average validation loss
            errors: Reconstruction errors for all samples
        """
        self.model.eval()
        total_loss = 0
        all_errors = []
        
        with torch.no_grad():
            for data, _ in val_loader:
                data = data.to(self.device)
                
                # Reconstruct
                reconstructed = self.model(data)
                
                # Loss
                loss = self.criterion(reconstructed, data)
                total_loss += loss.item()
                
                # Per-sample reconstruction errors
                errors = torch.mean((data - reconstructed) ** 2, dim=(1, 2))
                all_errors.extend(errors.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        return avg_loss, np.array(all_errors)
    
    def train(self, train_loader, val_loader, num_epochs=50, early_stopping_patience=10):
        """
        Complete training loop with early stopping
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Maximum number of epochs
            early_stopping_patience: Stop if no improvement for N epochs
            
        Returns:
            best_threshold: Anomaly detection threshold (95th percentile)
        """
        print("="*70)
        print("Starting Autoencoder Training (Unsupervised)")
        print("="*70)
        print(f"Device: {self.device}")
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        print(f"Epochs: {num_epochs}")
        print(f"Early stopping patience: {early_stopping_patience}")
        print("="*70 + "\n")
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_threshold = None
        
        for epoch in range(num_epochs):
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_errors = self.validate(val_loader)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            
            # Print progress
            print(f"Epoch [{epoch+1:3d}/{num_epochs}] | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f} | "
                  f"Val Error (95th): {np.percentile(val_errors, 95):.6f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Compute anomaly threshold (95th percentile of validation errors)
                best_threshold = float(np.percentile(val_errors, 95))
                
                # Save best model
                save_model(self.model, '../models/autoencoder_best.pth')
                print(f"  → Best model saved! Anomaly threshold: {best_threshold:.6f}")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"\nEarly stopping triggered at epoch {epoch+1}")
                    break
        
        # Save final model
        save_model(self.model, '../models/autoencoder_final.pth')
        
        print("\n" + "="*70)
        print("Training Complete!")
        print("="*70)
        print(f"Best validation loss: {best_val_loss:.6f}")
        print(f"Anomaly threshold (95th percentile): {best_threshold:.6f}")
        print("="*70)
        
        return best_threshold
    
    def plot_history(self, save_path='../results/autoencoder_training_history.png'):
        """
        Plot training history
        
        Args:
            save_path: Path to save plot
        """
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        plt.figure(figsize=(12, 5))
        
        # Loss plot
        plt.subplot(1, 1, 1)
        plt.plot(epochs, self.history['train_loss'], 'b-o', label='Training Loss', linewidth=2, markersize=4)
        plt.plot(epochs, self.history['val_loss'], 'r-s', label='Validation Loss', linewidth=2, markersize=4)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Reconstruction Error (MSE)', fontsize=12)
        plt.title('Autoencoder Training History', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Training history saved to {save_path}")
        plt.close()
    
    def save_history(self, save_path='../results/autoencoder_training_history.json'):
        """
        Save training history to JSON
        
        Args:
            save_path: Path to save JSON
        """
        with open(save_path, 'w') as f:
            json.dump(self.history, f, indent=4)
        print(f"✓ Training history saved to {save_path}")


def main():
    """Main training workflow"""
    print("="*70)
    print("UAV IDS - LSTM Autoencoder Training (Unsupervised)")
    print("="*70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Configuration
    WINDOW_SIZE = 10
    BATCH_SIZE = 64
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.001
    HIDDEN_SIZE = 32
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # Step 1: Load ONLY normal data
    print("Step 1: Loading normal data (Dataset-1, label=0 only)")
    print("-"*70)
    
    preprocessor = UAVDataPreprocessor(window_size=WINDOW_SIZE)
    
    # Load Dataset-1 and filter for normal samples only
    data = preprocessor.prepare_training_data(
        '../data/dataset_1_normal.csv',
        stride=1
    )
    
    X_normal = data['X_train']
    y_normal = data['y_train']
    
    # Verify all samples are normal
    print(f"Total samples: {len(y_normal)}")
    print(f"Normal samples: {np.sum(y_normal == 0)}")
    print(f"Attack samples: {np.sum(y_normal == 1)}")
    
    if np.sum(y_normal == 1) > 0:
        print("\n⚠️  Warning: Attack samples found in Dataset-1!")
        print("   Filtering to keep only normal samples...")
        normal_mask = y_normal == 0
        X_normal = X_normal[normal_mask]
        y_normal = y_normal[normal_mask]
        print(f"   Filtered to {len(y_normal)} normal samples")
    
    # Create dataset
    full_dataset = UAVSequenceDataset(X_normal, y_normal)
    
    # Step 2: Split into train/validation (80/20)
    print("\nStep 2: Splitting data (80% train, 20% validation)")
    print("-"*70)
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Step 3: Create autoencoder
    print("\nStep 3: Creating LSTM Autoencoder")
    print("-"*70)
    
    model = create_autoencoder(
        input_size=9,
        hidden_size=HIDDEN_SIZE,
        num_layers=1,
        dropout=0.2,
        device=device
    )
    
    # Step 4: Train
    print("\nStep 4: Training autoencoder")
    print("-"*70)
    
    trainer = AutoencoderTrainer(model, device=device, learning_rate=LEARNING_RATE)
    
    anomaly_threshold = trainer.train(
        train_loader,
        val_loader,
        num_epochs=NUM_EPOCHS,
        early_stopping_patience=10
    )
    
    # Step 5: Save artifacts
    print("\nStep 5: Saving artifacts")
    print("-"*70)
    
    # Save training plots and history
    trainer.plot_history()
    trainer.save_history()
    
    # Save threshold
    threshold_info = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'anomaly_threshold': anomaly_threshold,
        'percentile': 95,
        'training_samples': len(train_dataset),
        'validation_samples': len(val_dataset),
        'hidden_size': HIDDEN_SIZE,
        'window_size': WINDOW_SIZE,
        'note': 'Threshold is 95th percentile of validation reconstruction errors'
    }
    
    with open('../models/autoencoder_threshold.json', 'w') as f:
        json.dump(threshold_info, f, indent=4)
    print(f"✓ Threshold saved to ../models/autoencoder_threshold.json")
    
    # Step 6: Visualize reconstruction errors
    print("\nStep 6: Analyzing reconstruction errors")
    print("-"*70)
    
    model.eval()
    val_errors = []
    
    with torch.no_grad():
        for data, _ in val_loader:
            data = data.to(device)
            errors = model.get_reconstruction_error(data)
            val_errors.extend(errors.cpu().numpy())
    
    val_errors = np.array(val_errors)
    
    # Plot error distribution
    plt.figure(figsize=(12, 5))
    
    # Histogram
    plt.subplot(1, 2, 1)
    plt.hist(val_errors, bins=50, edgecolor='black', alpha=0.7)
    plt.axvline(anomaly_threshold, color='red', linestyle='--', linewidth=2, 
                label=f'Threshold (95th): {anomaly_threshold:.6f}')
    plt.xlabel('Reconstruction Error', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Reconstruction Errors (Normal Data)', 
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Box plot
    plt.subplot(1, 2, 2)
    plt.boxplot(val_errors, vert=True)
    plt.axhline(anomaly_threshold, color='red', linestyle='--', linewidth=2,
                label=f'Threshold: {anomaly_threshold:.6f}')
    plt.ylabel('Reconstruction Error', fontsize=12)
    plt.title('Reconstruction Error Distribution', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('../results/autoencoder_error_distribution.png', dpi=300, bbox_inches='tight')
    print(f"✓ Error distribution plot saved")
    plt.close()
    
    # Statistics
    print(f"\nValidation Error Statistics:")
    print(f"  Mean:   {np.mean(val_errors):.6f}")
    print(f"  Median: {np.median(val_errors):.6f}")
    print(f"  Std:    {np.std(val_errors):.6f}")
    print(f"  Min:    {np.min(val_errors):.6f}")
    print(f"  Max:    {np.max(val_errors):.6f}")
    print(f"  95th percentile: {anomaly_threshold:.6f} ← Threshold")
    
    print("\n" + "="*70)
    print("Autoencoder training complete!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Run: python evaluate_autoencoder.py")
    print("     → Test anomaly detection on Dataset-2 and Dataset-3")
    print("  2. Run: python ensemble_evaluate.py")
    print("     → Combine classifier + autoencoder predictions")
    print("="*70)


if __name__ == "__main__":
    main()
