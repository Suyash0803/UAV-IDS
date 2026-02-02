"""
Training Script for UAV LSTM IDS

This module handles:
1. Model training with progress tracking
2. Validation during training
3. Early stopping to prevent overfitting
4. Learning rate scheduling
5. Training history logging
6. Model checkpointing

Training strategy:
- Train on Dataset-1 (normal UAV behavior) to learn baseline patterns
- Use Binary Cross-Entropy loss (standard for binary classification)
- Adam optimizer with learning rate decay
- Early stopping based on validation loss
- Save best model based on validation performance
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime

from model import create_model, save_model
from preprocess import UAVDataPreprocessor, create_dataloaders


class EarlyStopping:
    """
    Early stopping to stop training when validation loss stops improving
    
    Prevents overfitting by monitoring validation loss and stopping
    when it hasn't improved for a specified number of epochs.
    """
    
    def __init__(self, patience=10, min_delta=0.001, verbose=True):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            verbose: Print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        """
        Check if training should stop
        
        Args:
            val_loss: Current validation loss
            
        Returns:
            True if should stop, False otherwise
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"  EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
            
        return self.early_stop


class IDSTrainer:
    """Trainer class for LSTM IDS"""
    
    def __init__(self, model, device='cpu', learning_rate=0.001, pos_weight=10.0):
        """
        Initialize trainer
        
        Args:
            model: LSTM IDS model
            device: 'cpu' or 'cuda'
            learning_rate: Initial learning rate for optimizer
            pos_weight: Weight for positive class (attacks) - higher = more sensitive
        """
        self.model = model
        self.device = device
        
        # Loss function: Weighted Binary Cross-Entropy
        # Applies higher weight to attack samples to improve detection
        # pos_weight=10.0 means attacks are 10x more important than normal
        self.pos_weight = pos_weight
        self.criterion = nn.BCELoss()
        
        # Optimizer: Adam (Adaptive Moment Estimation)
        # Advantages: Fast convergence, handles sparse gradients well
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        
        # Learning rate scheduler: Reduce LR when validation loss plateaus
        # Helps fine-tune the model in later epochs
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
    def train_epoch(self, train_loader):
        """
        Train for one epoch
        
        Args:
            train_loader: DataLoader for training data
            
        Returns:
            avg_loss: Average training loss
            avg_acc: Average training accuracy
        """
        self.model.train()  # Set model to training mode
        
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            # Move data to device
            data, target = data.to(self.device), target.to(self.device)
            target = target.unsqueeze(1)  # Shape: (batch_size, 1)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)
            
            # Apply class weights: attacks weighted more heavily
            weights = torch.where(target == 1, self.pos_weight, 1.0).to(self.device)
            criterion_weighted = nn.BCELoss(weight=weights)
            loss = criterion_weighted(output, target)
            
            # Backward pass and optimization
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            predictions = (output >= 0.5).float()
            correct += (predictions == target).sum().item()
            total += target.size(0)
            
        avg_loss = total_loss / len(train_loader)
        avg_acc = correct / total
        
        return avg_loss, avg_acc
    
    def validate(self, val_loader):
        """
        Validate model
        
        Args:
            val_loader: DataLoader for validation data
            
        Returns:
            avg_loss: Average validation loss
            avg_acc: Average validation accuracy
        """
        self.model.eval()  # Set model to evaluation mode
        
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():  # Disable gradient computation
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                target = target.unsqueeze(1)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                predictions = (output >= 0.5).float()
                correct += (predictions == target).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(val_loader)
        avg_acc = correct / total
        
        return avg_loss, avg_acc
    
    def train(self, train_loader, val_loader=None, epochs=50, early_stopping_patience=10):
        """
        Full training loop
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            epochs: Maximum number of epochs
            early_stopping_patience: Patience for early stopping
            
        Returns:
            history: Training history dictionary
        """
        print("\n" + "="*70)
        print("Training LSTM IDS")
        print("="*70 + "\n")
        
        # Initialize early stopping
        early_stopping = EarlyStopping(patience=early_stopping_patience, verbose=True)
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            
            # Validate
            if val_loader:
                val_loss, val_acc = self.validate(val_loader)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
                
                # Learning rate scheduling
                self.scheduler.step(val_loss)
                
                # Print progress
                print(f"Epoch [{epoch+1}/{epochs}] | "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_model(self.model, '../models/lstm_ids_best.pth')
                    print(f"  ✓ Best model saved (val_loss: {val_loss:.4f})")
                
                # Early stopping check
                if early_stopping(val_loss):
                    print(f"\nEarly stopping triggered at epoch {epoch+1}")
                    break
            else:
                # No validation set
                print(f"Epoch [{epoch+1}/{epochs}] | "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        
        print("\n" + "="*70)
        print("Training complete!")
        print("="*70 + "\n")
        
        # Save final model
        save_model(self.model, '../models/lstm_ids_final.pth')
        
        return self.history
    
    def plot_history(self, save_path='../results/training_history.png'):
        """
        Plot training history
        
        Args:
            save_path: Path to save plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss plot
        ax1.plot(self.history['train_loss'], label='Train Loss', linewidth=2)
        if self.history['val_loss']:
            ax1.plot(self.history['val_loss'], label='Validation Loss', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plot
        ax2.plot(self.history['train_acc'], label='Train Accuracy', linewidth=2)
        if self.history['val_acc']:
            ax2.plot(self.history['val_acc'], label='Validation Accuracy', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_path = os.path.join(os.path.dirname(__file__), save_path)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Training history plot saved to {output_path}")
        plt.close()
    
    def save_history(self, save_path='../results/training_history.json'):
        """Save training history to JSON"""
        output_path = os.path.join(os.path.dirname(__file__), save_path)
        with open(output_path, 'w') as f:
            json.dump(self.history, f, indent=4)
        print(f"✓ Training history saved to {output_path}")


def main():
    """Main training workflow"""
    print("="*70)
    print("UAV IDS Training Pipeline (with Attack Samples)")
    print("="*70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Configuration
    WINDOW_SIZE = 10
    BATCH_SIZE = 64
    EPOCHS = 50
    LEARNING_RATE = 0.001
    HIDDEN_SIZE = 64
    NUM_LAYERS = 2
    DROPOUT = 0.3
    POS_WEIGHT = 10.0  # Weight for attack samples (10x more important)
    
    # Device configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # Step 1: Prepare data with BOTH normal and attack samples
    print("Step 1: Preparing mixed training dataset...")
    print("-"*70)
    
    preprocessor = UAVDataPreprocessor(window_size=WINDOW_SIZE)
    
    # Load normal data (Dataset-1)
    print("Loading normal data (Dataset-1)...")
    data_normal = preprocessor.prepare_training_data(
        train_csv='../data/dataset_1_normal.csv',
        stride=1
    )
    
    # Load attack data (Dataset-2) for training
    print("\nLoading attack data (Dataset-2) for training...")
    X_attack, y_attack = preprocessor.prepare_unseen_data(
        '../data/dataset_2_injection_replay.csv',
        stride=1
    )
    
    # Mix training data: Use more normal samples to maintain realistic imbalance
    # Take 70% of normal data and 30% of attack data
    n_normal = int(0.7 * len(data_normal['X_train']))
    n_attack = int(0.3 * len(X_attack))
    
    print(f"\nMixing training data:")
    print(f"  Normal samples: {n_normal}")
    print(f"  Attack samples: {n_attack}")
    print(f"  Attack ratio: {n_attack/(n_normal+n_attack):.2%}")
    
    # Combine data
    X_train_mixed = np.concatenate([
        data_normal['X_train'][:n_normal],
        X_attack[:n_attack]
    ])
    y_train_mixed = np.concatenate([
        data_normal['y_train'][:n_normal],
        y_attack[:n_attack]
    ])
    
    # Shuffle the mixed data
    shuffle_idx = np.random.permutation(len(X_train_mixed))
    X_train_mixed = X_train_mixed[shuffle_idx]
    y_train_mixed = y_train_mixed[shuffle_idx]
    
    # Split for validation (80-20 split)
    split_idx = int(0.8 * len(X_train_mixed))
    X_train = X_train_mixed[:split_idx]
    y_train = y_train_mixed[:split_idx]
    X_val = X_train_mixed[split_idx:]
    y_val = y_train_mixed[split_idx:]
    
    print(f"\nData split:")
    print(f"  Training:   {len(X_train)} sequences")
    print(f"  Validation: {len(X_val)} sequences")
    
    # Save preprocessor
    preprocessor.save_scaler()
    
    # Step 2: Create data loaders
    print("\nStep 2: Creating data loaders...")
    print("-"*70)
    train_loader, val_loader = create_dataloaders(
        X_train, y_train, X_val, y_val,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    
    # Step 3: Create model
    print("\nStep 3: Creating model...")
    print("-"*70)
    model = create_model(
        input_size=9,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        use_attention=False,
        device=device
    )
    
    # Step 4: Train model
    print("\nStep 4: Training model...")
    print("-"*70)
    print(f"Using pos_weight={POS_WEIGHT} to emphasize attack detection\n")
    trainer = IDSTrainer(model, device=device, learning_rate=LEARNING_RATE, pos_weight=POS_WEIGHT)
    
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=EPOCHS,
        early_stopping_patience=10
    )
    
    # Step 5: Save results
    print("\nStep 5: Saving results...")
    print("-"*70)
    trainer.plot_history()
    trainer.save_history()
    
    # Print final results
    print("\n" + "="*70)
    print("Training Summary")
    print("="*70)
    print(f"Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"Final train accuracy: {history['train_acc'][-1]:.4f}")
    print(f"Final val loss: {history['val_loss'][-1]:.4f}")
    print(f"Final val accuracy: {history['val_acc'][-1]:.4f}")
    print(f"Best val loss: {min(history['val_loss']):.4f}")
    print("="*70)


if __name__ == "__main__":
    main()
