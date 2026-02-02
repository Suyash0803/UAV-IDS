"""
Data Preprocessing Module for UAV IDS

This module handles:
1. Feature normalization using MinMaxScaler
2. Time-series sliding window creation for LSTM input
3. Train/test split with support for cross-dataset evaluation
4. Data loading and transformation utilities

Design decisions:
- MinMaxScaler chosen over StandardScaler because UAV telemetry has bounded ranges
- Sliding windows capture temporal patterns (key for detecting replay attacks)
- Normalization fitted on training data only to prevent data leakage
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import pickle
import os


class UAVDataPreprocessor:
    """Preprocessing pipeline for UAV telemetry data"""
    
    def __init__(self, window_size=10):
        """
        Initialize preprocessor
        
        Args:
            window_size: Number of timesteps in each sequence window
                        (10 timesteps = 10 seconds at 1Hz sampling)
        """
        self.window_size = window_size
        self.scaler = MinMaxScaler()
        self.feature_columns = ['lat', 'lon', 'altitude', 'velocity', 
                               'pitch', 'roll', 'yaw', 'battery', 'command_id']
        self.is_fitted = False
        
    def load_data(self, csv_path):
        """
        Load UAV telemetry data from CSV
        
        Args:
            csv_path: Path to CSV file
            
        Returns:
            DataFrame with loaded data
        """
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} samples from {os.path.basename(csv_path)}")
        print(f"  Features: {len(self.feature_columns)}, Attack rate: {df['label'].mean():.2%}")
        return df
    
    def fit_scaler(self, df):
        """
        Fit normalization scaler on training data
        
        Important: Only fit on training data to avoid data leakage!
        The scaler learns min/max from training set and applies to all datasets.
        
        Args:
            df: Training DataFrame
        """
        X = df[self.feature_columns].values
        self.scaler.fit(X)
        self.is_fitted = True
        print(f"✓ Scaler fitted on {len(df)} training samples")
        
    def normalize_features(self, df):
        """
        Normalize features using fitted scaler
        
        Args:
            df: DataFrame to normalize
            
        Returns:
            Normalized feature array
        """
        if not self.is_fitted:
            raise ValueError("Scaler not fitted! Call fit_scaler() first.")
            
        X = df[self.feature_columns].values
        X_normalized = self.scaler.transform(X)
        return X_normalized
    
    def create_sequences(self, X, y, stride=1):
        """
        Create sliding window sequences for LSTM input
        
        Why sliding windows?
        - Captures temporal dependencies (e.g., sudden GPS jumps)
        - Enables detection of replay attacks (repeated patterns)
        - Provides context for each prediction
        
        Args:
            X: Normalized feature array (n_samples, n_features)
            y: Labels array (n_samples,)
            stride: Step size for sliding window (1 = overlapping windows)
            
        Returns:
            X_seq: Sequences (n_windows, window_size, n_features)
            y_seq: Labels for each window (n_windows,)
                   Label is 1 if ANY timestep in window is attack
        """
        X_seq = []
        y_seq = []
        
        # Slide window across the dataset
        for i in range(0, len(X) - self.window_size + 1, stride):
            window = X[i:i + self.window_size]
            # Label window as attack if any timestep is attack
            # This is conservative: flags potential attack windows
            label = int(np.any(y[i:i + self.window_size] == 1))
            
            X_seq.append(window)
            y_seq.append(label)
        
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)
        
        print(f"  Created {len(X_seq)} sequences (window_size={self.window_size}, stride={stride})")
        print(f"  Shape: {X_seq.shape}, Attack rate: {y_seq.mean():.2%}")
        
        return X_seq, y_seq
    
    def prepare_training_data(self, train_csv, test_csv=None, stride=1):
        """
        Prepare data for training
        
        Workflow:
        1. Load training data
        2. Fit scaler on training data
        3. Normalize and create sequences for training
        4. Optionally process test data with same scaler
        
        Args:
            train_csv: Path to training CSV
            test_csv: Optional path to test CSV (if None, no test set)
            stride: Stride for sliding windows
            
        Returns:
            Dictionary with processed data:
            {
                'X_train': Training sequences,
                'y_train': Training labels,
                'X_test': Test sequences (if test_csv provided),
                'y_test': Test labels (if test_csv provided)
            }
        """
        print("\n" + "="*70)
        print("Preprocessing Pipeline")
        print("="*70 + "\n")
        
        # Load and process training data
        print("Step 1: Loading training data...")
        df_train = self.load_data(train_csv)
        
        print("\nStep 2: Fitting scaler on training data...")
        self.fit_scaler(df_train)
        
        print("\nStep 3: Normalizing training features...")
        X_train_norm = self.normalize_features(df_train)
        y_train = df_train['label'].values
        
        print("\nStep 4: Creating training sequences...")
        X_train_seq, y_train_seq = self.create_sequences(X_train_norm, y_train, stride=stride)
        
        result = {
            'X_train': X_train_seq,
            'y_train': y_train_seq
        }
        
        # Process test data if provided
        if test_csv:
            print("\n" + "-"*70)
            print("Processing test data...")
            print("-"*70 + "\n")
            
            df_test = self.load_data(test_csv)
            X_test_norm = self.normalize_features(df_test)
            y_test = df_test['label'].values
            
            X_test_seq, y_test_seq = self.create_sequences(X_test_norm, y_test, stride=stride)
            
            result['X_test'] = X_test_seq
            result['y_test'] = y_test_seq
        
        print("\n" + "="*70)
        print("Preprocessing complete!")
        print("="*70 + "\n")
        
        return result
    
    def prepare_unseen_data(self, csv_path, stride=1):
        """
        Prepare unseen dataset for evaluation (cross-dataset testing)
        
        Uses the scaler fitted on training data.
        This simulates zero-day attack detection on unseen data distributions.
        
        Args:
            csv_path: Path to unseen dataset CSV
            stride: Stride for sliding windows
            
        Returns:
            X_seq: Sequences
            y_seq: Labels
        """
        if not self.is_fitted:
            raise ValueError("Scaler not fitted! Train the model first.")
        
        print(f"\nPreparing unseen dataset: {os.path.basename(csv_path)}")
        df = self.load_data(csv_path)
        
        X_norm = self.normalize_features(df)
        y = df['label'].values
        
        X_seq, y_seq = self.create_sequences(X_norm, y, stride=stride)
        
        return X_seq, y_seq
    
    def save_scaler(self, path='../models/scaler.pkl'):
        """Save fitted scaler for later use"""
        output_path = os.path.join(os.path.dirname(__file__), path)
        with open(output_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"✓ Scaler saved to {output_path}")
        
    def load_scaler(self, path='../models/scaler.pkl'):
        """Load previously fitted scaler"""
        input_path = os.path.join(os.path.dirname(__file__), path)
        with open(input_path, 'rb') as f:
            self.scaler = pickle.load(f)
        self.is_fitted = True
        print(f"✓ Scaler loaded from {input_path}")


class UAVSequenceDataset(Dataset):
    """PyTorch Dataset for UAV sequences"""
    
    def __init__(self, X, y):
        """
        Args:
            X: Sequences array (n_samples, window_size, n_features)
            y: Labels array (n_samples,)
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_dataloaders(X_train, y_train, X_test=None, y_test=None, batch_size=64, shuffle=True):
    """
    Create PyTorch DataLoaders for training and testing
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data (optional)
        batch_size: Batch size for training
        shuffle: Whether to shuffle training data
        
    Returns:
        train_loader: Training DataLoader
        test_loader: Test DataLoader (if test data provided)
    """
    train_dataset = UAVSequenceDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    
    test_loader = None
    if X_test is not None and y_test is not None:
        test_dataset = UAVSequenceDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def main():
    """Example preprocessing workflow"""
    print("UAV IDS Data Preprocessing Example\n")
    
    # Initialize preprocessor
    preprocessor = UAVDataPreprocessor(window_size=10)
    
    # Prepare training data (Dataset-1: Normal)
    data = preprocessor.prepare_training_data(
        train_csv='../data/dataset_1_normal.csv',
        stride=1
    )
    
    print(f"\nFinal training data shape: {data['X_train'].shape}")
    print(f"Features per timestep: {data['X_train'].shape[2]}")
    
    # Save scaler
    preprocessor.save_scaler()
    
    # Example: Prepare unseen dataset
    print("\n" + "="*70)
    print("Example: Cross-dataset evaluation")
    print("="*70)
    X_unseen, y_unseen = preprocessor.prepare_unseen_data('../data/dataset_2_injection_replay.csv')
    print(f"Unseen data shape: {X_unseen.shape}")


if __name__ == "__main__":
    main()
