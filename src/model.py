"""
LSTM-based Intrusion Detection System Model

This module implements a deep learning IDS using LSTM (Long Short-Term Memory)
networks for UAV telemetry sequence classification.

Model Architecture:
- Input: UAV telemetry sequences (window_size, n_features)
- LSTM layers: Capture temporal patterns and dependencies
- Fully connected layers: Classification head
- Output: Binary classification (normal vs attack)

Why LSTM?
1. Captures temporal dependencies in telemetry sequences
2. Learns normal behavior patterns over time
3. Detects anomalies like replay attacks (repeated sequences)
4. Handles variable-length patterns through gated architecture

Design decisions:
- Bidirectional LSTM: Learns from both past and future context
- Dropout: Prevents overfitting (crucial for security applications)
- Sigmoid activation: Outputs attack probability [0, 1]
- Binary Cross-Entropy loss: Standard for binary classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM_IDS(nn.Module):
    """
    LSTM-based Intrusion Detection System
    
    Architecture:
    Input → LSTM → LSTM → Dropout → FC → Dropout → FC → Sigmoid → Output
    """
    
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
        """
        Initialize LSTM IDS model
        
        Args:
            input_size: Number of features per timestep (9 for UAV telemetry)
            hidden_size: Number of LSTM hidden units (neurons per layer)
                        Larger = more capacity but slower and prone to overfitting
            num_layers: Number of stacked LSTM layers
                       2 layers is standard for sequence classification
            dropout: Dropout rate for regularization (0.3 = drop 30% of neurons)
                    Applied between LSTM layers and FC layers
        """
        super(LSTM_IDS, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        # - Bidirectional: Process sequence forward and backward
        # - batch_first=True: Input shape (batch, seq_len, features)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,  # Dropout between LSTM layers
            bidirectional=True  # Doubles hidden size: outputs 2*hidden_size
        )
        
        # Fully connected layers for classification
        # Input: 2*hidden_size (bidirectional doubles the output)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, 1)  # Binary classification
        
        # Batch normalization for stable training
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Input sequences (batch_size, window_size, input_size)
            
        Returns:
            out: Attack probability (batch_size, 1)
        """
        # LSTM forward pass
        # lstm_out: (batch_size, seq_len, hidden_size*2)
        # h_n: Final hidden state (num_layers*2, batch_size, hidden_size)
        # c_n: Final cell state (num_layers*2, batch_size, hidden_size)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use the last timestep output for classification
        # Shape: (batch_size, hidden_size*2)
        last_output = lstm_out[:, -1, :]
        
        # Fully connected layers
        out = F.relu(self.fc1(last_output))
        
        # Apply batch normalization (improves training stability)
        if out.size(0) > 1:  # Skip batch norm for single samples
            out = self.batch_norm(out)
            
        out = self.dropout(out)
        out = self.fc2(out)
        
        # Sigmoid activation for probability output
        out = torch.sigmoid(out)
        
        return out
    
    def predict(self, x, threshold=0.5):
        """
        Make binary predictions
        
        Args:
            x: Input sequences
            threshold: Classification threshold (default 0.5)
                      Lower threshold = more sensitive (catches more attacks but more false positives)
                      Higher threshold = more conservative (fewer false positives but might miss attacks)
        
        Returns:
            predictions: Binary predictions (0 or 1)
            probabilities: Attack probabilities
        """
        self.eval()  # Set to evaluation mode
        with torch.no_grad():
            probs = self.forward(x)
            preds = (probs >= threshold).float()
        return preds, probs


class LSTM_IDS_Advanced(nn.Module):
    """
    Advanced LSTM IDS with attention mechanism
    
    Adds attention layer to focus on important timesteps.
    Useful for detecting specific attack patterns within sequences.
    """
    
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
        """
        Initialize advanced LSTM IDS with attention
        
        Args:
            Same as LSTM_IDS
        """
        super(LSTM_IDS_Advanced, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention mechanism
        # Learns which timesteps are most important for classification
        self.attention = nn.Linear(hidden_size * 2, 1)
        
        # Classification layers
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        
    def attention_layer(self, lstm_out):
        """
        Apply attention mechanism to LSTM outputs
        
        Args:
            lstm_out: LSTM outputs (batch_size, seq_len, hidden_size*2)
            
        Returns:
            context: Weighted sum of outputs (batch_size, hidden_size*2)
        """
        # Calculate attention scores for each timestep
        # Shape: (batch_size, seq_len, 1)
        attention_scores = self.attention(lstm_out)
        
        # Apply softmax to get attention weights
        # Shape: (batch_size, seq_len, 1)
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # Weighted sum of LSTM outputs
        # Shape: (batch_size, hidden_size*2)
        context = torch.sum(attention_weights * lstm_out, dim=1)
        
        return context
    
    def forward(self, x):
        """
        Forward pass with attention
        
        Args:
            x: Input sequences (batch_size, window_size, input_size)
            
        Returns:
            out: Attack probability (batch_size, 1)
        """
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Apply attention to get context vector
        context = self.attention_layer(lstm_out)
        
        # Classification layers
        out = F.relu(self.fc1(context))
        
        if out.size(0) > 1:
            out = self.batch_norm(out)
            
        out = self.dropout(out)
        out = self.fc2(out)
        out = torch.sigmoid(out)
        
        return out
    
    def predict(self, x, threshold=0.5):
        """Make binary predictions"""
        self.eval()
        with torch.no_grad():
            probs = self.forward(x)
            preds = (probs >= threshold).float()
        return preds, probs


def create_model(input_size=9, hidden_size=64, num_layers=2, dropout=0.3, 
                use_attention=False, device='cpu'):
    """
    Factory function to create IDS model
    
    Args:
        input_size: Number of input features (9 for UAV telemetry)
        hidden_size: LSTM hidden size
        num_layers: Number of LSTM layers
        dropout: Dropout rate
        use_attention: Whether to use attention mechanism
        device: 'cpu' or 'cuda'
        
    Returns:
        model: Initialized model on specified device
    """
    if use_attention:
        model = LSTM_IDS_Advanced(input_size, hidden_size, num_layers, dropout)
    else:
        model = LSTM_IDS(input_size, hidden_size, num_layers, dropout)
    
    model = model.to(device)
    
    # Print model summary
    print("\n" + "="*70)
    print("Model Architecture")
    print("="*70)
    print(f"Model type: {'LSTM with Attention' if use_attention else 'Standard LSTM'}")
    print(f"Input size: {input_size} features")
    print(f"Hidden size: {hidden_size}")
    print(f"Number of layers: {num_layers}")
    print(f"Dropout rate: {dropout}")
    print(f"Device: {device}")
    print("\nModel structure:")
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("="*70 + "\n")
    
    return model


def save_model(model, path='../models/lstm_ids.pth'):
    """
    Save model state dict
    
    Args:
        model: Trained model
        path: Save path
    """
    import os
    output_path = os.path.join(os.path.dirname(__file__), path)
    torch.save(model.state_dict(), output_path)
    print(f"✓ Model saved to {output_path}")


def load_model(model, path='../models/lstm_ids.pth', device='cpu'):
    """
    Load model state dict
    
    Args:
        model: Model instance to load weights into
        path: Model file path
        device: Device to load model on
        
    Returns:
        model: Model with loaded weights
    """
    import os
    input_path = os.path.join(os.path.dirname(__file__), path)
    model.load_state_dict(torch.load(input_path, map_location=device))
    model = model.to(device)
    model.eval()
    print(f"✓ Model loaded from {input_path}")
    return model


def main():
    """Example model creation"""
    print("UAV LSTM IDS Model Example\n")
    
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # Create standard model
    model = create_model(
        input_size=9,
        hidden_size=64,
        num_layers=2,
        dropout=0.3,
        use_attention=False,
        device=device
    )
    
    # Test with dummy input
    print("Testing model with dummy input...")
    batch_size = 32
    window_size = 10
    n_features = 9
    
    dummy_input = torch.randn(batch_size, window_size, n_features).to(device)
    output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
    print("\n✓ Model test successful!")


if __name__ == "__main__":
    main()
