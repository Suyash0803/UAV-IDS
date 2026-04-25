# UAV IDS Preprocessing Guide

## ✅ Your Preprocessing Module is Ready!

The `preprocess.py` module provides a complete preprocessing pipeline for UAV telemetry data.

## Features

✅ **Feature Normalization**: MinMaxScaler for bounded telemetry ranges  
✅ **Sliding Window Sequences**: Time-series windows for LSTM input  
✅ **Cross-Dataset Support**: Same preprocessing for all datasets  
✅ **Data Leakage Prevention**: Scaler fitted only on training data  
✅ **PyTorch Integration**: DataLoader support for efficient training  
✅ **Reusable & Modular**: Easy to use across experiments  

---

## Quick Start

### 1. Basic Usage

```python
from preprocess import UAVDataPreprocessor, create_dataloaders

# Initialize preprocessor (10 timesteps per sequence)
preprocessor = UAVDataPreprocessor(window_size=10)

# Prepare training data
data = preprocessor.prepare_training_data(
    train_csv='data/dataset_1_normal.csv',
    stride=1  # Overlapping windows
)

# Create PyTorch DataLoaders
train_loader, _ = create_dataloaders(
    data['X_train'], 
    data['y_train'],
    batch_size=64,
    shuffle=True
)

# Save scaler for later use
preprocessor.save_scaler('models/scaler.pkl')
```

### 2. Cross-Dataset Evaluation

```python
# Load previously fitted scaler
preprocessor.load_scaler('models/scaler.pkl')

# Prepare unseen dataset with same preprocessing
X_unseen, y_unseen = preprocessor.prepare_unseen_data(
    'data/dataset_2_injection_replay.csv'
)

# Evaluate on unseen data
# ... (use your trained model)
```

### 3. Train/Test Split

```python
# Option A: Use separate datasets
data = preprocessor.prepare_training_data(
    train_csv='data/dataset_1_normal.csv',
    test_csv='data/dataset_2_injection_replay.csv',
    stride=1
)

train_loader, test_loader = create_dataloaders(
    data['X_train'], data['y_train'],
    data['X_test'], data['y_test'],
    batch_size=64
)

# Option B: Manual split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    data['X_train'], data['y_train'],
    test_size=0.2, random_state=42
)
```

---

## Output Shapes

| Step | Input | Output |
|------|-------|--------|
| Load CSV | CSV file | `(10000, 11)` DataFrame |
| Normalize | `(10000, 9)` features | `(10000, 9)` normalized |
| Create Sequences | `(10000, 9)` | `(9991, 10, 9)` sequences |

**Sequence Shape**: `(n_sequences, window_size, n_features)`
- `n_sequences`: 9991 (from 10000 samples with window_size=10)
- `window_size`: 10 timesteps per sequence
- `n_features`: 9 telemetry features

---

## Key Parameters

### Window Size
```python
preprocessor = UAVDataPreprocessor(window_size=10)
```
- **10 timesteps** = 10 seconds at 1Hz sampling
- Captures temporal patterns for attack detection
- Larger windows = more context, fewer sequences

### Stride
```python
X_seq, y_seq = preprocessor.create_sequences(X, y, stride=1)
```
- **stride=1**: Overlapping windows (more sequences, slower training)
- **stride=5**: Non-overlapping windows (fewer sequences, faster training)

### Batch Size
```python
train_loader, _ = create_dataloaders(X, y, batch_size=64)
```
- **64**: Good balance for LSTM training
- Adjust based on GPU memory

---

## Example Workflow

```python
# 1. Preprocess training data
preprocessor = UAVDataPreprocessor(window_size=10)
data = preprocessor.prepare_training_data('data/dataset_1_normal.csv')
preprocessor.save_scaler()

print(f"Training sequences: {data['X_train'].shape}")
# Output: (9991, 10, 9)

# 2. Create DataLoaders
train_loader, _ = create_dataloaders(
    data['X_train'], data['y_train'], 
    batch_size=64
)

# 3. Training loop
for X_batch, y_batch in train_loader:
    # X_batch shape: (64, 10, 9)
    # y_batch shape: (64,)
    pass

# 4. Evaluate on unseen datasets
preprocessor.load_scaler()
X_test, y_test = preprocessor.prepare_unseen_data('data/dataset_2_injection_replay.csv')
```

---

## Dataset Processing Results

**Dataset-1 (Normal):**
- Raw: 10,000 samples
- Sequences: 9,991 windows
- Attack rate: 0.00%

**Dataset-2 (Injection/Replay):**
- Raw: 10,000 samples (30% attacks)
- Sequences: 9,991 windows
- Attack rate: 96.79% (at window level)

**Dataset-3 (GPS Spoofing):**
- Raw: 10,000 samples (49% attacks)
- Sequences: 9,991 windows
- Attack rate: ~98% (at window level)

*Note: Window-level attack rate is higher because a window is labeled as attack if ANY timestep contains an attack.*

---

## Next Steps

Your preprocessing is complete! Now you can:
1. ✅ Train LSTM model on preprocessed sequences
2. ✅ Evaluate across datasets for zero-day detection
3. ✅ Tune hyperparameters (window size, stride, batch size)

See `train.py` and `evaluate.py` for the complete training pipeline.
