# UAV Intrusion Detection System (IDS) for Internet of Vehicles

A research-oriented Machine Learning-based Intrusion Detection System for detecting post-authentication UAV attacks in IoV environments.

## 🎯 Project Overview

This project implements a deep learning-based IDS to detect UAV attacks in Internet of Vehicles (IoV) scenarios. It extends blockchain-based UAV authentication schemes (like BASUV) by adding a runtime behavioral IDS layer to detect:

- **Command Injection Attacks**: Malicious commands injected into UAV control flow
- **Replay Attacks**: Stale telemetry data replayed to deceive the system
- **GPS Spoofing**: Manipulation of GPS coordinates
- **Sensor Anomalies**: Inconsistent or manipulated sensor readings

### Key Features

✅ Synthetic UAV telemetry dataset generation with realistic attack patterns  
✅ LSTM-based sequence classifier for temporal anomaly detection  
✅ Cross-dataset evaluation for zero-day attack detection  
✅ Comprehensive metrics: Accuracy, Precision, Recall, F1, FPR, ROC-AUC  
✅ Modular, research-quality implementation  

---

## 📁 Project Structure

```
btp/
├── data/                           # Generated datasets
│   ├── dataset_1_normal.csv        # Normal UAV telemetry (training)
│   ├── dataset_2_injection_replay.csv  # Command injection & replay attacks
│   └── dataset_3_gps_spoofing.csv  # GPS spoofing & sensor anomalies
│
├── src/                            # Source code
│   ├── generate_datasets.py        # Dataset generation
│   ├── preprocess.py               # Data preprocessing & windowing
│   ├── model.py                    # LSTM IDS model definition
│   ├── train.py                    # Training script
│   └── evaluate.py                 # Evaluation & cross-dataset testing
│
├── models/                         # Saved models
│   ├── lstm_ids_best.pth           # Best model checkpoint
│   ├── lstm_ids_final.pth          # Final trained model
│   └── scaler.pkl                  # Feature scaler
│
├── results/                        # Evaluation results
│   ├── training_history.png        # Training curves
│   ├── confusion_matrix_*.png      # Confusion matrices
│   ├── roc_curves.png              # ROC curves
│   ├── metrics_comparison.png      # Metrics comparison
│   └── evaluation_report.json      # Detailed metrics
│
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

---

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone or navigate to project directory
cd btp

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Datasets

```bash
cd src
python generate_datasets.py
```

**Output:**
- `dataset_1_normal.csv`: 10,000 normal UAV telemetry samples
- `dataset_2_injection_replay.csv`: 10,000 samples (30% attacks)
- `dataset_3_gps_spoofing.csv`: 10,000 samples (30% attacks)

### 3. Train the IDS

```bash
python train.py
```

**What happens:**
- Loads Dataset-1 (normal telemetry)
- Fits normalization scaler
- Creates time-series windows (10 timesteps each)
- Trains LSTM model with early stopping
- Saves best model and training history

### 4. Evaluate on Unseen Datasets

```bash
python evaluate.py
```

**What happens:**
- Loads trained model
- Tests on Dataset-2 (injection/replay attacks)
- Tests on Dataset-3 (GPS spoofing/sensor anomalies)
- Generates confusion matrices, ROC curves, and metrics
- Saves comprehensive evaluation report

---

## 📊 Datasets

### Dataset Features

Each dataset contains 9 UAV telemetry features:

| Feature      | Description                | Unit    |
|--------------|----------------------------|---------|
| `lat`        | Latitude                   | Degrees |
| `lon`        | Longitude                  | Degrees |
| `altitude`   | Flight altitude            | Meters  |
| `velocity`   | Speed                      | m/s     |
| `pitch`      | Pitch angle                | Degrees |
| `roll`       | Roll angle                 | Degrees |
| `yaw`        | Yaw angle                  | Degrees |
| `battery`    | Battery percentage         | %       |
| `command_id` | Command identifier         | Integer |
| `label`      | 0 = Normal, 1 = Attack     | Binary  |

### Dataset-1: Normal Telemetry

- **Purpose**: Training baseline
- **Samples**: 10,000
- **Attack Rate**: 0%
- **Characteristics**: Smooth trajectories, stable flight patterns

### Dataset-2: Command Injection & Replay Attacks

- **Purpose**: Test detection of command-based attacks
- **Samples**: 10,000
- **Attack Rate**: ~30%
- **Attack Types**:
  - **Command Injection**: Malicious command IDs (9000-9100 range)
  - **Replay Attacks**: Repeated/stale telemetry sequences

### Dataset-3: GPS Spoofing & Sensor Anomalies

- **Purpose**: Test detection of sensor manipulation
- **Samples**: 10,000
- **Attack Rate**: ~30%
- **Attack Types**:
  - **GPS Spoofing**: Sudden large jumps in coordinates (~11km)
  - **Sensor Anomalies**: Physically impossible readings (e.g., high velocity at low altitude)
  - **Telemetry Manipulation**: Fake battery/command data

---

## 🧠 Model Architecture

### LSTM-based IDS

```
Input (batch, 10, 9)
    ↓
Bidirectional LSTM (hidden_size=64, layers=2)
    ↓
Fully Connected (128 → 64)
    ↓
Batch Normalization
    ↓
Dropout (0.3)
    ↓
Fully Connected (64 → 1)
    ↓
Sigmoid Activation
    ↓
Output: Attack Probability [0, 1]
```

**Why LSTM?**
- Captures temporal dependencies in telemetry sequences
- Detects replay attacks (repeated patterns)
- Learns normal behavior over time windows
- Handles variable-length patterns through gated architecture

**Design Decisions:**
- **Bidirectional**: Learns from both past and future context
- **Window Size**: 10 timesteps (10 seconds at 1Hz sampling)
- **Loss**: Binary Cross-Entropy (standard for binary classification)
- **Optimizer**: Adam with learning rate scheduling
- **Regularization**: Dropout (0.3) + L2 weight decay

---

## 📈 Training Strategy

1. **Train on Dataset-1**: Learn normal UAV behavior patterns
2. **80-20 Train-Val Split**: Monitor validation performance
3. **Early Stopping**: Prevent overfitting (patience=10 epochs)
4. **Learning Rate Decay**: Reduce LR when validation loss plateaus
5. **Best Model Checkpoint**: Save model with lowest validation loss

**Hyperparameters:**
- Window Size: 10 timesteps
- Batch Size: 64
- Learning Rate: 0.001
- Epochs: 50 (with early stopping)
- Hidden Size: 64
- LSTM Layers: 2
- Dropout: 0.3

---

## 🎯 Evaluation Methodology

### Cross-Dataset Evaluation (Zero-Day Detection)

The model is trained **only on normal data** (Dataset-1) and tested on **unseen attack types** (Datasets 2 & 3). This simulates real-world zero-day attack detection.

### Metrics

- **Accuracy**: Overall classification correctness
- **Precision**: Of predicted attacks, how many are real? (Low FP)
- **Recall**: Of real attacks, how many detected? (Low FN)
- **F1-Score**: Harmonic mean of precision and recall
- **FPR** (False Positive Rate): Critical for IDS—low FPR means fewer false alarms
- **ROC-AUC**: Area under ROC curve (threshold-independent metric)

### Confusion Matrix Interpretation

```
                Predicted
              Normal  Attack
Actual Normal   TN      FP    ← False Alarms
Actual Attack   FN      TP    ← Missed Attacks
```

- **TN (True Negative)**: Correctly identified normal behavior
- **FP (False Positive)**: False alarms (normal flagged as attack)
- **FN (False Negative)**: Missed attacks (attack flagged as normal)
- **TP (True Positive)**: Correctly detected attacks

---

## 🔬 Research Contributions

1. **Behavioral IDS Layer**: Complements authentication-based security (BASUV)
2. **Zero-Day Detection**: Trained on normal data, detects unseen attack types
3. **Temporal Pattern Analysis**: LSTM captures sequential anomalies
4. **Cross-Dataset Generalization**: Tests on multiple attack distributions
5. **Reproducible Research**: Modular, well-documented implementation

---

## 📋 Usage Examples

### Generate Custom Dataset

```python
from generate_datasets import UAVDatasetGenerator

generator = UAVDatasetGenerator(random_seed=42)
df = generator.generate_dataset_2_injection_replay(
    n_samples=5000,
    attack_ratio=0.4  # 40% attacks
)
```

### Train with Custom Parameters

```python
from train import IDSTrainer
from model import create_model

model = create_model(
    input_size=9,
    hidden_size=128,  # Larger model
    num_layers=3,
    dropout=0.4
)

trainer = IDSTrainer(model, learning_rate=0.0005)
history = trainer.train(train_loader, val_loader, epochs=100)
```

### Evaluate with Custom Threshold

```python
from evaluate import IDSEvaluator

evaluator = IDSEvaluator(model, device='cuda')
metrics, y_true, y_pred, y_prob = evaluator.evaluate_dataset(
    test_loader,
    dataset_name='Custom Dataset',
    threshold=0.3  # More sensitive (catches more attacks, more FPs)
)
```

---

## 🛠️ Customization

### Adjust Attack Patterns

Edit [generate_datasets.py](src/generate_datasets.py):
- Modify attack injection logic
- Add new attack types
- Adjust attack intensity

### Change Model Architecture

Edit [model.py](src/model.py):
- Use attention mechanism: `use_attention=True`
- Add more LSTM layers
- Experiment with GRU or Transformer

### Tune Hyperparameters

Edit [train.py](src/train.py):
```python
WINDOW_SIZE = 20      # Longer temporal context
BATCH_SIZE = 128      # Larger batches
LEARNING_RATE = 0.0005  # Slower learning
HIDDEN_SIZE = 128     # More capacity
```

---

## 📊 Expected Results

After running the full pipeline, you should see:

### Training
- Training accuracy: ~95-99% (learns normal patterns well)
- Validation accuracy: ~95-98%
- Training converges in 20-30 epochs (with early stopping)

### Cross-Dataset Evaluation
- **Dataset-2** (Injection/Replay):
  - Accuracy: ~85-95%
  - F1-Score: ~0.80-0.90
  - FPR: <0.10 (low false alarms)
  
- **Dataset-3** (GPS/Sensor):
  - Accuracy: ~80-90%
  - F1-Score: ~0.75-0.85
  - May have slightly lower performance (harder attacks)

### Key Insights
- Model generalizes to unseen attack types
- Low FPR indicates practical deployability
- Some attacks harder to detect than others (GPS spoofing vs injection)

---

## 🔍 Troubleshooting

### Issue: Low recall on test sets
**Solution**: Lower classification threshold (e.g., 0.3 instead of 0.5)

### Issue: High false positive rate
**Solution**: 
- Increase classification threshold (e.g., 0.7)
- Train on more diverse normal data
- Add regularization (increase dropout)

### Issue: Model not training (loss not decreasing)
**Solution**:
- Check data loading (verify shapes)
- Reduce learning rate
- Check for NaN values in data
- Verify labels are binary (0/1)

### Issue: CUDA out of memory
**Solution**:
- Reduce batch size
- Reduce model size (hidden_size=32)
- Use `device='cpu'`

---

## 📚 Research Context

### Motivation

Traditional UAV security focuses on **authentication** (e.g., blockchain-based schemes like BASUV). However, post-authentication attacks can still occur:
- Compromised authenticated nodes
- Man-in-the-middle attacks
- Zero-day exploits

**Solution**: Add a **behavioral IDS layer** that monitors runtime telemetry to detect anomalies even from authenticated sources.

### Related Work

- **BASUV**: Blockchain-based authentication for UAV swarms
- **Anomaly Detection**: One-class SVM, Isolation Forest
- **Deep Learning IDS**: LSTM, CNN, Transformers for network intrusion detection

### Novelty

- UAV-specific attack patterns (GPS spoofing, command injection)
- Time-series behavioral analysis
- Cross-dataset zero-day detection evaluation

---

## 🤝 Contributing

This is a research project. Potential extensions:

1. **Real UAV Data**: Replace synthetic data with real drone telemetry
2. **Online Learning**: Update model with new attack patterns
3. **Federated Learning**: Distributed IDS across UAV swarm
4. **Explainability**: Add SHAP/LIME for attack attribution
5. **Multi-class Classification**: Classify attack types (not just binary)

---

## 📄 Citation

If you use this code in your research, please cite:

```
@misc{uav_ids_2026,
  author = {Your Name},
  title = {LSTM-based Intrusion Detection System for UAV-assisted IoV},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/yourusername/uav-ids}}
}
```

---

## 📝 License

This project is for academic and research purposes.

---

## 🙏 Acknowledgments

- Inspired by BASUV blockchain-based UAV authentication
- UAV telemetry patterns based on typical quadcopter specifications
- LSTM architecture adapted from sequence classification best practices

---

## 📧 Contact

For questions or collaboration:
- Email: your.email@example.com
- GitHub: [@yourusername](https://github.com/yourusername)

---

**Happy Researching! 🚁🔒**
